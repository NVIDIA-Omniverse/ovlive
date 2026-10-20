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

// Diff read orchestration: read an object's state from Valkey and build OMPF
// payloads.
//
// Implements SDD §10.5 (snapshot reads) and §10.1 (diff payload construction).
//
// The flow, end to end:
//   1. `read_object_snapshot` reads the WHOLE object (Spec SDD SR-1). For the
//      catch-up, legacy per-commit and checkpoint consumers it is the proven
//      read: the whole object in one atomic `EVALSHA` of
//      `scripts/object_snapshot_read.lua` on the snapshot lane, so its state is
//      exactly the state at the `seq` it returns, or a `TOO_LARGE` answer above
//      the atomic-read limits. Above those limits the proven read is the fenced
//      read (`storage::snapshot_fence::read`): the bounded fan-out below, run
//      under a per-object fence lease and proven by the lease's release —
//      catch-up's shared by a layer's joiners through its flight
//      (`storage::snapshot_flights`), the checkpoint assembler's its own. For
//      the structural `UpdateLayer` path it is the unfenced
//      bounded fan-out on the same lane: the header plus the four indices,
//      every section, tombstone, field and time-sample row, and the children
//      lists, in rounds with the outstanding commands and reply bytes windowed
//      (SR-4.3, SR-4.4, SR-11.3).
//   2. `snapshot_to_object_state` turns the snapshot into an `ObjectState`;
//      the Case 1/2/3 selection over it and the snapshot's tombstones (which
//      sections, fields and time samples a subscriber needs) runs in memory in
//      `domain::diff`.
//   3. `resolve_external_values` fetches the values the plan references in one
//      batch, and `build_diff_payload` builds the OODF delta and packs the OMPF
//      container. How it fetches is the caller's `ExtValueReads` (b.yuo, SDD
//      §11.2): a catch-up builds with `ExtValueReads::PROBE` first, where a
//      missing value builds nothing and answers `Built::Missing` — a commit
//      after the snapshot may have released it and the GC collected it, which
//      the catch-up resolves by re-reading under the `{gc}` read hold — and
//      with `ExtValueReads::recover` only for a value still missing under that
//      hold, once the store answered the hold live (the `LiveHoldProof` that
//      constructor takes), where the checkpoint-adapter fallback runs (SDD
//      §11) and a value still missing is `INTERNAL`, never an empty value.
// The `compute_*_at_cut` entry points below drive steps 2 and 3 from a
// snapshot step 1 proved.
//
// The structural write-planning cache's cold start reads the same atomic read
// under its own consumer, with the script's `return_dangling` flag, at the
// client's own ceilings (`read_object_snapshot_with_ceilings`); above them it
// takes the fenced read (`storage::snapshot_fence::read`).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use fred::interfaces::{HashesInterface, ListInterface, SetsInterface, SortedSetsInterface};
use ordered_float::OrderedFloat;

use oodf_ffi::{
    CommandDiffChangeSection as OodfDiffChangeSection,
    CommandDiffDeleteSection as OodfDiffDeleteSection, OwnedDelta, SetField as OodfSetField,
    StructCommand as OodfStructCommand, TimeSample as OodfTimeSample,
};

use crate::domain::delta::RewriteCount;
use crate::domain::diff::{collect_external_value_hashes, DiffEngine, DiffPlan, ExternalValueScan};
use crate::domain::ompf::OmpfContainer;
use crate::domain::types::{
    CompactFields, CompactSamples, CompactState, Field, FieldValue, ObjectState, ObjectStateOf,
    Section, ValueRows,
};
use crate::error::ServiceError;
use crate::observability::metrics::{self, SharedMetrics};
use crate::storage::client::{
    map_snapshot_load_err, FredErrOrigin, ValkeyClient, POOL_LABEL_SNAPSHOT,
};
use crate::storage::commit_charge::{
    ObjectStateNeed, QueuedCommands, SharedCommitCharge, WatchSnapshot,
};
use crate::storage::external_values::{hash_from_hex, ExternalValueStore, LiveHoldProof};
use crate::storage::fanout_thread::{PassAnswer, PassLane, PassLink, PassRequest};
use crate::storage::keys::{self, KeySchema, ROOT_PARENT_ID};
use crate::storage::memory_budget::SnapshotMemory;

// ── Object-snapshot read fault-injection seam (Spec SDD SR-1.5) ─────────────
//
// Compiled only under `#[cfg(any(test, feature = "test-fault-injection"))]`,
// modelled on `storage::txn_checkout::ProbeFaults` / `ProbeFaultInjector`. The
// release library has neither the injector nor the branch;
// `ObjectSnapshotReadFaults` degenerates to a zero-sized unit struct so no
// `#[cfg]` leaks into `ValkeyClient`'s field list or its constructor.
//
// **Why a seam is needed at all.** No healthy Valkey can be made to time out,
// drop a connection or answer `-LOADING` / `-NOAUTH` on one read on demand, so
// the transient and the permanent non-`WRONGTYPE` arms of the read's
// classification cannot be staged against a live server. The atomic read is
// one `EVALSHA`, so there is no per-key round trip left to fail on its own: the
// seam pops a
// canned `fred::error::Error` in place of the `EVALSHA`'s result, and that
// error takes the same classification a real one takes
// (`classify_object_snapshot_read_err` → `classify_snapshot_read_err` →
// `count_object_key_refusal` + `map_snapshot_load_err`), so the refusal
// counters, the log lines and the returned `ServiceError` are the real ones. A
// canned `SNAPSHOT_KEY_REFUSED: <family> <key> <raw>` error (`ErrorKind::
// Unknown`, the shape a script error reply arrives in) stands in for a per-key
// refusal inside the script; any other error for a script-level failure.
//
// On the two fan-outs — unfenced, and fenced (the fan-out the fenced read
// runs under its lease, Spec SDD SR-5.3) — the seam pops the canned error in
// place of one command, and it takes the classification a real failure of
// that command takes (`classify_snapshot_read_err` under that command's
// family, key and op). Which command is the queued entry's target
// (`ObjectSnapshotReadFaultTarget`): by default the first command of the
// fan-out window that consults it (on a fresh read, round 1's header
// `HGETALL`), or a command picked by its position in the read, overall or
// within one key family — or, for external-value clean-up's size-read pipeline
// alone, one of its seven commands by its key's family, whose reply the error
// replaces (`ObjectSnapshotReadFaultTarget::retirement_size_read`). The
// positional targets exist to stage b.zvd's defect
// shape — a transient failure of one children `LRANGE` deep in the read, which
// a reader that swallowed it would turn into a childless snapshot — on either
// fan-out. The seam answers per command on both transports: the plain
// commands of either fan-out, and, on the structural read's script windows
// (which windows those are: the bounded fan-out's section header below), each
// command a script window carries, before its `EVALSHA` is sent
// ([`issue_script_window`]). So a `SNAPSHOT_KEY_REFUSED:` tag means nothing to
// either fan-out: such an error is classified as that command's failure like
// any other. The one exception is on a script window, where a cluster-state
// text ([`is_script_cluster_state_refusal`]) takes the read's fallback to
// plain commands instead.
//
// Two more test-only levers serve the fenced read, whose lease must outlive a
// slow read (Spec SDD SR-5.3): a queued **delay** holds one fan-out command
// back for a set time before it is issued (on either fan-out; the read then
// goes on normally), and **stalled renewals** make the fenced read's renewal
// timer skip its renewals while set, so its lease lapses on its own deadline.
// A third, the queued **hold** ([`SnapshotReadHold`]), parks one fan-out
// command before it is issued until the test releases it, so a test can order
// a read against other work without a timed wait. None of the three changes
// what a command returns.
//
// Per-instance rather than process-global for the reason CLAUDE.md gives about
// `APPLY_STEP_FAULT_INJECTORS`: a process-global queue is shared-static state
// that leaks across concurrently-running tests.

/// Queued canned errors for [`read_object_snapshot`] and the fenced read's
/// fan-out (Spec SDD SR-1.5).
///
/// Each entry pairs an error with the [`ObjectSnapshotReadFaultTarget`] it
/// answers in place of. Only the front entry is ever consulted: it is popped
/// when the command (or `EVALSHA`) its target names comes up, and until then
/// every read takes the real path and the entries behind it wait. A read
/// consumes at most one entry — the injected command fails the read — and an
/// entry whose target the read never reached (a fan-out target on a read that
/// went atomic, a target naming the other fan-out, an index past the read's
/// last command) stays at the front for the next read.
/// An empty queue falls through to the real round trip, so the seam is
/// invisible to every other test.
///
/// Beside the errors it holds a second queue, of **delays**
/// ([`Self::push_delay_at`]), and a third, of **holds**
/// ([`Self::push_hold_at`]), both consulted by the two fan-outs only, and a
/// **renewal stall** switch ([`Self::stall_fence_renewals`]) the fenced read's
/// renewal timer obeys; see the section comment above.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Default)]
pub struct ObjectSnapshotReadFaultInjector {
    queue: std::sync::Mutex<
        std::collections::VecDeque<(ObjectSnapshotReadFaultTarget, fred::error::Error)>,
    >,
    delays: std::sync::Mutex<std::collections::VecDeque<(ObjectSnapshotReadFaultTarget, Duration)>>,
    holds: std::sync::Mutex<
        std::collections::VecDeque<(
            ObjectSnapshotReadFaultTarget,
            std::sync::Arc<SnapshotReadHold>,
        )>,
    >,
    renewals_stalled: std::sync::atomic::AtomicBool,
    /// The window recorder ([`Self::record_window_threads`]): `None` until
    /// a test starts it, then every fan-out window's [`FanoutWindowThread`]
    /// in the order the windows were issued.
    window_threads: std::sync::Mutex<Option<Vec<FanoutWindowThread>>>,
}

/// One fan-out window as [`ObjectSnapshotReadFaultInjector::record_window_threads`]
/// records it: which fan-out issued it, and the name of the thread that
/// issued it — the thread that polled the read's pass when the window's
/// commands were built (`storage::fanout_thread`'s `fanout-read` or one of
/// its `watch-read-<i>` threads, or `None` for an unnamed thread).
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FanoutWindowThread {
    pub reader: ObjectSnapshotReader,
    pub thread: Option<String>,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl ObjectSnapshotReadFaultInjector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Start recording every fan-out window a read on this client issues
    /// from now on, with the thread that issued it
    /// ([`Self::window_threads`]); a recording already started is cleared.
    /// The seam records a window as it builds the window's commands, before
    /// any of them is issued, on the thread polling the read's pass. Changes
    /// nothing a read does.
    pub fn record_window_threads(&self) -> &Self {
        if let Ok(mut recorded) = self.window_threads.lock() {
            *recorded = Some(Vec::new());
        }
        self
    }

    /// Every window recorded since [`Self::record_window_threads`], in
    /// issue order; empty when the recorder was never started.
    pub fn window_threads(&self) -> Vec<FanoutWindowThread> {
        self.window_threads
            .lock()
            .ok()
            .and_then(|recorded| recorded.clone())
            .unwrap_or_default()
    }

    /// How many windows [`Self::window_threads`] holds.
    pub fn windows_recorded(&self) -> usize {
        self.window_threads
            .lock()
            .ok()
            .and_then(|recorded| recorded.as_ref().map(Vec::len))
            .unwrap_or(0)
    }

    /// Record one window of `reader`, issued on the calling thread, when the
    /// recorder runs.
    fn note_window(&self, reader: ObjectSnapshotReader) {
        if let Ok(mut recorded) = self.window_threads.lock() {
            if let Some(recorded) = recorded.as_mut() {
                recorded.push(FanoutWindowThread {
                    reader,
                    thread: std::thread::current().name().map(str::to_owned),
                });
            }
        }
    }

    /// Queue one error for the next read attempt:
    /// [`push_at`](Self::push_at) with [`ObjectSnapshotReadFaultTarget::next_read`].
    pub fn push(&self, err: fred::error::Error) -> &Self {
        self.push_at(ObjectSnapshotReadFaultTarget::next_read(), err)
    }

    /// Queue one error to answer in place of the command `target` names.
    pub fn push_at(&self, target: ObjectSnapshotReadFaultTarget, err: fred::error::Error) -> &Self {
        if let Ok(mut q) = self.queue.lock() {
            q.push_back((target, err));
        }
        self
    }

    /// How many canned errors are still queued. Zero means every subsequent
    /// read takes the real path. Delays and holds are not counted
    /// ([`Self::delays_remaining`], [`Self::holds_remaining`]).
    pub fn remaining(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Queue one delay: the fan-out command `target` names is held back for
    /// `delay` before it is issued, and then issued for real. Only the two
    /// fan-outs consult delays — [`ObjectSnapshotReader::UnfencedFanout`] and
    /// [`ObjectSnapshotReader::FencedFanout`]; the atomic read never does.
    /// [`ObjectSnapshotReadFaultTarget::next_read`] names the first command of
    /// the next fan-out window. The delay queue is separate from the error
    /// queue and, like it, consulted only at its front; a read can consume any
    /// number of delays, since a delay fails nothing.
    ///
    /// # Panics
    ///
    /// On an [`ObjectSnapshotReadFaultTarget::retirement_size_read`] target:
    /// no fan-out command matches it, so the delay would never be taken, and
    /// every delay queued behind it would wait forever — a test arming it
    /// would pass without delaying anything.
    pub fn push_delay_at(&self, target: ObjectSnapshotReadFaultTarget, delay: Duration) -> &Self {
        target.assert_fanout_target("push_delay_at");
        if let Ok(mut q) = self.delays.lock() {
            q.push_back((target, delay));
        }
        self
    }

    /// How many delays are still queued.
    pub fn delays_remaining(&self) -> usize {
        self.delays.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Queue one hold: the fan-out command `target` names parks before it is
    /// issued until the test calls [`SnapshotReadHold::release`] on the
    /// returned handle, and is then issued for real. For ordering a read
    /// against other work without a timed wait.
    ///
    /// Arm, wait, release:
    ///
    /// ```ignore
    /// let hold = reads.push_hold_at(ObjectSnapshotReadFaultTarget::next_read());
    /// client.install_object_snapshot_read_fault_injector_for_test(reads.clone());
    /// // start the read in another task or `join!` branch, then:
    /// hold.parked().await; // the read is parked on the named command
    /// // ... the work that must happen while the read is parked ...
    /// hold.release();      // the read issues the command and goes on
    /// ```
    ///
    /// Like delays, holds are consulted by the two fan-outs only —
    /// [`ObjectSnapshotReader::UnfencedFanout`] and
    /// [`ObjectSnapshotReader::FencedFanout`]; the atomic read never parks —
    /// and only at the front of their own queue, separate from the error and
    /// delay queues. A read can consume any number of holds, since a hold
    /// fails nothing. Only the named command parks: the other commands of its
    /// fan-out window are issued meanwhile. The read does not succeed until
    /// the held command has been released and answered, but it can end
    /// without that: another command of the same window failing ends the
    /// window and drops the parked command, and so does the read being
    /// dropped (a lost fence lease, a cancelled caller). The park is an async
    /// wait: it blocks no OS thread. A hold the test never releases parks its
    /// command until the read ends, so a test that arms one must release it on
    /// every path.
    ///
    /// If the intended read ends without reaching the named command — an
    /// earlier injected or real error, a target past the read's last command,
    /// a read that went atomic — [`SnapshotReadHold::parked`] does not return
    /// for it, so a test that awaits it must bound that wait (for example with
    /// `tokio::time::timeout`). A hold is taken only by a fan-out command its
    /// target names, so an unreached hold stays queued
    /// ([`Self::holds_remaining`]), and it blocks every hold queued behind it,
    /// since only the front entry is checked. The next read on this client to
    /// reach its target then parks on it, unless the test has released it;
    /// releasing on every path covers that read too.
    ///
    /// # Panics
    ///
    /// On an [`ObjectSnapshotReadFaultTarget::retirement_size_read`] target:
    /// no fan-out command matches it, so the hold would never be taken, and
    /// every hold queued behind it would wait forever — a test arming it
    /// would pass without parking anything.
    pub fn push_hold_at(
        &self,
        target: ObjectSnapshotReadFaultTarget,
    ) -> std::sync::Arc<SnapshotReadHold> {
        target.assert_fanout_target("push_hold_at");
        let hold = std::sync::Arc::new(SnapshotReadHold::default());
        if let Ok(mut q) = self.holds.lock() {
            q.push_back((target, hold.clone()));
        }
        hold
    }

    /// How many holds are still queued, not yet taken by a read. A taken hold
    /// may still be parked: [`SnapshotReadHold::parked`] tells.
    pub fn holds_remaining(&self) -> usize {
        self.holds.lock().map(|q| q.len()).unwrap_or(0)
    }

    /// Stall (`true`) or resume (`false`) every fenced read's lease renewals
    /// on this client. While stalled, the renewal timer keeps its period but
    /// skips the renewal, so a read held longer than its lease loses it and
    /// its release answers `BROKEN`.
    pub fn stall_fence_renewals(&self, stalled: bool) -> &Self {
        self.renewals_stalled
            .store(stalled, std::sync::atomic::Ordering::SeqCst);
        self
    }

    /// Pop the front entry's error if `hits` says its target is the command
    /// at hand; otherwise leave the queue as it is.
    fn take_if(
        &self,
        hits: impl FnOnce(&ObjectSnapshotReadFaultTarget) -> bool,
    ) -> Option<fred::error::Error> {
        let mut q = self.queue.lock().ok()?;
        if hits(&q.front()?.0) {
            q.pop_front().map(|(_, err)| err)
        } else {
            None
        }
    }

    /// [`Self::take_if`] for the delay queue.
    fn take_delay_if(
        &self,
        hits: impl FnOnce(&ObjectSnapshotReadFaultTarget) -> bool,
    ) -> Option<Duration> {
        let mut q = self.delays.lock().ok()?;
        if hits(&q.front()?.0) {
            q.pop_front().map(|(_, delay)| delay)
        } else {
            None
        }
    }

    /// [`Self::take_if`] for the hold queue.
    fn take_hold_if(
        &self,
        hits: impl FnOnce(&ObjectSnapshotReadFaultTarget) -> bool,
    ) -> Option<std::sync::Arc<SnapshotReadHold>> {
        let mut q = self.holds.lock().ok()?;
        if hits(&q.front()?.0) {
            q.pop_front().map(|(_, hold)| hold)
        } else {
            None
        }
    }
}

/// One queued hold of [`ObjectSnapshotReadFaultInjector::push_hold_at`]: the
/// handle a test waits on and releases.
///
/// Two level latches, each set once and never cleared: **parked**, set when
/// the read reaches the held command, and **released**, set by the test. Both
/// are levels, not edges, so the order of the two sides does not matter: a
/// [`Self::parked`] awaited after the read parked returns at once, and a
/// [`Self::release`] made before the read arrives lets it pass without
/// parking (it still sets parked as it passes). One hold parks one command of
/// one read.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Default)]
pub struct SnapshotReadHold {
    parked: std::sync::atomic::AtomicBool,
    parked_notify: tokio::sync::Notify,
    released: std::sync::atomic::AtomicBool,
    released_notify: tokio::sync::Notify,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl SnapshotReadHold {
    /// Wait until the read has reached the held command and parked on it.
    /// Returns at once if it already has; never returns if no read reaches
    /// it, so bound the wait
    /// ([`ObjectSnapshotReadFaultInjector::push_hold_at`]).
    pub async fn parked(&self) {
        wait_for_latch(&self.parked, &self.parked_notify).await;
    }

    /// Whether the read has reached the held command.
    pub fn is_parked(&self) -> bool {
        self.parked.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Let the parked read issue the held command and go on. Idempotent; made
    /// before the read arrives, the read does not park at all.
    pub fn release(&self) {
        set_latch(&self.released, &self.released_notify);
    }

    /// The fan-out's side: mark parked, then wait for the release.
    async fn park(&self) {
        set_latch(&self.parked, &self.parked_notify);
        wait_for_latch(&self.released, &self.released_notify).await;
    }
}

/// Set a [`SnapshotReadHold`] latch and wake every waiter on it. Shared with
/// `external_values::ReadHoldTakeHold`, whose latches work the same way.
#[cfg(any(test, feature = "test-fault-injection"))]
pub(crate) fn set_latch(flag: &std::sync::atomic::AtomicBool, notify: &tokio::sync::Notify) {
    flag.store(true, std::sync::atomic::Ordering::SeqCst);
    notify.notify_waiters();
}

/// Wait until a [`SnapshotReadHold`] latch is set. The waiter registers with
/// `notify` before it reads the flag, so a `set_latch` racing the read either
/// is seen by the read or wakes the registered waiter — never lost. Shared
/// with `external_values::ReadHoldTakeHold`.
#[cfg(any(test, feature = "test-fault-injection"))]
pub(crate) async fn wait_for_latch(
    flag: &std::sync::atomic::AtomicBool,
    notify: &tokio::sync::Notify,
) {
    loop {
        let notified = notify.notified();
        tokio::pin!(notified);
        notified.as_mut().enable();
        if flag.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        notified.await;
    }
}

/// Which reader a positional [`ObjectSnapshotReadFaultTarget`] answers in.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjectSnapshotReader {
    /// [`read_object_snapshot`]'s unfenced bounded fan-out: the structural
    /// `UpdateLayer` path, at every object size. External-value clean-up's
    /// size read and reference walk (`storage::retirement`) consult the seam
    /// as this reader too, each on a cursor of its own.
    UnfencedFanout,
    /// The fan-out the fenced read runs under its lease
    /// (`storage::snapshot_fence::read::read_object_snapshot_fenced`, Spec SDD
    /// SR-5.3): the same commands in the same order as
    /// [`Self::UnfencedFanout`], counted afresh for every fenced read attempt.
    FencedFanout,
}

/// Which command a queued [`ObjectSnapshotReadFaultInjector`] error answers in
/// place of.
///
/// Commands are numbered from 0 per read, in the order the fan-out issues them
/// — the order below, which is the order it builds its commands in
/// (concurrent commands of one round are numbered in that order too, whatever
/// order their replies arrive in). Commands named by an index's members come
/// in the order round 1 collected them: rank order for a sorted index, and for
/// a set the order its `SSCAN` steps returned them, the server's order: to hit the command of one particular set
/// member, arm the seam on an object whose set holds only that member — or,
/// for an unmarked object's children lists, target the root's, which is
/// always the last.
///
/// - [`ObjectSnapshotReader::UnfencedFanout`] and
///   [`ObjectSnapshotReader::FencedFanout`]: round 1's first window is the
///   header (0) and the member counts of the sections index (1), the
///   deleted-sections index (2), the fields index (3) and the time-samples
///   index (4); then its page windows, each the next `SSCAN` step of every
///   set index still being scanned (sections, then deleted sections) and the
///   next `ZRANGE` page of every sorted index with ranks left (fields, then
///   time samples), an empty index having none. Round 2 is every section hash, every
///   tombstone hash, every field hash and every time-sample hash, then the
///   children: on an object without the `chs_complete` marker one `LRANGE`
///   per live section and last the root's; on a marked object the
///   `SMEMBERS obj:chs`, followed by round 3's `LRANGE`s of the lists it
///   names under a live parent, in its reply order. A fenced read counts
///   each attempt (a retry after a broken attempt is a new attempt) from 0.
///
/// The positional targets exist to stage b.zvd's defect shape — a transient
/// failure of one children `LRANGE`, which a reader that swallowed it would
/// return as a childless snapshot — on either fan-out:
/// [`Self::nth_of_family`] with `"children"` picks it out without counting
/// the commands before it.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectSnapshotReadFaultTarget(FaultTarget);

#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FaultTarget {
    NextRead,
    Command {
        reader: ObjectSnapshotReader,
        /// `None`: `nth` counts every command of the read; `Some`: only the
        /// commands of that family.
        family: Option<SnapshotKeyFamily>,
        nth: usize,
    },
    /// One command of external-value clean-up's size-read pipeline, by its
    /// key's family ([`ObjectSnapshotReadFaultTarget::retirement_size_read`]).
    /// Never named by a fan-out command, so [`target_names`] answers `false`.
    SizeReadCommand(SnapshotKeyFamily),
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl ObjectSnapshotReadFaultTarget {
    /// The seam's original target, and what
    /// [`ObjectSnapshotReadFaultInjector::push`] queues: the atomic read's
    /// `EVALSHA`, or on either fan-out (unfenced or fenced) the first command
    /// of the window that consults the seam — on a fresh read, round 1's
    /// header `HGETALL` — or a lone command a reader issues ahead of its pass
    /// (external-value clean-up's size read, which a retirement route issues
    /// before its reference walk).
    pub fn next_read() -> Self {
        Self(FaultTarget::NextRead)
    }

    /// Command `nth` (from 0, every family counted) of the next `reader` read
    /// that issues that many commands, in the order the type's doc lists.
    pub fn nth_command(reader: ObjectSnapshotReader, nth: usize) -> Self {
        Self(FaultTarget::Command {
            reader,
            family: None,
            nth,
        })
    }

    /// The `nth` command (from 0) of one key family in the next `reader` read
    /// that issues that many — `nth_of_family(reader, "children", 0)` is the
    /// read's first children-list `LRANGE`. `key_family` is the family's
    /// `key_family` metric label, one of
    /// [`metrics::SNAPSHOT_READ_KEY_FAMILIES`]; `"children"` is the lists
    /// themselves, `"children_index"` the marked object's `obj:chs`.
    ///
    /// # Panics
    ///
    /// On a `key_family` that is not one of those labels: a misspelt family
    /// would otherwise never match, and the test arming it would pass
    /// without injecting anything.
    pub fn nth_of_family(reader: ObjectSnapshotReader, key_family: &str, nth: usize) -> Self {
        let family = SnapshotKeyFamily::ALL
            .into_iter()
            .find(|f| f.label() == key_family)
            .unwrap_or_else(|| {
                panic!(
                    "{key_family:?} is not a snapshot-read key family; expected one of {:?}",
                    metrics::SNAPSHOT_READ_KEY_FAMILIES
                )
            });
        Self(FaultTarget::Command {
            reader,
            family: Some(family),
            nth,
        })
    }

    /// One command of external-value clean-up's size read
    /// (`storage::retirement::read_retirement_size`), by the `key_family` of
    /// the key it addresses: `"header"` its `HGET obj:hdr ext_gc_scheduled`,
    /// `"ext_index"` its `HLEN obj:ext`, and `"sections_index"`,
    /// `"deleted_sections_index"`, `"children_index"`, `"fields_index"` or
    /// `"time_samples_index"` that index's `SCARD` / `ZCARD`.
    ///
    /// Unlike every other target this one answers per command of the size
    /// read's one pipeline: the pipeline is sent whole, and the named
    /// command's reply is replaced with the queued error while the other six
    /// keep their real replies, so the error is counted and classified as
    /// that command's own failure would be. The size read still consults the
    /// seam first as one lone `header` command ([`Self::next_read`], or a
    /// `header` target at 0), which answers in place of the whole pipeline;
    /// this target is not that consult and never names it. Errors only
    /// ([`ObjectSnapshotReadFaultInjector::push_at`]): no fan-out command
    /// matches it, so [`ObjectSnapshotReadFaultInjector::push_delay_at`] and
    /// [`ObjectSnapshotReadFaultInjector::push_hold_at`] refuse it. Only the
    /// size read takes it, so at the front of the queue it waits for the next
    /// size read on this client, blocking the entries behind it.
    ///
    /// # Panics
    ///
    /// On a `key_family` that is not one of those seven: no other family's
    /// key is in the pipeline, so the target would never match and the test
    /// arming it would pass without injecting anything. The two queueing
    /// methods above panic, for the same reason, when given this target.
    pub fn retirement_size_read(key_family: &str) -> Self {
        // The size read's pipeline, in order (`storage::retirement`'s
        // `SIZE_READ_INDEX_COUNTS` after the `HGET` and the `HLEN`).
        const FAMILIES: [SnapshotKeyFamily; 7] = [
            SnapshotKeyFamily::Header,
            SnapshotKeyFamily::ExtIndex,
            SnapshotKeyFamily::SectionsIndex,
            SnapshotKeyFamily::DeletedSectionsIndex,
            SnapshotKeyFamily::ChildrenIndex,
            SnapshotKeyFamily::FieldsIndex,
            SnapshotKeyFamily::TimeSamplesIndex,
        ];
        let family = FAMILIES
            .into_iter()
            .find(|f| f.label() == key_family)
            .unwrap_or_else(|| {
                panic!(
                    "{key_family:?} names no command of the retirement size read; expected one \
                     of {:?}",
                    FAMILIES.map(SnapshotKeyFamily::label)
                )
            });
        Self(FaultTarget::SizeReadCommand(family))
    }

    /// Panic, naming `method`, unless a fan-out command can match this
    /// target: the delay and hold queues are consulted by the fan-outs only,
    /// so a [`Self::retirement_size_read`] target there would never be taken.
    fn assert_fanout_target(&self, method: &str) {
        if let FaultTarget::SizeReadCommand(family) = self.0 {
            panic!(
                "{method}: a retirement_size_read({:?}) target names no fan-out command, so it \
                 would never be taken; the size read's per-command seam injects errors only \
                 (push_at)",
                family.label()
            );
        }
    }
}

/// One read's position, as [`ObjectSnapshotReadFaults::take_fanout_command`]
/// counts it: which fan-out, how many commands it has issued in all and per family,
/// and whether the seam has already failed one of them. A reader makes one per
/// read and passes every command through it in issue order.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug)]
pub(crate) struct SnapshotReadFaultCursor {
    reader: ObjectSnapshotReader,
    commands: usize,
    /// Commands issued so far per family; a family not yet listed has issued
    /// none.
    per_family: Vec<(SnapshotKeyFamily, usize)>,
    /// Set once an error is injected: that command fails the read, so the
    /// read consults the seam no further.
    fired: bool,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl SnapshotReadFaultCursor {
    fn new(reader: ObjectSnapshotReader) -> Self {
        Self {
            reader,
            commands: 0,
            per_family: Vec::new(),
            fired: false,
        }
    }

    /// The cursor of one unfenced fan-out read.
    pub(crate) fn unfenced_fanout() -> Self {
        Self::new(ObjectSnapshotReader::UnfencedFanout)
    }

    /// The cursor of one fenced fan-out read attempt.
    pub(crate) fn fenced_fanout() -> Self {
        Self::new(ObjectSnapshotReader::FencedFanout)
    }

    /// Count one command of `family` and return its index in the read and its
    /// index within the family.
    fn advance(&mut self, family: SnapshotKeyFamily) -> (usize, usize) {
        let index = self.commands;
        self.commands += 1;
        let in_family = match self.per_family.iter_mut().find(|(f, _)| *f == family) {
            Some((_, n)) => {
                *n += 1;
                *n - 1
            }
            None => {
                self.per_family.push((family, 1));
                0
            }
        };
        (index, in_family)
    }
}

/// The handle [`crate::storage::client::ValkeyClient`] carries for
/// [`ObjectSnapshotReadFaultInjector`]: shared behind an `Arc`, so an injector
/// installed on one clone is seen by every other — a `LayerStore` built before
/// the injector was installed must still see it, or the seam would be a
/// construction-order trap.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Clone, Debug, Default)]
pub struct ObjectSnapshotReadFaults {
    slot: std::sync::Arc<std::sync::Mutex<Option<std::sync::Arc<ObjectSnapshotReadFaultInjector>>>>,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl ObjectSnapshotReadFaults {
    /// The inert seam: no injector installed, so every read takes the real
    /// path. Named rather than `Default::default()` because the construction
    /// site in `ValkeyClient` is shared by both cfg shapes, and the release
    /// shape's unit struct makes `::default()` a clippy error
    /// (`default_constructed_unit_structs`).
    pub fn none() -> Self {
        Self::default()
    }

    /// Install (or replace) the injector every read on this client consults.
    pub fn install(&self, injector: std::sync::Arc<ObjectSnapshotReadFaultInjector>) {
        if let Ok(mut slot) = self.slot.lock() {
            *slot = Some(injector);
        }
    }

    /// The installed injector, if any.
    fn injector(&self) -> Option<std::sync::Arc<ObjectSnapshotReadFaultInjector>> {
        self.slot.lock().ok()?.clone()
    }

    /// The atomic read's canned `EVALSHA` error: the front entry's, if an
    /// injector is installed and that entry targets
    /// [`ObjectSnapshotReadFaultTarget::next_read`].
    pub(crate) fn take_next_read(&self) -> Option<fred::error::Error> {
        self.injector()?
            .take_if(|target| target.0 == FaultTarget::NextRead)
    }

    /// What the seam does to the next command of a fan-out read (unfenced or
    /// fenced): one call per command, in issue order, with `window_first`
    /// true for the first command of a fan-out window (the command a
    /// [`ObjectSnapshotReadFaultTarget::next_read`] entry names). Counts the
    /// command on `cursor` once, records a window's first command with the
    /// recorder ([`ObjectSnapshotReadFaultInjector::record_window_threads`]),
    /// then takes the front hold and the front delay if each names this
    /// command, and the front error if it does — the latter only until one
    /// error has been injected in this read.
    pub(crate) fn take_fanout_command(
        &self,
        cursor: &mut SnapshotReadFaultCursor,
        family: SnapshotKeyFamily,
        window_first: bool,
    ) -> FanoutCommandFault {
        self.take_command(cursor, family, window_first, window_first)
    }

    /// What the seam does to a lone command a reader issues ahead of its
    /// pass, outside any fan-out window — external-value clean-up's size read
    /// (`storage::retirement::read_retirement_size`), on a cursor of its own:
    /// as [`Self::take_fanout_command`] for a window's first command, so a
    /// [`ObjectSnapshotReadFaultTarget::next_read`] entry names it, but not
    /// recorded as a window, since no fan-out issued it.
    pub(crate) fn take_lone_command(
        &self,
        cursor: &mut SnapshotReadFaultCursor,
        family: SnapshotKeyFamily,
    ) -> FanoutCommandFault {
        self.take_command(cursor, family, true, false)
    }

    /// The queued error for the size read's pipeline command on a key of
    /// `family` (`storage::retirement::read_retirement_size`): the front
    /// entry's, if it is an
    /// [`ObjectSnapshotReadFaultTarget::retirement_size_read`] target naming
    /// `family`. The size read asks once per command, in pipeline order, after
    /// the pipeline has answered, and stops at the first error it is given;
    /// no other reader asks, and no positional cursor counts the ask.
    pub(crate) fn take_size_read_command(
        &self,
        family: SnapshotKeyFamily,
    ) -> Option<fred::error::Error> {
        self.injector()?
            .take_if(|target| target.0 == FaultTarget::SizeReadCommand(family))
    }

    /// [`Self::take_fanout_command`] and [`Self::take_lone_command`]:
    /// `record` says whether the command opens a fan-out window the recorder
    /// notes.
    fn take_command(
        &self,
        cursor: &mut SnapshotReadFaultCursor,
        family: SnapshotKeyFamily,
        window_first: bool,
        record: bool,
    ) -> FanoutCommandFault {
        let (index, in_family) = cursor.advance(family);
        let reader = cursor.reader;
        let names = |target: &ObjectSnapshotReadFaultTarget| {
            target_names(target, reader, family, window_first, index, in_family)
        };
        let injector = self.injector();
        if record {
            if let Some(injector) = &injector {
                injector.note_window(reader);
            }
        }
        let hold = injector
            .as_ref()
            .and_then(|injector| injector.take_hold_if(names));
        let delay = injector
            .as_ref()
            .and_then(|injector| injector.take_delay_if(names));
        FanoutCommandFault {
            error: self.take_error(cursor, family, window_first, index, in_family),
            delay,
            hold: hold.map(FanoutCommandHold),
        }
    }

    /// Whether the fenced read's renewal timer must skip its renewals
    /// ([`ObjectSnapshotReadFaultInjector::stall_fence_renewals`]).
    pub(crate) fn fence_renewals_stalled(&self) -> bool {
        self.injector().is_some_and(|injector| {
            injector
                .renewals_stalled
                .load(std::sync::atomic::Ordering::SeqCst)
        })
    }

    /// The front error, if it names the command `cursor` just counted.
    fn take_error(
        &self,
        cursor: &mut SnapshotReadFaultCursor,
        family: SnapshotKeyFamily,
        window_first: bool,
        index: usize,
        in_family: usize,
    ) -> Option<fred::error::Error> {
        if cursor.fired {
            return None;
        }
        let reader = cursor.reader;
        let err = self.injector()?.take_if(|target| {
            target_names(target, reader, family, window_first, index, in_family)
        })?;
        cursor.fired = true;
        Some(err)
    }
}

/// Whether `target` names a command of `reader` that is the read's command
/// `index`, its family's `in_family`-th, and — for
/// [`ObjectSnapshotReadFaultTarget::next_read`] — the first of a fan-out
/// window.
#[cfg(any(test, feature = "test-fault-injection"))]
fn target_names(
    target: &ObjectSnapshotReadFaultTarget,
    reader: ObjectSnapshotReader,
    family: SnapshotKeyFamily,
    window_first: bool,
    index: usize,
    in_family: usize,
) -> bool {
    match target.0 {
        FaultTarget::NextRead => window_first,
        // Taken only by the size read's per-command consult
        // ([`ObjectSnapshotReadFaults::take_size_read_command`]).
        FaultTarget::SizeReadCommand(_) => false,
        FaultTarget::Command {
            reader: target_reader,
            family: target_family,
            nth,
        } => {
            target_reader == reader
                && match target_family {
                    None => nth == index,
                    Some(f) => f == family && nth == in_family,
                }
        }
    }
}

/// What the test-only seam does to one fan-out command
/// ([`ObjectSnapshotReadFaults::take_fanout_command`]): park it on `hold`
/// until the test releases it, hold it back for `delay`, then answer `error`
/// in its place (or issue it, when `error` is `None`). All three are always
/// `None` in release builds.
pub(crate) struct FanoutCommandFault {
    pub(crate) error: Option<fred::error::Error>,
    pub(crate) delay: Option<Duration>,
    pub(crate) hold: Option<FanoutCommandHold>,
}

impl FanoutCommandFault {
    /// A fault with nothing in it: the command is issued as it is. What a
    /// script window's plain re-issue after its fallback passes, without
    /// consulting the seam again.
    pub(crate) fn none() -> Self {
        Self {
            error: None,
            delay: None,
            hold: None,
        }
    }
}

/// The fan-out's view of a queued [`SnapshotReadHold`]: only
/// [`Self::park`], so the reader's call site is the same in both cfg shapes.
#[cfg(any(test, feature = "test-fault-injection"))]
pub(crate) struct FanoutCommandHold(std::sync::Arc<SnapshotReadHold>);

#[cfg(any(test, feature = "test-fault-injection"))]
impl FanoutCommandHold {
    /// Mark the hold parked and wait, without blocking a thread, until the
    /// test releases it.
    pub(crate) async fn park(self) {
        self.0.park().await;
    }
}

/// Production stand-in for the test shape's fan-out hold: uninhabited, so
/// [`FanoutCommandFault::hold`] is always `None` and the park is unreachable.
#[cfg(not(any(test, feature = "test-fault-injection")))]
pub(crate) enum FanoutCommandHold {}

#[cfg(not(any(test, feature = "test-fault-injection")))]
impl FanoutCommandHold {
    /// Unreachable: release builds construct no hold.
    #[inline]
    pub(crate) async fn park(self) {
        match self {}
    }
}

/// Production stand-in for [`ObjectSnapshotReadFaults`]: a zero-sized unit
/// struct whose [`Self::take_next_read`], [`Self::take_fanout_command`],
/// [`Self::take_lone_command`] and [`Self::take_size_read_command`] inject
/// nothing and whose
/// [`Self::fence_renewals_stalled`] is always `false`, so the readers consult
/// the seam with no `#[cfg]` at the call site and the release build folds the
/// branch away.
#[cfg(not(any(test, feature = "test-fault-injection")))]
#[derive(Clone, Debug, Default)]
pub struct ObjectSnapshotReadFaults;

#[cfg(not(any(test, feature = "test-fault-injection")))]
impl ObjectSnapshotReadFaults {
    /// Shape-independent counterpart of the test-shape constructor.
    pub fn none() -> Self {
        Self
    }

    /// Always `None`: release builds carry no injector.
    #[inline]
    pub(crate) fn take_next_read(&self) -> Option<fred::error::Error> {
        None
    }

    /// Always nothing: release builds carry no injector.
    #[inline]
    pub(crate) fn take_fanout_command(
        &self,
        _cursor: &mut SnapshotReadFaultCursor,
        _family: SnapshotKeyFamily,
        _window_first: bool,
    ) -> FanoutCommandFault {
        FanoutCommandFault::none()
    }

    /// Always nothing: release builds carry no injector.
    #[inline]
    pub(crate) fn take_lone_command(
        &self,
        _cursor: &mut SnapshotReadFaultCursor,
        _family: SnapshotKeyFamily,
    ) -> FanoutCommandFault {
        FanoutCommandFault::none()
    }

    /// Always `None`: release builds carry no injector.
    #[inline]
    pub(crate) fn take_size_read_command(
        &self,
        _family: SnapshotKeyFamily,
    ) -> Option<fred::error::Error> {
        None
    }

    /// Always `false`: release builds carry no injector.
    #[inline]
    pub(crate) fn fence_renewals_stalled(&self) -> bool {
        false
    }
}

/// Production stand-in for the test shape's per-read seam cursor: zero-sized,
/// so the readers thread it through with no `#[cfg]` at the call site.
#[cfg(not(any(test, feature = "test-fault-injection")))]
#[derive(Debug)]
pub(crate) struct SnapshotReadFaultCursor;

#[cfg(not(any(test, feature = "test-fault-injection")))]
impl SnapshotReadFaultCursor {
    /// Shape-independent counterpart of the test-shape constructor.
    #[inline]
    pub(crate) fn unfenced_fanout() -> Self {
        Self
    }

    /// Shape-independent counterpart of the test-shape constructor.
    #[inline]
    pub(crate) fn fenced_fanout() -> Self {
        Self
    }
}

// ── Checkpoint unprovable-cut seam (test-only) ──────────────────────────────
//
// Every checkpoint snapshot read proves its cut — atomically at or below the
// atomic-read limits, under the snapshot fence above them (Spec SDD SR-4.1,
// SR-11.5) — so only a header-sequence regression leaves the assembler without
// a cut to persist, and nothing short of a DR restore with an undersized band
// stages one. The checkpoint loop's tail-capture fallback sits behind that
// skip, and its tests need to reach it on demand. This seam is that route: an
// object armed on it has the checkpoint assembler's pin read answer above
// every cut, so the proven read that follows is judged regressed by the
// production arm (`checkpoint::assembler`, module docs).
//
// Declared here rather than in `checkpoint` because `ValkeyClient` carries it
// (the checkpoint loop is handed a client, and its tests drive that loop), and
// `storage` names no `checkpoint` type. Compiled only under
// `#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]`, as are the
// client's field and the seam's one consumer: the release library has none of
// them.

/// The set of objects whose checkpoint reads are forced to an unprovable cut;
/// the handle [`crate::storage::client::ValkeyClient`] carries.
///
/// Shared behind an `Arc`, so an object armed through one clone of a client is
/// armed for every clone, present or future — a checkpoint loop handed its
/// client by value must still see it.
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
#[derive(Clone, Debug, Default)]
pub struct UnprovableCutFaults {
    armed: std::sync::Arc<std::sync::Mutex<std::collections::HashSet<u64>>>,
}

#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
impl UnprovableCutFaults {
    /// The inert seam: nothing armed.
    pub fn none() -> Self {
        Self::default()
    }

    /// Arm `object_id`: from now on, until [`Self::disarm`], every checkpoint
    /// assembly of it is judged against a pin above every cut.
    pub fn arm(&self, object_id: u64) {
        if let Ok(mut armed) = self.armed.lock() {
            armed.insert(object_id);
        }
    }

    /// Disarm `object_id`: its next checkpoint assembly pins the real header
    /// sequence again.
    pub fn disarm(&self, object_id: u64) {
        if let Ok(mut armed) = self.armed.lock() {
            armed.remove(&object_id);
        }
    }

    /// Whether `object_id` is armed.
    pub fn is_armed(&self, object_id: u64) -> bool {
        self.armed
            .lock()
            .is_ok_and(|armed| armed.contains(&object_id))
    }
}

// ── ObjectSnapshot ──────────────────────────────────────────────────────────

/// All data needed for diff computation: an object's state as one
/// [`read_object_snapshot`] read it (atomically, or through the bounded
/// fan-out, unfenced or under the snapshot fence), carried as
/// [`AtomicSnapshot::state`].
///
/// An `obj:secs` member whose section hash is empty is left out of
/// [`Self::sections`]; on the proven paths (atomic and fenced) that is a
/// corrupt index, also reported on
/// [`AtomicSnapshot::corrupt_section_index_members`].
pub struct ObjectSnapshot {
    // Header fields
    pub object_id: u64,
    pub sequence_num: u64,
    pub first_version: u64,
    pub source_format: String,
    pub next_set_order: u64,

    // Structural data
    pub sections: HashMap<u64, Section>,
    pub children: HashMap<u64, Vec<u64>>,

    // Value data
    pub fields: Vec<Field>,
    pub time_samples: Vec<crate::domain::types::TimeSample>,

    // Tombstones: (section_id, first_version, last_version)
    pub tombstones: Vec<(u64, u64, u64)>,
}

// ── SubscriberState ─────────────────────────────────────────────────────────

/// A subscriber's last-known position, used to determine the diff case.
pub struct SubscriberState {
    /// The object generation the subscriber last received.
    pub have_object_id: u64,
    /// The sequence number the subscriber last received.
    pub have_version: u64,
}

// ── DiffResponse ───────────────────────────────────────────────────────────

/// The result of a diff computation: OMPF bytes ready for streaming to the subscriber.
pub struct DiffResponse {
    /// Serialized OMPF container. Empty if the subscriber is up to date.
    pub ompf_bytes: Vec<u8>,
    /// The diff case label (for metrics/logging).
    pub case: DiffCase,
    /// Number of struct command entries (sections + deleted sections) in the payload.
    pub sections_count: usize,
    /// The cut the payload was built at, and how it was read (Spec SDD
    /// SR-3.1).
    pub cut: DiffCut,
}

/// The cut a [`DiffResponse`] was built at (Spec SDD SR-3.1).
///
/// A payload built from a proven read — the atomic read, or the fenced read
/// above the atomic-read limits — carries a proven cut and nothing else, so no
/// variant can say "unproven".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffCut {
    /// No cut was asked for: a per-commit payload, which is labelled with the
    /// commit's own sequence number instead — whether it was built from an
    /// atomic read and version-bounded by its `upper_version`
    /// ([`compute_per_commit_diff_at_cut`]) or, for an empty commit, with no
    /// read ([`empty_commit_payload`]).
    Unpinned,
    /// Built from a snapshot proven at its cut — [`read_object_snapshot`]'s
    /// atomic read, or the fenced read (`storage::snapshot_fence::read`) a
    /// catch-up shares through its flight: every commit at or below `cut` is
    /// folded into the payload and no commit above it is. The
    /// two timestamps are the header's, read with `cut`, so the content
    /// header a catch-up stamps carries the cut's own timestamps (Spec SDD
    /// SR-3.3).
    Atomic {
        cut: u64,
        last_server_ts: u64,
        last_client_ts: u64,
    },
}

/// Which diff case was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffCase {
    FullContent,
    Incremental,
    UpToDate,
}

impl DiffCase {
    /// Prometheus label value.
    pub fn as_label(&self) -> &'static str {
        match self {
            DiffCase::FullContent => "full_content",
            DiffCase::Incremental => "incremental",
            DiffCase::UpToDate => "up_to_date",
        }
    }
}

// ── How a build reads its external values (b.yuo) ──────────────────────────

/// How a catch-up build reads the external values its plan references (b.yuo,
/// the catch-up miss protocol; SDD §11.2): [`Self::PROBE`], or the
/// fetch-on-miss mode [`Self::recover`] builds.
///
/// A value of this type, not an enum: the fetch-on-miss mode can be built
/// only from a [`LiveHoldProof`], which only a `{gc}` read hold the store
/// answered live produces
/// ([`ReadHold::confirm_live`](crate::storage::external_values::ReadHold::confirm_live)),
/// so the compiler — not
/// a source scan — keeps a build from reading through the checkpoint adapter
/// before the genuine-miss check. (A `test-support` build adds
/// [`Self::recover_for_test`] for the integration tests of the builders.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtValueReads(ExtReadMode);

/// The two read modes behind [`ExtValueReads`]; private, so a mode is chosen
/// only through its constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtReadMode {
    /// [`ExtValueReads::PROBE`].
    Probe,
    /// [`ExtValueReads::recover`].
    Recover,
}

impl ExtValueReads {
    /// [`ExternalValueStore::probe_values`]: a hash that is not stored makes
    /// the build answer [`Built::Missing`] with nothing built, nothing logged
    /// and the checkpoint adapter never consulted. Every catch-up build reads
    /// this way first, because the blob of a value its snapshot holds can be
    /// gone for a benign reason: a commit after the snapshot released it and
    /// the GC collected it.
    pub const PROBE: ExtValueReads = ExtValueReads(ExtReadMode::Probe);

    /// [`ExternalValueStore::read_batch_with_recovery_counter`]: a hash that
    /// is not stored is fetched through the checkpoint adapter
    /// (fetch-on-miss) and fails the build `INTERNAL` when that cannot restore
    /// it. For a hash still missing when a catch-up rebuilt from a re-read
    /// under the `{gc}` read hold, which makes it genuinely missing — which
    /// `_live` attests: the store answered the hold live on the genuine-miss
    /// check
    /// ([`ReadHold::confirm_live`](crate::storage::external_values::ReadHold::confirm_live)).
    /// Never answers [`Built::Missing`].
    pub fn recover(_live: &LiveHoldProof<'_>) -> Self {
        ExtValueReads(ExtReadMode::Recover)
    }

    /// The fetch-on-miss mode with no read hold behind it, for the
    /// integration tests that drive the builders directly (`test-support`
    /// only, never in a production build).
    #[cfg(any(test, feature = "test-support"))]
    pub fn recover_for_test() -> Self {
        ExtValueReads(ExtReadMode::Recover)
    }
}

/// What a catch-up build produced (b.yuo): the payload, or the external
/// values a [`ExtValueReads::PROBE`] build found missing, in which case it
/// built nothing.
#[must_use]
#[derive(Debug)]
pub enum Built<T> {
    /// Built.
    Ready(T),
    /// The hashes of the referenced external values that are not stored, in
    /// the plan's order. Only a [`ExtValueReads::PROBE`] build answers this.
    Missing(Vec<[u8; 32]>),
}

impl<T> Built<T> {
    /// The built value, or `Err` with the missing hashes.
    pub fn into_ready(self) -> Result<T, Vec<[u8; 32]>> {
        match self {
            Built::Ready(value) => Ok(value),
            Built::Missing(missing) => Err(missing),
        }
    }

    /// Apply `f` to a built value; a miss passes through.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Built<U> {
        match self {
            Built::Ready(value) => Built::Ready(f(value)),
            Built::Missing(missing) => Built::Missing(missing),
        }
    }
}

// ── compute_diff_at_cut ─────────────────────────────────────────────────────

/// The legacy version diff of a catch-up, built from its proven snapshot (Spec
/// SDD SR-3.1): plans the diff via domain logic ([`DiffEngine::plan`]),
/// resolves external values, builds the OMPF payload, and records metrics. It
/// emits everything newer than `subscriber.have_version`, with no upper bound:
/// the rollback-mode (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`) brand-new
/// and out-of-window catch-ups, from their atomic read at or below the
/// atomic-read ceilings and from their flight's shared fenced read above them.
/// The response's cut is [`DiffCut::Atomic`] at `snapshot.cut`. External
/// values are read as `ext_reads` says ([`Built::Missing`] under
/// [`ExtValueReads::PROBE`] when one is not stored).
///
/// `snapshot` must come from the atomic or the fenced path; an unfenced one is
/// refused `Internal`, because its cut is not proven ([`proven_cut`]).
pub async fn compute_diff_at_cut(
    snapshot: &AtomicSnapshot,
    subscriber: &SubscriberState,
    ext_store: &dyn ExternalValueStore,
    ext_reads: ExtValueReads,
    metrics: Option<&SharedMetrics>,
    service_label: &str,
) -> Result<Built<DiffResponse>, ServiceError> {
    let start = std::time::Instant::now();
    let object_id = snapshot.state.object_id.to_string();
    let cut = proven_cut(&object_id, snapshot)?;
    build_diff(
        &object_id,
        &snapshot.state,
        subscriber,
        None,
        cut,
        ext_store,
        ext_reads,
        metrics,
        service_label,
        start,
    )
    .await
}

/// [`compute_diff_at_cut`] for one commit, built from a legacy per-commit
/// atomic read (Spec SDD SR-3.4): a rollback-mode in-window replay's
/// commit at or below the atomic-read ceilings. A commit without relay bytes
/// above sequence 1 is an empty commit and takes no read at all
/// ([`empty_commit_payload`]), in the replay and on the steady-state path
/// alike.
///
/// Version-bounded by `upper_version`: the plan keeps only what changed in the
/// half-open interval `(subscriber.have_version, upper_version]`, so commits
/// the read's cut holds above `upper_version` are left out of this payload. The response's cut is therefore
/// [`DiffCut::Unpinned`], acked at the commit's own sequence number, not at the
/// read's cut.
///
/// `snapshot` must come from a proven path (in the service, always the atomic
/// one: a replay hands a layer above the atomic-read ceilings to catch-up's
/// coalesced route instead); an unfenced one is refused `Internal`
/// ([`proven_cut`]), as for the catch-up builds.
///
/// **A cut below `upper_version` is refused, here, so no caller can skip the
/// check** (Spec SDD SR-3.2's rule, applied to a per-commit read). A read
/// taken after the commit was streamed has its cut at or above the commit,
/// unless the layer's sequence regressed in between, as a restore with an
/// undersized sequence band does; its state then lacks changes the commit
/// made, and a diff built from it would be labelled `upper_version` while
/// missing them. So nothing is built and [`PerCommitDiff::Regressed`] is
/// returned instead. The caller owns what the refusal means for it: the
/// subscribe rollback-mode replay (`grpc::subscribe`'s
/// `build_per_commit_diff_at_cut`) WARNs, records an `unverified` stable cut
/// and ends the subscription `UNAVAILABLE` before any content.
///
/// External values are read as `ext_reads` says: under
/// [`ExtValueReads::PROBE`] a value that is not stored answers
/// [`Built::Missing`] and builds nothing. A regressed read answers
/// [`PerCommitDiff::Regressed`] before any value is read.
pub async fn compute_per_commit_diff_at_cut(
    snapshot: &AtomicSnapshot,
    subscriber: &SubscriberState,
    upper_version: u64,
    ext_store: &dyn ExternalValueStore,
    ext_reads: ExtValueReads,
    metrics: Option<&SharedMetrics>,
    service_label: &str,
) -> Result<Built<PerCommitDiff>, ServiceError> {
    let start = std::time::Instant::now();
    let object_id = snapshot.state.object_id.to_string();
    proven_cut(&object_id, snapshot)?;
    if snapshot.cut < upper_version {
        return Ok(Built::Ready(PerCommitDiff::Regressed {
            cut: snapshot.cut,
            to_version: upper_version,
        }));
    }
    Ok(build_diff(
        &object_id,
        &snapshot.state,
        subscriber,
        Some(upper_version),
        DiffCut::Unpinned,
        ext_store,
        ext_reads,
        metrics,
        service_label,
        start,
    )
    .await?
    .map(PerCommitDiff::Built))
}

/// What [`compute_per_commit_diff_at_cut`] built from a per-commit atomic read.
pub enum PerCommitDiff {
    /// The commit's single-version diff.
    Built(DiffResponse),
    /// The read's `cut` is below the commit it was asked to rebuild,
    /// `to_version`: the layer's sequence regressed (Spec SDD SR-3.2). Nothing
    /// was built.
    Regressed { cut: u64, to_version: u64 },
}

/// The per-commit payload for an empty commit, based at `base_version`, built
/// with no read: the steady-state path's and the rollback-mode in-window
/// replay's delivery for an `update` stream entry without relay bytes above
/// sequence 1, which on a current build is an empty commit (SDD §12.2).
///
/// Byte-identical to what [`compute_per_commit_diff_at_cut`] builds
/// ([`PerCommitDiff::Built`]) for that commit from a read whose cut covers
/// it: nothing in the layer is versioned in
/// `(base_version, commit]`, so its plan is [`DiffPlan::Incremental`] with
/// every list empty and no external value, and this hands
/// [`build_diff_payload`] exactly that plan. So the subscriber still receives
/// one message at the commit's sequence — an empty delta with `is_diff=true`
/// and `base_version`, never zero bytes — and its sequence stays contiguous.
/// The cut is [`DiffCut::Unpinned`], as for every per-commit payload.
///
/// `base_version` must be at least 1, which both callers guarantee: at 0 a
/// read would plan full content ([`DiffEngine::plan`]'s Case 1), not this.
pub fn empty_commit_payload(base_version: u64) -> Result<DiffResponse, ServiceError> {
    let plan = DiffPlan::Incremental {
        new_sections: Vec::new(),
        modified_sections: Vec::new(),
        deleted_sections: Vec::new(),
        changed_fields: Vec::new(),
        changed_time_samples: Vec::new(),
    };
    let scan = ExternalValueScan {
        ordered_hashes: Vec::new(),
        hash_to_index: HashMap::new(),
    };
    let ompf_bytes = build_diff_payload(&plan, "", base_version, &scan, &HashMap::new())?;
    Ok(DiffResponse {
        ompf_bytes,
        case: DiffCase::Incremental,
        sections_count: 0,
        cut: DiffCut::Unpinned,
    })
}

/// The [`DiffCut::Atomic`] a build from `snapshot` serves, or `Internal` when
/// `snapshot`'s cut is not proven (Spec SDD SR-3.1; policy: no snapshot is
/// served unverified). Proven: an atomic read's, and a fenced read's — every
/// `Fenced` snapshot outside `storage::snapshot_fence::read` is one its
/// release verified (`read_object_snapshot_fenced` returns nothing else, and
/// the unproven per-pass form, `read_object_snapshot_fenced_pass`, is
/// `pub(crate)` and called only there). Not proven: an unfenced read's `cut`,
/// only the sequence its first round saw, which a catch-up must never ack.
fn proven_cut(object_id: &str, snapshot: &AtomicSnapshot) -> Result<DiffCut, ServiceError> {
    match snapshot.path {
        SnapshotReadPath::Atomic | SnapshotReadPath::Fenced => Ok(DiffCut::Atomic {
            cut: snapshot.cut,
            last_server_ts: snapshot.last_server_ts,
            last_client_ts: snapshot.last_client_ts,
        }),
        other @ SnapshotReadPath::Unfenced => Err(ServiceError::Internal(format!(
            "catch-up build for object {object_id} was handed a snapshot read on the {} path, \
             whose cut is not proven",
            other.label()
        ))),
    }
}

/// The shared body of [`compute_diff_at_cut`] and
/// [`compute_per_commit_diff_at_cut`]: plan, resolve external values, build
/// the payload and record metrics, from a snapshot already read. Its steps are
/// numbered from 2 because step 1 is the caller's proven read.
#[allow(clippy::too_many_arguments)]
async fn build_diff(
    object_id: &str,
    snapshot: &ObjectSnapshot,
    subscriber: &SubscriberState,
    upper_version: Option<u64>,
    cut: DiffCut,
    ext_store: &dyn ExternalValueStore,
    ext_reads: ExtValueReads,
    metrics: Option<&SharedMetrics>,
    service_label: &str,
    start: std::time::Instant,
) -> Result<Built<DiffResponse>, ServiceError> {
    // 2. Convert snapshot to domain types.
    let state = snapshot_to_object_state(snapshot);
    let tombstones = &snapshot.tombstones;

    // 3. Plan the diff (pure domain logic).
    let plan = DiffEngine::plan(
        &state,
        tombstones,
        subscriber.have_object_id,
        subscriber.have_version,
        upper_version,
    )?;

    // 4. Scan for external values and fetch data from Valkey, as `ext_reads`
    //    says; a probe that finds one missing builds nothing. Under
    //    `Recover` the recovery counter is bumped from inside `read_batch`
    //    whenever the SR-3.2 ext-value fallback successfully reconstructs a
    //    missing blob via the configured `CheckpointStore`. We allocate
    //    the counter here so the operator-readable INFO log below can
    //    join `object_id` (in scope here, NOT in scope inside
    //    `read_batch`) with the recovered count — Subtask
    //    `t3.fgj.zc.p6.78`.
    let recovery_counter = AtomicUsize::new(0);
    let (scan, ext_data) = match resolve_external_values(
        &plan,
        ext_store,
        ext_reads,
        Some(&recovery_counter),
    )
    .await?
    {
        Built::Ready(resolved) => resolved,
        Built::Missing(missing) => return Ok(Built::Missing(missing)),
    };

    // 5. Build OMPF payload with external value data. Errors propagate as
    //    `Internal` — see `build_diff_payload` for the b.kf1 H-stor-10
    //    rationale on why a missing external blob must surface to the
    //    subscriber as INTERNAL rather than a silent zero-length payload.
    let ompf_bytes = build_diff_payload(
        &plan,
        &snapshot.source_format,
        subscriber.have_version,
        &scan,
        &ext_data,
    )?;

    // 6. Derive response metadata.
    let (case, sections_count) = match &plan {
        DiffPlan::UpToDate => (DiffCase::UpToDate, 0),
        DiffPlan::FullContent { sections, .. } => (DiffCase::FullContent, sections.len()),
        DiffPlan::Incremental {
            new_sections,
            modified_sections,
            deleted_sections,
            ..
        } => (
            DiffCase::Incremental,
            new_sections.len() + modified_sections.len() + deleted_sections.len(),
        ),
    };

    // 7. Record metrics.
    if let Some(m) = metrics {
        m.observe_diff_build(case.as_label(), service_label, start.elapsed());
    }

    // 8. Operator-observability INFO log (SDD SR-3.2, Subtask
    //    `t3.fgj.zc.p6.78`). Fires only when at least one ext-value
    //    hash was recovered via the checkpoint adapter during this
    //    build so successful no-miss calls stay silent —
    //    operators see the line iff a recovery actually happened, and
    //    it correlates `object_id` (this scope) with the count tick
    //    from inside `read_batch`. The per-outcome
    //    `live_edit_checkpoint_ext_recovery_total` counter is the
    //    aggregate-rate signal; this log is the per-call signal.
    let recovered_count = recovery_counter.load(Ordering::Relaxed);
    if recovered_count > 0 {
        tracing::info!(
            object_id = object_id,
            recovered_count = recovered_count,
            case = case.as_label(),
            "ext-value recovery via checkpoint adapter reconstructed missing blobs"
        );
    }

    Ok(Built::Ready(DiffResponse {
        ompf_bytes,
        case,
        sections_count,
        cut,
    }))
}

// ── compute_faithful_full_content_at_cut ─────────────────────────────────────

/// DR-independent faithful catch-up entrypoint for the brand-new / late-join
/// subscribe path (Plan b.msb, Task 2 wiring in `src/grpc/subscribe.rs`),
/// built from the catch-up's proven snapshot (Spec SDD SR-3.1): its atomic
/// read at or below the atomic-read ceilings, its flight's shared fenced read
/// above them. The response's cut is [`DiffCut::Atomic`] at `snapshot.cut`;
/// an unfenced snapshot is refused `Internal` ([`proven_cut`]).
///
/// Mirrors [`compute_diff_at_cut`]'s pipeline but routes through the domain faithful
/// builder [`DiffEngine::plan_faithful_snapshot`] instead of the version-diffing
/// [`DiffEngine::plan`]: the emitted plan is always a `DiffPlan::FullContent`
/// carrying net-live sections only (abandoned / unreachable prims dropped),
/// fields stamped with their TRUE authoring `set_order`, and no tombstones — the
/// shape a from-empty joiner needs.
///
/// Unlike the checkpoint assembler's `build_reset_ompf` (which is `pub(crate)`
/// inside the optional DR module, returns `CheckpointError`, attaches no
/// ext-value blobs, and inherits both catch-up defects), this path is always-on
/// and independent of `LIVE_EDIT_CHECKPOINT_ENABLED` (SR-4.1): it takes NO
/// `crate::checkpoint` dependency. A live from-empty joiner holds no external
/// values locally, so — exactly like [`compute_diff_at_cut`] — the actual blob
/// bytes are fetched via [`resolve_external_values`] and attached as OMPF
/// parts, not hash refs only.
///
/// `have_version = 0` / `base_version = 0` is threaded into
/// [`build_diff_payload`] so message #1 is a valid full-state create per SR-1.2.
///
/// External values are read as `ext_reads` says ([`Built::Missing`] under
/// [`ExtValueReads::PROBE`] when one is not stored).
pub async fn compute_faithful_full_content_at_cut(
    snapshot: &AtomicSnapshot,
    ext_store: &dyn ExternalValueStore,
    ext_reads: ExtValueReads,
    metrics: Option<&SharedMetrics>,
    service_label: &str,
) -> Result<Built<DiffResponse>, ServiceError> {
    let start = std::time::Instant::now();
    let cut = proven_cut(&snapshot.state.object_id.to_string(), snapshot)?;
    build_faithful_full_content(
        &snapshot.state,
        cut,
        ext_store,
        ext_reads,
        metrics,
        service_label,
        start,
    )
    .await
}

/// The body of [`compute_faithful_full_content_at_cut`], from a snapshot
/// already read. Its steps are numbered from 2 because step 1 is the caller's
/// proven read.
async fn build_faithful_full_content(
    snapshot: &ObjectSnapshot,
    cut: DiffCut,
    ext_store: &dyn ExternalValueStore,
    ext_reads: ExtValueReads,
    metrics: Option<&SharedMetrics>,
    service_label: &str,
    start: std::time::Instant,
) -> Result<Built<DiffResponse>, ServiceError> {
    // 2. Convert snapshot to domain types (carries Task 1's authoring
    //    `set_order` onto the domain `Field`s).
    let state = snapshot_to_object_state(snapshot);

    // 3. Plan the faithful full-content snapshot (pure domain logic): net-live
    //    sections only, fields carrying true authoring order, no tombstones.
    let plan = DiffEngine::plan_faithful_snapshot(&state)?;

    // 4. Resolve external values, as `ext_reads` says. A live joiner has no
    //    local ext values, so this fetches the actual blob bytes (under
    //    `Recover`, incl. the SR-3.2 recovery path) for attachment as OMPF
    //    parts. Recovery-counter participation is unnecessary here (the
    //    subscribe wiring has no per-call recovery log), so pass `None`.
    let (scan, ext_data) = match resolve_external_values(&plan, ext_store, ext_reads, None).await? {
        Built::Ready(resolved) => resolved,
        Built::Missing(missing) => return Ok(Built::Missing(missing)),
    };

    // 5. Build OMPF payload. `have_version = 0` → `base_version = 0` so the
    //    payload is a valid full-state create (SR-1.2). Actual ext-value blob
    //    bytes ride as OMPF Parts[1..N] (not hash-refs only).
    let ompf_bytes = build_diff_payload(&plan, &snapshot.source_format, 0, &scan, &ext_data)?;

    // 6. Derive response metadata. `plan_faithful_snapshot` only ever returns
    //    `FullContent`; the other arms are unreachable but handled exhaustively.
    let (case, sections_count) = match &plan {
        DiffPlan::FullContent { sections, .. } => (DiffCase::FullContent, sections.len()),
        DiffPlan::UpToDate => (DiffCase::UpToDate, 0),
        DiffPlan::Incremental {
            new_sections,
            modified_sections,
            deleted_sections,
            ..
        } => (
            DiffCase::Incremental,
            new_sections.len() + modified_sections.len() + deleted_sections.len(),
        ),
    };

    // 7. Record metrics.
    if let Some(m) = metrics {
        m.observe_diff_build(case.as_label(), service_label, start.elapsed());
    }

    Ok(Built::Ready(DiffResponse {
        ompf_bytes,
        case,
        sections_count,
        cut,
    }))
}

// ── compute_faithful_coalesced_at_cut ────────────────────────────────────────

/// DR-independent faithful catch-up entrypoint for a reconnector — BOTH the
/// in-window (Regime 2) and out-of-window / trimmed (Regime 3) paths under the
/// shipped default `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=true`, which converge
/// here and differ only in their regime metric label (Plan b.msb, Task 2/3
/// wiring in `src/grpc/subscribe.rs`). `false` reverts each regime to its
/// legacy path (per-commit replay for Regime 2, [`compute_diff_at_cut`] for
/// Regime 3). Built from the reconnect's proven snapshot (Spec SDD SR-3.1):
/// its atomic read at or below the atomic-read ceilings, its flight's shared
/// fenced read above them. The response's cut is [`DiffCut::Atomic`] at
/// `snapshot.cut`; an unfenced snapshot is refused `Internal`
/// ([`proven_cut`]).
///
/// Mirrors [`compute_faithful_full_content_at_cut`]'s pipeline but routes
/// through [`DiffEngine::plan_faithful_coalesced`] instead of the from-empty
/// [`DiffEngine::plan_faithful_snapshot`], and threads `have_version` (NOT `0`)
/// into [`build_diff_payload`] so the emitted payload is a valid coalesced
/// INCREMENTAL delta with `is_diff = true` and `base_version == have_version`.
///
/// This is the load-bearing difference from the from-empty joiner: a
/// reconnector (either regime) has `0 < have_version < current_seq` on the SAME
/// `object_id`, so its sequence has advanced past 0 and the client REJECTS a
/// `base_version = 0` from-empty create (SDD §12.3, §12.9). Emitting `Incremental`
/// with `base_version = have_version` keeps the exact envelope today's legacy
/// coalesced diff already satisfies while correcting the CONTENT: net-live prims
/// only, plus the unified ghost-retraction set (see the domain builder).
///
/// Like [`compute_faithful_full_content_at_cut`], this path is always-on and
/// takes NO `crate::checkpoint` dependency (independent of
/// `LIVE_EDIT_CHECKPOINT_ENABLED`, SR-4.1). A reconnector may not hold blobs
/// for newly-referenced hashes, so the actual ext-value blob bytes are fetched
/// via [`resolve_external_values`] and attached as OMPF parts, not hash-refs
/// only, read as `ext_reads` says ([`Built::Missing`] under
/// [`ExtValueReads::PROBE`] when one is not stored).
pub async fn compute_faithful_coalesced_at_cut(
    snapshot: &AtomicSnapshot,
    have_version: u64,
    ext_store: &dyn ExternalValueStore,
    ext_reads: ExtValueReads,
    metrics: Option<&SharedMetrics>,
    service_label: &str,
) -> Result<Built<DiffResponse>, ServiceError> {
    let start = std::time::Instant::now();
    let cut = proven_cut(&snapshot.state.object_id.to_string(), snapshot)?;
    build_faithful_coalesced(
        &snapshot.state,
        have_version,
        cut,
        ext_store,
        ext_reads,
        metrics,
        service_label,
        start,
    )
    .await
}

/// The body of [`compute_faithful_coalesced_at_cut`], from a snapshot already
/// read. Its steps are numbered from 2 because step 1 is the caller's proven
/// read.
#[allow(clippy::too_many_arguments)]
async fn build_faithful_coalesced(
    snapshot: &ObjectSnapshot,
    have_version: u64,
    cut: DiffCut,
    ext_store: &dyn ExternalValueStore,
    ext_reads: ExtValueReads,
    metrics: Option<&SharedMetrics>,
    service_label: &str,
    start: std::time::Instant,
) -> Result<Built<DiffResponse>, ServiceError> {
    // 2. Convert snapshot to domain types.
    let state = snapshot_to_object_state(snapshot);
    let tombstones = &snapshot.tombstones;

    // 3. Plan the faithful coalesced incremental diff (pure domain logic):
    //    net-live sections only, every child of a parent with a new or
    //    changed child restated at its sibling position, unified ghost
    //    retraction, dense-unique `set_order` ordered by the retained
    //    authoring rank (b.3pe).
    let plan = DiffEngine::plan_faithful_coalesced(&state, tombstones, have_version)?;

    // 4. Resolve external values, as `ext_reads` says (fetch actual blob
    //    bytes for OMPF parts; a reconnector may not hold blobs for
    //    newly-referenced hashes). No per-call recovery log on the subscribe
    //    path, so pass `None`.
    let (scan, ext_data) = match resolve_external_values(&plan, ext_store, ext_reads, None).await? {
        Built::Ready(resolved) => resolved,
        Built::Missing(missing) => return Ok(Built::Missing(missing)),
    };

    // 5. Build OMPF payload. `have_version` (NOT 0) → `base_version =
    //    have_version` so the reconnector's already-advanced sequence accepts
    //    message #1 (no from-empty reset). Actual ext-value blob bytes ride as
    //    OMPF Parts[1..N].
    let ompf_bytes = build_diff_payload(
        &plan,
        &snapshot.source_format,
        have_version,
        &scan,
        &ext_data,
    )?;

    // 6. Derive response metadata. `plan_faithful_coalesced` only ever returns
    //    `Incremental`; the other arms are unreachable but handled exhaustively.
    let (case, sections_count) = match &plan {
        DiffPlan::Incremental {
            new_sections,
            modified_sections,
            deleted_sections,
            ..
        } => (
            DiffCase::Incremental,
            new_sections.len() + modified_sections.len() + deleted_sections.len(),
        ),
        DiffPlan::FullContent { sections, .. } => (DiffCase::FullContent, sections.len()),
        DiffPlan::UpToDate => (DiffCase::UpToDate, 0),
    };

    // 7. Record metrics.
    if let Some(m) = metrics {
        m.observe_diff_build(case.as_label(), service_label, start.elapsed());
    }

    Ok(Built::Ready(DiffResponse {
        ompf_bytes,
        case,
        sections_count,
        cut,
    }))
}

/// Scan a DiffPlan for external value references and fetch the data, as
/// `ext_reads` says (b.yuo).
///
/// Returns the scan result and a hash→data map. If the plan contains no
/// external values, both are empty (no Valkey calls). Under
/// [`ExtValueReads::PROBE`] the fetch is
/// [`ExternalValueStore::probe_values`], and a hash that is not stored answers
/// [`Built::Missing`] with every such hash; under [`ExtValueReads::recover`]
/// it is [`ExternalValueStore::read_batch_with_recovery_counter`], which
/// recovers a missing hash through the checkpoint adapter or fails.
///
/// `recovery_counter` is plumbed through to
/// [`ExternalValueStore::read_batch_with_recovery_counter`] so a
/// successful SR-3.2 checkpoint-adapter fallback bumps it per recovered
/// hash. The version-diff builder (`build_diff`, behind
/// [`compute_diff_at_cut`] and [`compute_per_commit_diff_at_cut`]) reads it
/// back after the call to emit one operator-readable INFO log line per build
/// with the `object_id` joined onto the count — Subtask `t3.fgj.zc.p6.78`. Pass
/// `None` to skip counter participation. A probe never recovers, so it never
/// moves the counter.
async fn resolve_external_values(
    plan: &DiffPlan,
    ext_store: &dyn ExternalValueStore,
    ext_reads: ExtValueReads,
    recovery_counter: Option<&AtomicUsize>,
) -> Result<Built<(ExternalValueScan, HashMap<[u8; 32], Vec<u8>>)>, ServiceError> {
    let (fields, time_samples): (&[Field], &[crate::domain::types::TimeSample]) = match plan {
        DiffPlan::UpToDate => {
            return Ok(Built::Ready((
                ExternalValueScan {
                    ordered_hashes: Vec::new(),
                    hash_to_index: HashMap::new(),
                },
                HashMap::new(),
            )));
        }
        DiffPlan::FullContent {
            fields,
            time_samples,
            ..
        } => (fields, time_samples),
        DiffPlan::Incremental {
            changed_fields,
            changed_time_samples,
            ..
        } => (changed_fields, changed_time_samples),
    };

    let scan = collect_external_value_hashes(fields, time_samples);

    if scan.is_empty() {
        return Ok(Built::Ready((scan, HashMap::new())));
    }

    match ext_reads.0 {
        ExtReadMode::Probe => {
            let probe = ext_store.probe_values(&scan.ordered_hashes).await?;
            if !probe.missing.is_empty() {
                return Ok(Built::Missing(probe.missing));
            }
            Ok(Built::Ready((scan, probe.found)))
        }
        ExtReadMode::Recover => {
            // Fetch external value data from Valkey (pipelined). The counter
            // is bumped inside `read_batch_with_recovery_counter` per hash
            // whose blob was reconstructed via the SR-3.2 fallback.
            let ext_data = ext_store
                .read_batch_with_recovery_counter(&scan.ordered_hashes, recovery_counter)
                .await?;
            Ok(Built::Ready((scan, ext_data)))
        }
    }
}

/// Convert an `ObjectSnapshot` (storage layer) into an `ObjectState` (domain layer).
///
/// The one full-form decode every reader goes through (catch-up, the
/// checkpoint assembler). Its time samples are canonical:
/// a stored `-0.0` row is held at `+0.0`, and a stored signed-zero pair
/// resolves by `domain::types::resolve_signed_zero_pair`, the rule the
/// reconciler's merge applies, so the served state does not depend on read
/// order and does not change when the pair is merged
/// (`domain::types::time_samples_from_stored_rows`).
pub fn snapshot_to_object_state(snap: &ObjectSnapshot) -> ObjectState {
    let mut fields = HashMap::new();
    for f in &snap.fields {
        fields.insert((f.section_id, f.key_name.clone()), f.clone());
    }

    let time_samples =
        crate::domain::types::time_samples_from_stored_rows(snap.time_samples.iter().cloned());

    ObjectState {
        object_id: snap.object_id,
        sequence_num: snap.sequence_num,
        first_version: snap.first_version,
        source_format: snap.source_format.clone(),
        next_set_order: snap.next_set_order,
        sections: snap.sections.clone(),
        children: snap.children.clone(),
        fields,
        time_samples,
    }
}

/// Build a minimal state from just the object's header hash, with empty
/// section/children/field/time-sample maps, in either row form
/// ([`ObjectStateOf`]).
///
/// Used by `update_layer_once_watch`'s field-only fast path: when a delta contains no
/// structural commands and is not a reset, `apply_delta` only reads
/// `sequence_num` and `next_set_order` (both stored in the header) and only
/// writes to fields/samples — so loading the full snapshot is wasted work
/// that widens the WATCH window and amplifies contention. See the fast-path
/// predicate in `src/storage/layer.rs::update_layer_once_watch`.
pub fn header_to_object_state<R: ValueRows>(
    header: &HashMap<String, String>,
    object_id: u64,
) -> ObjectStateOf<R> {
    // b.k1r medium: warn on missing/corrupt u64 fields so a partial header
    // on the fast path does not silently produce a zeroed `sequence_num`
    // that would collide with the initial `1-0` stream entry.
    let owner = object_id.to_string();
    ObjectStateOf {
        object_id,
        sequence_num: parse_u64_warn(header, "seq", &owner),
        first_version: parse_u64_warn(header, "first_ver", &owner),
        source_format: header.get("src_fmt").cloned().unwrap_or_default(),
        next_set_order: parse_u64_warn(header, "next_set_order", &owner),
        sections: HashMap::new(),
        children: HashMap::new(),
        fields: Default::default(),
        time_samples: Default::default(),
    }
}

// ── Snapshot-read wrong-type observability (b.atj) ──────────────────────────

/// Which key family a failing snapshot read was issued against.
///
/// The `key_family` label vocabulary of
/// `live_edit_snapshot_read_type_error_total`, one variant per key family
/// [`read_object_snapshot`]'s reads touch (its atomic script and its bounded
/// fan-out alike): the header, the four indices, every section, tombstone,
/// field and time-sample row, the children lists, and — on an object carrying
/// the `chs_complete` marker — `obj:chs` ([`Self::ChildrenIndex`]); plus
/// `obj:ext` ([`Self::ExtIndex`]), which only external-value clean-up reads.
///
/// Three guards, of decreasing strength, keep this enum and the pre-registered
/// set in [`metrics::SNAPSHOT_READ_KEY_FAMILIES`] in step. Stated precisely,
/// because the ladder is weaker than it looks and only the first rung is
/// unconditional:
///
/// 1. [`Self::label`] has **no `_` arm**, so an added variant fails to compile
///    until it is given a LABEL. That much the compiler forces, always. It does
///    *not* reach the metrics array.
/// 2. The `const` assertion below [`Self::ALL`] compares `ALL`'s fixed length
///    against [`metrics::SNAPSHOT_READ_KEY_FAMILIES`]'s — but it can only fire
///    once the new variant has been LISTED in `ALL`. Listing it is a convention
///    the assertion rewards (it then fails to compile, naming the metrics array,
///    until that array grows too), not a step the compiler imposes.
/// 3. Set *equality* — as opposed to cardinality — is pinned by the unit test
///    `snapshot_key_family_all_matches_the_registered_label_set` at the bottom
///    of this file, which checks containment in both directions. It iterates
///    `ALL`, so it too sees only listed variants; and it is a test, not a
///    compile error. What it catches is a duplicated or omitted entry that
///    leaves the length intact.
///
/// **Residual hole.** A variant given a label (rung 1) and a call site, but
/// added to neither [`Self::ALL`] nor [`metrics::SNAPSHOT_READ_KEY_FAMILIES`],
/// compiles clean and leaves the unit test green — while emitting a
/// `key_family` value that was never registered, hence never pre-seeded to zero.
/// That defeats the `increase(...) > 0` alert recipe the pre-registration exists
/// to serve. Nothing in the build closes it: the review-time check is "every
/// `SnapshotKeyFamily` variant appears in `ALL`". The only shape that would
/// close it mechanically is a declarative macro minting the enum, `ALL` and the
/// label constants from one list — deliberately not built for twelve values.
///
/// `pub(crate)` for the reason [`classify_snapshot_read_err`] gives: the
/// checkpoint lane routes its two direct reads through the helper (b.muu) and
/// needs to name the family, as do `storage::orphan_sweep`'s `obj:hdr` discovery
/// read and retirement fence through [`count_object_key_refusal`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SnapshotKeyFamily {
    /// `le:{object_id}:obj:hdr` — `HGETALL` in [`read_object_snapshot`]'s
    /// bounded fan-out (unfenced, and the fenced read's pass), and the
    /// `HGET … seq` reads that hit the same key: the checkpoint
    /// assembler's pin and the structural path's proof after its unfenced
    /// read.
    ///
    /// On [`read_object_snapshot`] it also carries every failure of the
    /// `EVALSHA` itself (a `NOSCRIPT` reload failure, `-BUSY`, `-LOADING`, a
    /// timeout, a cluster redirect) besides the script's own header read: the
    /// header is the script's one declared key, so a script-level failure is
    /// attributed to it.
    ///
    /// Since b.muu also carries four touches issued outside this module on the
    /// same key: the checkpoint assembler's `HGET obj:hdr seq` and `HGETALL
    /// obj:hdr`, and the orphan sweep's `HGETALL obj:hdr` discovery read (its
    /// first touch of the key, upstream of everything else the sweep does to
    /// the object) and retirement fence `HSET obj:hdr superseded 1` — the one
    /// **write** on the counter, kept here rather than given a sibling because
    /// the key, the condition and the remediation are the read side's exactly.
    /// Four further `obj:hdr` touches in `storage::layer` remain uncounted
    /// (b.iy5), which is what leaves this family meaning four different things
    /// by consumer: see [`SNAPSHOT_READ_TYPE_ERROR_REMEDIATION`] for the
    /// criterion, that per-consumer reading, the per-site inventory and the
    /// exclusions the criterion drops and why (b.iy5 is the follow-up that
    /// routes them).
    Header,
    /// `le:{object_id}:obj:secs` — `SMEMBERS` (the atomic read), or `SCARD`
    /// then `SSCAN` steps (the bounded fan-out's round 1); and `SCARD` in
    /// external-value clean-up's size read (`storage::retirement`).
    SectionsIndex,
    /// `le:{object_id}:obj:dels` — as [`Self::SectionsIndex`].
    DeletedSectionsIndex,
    /// `le:{object_id}:obj:sec:{section_id}` — `HGETALL`.
    Section,
    /// `le:{object_id}:obj:del:{section_id}` — `HGETALL`.
    DeletedSection,
    /// `le:{object_id}:obj:ch:{parent_id}` — `LRANGE`.
    Children,
    /// `le:{object_id}:obj:chs` — `SCARD` / `SMEMBERS`, which the snapshot
    /// reads issue only on a marked object: the atomic read
    /// (`object_snapshot_read.lua`) and the bounded fan-out (`SMEMBERS`;
    /// unfenced, and the fenced read's pass). Every retirement route's size
    /// read (`storage::retirement`) issues `SCARD` on it, marked or not. A
    /// refusal on this key is labelled here, never
    /// as [`Self::Children`].
    ChildrenIndex,
    /// `le:{object_id}:obj:flds` — `ZRANGE` (the atomic read whole; the
    /// bounded fan-out's round 1 `ZCARD`, then `ZRANGE` pages by rank); and
    /// in external-value clean-up (`storage::retirement`), its size read's
    /// `ZCARD` and its reference walk's `ZRANGE` pages.
    FieldsIndex,
    /// `le:{object_id}:obj:tss` — as [`Self::FieldsIndex`].
    TimeSamplesIndex,
    /// `le:{object_id}:obj:fld:{section_id}:{hex(key_name)}` — `HGETALL`.
    Field,
    /// `le:{object_id}:obj:ts:{section_id}:{hex_f64(time)}` — `HGETALL`.
    TimeSample,
    /// `le:{object_id}:obj:ext` — the external-value reference index. No
    /// snapshot read touches it: external-value clean-up's size read (`HLEN`)
    /// and reference walk (`HSCAN`) do (`storage::retirement`). A foreign value here
    /// stops none of the four snapshot reads; the clean-up skips its release
    /// and removes the key. Edits refuse it on their own
    /// (`live_edit_keyspace_type_error_total`).
    ExtIndex,
}

impl SnapshotKeyFamily {
    /// Every variant, in the order [`metrics::SNAPSHOT_READ_KEY_FAMILIES`]
    /// registers them.
    ///
    /// The one place in the crate that enumerates the enum, and therefore the
    /// hinge of guards 2 and 3 in the type's doc — both of which see a variant
    /// only after it is listed here. Nothing in the build requires that listing
    /// (see the type's **Residual hole**), so keeping this array complete is the
    /// review-time obligation the other two guards hang off: with the variant
    /// listed, the fixed length below and the `const` assertion beside it force
    /// the metrics array to grow in the same edit.
    pub(crate) const ALL: [Self; 12] = [
        Self::Header,
        Self::SectionsIndex,
        Self::DeletedSectionsIndex,
        Self::Section,
        Self::DeletedSection,
        Self::Children,
        Self::ChildrenIndex,
        Self::FieldsIndex,
        Self::TimeSamplesIndex,
        Self::Field,
        Self::TimeSample,
        Self::ExtIndex,
    ];

    /// The metric label value.
    ///
    /// Returns the constants `observability::metrics` pre-registers — this enum
    /// never mints a label string of its own, so no label can be misspelled into
    /// existence here. It can still be *unregistered*: a variant labelled here
    /// but absent from [`Self::ALL`] and from
    /// [`metrics::SNAPSHOT_READ_KEY_FAMILIES`] emits a family neither the length
    /// assertion nor the set-equality test can see (the type's **Residual
    /// hole**). This match having no `_` arm is guard 1, and guard 1 only forces
    /// the label to exist.
    fn label(self) -> &'static str {
        match self {
            Self::Header => metrics::SNAPSHOT_KEY_FAMILY_HEADER,
            Self::SectionsIndex => metrics::SNAPSHOT_KEY_FAMILY_SECTIONS_INDEX,
            Self::DeletedSectionsIndex => metrics::SNAPSHOT_KEY_FAMILY_DELETED_SECTIONS_INDEX,
            Self::Section => metrics::SNAPSHOT_KEY_FAMILY_SECTION,
            Self::DeletedSection => metrics::SNAPSHOT_KEY_FAMILY_DELETED_SECTION,
            Self::Children => metrics::SNAPSHOT_KEY_FAMILY_CHILDREN,
            Self::ChildrenIndex => metrics::SNAPSHOT_KEY_FAMILY_CHILDREN_INDEX,
            Self::FieldsIndex => metrics::SNAPSHOT_KEY_FAMILY_FIELDS_INDEX,
            Self::TimeSamplesIndex => metrics::SNAPSHOT_KEY_FAMILY_TIME_SAMPLES_INDEX,
            Self::Field => metrics::SNAPSHOT_KEY_FAMILY_FIELD,
            Self::TimeSample => metrics::SNAPSHOT_KEY_FAMILY_TIME_SAMPLE,
            Self::ExtIndex => metrics::SNAPSHOT_KEY_FAMILY_EXT_INDEX,
        }
    }
}

/// Compile-time guard 2 (see [`SnapshotKeyFamily`]): the enum's cardinality and
/// the pre-registered label set's must agree.
///
/// Listing a thirteenth variant in [`SnapshotKeyFamily::ALL`] breaks that
/// array's fixed `[Self; 12]` length; widening it to `[Self; 13]` then breaks
/// this assertion, and the error points at [`metrics::SNAPSHOT_READ_KEY_FAMILIES`]
/// as the array that has not grown yet. Note the precondition: this fires only
/// for a variant somebody put in `ALL`. A variant labelled and called but never
/// listed there leaves both arrays at twelve and this assertion silent — the
/// residual hole named on [`SnapshotKeyFamily`], which review closes, not the
/// build. Cardinality only, besides — element-for-element equality is the job of
/// `snapshot_key_family_all_matches_the_registered_label_set`.
const _: () = assert!(
    SnapshotKeyFamily::ALL.len() == crate::observability::metrics::SNAPSHOT_READ_KEY_FAMILIES.len(),
    "SnapshotKeyFamily::ALL and metrics::SNAPSHOT_READ_KEY_FAMILIES must list the same families"
);

/// Operator-facing remediation sentence on an object-state touch refused by a
/// wrong-typed key (b.atj).
///
/// Deliberately not `storage::layer`'s `KEYSPACE_TYPE_ERROR_REMEDIATION`: that
/// one says "writes to this layer fail closed", which understates this: the
/// key can stop up to four snapshot consumers plus external-value clean-up of
/// a deleted or replaced layer — and that clean-up is often a background task
/// whose only evidence is the log line this constant rides on.
///
/// **One read, at every layer size, and which consumers stop depends only on
/// the key.** Four consumers read [`read_object_snapshot`]'s key set: at or
/// below the atomic-read limits catch-up, the checkpoint assembler and
/// per-commit reconstruction through its atomic read; above them catch-up (a
/// legacy replay's coalesced route included) through the fenced read's fan-out
/// its shared fenced-read flight runs (`storage::snapshot_flights`), and the
/// checkpoint assembler through a fenced read of its own; and the structural
/// commit path's WATCH route through its unfenced
/// fan-out at every size. No per-commit reconstruction is left above the
/// limits: a legacy in-window replay's first per-commit atomic read answers
/// `TOO_LARGE`, and the replay then reads through catch-up's coalesced route.
/// (A live commit never reconstructs: one without relay bytes is an empty
/// commit, sent with no read.) Under the `lua` backend a structural call that
/// commits through the structural write-planning cache reads the layer only on
/// a cold start or reload, through the same atomic read at the cache's
/// ceilings (`storage::structural_cache_cold_start`), so a refused key stops
/// only those calls through this read; a cached call reads the header, stream
/// and commit token through `structural_cache_sync.lua`, which this counting
/// does not cover. A cached commit (`update_layer_fast.lua`, structural mode)
/// still `TYPE`-probes every record and index key it writes, `obj:chs`
/// included whenever its plan writes a children list; a foreign value on such
/// a key refuses the commit with `WRONGTYPE_KEY`, counted on
/// `live_edit_keyspace_type_error_total` (`backend="lua"`, `site="update"`,
/// `mode="structural"`), not here. The cache's sampled audit
/// (`storage::structural_cache_audit`) issues that atomic read too, in a
/// background task: its refusal is counted and stops no call. Each form reads
/// the header, the four indices and every section, tombstone, field and
/// time-sample hash, so a refused key there stops all four consumers. On an
/// object carrying the `chs_complete` marker it reads `obj:chs` and only the
/// children lists it names under a live parent, so a refused `obj:chs` stops
/// all four at every size (the atomic read's size check, `SCARD obj:chs`, runs
/// before it can answer `TOO_LARGE`), and a refused list `obj:chs` does not
/// name is read by none of them; on an unmarked object it reads the lists of
/// the root and every live section, so a refused list stops all four. The
/// structural path's `HGET obj:hdr seq` proof is the one further touch, on the
/// header, which the others already read.
///
/// **The fifth consumer reads no snapshot.** External-value clean-up of a
/// deleted or replaced layer (`storage::retirement`) reads only sizes and
/// references: its size read, one pipeline of seven commands
/// (`HGET obj:hdr ext_gc_scheduled`, `HLEN obj:ext`, and the cardinality of
/// each structural index — `SCARD` of `obj:secs`, `obj:dels` and `obj:chs`,
/// `ZCARD` of `obj:flds` and `obj:tss`), and its reference walk (the
/// header's `EXISTS`, `obj:flds` and `obj:tss` by rank, their field and
/// time-sample rows, `obj:ext` by `HSCAN`), each failure counted under its
/// own family — `obj:ext` under [`SnapshotKeyFamily::ExtIndex`], each index
/// under its own. Where a refused key stops it depends on the read:
///
/// - **The size read.** A wrong-typed `obj:hdr` or `obj:ext` is counted and
///   skips the release, and the teardown removes the key. A wrong-typed
///   index is counted and WARNed, and the clean-up goes on at its full page
///   to the walk and the teardown, which then meet the index (below): a
///   wrong-typed `obj:flds` or `obj:tss` stops the release at the walk, and
///   every wrong-typed index stops the teardown. Any other refusal on any of the seven commands stops the
///   whole clean-up before it changes anything: `DeleteLayer` answers the
///   mapped error, and the overwrite body and the sweep keep the object.
/// - **The walk.** A refused row stops only the release (the
///   reference-count reconciler frees what was not released); the teardown
///   `UNLINK`s a row whatever its type. A refused `obj:flds` or `obj:tss`
///   stops the release here, and a wrong-typed one the teardown too (below).
/// - **The teardown.** After its first `UNLINK` it pages all five indices —
///   `obj:secs`, `obj:dels` and `obj:chs` by `SSCAN`, then `obj:flds` and
///   `obj:tss` by `ZRANGE` — so a wrong-typed index (or a refusal of its
///   page command that persists, which the size read's count did not meet)
///   stops it there. The five indices and every row not yet removed stay,
///   and on `DeleteLayer` so does the layers-index member; `DeleteLayer`
///   fails (`INTERNAL` for a wrong type) with the mapping already deleted
///   (b.yz5).
///
/// The teardown's index pages are not counted (b.yz5). The size read counts
/// each index before the teardown meets it — a wrong-typed one once, under
/// its own family, and `obj:flds` and `obj:tss` again at the walk — and the
/// teardown's failure names the index it stopped at, with the object and the
/// key. An orphan whose retirement fence failed, or whose id is not numeric,
/// gets no size read and no walk from the sweep: there only the teardown's
/// failure names a wrong-typed index.
///
/// The checkpoint assembler's two direct reads —
/// `ValkeySnapshotLoader::read_header_seq` (`HGET obj:hdr seq`) and
/// `read_header_identity` (`HGETALL obj:hdr`) — route through
/// [`classify_snapshot_read_err`] too (b.muu), each naming its own family. So do the orphan sweep's
/// two `obj:hdr` touches, both reaching [`count_object_key_refusal`] directly
/// rather than through this helper: its `HGETALL` **discovery read**
/// (`storage::orphan_sweep::sweep_orphaned_objects`), which is the sweep's first
/// touch of the key and `?`-propagates, so a refused header ends the pass before
/// any fence runs; and its retirement **fence write** (`HSET obj:hdr superseded
/// 1`, `cleanup_orphaned_object`) — the one non-read tick site, named on the
/// counter's own doc, and reached only on a header that read cleanly. So do
/// external-value clean-up's size read, through [`count_object_key_refusal`]
/// directly, and its reference walk, through this helper.
///
/// **What remains uncounted, and what that leaves [`SnapshotKeyFamily::Header`]
/// meaning per consumer.** The criterion, stated so a later reader can
/// re-derive the set from the key rather than trusting a count: an `obj:hdr`
/// touch in `storage::layer` ticked by neither this counter nor the commit-side
/// `keyspace_type_error_total` probe, type-sensitive enough to answer
/// `WRONGTYPE` at all, and with no counted touch of the same key ahead of it on
/// the same path. **Four** touches meet it, and they are why the `header` cell
/// means four different things by consumer: **counted on first touch** from
/// checkpoint assembly, from the orphan sweep and from external-value
/// clean-up (whose counted size read precedes `schedule_old_object_ext_gc`'s
/// own `HGET obj:hdr ext_gc_scheduled` on every route), none of which has an
/// uncounted touch in front of its counted one; **race-only** behind each of
/// two `storage::layer` header pre-reads, which meet a wrong-typed `obj:hdr` first
/// and map their own error; **blind** on `StatLayer`, where nothing counted
/// runs behind the header read at all, so the RPC fails closed with no metric,
/// no log line and no key name (`ServiceError::Internal` is scrubbed at the
/// gRPC boundary); and one **uncounted write**, the CreateLayer-overwrite
/// retirement fence, not race-shaped because a write shadows no read.
///
/// Those four are the follow-up Issue **b.iy5**, whose scope is all four of
/// them rather than a subset; b.iy5 also carries the checkpoint restore lane
/// and the `obj:stream` family. The four, by site: the two pre-reads
/// (`update_layer`'s slow-path `HGETALL obj:hdr` on the WATCHed connection →
/// `map_snapshot_load_err`; subscribe / catch-up's
/// `LayerStore::read_object_header` → `map_fred_err_for_grpc`),
/// `stat_layer_for_object_id`'s `HGETALL obj:hdr`,
/// and the CreateLayer-overwrite fence `HSET le:{old_oid}:obj:hdr superseded 1`
/// (`create_probe_keys` is built over the NEW `object_id`, so
/// `keyspace_type_error_total` does not reach it). The restore lane's reads
/// are `checkpoint::apply::header_is_structurally_complete`,
/// `checkpoint::residue`'s two `obj:hdr` re-reads and `enumerate_residue`'s
/// five index reads, all through `checkpoint::apply::classify_object_key_read_error`.
///
/// **Touches the criterion drops, and why** — counting any of them would be
/// wrong, not merely redundant:
/// - `schedule_old_object_ext_gc`'s `HGET obj:hdr ext_gc_scheduled` (its
///   re-entry guard → `map_transient_fred_err`) and the header touches of
///   `scripts/ext_retire_begin.lua` (`HGET` and `HSET obj:hdr
///   ext_gc_scheduled`, the retirement intent's begin): every route runs
///   them behind its counted size read of the same key
///   (`storage::retirement::read_retirement_size`), so they can only re-meet
///   a key already met.
/// - The three `WATCH` of the object header: type-agnostic — the command
///   cannot answer `WRONGTYPE`.
/// - The two MULTI-queued header `HSET`s and `write_restored_object`'s two
///   fence `HGETALL`s: probe-covered. `create_probe_keys` covers the
///   materializer write and the restore fences; `mutation_probe_keys` covers
///   the commit path's `StorageMutation::SetObjectHeader`. Both probe
///   `obj:hdr` on the same `object_id` first and tick
///   `keyspace_type_error_total`, so counting here too would double-count one
///   condition across two counters whose zero baselines mean different things.
/// - `EVALSHA update_layer_fast`: raises its own `WRONGTYPE_KEY:` verdict, not
///   a server `WRONGTYPE`; this counter's contract is "the server answered
///   `WRONGTYPE`".
///
/// **One touch has a wider radius than the per-layer sentence below.** The
/// orphan sweep's discovery read (`op` = `HGETALL obj:hdr (orphan discovery)`)
/// `?`-propagates out of `sweep_orphaned_objects` through `run_sweep`, so
/// `sweep_orphaned_layers` and the b.yda retire-retry drain `sweep_retire_retry`
/// never run on that pass either.
///
/// The three arms are not equally damaged, and the clause says so rather than
/// rounding up to "every object". The `?` fires inside the `while` over the
/// SCAN stream (`orphan_sweep.rs:~902`) and `cleanup_orphaned_object` runs
/// **inline, per object**, at `:~941` — so every object the pass had already
/// delivered was cleaned, and only the ones the SCAN had not yet reached stall.
/// The two later arms are exact: `sweep_orphaned_layers` and
/// `sweep_retire_retry` are never entered at all, so orphaned-mapping cleanup
/// and the deferred-release drain stall entirely. SCAN order is not stable
/// across passes, so a permanently-failing `le:{x}:obj:hdr` is nonetheless a
/// standing stall rather than a fixed prefix of progress — it repeats every
/// 300 s (`orphan_sweep::ORPHAN_SWEEP_INTERVAL`) pass, with `ref_count` leaking
/// meanwhile. The remediation string therefore carries an `op`-conditioned
/// clause; the sibling [`SNAPSHOT_READ_REFUSAL_REMEDIATION`] carries the same
/// clause, since the discovery read reaches both arms.
const SNAPSHOT_READ_TYPE_ERROR_REMEDIATION: &str =
    "reads of this layer fail closed until the named key is removed. One read loads this \
     object's state, at every layer size, for four consumers: structural UpdateLayer commits \
     (under the lua backend's structural cache, only on a cold start or reload, and its \
     sampled audit, which stops no commit; a cached commit refuses a foreign value on a record \
     or index key it writes, obj:chs included, on its own, as WRONGTYPE_KEY on \
     live_edit_keyspace_type_error_total), catch-up and the \
     checkpoint assembler (above the atomic-read limits through a fenced read, catch-up's \
     shared by the layer's joiners) and subscribe's per-commit reconstruction (a legacy \
     in-window replay, which above those limits reads through catch-up's shared fenced read). \
     A fifth, the external-value clean-up of a deleted or replaced layer, first reads the \
     header's release flag and the sizes of obj:ext and the five indices (obj:secs, obj:dels, \
     obj:chs, obj:flds, obj:tss) in one size read (op names the retirement size read), then \
     reads only the header, obj:flds, obj:tss, their field and time-sample rows and obj:ext \
     for its release; its key deletion then pages all five indices. A foreign value on the \
     header, an index or a section, tombstone, field or time-sample row stops all four. One on \
     the header or a field or time-sample row also stops the clean-up's release of the layer's \
     external values (the layer is still deleted; the reference-count reconciler frees them \
     later). One on any of the five indices is counted at the clean-up's size read, under that \
     index's key_family (obj:flds and obj:tss again at its release), and the clean-up goes on \
     with its full memory page; the orphan sweep's clean-up of an orphan whose retirement \
     fence failed, or whose object id is not numeric, runs no size read and no release, so \
     there it is not counted. Either way it then stops the clean-up's key deletion after it \
     has removed the header, and on obj:flds or obj:tss its release too. The five \
     indices and every row not yet removed stay, and on a DeleteLayer so does the layer's entry \
     in the session's layer list; the DeleteLayer answers INTERNAL with the layer already \
     unmapped. The key deletion's own failure line names the object and the index it stopped \
     at. A foreign value on obj:ext (key_family ext_index) stops none of the \
     four snapshot reads; the clean-up skips its release and removes the key, its references \
     freed later by the reference-count reconciler. Edits refuse it on their own \
     (live_edit_keyspace_type_error_total). On an object whose header \
     carries chs_complete=1 the read also reads obj:chs and only the children lists it names, \
     so a foreign value on obj:chs stops all four at every layer size (the atomic read's size \
     check reads obj:chs), and one on a children list obj:chs does not name stops none of \
     them; on any other object a children list stops all four. The orphan sweep's retirement \
     fence \
     writes the same header key. When op is HGETALL obj:hdr (orphan \
     discovery) the radius is wider than per-layer: that read aborts the rest of the \
     orphan-sweep pass, so orphan cleanup stalls for every object the pass had not yet reached, and orphaned-mapping \
     cleanup and the deferred-release drain stall entirely, on every 300s pass until this key is \
     removed. (Four header touches on the edit, subscribe, StatLayer and re-upload paths are \
     not yet counted on \
     live_edit_snapshot_read_type_error_total.) Nothing in \
     this service retypes a key it owns, so investigate the out-of-band writer that placed a \
     foreign value on the le: keyspace before \
     deleting it";

/// Operator-facing remediation sentence on an object-state touch refused by a
/// permanent, non-`WRONGTYPE` backend failure (b.muu).
///
/// Its own constant rather than a second clause on
/// [`SNAPSHOT_READ_TYPE_ERROR_REMEDIATION`] because the triage is the opposite
/// one. A wrong-typed key is a keyspace fault with a keyspace repair ("remove
/// the key, hunt the out-of-band writer"); everything counted here is a fault in
/// this service, its client library, its credentials or the backing store
/// itself, and nothing an operator does to the `le:` keyspace will fix it. The
/// `kind` label on `live_edit_snapshot_read_refused_total` is what splits those
/// further, so the sentence stays kind-agnostic and points at the label.
///
/// The blast radius is the wrong-typed case's — the same touches fail closed,
/// and which of the four snapshot consumers stop depends on the key (and, for
/// structural `UpdateLayer`, its route), by the split spelled out on
/// [`SNAPSHOT_READ_TYPE_ERROR_REMEDIATION`] — with one difference, at the
/// fifth consumer's size read: a wrong-typed key there (`obj:hdr` or
/// `obj:ext`, which skips the release, or a structural index, which only
/// widens the reservation to the full page) leaves the clean-up going on and
/// the layer still deleted, while a refusal of any of its seven commands
/// stops the whole clean-up before it changes anything;
/// the orphan sweep's retirement fence writes the same header key, and its
/// discovery read widens the radius to the whole sweep pass on either arm, for
/// the control-flow reason spelled out there too — which is why the sentence
/// repeats it rather than deferring to the sibling: a log line is read on its
/// own.
const SNAPSHOT_READ_REFUSAL_REMEDIATION: &str =
    "this object's state is unreadable until the named command stops failing. One read loads \
     it, at every layer size, for four consumers: structural UpdateLayer commits (under the lua \
     backend's structural cache, only on a cold start or reload, and its sampled audit, which \
     stops no commit), catch-up and the checkpoint assembler \
     (above the atomic-read limits through a fenced read, catch-up's shared by the layer's \
     joiners) and subscribe's per-commit reconstruction (a legacy in-window replay, which above \
     those limits reads through catch-up's shared fenced read). A fifth, the external-value \
     clean-up of a deleted or replaced layer, first reads the header's release flag and the \
     sizes of obj:ext and the five indices (obj:secs, obj:dels, obj:chs, obj:flds, obj:tss) in \
     one size read, then reads only the header, obj:flds, obj:tss, their field and time-sample \
     rows and obj:ext for its release; its key deletion then pages all five indices. A command \
     on the header, an \
     index or a section, tombstone, field or time-sample row stops all four. Any command of \
     the clean-up's size read (op names the retirement size read), on the header, obj:ext or \
     any of the five indices, also stops the clean-up before it changes anything: a \
     DeleteLayer of the layer fails, and a replaced or orphaned layer's keys stay for the \
     orphan sweep. One on a field or time-sample row, or on \
     the header or obj:ext later in the clean-up's reference walk, stops only the clean-up's \
     release of the layer's external values (the layer is still deleted; the reference-count \
     reconciler frees them later). One that persists on any of the five indices that the size \
     read's count did not meet (its count answered) stops the clean-up's key deletion after it \
     has removed the header, and on obj:flds or obj:tss its release too. The five indices and \
     every row not yet removed stay, and on a DeleteLayer so does the layer's entry in the \
     session's layer list; the DeleteLayer fails with the layer already unmapped. The key \
     deletion's own failure line names the object and the index it stopped at. On an object \
     whose \
     header carries chs_complete=1 the read also reads obj:chs and only the children lists it \
     names, so a command on obj:chs stops all four at every layer size (the atomic read's size \
     check reads obj:chs), and one on a children list obj:chs does not name stops none of them; \
     on any other object a children list stops all four. The orphan sweep's retirement fence writes \
     the same header key. When op is HGETALL obj:hdr (orphan discovery) the radius is \
     wider than per-layer: that read aborts the rest of the orphan-sweep pass, so orphan cleanup \
     stalls for every object the pass had not yet reached, and orphaned-mapping cleanup and the \
     deferred-release drain stall entirely, on every 300s pass until this command stops failing. \
     Unlike a wrong-typed key this is \
     not a keyspace fault and removing the key will not help — the kind label says which: \
     protocol / parse / invalid_argument / invalid_command / not_found are service or \
     client-library bugs, auth / config / tls are credential or deployment faults, routing is a \
     cluster-topology fault, backend_unavailable is the backing store refusing a write — \
     -MISCONF (it cannot persist data to disk; this lasts until a background save succeeds, and \
     on a full disk needs space freed), -READONLY (the write reached a read-only replica) or \
     -NOREPLICAS (too few replicas reachable to accept writes) — which reads never receive, so \
     it arrives here from the orphan sweep's header fence write alone; -LOADING while a node \
     replays its data and -BUSY under a long-running script are retried as transient and never \
     counted, so either one in the detail field means that classification has regressed; \
     backend_out_of_memory means used_memory exceeded maxmemory under \
     the required noeviction policy — the store will refuse every memory-growing (denyoom) write \
     until memory is freed or maxmemory raised; the GC's out-of-memory collection frees only \
     released external values, so a store full of data in use does not clear on its own, and \
     unknown is \
     the catch-all — an unmapped \
     fred kind or a server reply this service does not recognise: read the detail field for the \
     raw error, and note that a new fred release may have added a kind nothing here maps yet";

/// Whether a fred error is the backing store answering but declining to serve
/// (b.muu): a `-MISCONF`, `-READONLY` or `-NOREPLICAS` write refusal, or a
/// `-LOADING` / `-BUSY` not-serving reply — each bare or after an `EXEC`'s
/// `EXECABORT Transaction discarded because of: ` prefix.
///
/// **Counting, plus one startup verdict — never read-path classification.**
/// This predicate exists to split one metric cell out of `unknown`, and it must
/// never be consulted by the classification of a read or a commit. It has
/// exactly one other caller, [`run_snapshot_read_self_test`], where it decides
/// only "fatal or not" for the startup self-test: a store answering one of
/// these five is temporarily not serving, not refusing the script, so the
/// self-test reports `Unavailable` and startup continues instead of exiting
/// (exiting would crash-loop the pod through a primary's RDB replay). That is
/// not a retry decision: the self-test makes one attempt either way.
///
/// It is composed from the two classification predicates in `storage::client`
/// ([`crate::storage::client::is_store_not_serving_reply`] and
/// [`crate::storage::client::store_write_refusal`]) rather than restating
/// their tokens, so the cell and the classification match every token the
/// same way, trailing space included (b.xd1). Composing them does not make
/// this predicate steer anything: it still only counts.
///
/// # The live members: the write-gated three
///
/// `-MISCONF` (`stop-writes-on-bgsave-error`), `-READONLY` ("You can't write
/// against a read only replica") and `-NOREPLICAS` (`min-replicas-to-write`)
/// are Valkey refusals of **write** commands only. fred maps none of them
/// (`fred-10.1.0/src/protocol/utils.rs:96-129`, `_ => ErrorKind::Unknown` at
/// `:119`), and [`crate::storage::client::is_transient_fred_error`] calls
/// none of them transient, so each reaches [`count_object_key_refusal`]'s
/// counting arm. The touch set enumerated there is all reads but one — the
/// orphan sweep's retirement fence `HSET obj:hdr superseded 1` — so these
/// three are reachable here from that single write and nowhere else: at most
/// one tick per orphan per 300 s sweep pass, never a request-rate burst.
///
/// `-MISCONF` is also the one member that can need operator action: under
/// `stop-writes-on-bgsave-error yes` it lasts until a background save succeeds,
/// which on a full disk means until an operator frees space. It stays in this
/// cell rather than joining `-OOM` because it also arises transiently (the next
/// BGSAVE simply succeeds); the 15 m sustained-rate alert is what separates the
/// two — a burst that clears was transient, a rate outliving the window wants an
/// operator.
///
/// # `-LOADING` and `-BUSY`: a regression tripwire
///
/// Both refuse every command, reads included, and used to give this cell a
/// restart-shaped rate. They are now transient
/// ([`crate::storage::client::is_store_not_serving_reply`], b.xd1), so the
/// counting arm's `!is_transient_fred_error` gate keeps them out and a
/// healthy restart does not move the cell. They stay matched here so that a
/// change which made them permanent again would land them in this cell,
/// where the sustained-rate alert sees them, rather than in `unknown`.
///
/// # `-OOM` is deliberately NOT in this family
///
/// It has its own cell and its own predicate,
/// [`is_backend_out_of_memory_reply`]. What every member above has — the
/// property that licenses this cell's WARN level and its sustained-rate rule
/// rather than an `increase(...) > 0` page — is a *path back that can close
/// without an operator*: the next background save succeeds, the replica is
/// promoted, the replicas come back (and, for the tripwire pair, the replay
/// finishes, the script ends).
/// Under the `maxmemory-policy: noeviction` this project mandates (CLAUDE.md
/// § Deployment) `-OOM` has no such path for data in use: nothing is evicted,
/// the only memory the service frees by itself is what released external
/// values hold (the GC's out-of-memory collection, b.pa1), and a store full of
/// data in use refuses every `denyoom` command until an operator raises
/// `maxmemory` or removes data. Even a store that clears was sized too small.
/// Folding it in here would make this cell's documented contract false for
/// one member and route an operator-action condition to the no-page arm.
///
/// A request's write refused by `-OOM` (or by `-MISCONF`, `-READONLY`,
/// `-NOREPLICAS`) is not this touch set: the upload write paths answer it
/// `UNAVAILABLE` and count it on `live_edit_store_write_refused_total`, and
/// every other write answers it `UNAVAILABLE` through the mappers, uncounted
/// ([`crate::storage::client::store_write_refusal`]).
///
/// # Anti-creep
///
/// Kind-gated and prefix-anchored, never a substring search and never a bare
/// kind widening — the rule both composed predicates carry, and for the same
/// reason: `ErrorKind::Unknown` also carries `-UNBLOCKED`, the router's
/// message-matched failures, this service's own `WRONGTYPE_KEY:`-tagged Lua
/// verdict, and every other un-parsed server error. Every token is matched
/// with its trailing space, so `-BUSYGROUP` and `-BUSYKEY` are not matched by
/// the string, whatever the touch set. A future addition belongs in one of the
/// composed predicates, with the evidence they require.
fn is_backend_unavailable_reply(e: &fred::error::Error) -> bool {
    use crate::storage::client::StoreWriteRefusal;
    crate::storage::client::is_store_not_serving_reply(e)
        || matches!(
            crate::storage::client::store_write_refusal(e),
            Some(
                StoreWriteRefusal::Misconf
                    | StoreWriteRefusal::ReadOnly
                    | StoreWriteRefusal::NoReplicas
            )
        )
}

/// Whether a fred error is the backing store answering that it is full (b.muu):
/// an `-OOM` reply.
///
/// **Counting only** — it splits one metric cell out of `unknown` and must never
/// be consulted by classification. Unlike its sibling
/// [`is_backend_unavailable_reply`] it has no startup caller either: a store
/// over `maxmemory` refuses no read, so it cannot fail the self-test.
///
/// # The two Valkey origins, and which one is reachable here
///
/// 1. `valkey/src/server.c:~2240` — `shared.oomerr`,
///    `-OOM command not allowed when used memory > 'maxmemory'.`, returned for
///    any command flagged `denyoom` once `used_memory` exceeds `maxmemory`.
/// 2. `valkey/src/script.c:~200` —
///    `-OOM allow-oom flag is not set on the script, can not run it when used
///    memory > 'maxmemory'`, returned for a script that did not opt in.
///
/// Only the first is reachable on this counter's touch set. The second is
/// raised only for a *flagged* script, one with a `#!lua` shebang (Valkey's
/// `lua-api.md`, `allow-oom`), and the set's one script —
/// [`read_object_snapshot`]'s `object_snapshot_read.lua` — has none; every other
/// touch in the set (the reads of the bounded fan-out, unfenced and the
/// fenced read's pass, `SMEMBERS obj:chs` included; the structural path's `HGET
/// obj:hdr seq` proof, the checkpoint assembler's two, the orphan sweep's two,
/// external-value clean-up's size read)
/// is a plain key command. The prefix nonetheless matches both, which is the right
/// failure mode: if a future touch set gains a flagged script the cell is
/// already correct rather than silently landing in `unknown`.
///
/// Within the first origin, **reads are never `denyoom`**, so the only emitter
/// the counter can actually observe is the orphan sweep's retirement fence
/// `HSET obj:hdr superseded 1`
/// (`storage::orphan_sweep::cleanup_orphaned_object`) — the one write the
/// counter covers, and the reason this predicate did not need to exist before
/// b.muu widened the touch set to include a write.
///
/// # Anti-creep
///
/// Kind-gated and prefix-anchored, never a substring search and never a bare
/// kind widening — the same rule [`is_backend_unavailable_reply`] carries and
/// for the same reason: `ErrorKind::Unknown` is where every un-parsed server
/// error lands. It is [`crate::storage::client::store_write_refusal`]'s
/// `OutOfMemory` verdict (b.xd1), so it matches `OOM ` with its trailing
/// space, bare or after an `EXEC`'s `EXECABORT Transaction discarded because
/// of: ` prefix, exactly as the classification does; both origins above start
/// that way.
fn is_backend_out_of_memory_reply(e: &fred::error::Error) -> bool {
    crate::storage::client::store_write_refusal(e)
        == Some(crate::storage::client::StoreWriteRefusal::OutOfMemory)
}

/// Map a fred error onto its `live_edit_snapshot_read_refused_total{kind}` cell
/// (b.muu).
///
/// Returns a constant from
/// [`metrics::SNAPSHOT_READ_REFUSAL_KINDS`] — this function never mints a label
/// string of its own, the same discipline [`SnapshotKeyFamily::label`] follows.
///
/// **The `_` arm is the guard here, not a hole — the opposite reading of the
/// `_`-less match on [`SnapshotKeyFamily::label`], and the difference is
/// ownership.** `SnapshotKeyFamily` is this crate's enum, so exhaustiveness is a
/// free compile-time reminder that a new variant needs a label.
/// `fred::error::ErrorKind` is a third-party enum: a fred release can add a
/// variant, and `#[non_exhaustive]` is not what stops this match from
/// compiling — nothing is. With an arm per known kind and no `_`, a fred bump
/// that adds a kind would break the build; with the `_` arm, that kind lands on
/// `unknown`, a cell [`Metrics::new`] has already seeded. **Bounded cardinality
/// is the property the alert recipe rests on**, so the catch-all stays. Do not
/// "fix" it into an exhaustive match.
///
/// Two arms are not a plain kind mapping:
///
/// - `InvalidArgument` reaches here only when it is **not** a `WRONGTYPE` —
///   [`count_object_key_refusal`]'s first arm claims that case for the sibling
///   counter and returns before this function is called. The label is therefore
///   honest as written; it does not need to say "non-WRONGTYPE".
/// - `Unknown` splits three ways, on [`is_backend_out_of_memory_reply`] and
///   [`is_backend_unavailable_reply`], for the reasons those predicates
///   document. The two guards are disjoint by construction — `-OOM` is not in
///   the unavailable family's token list — so their order is not load-bearing;
///   the OOM guard is written first only because it is the narrower of the two,
///   which is the order a reader expects and the order that stays correct if
///   the other list is ever widened.
///
/// The transient kinds (`Timeout`, `IO`, `Canceled`, `Cluster`, `Backpressure`)
/// have no cell because they never reach here: the caller gates on
/// `!is_transient_fred_error(e)` first. Listing them would invite the impression
/// that this function decides anything, which it does not. The same gate keeps
/// the transient `-LOADING` / `-BUSY` out, so the `backend_unavailable` arm
/// sees them only if they stop being transient (the tripwire
/// [`is_backend_unavailable_reply`] documents).
fn snapshot_refusal_kind_label(e: &fred::error::Error) -> &'static str {
    use fred::error::ErrorKind as K;
    match e.kind() {
        K::Protocol => metrics::SNAPSHOT_READ_REFUSAL_KIND_PROTOCOL,
        K::Parse => metrics::SNAPSHOT_READ_REFUSAL_KIND_PARSE,
        K::Auth => metrics::SNAPSHOT_READ_REFUSAL_KIND_AUTH,
        K::Config => metrics::SNAPSHOT_READ_REFUSAL_KIND_CONFIG,
        // The variant itself is feature-gated in fred
        // (`fred-10.1.0/src/error.rs:38-51`), reached here through
        // `Cargo.toml`'s `valkey-tls = ["fred/enable-rustls-ring"]`. The arm is
        // gated to match; the *cell* is registered unconditionally so the
        // published series set does not depend on the build shape. An ungated
        // arm fails CLAUDE.md's third lint shape (`--no-default-features`).
        #[cfg(feature = "valkey-tls")]
        K::Tls => metrics::SNAPSHOT_READ_REFUSAL_KIND_TLS,
        K::InvalidArgument => metrics::SNAPSHOT_READ_REFUSAL_KIND_INVALID_ARGUMENT,
        K::InvalidCommand => metrics::SNAPSHOT_READ_REFUSAL_KIND_INVALID_COMMAND,
        K::Routing => metrics::SNAPSHOT_READ_REFUSAL_KIND_ROUTING,
        K::NotFound => metrics::SNAPSHOT_READ_REFUSAL_KIND_NOT_FOUND,
        K::Unknown if is_backend_out_of_memory_reply(e) => {
            metrics::SNAPSHOT_READ_REFUSAL_KIND_BACKEND_OUT_OF_MEMORY
        }
        K::Unknown if is_backend_unavailable_reply(e) => {
            metrics::SNAPSHOT_READ_REFUSAL_KIND_BACKEND_UNAVAILABLE
        }
        _ => metrics::SNAPSHOT_READ_REFUSAL_KIND_UNKNOWN,
    }
}

/// Count and log one failed touch of an object-state key, steering nothing
/// (b.muu).
///
/// The counting half of [`classify_snapshot_read_err`], split out so a call site
/// that must keep its **own** error mapping can still tick the same counters.
/// Three properties of the signature are load-bearing:
///
/// - **It borrows the error.** A site counts, then hands the owned `e` to
///   whatever mapper it already used — `map_fred_err_for_grpc`,
///   `map_transient_fred_err`, or none at all. That is what makes each of the
///   remaining `storage::layer` touches b.iy5 routes a one-line wrap rather
///   than a classification change, and it is why `classify_snapshot_read_err`
///   could not be adopted at those sites: its return type *is*
///   `map_snapshot_load_err`'s verdict.
/// - **It takes the metrics handle explicitly, not a `&ValkeyClient`.** The
///   orphan sweep threads `Option<&SharedMetrics>` down as a parameter
///   (`storage::orphan_sweep::sweep_orphaned_objects` and, below it,
///   `cleanup_orphaned_object`) and its tests wire
///   metrics through `spawn_orphan_sweep`'s argument rather than through
///   `ValkeyClient::install_observability_hooks`. A helper keyed on
///   `client.metrics_snapshot()` would compile there and silently count nothing
///   under test.
/// - **It returns `()`.** Nothing downstream can branch on it, which is the
///   b.zvd fail-closed posture restated: these predicates may count and log,
///   never steer.
///
/// `metrics = None` is a silent skip on every arm — the log lines are emitted
/// either way, so a metrics-less build or a poisoned metrics lock loses the
/// count but not the evidence.
///
/// **Exactly one counter, at most, per call** (the b.atj argument, restated for
/// the second counter). The two arms are mutually exclusive and ordered:
/// `WRONGTYPE` is claimed first by the unchanged
/// [`crate::storage::client::is_server_wrongtype_error`] and returns; only then
/// is the permanent/transient split consulted. So no refusal can tick both, and
/// `live_edit_snapshot_read_type_error_total`'s zero baseline ("no foreign value
/// on the `le:` keyspace") is untouched by this function's existence.
///
/// **The permanent/transient split is
/// [`crate::storage::client::is_transient_fred_error`] and nothing else**,
/// including its message-matched carve-outs for the router
/// (`is_router_no_connection_error`, `is_router_retry_budget_exhausted_error`),
/// for the store not serving (`is_store_not_serving_reply`: `-LOADING`,
/// `-BUSY`) and for a multi-key command during a slot migration
/// (`is_tryagain_reply`: `-TRYAGAIN`). That is the identical predicate
/// [`map_snapshot_load_err`] uses to choose `Aborted`, so arm 2 fires on
/// exactly the set that reaches a caller as `ServiceError::Internal`, minus
/// `WRONGTYPE`, plus the store write refusals the mapper answers plain
/// `Unavailable` (`storage::client::store_write_refusal`; on this touch set
/// only the orphan sweep's fence write can receive one, and the sweep does not
/// use the mapper). No second, drifting notion of
/// "permanent" is introduced.
///
/// **Deliberately not folded into [`map_snapshot_load_err`]**, which would be
/// the shorter edit and is wrong: `crate::storage::client::map_transient_fred_err`
/// is a bare alias of it, so a mapper-level tick would sweep in `WATCH obj
/// header`, `EVALSHA update_layer_fast`, `EVALSHA keyspace_type_probe`, EXEC
/// replies and `storage::tail_slice`'s two stream `XRANGE`s. The counter would
/// then mean "any permanent Valkey command failure anywhere" and its name would
/// be a lie. In scope is the enumerated set: the reads inside
/// [`read_object_snapshot`]'s script (each per-key refusal arrives tagged with
/// its family and key, and a failure of the `EVALSHA` itself counts as
/// [`SnapshotKeyFamily::Header`]), the reads of the bounded fan-out
/// (unfenced, and the fenced read's pass; each under its own family) and the
/// structural path's `HGET obj:hdr seq` proof after it
/// ([`SnapshotKeyFamily::Header`]), the checkpoint assembler's two direct
/// reads, the orphan sweep's two `obj:hdr` touches — its `HGETALL`
/// discovery read and its header fence write — and external-value
/// clean-up's reads (`storage::retirement`): its size read's seven commands,
/// each under its own key's family, and its reference walk's.
///
/// The checkpoint **restore** lane touches the same key families with the same
/// type sensitivity and is deliberately NOT in that set: its reads carry their
/// own ERROR line and their own restore-specific remediation and tick no
/// counter. Bringing it in rides the same follow-up Issue (b.iy5) as the
/// `storage::layer` touches; [`SNAPSHOT_READ_TYPE_ERROR_REMEDIATION`] names the
/// reads.
pub(crate) fn count_object_key_refusal(
    metrics: Option<&SharedMetrics>,
    object_id: &str,
    family: SnapshotKeyFamily,
    key: &str,
    op: &str,
    e: &fred::error::Error,
) {
    if crate::storage::client::is_server_wrongtype_error(e) {
        if let Some(m) = metrics {
            m.inc_snapshot_read_type_error(family.label());
        }
        tracing::error!(
            object_id = %object_id,
            key = %key,
            key_family = %family.label(),
            op = %op,
            detail = %e.details(),
            remediation = SNAPSHOT_READ_TYPE_ERROR_REMEDIATION,
            "object-state touch refused: the named key holds the wrong Valkey type; every \
             consumer whose read of this object's state touches it fails closed until it is \
             removed"
        );
    } else if !crate::storage::client::is_transient_fred_error(e) {
        let kind = snapshot_refusal_kind_label(e);
        if let Some(m) = metrics {
            m.inc_snapshot_read_refused(kind);
        }
        // One event, two levels. Written once through a local macro because the
        // field set must not drift between them: a dashboard or log-based alert
        // that keys on `kind` has to find the same keys whichever level fired.
        // (`tracing::event!` cannot take the level as a runtime value — its
        // metadata is a `static` — so an `if` over two invocations is the only
        // shape available, and the macro is what keeps that from meaning two
        // hand-maintained copies.)
        //
        // `detail` prints fred's message verbatim, `auth` cell included. Valkey
        // answers -NOAUTH / -WRONGPASS with a fixed sentence and echoes no
        // credential, and no credential transits this path in the first place —
        // so there is nothing to redact, and adding a redaction the WRONGTYPE
        // arm above does not have would only make the two lines disagree about
        // what an operator is allowed to see.
        macro_rules! refusal_event {
            ($level:ident) => {
                tracing::$level!(
                    object_id = %object_id,
                    key = %key,
                    key_family = %family.label(),
                    kind = %kind,
                    op = %op,
                    detail = %e.details(),
                    remediation = SNAPSHOT_READ_REFUSAL_REMEDIATION,
                    "object-state touch refused by a permanent backend failure; every consumer \
                     whose read of this object's state issues it fails closed until the command \
                     stops failing"
                )
            };
        }
        // WARN for `backend_unavailable`, ERROR for the other eleven. Its live
        // members — the orphan sweep's fence write refused `-MISCONF`,
        // `-READONLY` or `-NOREPLICAS` — are the store declining a write for a
        // condition that can clear on its own (the next background save, a
        // promotion, the replicas returning), and its alert recipe is
        // correspondingly a sustained-rate rule, not `increase(...) > 0`. The
        // transient `-LOADING` / `-BUSY` never reach this arm; they stay in the
        // cell only as a tripwire (`is_backend_unavailable_reply`). The other
        // eleven cells should never move at all, so each of those lines is a
        // page-worthy event and stays at ERROR. Emitting ERRORs for a condition
        // that clears on its own is how an operator learns to filter this
        // logger out, which would cost the eleven cells the evidence line their
        // alert depends on.
        //
        // The gate is this one cell by name, NOT "came from the backing store":
        // `backend_out_of_memory` is the counter-example that keeps the two from
        // being the same test. It is also the store answering rather than a
        // service bug, but under the mandated `noeviction` policy a store full
        // of data in use does not clear on its own — the GC's out-of-memory
        // collection frees only released external values, and the rest
        // persists until an operator raises `maxmemory` or frees memory — so it
        // wants the page and falls through to ERROR with the rest. Level only: the counter ticks
        // identically on both arms, and there is deliberately no rate-limiter or
        // any other process-global state here (see the b.zvd posture on this
        // helper — it may count and log, never steer or accumulate).
        if kind == metrics::SNAPSHOT_READ_REFUSAL_KIND_BACKEND_UNAVAILABLE {
            refusal_event!(warn);
        } else {
            refusal_event!(error);
        }
    }
    // Transient: nothing to count or log. On the read call sites that is because
    // the failure is retried — in-process where a `retry_aborted_with_jitter`
    // loop wraps the read, otherwise by whatever re-drives the operation (a
    // single-attempt path such as the SubscribeReadLayer catch-up hands the
    // client `UNAVAILABLE`, which it retries: the escaped `Aborted` maps there
    // in `src/error.rs`) — and a retried failure is not an operator-visible
    // event. The orphan sweep's two touches are silent by a different argument
    // each. Its discovery read does not retry in-pass — the `?` ends the sweep
    // pass, and the next periodic tick re-drives the whole SCAN from the top, so
    // a transient header failure costs one interval of sweep progress and
    // nothing else. Its fence write does not retry either — it skips ext-GC
    // scheduling (which leaks up to one `ref_count` per hash the orphan holds, the recoverable
    // direction that b.qpf's leak accounting and the ext-value reconciler own),
    // WARNs, and proceeds to UNLINK. Its transient fence failures are therefore
    // left to that existing `warn!` and to the leak accounting by choice, not
    // because a retry covers them.
}

/// Classify one failed snapshot read, counting and logging the refusal on the
/// way through (b.atj; second counter b.muu).
///
/// The classification itself is **unchanged** — every error, `WRONGTYPE`
/// included, is handed to [`map_snapshot_load_err`] verbatim, so the returned
/// variant and message are byte-identical to what the bare call produced. The
/// b.zvd fail-closed posture depends on that: see
/// [`crate::storage::client::is_server_wrongtype_error`] for why this predicate
/// may count and log but must never steer control flow. b.muu left that
/// property intact by moving the counting into [`count_object_key_refusal`],
/// which borrows the error and returns `()`: this function is now that call
/// followed by the same mapper it always ended in.
///
/// `pool_label` names the pool the failed read was issued on
/// ([`crate::storage::client::POOL_LABEL_DATA`] for the data pool,
/// [`crate::storage::client::POOL_LABEL_SNAPSHOT`] for the snapshot lane). It
/// decides only which `valkey_errors_total{pool, kind="command_timeout"}` cell
/// a command timeout ticks, never the returned variant.
///
/// **One tick per refused key met, not per key on the object** (b.atj's
/// argument, and it carries over to b.muu's counter unchanged because both ride
/// the same call). One planted
/// key ticks exactly once per refused call, which is the shape an operator
/// actually sees and the property the counter's baseline rests on: the bounded
/// fan-out's windows are `?`-propagated in sequence so a later window never
/// runs after an earlier one failed, and inside a window `try_join_all`'s
/// small-vec arm breaks out of the poll pass on the first `Err`, leaving the
/// sibling futures un-polled. The one shape that can exceed it is several
/// wrong-typed keys in the SAME window, when the window is large enough for
/// `try_join_all` to take its `FuturesOrdered` arm (> 30 futures), where more
/// than one sibling can complete in the pass that errors.
/// That is a louder count of a louder fault, so it is left alone: suppressing it
/// would need per-call state whose only effect is to under-report.
///
/// `key` is logged in full (hex-encoded field-name segments and all): it is the
/// single thing an operator needs, and the client-visible surface is unchanged
/// because `ServiceError::Internal` is scrubbed to a generic message at the
/// gRPC boundary.
///
/// **`pub(crate)` for its second caller, which b.muu landed.**
/// `checkpoint::assembler::ValkeySnapshotLoader` calls this for its two direct
/// reads, which used to call [`crate::storage::client::map_snapshot_load_err`]
/// straight: `read_header_seq` (`HGET obj:hdr seq`) and `read_header_identity`
/// (`HGETALL obj:hdr`), both with [`SnapshotKeyFamily::Header`]. Each wraps the
/// result in the `service_err_to_checkpoint_err` it already used, and since the
/// `ServiceError` is identical by construction that conversion's verdict is
/// unchanged. `load_consistent_snapshot` is not one of them — it issues no fred
/// command itself, calling through `&dyn SnapshotLoader`, and reaches only the
/// first of the two (the identity read is `assemble_inner`'s, issued after the
/// snapshot). The direction of that edge is the one SR-4.1 already mandates
/// (`checkpoint → storage::diff_read`), so it adds no module dependency.
///
/// A site that needs its **own** mapping calls [`count_object_key_refusal`]
/// instead — as `storage::orphan_sweep`'s discovery read (which must keep
/// `SweepError::from_fred`) and its retirement fence do, and as the remaining
/// `storage::layer` `obj:hdr` touches b.iy5 routes will. The criterion that
/// picks those out, the per-site inventory and the exclusions it drops are on
/// [`SNAPSHOT_READ_TYPE_ERROR_REMEDIATION`].
pub(crate) fn classify_snapshot_read_err(
    client: &ValkeyClient,
    pool_label: &'static str,
    object_id: &str,
    family: SnapshotKeyFamily,
    key: &str,
    op: &str,
    e: fred::error::Error,
) -> ServiceError {
    count_object_key_refusal(
        client.metrics_snapshot().as_ref(),
        object_id,
        family,
        key,
        op,
        &e,
    );
    // `pool_label` is the pool the failed read was issued on, so a command
    // timeout is counted against the connection that actually stalled:
    // `POOL_LABEL_DATA` for a read on the data pool, `POOL_LABEL_SNAPSHOT` for
    // one on the snapshot lane.
    map_snapshot_load_err(FredErrOrigin::client(client, pool_label), op, e)
}

// ── Shared per-entry decoding ───────────────────────────────────────────────
//
// One decoder per entry family, used by both of `read_object_snapshot`'s
// readers — the atomic script's reply and the bounded fan-out (unfenced, and
// the fenced read's pass) — so the two turn the same hashes and lists into the
// same state by construction: the atomic read's equality contract with the
// fan-out (Spec SDD SR-1.2) rests on it.

/// Decode one section hash. `None` for an empty hash (an indexed section
/// whose hash is gone), which both readers skip.
fn decode_section(sid_str: &str, data: &HashMap<String, String>) -> Option<Section> {
    if data.is_empty() {
        return None;
    }
    Some(Section {
        section_id: sid_str.parse::<u64>().unwrap_or(0),
        parent_id: parse_u64(data, "parent_id").unwrap_or(0),
        name: data.get("name").cloned().unwrap_or_default(),
        section_type: data
            .get("type")
            .and_then(|s| s.parse::<u8>().ok())
            .unwrap_or(0),
        section_order: data
            .get("order")
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0),
        first_version: parse_u64(data, "first_ver").unwrap_or(0),
        last_version: parse_u64(data, "last_ver").unwrap_or(0),
    })
}

/// Decode one tombstone hash into `(section_id, first_version,
/// last_version)`. `None` for an empty hash, which both readers skip.
fn decode_tombstone(sid_str: &str, data: &HashMap<String, String>) -> Option<(u64, u64, u64)> {
    if data.is_empty() {
        return None;
    }
    let section_id = sid_str.parse::<u64>().unwrap_or(0);
    let first_ver = parse_u64(data, "first_ver").unwrap_or(0);
    let last_ver = parse_u64(data, "last_ver").unwrap_or(0);
    Some((section_id, first_ver, last_ver))
}

/// Decode one parent's children list. `None` when the list holds no
/// parseable child id, which both readers leave out of `children`.
fn decode_children_list(pid_str: &str, child_strs: &[String]) -> Option<(u64, Vec<u64>)> {
    if child_strs.is_empty() {
        return None;
    }
    let pid = pid_str.parse::<u64>().unwrap_or(0);
    let child_ids: Vec<u64> = child_strs
        .iter()
        .filter_map(|s| s.parse::<u64>().ok())
        .collect();
    (!child_ids.is_empty()).then_some((pid, child_ids))
}

/// Decode an object's field rows `(section_id, key_name, hash)`, in the order
/// given, skipping empty hashes.
///
/// # Errors
///
/// [`ServiceError::Internal`] for a row whose external `hash` column is
/// malformed (see [`decode_field_value`]).
fn decode_field_rows(
    object_id: &str,
    rows: Vec<(String, String, HashMap<String, String>)>,
) -> Result<Vec<Field>, ServiceError> {
    let mut decoder = FieldRowDecoder::with_capacity(rows.len());
    for (sid_str, key_name, data) in rows {
        decoder.push(object_id, &sid_str, key_name, &data)?;
    }
    Ok(decoder.finish(object_id))
}

/// The incremental form of [`decode_field_rows`]: rows are decoded as they are
/// pushed, and the aggregated warn lines are emitted once, at
/// [`Self::finish`]. The bounded fan-out pushes each window's rows as the
/// window completes, so no raw reply outlives its window.
#[derive(Default)]
struct FieldRowDecoder {
    fields: Vec<Field>,
    // b.3pe: `set_order` and `version` are written by BOTH commit paths on
    // every field write, so a missing/corrupt column means a partial or corrupt
    // field hash — do not silently zero it (a zeroed `version` mis-sorts the
    // catch-up builders' dense re-index; a zeroed `set_order` mis-orders the
    // legacy paths that emit it verbatim). Because the trigger is all-or-nothing
    // across the object's rows, the misses are TALLIED here and reported as one
    // aggregated warn line per column at `finish` rather than per field.
    set_order_misses: ColumnMissTally,
    version_misses: ColumnMissTally,
    // b.zvd: rows carrying both value shapes at once. See
    // `ValueShapeConflictTally` for why the reader reports rather than repairs.
    shape_conflicts: ValueShapeConflictTally,
}

impl FieldRowDecoder {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            fields: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// Decode one field row `(section_id, key_name, hash)`, skipping an empty
    /// hash. Fails on a malformed external `hash` column (see
    /// [`decode_field_value`]).
    fn push(
        &mut self,
        object_id: &str,
        sid_str: &str,
        key_name: String,
        data: &HashMap<String, String>,
    ) -> Result<(), ServiceError> {
        if data.is_empty() {
            return Ok(());
        }
        let section_id = sid_str.parse::<u64>().unwrap_or(0);
        self.shape_conflicts
            .observe(data, section_id, || key_name.clone());
        let value = decode_field_value(
            data,
            object_id,
            format_args!("field ({section_id}, {key_name:?})"),
        )?;
        let set_order =
            self.set_order_misses
                .parse_or_tally(data, "set_order", section_id, &key_name);
        // t3.msb.35.ey.9u: raw per-delta authoring rank. Falls back to the
        // cumulative `set_order` when the hash field is absent (pre-release,
        // no on-disk migration of already-persisted field hashes) — absence is
        // EXPECTED on those rows, so this one stays quiet and untallied.
        let authoring_set_order = parse_u64(data, "authoring_set_order").unwrap_or(set_order);
        let version = self
            .version_misses
            .parse_or_tally(data, "version", section_id, &key_name);
        self.fields.push(Field {
            section_id,
            key_name,
            value,
            set_order,
            authoring_set_order,
            version,
        });
        Ok(())
    }

    /// Emit the aggregated warn lines and return the fields, in push order.
    fn finish(self, object_id: &str) -> Vec<Field> {
        self.set_order_misses.emit("set_order", object_id);
        self.version_misses.emit("version", object_id);
        self.shape_conflicts.emit("field", object_id);
        self.fields
    }
}

/// Decode an object's time-sample rows `(section_id, time, hash)`, in the
/// order given, skipping empty hashes.
///
/// # Errors
///
/// [`ServiceError::Internal`] for a row whose external `hash` column is
/// malformed (see [`decode_field_value`]).
fn decode_time_sample_rows(
    object_id: &str,
    rows: Vec<(String, f64, HashMap<String, String>)>,
) -> Result<Vec<crate::domain::types::TimeSample>, ServiceError> {
    let mut decoder = TimeSampleRowDecoder::with_capacity(rows.len());
    for (sid_str, time_f64, data) in rows {
        decoder.push(object_id, &sid_str, time_f64, &data)?;
    }
    Ok(decoder.finish(object_id))
}

/// Decode one stored time-sample row — `data` is its `HGETALL` — with the
/// full readers' decoder ([`TimeSampleRowDecoder`]): `None` for an absent
/// (empty) row, the decoded sample otherwise, at `time` exactly as given.
///
/// For the reference-count reconciler's signed-zero merge
/// (`storage::time_sample_merge`), which must decide on exactly the samples
/// the readers decode.
///
/// # Errors
///
/// As [`decode_time_sample_rows`]: a malformed external `hash` column.
pub(crate) fn decode_time_sample_row(
    object_id: &str,
    section_id: &str,
    time: f64,
    data: &HashMap<String, String>,
) -> Result<Option<crate::domain::types::TimeSample>, ServiceError> {
    let mut decoder = TimeSampleRowDecoder::with_capacity(1);
    decoder.push(object_id, section_id, time, data)?;
    Ok(decoder.finish(object_id).pop())
}

/// The incremental form of [`decode_time_sample_rows`], for the reason
/// [`FieldRowDecoder`] gives.
#[derive(Default)]
struct TimeSampleRowDecoder {
    time_samples: Vec<crate::domain::types::TimeSample>,
    // Time samples share `encode_value_fields` / `decode_field_value` with
    // fields, so they inherit the same coexistence hazard and get their own
    // aggregated line (b.zvd).
    shape_conflicts: ValueShapeConflictTally,
}

impl TimeSampleRowDecoder {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            time_samples: Vec::with_capacity(capacity),
            ..Self::default()
        }
    }

    /// Decode one time-sample row `(section_id, time, hash)`, skipping an
    /// empty hash. Fails on a malformed external `hash` column (see
    /// [`decode_field_value`]).
    fn push(
        &mut self,
        object_id: &str,
        sid_str: &str,
        time_f64: f64,
        data: &HashMap<String, String>,
    ) -> Result<(), ServiceError> {
        if data.is_empty() {
            return Ok(());
        }
        let section_id = sid_str.parse::<u64>().unwrap_or(0);
        self.shape_conflicts
            .observe(data, section_id, || format!("t={time_f64}"));
        let value = decode_field_value(
            data,
            object_id,
            format_args!("time sample ({section_id}, t={time_f64})"),
        )?;
        let version = parse_u64(data, "version").unwrap_or(0);
        self.time_samples.push(crate::domain::types::TimeSample {
            section_id,
            time: OrderedFloat(time_f64),
            value,
            version,
        });
        Ok(())
    }

    /// Emit the aggregated warn line and return the time samples, in push
    /// order.
    fn finish(self, object_id: &str) -> Vec<crate::domain::types::TimeSample> {
        self.shape_conflicts.emit("time_sample", object_id);
        self.time_samples
    }
}

// ── Atomic snapshot read (Spec SDD SR-1) ────────────────────────────────────
//
// `read_object_snapshot` reads one object's whole state in one `EVALSHA` of
// `scripts/object_snapshot_read.lua` on the snapshot lane. The script runs
// atomically, so the state it returns is exactly the state at the header `seq`
// it returns: the cut, proven by construction rather than by re-reading. Above
// the item or byte limit the script answers `TOO_LARGE` instead, without
// partial state, and the caller takes its consumer's above-ceiling path.

/// Items one section hash holds: `parent_id`, `name`, `type`, `order`,
/// `first_ver`, `last_ver`.
///
/// The writers: `storage::layer::queue_object_state`'s sections loop,
/// `storage::layer::execute_mutation`'s `SetSection` arm and
/// `update_layer_fast.lua`'s structural-mode `SEC` record (whose six columns
/// are `storage::layer::section_hash_fields`', carried by
/// `storage::lua::encode_mutations`), the only `HSET`s of an `obj:sec:*` key.
/// One of the four widths the atomic read's item bound is
/// computed from (Spec SDD SR-1.2 step 2): a writer that adds a column must
/// raise the width it writes, or the bound under-counts and an object can pass
/// the early-out and trip the item tally mid-read instead.
pub const SNAPSHOT_SECTION_HASH_WIDTH: u64 = 6;

/// Items one tombstone hash holds: `first_ver`, `last_ver`. The writers:
/// `storage::layer::execute_mutation`'s `CreateTombstone` arm and
/// `update_layer_fast.lua`'s structural-mode `TOMB` record, the only `HSET`s of
/// an `obj:del:*` key. See [`SNAPSHOT_SECTION_HASH_WIDTH`] for what the
/// width is for.
pub const SNAPSHOT_TOMBSTONE_HASH_WIDTH: u64 = 2;

/// Items one field hash holds at most: the value in one of its two shapes —
/// `value` (inline) or `hash` + `size` (external), never both, since every
/// writer clears the other shape first (`storage::layer::VALUE_HFIELDS`) — plus
/// `set_order`, `authoring_set_order` and `version`.
///
/// The writers: `storage::layer::field_value_hash_fields` (behind
/// `queue_object_state`'s fields loop and `execute_mutation`'s `SetField` arm)
/// and `update_layer_fast.lua`'s fast-mode `SF` arm, whose value columns are
/// `storage::lua::encode_delta`'s, and its structural-mode `SFA` arm, which
/// writes `field_value_hash_fields`' columns as `storage::lua::encode_mutations`
/// carries them. An inline field uses four of the five, which
/// is what makes the bound about 6 % conservative on Kitchen_set. See
/// [`SNAPSHOT_SECTION_HASH_WIDTH`] for what the width is for.
pub const SNAPSHOT_FIELD_HASH_WIDTH: u64 = 5;

/// Items one time-sample hash holds at most: the value in one of its two
/// shapes (see [`SNAPSHOT_FIELD_HASH_WIDTH`]) plus `version`.
///
/// The writers: `storage::layer::time_sample_hash_fields` (behind
/// `queue_object_state`'s time-samples loop and `execute_mutation`'s
/// `SetTimeSample` arm) and `update_layer_fast.lua`'s fast-mode `ST` and
/// structural-mode `STA` arms. See
/// [`SNAPSHOT_SECTION_HASH_WIDTH`] for what the width is for.
pub const SNAPSHOT_TIME_SAMPLE_HASH_WIDTH: u64 = 3;

/// The reply format tag `object_snapshot_read.lua` puts first in every reply.
/// A reply carrying another tag came from a script this binary does not carry
/// and is refused as `Internal` rather than decoded.
const OBJECT_SNAPSHOT_READ_FORMAT: &str = "object-snapshot-read-v1";

/// Error tag of a per-key refusal inside the script:
/// `SNAPSHOT_KEY_REFUSED: <family> <key> <raw error>`.
const SNAPSHOT_KEY_REFUSED_TAG: &str = "SNAPSHOT_KEY_REFUSED: ";

/// Error tag of the self-test's refused read:
/// `SNAPSHOT_SELFTEST_REFUSED: <key> <raw error>`.
const SNAPSHOT_SELFTEST_REFUSED_TAG: &str = "SNAPSHOT_SELFTEST_REFUSED: ";

/// The object id the startup self-test reads under (Spec SDD SR-1.8).
///
/// Never written: object ids are decimal `u64`s minted by the object-id
/// counter, so this non-numeric id can never name a real object. Its header
/// key is absent and the self-test reads one key beside it, so the self-test
/// touches no state.
pub const SNAPSHOT_SELF_TEST_OBJECT_ID: &str = "snapshot-read-self-test";

/// The `consumer` vocabulary of a reader's snapshot read (Spec SDD SR-8.1):
/// every consumer whose read returns the full state ([`ObjectSnapshot`]).
/// Each caller of [`read_object_snapshot`] names itself; the label rides on
/// every `live_edit_snapshot_read_*` series, and the consumer decides the read
/// path. With [`SlimSnapshotReadConsumer`] — the structural write-planning
/// consumers, whose reads return the compact form — it makes up the closed
/// label set [`metrics::SNAPSHOT_READ_CONSUMERS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotReadConsumer {
    /// A `SubscribeReadLayer` catch-up.
    Catchup,
    /// A legacy (rollback-mode) in-window replay's per-commit read. An empty
    /// commit takes none, and the steady-state path never reads.
    CatchupLegacyPerCommit,
    /// The checkpoint assembler.
    Checkpoint,
}

impl SnapshotReadConsumer {
    /// The metric label (a [`metrics::SNAPSHOT_READ_CONSUMERS`] value).
    pub fn label(self) -> &'static str {
        match self {
            Self::Catchup => metrics::SNAPSHOT_READ_CONSUMER_CATCHUP,
            Self::CatchupLegacyPerCommit => {
                metrics::SNAPSHOT_READ_CONSUMER_CATCHUP_LEGACY_PER_COMMIT
            }
            Self::Checkpoint => metrics::SNAPSHOT_READ_CONSUMER_CHECKPOINT,
        }
    }

    /// Whether this consumer reads through the atomic script (Spec SDD
    /// SR-1.9): every reader does. (External-value clean-up's reference walk,
    /// `storage::retirement`, is no snapshot read: a superseded object needs
    /// no cut, SR-4.4.) The structural consumers' rule is
    /// [`SlimSnapshotReadConsumer::takes_atomic_path`].
    pub fn takes_atomic_path(self) -> bool {
        matches!(
            self,
            Self::Catchup | Self::CatchupLegacyPerCommit | Self::Checkpoint
        )
    }
}

/// The structural write-planning consumers of a snapshot read, whose reads
/// are **slim**: their state is the compact form ([`CompactState`]: per value
/// row, its key and its external hash), read by [`read_object_snapshot_slim`]
/// and the fenced read. A separate type from [`SnapshotReadConsumer`], whose
/// reads return the full [`ObjectSnapshot`], so a compact state never reaches
/// a reader: the read's result type follows the consumer's type.
///
/// A slim read tallies what a full read of the same path would, and decodes
/// into the compact state:
///
/// - on the atomic path the script's `slim` flag is set, so no inline value
///   is transferred; its item, byte and entry tallies are a full read's
///   (`scripts/object_snapshot_read.lua`, `ARGV[13]`);
/// - the fenced read's pass reads each value row whole, as for every
///   consumer, and keeps only its key and validated external hash as it
///   decodes;
/// - [`Self::StructuralUpdate`]'s unfenced fan-out reads each value row with
///   two commands that transfer no inline value, its children lists by
///   length and then a ranged read, and a marked object's `obj:chs` in pages
///   ([`FanoutReplies::Bounded`]), each window of its rounds 2 and 3 in one
///   window script except a lone children list whose modelled shard time
///   alone exceeds τ, issued as one plain `LRANGE`, and every window after
///   the read falls back to plain commands. Its tallies
///   are rebuilt as a whole-row read's would be, and equal them for every
///   row a writer of this service shapes; a column no writer writes is not
///   counted.
///
/// Either way an external row's `hash` column is validated by the readers'
/// strict rule, with the same error. The fenced read's pass still reports a
/// row holding both value shapes (b.zvd), and the atomic slim read reports it
/// from the row's `b` shape; the bounded-reply read reports it on its script
/// windows, whose commands per row run atomically, and decodes such a row
/// inline without reporting it on its plain windows, whose two commands per
/// row can tear across a rewrite (`ValueRowReply::shape`). None of them
/// tallies the `set_order` / `version` column misses the full decode reports,
/// since the compact form holds neither column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlimSnapshotReadConsumer {
    /// The structural `UpdateLayer` slow path (the WATCH path), once per
    /// attempt. Never takes the atomic path: it reads through the unfenced
    /// fan-out at every size, with bounded replies
    /// ([`FanoutReplies::Bounded`]), charged as it reads
    /// (`read_object_snapshot_slim_charged`), so its unfenced `too_large` is
    /// a stop for the call's upload charge, not a ceiling.
    StructuralUpdate,
    /// The structural write-planning cache's cold-start read
    /// (`storage::structural_cache_cold_start`; structural-edits SDD SR-4.3):
    /// the plain atomic read at the shared ceilings, and above them — or when
    /// the cold start's route predictor puts the layer over the byte ceiling —
    /// b.y3m's fenced read, through the cold start's own hook
    /// (`storage::structural_cache_fenced`). Its [`SnapshotReadOf::TooLarge`]
    /// sends it to that fenced read and never to the unfenced fan-out, which
    /// under write traffic never completes. So on this consumer an atomic
    /// `too_large` is a route-predictor miss, a layer that grew since the
    /// probe, or an unprobed cold start's first read. With
    /// [`Self::StructuralCacheAudit`], the only consumers that set
    /// `return_dangling` and get [`AtomicSnapshotOf::structural`] on the
    /// atomic path; the fenced pass fills it too.
    StructuralCache,
    /// The structural write-planning cache's sampled audit read
    /// (`storage::structural_cache_audit`; structural-edits SDD SR-3.6): the
    /// same reads as [`Self::StructuralCache`] — the plain atomic read, and
    /// above its ceilings, with the above-ceiling switch on, the fenced read —
    /// under its own label, so `consumer="structural_cache"` counts cold
    /// starts only. With `return_dangling` set and
    /// [`AtomicSnapshotOf::structural`].
    StructuralCacheAudit,
}

impl SlimSnapshotReadConsumer {
    /// The metric label (a [`metrics::SNAPSHOT_READ_CONSUMERS`] value).
    pub fn label(self) -> &'static str {
        match self {
            Self::StructuralUpdate => metrics::SNAPSHOT_READ_CONSUMER_STRUCTURAL_UPDATE,
            Self::StructuralCache => metrics::SNAPSHOT_READ_CONSUMER_STRUCTURAL_CACHE,
            Self::StructuralCacheAudit => metrics::SNAPSHOT_READ_CONSUMER_STRUCTURAL_CACHE_AUDIT,
        }
    }

    /// Whether this consumer reads through the atomic script (Spec SDD
    /// SR-1.9). The structural `UpdateLayer` WATCH path never does: it reads
    /// once per attempt, and it is the most frequent writer in an
    /// agent-built scene, so a read that blocks the shard for the whole
    /// object per attempt would dominate the shard-delay allowance (SR-4.3);
    /// its per-window script blocks the shard one bounded window at a time
    /// (SDD §10.5). The structural cache's cold start
    /// always does: it reads once per object per replica, and needs the
    /// proven cut its entry is fenced on. So does its sampled audit, which
    /// compares the entry with the state at that cut.
    pub fn takes_atomic_path(self) -> bool {
        matches!(self, Self::StructuralCache | Self::StructuralCacheAudit)
    }

    /// Whether this consumer's atomic read sets the script's `return_dangling`
    /// flag and decodes [`AtomicSnapshotOf::structural`]: the structural
    /// cache's cold start and its audit only. Every other consumer's atomic
    /// read sends the script's original eleven arguments (a reader's) or
    /// passes `'0'` before the `slim` flag.
    fn reads_structural_facts(self) -> bool {
        matches!(self, Self::StructuralCache | Self::StructuralCacheAudit)
    }
}

/// A snapshot-read consumer vocabulary, and the state its reads decode into:
/// [`SnapshotReadConsumer`] (the full [`ObjectSnapshot`]) or
/// [`SlimSnapshotReadConsumer`] (the compact [`CompactState`]). Sealed; the
/// fenced read is generic over it, so its result type follows its consumer's.
pub trait SnapshotConsumer: sealed::ConsumerRules + Copy + std::fmt::Debug + Send + Sync {
    /// What this consumer's reads decode the state into.
    type State: SnapshotState;

    /// The metric label (a [`metrics::SNAPSHOT_READ_CONSUMERS`] value).
    fn label(self) -> &'static str;

    /// Whether this consumer reads through the atomic script (Spec SDD
    /// SR-1.9), and so has an above-ceiling fenced path.
    fn takes_atomic_path(self) -> bool;
}

impl SnapshotConsumer for SnapshotReadConsumer {
    type State = ObjectSnapshot;

    fn label(self) -> &'static str {
        SnapshotReadConsumer::label(self)
    }

    fn takes_atomic_path(self) -> bool {
        SnapshotReadConsumer::takes_atomic_path(self)
    }
}

impl SnapshotConsumer for SlimSnapshotReadConsumer {
    type State = CompactState;

    fn label(self) -> &'static str {
        SlimSnapshotReadConsumer::label(self)
    }

    fn takes_atomic_path(self) -> bool {
        SlimSnapshotReadConsumer::takes_atomic_path(self)
    }
}

/// What a snapshot read decodes an object's state into: [`ObjectSnapshot`],
/// every row whole, or [`CompactState`], the compact form. Sealed: the two
/// differ only in how a field or time-sample row is decoded (module
/// `sealed`), never in what is read or tallied.
pub trait SnapshotState: sealed::SnapshotDecode + Send + Sync + Sized + 'static {}

impl SnapshotState for ObjectSnapshot {}

impl SnapshotState for CompactState {}

/// The sealed halves of [`SnapshotConsumer`] and [`SnapshotState`]: what the
/// read paths need of them and nothing outside this module may implement.
mod sealed {
    use super::*;

    /// The parts of a consumer's read rules the read paths alone use.
    pub trait ConsumerRules {
        /// Whether the atomic read sets `return_dangling` and decodes
        /// [`AtomicSnapshotOf::structural`].
        fn reads_structural_facts(self) -> bool;
    }

    impl ConsumerRules for SnapshotReadConsumer {
        fn reads_structural_facts(self) -> bool {
            false
        }
    }

    impl ConsumerRules for SlimSnapshotReadConsumer {
        fn reads_structural_facts(self) -> bool {
            SlimSnapshotReadConsumer::reads_structural_facts(self)
        }
    }

    /// Everything of a read's state but its value rows.
    pub struct SnapshotParts {
        pub object_id: u64,
        pub sequence_num: u64,
        pub first_version: u64,
        pub source_format: String,
        pub next_set_order: u64,
        pub sections: HashMap<u64, Section>,
        pub children: HashMap<u64, Vec<u64>>,
        pub tombstones: Vec<(u64, u64, u64)>,
    }

    /// How a state is decoded from what a read read. Both readers share every
    /// decoder but the value rows'.
    pub trait SnapshotDecode: Sized {
        /// Whether the atomic read sets the script's `slim` flag.
        const SLIM: bool;

        /// The value rows decoded so far.
        type Rows: Send;

        /// An empty row decoder, sized for the index members the read found.
        fn rows_with_capacity(fields: usize, time_samples: usize) -> Self::Rows;

        /// Decode one field row `(section id, key name)` the fan-out read
        /// whole, skipping an empty hash.
        fn push_field(
            rows: &mut Self::Rows,
            object_id: &str,
            sid_str: &str,
            key_name: String,
            data: &HashMap<String, String>,
        ) -> Result<(), ServiceError>;

        /// Decode one time-sample row `(section id, time)` the fan-out read
        /// whole, skipping an empty hash.
        fn push_time_sample(
            rows: &mut Self::Rows,
            object_id: &str,
            sid_str: &str,
            time: f64,
            data: &HashMap<String, String>,
        ) -> Result<(), ServiceError>;

        /// Decode one field row `(section id, key name)` from its shape, as
        /// the bounded-reply fan-out reads it ([`FanoutReplies::Bounded`])
        /// and a slim atomic reply carries it, skipping an absent row. The
        /// compact form only: that read is the structural update's, whose
        /// state is compact; the full form refuses it `Internal`.
        fn push_field_shape(
            rows: &mut Self::Rows,
            object_id: &str,
            sid_str: &str,
            key_name: String,
            shape: &SlimRow,
        ) -> Result<(), ServiceError>;

        /// [`Self::push_field_shape`] for a time-sample row `(section id,
        /// time)`.
        fn push_time_sample_shape(
            rows: &mut Self::Rows,
            object_id: &str,
            sid_str: &str,
            time: f64,
            shape: &SlimRow,
        ) -> Result<(), ServiceError>;

        /// The fan-out's state, emitting the decode's aggregated warn lines.
        fn assemble(parts: SnapshotParts, rows: Self::Rows, object_id: &str) -> Self;

        /// An atomic reply's value-row element (the fields' or the time
        /// samples'), its per-row bodies not yet decoded.
        type AtomicRows;

        /// The atomic reply's value rows, decoded.
        type AtomicDecoded;

        /// Split one value-row element of an atomic reply (full or slim, as
        /// [`Self::SLIM`] set the flag) into its rows. Strict on shape.
        fn atomic_rows(
            value: Option<fred::types::Value>,
            object_id: &str,
            what: &str,
        ) -> Result<Self::AtomicRows, ServiceError>;

        /// Decode an atomic reply's field and time-sample rows, returning
        /// them and the bytes of their bodies
        /// ([`AtomicSnapshotOf::value_row_bytes_received`]; tallied in
        /// test-support builds only, 0 otherwise).
        fn decode_atomic(
            object_id: &str,
            fields: Self::AtomicRows,
            time_samples: Self::AtomicRows,
        ) -> Result<(Self::AtomicDecoded, u64), ServiceError>;

        /// The atomic read's state.
        fn assemble_atomic(
            parts: SnapshotParts,
            rows: Self::AtomicDecoded,
            object_id: &str,
        ) -> Self;
    }

    /// [`ObjectSnapshot`]'s value rows: every row whole.
    pub struct FullValueRows {
        fields: FieldRowDecoder,
        time_samples: TimeSampleRowDecoder,
    }

    impl SnapshotDecode for ObjectSnapshot {
        const SLIM: bool = false;
        type Rows = FullValueRows;

        fn rows_with_capacity(fields: usize, time_samples: usize) -> FullValueRows {
            FullValueRows {
                fields: FieldRowDecoder::with_capacity(fields),
                time_samples: TimeSampleRowDecoder::with_capacity(time_samples),
            }
        }

        fn push_field(
            rows: &mut FullValueRows,
            object_id: &str,
            sid_str: &str,
            key_name: String,
            data: &HashMap<String, String>,
        ) -> Result<(), ServiceError> {
            rows.fields.push(object_id, sid_str, key_name, data)
        }

        fn push_time_sample(
            rows: &mut FullValueRows,
            object_id: &str,
            sid_str: &str,
            time: f64,
            data: &HashMap<String, String>,
        ) -> Result<(), ServiceError> {
            rows.time_samples.push(object_id, sid_str, time, data)
        }

        fn push_field_shape(
            _rows: &mut FullValueRows,
            object_id: &str,
            _sid_str: &str,
            _key_name: String,
            _shape: &SlimRow,
        ) -> Result<(), ServiceError> {
            Err(shape_read_into_full_state(object_id))
        }

        fn push_time_sample_shape(
            _rows: &mut FullValueRows,
            object_id: &str,
            _sid_str: &str,
            _time: f64,
            _shape: &SlimRow,
        ) -> Result<(), ServiceError> {
            Err(shape_read_into_full_state(object_id))
        }

        fn assemble(parts: SnapshotParts, rows: FullValueRows, object_id: &str) -> Self {
            let fields = rows.fields.finish(object_id);
            let time_samples = rows.time_samples.finish(object_id);
            Self::assemble_atomic(parts, (fields, time_samples), object_id)
        }

        type AtomicRows = Vec<(String, fred::types::Value)>;
        type AtomicDecoded = (Vec<Field>, Vec<crate::domain::types::TimeSample>);

        fn atomic_rows(
            value: Option<fred::types::Value>,
            object_id: &str,
            what: &str,
        ) -> Result<Self::AtomicRows, ServiceError> {
            reply_entries(value, object_id, what)
        }

        fn decode_atomic(
            object_id: &str,
            fields: Self::AtomicRows,
            time_samples: Self::AtomicRows,
        ) -> Result<(Self::AtomicDecoded, u64), ServiceError> {
            let mut received = 0;
            let mut field_rows = Vec::with_capacity(fields.len());
            for (member, value) in fields {
                // Member format: "{section_id}:{hex(key_name)}"
                let (sid, hex_key) = split_member(&member);
                let data = reply_flat_hash(Some(value), object_id, "field hash")?;
                note_value_row_bytes(&mut received, &data);
                field_rows.push((sid, keys::hex_decode(&hex_key), data));
            }
            let mut time_sample_rows = Vec::with_capacity(time_samples.len());
            for (member, value) in time_samples {
                // Member format: "{section_id}:{hex_f64(time)}"
                let (sid, time_hex) = split_member(&member);
                let data = reply_flat_hash(Some(value), object_id, "time-sample hash")?;
                note_value_row_bytes(&mut received, &data);
                time_sample_rows.push((sid, keys::hex_decode_f64(&time_hex), data));
            }
            let fields = decode_field_rows(object_id, field_rows)?;
            let time_samples = decode_time_sample_rows(object_id, time_sample_rows)?;
            Ok(((fields, time_samples), received))
        }

        fn assemble_atomic(
            parts: SnapshotParts,
            (fields, time_samples): Self::AtomicDecoded,
            _object_id: &str,
        ) -> Self {
            ObjectSnapshot {
                object_id: parts.object_id,
                sequence_num: parts.sequence_num,
                first_version: parts.first_version,
                source_format: parts.source_format,
                next_set_order: parts.next_set_order,
                sections: parts.sections,
                children: parts.children,
                fields,
                time_samples,
                tombstones: parts.tombstones,
            }
        }
    }

    /// [`CompactState`]'s value rows: per row its key and its validated
    /// external hash, plus the b.zvd shape-conflict tallies the full decode
    /// keeps.
    pub struct CompactValueRows {
        fields: CompactFields,
        time_samples: CompactSamples,
        field_shape_conflicts: ValueShapeConflictTally,
        time_sample_shape_conflicts: ValueShapeConflictTally,
    }

    impl SnapshotDecode for CompactState {
        const SLIM: bool = true;
        type Rows = CompactValueRows;

        fn rows_with_capacity(fields: usize, time_samples: usize) -> CompactValueRows {
            CompactValueRows {
                fields: CompactFields::with_capacity(fields),
                time_samples: CompactSamples::with_capacity(time_samples),
                field_shape_conflicts: ValueShapeConflictTally::default(),
                time_sample_shape_conflicts: ValueShapeConflictTally::default(),
            }
        }

        fn push_field(
            rows: &mut CompactValueRows,
            object_id: &str,
            sid_str: &str,
            key_name: String,
            data: &HashMap<String, String>,
        ) -> Result<(), ServiceError> {
            if data.is_empty() {
                return Ok(());
            }
            let section_id = sid_str.parse::<u64>().unwrap_or(0);
            rows.field_shape_conflicts
                .observe(data, section_id, || key_name.clone());
            let hash = decode_external_hash(
                data,
                object_id,
                format_args!("field ({section_id}, {key_name:?})"),
            )?;
            rows.fields.insert(section_id, &key_name, hash);
            Ok(())
        }

        fn push_time_sample(
            rows: &mut CompactValueRows,
            object_id: &str,
            sid_str: &str,
            time: f64,
            data: &HashMap<String, String>,
        ) -> Result<(), ServiceError> {
            if data.is_empty() {
                return Ok(());
            }
            let section_id = sid_str.parse::<u64>().unwrap_or(0);
            rows.time_sample_shape_conflicts
                .observe(data, section_id, || format!("t={time}"));
            let hash = decode_external_hash(
                data,
                object_id,
                format_args!("time sample ({section_id}, t={time})"),
            )?;
            rows.time_samples.insert(section_id, time.to_bits(), hash);
            Ok(())
        }

        fn push_field_shape(
            rows: &mut CompactValueRows,
            object_id: &str,
            sid_str: &str,
            key_name: String,
            shape: &SlimRow,
        ) -> Result<(), ServiceError> {
            let section_id = sid_str.parse::<u64>().unwrap_or(0);
            let hash = shape.external_hash(
                &mut rows.field_shape_conflicts,
                section_id,
                || key_name.clone(),
                object_id,
                format_args!("field ({section_id}, {key_name:?})"),
            )?;
            if let Some(hash) = hash {
                rows.fields.insert(section_id, &key_name, hash);
            }
            Ok(())
        }

        fn push_time_sample_shape(
            rows: &mut CompactValueRows,
            object_id: &str,
            sid_str: &str,
            time: f64,
            shape: &SlimRow,
        ) -> Result<(), ServiceError> {
            let section_id = sid_str.parse::<u64>().unwrap_or(0);
            let hash = shape.external_hash(
                &mut rows.time_sample_shape_conflicts,
                section_id,
                || format!("t={time}"),
                object_id,
                format_args!("time sample ({section_id}, t={time})"),
            )?;
            if let Some(hash) = hash {
                rows.time_samples.insert(section_id, time.to_bits(), hash);
            }
            Ok(())
        }

        fn assemble(parts: SnapshotParts, rows: CompactValueRows, object_id: &str) -> Self {
            rows.field_shape_conflicts.emit("field", object_id);
            rows.time_sample_shape_conflicts
                .emit("time_sample", object_id);
            // A compact state holds no tombstones: no structural plan reads
            // them.
            CompactState {
                object_id: parts.object_id,
                sequence_num: parts.sequence_num,
                first_version: parts.first_version,
                source_format: parts.source_format,
                next_set_order: parts.next_set_order,
                sections: parts.sections,
                children: parts.children,
                fields: rows.fields,
                time_samples: rows.time_samples,
            }
        }

        type AtomicRows = Vec<(String, SlimRow)>;
        type AtomicDecoded = CompactValueRows;

        fn atomic_rows(
            value: Option<fred::types::Value>,
            object_id: &str,
            what: &str,
        ) -> Result<Self::AtomicRows, ServiceError> {
            reply_slim_rows(value, object_id, what)
        }

        fn decode_atomic(
            object_id: &str,
            fields: Self::AtomicRows,
            time_samples: Self::AtomicRows,
        ) -> Result<(Self::AtomicDecoded, u64), ServiceError> {
            let mut rows = Self::rows_with_capacity(fields.len(), time_samples.len());
            let mut received = 0;
            for (member, shape) in fields {
                note_slim_row_bytes(&mut received, &shape);
                // Member format: "{section_id}:{hex(key_name)}"
                let (sid, hex_key) = split_member(&member);
                Self::push_field_shape(
                    &mut rows,
                    object_id,
                    &sid,
                    keys::hex_decode(&hex_key),
                    &shape,
                )?;
            }
            for (member, shape) in time_samples {
                note_slim_row_bytes(&mut received, &shape);
                // Member format: "{section_id}:{hex_f64(time)}"
                let (sid, time_hex) = split_member(&member);
                Self::push_time_sample_shape(
                    &mut rows,
                    object_id,
                    &sid,
                    keys::hex_decode_f64(&time_hex),
                    &shape,
                )?;
            }
            Ok((rows, received))
        }

        fn assemble_atomic(
            parts: SnapshotParts,
            rows: Self::AtomicDecoded,
            object_id: &str,
        ) -> Self {
            Self::assemble(parts, rows, object_id)
        }
    }
}

/// The string bytes of one decoded value row's columns, names included.
/// Test-support only: [`AtomicSnapshotOf::value_row_bytes_received`] is its
/// one reader.
#[cfg(any(test, feature = "test-support"))]
fn flat_hash_bytes(data: &HashMap<String, String>) -> u64 {
    data.iter().map(|(k, v)| (k.len() + v.len()) as u64).sum()
}

/// Add one decoded value row's bytes to `received`, the full atomic decode's
/// [`AtomicSnapshotOf::value_row_bytes_received`] tally. A no-op outside
/// test-support builds, so a reader's atomic read does no work for a value
/// only tests read (the fan-out gates its own tally the same way).
#[cfg(any(test, feature = "test-support"))]
fn note_value_row_bytes(received: &mut u64, data: &HashMap<String, String>) {
    *received += flat_hash_bytes(data);
}

/// See the test-support form.
#[cfg(not(any(test, feature = "test-support")))]
fn note_value_row_bytes(_received: &mut u64, _data: &HashMap<String, String>) {}

/// The slim atomic decode's form of [`note_value_row_bytes`]: add one slim
/// row element's shape and hash bytes to `received`. A no-op outside
/// test-support builds, so a slim read (cold start, audit) does no work for a
/// value only tests read.
#[cfg(any(test, feature = "test-support"))]
fn note_slim_row_bytes(received: &mut u64, shape: &SlimRow) {
    *received += shape.bytes();
}

/// See the test-support form.
#[cfg(not(any(test, feature = "test-support")))]
fn note_slim_row_bytes(_received: &mut u64, _shape: &SlimRow) {}

/// [`SlimRow`], in its own module: it names the compact form's atomic rows
/// (`sealed::SnapshotDecode::AtomicRows`), so it must be nominally `pub`.
mod slim_row {
    use super::*;

    /// One value row's element in a slim atomic reply (the script's `slim`
    /// flag): the row's shape. The bounded-reply fan-out decides the same
    /// shape from its two-command read of the row
    /// ([`ValueRowReply::shape`](super::ValueRowReply::shape)), except that
    /// on its plain windows, whose commands can tear, it never answers
    /// [`Self::InlineBesideHash`].
    pub enum SlimRow {
        /// `'-'`: the row's hash is absent (empty); skipped, as the full decode
        /// skips it.
        Absent,
        /// `'i'`: inline (a `value` column, or neither value column).
        Inline,
        /// `'b'`: inline, with a `hash` column beside the `value` (b.zvd).
        InlineBesideHash,
        /// `'x'`: external, with the raw `hash` column.
        External(String),
    }

    impl SlimRow {
        /// The bytes of the element's shape and hash strings. Test-support
        /// only: `note_slim_row_bytes` is its one caller.
        #[cfg(any(test, feature = "test-support"))]
        pub(super) fn bytes(&self) -> u64 {
            match self {
                Self::External(hash) => 1 + hash.len() as u64,
                Self::Absent | Self::Inline | Self::InlineBesideHash => 1,
            }
        }

        /// The row's external hash, decoded by the full decode's rule: `None`
        /// for an absent row (not held), `Some(None)` for an inline one,
        /// `Some(Some(hash))` for an external one. A `b` row is noted in
        /// `conflicts` as the full decode notes it.
        ///
        /// # Errors
        ///
        /// [`ServiceError::Internal`] for a malformed `hash` column, with
        /// [`decode_field_value`]'s message.
        pub(super) fn external_hash(
            &self,
            conflicts: &mut ValueShapeConflictTally,
            section_id: u64,
            name: impl FnOnce() -> String,
            object_id: &str,
            row: impl std::fmt::Display,
        ) -> Result<Option<Option<[u8; 32]>>, ServiceError> {
            match self {
                Self::Absent => Ok(None),
                Self::Inline => Ok(Some(None)),
                Self::InlineBesideHash => {
                    conflicts.note(section_id, name);
                    Ok(Some(None))
                }
                Self::External(hash_hex) => {
                    parse_external_hash(hash_hex, object_id, row).map(|hash| Some(Some(hash)))
                }
            }
        }
    }
}

use slim_row::SlimRow;

/// The `Internal` error for a value row read by its shape (the bounded-reply
/// fan-out) into the full state. Unreachable while only the structural
/// update's read, whose state is compact, reads rows that way: the unfenced
/// fan-out is its read alone ([`read_object_snapshot_unfenced`]).
fn shape_read_into_full_state(object_id: &str) -> ServiceError {
    ServiceError::Internal(format!(
        "snapshot read of object {object_id}: a value row read by its shape cannot be decoded \
         into the full state"
    ))
}

/// A slim atomic reply's value-row element: `{member, shape}` or `{member,
/// 'x', hash}` per row (`scripts/object_snapshot_read.lua`, `ARGV[13]`).
/// Strict: any other element is a malformed reply.
fn reply_slim_rows(
    value: Option<fred::types::Value>,
    object_id: &str,
    what: &str,
) -> Result<Vec<(String, SlimRow)>, ServiceError> {
    value
        .map(|v| v.into_array())
        .unwrap_or_default()
        .into_iter()
        .map(|entry| {
            let mut parts = entry.into_array().into_iter().map(|v| v.into_string());
            let row = match (parts.next(), parts.next(), parts.next(), parts.next()) {
                (Some(Some(member)), Some(Some(shape)), None, None) => match shape.as_str() {
                    "-" => Some((member, SlimRow::Absent)),
                    "i" => Some((member, SlimRow::Inline)),
                    "b" => Some((member, SlimRow::InlineBesideHash)),
                    _ => None,
                },
                (Some(Some(member)), Some(Some(shape)), Some(Some(hash)), None) if shape == "x" => {
                    Some((member, SlimRow::External(hash)))
                }
                _ => None,
            };
            row.ok_or_else(|| {
                reply_shape_err(
                    object_id,
                    format!("a slim {what} row is not {{member, shape}} or {{member, 'x', hash}}"),
                )
            })
        })
        .collect()
}

/// How a snapshot was read (Spec SDD SR-1.5, SR-8.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotReadPath {
    /// One `EVALSHA` of `object_snapshot_read.lua`: the state at its cut.
    Atomic,
    /// The fan-out read under a per-object snapshot fence.
    Fenced,
    /// The bounded fan-out read with no fence.
    Unfenced,
}

impl SnapshotReadPath {
    /// The metric label (a [`metrics::SNAPSHOT_READ_PATHS`] value).
    pub fn label(self) -> &'static str {
        match self {
            Self::Atomic => metrics::SNAPSHOT_READ_PATH_ATOMIC,
            Self::Fenced => metrics::SNAPSHOT_READ_PATH_FENCED,
            Self::Unfenced => metrics::SNAPSHOT_READ_PATH_UNFENCED,
        }
    }
}

/// One object's state as a snapshot read read it (Spec SDD SR-1.5): `S` is
/// what the read decoded the state into ([`SnapshotState`]) — an
/// [`ObjectSnapshot`] for every reader ([`AtomicSnapshot`],
/// [`read_object_snapshot`]), a [`CompactState`] for the structural
/// write-planning consumers' slim reads ([`read_object_snapshot_slim`]).
pub struct AtomicSnapshotOf<S> {
    /// The object's state. On the atomic and fenced paths it is exactly the
    /// state at [`Self::cut`]; on the unfenced path it is unproven (see
    /// [`Self::cut`]). There is no raced section to reconcile under a proven
    /// read (an indexed section without a hash there is a corrupt index,
    /// reported on [`Self::corrupt_section_index_members`]), and the unfenced
    /// path's consumers do not reconcile.
    pub state: S,
    /// The header `seq` at the cut — on the atomic path every commit at or
    /// below it is in [`Self::state`] and none above it is. On the fenced
    /// path it is the `seq` the fence's acquire script read, and the same
    /// holds: the read ran under a lease no commit could land through, and
    /// its release proved the lease still held and the header still at it
    /// (Spec SDD SR-5.3). On the unfenced path it is the `seq` the header held
    /// when the read's first round read it, and nothing proves the rest of the
    /// read saw the same commit: the caller proves it if it needs to (the
    /// structural path's re-read of the header, Spec SDD SR-4.3). Equal to
    /// `state.sequence_num`.
    pub cut: u64,
    /// The header's `last_server_ts`, read with [`Self::cut`].
    pub last_server_ts: u64,
    /// The header's `last_client_ts`, read with [`Self::cut`].
    pub last_client_ts: u64,
    /// Bytes of every string the read read for the object's indices, entries
    /// and children: on the atomic path, the tally the byte limit is checked
    /// against, and counted the same way on the fan-out paths. It counts the
    /// `obj:chs` members a marked object's read reads but never returns in
    /// [`Self::state`], and not the header or the reply framing.
    pub reply_bytes: u64,
    /// Items the read read (hash fields, set and sorted-set members, list
    /// elements, the `obj:chs` members of a marked object included): on the
    /// atomic path, the tally the item limit is checked against, and counted
    /// the same way on the fan-out paths.
    pub item_count: u64,
    /// Entries the read issued a command for: one per section, tombstone,
    /// field and time-sample hash, and one per children list.
    pub entry_count: u64,
    /// How the state was read.
    pub path: SnapshotReadPath,
    /// The `obj:secs` members that name no readable section, in index order:
    /// a member whose section hash came back empty (left out of
    /// `state.sections`, as on every path), or a member that is not a decimal
    /// section id (decoded as section `0`, as on every path).
    ///
    /// Filled on the proven paths only — atomic and fenced — where either is a
    /// corrupt index rather than a race: the read saw the index and every hash
    /// at one cut (one script, or a fan-out no commit could land through), so
    /// a delete that removed the hash removed the member too (Spec SDD
    /// SR-4.1). The decoders skip such members as before, and each consumer
    /// decides what a corrupt index means to it: the checkpoint assembler
    /// refuses the layer as `Permanent` rather than persist it, while catch-up
    /// decodes the state as the fan-out always has. Empty on the unfenced
    /// path, where a member whose hash is gone can be a commit landing between
    /// rounds, and whose consumer (the structural path) does not consult it.
    pub corrupt_section_index_members: Vec<String>,
    /// What the structural paths need beyond the state, read at the same cut:
    /// on the atomic path, `Some` for the
    /// [`SlimSnapshotReadConsumer::StructuralCache`] and
    /// [`SlimSnapshotReadConsumer::StructuralCacheAudit`] consumers only,
    /// `None` for every other consumer; `Some` on both fan-out paths, which read the
    /// header and `obj:chs` anyway — the unfenced one for the WATCH path's
    /// structural commit.
    pub structural: Option<StructuralSnapshotFacts>,
    /// The catch-up memory this snapshot is accounted under (Spec SDD SR-7.2
    /// step 1), repaid when the snapshot is dropped. A shared fenced-read
    /// flight (`storage::snapshot_flights`) sets it to the flight's combined
    /// grant and its top-up, so the memory stays reserved exactly while some
    /// attached catch-up still holds the shared snapshot. Every read here
    /// returns it empty: [`read_object_snapshot`] and the fenced read reserve
    /// nothing themselves.
    pub memory: SnapshotMemory,
    /// Bytes of the field and time-sample rows' reply elements this read
    /// received, their member ids excluded: every column of every value row
    /// on a full read and on the whole-row fan-out (the fenced read's pass);
    /// each row's shape and external hash on an atomic slim read;
    /// and on the structural update's bounded-reply fan-out the value-shape
    /// columns it received (`hash` and `size`), so 0 for an inline row, whose
    /// value is never transferred. **Test-support only**: it tells a test a
    /// real slim read from a full read decoded into the compact form, which
    /// [`Self::reply_bytes`] (the store-side tally, the same for both)
    /// cannot.
    #[cfg(any(test, feature = "test-support"))]
    pub value_row_bytes_received: u64,
}

/// What every reader's snapshot read returns: the state with every row whole.
pub type AtomicSnapshot = AtomicSnapshotOf<ObjectSnapshot>;

impl<S> AtomicSnapshotOf<S> {
    /// The layer's size as this read measured it, `L = (item_count,
    /// reply_bytes)` (Spec SDD SR-11.1): what a fenced read's release records
    /// on the fence key for the layer's next epoch, and the weight its memory
    /// reservation is adjusted to.
    pub fn measured_size(&self) -> crate::storage::snapshot_fence::LayerSize {
        crate::storage::snapshot_fence::LayerSize {
            items: self.item_count,
            bytes: self.reply_bytes,
        }
    }
}

/// The header facts and dangling parents a snapshot read learned beyond the
/// state it returns ([`AtomicSnapshot::structural`]). Reported, not mapped:
/// the caller decides what a superseded or unmarked header means.
///
/// Two readers use them. The structural cache's cold start and audit seed
/// an entry from them (structural-edits SDD SR-3.2, SR-4.3, SR-3.6). The
/// structural commit paths read [`Self::dangling_parents`] to delete, when a
/// commit creates a section, a children list an earlier build stored under
/// that id while the id had no row (SDD §7.7, §10.5): no snapshot read
/// returns such a list, so the commit would otherwise leave it in place
/// under a now-live parent.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StructuralSnapshotFacts {
    /// The header's `commit_chain` column; `""` when absent (an object
    /// materialized before the column existed). The fence identity the cache
    /// entry is loaded at (SR-4.5).
    pub commit_chain: String,
    /// The `superseded` column, verbatim; `None` when absent. Any value marks
    /// the object superseded.
    pub superseded: Option<String>,
    /// The `chs_complete` column, verbatim; `None` when absent.
    pub chs_complete: Option<String>,
    /// The parents of the stored children lists the read left out because
    /// the parent is neither the root nor a live section: on the atomic path
    /// from the script's `return_dangling` element, in the script's order (the
    /// cache entry's `dangling_parents`, seeded at the read's cut); on the
    /// fan-out paths in `obj:chs`'s reply order. A member that is not a
    /// decimal id is left out, since no section can ever be created with it.
    /// Always empty on an unmarked object, whose `obj:chs` cannot enumerate
    /// them — check [`Self::chs_complete_marked`] before trusting an empty
    /// list.
    pub dangling_parents: Vec<u64>,
}

impl StructuralSnapshotFacts {
    /// Whether the header marks the object's children-list index complete
    /// (`chs_complete` = `CHS_COMPLETE_MARKED`), so that
    /// [`Self::dangling_parents`] is the complete set.
    pub fn chs_complete_marked(&self) -> bool {
        self.chs_complete.as_deref() == Some(crate::storage::layer::CHS_COMPLETE_MARKED)
    }

    /// Whether the header carries the `superseded` column.
    pub fn is_superseded(&self) -> bool {
        self.superseded.is_some()
    }
}

/// The item and byte ceilings one atomic read runs under (the script's
/// `ARGV[4..5]`).
///
/// Every production atomic read runs at the shared defaults on the client
/// (`LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS` / `…_MAX_BYTES`,
/// [`Self::of_client`]), through [`read_object_snapshot`] or, for the
/// structural cache's cold start and sampled audit, through
/// [`read_object_snapshot_with_ceilings`] with the client's own ceilings;
/// above them both take the fenced read. Only the measurement tools pass
/// other ceilings (the raised-ceiling read); the client's defaults never
/// change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotCeilings {
    /// The item limit.
    pub max_items: u64,
    /// The byte limit.
    pub max_bytes: u64,
}

impl SnapshotCeilings {
    /// The client's configured ceilings, the ones every consumer's plain read
    /// runs under.
    pub fn of_client(client: &ValkeyClient) -> Self {
        Self {
            max_items: client.snapshot_atomic_read_max_items,
            max_bytes: client.snapshot_atomic_read_max_bytes,
        }
    }
}

/// The cardinalities step 2 of the atomic read takes before deciding whether
/// to read on (Spec SDD SR-1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotCardinalities {
    /// `SCARD obj:secs`.
    pub sections: u64,
    /// `SCARD obj:dels`.
    pub tombstones: u64,
    /// `ZCARD obj:flds`.
    pub fields: u64,
    /// `ZCARD obj:tss`.
    pub time_samples: u64,
    /// `SCARD obj:chs`, read only when the object is marked
    /// (`chs_complete = "1"`); `None` on an unmarked object.
    pub children_index: Option<u64>,
}

/// `object_snapshot_read.lua`'s step-2 item bound over `cards`, computed the
/// way the script computes it: per live section its index member, its hash
/// ([`SNAPSHOT_SECTION_HASH_WIDTH`]) and one children-list element; per
/// tombstone, field and time sample its index member and its hash (the other
/// three widths); plus the `obj:chs` members on a marked object. Saturating.
///
/// The structural cache's cold start computes it from the probe's
/// cardinalities (`storage::structural_cache_sync`) to choose its route
/// before any full read (`storage::structural_cache_cold_start`): the script
/// answers `TOO_LARGE` at its early-out exactly when this exceeds the read's
/// item ceiling, on the same cardinalities.
pub fn snapshot_item_bound(cards: &SnapshotCardinalities) -> u64 {
    let per = |n: u64, width: u64| n.saturating_mul(width.saturating_add(1));
    per(cards.sections, SNAPSHOT_SECTION_HASH_WIDTH + 1)
        .saturating_add(per(cards.tombstones, SNAPSHOT_TOMBSTONE_HASH_WIDTH))
        .saturating_add(per(cards.fields, SNAPSHOT_FIELD_HASH_WIDTH))
        .saturating_add(per(cards.time_samples, SNAPSHOT_TIME_SAMPLE_HASH_WIDTH))
        .saturating_add(cards.children_index.unwrap_or(0))
}

/// The limit a mid-read `TOO_LARGE` crossed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotLimit {
    /// The item limit (`LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS`).
    Items,
    /// The byte limit (`LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_BYTES`).
    Bytes,
}

/// Where the atomic read stopped with `TOO_LARGE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotTooLargeAt {
    /// Step 2: the item bound exceeded the item limit, and no entry was read.
    EarlyOut,
    /// Steps 3–4: a running tally crossed `limit` after the command that
    /// pushed it over. The tallies cover what was read up to and including
    /// that command — partial, so they under-size the layer on their own.
    MidRead {
        limit: SnapshotLimit,
        item_tally: u64,
        byte_tally: u64,
    },
}

/// An atomic read's `TOO_LARGE` answer: the layer is above the atomic-read
/// ceiling (Spec SDD SR-1.9). The caller takes its consumer's above-ceiling
/// path; no partial state comes with it.
///
/// Carries what sizing the layer needs (Spec SDD SR-11.1): the cardinalities
/// and item bound on every answer, and the tallies and the crossed limit on a
/// mid-read one. `Copy` (so `Clone`), so one answer can be handed to every
/// waiter that shares the read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotTooLarge {
    /// Step 2's cardinalities.
    pub cardinalities: SnapshotCardinalities,
    /// Step 2's upper bound on the items the full read would tally.
    pub item_bound: u64,
    /// Where the read stopped.
    pub at: SnapshotTooLargeAt,
}

/// What a snapshot read read, its state decoded into `S` (see
/// [`AtomicSnapshotOf`]).
pub enum SnapshotReadOf<S> {
    /// The object's state. Boxed so the `TooLarge` answer does not carry the
    /// snapshot's size.
    Snapshot(Box<AtomicSnapshotOf<S>>),
    /// The atomic read answered `TOO_LARGE`.
    TooLarge(SnapshotTooLarge),
}

/// What [`read_object_snapshot`] read.
pub type SnapshotRead = SnapshotReadOf<ObjectSnapshot>;

/// Read one object's state for `consumer`, a reader (Spec SDD SR-1.5). The
/// structural write-planning consumers read through
/// [`read_object_snapshot_slim`] instead, the same reads decoded into the
/// compact form; what follows describes both, by path.
///
/// On the atomic path — the catch-up, legacy per-commit, checkpoint and
/// structural-cache consumers — this is one `EVALSHA` of `object_snapshot_read.lua` on the
/// snapshot lane (`ValkeyClient::acquire_snapshot`), under the client's item and
/// byte limits (`LIVE_EDIT_SNAPSHOT_ATOMIC_READ_MAX_ITEMS` / `…_MAX_BYTES`).
/// It returns:
///
/// - [`SnapshotRead::Snapshot`] with the state at its cut;
/// - [`SnapshotRead::TooLarge`] when the layer is above either limit, with no
///   partial state — the caller decides its above-ceiling path (the proven
///   one is `storage::snapshot_fence::read::read_object_snapshot_fenced`,
///   which records its own reads under `path="fenced"`);
/// - `ServiceError::NotFound` when the header is absent, or present without a
///   numeric `seq` (`read_object_header`'s rule);
/// - a classified error otherwise. A per-key refusal inside the script ticks
///   the refusal counters under its key's family and is permanent
///   (`Internal`) — except a refusal whose raw error is the cluster refusing
///   the key for its slot's state (a migrating slot, a slot not served, the
///   cluster down; [`is_script_cluster_state_refusal`]), which is attributed
///   to [`SnapshotKeyFamily::Header`], is transient (`Aborted`) and ticks no
///   refusal counter. A failure of the `EVALSHA` itself is attributed to
///   [`SnapshotKeyFamily::Header`] and classified by `map_snapshot_load_err`
///   (transient → `Aborted`, permanent → `Internal`); a reply in another
///   format is `Internal`.
///
/// The structural `UpdateLayer` WATCH path's consumer
/// ([`SlimSnapshotReadConsumer::StructuralUpdate`]) never takes the atomic
/// path ([`SlimSnapshotReadConsumer::takes_atomic_path`]). It
/// reads through the unfenced bounded fan-out at every size (Spec SDD SR-4.3),
/// on the
/// same snapshot lane, so for it this never answers `TooLarge`. It returns:
///
/// - [`SnapshotRead::Snapshot`] with `path = Unfenced` and `cut` = the header
///   `seq` the read's first round saw. Unproven: the fan-out holds no fence
///   and runs in several rounds, so a commit can land between them; the
///   structural path proves its read with one `HGET obj:hdr seq` afterwards
///   (SR-4.3);
/// - `ServiceError::NotFound` when the header is absent, or present without
///   a numeric `seq`;
/// - a classified error otherwise: a failed command is attributed to its own
///   key family and classified by [`classify_snapshot_read_err`] on the
///   snapshot lane's pool label (transient → `Aborted`, permanent →
///   `Internal`, the refusal counters ticked).
///
/// On either path, a field or time-sample row whose external `hash` column is
/// not 64 hex characters fails the read `Internal` (see `decode_field_value`)
/// rather than decoding to a wrong hash.
///
/// The structural cache's consumers
/// ([`SlimSnapshotReadConsumer::StructuralCache`] and
/// [`SlimSnapshotReadConsumer::StructuralCacheAudit`]) are atomic-only, like
/// the catch-up and checkpoint consumers, and their `TooLarge` is final here as
/// theirs is. Their reads also set the script's `return_dangling` flag, so
/// their snapshots carry
/// [`AtomicSnapshotOf::structural`]: the header's `commit_chain`, `superseded`
/// and `chs_complete`, and the dangling parents the read excluded, all at the
/// cut. Every reader's atomic read carries `None` there.
///
/// Every completed read records `live_edit_snapshot_read_total` and
/// `live_edit_snapshot_read_duration_seconds` under its path; a successful one
/// also records the reply-bytes and entries histograms. It also emits one
/// DEBUG line with its own duration ([`SNAPSHOT_READ_TIMING_TARGET`]), off at
/// the default log level.
pub async fn read_object_snapshot(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    consumer: SnapshotReadConsumer,
) -> Result<SnapshotRead, ServiceError> {
    read_object_snapshot_with_ceilings(
        client,
        keys,
        object_id,
        consumer,
        SnapshotCeilings::of_client(client),
    )
    .await
}

/// The `tracing` target of the one DEBUG line every completed snapshot read
/// emits ([`log_snapshot_read_timing`]). Nothing enables it by default; a
/// measurement that needs each read's own duration, not the
/// `live_edit_snapshot_read_duration_seconds` buckets, turns it on alone with
/// the directive `LIVE_EDIT_LOG_LEVEL=info,live_edit::snapshot_read_timing=debug`.
///
/// The structural-edits validation harness parses this line
/// (`docs-internal/validation/structural-edits-b2wo-20261002/scripts/analyze.py`,
/// its `SNAP_TIMING_*` constants): it matches this target, the message
/// [`SNAPSHOT_READ_TIMING_MSG`] and the field names `consumer`, `path`,
/// `outcome` and `elapsed_us`. Renaming any of them breaks its per-read
/// figures, so change that script with them.
pub const SNAPSHOT_READ_TIMING_TARGET: &str = "live_edit::snapshot_read_timing";

/// The message of that line (matched by the harness; see
/// [`SNAPSHOT_READ_TIMING_TARGET`]).
pub const SNAPSHOT_READ_TIMING_MSG: &str = "snapshot read timing";

/// One completed snapshot read's DEBUG line: its object id and the consumer,
/// path, outcome and duration `live_edit_snapshot_read_duration_seconds`
/// records for it, the duration in whole microseconds (`elapsed_us`). The
/// validation harness reads these field names by name
/// ([`SNAPSHOT_READ_TIMING_TARGET`]).
fn log_snapshot_read_timing(
    object_id: &str,
    consumer: impl SnapshotConsumer,
    path: SnapshotReadPath,
    outcome: &'static str,
    elapsed: std::time::Duration,
) {
    tracing::debug!(
        target: SNAPSHOT_READ_TIMING_TARGET,
        object_id,
        consumer = consumer.label(),
        path = path.label(),
        outcome,
        elapsed_us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
        "{SNAPSHOT_READ_TIMING_MSG}"
    );
}

/// [`read_object_snapshot`] with the atomic read's item and byte ceilings
/// passed for this call instead of read from the client.
///
/// For the structural cache's reads: the cold start's plain read, at the
/// client's ceilings (structural-edits SDD SR-4.3), and the sampled audit's
/// plain read (SR-3.6), also at the client's ceilings, before its fenced
/// read above them. The measurement tools pass larger ceilings (the
/// raised-ceiling read). `ceilings` binds the atomic path only: a consumer
/// that reads through the unfenced fan-out has no ceiling, and ignores it.
/// Everything else — metrics, the timing line, errors, the reply — is
/// exactly [`read_object_snapshot`]'s.
pub async fn read_object_snapshot_with_ceilings(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    consumer: SnapshotReadConsumer,
    ceilings: SnapshotCeilings,
) -> Result<SnapshotRead, ServiceError> {
    uncharged(
        object_id,
        read_snapshot_with_ceilings(client, keys, object_id, consumer, ceilings, None).await?,
    )
}

/// The answer of a read that carried no [`FanoutAllowance`], which never
/// stops: a stop is refused `Internal` rather than guessed at.
fn uncharged<T>(object_id: &str, read: Result<T, u64>) -> Result<T, ServiceError> {
    read.map_err(|_| {
        ServiceError::Internal(format!(
            "snapshot read of object {object_id} stopped for its upload charge, which only the \
             structural update's charged read carries"
        ))
    })
}

/// The slim read of one object's state for a structural write-planning
/// consumer: the read [`read_object_snapshot`] documents for that consumer's
/// path — the same ceilings, errors, metrics (under the consumer's label,
/// with a full read's tallies) and timing line — with the state decoded into
/// the compact form ([`CompactState`]; see [`SlimSnapshotReadConsumer`] for
/// how each path reads it).
///
/// The state carries no tombstones, which no structural plan reads; on the
/// atomic path [`AtomicSnapshotOf::reply_bytes`] still counts the inline
/// values the script read but did not send.
pub async fn read_object_snapshot_slim(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    consumer: SlimSnapshotReadConsumer,
) -> Result<SnapshotReadOf<CompactState>, ServiceError> {
    read_object_snapshot_slim_with_ceilings(
        client,
        keys,
        object_id,
        consumer,
        SnapshotCeilings::of_client(client),
    )
    .await
}

/// [`read_object_snapshot_slim`] with the atomic read's ceilings passed for
/// this call, as [`read_object_snapshot_with_ceilings`] is to
/// [`read_object_snapshot`].
pub async fn read_object_snapshot_slim_with_ceilings(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    consumer: SlimSnapshotReadConsumer,
    ceilings: SnapshotCeilings,
) -> Result<SnapshotReadOf<CompactState>, ServiceError> {
    uncharged(
        object_id,
        read_snapshot_with_ceilings(client, keys, object_id, consumer, ceilings, None).await?,
    )
}

/// What [`read_object_snapshot_slim_charged`] answers.
pub(crate) enum ChargedSnapshotRead {
    /// The whole object, read as [`read_object_snapshot_slim`] reads it for
    /// the structural update.
    Snapshot(Box<AtomicSnapshotOf<CompactState>>),
    /// The read stopped before its next window: the upload charge could not
    /// grow, without waiting, to cover it. `target` is the best-known whole
    /// object-state target the attempt needs (see [`FanoutAllowance`]).
    Stopped { target: u64 },
}

/// The structural `UpdateLayer` WATCH path's read of one object
/// ([`SlimSnapshotReadConsumer::StructuralUpdate`], the unfenced fan-out),
/// charged as it reads: [`read_object_snapshot_slim`]'s read — that
/// consumer's bounded-reply read, which transfers no inline value
/// ([`FanoutReplies::Bounded`]) — with `allowance` growing the call's upload
/// charge before each part of the read it covers (SDD §5.4):
///
/// - after the count windows, before any index member is read, to the read's
///   whole count-based need — the attempt's fixed cost, every entry's
///   round-1 member and command, one window of wire commands in flight and a
///   byte window: the smaller of `FANOUT_WINDOW_BYTES` and the read's
///   projected reply and script-request bound from its counts at the name
///   floor ([`projected_read_reply_bound`], [`projected_read_request_bound`]),
///   so a small object holds about its own windows' bound rather than a
///   whole window (WA1);
/// - before each later window and index page is issued, to the sum of its
///   commands' per-command reply bounds, a script window's request bytes and
///   the items they can add, with names sized at the longest the read has
///   seen, at least [`NAME_BYTES_SIZING_FLOOR`], and for a plain window
///   (an index page, a lone long children list, a window after the read's
///   script fallback) its wire commands, each holding its own future;
/// - after each, to what it received, the items tallied and the name bytes
///   the read retains.
///
/// Each growth never waits and never shrinks the charge
/// ([`SharedCommitCharge::try_hold_at_least`]); a refused one stops the read
/// before its next window, answers [`ChargedSnapshotRead::Stopped`] and
/// records the read as `live_edit_snapshot_read_total{consumer=
/// "structural_update", path="unfenced", outcome="too_large"}`. Three
/// residuals are charged after they arrive:
///
/// - a page or window of names longer than any the read has seen, at most
///   one at a time per attempt — 32,715,522 B for a field-index page and
///   8,404,709 B for a section window at `MAX_NAME_BYTES` [code] — outside
///   §21.2's line until its own Issue lands;
/// - a round-1 `SSCAN` step on a set index the store keeps in its compact
///   encoding, which answers the whole set whatever its `COUNT`: at most
///   once per set index per read (its scan then ends), and at most
///   max(`set-max-intset-entries`, `set-max-listpack-entries`) ×
///   [`ID_REPLY_BOUND`] per index — 10,240 B at the store's defaults (512,
///   128), 30,720 B per read over the three set indices;
/// - a round-1 `SSCAN` step on a hashtable-encoded set index that returns
///   more than its charged share (at least what its `COUNT` asks for, at
///   least twice that for a share above one): the store walks the table
///   until it has collected `COUNT` members, each bucket adding its whole
///   chain and, during a rehash, every larger-table bucket the smaller
///   table's bucket expands to (Valkey 8.0 `scanGenericCommand`). A few
///   members × [`ID_REPLY_BOUND`] per step; over a read never more than the
///   set's own size × [`ID_REPLY_BOUND`], but no constant bounds it.
///
/// After the read, `allowance` holds the name bytes the read retained
/// ([`FanoutAllowance::name_bytes`]), which the attempt's later charge points
/// keep.
///
/// Crate-internal and for that consumer only. The fenced pass carries no
/// allowance and reads whole rows, as before.
pub(crate) async fn read_object_snapshot_slim_charged(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    allowance: &mut FanoutAllowance<'_>,
) -> Result<ChargedSnapshotRead, ServiceError> {
    match read_snapshot_with_ceilings(
        client,
        keys,
        object_id,
        SlimSnapshotReadConsumer::StructuralUpdate,
        SnapshotCeilings::of_client(client),
        Some(allowance),
    )
    .await?
    {
        Ok(SnapshotReadOf::Snapshot(snapshot)) => Ok(ChargedSnapshotRead::Snapshot(snapshot)),
        // The unfenced fan-out has no ceiling, so this consumer is never
        // answered `TOO_LARGE` by a ceiling; refuse rather than guess.
        Ok(SnapshotReadOf::TooLarge(_)) => Err(ServiceError::Internal(format!(
            "structural snapshot read of object {object_id} answered TOO_LARGE, which the \
             unfenced fan-out never answers"
        ))),
        Err(target) => Ok(ChargedSnapshotRead::Stopped { target }),
    }
}

/// The read [`read_object_snapshot_with_ceilings`],
/// [`read_object_snapshot_slim_with_ceilings`] and
/// [`read_object_snapshot_slim_charged`] run: `consumer` picks the path and
/// its type the state's form, and on the unfenced fan-out its reply shape
/// ([`FanoutReplies`]). `allowance`, the charged read's only, rides the
/// unfenced fan-out; `Err(target)` inside the `Ok` is its stop.
async fn read_snapshot_with_ceilings<C: SnapshotConsumer>(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    consumer: C,
    ceilings: SnapshotCeilings,
    allowance: Option<&mut FanoutAllowance<'_>>,
) -> Result<Result<SnapshotReadOf<C::State>, u64>, ServiceError> {
    let started = std::time::Instant::now();
    let (path, result) = if consumer.takes_atomic_path() {
        (
            SnapshotReadPath::Atomic,
            read_object_snapshot_atomic::<C::State>(
                client,
                keys,
                object_id,
                ceilings,
                consumer.reads_structural_facts(),
            )
            .await
            .map(Ok),
        )
    } else {
        (
            SnapshotReadPath::Unfenced,
            read_object_snapshot_unfenced::<C::State>(client, keys, object_id, allowance)
                .await
                .map(|outcome| match outcome {
                    FanoutOutcome::Read(snapshot) => Ok(SnapshotReadOf::Snapshot(snapshot)),
                    FanoutOutcome::Stopped { target } => Err(target),
                }),
        )
    };
    let elapsed = started.elapsed();
    let outcome = match &result {
        Ok(Ok(SnapshotReadOf::Snapshot(_))) => metrics::SNAPSHOT_READ_OUTCOME_OK,
        // A ceiling's `TOO_LARGE`, or a charged read's stop for its upload
        // charge (the structural update's unfenced read only).
        Ok(Ok(SnapshotReadOf::TooLarge(_)) | Err(_)) => metrics::SNAPSHOT_READ_OUTCOME_TOO_LARGE,
        Err(ServiceError::NotFound(_)) => metrics::SNAPSHOT_READ_OUTCOME_NOT_FOUND,
        Err(ServiceError::Aborted(_)) => metrics::SNAPSHOT_READ_OUTCOME_TRANSIENT_ERROR,
        Err(_) => metrics::SNAPSHOT_READ_OUTCOME_PERMANENT_ERROR,
    };
    log_snapshot_read_timing(object_id, consumer, path, outcome, elapsed);
    if let Some(m) = client.metrics_snapshot() {
        m.observe_snapshot_read(consumer.label(), path.label(), outcome, elapsed);
        if let Ok(Ok(SnapshotReadOf::Snapshot(snapshot))) = &result {
            m.observe_snapshot_read_size(
                consumer.label(),
                snapshot.reply_bytes,
                snapshot.entry_count,
            );
        }
    }
    result
}

/// The script's `ARGV[12]` value that sets `return_dangling`. Unset, the
/// argument is not passed at all, so the eleven arguments every other read
/// sends are unchanged.
const RETURN_DANGLING_SET: &str = "1";

/// The script's `ARGV[12]` value that leaves `return_dangling` unset, passed
/// only when `ARGV[13]` follows it.
const RETURN_DANGLING_UNSET: &str = "0";

/// The script's `ARGV[13]` value that sets the `slim` value-rows flag. Unset,
/// the argument is not passed at all.
const SLIM_SET: &str = "1";

/// The atomic path of a snapshot read: the `EVALSHA`, its error
/// classification and the reply's decode into `S`, under `ceilings`, with the
/// script's `return_dangling` flag set when `structural_facts` is and its
/// `slim` flag when `S` is the compact form.
async fn read_object_snapshot_atomic<S: SnapshotState>(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    ceilings: SnapshotCeilings,
    structural_facts: bool,
) -> Result<SnapshotReadOf<S>, ServiceError> {
    let oid = object_id.parse::<u64>().map_err(|e| {
        ServiceError::Internal(format!(
            "read_object_snapshot called with non-numeric object_id '{object_id}': {e}"
        ))
    })?;
    let header_key = keys.obj_header(object_id);
    let mut argv = object_snapshot_read_argv(
        keys,
        object_id,
        "read",
        ceilings.max_items,
        ceilings.max_bytes,
    );
    if structural_facts {
        argv.push(RETURN_DANGLING_SET.to_owned());
    } else if S::SLIM {
        argv.push(RETURN_DANGLING_UNSET.to_owned());
    }
    if S::SLIM {
        argv.push(SLIM_SET.to_owned());
    }

    // The test-only fault seam answers in place of the `EVALSHA` and takes the
    // same classification below as a real error.
    let reply = match client.object_snapshot_read_faults.take_next_read() {
        Some(injected) => Err(injected),
        None => eval_object_snapshot_read(client, &header_key, argv).await,
    }
    .map_err(|e| classify_object_snapshot_read_err(client, object_id, &header_key, e))?;

    decode_object_snapshot_reply::<S>(object_id, oid, ceilings, structural_facts, reply)
}

/// The script's `ARGV` (see `scripts/object_snapshot_read.lua`).
fn object_snapshot_read_argv(
    keys: &KeySchema,
    object_id: &str,
    mode: &str,
    max_items: u64,
    max_bytes: u64,
) -> Vec<String> {
    vec![
        keys.prefix().to_owned(),
        object_id.to_owned(),
        mode.to_owned(),
        max_items.to_string(),
        max_bytes.to_string(),
        SNAPSHOT_SECTION_HASH_WIDTH.to_string(),
        SNAPSHOT_TOMBSTONE_HASH_WIDTH.to_string(),
        SNAPSHOT_FIELD_HASH_WIDTH.to_string(),
        SNAPSHOT_TIME_SAMPLE_HASH_WIDTH.to_string(),
        crate::storage::layer::HEADER_FIELD_CHS_COMPLETE.to_owned(),
        crate::storage::layer::CHS_COMPLETE_MARKED.to_owned(),
    ]
}

/// One `EVALSHA` of the script on the snapshot lane, with the header as its one
/// declared key (the slot pin; never an empty key list, which fred would route
/// to an arbitrary primary).
async fn eval_object_snapshot_read(
    client: &ValkeyClient,
    header_key: &str,
    argv: Vec<String>,
) -> Result<fred::types::Value, fred::error::Error> {
    let (pool, _in_use) = client.acquire_snapshot();
    crate::storage::lua::OBJECT_SNAPSHOT_READ_SCRIPT
        .evalsha_with_reload::<fred::types::Value, _, _>(
            pool.next(),
            vec![header_key.to_owned()],
            argv,
        )
        .await
}

/// Classify a failed atomic read, ticking the refusal counters on the way
/// (Spec SDD SR-1.4).
///
/// A tagged per-key refusal (`SNAPSHOT_KEY_REFUSED: <family> <key> <raw>`) is
/// counted under its own family and key through [`count_object_key_refusal`]:
/// a `WRONGTYPE` raw error is handed over as the server's `WRONGTYPE` reply
/// (`ErrorKind::InvalidArgument`, the shape
/// `client::is_server_wrongtype_error` recognises), so it ticks
/// `live_edit_snapshot_read_type_error_total{key_family}`; anything else but a
/// cluster-state refusal (below) stays the script's own `ErrorKind::Unknown`
/// reply, which ticks
/// `live_edit_snapshot_read_refused_total{kind="unknown"}`. Both are permanent
/// by `map_snapshot_load_err`'s rule, so the read fails `Internal` — no tag
/// steers anything beyond "fail" (the b.zvd posture).
///
/// Every other failure is a script-level one (the `EVALSHA` itself: a
/// `NOSCRIPT` reload failure, `-BUSY`, `-LOADING`, a timeout, IO, a cluster
/// redirect), attributed to [`SnapshotKeyFamily::Header`] and classified by
/// `map_snapshot_load_err` unchanged. A tag naming a family this binary does
/// not know is a script/binary mismatch and is treated the same way.
///
/// One family of tagged refusals is not a refusal of the key at all: the
/// cluster refusing a script's undeclared key because of the slot's state
/// ([`is_script_cluster_state_refusal`]: the slot migrating under the read, the
/// key gone to another node, the cluster down). Those are the in-script form of
/// the cluster redirects listed above, so they are attributed to
/// [`SnapshotKeyFamily::Header`] and classified as the transient cluster error
/// they are (`ErrorKind::Cluster` → `Aborted`), ticking no refusal counter.
/// Without this, every atomic read of an object in a migrating slot — whose
/// undeclared keys are often absent (`obj:dels`, `obj:tss`, a leaf section's
/// children list), and an absent key in a migrating slot answers ASK — would
/// fail non-retryably for the whole resharding.
fn classify_object_snapshot_read_err(
    client: &ValkeyClient,
    object_id: &str,
    header_key: &str,
    e: fred::error::Error,
) -> ServiceError {
    if let Some((family, key, raw)) = parse_snapshot_key_refusal(&e) {
        if is_script_cluster_state_refusal(raw) {
            let op = format!(
                "EVALSHA object_snapshot_read (the cluster refused the in-script read of {key})"
            );
            return classify_snapshot_read_err(
                client,
                POOL_LABEL_SNAPSHOT,
                object_id,
                SnapshotKeyFamily::Header,
                header_key,
                &op,
                fred::error::Error::new(fred::error::ErrorKind::Cluster, e.details().to_owned()),
            );
        }
        let op = "EVALSHA object_snapshot_read (in-script read)";
        let counted = if raw.starts_with("WRONGTYPE ") {
            fred::error::Error::new(fred::error::ErrorKind::InvalidArgument, raw.to_owned())
        } else {
            e.clone()
        };
        let key = key.to_owned();
        return classify_snapshot_read_err(
            client,
            POOL_LABEL_SNAPSHOT,
            object_id,
            family,
            &key,
            op,
            counted,
        );
    }
    classify_snapshot_read_err(
        client,
        POOL_LABEL_SNAPSHOT,
        object_id,
        SnapshotKeyFamily::Header,
        header_key,
        "EVALSHA object_snapshot_read",
        e,
    )
}

/// Parse a `SNAPSHOT_KEY_REFUSED: <family> <key> <raw error>` script reply into
/// its family, key and raw error.
///
/// Kind-gated and prefix-anchored: a script's own error reply arrives as
/// `ErrorKind::Unknown` (fred maps only the leading tokens it knows), so the tag
/// is trusted only there. Keys carry no spaces (`KeySchema` builds them from a
/// prefix, an id and hex segments), so the key ends at the first space after
/// the family.
fn parse_snapshot_key_refusal(e: &fred::error::Error) -> Option<(SnapshotKeyFamily, &str, &str)> {
    if !matches!(e.kind(), fred::error::ErrorKind::Unknown) {
        return None;
    }
    let rest = e.details().strip_prefix(SNAPSHOT_KEY_REFUSED_TAG)?;
    let (label, rest) = rest.split_once(' ')?;
    let (key, raw) = rest.split_once(' ')?;
    let family = SnapshotKeyFamily::ALL
        .into_iter()
        .find(|f| f.label() == label)?;
    Some((family, key, raw))
}

/// Whether an in-script error (the raw error a `redis.pcall` returned, as a
/// tagged refusal carries it) is the cluster refusing an undeclared key
/// because of the slot's state, rather than refusing the key itself.
///
/// # Provenance
///
/// Valkey's `scriptVerifyClusterState` (`src/script.c:423-461` at 8.1.1 and
/// 9.0.3; the same texts in Redis OSS 7.4.3's, `src/script.c:462-502`) answers
/// a script command whose key this node cannot serve with one of these. The
/// message carries no `-CODE` of its own, so `luaPushErrorBuff`
/// (`src/lua/script_lua.c:550` at 9.0.3, `:551` at 8.1.1) prepends `ERR `
/// before `redis.pcall` returns it:
///
/// - `Script attempted to access a non local key in a cluster node` — MOVED or
///   ASK, which is what an absent key in a MIGRATING slot answers;
/// - `Unable to execute command '<cmd>' in script because undeclared keys were
///   accessed during rehashing of the slot` — the slot is importing or
///   migrating and the key is undeclared;
/// - `Script attempted to execute a command while the cluster is down`;
/// - `Script attempted to access a slot not served`.
///
/// Azure Managed Redis (closed source) words the first of these
/// `Lua script attempted to access a non local key in a cluster node`: the
/// Epic 1 cost report captured it refusing a cross-node key
/// (`docs-internal/validation/snapshot-read-cost-20260927/report.md`, the
/// runner console lines of its step 2 self-test). That is the older Redis OSS
/// wording, `src/scripting.c:820-822` at 6.2.14 (Redis 7.2.5 says `Script
/// attempted …`, `src/script.c:441`). Same condition, same answer: transient.
/// Matching it is safe on AMR because AMR was measured accepting undeclared
/// same-slot reads, so this text there cannot be a policy refusal of such a
/// read.
///
/// The capture came from `redis.call`, not `redis.pcall`, as
/// `Lua script … cluster node script: <sha>, on @user_script:30.` — the
/// ` script: <sha>, on <source>:<line>.` suffix is what Redis 7.x's
/// `luaCallFunction` appends to a raised error (`src/script_lua.c:1704-1710`
/// at 7.2.5). In that lineage `redis.pcall` returns `luaPushErrorBuff`'s
/// `ERR <msg>` with no source or line (`src/script_lua.c:560-567` at 7.2.5),
/// which is the form matched here (with or without its `ERR ` code, like the
/// Valkey texts). Redis 6.2's `luaPushError` instead prefixes
/// the pcall form with `<source>: <line>: ` (`src/scripting.c:390-394` at
/// 6.2.14), but a 6.2 engine would have answered that `redis.call` as
/// `Error running script (call to f_<sha>): …` (`src/scripting.c:1802`), not
/// with the 7.x suffix AMR sent. So the prefixed form is deliberately not
/// accepted: the captured bytes contradict it. AMR's actual `redis.pcall`
/// bytes have never been captured; if they differ from `ERR <msg>`, this
/// match misses: the read fails as a permanent refusal and the startup
/// self-test answers `UndeclaredKeyRefused`, as both did before AMR's wording
/// was listed.
///
/// All of these clear without an operator: the migration finishes, the cluster
/// recovers or the slot is served again. So a retry is the right answer. The
/// cross-slot texts of the same function are deliberately not here: every key
/// the script builds carries the object's own hash tag, so they cannot arise
/// from a slot's state, only from a bug.
///
/// # Anti-creep
///
/// Anchored and exact, in the shape of `client::is_router_no_connection_error`
/// and `client::is_router_retry_budget_exhausted_error`: the text must be the
/// whole message, after an optional `ERR ` code, and only the command name of
/// the rehashing text varies. Never a substring search. A future addition needs
/// the same evidence: a named server origin at a `file:line`, or, for a
/// closed-source server, a captured reply.
fn is_script_cluster_state_refusal(raw: &str) -> bool {
    const EXACT: [&str; 4] = [
        "Script attempted to access a non local key in a cluster node",
        "Script attempted to execute a command while the cluster is down",
        "Script attempted to access a slot not served",
        // Azure Managed Redis's wording of the first (Redis OSS 6.2.14's).
        "Lua script attempted to access a non local key in a cluster node",
    ];
    const REHASHING_PREFIX: &str = "Unable to execute command '";
    const REHASHING_SUFFIX: &str =
        "' in script because undeclared keys were accessed during rehashing of the slot";
    let msg = raw.strip_prefix("ERR ").unwrap_or(raw);
    EXACT.contains(&msg)
        || msg
            .strip_prefix(REHASHING_PREFIX)
            .and_then(|rest| rest.strip_suffix(REHASHING_SUFFIX))
            .is_some_and(|cmd| !cmd.is_empty() && !cmd.contains('\''))
}

/// Decode the script's reply (shapes in `scripts/object_snapshot_read.lua`).
/// `ceilings` are the ones the read ran under; `structural_facts` is whether
/// it set `return_dangling`, which adds one trailing element to an `OK` reply
/// and nothing to any other.
fn decode_object_snapshot_reply<S: SnapshotState>(
    object_id: &str,
    oid: u64,
    ceilings: SnapshotCeilings,
    structural_facts: bool,
    reply: fred::types::Value,
) -> Result<SnapshotReadOf<S>, ServiceError> {
    let parts = reply.into_array();
    let arity = parts.len();
    let mut parts = parts.into_iter();
    let format = reply_string(parts.next(), object_id, "format tag")?;
    if format != OBJECT_SNAPSHOT_READ_FORMAT {
        return Err(ServiceError::Internal(format!(
            "object_snapshot_read.lua answered for object {object_id} in format {format:?}, \
             expected {OBJECT_SNAPSHOT_READ_FORMAT:?}: the script that ran is not the one this \
             binary carries"
        )));
    }
    let status = reply_string(parts.next(), object_id, "status")?;
    // Every element below is required: a short reply must not decode as empty
    // indices or an empty header. `return_dangling` adds the trailing
    // dangling-parents element to `OK` only.
    let expected_arity = match status.as_str() {
        "OK" if structural_facts => 10,
        "OK" => 9,
        "NOT_FOUND" => 2,
        "TOO_LARGE" => 6,
        _ => arity,
    };
    if arity != expected_arity {
        return Err(reply_shape_err(
            object_id,
            format!("a {status} reply has {arity} elements, expected {expected_arity}"),
        ));
    }
    match status.as_str() {
        "OK" => decode_object_snapshot_ok::<S>(object_id, oid, structural_facts, parts)
            .map(|snapshot| SnapshotReadOf::Snapshot(Box::new(snapshot))),
        "NOT_FOUND" => Err(ServiceError::NotFound(format!(
            "object {object_id} not found"
        ))),
        "TOO_LARGE" => {
            let too_large = decode_object_snapshot_too_large(object_id, parts)?;
            if let SnapshotTooLargeAt::MidRead {
                limit: SnapshotLimit::Items,
                item_tally,
                byte_tally,
            } = too_large.at
            {
                // The anomaly arm (Spec SDD SR-1.2): step 2's bound covers every
                // item a well-formed object holds, so only an object that broke
                // the per-family width or the one-parent-per-section invariant
                // can cross the item limit after passing it.
                let c = too_large.cardinalities;
                tracing::warn!(
                    object_id = %object_id,
                    sections = c.sections,
                    tombstones = c.tombstones,
                    fields = c.fields,
                    time_samples = c.time_samples,
                    children_index = ?c.children_index,
                    item_bound = too_large.item_bound,
                    item_tally,
                    byte_tally,
                    item_limit = ceilings.max_items,
                    "atomic snapshot read stopped on the item tally after its item bound had \
                     passed: the object holds more items than the per-family hash widths and \
                     the children tree allow, so a writer added a hash column the width \
                     constants do not count or a children list holds ids that are not live \
                     sections; the layer takes the above-ceiling path"
                );
            }
            Ok(SnapshotReadOf::TooLarge(too_large))
        }
        other => Err(reply_shape_err(
            object_id,
            format!("unknown status {other:?}"),
        )),
    }
}

/// Decode an `OK` reply's body:
/// `hdr, secs, dels, flds, tss, children, {items, bytes, entries}`, then, when
/// `structural_facts` (the read set `return_dangling`), `dangling`. `flds`
/// and `tss` are decoded into `S`'s rows: whole, or, for the compact form,
/// slim (the read set the script's `slim` flag).
fn decode_object_snapshot_ok<S: SnapshotState>(
    object_id: &str,
    oid: u64,
    structural_facts: bool,
    mut parts: impl Iterator<Item = fred::types::Value>,
) -> Result<AtomicSnapshotOf<S>, ServiceError> {
    let header = reply_flat_hash(parts.next(), object_id, "header")?;
    let sections_raw = reply_entries(parts.next(), object_id, "sections")?;
    let tombstones_raw = reply_entries(parts.next(), object_id, "tombstones")?;
    let fields_raw = S::atomic_rows(parts.next(), object_id, "fields")?;
    let time_samples_raw = S::atomic_rows(parts.next(), object_id, "time samples")?;
    let children_raw = reply_entries(parts.next(), object_id, "children")?;
    let [item_count, reply_bytes, entry_count] = reply_tallies(parts.next(), object_id)?;
    let structural = if structural_facts {
        Some(StructuralSnapshotFacts {
            commit_chain: header
                .get(crate::storage::layer::HEADER_FIELD_COMMIT_CHAIN)
                .cloned()
                .unwrap_or_default(),
            superseded: header
                .get(crate::storage::layer::HEADER_FIELD_SUPERSEDED)
                .cloned(),
            chs_complete: header
                .get(crate::storage::layer::HEADER_FIELD_CHS_COMPLETE)
                .cloned(),
            dangling_parents: reply_dangling_parents(parts.next(), object_id)?,
        })
    } else {
        None
    };

    // A header without a numeric `seq` is not a live object for these
    // consumers: `read_object_header`'s rule, which every catch-up pin read
    // already applies (Spec SDD SR-1.5).
    let Some(cut) = header
        .get(crate::storage::layer::HEADER_FIELD_SEQ)
        .and_then(|s| s.parse::<u64>().ok())
    else {
        return Err(ServiceError::NotFound(format!(
            "object {object_id} header partial (missing or non-numeric seq); treating as \
             not-found"
        )));
    };

    let mut sections = HashMap::with_capacity(sections_raw.len());
    // Under one atomic cut an indexed section always has its hash and a decimal
    // id, so a member without either is a corrupt index. It is decoded exactly as
    // the fan-out decodes it and recorded for the consumer to judge (see
    // `AtomicSnapshot::corrupt_section_index_members`).
    let mut corrupt_section_index_members = Vec::new();
    for (sid, value) in sections_raw {
        let data = reply_flat_hash(Some(value), object_id, "section hash")?;
        let decoded = decode_section(&sid, &data);
        let corrupt = decoded.is_none() || sid.parse::<u64>().is_err();
        if let Some(section) = decoded {
            sections.insert(section.section_id, section);
        }
        if corrupt {
            corrupt_section_index_members.push(sid);
        }
    }
    let mut tombstones = Vec::with_capacity(tombstones_raw.len());
    for (sid, value) in tombstones_raw {
        let data = reply_flat_hash(Some(value), object_id, "tombstone hash")?;
        if let Some(tombstone) = decode_tombstone(&sid, &data) {
            tombstones.push(tombstone);
        }
    }
    let mut children: HashMap<u64, Vec<u64>> = HashMap::new();
    for (pid, value) in children_raw {
        let child_strs = reply_string_list(Some(value), object_id, "children list")?;
        if let Some((pid, child_ids)) = decode_children_list(&pid, &child_strs) {
            children.insert(pid, child_ids);
        }
    }
    let (value_rows, value_row_bytes_received) =
        S::decode_atomic(object_id, fields_raw, time_samples_raw)?;
    #[cfg(not(any(test, feature = "test-support")))]
    let _ = value_row_bytes_received;

    let state = S::assemble_atomic(
        sealed::SnapshotParts {
            object_id: oid,
            sequence_num: cut,
            first_version: parse_u64_warn(&header, "first_ver", object_id),
            source_format: header.get("src_fmt").cloned().unwrap_or_default(),
            next_set_order: parse_u64_warn(&header, "next_set_order", object_id),
            sections,
            children,
            tombstones,
        },
        value_rows,
        object_id,
    );
    Ok(AtomicSnapshotOf {
        state,
        cut,
        last_server_ts: parse_u64_warn(&header, "last_server_ts", object_id),
        last_client_ts: parse_u64_warn(&header, "last_client_ts", object_id),
        reply_bytes,
        item_count,
        entry_count,
        path: SnapshotReadPath::Atomic,
        corrupt_section_index_members,
        structural,
        memory: SnapshotMemory::default(),
        #[cfg(any(test, feature = "test-support"))]
        value_row_bytes_received,
    })
}

/// The `return_dangling` element of an `OK` reply: an array of `obj:chs`
/// members (`{}` when there are none). Strict on shape — a scalar or a
/// non-string member is a malformed reply, never an empty set, since an
/// under-reported set would let a replayed creation adopt a list the cache
/// entry does not hold. A member that is a string but not a decimal id is
/// skipped: no section can ever be created with that id, so its list can
/// never become visible, and the script never returns a list under a parent
/// that is not live.
fn reply_dangling_parents(
    value: Option<fred::types::Value>,
    object_id: &str,
) -> Result<Vec<u64>, ServiceError> {
    match value {
        Some(fred::types::Value::Array(members)) => {
            let mut parents = Vec::with_capacity(members.len());
            for member in members {
                let member = member.into_string().ok_or_else(|| {
                    reply_shape_err(
                        object_id,
                        "dangling parents hold a non-UTF-8 or non-string member",
                    )
                })?;
                if let Ok(parent) = member.parse::<u64>() {
                    parents.push(parent);
                }
            }
            Ok(parents)
        }
        other => Err(reply_shape_err(
            object_id,
            format!("dangling parents are not an array: {other:?}"),
        )),
    }
}

/// Decode a `TOO_LARGE` reply's body: `which, cards, bound, {items, bytes,
/// entries}`.
fn decode_object_snapshot_too_large(
    object_id: &str,
    mut parts: impl Iterator<Item = fred::types::Value>,
) -> Result<SnapshotTooLarge, ServiceError> {
    let which = reply_string(parts.next(), object_id, "TOO_LARGE limit")?;
    let cards = parts.next().map(|v| v.into_array()).unwrap_or_default();
    let cards: Vec<i64> = cards
        .iter()
        .map(|v| v.as_i64())
        .collect::<Option<Vec<_>>>()
        .filter(|c| c.len() == 6)
        .ok_or_else(|| {
            reply_shape_err(object_id, "TOO_LARGE cardinalities are not six integers")
        })?;
    let card = |i: usize| u64::try_from(cards[i]).unwrap_or(0);
    let cardinalities = SnapshotCardinalities {
        sections: card(0),
        tombstones: card(1),
        fields: card(2),
        time_samples: card(3),
        children_index: (cards[5] == 1).then(|| card(4)),
    };
    let item_bound = reply_u64(parts.next(), object_id, "TOO_LARGE item bound")?;
    let [item_tally, byte_tally, _entries] = reply_tallies(parts.next(), object_id)?;
    let at = match which.as_str() {
        "bound" => SnapshotTooLargeAt::EarlyOut,
        "items" => SnapshotTooLargeAt::MidRead {
            limit: SnapshotLimit::Items,
            item_tally,
            byte_tally,
        },
        "bytes" => SnapshotTooLargeAt::MidRead {
            limit: SnapshotLimit::Bytes,
            item_tally,
            byte_tally,
        },
        other => {
            return Err(reply_shape_err(
                object_id,
                format!("unknown TOO_LARGE limit {other:?}"),
            ))
        }
    };
    Ok(SnapshotTooLarge {
        cardinalities,
        item_bound,
        at,
    })
}

/// The `Internal` error for a reply that does not have the script's shape.
fn reply_shape_err(object_id: &str, what: impl std::fmt::Display) -> ServiceError {
    ServiceError::Internal(format!(
        "object_snapshot_read.lua reply for object {object_id} is malformed: {what}"
    ))
}

/// One string element of the reply (`None` = the reply ran out).
fn reply_string(
    value: Option<fred::types::Value>,
    object_id: &str,
    what: &str,
) -> Result<String, ServiceError> {
    value
        .and_then(|v| v.into_string())
        .ok_or_else(|| reply_shape_err(object_id, format!("{what} is absent or not a string")))
}

/// One non-negative integer element of the reply.
fn reply_u64(
    value: Option<fred::types::Value>,
    object_id: &str,
    what: &str,
) -> Result<u64, ServiceError> {
    value
        .and_then(|v| v.as_u64())
        .ok_or_else(|| reply_shape_err(object_id, format!("{what} is absent or not an integer")))
}

/// The `{items, bytes, entries}` tally triple.
fn reply_tallies(
    value: Option<fred::types::Value>,
    object_id: &str,
) -> Result<[u64; 3], ServiceError> {
    let tallies: Vec<u64> = value
        .map(|v| v.into_array())
        .unwrap_or_default()
        .iter()
        .map(|v| v.as_u64())
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    <[u64; 3]>::try_from(tallies)
        .map_err(|_| reply_shape_err(object_id, "tallies are not three integers"))
}

/// An array of strings (an index or a children list).
fn reply_string_list(
    value: Option<fred::types::Value>,
    object_id: &str,
    what: &str,
) -> Result<Vec<String>, ServiceError> {
    value
        .map(|v| v.into_array())
        .unwrap_or_default()
        .into_iter()
        .map(|v| {
            v.into_string().ok_or_else(|| {
                reply_shape_err(
                    object_id,
                    format!("{what} holds a non-UTF-8 or non-string member"),
                )
            })
        })
        .collect()
}

/// A flat `HGETALL` array, as a map. An absent hash is an empty array.
fn reply_flat_hash(
    value: Option<fred::types::Value>,
    object_id: &str,
    what: &str,
) -> Result<HashMap<String, String>, ServiceError> {
    let flat = reply_string_list(value, object_id, what)?;
    if flat.len() % 2 != 0 {
        return Err(reply_shape_err(
            object_id,
            format!("{what} has an odd-length ({}) field array", flat.len()),
        ));
    }
    let mut map = HashMap::with_capacity(flat.len() / 2);
    let mut it = flat.into_iter();
    while let (Some(field), Some(value)) = (it.next(), it.next()) {
        map.insert(field, value);
    }
    Ok(map)
}

/// A per-entry array of `{id, flat_hash_or_list}` pairs.
fn reply_entries(
    value: Option<fred::types::Value>,
    object_id: &str,
    what: &str,
) -> Result<Vec<(String, fred::types::Value)>, ServiceError> {
    value
        .map(|v| v.into_array())
        .unwrap_or_default()
        .into_iter()
        .map(|entry| {
            let mut pair = entry.into_array().into_iter();
            match (
                pair.next().and_then(|id| id.into_string()),
                pair.next(),
                pair.next(),
            ) {
                (Some(id), Some(body), None) => Ok((id, body)),
                _ => Err(reply_shape_err(
                    object_id,
                    format!("a {what} entry is not an {{id, body}} pair"),
                )),
            }
        })
        .collect()
}

// ── Bounded fan-out: unfenced, and the fenced read's pass (SR-4.3, SR-4.4, SR-5.3, SR-11.3)
//
// The structural `UpdateLayer` slow path reads through this at every size,
// never through the atomic script, so it does not block its shard for a whole
// read. The structural read's plain commands (round 1, a lone long children
// list, windows after a script fallback) interleave with other traffic; each of
// the structural read's script windows (below) blocks its shard for at most
// τ (`config::FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS`), by the recorded model,
// with one residual, the same as the byte window's (SDD §5.4 *Residuals*):
// names longer than any the read has seen. A window is sized at the longest
// name seen, so the window that brings longer names runs past its model, as
// its replies run past their bound; the recorded worst was 397 sections named 4,096 B sized at 64 B, 1.37 × τ
// (`config::FANOUT_SCRIPT_WINDOW_FIXED_NS`'s record).
// It reads the same keys the atomic read reads, under the same marked/unmarked
// children rule (SR-1.2), so on a quiescent object the two return the same
// state and the same tallies.
// Nothing fences it, so its state is not proven to be the state at any one
// `seq`: the `cut` it returns is the header `seq` its first round read, and a
// caller that needs a proof re-reads the header afterwards (the structural
// path's one `HGET obj:hdr seq`, SR-4.3).
//
// The fenced read (`storage::snapshot_fence::read`, SR-5.3) runs this same
// fan-out, one pass per attempt, under a fence lease no commit can land
// through; its release script is the proof. Its pass
// (`read_object_snapshot_fenced_pass`) differs only in what it does with what
// it reads: a header without a numeric `seq` is always `NotFound`, indexed
// sections without a readable hash are recorded as a corrupt index, as on the
// atomic path, and it reports its progress — the reply bytes tallied so far,
// after round 1 and after each later window — to the fenced read's hook, which
// tops up its memory grant from it. The unfenced fan-out reports to no one,
// except that the structural update's charged read
// (`read_object_snapshot_slim_charged`) grows the call's upload charge after
// round 1's count windows, before each later window and index page is issued
// and after each, through its `FanoutAllowance`, and stops before the next
// window when a growth is refused.
//
// Every pass runs on a dedicated fan-out thread (`storage::fanout_thread`),
// never on a service worker — a fenced pass on `fanout-read`, an unfenced one
// on the least-loaded of the `watch-read` threads, where it stays — so its
// windows allocate from that thread's one allocator arena whatever the worker
// count.
// The hook and the allowance stay with the caller: the pass reaches them at
// the same seams, in the same order, as requests over its `PassLink` that the
// caller answers before the pass issues anything more (`ProgressSink`,
// `AllowanceSeam`).
//
// One code path, two reply shapes (`FanoutReplies`). Every reader but one
// reads whole rows and whole lists (`Whole`). The structural update's read —
// both its entry points, charged or not — reads with bounded replies
// (`Bounded`), so that every reply is bounded per command before it is issued
// and no inline value is transferred:
//
// - a field or time-sample row is one command of two wire commands, `HSTRLEN
//   row value` and `HMGET row` of its other columns, with a `HEXISTS row
//   value` follow-up only for a row with a value length of 0 and either a
//   `hash` or no known column, and an `EXISTS row` follow-up only for a row
//   with no known column and no `value` (`FanoutPage::ValueRow`; the rule is
//   `ValueRowReply::asks_value_exists` / `asks_row_exists`). The row's shape
//   follows `decode_field_value`'s rule, and its tally is rebuilt as
//   `HGETALL`'s would be (column names, the `HMGET` lengths and the `HSTRLEN`
//   length): equal for every row a writer of this service shapes; a column
//   no writer writes is not counted;
// - a children list is `LLEN` in round 2 and, only when it is not empty,
//   `LRANGE key 0 n-1` in round 3 (`FanoutPage::Length`, `FanoutPage::Prefix`):
//   the range caps the reply, and a list that grew in between is truncated
//   inside a read whose proof then fails;
// - on a marked object `obj:chs` is counted (`SCARD`) after the count window
//   and paged in round 1 by the set-index rule, instead of one `SMEMBERS`;
// - every window and index page is sized so that the sum of its commands'
//   per-command reply bounds (`FanoutWindows::reply_bound`) stays within
//   `FANOUT_WINDOW_BYTES`, and its wire commands within `FANOUT_WINDOW`; a
//   command whose bound alone exceeds the byte window runs alone. Names are
//   sized at the longest the read has seen, at least `NAME_BYTES_SIZING_FLOOR`,
//   so a reply carrying longer names can exceed its bound: the one reply
//   shape no plain command bounds before it arrives.
//
// How the bounded-reply read issues a window (`WindowTransport`). Round 1 is
// plain commands, each its own call. Each window of rounds 2 and 3 is one
// `EVALSHA` of `scripts/fanout_window_read.lua` (`issue_script_window`): the
// script issues, per entry, exactly the commands the plain window would
// issue, and returns their raw answers, which are decoded into the plain
// issuer's replies (`decode_fanout_window_reply`), so tallies, the compact
// decode and the `seq` proof are the same whichever transport read a window.
// One call per window replaces two or three per value row, and one window's
// commands run atomically, so a row's commands cannot be torn across another
// client's write: a value row read by a script window reports a value beside
// a hash as a row holding both shapes (`ValueRowReply::shape`), as the
// whole-row readers do. A script window blocks its shard while it runs, so
// besides the cuts above it is cut so its request and reply bytes together
// stay within `FANOUT_WINDOW_BYTES` and its modelled shard time within τ
// (`config::fanout_script_window_modelled_shard_ns`); a lone entry over τ — a
// long children list — is issued as one plain `LRANGE` instead. A read whose
// script meets a cluster-state refusal of an undeclared same-slot key (a slot
// migrating, not served, or the cluster down) re-issues that window, and every
// later one, as plain commands, whose redirects fred's router follows per key:
// an `-ASK` with a pipelined `ASKING` to the migration's target (the vendored
// fred's `live_edit::ask`, SDD §15.3), a `-MOVED` after a cluster sync
// (`live_edit_snapshot_read_window_fallback_total`); the next read tries the
// script again. A plain window reports no row holding both shapes: its two
// commands per row can tear.
//
// It reads in round trips with the collections bounded. Three rounds: the
// header and the four indices; every hash, plus the children lists of an
// unmarked object or the `obj:chs` index of a marked one; then a marked
// object's children lists. Each round is issued in windows, one at a time, and
// per read:
//
// - round 1 (`fanout_round_one`) is bounded in index members: its first window
//   is the header and the four indices' member counts (`SCARD` / `ZCARD`),
//   and every later window reads at most `FANOUT_INDEX_WINDOW_MEMBERS`
//   members of them, all indices together — `ZRANGE` pages by rank for the
//   sorted indices, `SSCAN` steps for the sets (asked for half their share,
//   `COUNT` being approximate; a set's repeats dropped). Read whole, round 1
//   would be one window whose cost grows with the layer, out of the command
//   cap's reach. On the bounded-reply read a page is also bounded in bytes:
//   the sum of its shares × each index's per-member bound stays within
//   `FANOUT_WINDOW_BYTES`;
// - rounds 2 and 3 hold at most `FANOUT_WINDOW` commands outstanding — one
//   window's;
// - the raw replies awaiting processing are one window's: every reply is
//   decoded into the snapshot (`FanoutEntries::take_in`) and dropped as soon
//   as its window completes, before the next window is issued. Round 1's
//   index members are the exception: they are kept, whole, because round 2's
//   commands are built from them;
// - on the whole-row read, a round-2 or round-3 window holds as many commands
//   as fit `FANOUT_WINDOW_BYTES` at the average reply bytes per entry command
//   observed so far in the read (per command of any family, header and
//   indices included, until an entry reply with any bytes in it arrives),
//   never fewer than one (`fanout_window_len`). So for it
//   `FANOUT_WINDOW_BYTES` is a target, not a hard maximum: a window's replies
//   come to about that figure when its entries are near that average, and
//   more when they run larger, up to `FANOUT_WINDOW` × the largest entry.
//   Once the observed average exceeds the byte window, windows shrink to one
//   command each (one reply cannot be split); but a window is sized before
//   its own replies arrive, so a reply larger than the byte window can still
//   share its window with up to `FANOUT_WINDOW − 1` other commands. The
//   bounded-reply read sizes by per-command bounds instead (above), so for it
//   the byte window is a bound, charged before issue, except for names
//   longer than any it has seen.
//
// The decoded state itself grows to the whole object, as every reader's does.
// Both command and byte windows come from the configured atomic-read limits
// (`config::snapshot_fanout_window` / `snapshot_fanout_window_bytes`), so
// lowering the ceiling narrows them. The command window is also capped at
// `config::FANOUT_WINDOW_MAX` (2,141 commands, from SR-11.6 (i)'s queue limit),
// so at the default ceiling it is the cap, not ⌊ceiling ÷ 5.266⌋, and only a
// ceiling below 11,275 items narrows it further. Round 1's member bound,
// `config::FANOUT_INDEX_WINDOW_MEMBERS` (26,855), comes from the same limit at
// the shipped ceiling and does not move with the configured one. Together they
// keep every window of every round under SR-11.6 (i)'s per-stream allowance.

/// Which fan-out read [`fanout_snapshot_read`] runs. On either, a header
/// that holds no numeric `seq` is not a live object: `ServiceError::NotFound`,
/// `read_object_header`'s rule (Spec SDD SR-1.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FanoutRead {
    /// The unfenced read (`path = Unfenced`).
    Unfenced,
    /// One pass of the fenced read (`path = Fenced`): a corrupt section index
    /// is recorded.
    Fenced,
}

impl FanoutRead {
    fn path(self) -> SnapshotReadPath {
        match self {
            Self::Unfenced => SnapshotReadPath::Unfenced,
            Self::Fenced => SnapshotReadPath::Fenced,
        }
    }

    /// The read's position for the test-only seam.
    fn fault_cursor(self) -> SnapshotReadFaultCursor {
        match self {
            Self::Unfenced => SnapshotReadFaultCursor::unfenced_fanout(),
            Self::Fenced => SnapshotReadFaultCursor::fenced_fanout(),
        }
    }

    /// Whether an indexed section without a readable hash (or with a
    /// non-decimal id) is recorded as a corrupt index rather than taken for a
    /// commit landing between rounds: only under the fence, where no commit
    /// can land ([`AtomicSnapshot::corrupt_section_index_members`]).
    fn records_corrupt_index(self) -> bool {
        matches!(self, Self::Fenced)
    }
}

/// The reply shape of a fan-out read (see the section comment above).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FanoutReplies {
    /// Whole rows and whole lists: `HGETALL` per hash, `LRANGE 0 -1` per
    /// children list, `SMEMBERS obj:chs` on a marked object; windows
    /// sized from the average reply observed. The fenced read's pass.
    Whole,
    /// Every reply bounded per command before it is issued: a value row
    /// by its two-command read (`FanoutPage::ValueRow`), a children list
    /// by `LLEN` then a ranged `LRANGE` (`FanoutPage::Length`,
    /// `FanoutPage::Prefix`), a marked object's `obj:chs` in round-1
    /// pages; every window and page sized from its commands'
    /// per-command reply bounds (`FanoutWindows::reply_bound`). Each
    /// window of rounds 2 and 3 is one window script
    /// (`WindowTransport::Script`), except a lone children list whose
    /// modelled shard time alone exceeds τ (one plain `LRANGE`) and
    /// every window after the read falls back to plain commands. The
    /// unfenced fan-out, which is the structural update's read only
    /// (`SlimSnapshotReadConsumer::StructuralUpdate`), through both its
    /// entry points; its state is the compact form.
    Bounded,
}

/// One command of the bounded fan-out (unfenced, and the fenced read's pass).
/// The key family decides the command ([`fanout_op`]): `HGETALL` for the
/// header and every entry hash, `SMEMBERS` for the set indices, `ZRANGE 0 -1`
/// for the sorted-set indices and `LRANGE 0 -1` for a children list — read
/// whole ([`FanoutPage::Whole`]). Round 1 reads the four indices in parts
/// instead ([`FanoutPage`]): their counts, then their members in pages; and
/// the bounded-reply read ([`FanoutReplies::Bounded`]) reads value rows and
/// children lists in parts too.
struct FanoutCommand {
    family: SnapshotKeyFamily,
    key: String,
    /// What the command reads: a section or tombstone id, a field or
    /// time-sample index member, or a children list's parent id. Empty for
    /// the header and the indices.
    id: String,
    /// Which part of the key it reads.
    page: FanoutPage,
    /// The wire command that failed, for a command of several wire commands
    /// ([`FanoutPage::ValueRow`]): set by [`value_row_read`] before it
    /// returns that command's error, or from a script window's
    /// `FANOUT_WINDOW_REFUSED` tag ([`classify_script_window_err`]), so the
    /// failure is logged and classified under the command that met it
    /// ([`Self::op_label`]). Unset otherwise.
    failed_op: std::sync::OnceLock<&'static str>,
}

/// Which part of its key a fan-out command reads. Every command of the
/// whole-row read reads its key whole except round 1's index reads (Spec SDD
/// SR-11.3), which are bounded in members by
/// `config::FANOUT_INDEX_WINDOW_MEMBERS`. The bounded-reply read
/// ([`FanoutReplies::Bounded`]) also reads value rows and children lists in
/// parts, so every reply it receives is bounded before it is issued.
#[derive(Debug, Clone, PartialEq, Eq)]
enum FanoutPage {
    /// The whole key: `HGETALL`, `SMEMBERS`, `ZRANGE 0 -1` or `LRANGE 0 -1`.
    Whole,
    /// An index's member count: `SCARD` for a set, `ZCARD` for a sorted set.
    Count,
    /// Ranks `start..=stop` of a sorted-set index: `ZRANGE key start stop`.
    Ranks { start: u64, stop: u64 },
    /// One step of a set index's scan: `SSCAN key cursor COUNT count`.
    Scan { cursor: String, count: u64 },
    /// A field or time-sample row's two-command read, which transfers no
    /// inline value: `HSTRLEN row value` and `HMGET row` of the family's other
    /// columns ([`value_row_columns`]) together, then `HEXISTS row value` only
    /// for a row with a value length of 0 and either a `hash` or no known
    /// column, and `EXISTS row` only for a row with no known column and no
    /// `value` ([`ValueRowReply`]). In a script window the same commands run
    /// inside the window's script, in that order.
    ValueRow,
    /// A children list's length: `LLEN`, in round 2.
    Length,
    /// The first `len` entries of a children list, `len` > 0: `LRANGE key 0
    /// len-1`, in round 3, `len` its round-2 `LLEN`.
    Prefix { len: u64 },
}

impl FanoutCommand {
    fn new(family: SnapshotKeyFamily, key: String, id: &str) -> Self {
        Self::part(family, key, id, FanoutPage::Whole)
    }

    /// A round-1 command reading part of an index (`family` one of the
    /// round-1 indices).
    fn index_part(family: SnapshotKeyFamily, key: String, page: FanoutPage) -> Self {
        Self::part(family, key, "", page)
    }

    /// A command reading `page` of `key`.
    fn part(family: SnapshotKeyFamily, key: String, id: &str, page: FanoutPage) -> Self {
        Self {
            family,
            key,
            id: id.to_owned(),
            page,
            failed_op: std::sync::OnceLock::new(),
        }
    }

    /// The wire commands this command issues at once, which count toward
    /// `FANOUT_WINDOW`: two for a value row's read, one for every other.
    fn wire_commands(&self) -> u64 {
        match self.page {
            FanoutPage::ValueRow => 2,
            _ => 1,
        }
    }

    /// The `op` a failure of this command is logged and classified under —
    /// one fixed label per command shape, the same on both fan-outs, so a
    /// refusal reads the same whichever fan-out met it. A value row's read is
    /// labelled by the wire command that failed, `HSTRLEN` when none is
    /// recorded (a canned error of the test-only seam answers in place of
    /// the whole command).
    fn op_label(&self) -> String {
        let sorted = fanout_op(self.family) == FanoutOp::Zrange;
        match &self.page {
            FanoutPage::Whole | FanoutPage::Ranks { .. } | FanoutPage::Prefix { .. } => {
                self.whole_op_label()
            }
            FanoutPage::Count => format!(
                "{} {}",
                if sorted { "ZCARD" } else { "SCARD" },
                index_label(self.family)
            ),
            FanoutPage::Scan { .. } => format!("SSCAN {}", index_label(self.family)),
            FanoutPage::ValueRow => format!(
                "{} {}",
                self.failed_op.get().copied().unwrap_or("HSTRLEN"),
                value_row_label(self.family)
            ),
            FanoutPage::Length => format!("LLEN children list (parent {})", self.id),
        }
    }

    /// [`Self::op_label`] for a command reading its whole key (and a
    /// `ZRANGE` page, which is the same command over fewer ranks).
    fn whole_op_label(&self) -> String {
        match self.family {
            SnapshotKeyFamily::Header => "HGETALL obj header".to_owned(),
            SnapshotKeyFamily::SectionsIndex => "SMEMBERS sections index".to_owned(),
            SnapshotKeyFamily::DeletedSectionsIndex => "SMEMBERS deleted sections index".to_owned(),
            SnapshotKeyFamily::Section => "HGETALL section".to_owned(),
            SnapshotKeyFamily::DeletedSection => "HGETALL deleted section".to_owned(),
            SnapshotKeyFamily::Children => format!("LRANGE children list (parent {})", self.id),
            SnapshotKeyFamily::ChildrenIndex => "SMEMBERS children index".to_owned(),
            SnapshotKeyFamily::FieldsIndex => "ZRANGE fields index".to_owned(),
            SnapshotKeyFamily::TimeSamplesIndex => "ZRANGE time samples index".to_owned(),
            SnapshotKeyFamily::Field => "HGETALL field".to_owned(),
            SnapshotKeyFamily::TimeSample => "HGETALL time sample".to_owned(),
            // No fan-out command reads it (external-value clean-up's own reads
            // carry their own `op`).
            SnapshotKeyFamily::ExtIndex => "HGETALL ext index".to_owned(),
        }
    }

    /// The command as an entry of a script window
    /// (`scripts/fanout_window_read.lua`): a field or time-sample value row,
    /// a section or tombstone hash, a children list's length or its ranged
    /// read. `None` for every other command — the header and the index parts
    /// of round 1, and every whole-row command — which no script window
    /// carries.
    fn script_op(&self) -> Option<ScriptOp> {
        match (self.family, &self.page) {
            (SnapshotKeyFamily::Field, FanoutPage::ValueRow) => Some(ScriptOp::Field),
            (SnapshotKeyFamily::TimeSample, FanoutPage::ValueRow) => Some(ScriptOp::TimeSample),
            (SnapshotKeyFamily::Section, FanoutPage::Whole) => Some(ScriptOp::Section),
            (SnapshotKeyFamily::DeletedSection, FanoutPage::Whole) => Some(ScriptOp::Tombstone),
            (SnapshotKeyFamily::Children, FanoutPage::Length) => Some(ScriptOp::Length),
            (SnapshotKeyFamily::Children, FanoutPage::Prefix { len }) => {
                Some(ScriptOp::Prefix { len: *len })
            }
            _ => None,
        }
    }

    /// The bytes this command adds to a script window's request: its op (one
    /// byte), its id and, for a ranged list, its length's decimal digits. 0
    /// for a command no script window carries. The window adds its column
    /// lists once ([`SCRIPT_WINDOW_COLUMN_ARG_BYTES`]).
    fn script_request_bytes(&self) -> u64 {
        match self.script_op() {
            None => 0,
            Some(op) => {
                let digits = match op {
                    ScriptOp::Prefix { len } => u64::from(len.checked_ilog10().unwrap_or(0)) + 1,
                    _ => 0,
                };
                (1 + self.id.len() as u64).saturating_add(digits)
            }
        }
    }
}

/// One entry of a script window (`scripts/fanout_window_read.lua`, `ARGV[5..]`):
/// what it reads, and so its op letter and the in-script commands that may
/// refuse it ([`FanoutCommand::script_op`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScriptOp {
    /// `f`: a field row, by its value-row commands.
    Field,
    /// `t`: a time-sample row, by its value-row commands.
    TimeSample,
    /// `s`: a section hash, `HGETALL`.
    Section,
    /// `d`: a tombstone hash, `HGETALL`.
    Tombstone,
    /// `l`: a children list's length, `LLEN`.
    Length,
    /// `r`: a children list's first `len` entries, `LRANGE key 0 len-1`.
    Prefix { len: u64 },
}

impl ScriptOp {
    /// The entry's op letter.
    fn letter(self) -> &'static str {
        match self {
            Self::Field => "f",
            Self::TimeSample => "t",
            Self::Section => "s",
            Self::Tombstone => "d",
            Self::Length => "l",
            Self::Prefix { .. } => "r",
        }
    }

    /// The in-script commands the script issues for this entry: the only
    /// commands a `FANOUT_WINDOW_REFUSED` tag may name for it.
    fn commands(self) -> &'static [&'static str] {
        match self {
            Self::Field | Self::TimeSample => &["HSTRLEN", "HMGET", "HEXISTS", "EXISTS"],
            Self::Section | Self::Tombstone => &["HGETALL"],
            Self::Length => &["LLEN"],
            Self::Prefix { .. } => &["LRANGE"],
        }
    }
}

/// How the bounded-reply read ([`FanoutReplies::Bounded`]) issues one window
/// of its rounds 2 and 3 (the section comment above; SDD §10.5). Round 1 is
/// always plain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowTransport {
    /// One `EVALSHA` of the window script
    /// (`storage::lua::FANOUT_WINDOW_READ_SCRIPT`) per window: the
    /// production rule.
    Script,
    /// Each command its own call (each value row two, plus its follow-ups):
    /// what a read falls back to on a cluster-state refusal, and a lone
    /// children list too long for a script window.
    Plain,
}

/// The format tag a window script's reply carries
/// (`scripts/fanout_window_read.lua`'s `FORMAT`).
const FANOUT_WINDOW_READ_FORMAT: &str = "fanout-window-read-v1";

/// The tag of a window script's per-entry refusal:
/// `FANOUT_WINDOW_REFUSED: <entry> <command> <raw error>`.
const FANOUT_WINDOW_REFUSED_TAG: &str = "FANOUT_WINDOW_REFUSED: ";

/// The `op` a window script's script-level failure is logged and classified
/// under, attributed to the object's header: the atomic read's rule for a
/// failure that says nothing about any one entry's key.
const FANOUT_WINDOW_READ_OP: &str = "EVALSHA fanout_window_read";

/// The bytes of a value row's column list as the window script takes it
/// (`ARGV[3]` / `ARGV[4]`): the columns joined by single spaces.
const fn column_arg_bytes(columns: &[&str]) -> u64 {
    let mut bytes = 0;
    let mut i = 0;
    while i < columns.len() {
        bytes += columns[i].len();
        i += 1;
    }
    (bytes + columns.len().saturating_sub(1)) as u64
}

/// A value row's column list as the window script takes it: the family's
/// [`value_row_columns`] joined by single spaces, so the script and the plain
/// issuer read one list.
fn column_arg(family: SnapshotKeyFamily) -> String {
    value_row_columns(family).join(" ")
}

/// The bytes a script window's request carries beside its entries: its two
/// column lists ([`column_arg`]).
const SCRIPT_WINDOW_COLUMN_ARG_BYTES: u64 =
    column_arg_bytes(value_row_columns(SnapshotKeyFamily::Field))
        + column_arg_bytes(value_row_columns(SnapshotKeyFamily::TimeSample));

/// Whether two strings are equal, in a `const` context.
const fn const_str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

// The window script reads a row's `hash` as its first column (its follow-up
// rule asks for it by position), as `ValueRowReply::hash` does.
const _: () = {
    assert!(const_str_eq(
        value_row_columns(SnapshotKeyFamily::Field)[0],
        "hash"
    ));
    assert!(const_str_eq(
        value_row_columns(SnapshotKeyFamily::TimeSample)[0],
        "hash"
    ));
};

/// The name an index-part command's `op` label gives its index.
fn index_label(family: SnapshotKeyFamily) -> &'static str {
    match family {
        SnapshotKeyFamily::SectionsIndex => "sections index",
        SnapshotKeyFamily::DeletedSectionsIndex => "deleted sections index",
        SnapshotKeyFamily::ChildrenIndex => "children index",
        SnapshotKeyFamily::FieldsIndex => "fields index",
        SnapshotKeyFamily::TimeSamplesIndex => "time samples index",
        // Not an index: never given an index part.
        SnapshotKeyFamily::Header
        | SnapshotKeyFamily::Section
        | SnapshotKeyFamily::DeletedSection
        | SnapshotKeyFamily::Children
        | SnapshotKeyFamily::Field
        | SnapshotKeyFamily::TimeSample
        | SnapshotKeyFamily::ExtIndex => "key",
    }
}

/// The name a value row's `op` label gives its row: `HSTRLEN field`,
/// `HMGET time sample`.
fn value_row_label(family: SnapshotKeyFamily) -> &'static str {
    match family {
        SnapshotKeyFamily::Field => "field",
        SnapshotKeyFamily::TimeSample => "time sample",
        // Not a value row: never given a value row's read.
        SnapshotKeyFamily::Header
        | SnapshotKeyFamily::SectionsIndex
        | SnapshotKeyFamily::DeletedSectionsIndex
        | SnapshotKeyFamily::Section
        | SnapshotKeyFamily::DeletedSection
        | SnapshotKeyFamily::Children
        | SnapshotKeyFamily::ChildrenIndex
        | SnapshotKeyFamily::FieldsIndex
        | SnapshotKeyFamily::TimeSamplesIndex
        | SnapshotKeyFamily::ExtIndex => "row",
    }
}

/// The columns a value row's two-command read asks `HMGET` for
/// ([`FanoutPage::ValueRow`]): every column a writer of this service stores
/// on the row but `value`, whose length `HSTRLEN` reads instead — the
/// external shape's `hash` and `size` (`storage::layer::encode_value_fields`)
/// and the family's own columns (`storage::layer::field_value_hash_fields`,
/// `time_sample_hash_fields`). `hash` comes first: the row's shape reads it,
/// and the window script's follow-up rule asks for it by position. The
/// window script takes the same lists ([`column_arg`]).
const fn value_row_columns(family: SnapshotKeyFamily) -> &'static [&'static str] {
    match family {
        SnapshotKeyFamily::TimeSample => &["hash", "size", "version"],
        _ => &[
            "hash",
            "size",
            "set_order",
            "authoring_set_order",
            "version",
        ],
    }
}

/// The Valkey command a fan-out read of `family` issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FanoutOp {
    Hgetall,
    Smembers,
    Zrange,
    Lrange,
}

fn fanout_op(family: SnapshotKeyFamily) -> FanoutOp {
    match family {
        SnapshotKeyFamily::Header
        | SnapshotKeyFamily::Section
        | SnapshotKeyFamily::DeletedSection
        | SnapshotKeyFamily::Field
        | SnapshotKeyFamily::TimeSample
        // Never read by the fan-out; a hash, like the rows above.
        | SnapshotKeyFamily::ExtIndex => FanoutOp::Hgetall,
        SnapshotKeyFamily::SectionsIndex
        | SnapshotKeyFamily::DeletedSectionsIndex
        | SnapshotKeyFamily::ChildrenIndex => FanoutOp::Smembers,
        SnapshotKeyFamily::FieldsIndex | SnapshotKeyFamily::TimeSamplesIndex => FanoutOp::Zrange,
        SnapshotKeyFamily::Children => FanoutOp::Lrange,
    }
}

/// One fan-out reply: a hash, the members of a set, sorted set or list (or of
/// one `ZRANGE` page or ranged `LRANGE`), a count (an index's members, a
/// list's length), one `SSCAN` step, or a value row's two-command read.
enum FanoutReply {
    Hash(HashMap<String, String>),
    List(Vec<String>),
    Count(u64),
    /// One `SSCAN` step: the members it returned (a set may repeat a member
    /// across steps) and the cursor to continue from, `"0"` when the scan is
    /// complete.
    Scan {
        cursor: String,
        members: Vec<String>,
    },
    /// A value row's two-command read ([`FanoutPage::ValueRow`]).
    ValueRow(ValueRowReply),
}

impl FanoutReply {
    /// Items, counted as the atomic read counts them: hash fields, or
    /// set / sorted-set members / list elements; for a value row's
    /// two-command read, the whole row's, rebuilt
    /// ([`ValueRowReply::tally`]). A count holds none.
    fn items(&self) -> u64 {
        match self {
            Self::Hash(h) => h.len() as u64,
            Self::List(l) | Self::Scan { members: l, .. } => l.len() as u64,
            Self::Count(_) => 0,
            Self::ValueRow(row) => row.tally().0,
        }
    }

    /// Bytes received: the length of every string the reply carried, hash
    /// field names included. A count holds none, and a value row's
    /// two-command read only its `HMGET` columns' values. What a window's
    /// replies are sized and charged on.
    fn bytes(&self) -> u64 {
        match self {
            Self::Hash(h) => h.iter().map(|(k, v)| (k.len() + v.len()) as u64).sum(),
            Self::List(l) | Self::Scan { members: l, .. } => l.iter().map(|s| s.len() as u64).sum(),
            Self::Count(_) => 0,
            Self::ValueRow(row) => row.received_bytes(),
        }
    }

    /// Bytes counted as the atomic read counts them: [`Self::bytes`] for
    /// every reply but a value row's two-command read, whose whole row's
    /// bytes are rebuilt ([`ValueRowReply::tally`]). What the read's tally
    /// keeps.
    fn tally_bytes(&self) -> u64 {
        match self {
            Self::ValueRow(row) => row.tally().1,
            _ => self.bytes(),
        }
    }
}

/// What a value row's two-command read learned of the row
/// ([`FanoutPage::ValueRow`]): its `value` column's length, every other
/// column a writer of this service stores on it, and the follow-ups' answers.
/// The inline value itself is never transferred.
struct ValueRowReply {
    family: SnapshotKeyFamily,
    /// `HSTRLEN row value`: 0 for an absent column or an empty value.
    value_len: u64,
    /// `HMGET row` of [`value_row_columns`], in that order.
    columns: Vec<Option<String>>,
    /// `HEXISTS row value`, asked only for a row with a value length of 0
    /// and either a `hash` or no known column: whether the row holds an empty
    /// `value` — beside its hash, or alone (a row holding only an empty
    /// `value`, which no writer of this service writes, but whose tally
    /// `HEXISTS` keeps equal to `HGETALL`'s).
    value_exists: Option<bool>,
    /// `EXISTS row`, asked only for a row with no known column and no `value`
    /// (`HEXISTS` answered false): whether the row holds only columns no
    /// writer of this service writes, rather than being absent.
    row_exists: Option<bool>,
    /// Whether the row's commands ran atomically: true when it was read by a
    /// script window (`decode_fanout_window_reply`), false when by the plain
    /// value-row read ([`value_row_read`]), whose commands can tear. Decides
    /// whether a value beside a `hash` is reported ([`Self::shape`]).
    atomic: bool,
}

impl ValueRowReply {
    /// Whether the read asks `HEXISTS row value` of this row, from its
    /// `HSTRLEN` and `HMGET` answers: only a row with a value length of 0
    /// and either a `hash` or no known column may still hold an empty
    /// `value` (a row with another known column but no `hash` is inline by
    /// the writers' rule and needs nothing). The one rule both the plain
    /// value-row read and the window script follow; the script's twin is in
    /// `scripts/fanout_window_read.lua`, and its decode rejects a reply that
    /// differs from this.
    fn asks_value_exists(&self) -> bool {
        self.value_len == 0 && (self.hash().is_some() || !self.any_column())
    }

    /// Whether the read asks `EXISTS row` of this row, once `HEXISTS` is in:
    /// only a row with no known column and no `value`, which is absent or
    /// holds only columns no writer writes. The same single source as
    /// [`Self::asks_value_exists`].
    fn asks_row_exists(&self) -> bool {
        self.value_exists == Some(false) && !self.any_column()
    }

    /// The row's `hash` column.
    fn hash(&self) -> Option<&str> {
        self.columns.first().and_then(Option::as_deref)
    }

    /// Whether `HMGET` found any of its columns.
    fn any_column(&self) -> bool {
        self.columns.iter().any(Option::is_some)
    }

    /// Whether the row holds a `value` column: a non-empty one by its length;
    /// an empty one beside a `hash`, or alone, by the `HEXISTS` follow-up;
    /// and, on a row with no `hash` but another known column, by the writers'
    /// rule — every writer of this service writes a value in one of its two
    /// shapes (`storage::layer::encode_value_fields`), so such a row is
    /// inline, and its value may be empty.
    fn has_value(&self) -> bool {
        self.value_len > 0
            || self.value_exists == Some(true)
            || (self.hash().is_none() && self.any_column())
    }

    /// The row's shape, by [`decode_field_value`]'s rule (a `value` column
    /// first): inline when it holds a value — empty or not, a `hash` beside
    /// it or not; external when it holds a `hash` alone; inline when it holds
    /// only columns no writer writes (`EXISTS`); and absent — an indexed row
    /// whose hash is gone, skipped as an empty `HGETALL` is — otherwise.
    ///
    /// A value beside a `hash` is decoded inline, as the whole-row decode
    /// decodes it. Read by a script window ([`Self::atomic`]) it is reported
    /// as a row holding both shapes ([`SlimRow::InlineBesideHash`], b.zvd,
    /// [`ValueShapeConflictTally`]), as every whole-row reader (the cold
    /// start and audit, catch-up, checkpoint) reports it: the row's
    /// commands ran atomically, so both shapes were on the row at once.
    /// Read by the plain value-row read it is never reported: those commands
    /// are not atomic — they can run on different connections with other
    /// clients' commands between them — so an inline→external rewrite
    /// landing between them reads as both shapes on a row that never held
    /// both. Such a torn read is rejected whole by the structural update's
    /// `seq` proof, but the conflict WARN would be emitted when the state is
    /// assembled, before that proof, on exactly the contended reads this
    /// path serves.
    fn shape(&self) -> SlimRow {
        match (self.has_value(), self.hash()) {
            (true, Some(_)) if self.atomic => SlimRow::InlineBesideHash,
            (true, _) => SlimRow::Inline,
            (false, Some(hash)) => SlimRow::External(hash.to_owned()),
            (false, None) if self.row_exists == Some(true) => SlimRow::Inline,
            (false, None) => SlimRow::Absent,
        }
    }

    /// The row's `(items, bytes)` as a whole-row `HGETALL` would tally them:
    /// each column found (its name and value), and the `value` column (its
    /// name and [`Self::value_len`]) when the row holds one, an empty one
    /// included. Equal to `HGETALL`'s for every row a writer of this service
    /// shapes, and also for a row holding only an empty `value`, which no
    /// writer writes but whose `value` the `HEXISTS` follow-up finds; a
    /// column no writer writes is not counted, by design.
    fn tally(&self) -> (u64, u64) {
        let mut items = 0u64;
        let mut bytes = 0u64;
        for (name, value) in value_row_columns(self.family).iter().zip(&self.columns) {
            if let Some(value) = value {
                items += 1;
                bytes += (name.len() + value.len()) as u64;
            }
        }
        if self.has_value() {
            items += 1;
            bytes += "value".len() as u64 + self.value_len;
        }
        (items, bytes)
    }

    /// The string bytes the read received: its `HMGET` columns' values (the
    /// length, `HEXISTS` and `EXISTS` replies are integers).
    fn received_bytes(&self) -> u64 {
        self.columns.iter().flatten().map(|v| v.len() as u64).sum()
    }

    /// The value-shape columns received (`hash`, `size`): the bounded read's
    /// [`AtomicSnapshotOf::value_row_bytes_received`] term. Test-support
    /// only.
    #[cfg(any(test, feature = "test-support"))]
    fn value_shape_bytes(&self) -> u64 {
        self.columns
            .iter()
            .take(2)
            .flatten()
            .map(|v| v.len() as u64)
            .sum()
    }
}

/// How many commands the next fan-out window issues (Spec SDD SR-11.3).
///
/// At most `max_commands` (`FANOUT_WINDOW`) and at most as many as fit
/// `max_bytes` (`FANOUT_WINDOW_BYTES`) at the average reply bytes per command
/// observed so far (`bytes_seen` over `commands_seen`, on the basis
/// [`FanoutWindows::run`] picks: entry replies once one with any bytes in it
/// has arrived, every reply before that), never more than
/// `remaining` and never fewer than one, because one reply cannot be split:
/// once the average exceeds the byte window, the window is one command. The
/// size comes from replies already seen, before this window's own arrive, so a
/// reply larger than the byte window can land in a window of up to
/// `max_commands` commands sized from a smaller average. Before any reply
/// has been observed, or while every reply on that basis was empty, the byte
/// window has nothing to scale from and the command window alone applies.
fn fanout_window_len(
    max_commands: u64,
    max_bytes: u64,
    commands_seen: u64,
    bytes_seen: u64,
    remaining: usize,
) -> usize {
    let by_bytes = if commands_seen == 0 || bytes_seen == 0 {
        u128::from(max_commands)
    } else {
        // floor(max_bytes ÷ (bytes_seen ÷ commands_seen)), in u128 so the
        // product cannot overflow.
        u128::from(max_bytes) * u128::from(commands_seen) / u128::from(bytes_seen)
    };
    let len = u128::from(max_commands).min(by_bytes).max(1);
    usize::try_from(len).unwrap_or(usize::MAX).min(remaining)
}

/// The items a WATCH attempt's read is projected to tally per entry, before
/// it has tallied any: the projection a stop at the count window carries, so
/// the attempt's one wait normally covers the whole read.
const PROJECTED_ITEMS_PER_ENTRY: u64 = 5;

/// The shortest name the bounded-reply read ([`FanoutReplies::Bounded`])
/// sizes a name-bearing reply at — a field-index member, a section row —
/// before it has seen a longer name: the sizing name is the longest the
/// read has seen, at least this. Names up to it are charged before they
/// arrive; a page or window of longer names than any seen yet is charged
/// after it arrives (the residual, SDD §5.4). 64 B covers the names of the
/// product range's layers [model]; raising it shrinks the residual and the
/// field-index pages with it (at 128 B a page holds about 2,182 members
/// instead of 4,057, about 80 ms more on a Kitchen-sized layer on a managed
/// store [model]). A constant, not an env var (SDD §16).
pub(crate) const NAME_BYTES_SIZING_FLOOR: u64 = 64;

/// The most decimal digits a `u64` column or id holds: 20 (`u64::MAX`). Every
/// id, version, order and size a writer of this service stores is a `u64`
/// in decimal.
const U64_DECIMAL_BYTES: u64 = u64::MAX.ilog10() as u64 + 1;

/// An external value's `hash` column: SHA-256 as hex, the 64 characters
/// [`hash_from_hex`] accepts and every writer stores
/// (`storage::layer::encode_value_fields`).
const EXTERNAL_HASH_HEX_BYTES: u64 = 2 * 32;

/// The reply bound of a field row's two-command read: what `HMGET` returns —
/// `hash` (64) and `size`, `set_order`, `authoring_set_order` and `version`
/// (20 each) — whatever the row's value and whichever build wrote it; the
/// length, `HEXISTS` and `EXISTS` replies are integers. 144 B.
const FIELD_ROW_REPLY_BOUND: u64 = EXTERNAL_HASH_HEX_BYTES + 4 * U64_DECIMAL_BYTES;

/// The reply bound of a time-sample row's two-command read: `hash`, `size`
/// and `version`. 104 B.
const TIME_SAMPLE_ROW_REPLY_BOUND: u64 = EXTERNAL_HASH_HEX_BYTES + 2 * U64_DECIMAL_BYTES;

/// The reply bound of a tombstone row's `HGETALL`: its two columns, names
/// and `u64` values (`storage::layer::execute_mutation`'s `CreateTombstone`
/// arm). 57 B.
const TOMBSTONE_ROW_REPLY_BOUND: u64 =
    ("first_ver".len() + "last_ver".len()) as u64 + 2 * U64_DECIMAL_BYTES;

/// The reply bound of a section row's `HGETALL` less its name's bytes: the
/// six column names (`storage::layer::section_hash_fields`), `parent_id`,
/// `first_ver` and `last_ver` (`u64`), `type` (`u8`) and `order` (`u32`).
/// 112 B; the row is that plus its name ([`section_row_reply_bound`]).
const SECTION_ROW_REPLY_BOUND_WITHOUT_NAME: u64 = ("parent_id".len()
    + "name".len()
    + "type".len()
    + "order".len()
    + "first_ver".len()
    + "last_ver".len()) as u64
    + 3 * U64_DECIMAL_BYTES
    + (u8::MAX.ilog10() as u64 + 1)
    + (u32::MAX.ilog10() as u64 + 1);

/// The reply bound of one decimal id: a sections, deleted-sections or
/// `obj:chs` index member, or one children-list entry. 20 B.
const ID_REPLY_BOUND: u64 = U64_DECIMAL_BYTES;

/// The reply bound of one time-samples index member,
/// `{section_id}:{hex_f64(time)}` (`KeySchema::time_sample_member`): 20 + 1 +
/// 16 = 37 B.
const TIME_SAMPLE_MEMBER_REPLY_BOUND: u64 = U64_DECIMAL_BYTES + 1 + 16;

/// The reply bound of a section row whose name is `name_bytes` long.
const fn section_row_reply_bound(name_bytes: u64) -> u64 {
    SECTION_ROW_REPLY_BOUND_WITHOUT_NAME.saturating_add(name_bytes)
}

/// The reply bound of one fields index member whose name is `name_bytes`
/// long, `{section_id}:{hex(key_name)}` (`KeySchema::field_member`): 20 + 1 +
/// 2 × `name_bytes`.
const fn field_member_reply_bound(name_bytes: u64) -> u64 {
    (U64_DECIMAL_BYTES + 1).saturating_add(name_bytes.saturating_mul(2))
}

/// The bounded-reply read's projected reply bound ([`FanoutReplies::Bounded`]),
/// from its count windows' counts — `[sections, tombstones, fields, time
/// samples, obj:chs]`, the last 0 on an object whose `obj:chs` the read does
/// not page — with every name at [`NAME_BYTES_SIZING_FLOOR`]: the
/// per-command reply bounds the read charges before issue
/// ([`FanoutWindows::reply_bound`], [`FanoutWindows::index_member_bound`]),
/// summed over every reply the rest of the read receives:
///
/// - round 1's pages: each set index (sections, deleted sections, `obj:chs`)
///   at twice its count × [`ID_REPLY_BOUND`] — a set's step is charged for its
///   share, and its share is at most twice its count (its `SSCAN` asks for
///   half) — and each sorted index at its count × its member bound
///   ([`field_member_reply_bound`] at the floor, 149 B;
///   [`TIME_SAMPLE_MEMBER_REPLY_BOUND`]);
/// - round 2's rows: a section [`section_row_reply_bound`] at the floor
///   (176 B), a tombstone [`TOMBSTONE_ROW_REPLY_BOUND`], a field
///   [`FIELD_ROW_REPLY_BOUND`] and a time sample
///   [`TIME_SAMPLE_ROW_REPLY_BOUND`]; a children list's `LLEN` is an integer,
///   bounded at nothing;
/// - round 3's children lists: [`ID_REPLY_BOUND`] per entry, for the larger
///   of sections + 2 and the `obj:chs` count. At a consistent cut every live
///   section is an entry of exactly one list (its parent's), so the lists
///   hold about the live sections between them; no count window counts them,
///   and this is the projection.
///
/// So at the floor no window's or page's pre-issue bound exceeds it. Two
/// things can carry a read past it:
///
/// - **lists longer than projected**: each round-3 `LRANGE`'s bound is its
///   list's length from round 2's `LLEN` × [`ID_REPLY_BOUND`], so its window
///   is charged in advance, exactly, before it is issued
///   ([`FanoutWindows::allowance_grow`]);
/// - **names longer than any the read has seen**: every pre-issue bound sizes
///   names at the longest seen so far, so the page or window that brings the
///   first longer names exceeds its bound, and the excess is charged only
///   after it arrives, by the growth after that page or window — the residual
///   (SDD §5.4, [`FanoutWindows::allowance_grow`]).
///
/// WA1 charges the smaller of it plus [`projected_read_request_bound`] and
/// `FANOUT_WINDOW_BYTES` as its byte window ([`FanoutAllowance`]). With a
/// whole window at WA1, a small object had that
/// residual held in advance; with the projection it does not, so up to
/// `W_BYTE` × (`FANOUT_WINDOW_BYTES` − the projection) more of a long-name
/// residual is held after its replies arrive rather than before.
pub(crate) fn projected_read_reply_bound(counts: [u64; 5]) -> u64 {
    let [sections, tombstones, fields, time_samples, children_index] = counts;
    let floor = NAME_BYTES_SIZING_FLOOR;
    let set_members = sections
        .saturating_add(tombstones)
        .saturating_add(children_index);
    let round_one = [
        (set_members, 2 * ID_REPLY_BOUND),
        (fields, field_member_reply_bound(floor)),
        (time_samples, TIME_SAMPLE_MEMBER_REPLY_BOUND),
    ];
    let round_two = [
        (sections, section_row_reply_bound(floor)),
        (tombstones, TOMBSTONE_ROW_REPLY_BOUND),
        (fields, FIELD_ROW_REPLY_BOUND),
        (time_samples, TIME_SAMPLE_ROW_REPLY_BOUND),
    ];
    let list_entries = sections.saturating_add(2).max(children_index);
    let round_three = [(list_entries, ID_REPLY_BOUND)];
    round_one
        .into_iter()
        .chain(round_two)
        .chain(round_three)
        .map(|(count, bound)| count.saturating_mul(bound))
        .fold(0, u64::saturating_add)
}

/// The bounded-reply read's projected script-request bound, from its count
/// windows' counts (`[sections, tombstones, fields, time samples, obj:chs]`,
/// as [`projected_read_reply_bound`] takes them), with every id at the name
/// floor: the bytes every script window of the read sends, summed over the
/// read ([`FanoutCommand::script_request_bytes`]) — per entry its op byte and
/// its id: a section or tombstone 1 + 20 B, a field 1 + 149 B
/// ([`field_member_reply_bound`] at the floor), a time sample 1 + 37 B; per
/// children list (the larger of sections + 2 and the `obj:chs` count, as the
/// reply projection counts them) its `LLEN` entry, 1 + 20 B, and its ranged
/// read, 1 + 20 B and its length's at most 20 digits — plus one window's
/// column lists ([`SCRIPT_WINDOW_COLUMN_ARG_BYTES`]).
///
/// WA1 adds it to the reply projection before taking the smaller of the two
/// and `FANOUT_WINDOW_BYTES` as its byte window ([`FanoutAllowance::counted`]),
/// so the one wait a refused growth's stop projects normally covers every
/// window's request and reply bytes at the floor. A field whose name is
/// longer than the floor sends more: each script window's request bytes are
/// exact, and charged before it is issued.
pub(crate) fn projected_read_request_bound(counts: [u64; 5]) -> u64 {
    let [sections, tombstones, fields, time_samples, children_index] = counts;
    let id_entry = 1 + ID_REPLY_BOUND;
    let lists = sections.saturating_add(2).max(children_index);
    [
        (sections, id_entry),
        (tombstones, id_entry),
        (
            fields,
            1 + field_member_reply_bound(NAME_BYTES_SIZING_FLOOR),
        ),
        (time_samples, 1 + TIME_SAMPLE_MEMBER_REPLY_BOUND),
        (lists, id_entry + id_entry + U64_DECIMAL_BYTES),
    ]
    .into_iter()
    .map(|(count, bytes)| count.saturating_mul(bytes))
    .fold(SCRIPT_WINDOW_COLUMN_ARG_BYTES, u64::saturating_add)
}

// The [code] figures SDD §5.4 and the source docs quote, pinned: the bounds
// above, and the name-bearing replies at `MAX_NAME_BYTES`, the longest name
// any writer of this service accepts (`domain::limits`), which size the
// residual (a page or window of names longer than the read has seen).
const _: () = {
    let max_name = crate::domain::limits::MAX_NAME_BYTES as u64;
    assert!(FIELD_ROW_REPLY_BOUND == 144);
    assert!(TIME_SAMPLE_ROW_REPLY_BOUND == 104);
    assert!(TOMBSTONE_ROW_REPLY_BOUND == 57);
    assert!(SECTION_ROW_REPLY_BOUND_WITHOUT_NAME == 112);
    assert!(TIME_SAMPLE_MEMBER_REPLY_BOUND == 37);
    assert!(section_row_reply_bound(max_name) == 4_208);
    assert!(field_member_reply_bound(max_name) == 8_213);
    assert!(field_member_reply_bound(NAME_BYTES_SIZING_FLOOR) == 149);
};

/// A structural WATCH attempt's allowance on its whole-object read (SDD
/// §5.4): the read grows the call's upload charge, never waiting and never
/// shrinking it, before each part of the read it covers, and stops before
/// its next window when a growth is refused
/// ([`read_object_snapshot_slim_charged`]).
///
/// Created per read by the attempt and lent to it; after the read it holds
/// the name bytes the read retained ([`Self::name_bytes`]). It never leaves
/// the attempt's task: the read's pass, running on a `watch-read` thread, reaches it
/// through requests the attempt answers in order
/// ([`read_object_snapshot_unfenced`]). Each target is
/// the whole object-state need the read reaches by then
/// (`ObjectStateNeed::Watch`, the attempt's fixed cost included, no rewrite:
/// the plan grows its own).
pub(crate) struct FanoutAllowance<'a> {
    charge: &'a dyn SharedCommitCharge,
    /// The object's entries, from the count windows (0 before them).
    entries: u64,
    /// `min(wire commands, FANOUT_WINDOW)`.
    in_flight: u64,
    /// The most plain wire commands any plain window of the read has asked
    /// for so far (`WatchSnapshot::plain_in_flight`): 0 until WA1, which
    /// asks for round 1's index pages, then grow-only.
    plain_in_flight: u64,
    /// The byte window WA1 charges: the smaller of `FANOUT_WINDOW_BYTES` and
    /// the read's projected reply and script-request bound
    /// ([`projected_read_reply_bound`], [`projected_read_request_bound`]), so
    /// a small object's read holds about its own windows' bound, not a whole
    /// window. A refused growth's stop projects at least it (0 before WA1).
    byte_window: u64,
    /// The name bytes the read retains, as of its last growth.
    name_bytes: u64,
    /// The best-known whole target, once a growth was refused.
    stopped: Option<u64>,
}

impl<'a> FanoutAllowance<'a> {
    /// An allowance on `charge`.
    pub(crate) fn new(charge: &'a dyn SharedCommitCharge) -> Self {
        Self {
            charge,
            entries: 0,
            in_flight: 0,
            plain_in_flight: 0,
            byte_window: 0,
            name_bytes: 0,
            stopped: None,
        }
    }

    /// The name bytes the read retained: the round-1 field-index member
    /// bytes and the section-name bytes it read (`WatchSnapshot::name_bytes`),
    /// which the attempt's plan and its WA3 keep charged.
    pub(crate) fn name_bytes(&self) -> u64 {
        self.name_bytes
    }

    /// The read's need with `items` and `window_bytes`.
    fn need(&self, items: u64, window_bytes: u64) -> ObjectStateNeed {
        ObjectStateNeed::Watch {
            snapshot: WatchSnapshot {
                entries: self.entries,
                in_flight: self.in_flight,
                plain_in_flight: self.plain_in_flight,
                items,
                window_bytes,
                name_bytes: self.name_bytes,
            },
            fixed: true,
            rewrite: RewriteCount::default(),
            queued: QueuedCommands::default(),
        }
    }

    /// Grow, at least, to the need with `items`, `window_bytes`,
    /// `name_bytes` and a plain window of `plain_commands` wire commands (0
    /// for a script window and after a window); on a refusal record the stop
    /// with the larger of the refused target and the count projection (items
    /// at [`PROJECTED_ITEMS_PER_ENTRY`] per entry, and the larger of
    /// `window_bytes` and WA1's byte window, [`Self::byte_window`], with the
    /// plain commands asked so far). Whether the read goes on.
    fn grow(
        &mut self,
        items: u64,
        window_bytes: u64,
        name_bytes: u64,
        plain_commands: u64,
    ) -> bool {
        self.name_bytes = self.name_bytes.max(name_bytes);
        self.plain_in_flight = self.plain_in_flight.max(plain_commands);
        match self
            .charge
            .try_hold_at_least_need(&self.need(items, window_bytes))
        {
            Ok(_) => true,
            Err(refused) => {
                let projected = self.charge.target(&self.need(
                    self.entries.saturating_mul(PROJECTED_ITEMS_PER_ENTRY),
                    window_bytes.max(self.byte_window),
                ));
                self.stopped = Some(refused.max(projected));
                false
            }
        }
    }

    /// WA1: the count windows' member counts are in — `[sections,
    /// tombstones, fields, time samples, obj:chs]`, the last 0 on an object
    /// whose `obj:chs` the read does not read. `entries` = sections +
    /// tombstones + fields + time samples + 2 × the larger of sections + 2
    /// and the `obj:chs` count (one `LLEN` and one ranged `LRANGE` per list);
    /// `in_flight` counts wire commands, a value row's two. Held at least:
    /// the attempt's fixed cost, those entries, one window of commands in
    /// flight, a byte window, and round 1's index pages as plain commands
    /// (one per index with members, at most five: `plain_in_flight`), before
    /// any index member is read.
    ///
    /// The byte window is the smaller of `max_bytes` (`FANOUT_WINDOW_BYTES`)
    /// and the read's projected reply and script-request bound from these
    /// counts at the name floor ([`projected_read_reply_bound`] +
    /// [`projected_read_request_bound`]): an object whose windows project
    /// past one window holds the whole window, as before, and a small one
    /// holds what its windows can come to. Nothing after WA1 relies on
    /// it: every round-1 page and every round-2 and round-3 window is
    /// charged before it is issued, at its own summed reply bound and a
    /// script window's request bytes ([`FanoutWindows::allowance_grow`]).
    /// Lists longer than projected are
    /// charged so, in advance and exactly, from their round-2 `LLEN`. Names
    /// longer than any the read has seen are not: each pre-issue bound sizes
    /// names at the longest seen so far, so the page or window bringing the
    /// first longer names exceeds its bound, and the excess is the residual,
    /// charged only after it arrives by the growth after that page or window
    /// (SDD §5.4). With the projection in place of a whole window, a small
    /// object no longer has that residual held in advance: up to `W_BYTE` ×
    /// (`FANOUT_WINDOW_BYTES` − the projection) more is held after the
    /// replies arrive rather than before. The byte window is also the floor
    /// of a refused growth's stop projection ([`Self::grow`]), so one wait
    /// normally covers the read.
    fn counted(&mut self, counts: [u64; 5], max_commands: u64, max_bytes: u64) -> bool {
        let [sections, tombstones, fields, time_samples, children_index] = counts;
        let lists = sections.saturating_add(2).max(children_index);
        self.entries = sections
            .saturating_add(tombstones)
            .saturating_add(fields)
            .saturating_add(time_samples)
            .saturating_add(lists.saturating_mul(2));
        let wire_commands = self
            .entries
            .saturating_add(fields)
            .saturating_add(time_samples);
        self.in_flight = wire_commands.min(max_commands);
        self.byte_window = max_bytes.min(
            projected_read_reply_bound(counts).saturating_add(projected_read_request_bound(counts)),
        );
        // Round 1's pages, issued next, are plain: one command per index
        // with members. Held from here, so a stop here projects them and its
        // one wait covers the pages too.
        let page_commands = counts.iter().filter(|&&n| n > 0).count() as u64;
        self.grow(0, self.byte_window, 0, page_commands)
    }
}

/// Where a fan-out pass reports its progress, the reply bytes tallied so far
/// ([`FanoutWindows::report_progress`]).
enum ProgressSink<'a> {
    /// A synchronous sink in the pass's own task: the unfenced read's no-op.
    Local(&'a mut (dyn FnMut(u64) + Send)),
    /// The fenced pass's caller, whose hook tops up its memory grant: each
    /// report a [`PassRequest::Progress`], awaited before the pass goes on
    /// (`storage::fanout_thread`).
    Caller(&'a PassLink),
}

/// How a fan-out pass reaches the charged read's allowance
/// ([`FanoutAllowance`]), which stays with its caller.
enum AllowanceSeam<'a> {
    /// No allowance: the read never stops (every read but the structural
    /// update's charged one).
    None,
    /// The caller's allowance: each seam a request ([`PassRequest::Counted`],
    /// [`PassRequest::Grow`], [`PassRequest::NameBytes`]) the caller answers
    /// before the pass goes on; `stopped` holds the target of a refused
    /// growth's answer.
    Caller {
        link: &'a PassLink,
        stopped: Option<u64>,
    },
}

/// The fan-out's windows over one read, and what it has observed so far.
struct FanoutWindows<'a> {
    client: &'a ValkeyClient,
    pool: &'a fred::clients::Pool,
    /// The key schema: a script window's key prefix and header key.
    keys: &'a KeySchema,
    object_id: &'a str,
    /// The read's tallies ([`FanoutTally`]): what its sinks take in, and what
    /// [`Self::report_progress`] reports.
    tally: FanoutTally,
    /// Where the read reports its progress, the reply bytes tallied so far
    /// (`tally.bytes`): after round 1 and after each window of [`Self::run`].
    /// Between windows, before the next is issued; a local no-op for the
    /// unfenced read.
    progress: ProgressSink<'a>,
    /// `FANOUT_WINDOW` (`config::snapshot_fanout_window`: the ceiling's
    /// window, capped at `config::FANOUT_WINDOW_MAX`).
    max_commands: u64,
    /// `FANOUT_WINDOW_BYTES`.
    max_bytes: u64,
    /// Commands whose replies have been taken in, every family.
    commands_seen: u64,
    /// Reply bytes taken in (as [`FanoutReply::bytes`] counts them), every
    /// family, the header and the indices included: the basis of the
    /// per-command average until an entry reply with any bytes in it arrives.
    bytes_seen: u64,
    /// Entry commands (a section, tombstone, field or time-sample `HGETALL`,
    /// or a children-list `LRANGE`) whose replies have been taken in.
    entry_commands_seen: u64,
    /// Reply bytes of those entry commands: the basis of the per-command
    /// average once it is non-zero (entry replies that were all empty carry no
    /// size to scale from). Index replies are left out because one
    /// index reply holds a member per entry — a `ZRANGE obj:flds` on a
    /// Kitchen_set-sized object is ~100 KB — so averaging them in would size
    /// the entry windows tens of times too small.
    entry_bytes_seen: u64,
    /// The read's position for the test-only seam, across all its rounds
    /// (zero-sized in release builds).
    fault_cursor: SnapshotReadFaultCursor,
    /// The charged read's allowance ([`FanoutAllowance`]), reached through
    /// its caller; [`AllowanceSeam::None`] for every other read, which never
    /// stops.
    allowance: AllowanceSeam<'a>,
    /// The read's reply shape ([`FanoutReplies`]): how its rounds 2 and 3
    /// read and how its windows are sized.
    replies: FanoutReplies,
    /// The bounded-reply read's per-command reply bound summed over the
    /// window about to be issued ([`Self::reply_bound`]), for its `fan-out
    /// window` debug event; taken by that event. `None` on the whole-row read.
    window_bound: Option<u64>,
    /// How the bounded-reply read issues the windows of its rounds 2 and 3
    /// while it has not fallen back ([`ValkeyClient::watch_read_transport`]:
    /// [`WindowTransport::Script`] but under a test's override), but for a
    /// lone entry over τ, which [`Self::bounded_window`] issues plain under
    /// either. Unused on the whole-row read.
    transport: WindowTransport,
    /// Set, for the rest of the read, when a script window met a
    /// cluster-state refusal: that window and every later one are issued
    /// plain ([`Self::run_window`]). A new read starts unset, so it tries the
    /// script again.
    fell_back: bool,
}

/// One window of the bounded-reply read's rounds 2 and 3, as
/// [`FanoutWindows::bounded_window`] cut it.
struct BoundedWindow {
    commands: Vec<FanoutCommand>,
    /// How the window is issued.
    transport: WindowTransport,
    /// Its commands' wire commands (a value row's two-command read counts
    /// two).
    wire: u64,
    /// Its commands' summed per-command reply bounds.
    reply_bound: u64,
    /// A script window's request bytes: its entries' and its column lists'
    /// ([`FanoutCommand::script_request_bytes`]); 0 for a plain window.
    request_bytes: u64,
    /// A script window's modelled shard time on AMR, in ns
    /// ([`crate::config::fanout_script_window_modelled_shard_ns`]); `None`
    /// for a plain window.
    modelled_shard_ns: Option<u64>,
    /// The items its commands can add (their item widths).
    widths: u64,
}

/// What a window is issued as, for the `fan-out window` debug event: the
/// window's transport, and for a plain window whether the read fell back to
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowIssue {
    Script,
    Plain,
    PlainFallback,
}

impl WindowIssue {
    /// The debug event's `transport` value.
    fn label(self) -> &'static str {
        match self {
            Self::Script => "script",
            Self::Plain => "plain",
            Self::PlainFallback => "plain_fallback",
        }
    }
}

/// The items a bounded-reply read's round-2 or round-3 command can add to
/// the read's tally, as the atomic read's item bound counts a row (Spec SDD
/// SR-1.2): a hash its family's pinned width ([`SNAPSHOT_SECTION_HASH_WIDTH`]
/// and the others), a ranged `LRANGE` its entries, an `LLEN` none.
fn item_width(command: &FanoutCommand) -> u64 {
    match (command.family, &command.page) {
        (_, FanoutPage::Prefix { len }) => *len,
        (_, FanoutPage::Length) => 0,
        (SnapshotKeyFamily::Section, _) => SNAPSHOT_SECTION_HASH_WIDTH,
        (SnapshotKeyFamily::DeletedSection, _) => SNAPSHOT_TOMBSTONE_HASH_WIDTH,
        (SnapshotKeyFamily::Field, _) => SNAPSHOT_FIELD_HASH_WIDTH,
        (SnapshotKeyFamily::TimeSample, _) => SNAPSHOT_TIME_SAMPLE_HASH_WIDTH,
        _ => 0,
    }
}

/// Whether a fan-out reply of `family` reads one entry (and so is averaged
/// into the entry-reply basis the window is sized from) rather than the
/// header or an index.
fn is_fanout_entry(family: SnapshotKeyFamily) -> bool {
    matches!(
        family,
        SnapshotKeyFamily::Section
            | SnapshotKeyFamily::DeletedSection
            | SnapshotKeyFamily::Field
            | SnapshotKeyFamily::TimeSample
            | SnapshotKeyFamily::Children
    )
}

impl<'a> FanoutWindows<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        client: &'a ValkeyClient,
        pool: &'a fred::clients::Pool,
        keys: &'a KeySchema,
        object_id: &'a str,
        fault_cursor: SnapshotReadFaultCursor,
        progress: ProgressSink<'a>,
        replies: FanoutReplies,
        allowance: AllowanceSeam<'a>,
    ) -> Self {
        Self {
            client,
            pool,
            keys,
            object_id,
            tally: FanoutTally::default(),
            progress,
            allowance,
            replies,
            window_bound: None,
            transport: client.watch_read_transport(),
            fell_back: false,
            max_commands: crate::config::snapshot_fanout_window(
                client.snapshot_atomic_read_max_items,
            ),
            max_bytes: crate::config::snapshot_fanout_window_bytes(
                client.snapshot_atomic_read_max_bytes,
            ),
            commands_seen: 0,
            bytes_seen: 0,
            entry_commands_seen: 0,
            entry_bytes_seen: 0,
            fault_cursor,
        }
    }

    /// The name length the bounded-reply read sizes name-bearing replies at:
    /// the longest name it has seen, at least [`NAME_BYTES_SIZING_FLOOR`].
    fn sizing_name_bytes(&self) -> u64 {
        self.tally.names.longest.max(NAME_BYTES_SIZING_FLOOR)
    }

    /// The bound on `command`'s reply bytes on the bounded-reply read
    /// ([`FanoutReplies::Bounded`]), as [`FanoutReply::bytes`] counts them,
    /// from the writers' formats: a field row 144 B and a time-sample row
    /// 104 B (their two-command reads), a tombstone 57 B, a section 112 B +
    /// its name, an `LLEN` nothing and a ranged `LRANGE` 20 B per entry. A
    /// section's name is sized at [`Self::sizing_name_bytes`], so a longer
    /// name exceeds its bound. No other command is issued in rounds 2 and 3
    /// of that read; one would be bounded at the whole byte window.
    fn reply_bound(&self, command: &FanoutCommand) -> u64 {
        match (command.family, &command.page) {
            (SnapshotKeyFamily::Field, FanoutPage::ValueRow) => FIELD_ROW_REPLY_BOUND,
            (SnapshotKeyFamily::TimeSample, FanoutPage::ValueRow) => TIME_SAMPLE_ROW_REPLY_BOUND,
            (SnapshotKeyFamily::DeletedSection, FanoutPage::Whole) => TOMBSTONE_ROW_REPLY_BOUND,
            (SnapshotKeyFamily::Section, FanoutPage::Whole) => {
                section_row_reply_bound(self.sizing_name_bytes())
            }
            (_, FanoutPage::Length) => 0,
            (_, FanoutPage::Prefix { len }) => len.saturating_mul(ID_REPLY_BOUND),
            _ => self.max_bytes,
        }
    }

    /// The bound on one member's bytes in a round-1 page of `family`'s index
    /// on the bounded-reply read: a decimal id for the sets (sections,
    /// deleted sections, `obj:chs`), 37 B for a time-samples member and
    /// 21 B + twice the sizing name for a fields member.
    fn index_member_bound(&self, family: SnapshotKeyFamily) -> u64 {
        match family {
            SnapshotKeyFamily::FieldsIndex => field_member_reply_bound(self.sizing_name_bytes()),
            SnapshotKeyFamily::TimeSamplesIndex => TIME_SAMPLE_MEMBER_REPLY_BOUND,
            _ => ID_REPLY_BOUND,
        }
    }

    /// The next window of the bounded-reply read's round 2 or 3, taken from
    /// `rest`, to be issued by `transport`: commands while their wire
    /// commands stay within `FANOUT_WINDOW` and their summed reply bounds
    /// within `FANOUT_WINDOW_BYTES`, at least one (a command whose bound
    /// alone exceeds the byte window runs alone).
    ///
    /// A script window is cut by two rules more, since its script blocks the
    /// shard for the whole window and its request is one argument list:
    ///
    /// - its reply bound and its request bytes together
    ///   ([`FanoutCommand::script_request_bytes`], with the column lists)
    ///   stay within `FANOUT_WINDOW_BYTES`;
    /// - its modelled shard time
    ///   ([`crate::config::fanout_script_window_modelled_shard_ns`]: wire
    ///   commands, the entries its ranged `LRANGE`s ask for, and its reply
    ///   bound) stays within τ
    ///   ([`crate::config::FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS`]).
    ///
    /// A first entry whose modelled time alone exceeds τ — a children list
    /// longer than a script window may read — is issued alone as a plain
    /// window: one `LRANGE`, which is atomic on the store anyway and costs
    /// it less than the same read inside a script.
    fn bounded_window(
        &self,
        rest: &mut std::iter::Peekable<std::vec::IntoIter<FanoutCommand>>,
        transport: WindowTransport,
    ) -> BoundedWindow {
        let script = transport == WindowTransport::Script;
        let mut window = BoundedWindow {
            commands: Vec::new(),
            transport,
            wire: 0,
            reply_bound: 0,
            request_bytes: if script {
                SCRIPT_WINDOW_COLUMN_ARG_BYTES
            } else {
                0
            },
            modelled_shard_ns: None,
            widths: 0,
        };
        let mut list_entries = 0u64;
        while let Some(next) = rest.peek() {
            let wire = window.wire.saturating_add(next.wire_commands());
            let reply_bound = window.reply_bound.saturating_add(self.reply_bound(next));
            let (request_bytes, lists) = if script {
                let list_len = match next.page {
                    FanoutPage::Prefix { len } => len,
                    _ => 0,
                };
                (
                    window
                        .request_bytes
                        .saturating_add(next.script_request_bytes()),
                    list_entries.saturating_add(list_len),
                )
            } else {
                (0, 0)
            };
            let shard_ns =
                crate::config::fanout_script_window_modelled_shard_ns(wire, lists, reply_bound);
            let over_budget =
                script && shard_ns > crate::config::FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS;
            if window.commands.is_empty() {
                if over_budget {
                    // One entry over τ: a plain window of its own.
                    window.transport = WindowTransport::Plain;
                    window.request_bytes = 0;
                    window.wire = wire;
                    window.reply_bound = reply_bound;
                    window.widths = item_width(next);
                    window.commands.extend(rest.next());
                    return window;
                }
            } else if wire > self.max_commands
                || reply_bound.saturating_add(request_bytes) > self.max_bytes
                || over_budget
            {
                break;
            }
            window.wire = wire;
            window.reply_bound = reply_bound;
            window.request_bytes = request_bytes;
            list_entries = lists;
            if script {
                window.modelled_shard_ns = Some(shard_ns);
            }
            window.widths = window.widths.saturating_add(item_width(next));
            window.commands.extend(rest.next());
        }
        window
    }

    /// Run one round's commands in windows, handing every reply to `sink` in
    /// command order, with the read's tally, as soon as its window completes —
    /// before the next window is issued — so the raw replies awaiting
    /// processing are never more than one window's. After each window it
    /// reports the read's progress ([`Self::report_progress`]), and on the
    /// fenced pass waits until its caller has applied the report before the
    /// next window is issued. The first
    /// failure fails the round: a failed command, classified by
    /// [`classify_snapshot_read_err`] under its family and key on the
    /// snapshot lane's pool label, or an error `sink` returns.
    ///
    /// On the whole-row read each window is sized by [`fanout_window_len`]
    /// from the average reply bytes per entry command seen so far in this
    /// read, or, until an entry reply with any bytes in it has arrived, per
    /// command of any family, and issued plain. On the bounded-reply read
    /// each is sized from its commands' reply bounds and, for a script
    /// window, its request bytes and modelled shard time
    /// ([`Self::bounded_window`]), and issued as one window script unless it
    /// is a lone entry over τ, which [`Self::bounded_window`] issues plain,
    /// or the read has fallen back ([`Self::run_window`]).
    ///
    /// The test-only seam ([`ObjectSnapshotReadFaults`]) is consulted once per
    /// command, in command order across every round of the read, before the
    /// window is issued: a queued canned error answers in place of the command
    /// its target names (by default the window's first) and takes the same
    /// classification a real error of that command takes, and a queued delay
    /// holds the command it names back before it is issued. Since a failed
    /// command fails the read, one read consumes at most one queued error. A
    /// script window consults it the same way, once per command
    /// ([`issue_script_window`]).
    ///
    /// On a charged read the allowance grows the call's charge, never
    /// waiting and never shrinking it, before each window is issued — to its
    /// summed reply bound, a script window's request bytes, the items it can
    /// add, and a plain window's wire commands — and after it, to what it
    /// received, the items tallied and the names retained
    /// ([`Self::allowance_grow`]), each growth answered by the caller before
    /// the pass goes on; a refused growth ends the round before its next
    /// window, with `Ok`, and the caller reads the stop from
    /// [`Self::stopped`].
    async fn run<F>(
        &mut self,
        commands: Vec<FanoutCommand>,
        mut sink: F,
    ) -> Result<(), ServiceError>
    where
        F: FnMut(&mut FanoutTally, FanoutCommand, FanoutReply) -> Result<(), ServiceError>,
    {
        let mut rest = commands.into_iter().peekable();
        let mut remaining = rest.len();
        while rest.peek().is_some() {
            let bytes_before = self.bytes_seen;
            let mut call = WindowCall::Plain;
            let window: Vec<FanoutCommand> = match self.replies {
                FanoutReplies::Whole => {
                    // The entry basis only once it holds bytes: entry replies
                    // that were all empty (indexed hashes deleted under the
                    // read) carry no size to scale from, and would otherwise
                    // drop the byte window entirely.
                    let (commands_seen, bytes_seen) =
                        if self.entry_commands_seen > 0 && self.entry_bytes_seen > 0 {
                            (self.entry_commands_seen, self.entry_bytes_seen)
                        } else {
                            (self.commands_seen, self.bytes_seen)
                        };
                    let len = fanout_window_len(
                        self.max_commands,
                        self.max_bytes,
                        commands_seen,
                        bytes_seen,
                        remaining,
                    );
                    rest.by_ref().take(len).collect()
                }
                FanoutReplies::Bounded => {
                    let transport = if self.fell_back {
                        WindowTransport::Plain
                    } else {
                        self.transport
                    };
                    let window = self.bounded_window(&mut rest, transport);
                    // Before the window is issued: its replies' bound, a
                    // script window's request bytes, the items it can add and
                    // a plain window's wire commands.
                    let items = self.tally.items.saturating_add(window.widths);
                    let plain_commands = match window.transport {
                        WindowTransport::Plain => window.wire,
                        WindowTransport::Script => 0,
                    };
                    if !self
                        .allowance_grow(
                            items,
                            window.reply_bound.saturating_add(window.request_bytes),
                            plain_commands,
                        )
                        .await
                    {
                        return Ok(());
                    }
                    self.window_bound = Some(window.reply_bound);
                    if window.transport == WindowTransport::Script {
                        call = WindowCall::Script {
                            items,
                            reply_bound: window.reply_bound,
                            wire: window.wire,
                            request_bytes: window.request_bytes,
                            modelled_shard_ns: window.modelled_shard_ns.unwrap_or_default(),
                        };
                    }
                    window.commands
                }
            };
            remaining -= window.len();
            self.run_window(window, call, &mut sink).await?;
            // A script window's fallback whose growth was refused issued
            // nothing more: the read stops here.
            if self.stopped().is_some() {
                return Ok(());
            }
            self.report_progress().await;
            // After it: what it received, with the items tallied and the
            // names retained so far.
            let items = self.tally.items;
            if !self
                .allowance_grow(items, self.bytes_seen - bytes_before, 0)
                .await
            {
                return Ok(());
            }
        }
        Ok(())
    }

    /// Report the read's progress: the reply bytes tallied so far
    /// (`tally.bytes`, the tally [`AtomicSnapshot::measured_size`] ends
    /// with), to the read's progress sink. Between windows: the fenced pass
    /// waits here until its caller — whose renewal timer keeps the lease
    /// meanwhile — has applied the report, so no window is issued before the
    /// report of the one before it is applied.
    async fn report_progress(&mut self) {
        let bytes_read = self.tally.bytes;
        match &mut self.progress {
            ProgressSink::Local(sink) => (*sink)(bytes_read),
            ProgressSink::Caller(link) => {
                link.ask(PassRequest::Progress { bytes_read }).await;
            }
        }
    }

    /// Ask the charged read's caller to apply one allowance seam
    /// ([`FanoutAllowance`]); whether the read goes on. A refusal's answer
    /// carries the stop's target, kept for [`Self::stopped`]. Always goes on
    /// on an uncharged read, which asks nothing.
    async fn ask_allowance(&mut self, request: PassRequest) -> bool {
        match &mut self.allowance {
            AllowanceSeam::None => true,
            AllowanceSeam::Caller { link, stopped } => match link.ask(request).await {
                PassAnswer::GoOn => true,
                PassAnswer::Stop { target } => {
                    *stopped = Some(target);
                    false
                }
            },
        }
    }

    /// WA1 on a charged read: grow its charge once the count windows' member
    /// counts (`[sections, tombstones, fields, time samples, obj:chs]`) are
    /// in, before any member is read ([`FanoutAllowance::counted`]). Whether
    /// the read goes on; always on an uncharged read.
    async fn allowance_counted(&mut self, counts: [u64; 5]) -> bool {
        let (max_commands, max_bytes) = (self.max_commands, self.max_bytes);
        self.ask_allowance(PassRequest::Counted {
            counts,
            max_commands,
            max_bytes,
        })
        .await
    }

    /// Grow a charged read's charge, at least, to the need with `items`, one
    /// window of `window_bytes`, `plain_commands` plain wire commands in
    /// flight and the name bytes the read retains so far: before a window or
    /// index page is issued, with its summed reply bound (and a script
    /// window's request bytes), the items it can add and, for a plain window,
    /// its wire commands; after it, with what it received and no plain
    /// commands. Whether the read goes on; always on an uncharged read.
    /// Between windows: the pass issues nothing until its caller answered
    /// ([`FanoutAllowance::grow`]).
    ///
    /// After a window this charges what arrived, so a window that received
    /// more than its bound is charged here, after its replies arrived: the
    /// residuals (SDD §5.4, [`read_object_snapshot_slim_charged`]). Names
    /// longer than any the read had seen exceed it at most one page or window
    /// at a time, since the sizing name then rises to the longest seen; a
    /// compact-encoded set index answering its whole membership to one
    /// round-1 `SSCAN` step exceeds it at most once per index, by at most
    /// that encoding's entry limit × [`ID_REPLY_BOUND`] (10,240 B at the
    /// store's defaults); a hashtable-encoded set index's step, whose last
    /// bucket can carry it past its share, exceeds it by a few members ×
    /// [`ID_REPLY_BOUND`] per step, never more over a read than the set's own
    /// size × [`ID_REPLY_BOUND`].
    async fn allowance_grow(&mut self, items: u64, window_bytes: u64, plain_commands: u64) -> bool {
        let name_bytes = self.tally.names.bytes;
        self.ask_allowance(PassRequest::Grow {
            items,
            window_bytes,
            name_bytes,
            plain_commands,
        })
        .await
    }

    /// The charged read's stop: the best-known whole target, once a growth
    /// was refused; `None` while it reads, and always on an uncharged read.
    fn stopped(&self) -> Option<u64> {
        match &self.allowance {
            AllowanceSeam::Caller { stopped, .. } => *stopped,
            AllowanceSeam::None => None,
        }
    }

    /// Issue `window` as one fan-out window, as it stands — [`Self::run`]
    /// sizes its windows first; round 1 plans its own, bounded in index
    /// members ([`fanout_round_one`]) — as `call` says, and hand every reply
    /// to `sink` in command order, with the read's tally, once the window
    /// completes. The test-only seam, the classification of a failed command
    /// and the `fan-out window` debug event are [`Self::run`]'s, per window.
    ///
    /// A script window ([`WindowCall::Script`]) is one `EVALSHA` of the
    /// window script ([`issue_script_window`]). If it meets a cluster-state
    /// refusal (or the seam answers one in place of one of its commands),
    /// the read falls back, once and for the rest of the read
    /// ([`Self::fell_back`], `live_edit_snapshot_read_window_fallback_total`):
    /// the call's charge is grown for the same window as plain commands, and
    /// the same window is issued plain, without consulting the seam again. A
    /// refused growth issues nothing more and returns `Ok`, the stop in
    /// [`Self::stopped`].
    async fn run_window<F>(
        &mut self,
        window: Vec<FanoutCommand>,
        call: WindowCall,
        sink: &mut F,
    ) -> Result<(), ServiceError>
    where
        F: FnMut(&mut FanoutTally, FanoutCommand, FanoutReply) -> Result<(), ServiceError>,
    {
        let issued = std::time::Instant::now();
        let (replies, issue, request_bytes, modelled_shard_ns) = match call {
            WindowCall::Plain => {
                let issue = if self.fell_back {
                    WindowIssue::PlainFallback
                } else {
                    WindowIssue::Plain
                };
                (self.issue_plain(&window, true).await?, issue, None, None)
            }
            WindowCall::Script {
                items,
                reply_bound,
                wire,
                request_bytes,
                modelled_shard_ns,
            } => {
                match issue_script_window(
                    self.client,
                    self.pool,
                    self.keys,
                    self.object_id,
                    &mut self.fault_cursor,
                    &window,
                )
                .await?
                {
                    ScriptWindow::Replies(replies) => (
                        replies,
                        WindowIssue::Script,
                        Some(request_bytes),
                        Some(modelled_shard_ns),
                    ),
                    ScriptWindow::ClusterState => {
                        self.fell_back = true;
                        if let Some(metrics) = self.client.metrics_snapshot() {
                            metrics.inc_snapshot_read_window_fallback();
                        }
                        // The same window as plain commands, charged as such
                        // before it is issued.
                        if !self.allowance_grow(items, reply_bound, wire).await {
                            self.window_bound = None;
                            return Ok(());
                        }
                        (
                            self.issue_plain(&window, false).await?,
                            WindowIssue::PlainFallback,
                            None,
                            None,
                        )
                    }
                }
            }
        };
        let replied = std::time::Instant::now();
        let commands = window.len();
        let first_family = window.first().map(|c| c.family.label());
        let last_family = window.last().map(|c| c.family.label());
        let mut window_bytes = 0u64;
        let mut index_members = 0u64;
        for (command, reply) in window.into_iter().zip(replies) {
            let bytes = reply.bytes();
            window_bytes += bytes;
            self.commands_seen += 1;
            self.bytes_seen += bytes;
            if is_fanout_entry(command.family) {
                self.entry_commands_seen += 1;
                self.entry_bytes_seen += bytes;
            } else if command.page != FanoutPage::Whole {
                // Round 1's index parts only: the members `member_target`
                // bounds. A marked object's `obj:chs` `SMEMBERS` in round 2 is
                // read whole and is not bounded by it.
                index_members += reply.items();
            }
            sink(&mut self.tally, command, reply)?;
        }
        // One event per completed window, round 1's included, for the
        // fan-out's validation measurements (the Epic 3 report): how many
        // commands, reply bytes and round-1 index members it carried against
        // the command, byte and index-member targets, the families it spanned,
        // how long its replies took to arrive and how long `sink` took
        // over them; on the bounded-reply read, the reply bound it was sized
        // and charged within (`reply_byte_bound`, absent otherwise); how it
        // was issued (`transport`: `script`, `plain`, or `plain_fallback` for
        // a window issued plain because the read fell back from its script);
        // and on a script window its request bytes and modelled shard time
        // (`request_bytes`, `modelled_shard_ns`, absent otherwise). Debug
        // level: emitted only when the log filter enables debug for this
        // module.
        let reply_byte_bound = self.window_bound.take();
        tracing::debug!(
            object_id = self.object_id,
            commands,
            reply_bytes = window_bytes,
            reply_byte_bound,
            index_members,
            command_target = self.max_commands,
            byte_target = self.max_bytes,
            member_target = crate::config::FANOUT_INDEX_WINDOW_MEMBERS,
            first_family,
            last_family,
            transport = issue.label(),
            request_bytes,
            modelled_shard_ns,
            replies_us = replied.duration_since(issued).as_micros() as u64,
            sink_us = replied.elapsed().as_micros() as u64,
            "fan-out window"
        );
        Ok(())
    }

    /// Issue `window` as plain commands, each its own call, through the
    /// issuer of the read's reply shape, and return the replies in command
    /// order ([`issue_window`]). `consult_seam` is false only for a script
    /// window's plain re-issue after a fallback, whose commands the seam has
    /// already counted.
    async fn issue_plain(
        &mut self,
        window: &[FanoutCommand],
        consult_seam: bool,
    ) -> Result<Vec<FanoutReply>, ServiceError> {
        let (client, pool, object_id) = (self.client, self.pool, self.object_id);
        let cursor = consult_seam.then_some(&mut self.fault_cursor);
        // One per-command future type per reply shape, never one for both:
        // see `issue_bounded_fanout_command`.
        match self.replies {
            FanoutReplies::Whole => {
                issue_window(
                    client,
                    pool,
                    object_id,
                    cursor,
                    window,
                    issue_fanout_command,
                )
                .await
            }
            FanoutReplies::Bounded => {
                issue_window(
                    client,
                    pool,
                    object_id,
                    cursor,
                    window,
                    issue_bounded_fanout_command,
                )
                .await
            }
        }
    }
}

/// How [`FanoutWindows::run_window`] issues a window.
enum WindowCall {
    /// Plain commands: every window of round 1 and of the whole-row read, a
    /// lone entry over τ, and every window of a read that fell back.
    Plain,
    /// One window script, with what its pre-issue growth asked (`items`,
    /// `reply_bound`) and its wire commands, which a fallback's plain-window
    /// growth asks again; and its request bytes and modelled shard time, for
    /// the debug event.
    Script {
        items: u64,
        reply_bound: u64,
        wire: u64,
        request_bytes: u64,
        modelled_shard_ns: u64,
    },
}

/// How a script window ended without an error ([`issue_script_window`]).
enum ScriptWindow {
    /// Every entry's reply, in command order.
    Replies(Vec<FanoutReply>),
    /// A cluster-state refusal of an undeclared same-slot key
    /// ([`is_script_cluster_state_refusal`]), from the store or from the
    /// test-only seam: the read falls back to plain commands.
    ClusterState,
}

/// Issue `window`'s commands at once (`try_join_all`), each through `issue`,
/// the issuer of the read's reply shape ([`issue_fanout_command`] for
/// [`FanoutReplies::Whole`], [`issue_bounded_fanout_command`] for
/// [`FanoutReplies::Bounded`]), and return their replies in command order,
/// a failure classified under the command that met it: the plain transport
/// ([`WindowTransport::Plain`]). `try_join_all` collects the iterator before
/// polling anything, so the test-only seam sees the window's commands in
/// order, each exactly once. With no `cursor` the seam is not consulted: a
/// script window's plain re-issue after its fallback, whose commands the seam
/// has already counted, passes none.
async fn issue_window<'w, I, Fut>(
    client: &'w ValkeyClient,
    pool: &'w fred::clients::Pool,
    object_id: &'w str,
    mut cursor: Option<&mut SnapshotReadFaultCursor>,
    window: &'w [FanoutCommand],
    issue: I,
) -> Result<Vec<FanoutReply>, ServiceError>
where
    I: Fn(&'w fred::clients::Pool, &'w FanoutCommand) -> Fut + Copy,
    Fut: std::future::Future<Output = Result<FanoutReply, fred::error::Error>>,
{
    let futures = window.iter().enumerate().map(|(i, command)| {
        let injected = match cursor.as_deref_mut() {
            Some(cursor) => client.object_snapshot_read_faults.take_fanout_command(
                cursor,
                command.family,
                i == 0,
            ),
            None => FanoutCommandFault::none(),
        };
        issue_window_command(client, pool, object_id, injected, command, issue)
    });
    futures::future::try_join_all(futures).await
}

/// One command of a fan-out window, as [`issue_window`] holds it: the
/// test-only seam's `injected` fault first (park on its hold, wait out its
/// delay, answer its error in the command's place), then the command through
/// `issue`, a failure classified under the command. Every command of a
/// window holds this future at once, so its size is per command in flight;
/// it is `issue`'s future plus the seam's fault, about 200 B more (see
/// [`issue_bounded_fanout_command`]).
async fn issue_window_command<'w, I, Fut>(
    client: &'w ValkeyClient,
    pool: &'w fred::clients::Pool,
    object_id: &'w str,
    injected: FanoutCommandFault,
    command: &'w FanoutCommand,
    issue: I,
) -> Result<FanoutReply, ServiceError>
where
    I: Fn(&'w fred::clients::Pool, &'w FanoutCommand) -> Fut,
    Fut: std::future::Future<Output = Result<FanoutReply, fred::error::Error>>,
{
    if let Some(hold) = injected.hold {
        hold.park().await;
    }
    if let Some(delay) = injected.delay {
        tokio::time::sleep(delay).await;
    }
    let reply = match injected.error {
        Some(e) => Err(e),
        None => issue(pool, command).await,
    };
    reply.map_err(|e| {
        classify_snapshot_read_err(
            client,
            POOL_LABEL_SNAPSHOT,
            object_id,
            command.family,
            &command.key,
            &command.op_label(),
            e,
        )
    })
}

/// Issue one command of the bounded-reply read's plain windows
/// ([`FanoutReplies::Bounded`], [`WindowTransport::Plain`]) on the snapshot
/// lane: a value row ([`FanoutPage::ValueRow`]) by its two-command read
/// ([`value_row_read`]), every other command as [`issue_fanout_command`]
/// issues it. Its plain windows are round 1's, a lone children list too long
/// for a script window, and every window of a read that fell back from its
/// window script; it is also the oracle the script windows' decode is held
/// to.
///
/// One issuer per reply shape, because an async fn's future is as large as
/// its largest arm and every command of a window holds its future at once
/// ([`issue_window`]). The value-row read's two joined fred calls make its
/// future about twice the size of one fred call's, so a single issuer with
/// a value-row arm would carry that size on every command of every window.
/// The whole-row read (the fenced read's pass, whose window transient the
/// catch-up memory budget's window term `X` funds) never reads a
/// value row, so its issuer, [`issue_fanout_command`], has no value-row arm
/// and its per-command future is the size of its largest arm, one fred call
/// or one `SSCAN` step. This issuer keeps the value-row read inline and
/// unboxed: its plain windows are charged per plain command by
/// `W_PLAIN_CMD`
/// (`grpc::upload_admission::WATCH_SNAPSHOT_PLAIN_COMMAND_CHARGE_BYTES`),
/// whose figure was fitted on this shape, one future per command holding the
/// value-row read inline. Boxing it would add a second heap allocation to
/// every value-row command that the fit never saw.
///
/// Measured with `size_of_val` in a debug build of the tree before the
/// window script: this issuer's future is 13,568 B and its
/// [`issue_window_command`] future 13,768 B, equal to those of the one issuer
/// with the value-row arm inline that the figure was fitted on, rebuilt in
/// that tree to measure it (the value-row read alone is 13,544 B); the
/// whole-row issuer's future is 7,016 B and its [`issue_window_command`]
/// future 7,216 B. Release sizes not measured.
async fn issue_bounded_fanout_command(
    pool: &fred::clients::Pool,
    command: &FanoutCommand,
) -> Result<FanoutReply, fred::error::Error> {
    if matches!(
        (&command.page, fanout_op(command.family), command.family),
        (
            FanoutPage::ValueRow,
            FanoutOp::Hgetall,
            SnapshotKeyFamily::Field | SnapshotKeyFamily::TimeSample
        )
    ) {
        value_row_read(pool, command).await
    } else {
        issue_fanout_command(pool, command).await
    }
}

/// Issue one fan-out command on the snapshot lane: every command of the
/// whole-row read ([`FanoutReplies::Whole`]), and every command but a value
/// row of the bounded-reply read ([`issue_bounded_fanout_command`]). It has
/// no value-row arm, so its future is the size of its largest arm, one fred
/// call or one `SSCAN` step.
async fn issue_fanout_command(
    pool: &fred::clients::Pool,
    command: &FanoutCommand,
) -> Result<FanoutReply, fred::error::Error> {
    let key = command.key.as_str();
    match (&command.page, fanout_op(command.family)) {
        (FanoutPage::Whole, FanoutOp::Hgetall) => pool.hgetall(key).await.map(FanoutReply::Hash),
        (FanoutPage::Whole, FanoutOp::Smembers) => pool.smembers(key).await.map(FanoutReply::List),
        (FanoutPage::Whole, FanoutOp::Zrange) => pool
            .zrange(key, 0, -1, None, false, None, false)
            .await
            .map(FanoutReply::List),
        (FanoutPage::Whole, FanoutOp::Lrange) => {
            pool.lrange(key, 0, -1).await.map(FanoutReply::List)
        }
        (FanoutPage::Count, FanoutOp::Smembers) => pool.scard(key).await.map(FanoutReply::Count),
        (FanoutPage::Count, FanoutOp::Zrange) => pool.zcard(key).await.map(FanoutReply::Count),
        (FanoutPage::Ranks { start, stop }, FanoutOp::Zrange) => {
            let rank = |r: u64| i64::try_from(r).unwrap_or(i64::MAX);
            pool.zrange(key, rank(*start), rank(*stop), None, false, None, false)
                .await
                .map(FanoutReply::List)
        }
        (FanoutPage::Scan { cursor, count }, FanoutOp::Smembers) => {
            sscan_step(pool, key, cursor, *count).await
        }
        // The bounded-reply read's children lists: their length, then the
        // range the length names.
        (FanoutPage::Length, FanoutOp::Lrange) => pool.llen(key).await.map(FanoutReply::Count),
        (FanoutPage::Prefix { len }, FanoutOp::Lrange) => {
            let stop = i64::try_from(len.saturating_sub(1)).unwrap_or(i64::MAX);
            pool.lrange(key, 0, stop).await.map(FanoutReply::List)
        }
        // Round 1 builds index parts only for the round-1 indices, each with
        // the part its own command takes, and the bounded-reply read list
        // parts only for children lists; its value rows go to
        // `issue_bounded_fanout_command`'s value-row read and never reach
        // here. Nothing else ever does either.
        (page, op) => Err(fred::error::Error::new(
            fred::error::ErrorKind::Unknown,
            format!("fan-out command {op:?} cannot read part {page:?} of {key}"),
        )),
    }
}

/// A value row's two-command read ([`FanoutPage::ValueRow`]): `HSTRLEN row
/// value` and `HMGET row` of [`value_row_columns`], issued together; then,
/// only for a row with a value length of 0 and either a `hash` or no known
/// column, `HEXISTS row value` (an empty value beside the hash decides the
/// shape, and an empty value alone — a row no writer of this service writes —
/// is an inline row the tally counts), and only for a row with no known
/// column whose `HEXISTS` answered no `value`, `EXISTS row` (absent, or
/// holding only columns no writer writes). Every reply is an integer or one
/// of the `HMGET` columns, so the read's reply is bounded whatever the row
/// holds, and its inline value is never transferred.
///
/// A failure is returned as the failing command's error, with that command
/// recorded on `command` for its `op` label ([`FanoutCommand::op_label`]);
/// when both of the first two fail, `HSTRLEN`'s.
async fn value_row_read(
    pool: &fred::clients::Pool,
    command: &FanoutCommand,
) -> Result<FanoutReply, fred::error::Error> {
    use fred::interfaces::KeysInterface;
    let key = command.key.as_str();
    let failed = |op: &'static str, e: fred::error::Error| {
        let _ = command.failed_op.set(op);
        e
    };
    let (value_len, columns) = futures::future::join(
        pool.hstrlen::<u64, _, _>(key, "value"),
        pool.hmget::<Vec<Option<String>>, _, _>(key, value_row_columns(command.family).to_vec()),
    )
    .await;
    let value_len = value_len.map_err(|e| failed("HSTRLEN", e))?;
    let columns = columns.map_err(|e| failed("HMGET", e))?;
    let mut row = ValueRowReply {
        family: command.family,
        value_len,
        columns,
        value_exists: None,
        row_exists: None,
        atomic: false,
    };
    // A row with a `hash` or no known column, and no value length, may still
    // hold an empty `value`: `HEXISTS` decides it. A row with another known
    // column but no `hash` is inline by the writers' rule and needs nothing.
    if row.asks_value_exists() {
        row.value_exists = Some(
            pool.hexists::<bool, _, _>(key, "value")
                .await
                .map_err(|e| failed("HEXISTS", e))?,
        );
    }
    // No known column and no `value`: absent, or holding only columns no
    // writer writes.
    if row.asks_row_exists() {
        row.row_exists = Some(
            pool.exists::<u64, _>(key)
                .await
                .map_err(|e| failed("EXISTS", e))?
                > 0,
        );
    }
    Ok(FanoutReply::ValueRow(row))
}

/// External-value clean-up's read of one field or time-sample row
/// (`storage::retirement`'s reference walk; `family` is
/// [`SnapshotKeyFamily::Field`] or [`SnapshotKeyFamily::TimeSample`], `key`
/// the row's key): the row's external hash by the readers' strict rule, and
/// the string bytes the read received.
///
/// The row is read by the bounded value-row read ([`value_row_read`]: `HSTRLEN`
/// and `HMGET`, with the `HEXISTS` / `EXISTS` follow-ups), so its inline value
/// is never transferred, and its shape follows [`decode_field_value`]'s rule
/// ([`ValueRowReply::shape`]): a row holding a `value` — beside a `hash` or not
/// — is inline and names no hash, as every reader decodes it; a row holding a
/// `hash` alone names that hash, validated by the strict decode
/// ([`parse_external_hash`]: a malformed column is `Internal`); an absent row
/// names none. A failed command is classified under `family` on the snapshot
/// lane by [`classify_snapshot_read_err`] (transient → `Aborted`, permanent →
/// `Internal`, the refusal counters ticked), as the fan-out classifies it.
pub(crate) async fn read_value_row_external_hash(
    client: &ValkeyClient,
    pool: &fred::clients::Pool,
    object_id: &str,
    family: SnapshotKeyFamily,
    key: String,
) -> Result<(Option<[u8; 32]>, u64), ServiceError> {
    let command = FanoutCommand::part(family, key, "", FanoutPage::ValueRow);
    let reply = value_row_read(pool, &command).await.map_err(|e| {
        classify_snapshot_read_err(
            client,
            POOL_LABEL_SNAPSHOT,
            object_id,
            family,
            &command.key,
            &command.op_label(),
            e,
        )
    })?;
    let FanoutReply::ValueRow(row) = reply else {
        return Err(fanout_shape_err(object_id, &command));
    };
    let received = row.received_bytes();
    let hash = match row.shape() {
        SlimRow::External(hash_hex) => Some(parse_external_hash(
            &hash_hex,
            object_id,
            format_args!("{} row {}", value_row_label(family), command.key),
        )?),
        SlimRow::Absent | SlimRow::Inline | SlimRow::InlineBesideHash => None,
    };
    Ok((hash, received))
}

/// Issue `window` as one script window ([`WindowTransport::Script`]): one
/// `EVALSHA` of `scripts/fanout_window_read.lua` on one connection of
/// `pool`, the object's header its one declared key, whose reply is decoded
/// into each entry's reply as the plain issuer would have returned it
/// ([`decode_fanout_window_reply`]), in command order.
///
/// The test-only seam is consulted first, once per command in window order,
/// as [`issue_window`] consults it, so its cursor counts the same commands
/// whichever transport issues them. Then, before anything is sent: every hold
/// the window took is parked on, in order; the window waits out the longest
/// of its delays (the plain window's commands wait out theirs at once); and a
/// canned error on command `i` is that entry's in-script refusal — a
/// cluster-state text ([`is_script_cluster_state_refusal`]) answers
/// [`ScriptWindow::ClusterState`], anything else fails the read classified
/// under command `i`'s family, key and `op` label, as on a plain window. No
/// `EVALSHA` is sent for a window the seam answered.
///
/// The `EVALSHA` is sent by [`crate::storage::lua::evalsha_reload_rebuilding`]
/// (a `NOSCRIPT` loads the script on every primary and sends the window once
/// more), and a failure is classified by [`classify_script_window_err`]. A
/// command no script window carries is refused `Internal` before anything is
/// sent; [`FanoutWindows::bounded_window`] never builds one.
async fn issue_script_window(
    client: &ValkeyClient,
    pool: &fred::clients::Pool,
    keys: &KeySchema,
    object_id: &str,
    cursor: &mut SnapshotReadFaultCursor,
    window: &[FanoutCommand],
) -> Result<ScriptWindow, ServiceError> {
    let header_key = keys.obj_header(object_id);
    let mut holds = Vec::new();
    let mut delay: Option<Duration> = None;
    let mut injected: Option<(usize, fred::error::Error)> = None;
    for (i, command) in window.iter().enumerate() {
        let fault =
            client
                .object_snapshot_read_faults
                .take_fanout_command(cursor, command.family, i == 0);
        holds.extend(fault.hold);
        delay = delay.max(fault.delay);
        if let Some(e) = fault.error {
            injected = Some((i, e));
        }
    }
    for hold in holds {
        hold.park().await;
    }
    if let Some(delay) = delay {
        tokio::time::sleep(delay).await;
    }
    if let Some((i, e)) = injected {
        if is_script_cluster_state_refusal(e.details()) {
            return Ok(ScriptWindow::ClusterState);
        }
        let command = &window[i];
        return Err(classify_snapshot_read_err(
            client,
            POOL_LABEL_SNAPSHOT,
            object_id,
            command.family,
            &command.key,
            &command.op_label(),
            e,
        ));
    }

    let ops: Vec<ScriptOp> = window
        .iter()
        .map(|command| {
            command.script_op().ok_or_else(|| {
                ServiceError::Internal(format!(
                    "fan-out snapshot read of object {object_id}: {} ({}) cannot be read by a \
                     script window",
                    command.op_label(),
                    command.key
                ))
            })
        })
        .collect::<Result<_, _>>()?;
    let build_args = || {
        let mut args: Vec<fred::types::Value> = Vec::with_capacity(4 + 3 * window.len());
        args.push(keys.prefix().into());
        args.push(object_id.into());
        args.push(column_arg(SnapshotKeyFamily::Field).into());
        args.push(column_arg(SnapshotKeyFamily::TimeSample).into());
        for (command, op) in window.iter().zip(&ops) {
            args.push(op.letter().into());
            args.push(command.id.as_str().into());
            if let ScriptOp::Prefix { len } = op {
                args.push(i64::try_from(*len).unwrap_or(i64::MAX).into());
            }
        }
        args
    };
    let reply = crate::storage::lua::evalsha_reload_rebuilding::<fred::types::Value>(
        pool.next(),
        &crate::storage::lua::FANOUT_WINDOW_READ_SCRIPT,
        &header_key,
        build_args,
    )
    .await;
    match reply {
        Ok(reply) => decode_fanout_window_reply(object_id, &header_key, window, &ops, reply)
            .map(ScriptWindow::Replies),
        Err(e) => classify_script_window_err(client, object_id, &header_key, window, &ops, e),
    }
}

/// Classify a failed script window (SDD §10.5), ticking the refusal
/// counters on the way, as [`classify_object_snapshot_read_err`] classifies
/// the atomic read's:
///
/// - a `FANOUT_WINDOW_REFUSED: <entry> <command> <raw>` tag whose raw error is
///   a cluster-state refusal ([`is_script_cluster_state_refusal`]) is the
///   fallback: [`ScriptWindow::ClusterState`];
/// - any other tag naming an entry of the window and one of that entry's own
///   commands ([`ScriptOp::commands`]) is a per-key refusal of that entry,
///   classified under the entry's family, key and `op` label — a value row's
///   under the command that met it (`failed_op`), so a refusal reads the same
///   whichever transport met it — with a raw `WRONGTYPE ` handed over as the
///   server's `WRONGTYPE` reply (`ErrorKind::InvalidArgument`, which ticks
///   `live_edit_snapshot_read_type_error_total{key_family}`), anything else
///   as the script's own `ErrorKind::Unknown` reply (permanent);
/// - everything else — a tag with an entry out of range or a command foreign
///   to its entry, `FANOUT_WINDOW_ARGS`, a script-level failure (a timeout,
///   IO, `-BUSY`, `-LOADING`, a `NOSCRIPT` after the reload) — says nothing
///   about any one entry's key, so it is attributed to the object's header
///   under `op` `EVALSHA fanout_window_read`, the atomic read's rule.
fn classify_script_window_err(
    client: &ValkeyClient,
    object_id: &str,
    header_key: &str,
    window: &[FanoutCommand],
    ops: &[ScriptOp],
    e: fred::error::Error,
) -> Result<ScriptWindow, ServiceError> {
    if let Some((entry, command_name, raw)) = parse_fanout_window_refusal(&e) {
        let entry = entry
            .checked_sub(1)
            .and_then(|i| Some((window.get(i)?, ops.get(i)?)))
            .and_then(|(command, op)| {
                let name = op
                    .commands()
                    .iter()
                    .copied()
                    .find(|name| *name == command_name)?;
                Some((command, op, name))
            });
        if let Some((command, op, name)) = entry {
            if is_script_cluster_state_refusal(raw) {
                return Ok(ScriptWindow::ClusterState);
            }
            if matches!(op, ScriptOp::Field | ScriptOp::TimeSample) {
                let _ = command.failed_op.set(name);
            }
            let counted = if raw.starts_with("WRONGTYPE ") {
                fred::error::Error::new(fred::error::ErrorKind::InvalidArgument, raw.to_owned())
            } else {
                e.clone()
            };
            return Err(classify_snapshot_read_err(
                client,
                POOL_LABEL_SNAPSHOT,
                object_id,
                command.family,
                &command.key,
                &command.op_label(),
                counted,
            ));
        }
    }
    Err(classify_snapshot_read_err(
        client,
        POOL_LABEL_SNAPSHOT,
        object_id,
        SnapshotKeyFamily::Header,
        header_key,
        FANOUT_WINDOW_READ_OP,
        e,
    ))
}

/// Parse a `FANOUT_WINDOW_REFUSED: <entry> <command> <raw error>` script
/// reply into its 1-based entry index, command and raw error. Kind-gated and
/// prefix-anchored, as [`parse_snapshot_key_refusal`] is: a script's own
/// error reply arrives as `ErrorKind::Unknown`, so the tag is trusted only
/// there.
fn parse_fanout_window_refusal(e: &fred::error::Error) -> Option<(usize, &str, &str)> {
    if !matches!(e.kind(), fred::error::ErrorKind::Unknown) {
        return None;
    }
    let rest = e.details().strip_prefix(FANOUT_WINDOW_REFUSED_TAG)?;
    let (entry, rest) = rest.split_once(' ')?;
    let (command, raw) = rest.split_once(' ')?;
    Some((entry.parse().ok()?, command, raw))
}

/// The `Internal` error for a script window's reply that is not this
/// binary's shape (`scripts/fanout_window_read.lua`): attributed to the
/// object's header and the window's `EVALSHA`, naming the entry where there
/// is one. A script/binary mismatch, never a store state.
fn script_window_shape_err(
    object_id: &str,
    header_key: &str,
    entry: Option<&FanoutCommand>,
    what: impl std::fmt::Display,
) -> ServiceError {
    let at = entry
        .map(|command| format!(" at {} ({})", command.op_label(), command.key))
        .unwrap_or_default();
    ServiceError::Internal(format!(
        "fan-out snapshot read of object {object_id}: the {FANOUT_WINDOW_READ_OP} reply \
         ({header_key}) is malformed{at}: {what}"
    ))
}

/// Decode a script window's reply (`{fmt, 'OK', e_1 … e_N}`,
/// `scripts/fanout_window_read.lua`) into each entry's reply, in command
/// order, each with the `FromValue` target the plain issuer reads it with —
/// a section or tombstone hash as `HGETALL`'s map, a list's length as
/// `LLEN`'s count, a ranged list as `LRANGE`'s strings — and a value row as
/// the [`ValueRowReply`] the plain value-row read builds, marked atomic. So
/// the replies, their tallies and the state they decode into are the plain
/// window's by construction.
///
/// Strict: a wrong format tag or status, an arity other than 2 + N, a value
/// row of an arity other than 4 + its columns, a mask with a bit past its
/// columns, a non-empty string under an unset bit, and a follow-up answer
/// present where the follow-up rule ([`ValueRowReply::asks_value_exists`],
/// [`ValueRowReply::asks_row_exists`]) does not ask it, or absent where it
/// does, are each `Internal` ([`script_window_shape_err`]): the script's
/// copy of the rule cannot drift from the plain read's without failing
/// closed.
fn decode_fanout_window_reply(
    object_id: &str,
    header_key: &str,
    window: &[FanoutCommand],
    ops: &[ScriptOp],
    reply: fred::types::Value,
) -> Result<Vec<FanoutReply>, ServiceError> {
    let shape_err = |entry: Option<&FanoutCommand>, what: String| {
        script_window_shape_err(object_id, header_key, entry, what)
    };
    let parts = reply.into_array();
    if parts.len() != 2 + window.len() {
        return Err(shape_err(
            None,
            format!(
                "{} elements for a window of {} entries",
                parts.len(),
                window.len()
            ),
        ));
    }
    let mut parts = parts.into_iter();
    let format = parts.next().and_then(|v| v.into_string());
    let status = parts.next().and_then(|v| v.into_string());
    if format.as_deref() != Some(FANOUT_WINDOW_READ_FORMAT) || status.as_deref() != Some("OK") {
        return Err(shape_err(
            None,
            format!("format {format:?}, status {status:?}"),
        ));
    }
    window
        .iter()
        .zip(ops)
        .zip(parts)
        .map(|((command, op), value)| {
            let entry_err = |what: String| shape_err(Some(command), what);
            match op {
                ScriptOp::Field | ScriptOp::TimeSample => {
                    decode_script_value_row(command.family, value)
                        .map(FanoutReply::ValueRow)
                        .map_err(entry_err)
                }
                ScriptOp::Section | ScriptOp::Tombstone => value
                    .convert::<HashMap<String, String>>()
                    .map(FanoutReply::Hash)
                    .map_err(|e| entry_err(format!("not a flat hash: {e}"))),
                ScriptOp::Length => match value {
                    fred::types::Value::Integer(n) => u64::try_from(n)
                        .map(FanoutReply::Count)
                        .map_err(|_| entry_err(format!("a negative length {n}"))),
                    other => Err(entry_err(format!(
                        "a length that is not an integer: {other:?}"
                    ))),
                },
                ScriptOp::Prefix { .. } => {
                    if !matches!(value, fred::types::Value::Array(_)) {
                        return Err(entry_err(format!(
                            "a range that is not an array: {value:?}"
                        )));
                    }
                    value
                        .convert::<Vec<String>>()
                        .map(FanoutReply::List)
                        .map_err(|e| entry_err(format!("not a list of strings: {e}")))
                }
            }
        })
        .collect()
}

/// Decode one value row's element of a script window's reply,
/// `{value_len, mask, value_exists, row_exists, c_1 … c_k}`
/// ([`decode_fanout_window_reply`]), into the [`ValueRowReply`] the plain
/// value-row read builds, marked atomic. The error is the reason, for the
/// caller's `Internal`.
fn decode_script_value_row(
    family: SnapshotKeyFamily,
    value: fred::types::Value,
) -> Result<ValueRowReply, String> {
    let columns = value_row_columns(family);
    let parts = value.into_array();
    if parts.len() != 4 + columns.len() {
        return Err(format!(
            "a value row of {} elements, not {}",
            parts.len(),
            4 + columns.len()
        ));
    }
    let mut parts = parts.into_iter();
    let mut integer = |what: &str| match parts.next() {
        Some(fred::types::Value::Integer(n)) => Ok(n),
        other => Err(format!("a value row's {what} is not an integer: {other:?}")),
    };
    let value_len = u64::try_from(integer("value length")?)
        .map_err(|_| "a negative value length".to_owned())?;
    let mask = integer("mask")?;
    let value_exists = integer("HEXISTS answer")?;
    let row_exists = integer("EXISTS answer")?;
    let column_count = u32::try_from(columns.len()).unwrap_or(u32::MAX);
    if mask < 0 || (mask >> column_count) != 0 {
        return Err(format!("a mask {mask} past its {} columns", columns.len()));
    }
    let mut row = ValueRowReply {
        family,
        value_len,
        columns: Vec::with_capacity(columns.len()),
        value_exists: None,
        row_exists: None,
        atomic: true,
    };
    for (j, part) in parts.enumerate() {
        let text = part
            .into_string()
            .ok_or_else(|| format!("column {} is not a string", columns[j]))?;
        if mask & (1_i64 << j) != 0 {
            row.columns.push(Some(text));
        } else if text.is_empty() {
            row.columns.push(None);
        } else {
            return Err(format!(
                "column {} is absent by the mask but carries a value",
                columns[j]
            ));
        }
    }
    let follow_up = |asked: bool, answer: i64, what: &str| match (asked, answer) {
        (true, 0) => Ok(Some(false)),
        (true, 1) => Ok(Some(true)),
        (false, -1) => Ok(None),
        (asked, answer) => Err(format!(
            "{what} answered {answer} where the follow-up rule {} it",
            if asked { "asks" } else { "does not ask" }
        )),
    };
    row.value_exists = follow_up(row.asks_value_exists(), value_exists, "HEXISTS")?;
    row.row_exists = follow_up(row.asks_row_exists(), row_exists, "EXISTS")?;
    Ok(row)
}

/// One `SSCAN key cursor COUNT count` step, on one connection of `pool`
/// (routed by the key, like every other fan-out command). fred's own `sscan`
/// pages through the whole set as a stream; round 1 needs one step per
/// window, so the command is sent as itself.
async fn sscan_step(
    pool: &fred::clients::Pool,
    key: &str,
    cursor: &str,
    count: u64,
) -> Result<FanoutReply, fred::error::Error> {
    use fred::interfaces::ClientLike;
    let reply: fred::types::Value = pool
        .next()
        .custom(
            fred::types::CustomCommand::new_static(
                "SSCAN",
                fred::types::ClusterHash::FirstValue,
                false,
            ),
            vec![
                fred::types::Value::from(key),
                fred::types::Value::from(cursor),
                fred::types::Value::from("COUNT"),
                fred::types::Value::from(i64::try_from(count).unwrap_or(i64::MAX)),
            ],
        )
        .await?;
    let shape_err = || {
        fred::error::Error::new(
            fred::error::ErrorKind::Protocol,
            format!("SSCAN {key} returned an unexpected reply"),
        )
    };
    let mut parts = reply.into_array().into_iter();
    let (Some(next), Some(members), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err(shape_err());
    };
    let cursor = next.as_string().ok_or_else(shape_err)?;
    let members: Vec<String> = members.convert()?;
    Ok(FanoutReply::Scan { cursor, members })
}

/// The `Internal` error for a fan-out reply whose shape does not match its
/// command. Unreachable while [`fanout_op`] decides both.
fn fanout_shape_err(object_id: &str, command: &FanoutCommand) -> ServiceError {
    ServiceError::Internal(format!(
        "fan-out snapshot read of object {object_id}: the reply to {} ({}) has the wrong shape",
        command.op_label(),
        command.key
    ))
}

/// The tallies of a fan-out read, kept as the atomic read keeps its own
/// (`object_snapshot_read.lua`): items and bytes of every index, hash and list
/// read (the header excluded, a marked object's `obj:chs` members included),
/// and one entry per hash and per children list read. A value row's
/// two-command read is tallied as its whole row's read would be
/// ([`FanoutReply::tally_bytes`]); a children list's `LLEN` and ranged
/// `LRANGE` as one whole-list read. Beside them, the names the read has
/// taken in ([`NameTally`]).
#[derive(Default)]
struct FanoutTally {
    items: u64,
    bytes: u64,
    entries: u64,
    names: NameTally,
}

/// The names a fan-out read has taken in: the round-1 field-index members and
/// the decoded section names. The bounded-reply read sizes name-bearing
/// replies by [`Self::longest`] and its charge keeps [`Self::bytes`]
/// (`WatchSnapshot::name_bytes`).
#[derive(Default)]
struct NameTally {
    /// The longest name seen, in bytes: a field's key name (half its
    /// member's hex) or a section's name.
    longest: u64,
    /// The bytes the read retains for names: every field-index member's
    /// bytes, and every section name's.
    bytes: u64,
}

impl NameTally {
    /// Fields index members `{section_id}:{hex(key_name)}` taken in.
    fn field_members(&mut self, members: &[String]) {
        for member in members {
            let hex = member.split_once(':').map_or("", |(_, hex)| hex);
            self.longest = self.longest.max(hex.len() as u64 / 2);
            self.bytes = self.bytes.saturating_add(member.len() as u64);
        }
    }

    /// A decoded section name taken in.
    fn section_name(&mut self, name: &str) {
        self.longest = self.longest.max(name.len() as u64);
        self.bytes = self.bytes.saturating_add(name.len() as u64);
    }
}

impl FanoutTally {
    fn index(&mut self, reply: &FanoutReply) {
        self.items += reply.items();
        self.bytes += reply.tally_bytes();
    }

    fn entry(&mut self, reply: &FanoutReply) {
        self.index(reply);
        self.entries += 1;
    }

    /// One whole index, read in round 1's pages and collected: its members
    /// tallied as one whole-index reply would be.
    fn index_members(&mut self, members: &[String]) {
        self.items += members.len() as u64;
        self.bytes += members.iter().map(|m| m.len() as u64).sum::<u64>();
    }
}

/// The entries a fan-out read has taken in so far, decoded by the decoders
/// the atomic read shares, so the readers turn the same replies into the
/// same state; the value rows into `S`'s form, whole or compact. Each reply
/// is decoded as its window completes and the raw reply dropped, so no raw
/// reply outlives its window.
struct FanoutEntries<S: SnapshotState> {
    sections: HashMap<u64, Section>,
    tombstones: Vec<(u64, u64, u64)>,
    value_rows: <S as sealed::SnapshotDecode>::Rows,
    children: HashMap<u64, Vec<u64>>,
    /// The section members with no readable hash or a non-decimal id, in
    /// index order, when the read records a corrupt index
    /// ([`FanoutRead::records_corrupt_index`]); `None` when it does not.
    corrupt_section_index_members: Option<Vec<String>>,
    /// The bounded-reply read's non-empty children lists and their lengths,
    /// from round 2's `LLEN`s: `(parent id, length)`, round 3's ranged
    /// `LRANGE`s. Empty on the whole-row read.
    list_lengths: Vec<(String, u64)>,
    /// [`AtomicSnapshotOf::value_row_bytes_received`] so far.
    #[cfg(any(test, feature = "test-support"))]
    value_row_bytes_received: u64,
}

impl<S: SnapshotState> FanoutEntries<S> {
    /// Take in one round-2 or round-3 reply and tally it. Returns the members
    /// of a marked object's `obj:chs`, which name round 3's lists on the
    /// whole-row read, and `None` for every other reply.
    fn take_in(
        &mut self,
        object_id: &str,
        command: FanoutCommand,
        reply: FanoutReply,
        tally: &mut FanoutTally,
    ) -> Result<Option<Vec<String>>, ServiceError> {
        match (command.family, &command.page, &reply) {
            (SnapshotKeyFamily::ChildrenIndex, _, _) => tally.index(&reply),
            // A list's length is tallied as its whole-list read would be:
            // here for an empty list, which reads no range; by its range in
            // round 3 for any other.
            (_, FanoutPage::Length, FanoutReply::Count(0)) => tally.entry(&reply),
            (_, FanoutPage::Length, _) => {}
            _ => tally.entry(&reply),
        }
        match (command.family, reply) {
            (SnapshotKeyFamily::Section, FanoutReply::Hash(data)) => {
                let decoded = decode_section(&command.id, &data);
                if let Some(corrupt) = &mut self.corrupt_section_index_members {
                    // The atomic decode's rule, for the same reason: under a
                    // fence an indexed section always has its hash and a
                    // decimal id.
                    if decoded.is_none() || command.id.parse::<u64>().is_err() {
                        corrupt.push(command.id.clone());
                    }
                }
                if let Some(section) = decoded {
                    tally.names.section_name(&section.name);
                    self.sections.insert(section.section_id, section);
                }
            }
            (SnapshotKeyFamily::DeletedSection, FanoutReply::Hash(data)) => {
                if let Some(tombstone) = decode_tombstone(&command.id, &data) {
                    self.tombstones.push(tombstone);
                }
            }
            (SnapshotKeyFamily::Field, FanoutReply::Hash(data)) => {
                #[cfg(any(test, feature = "test-support"))]
                {
                    self.value_row_bytes_received += flat_hash_bytes(&data);
                }
                // Member format: "{section_id}:{hex(key_name)}"
                let (sid, hex_key) = split_member(&command.id);
                S::push_field(
                    &mut self.value_rows,
                    object_id,
                    &sid,
                    keys::hex_decode(&hex_key),
                    &data,
                )?;
            }
            (SnapshotKeyFamily::TimeSample, FanoutReply::Hash(data)) => {
                #[cfg(any(test, feature = "test-support"))]
                {
                    self.value_row_bytes_received += flat_hash_bytes(&data);
                }
                // Member format: "{section_id}:{hex_f64(time)}"
                let (sid, time_hex) = split_member(&command.id);
                S::push_time_sample(
                    &mut self.value_rows,
                    object_id,
                    &sid,
                    keys::hex_decode_f64(&time_hex),
                    &data,
                )?;
            }
            // The bounded-reply read's two-command reads: decoded by shape.
            (SnapshotKeyFamily::Field, FanoutReply::ValueRow(row)) => {
                #[cfg(any(test, feature = "test-support"))]
                {
                    self.value_row_bytes_received += row.value_shape_bytes();
                }
                // Member format: "{section_id}:{hex(key_name)}"
                let (sid, hex_key) = split_member(&command.id);
                S::push_field_shape(
                    &mut self.value_rows,
                    object_id,
                    &sid,
                    keys::hex_decode(&hex_key),
                    &row.shape(),
                )?;
            }
            (SnapshotKeyFamily::TimeSample, FanoutReply::ValueRow(row)) => {
                #[cfg(any(test, feature = "test-support"))]
                {
                    self.value_row_bytes_received += row.value_shape_bytes();
                }
                // Member format: "{section_id}:{hex_f64(time)}"
                let (sid, time_hex) = split_member(&command.id);
                S::push_time_sample_shape(
                    &mut self.value_rows,
                    object_id,
                    &sid,
                    keys::hex_decode_f64(&time_hex),
                    &row.shape(),
                )?;
            }
            (SnapshotKeyFamily::Children, FanoutReply::List(child_strs)) => {
                if let Some((pid, child_ids)) = decode_children_list(&command.id, &child_strs) {
                    self.children.insert(pid, child_ids);
                }
            }
            // The bounded-reply read's list lengths: a non-empty list is read
            // by its range in round 3.
            (SnapshotKeyFamily::Children, FanoutReply::Count(len)) => {
                if len > 0 {
                    self.list_lengths.push((command.id, len));
                }
            }
            (SnapshotKeyFamily::ChildrenIndex, FanoutReply::List(members)) => {
                return Ok(Some(members));
            }
            _ => return Err(fanout_shape_err(object_id, &command)),
        }
        Ok(None)
    }
}

/// What round 1 of the bounded fan-out read: the header and the four
/// indices' members — and on the bounded-reply read of a marked object,
/// `obj:chs`'s — each index whole and, for the sets, without repeats.
struct RoundOne {
    header: HashMap<String, String>,
    section_ids: Vec<String>,
    deleted_section_ids: Vec<String>,
    field_members: Vec<String>,
    time_sample_members: Vec<String>,
    /// `obj:chs`'s members, read in round 1 by the bounded-reply read of a
    /// marked object only; `None` otherwise (the whole-row read reads a
    /// marked object's `obj:chs` in round 2).
    children_index_ids: Option<Vec<String>>,
}

/// Whether `header` marks its object's `obj:chs` complete (`chs_complete`,
/// SDD §7.2): the readers then take the children lists from it alone.
fn chs_marked(header: &HashMap<String, String>) -> bool {
    header
        .get(crate::storage::layer::HEADER_FIELD_CHS_COMPLETE)
        .map(String::as_str)
        == Some(crate::storage::layer::CHS_COMPLETE_MARKED)
}

/// Each share of a bounded-reply round-1 page (`shares`, members per index,
/// from [`share_index_window`]) cut so the page's summed member bound —
/// share × each index's per-member bound (`bounds`) — stays within
/// `max_bytes`: every share scaled down by the same factor, rounded down.
/// Unchanged when the page already fits. A page always makes progress: if
/// every share would round to nothing, the first index with a share keeps
/// one member (one member's bound alone can exceed only a tiny configured
/// byte window).
fn cap_shares_to_bytes(shares: Vec<u64>, bounds: &[u64], max_bytes: u64) -> Vec<u64> {
    let total: u128 = shares
        .iter()
        .zip(bounds)
        .map(|(&share, &bound)| u128::from(share) * u128::from(bound))
        .sum();
    if total <= u128::from(max_bytes) {
        return shares;
    }
    let mut capped: Vec<u64> = shares
        .iter()
        .map(|&share| {
            u64::try_from(u128::from(share) * u128::from(max_bytes) / total).unwrap_or(u64::MAX)
        })
        .collect();
    if capped.iter().all(|&share| share == 0) {
        if let Some(first) = shares.iter().position(|&share| share > 0) {
            capped[first] = 1;
        }
    }
    capped
}

/// One set index being scanned in round 1 (`SSCAN`).
struct SetPages {
    family: SnapshotKeyFamily,
    key: String,
    /// `SCARD`'s answer: what the scan sizes its steps from.
    count: u64,
    /// The cursor to continue from; `None` once the scan is complete.
    cursor: Option<String>,
    /// The members collected, in the order the steps returned them.
    members: Vec<String>,
    /// The members already collected: `SSCAN` may return one more than once
    /// (when the set is rehashed between steps). Dropped with round 1.
    seen: std::collections::HashSet<String>,
}

/// One sorted-set index being read by rank in round 1 (`ZRANGE`).
struct RankPages {
    family: SnapshotKeyFamily,
    key: String,
    /// `ZCARD`'s answer: the ranks to read are `0..count`.
    count: u64,
    /// The next rank to read.
    next: u64,
    /// The members collected, in rank order.
    members: Vec<String>,
}

/// How many members of a page each still-unread index gets in the next
/// round-1 window: `budget` shared out to `demands` (what each index still
/// asks for; [`fanout_round_one`] says how much) by water-filling — every index
/// gets what it asks for when the asks fit, and otherwise the smallest asks
/// are met in full and the rest share the remainder equally. Every positive
/// ask gets at least one member while `budget` lasts, so a window always
/// makes progress.
fn share_index_window(budget: u64, demands: &[u64]) -> Vec<u64> {
    let mut order: Vec<usize> = (0..demands.len()).filter(|&i| demands[i] > 0).collect();
    order.sort_by_key(|&i| demands[i]);
    let mut shares = vec![0; demands.len()];
    let mut left = budget;
    let mut open = order.len() as u64;
    for i in order {
        if left == 0 {
            break;
        }
        let fair = (left / open).max(1);
        let share = demands[i].min(fair);
        shares[i] = share;
        left -= share;
        open -= 1;
    }
    shares
}

/// Round 1 of the bounded fan-out (Spec SDD SR-11.3): the header and the four
/// indices whose members name round 2's commands, in windows bounded in index
/// members by `config::FANOUT_INDEX_WINDOW_MEMBERS` (see that constant for
/// why: read whole, round 1's window grows with the layer, out of reach of
/// the command cap).
///
/// - The first window is the header `HGETALL` and the four member counts
///   (`SCARD` of the sections and deleted-sections sets, `ZCARD` of the
///   fields and time-samples sorted sets).
/// - Then each window reads up to the bound in members, shared among the
///   indices still unread ([`share_index_window`]): a `ZRANGE key start stop`
///   page by rank for a sorted index, and one `SSCAN key cursor COUNT n` step
///   for a set, asked for half its share because `COUNT` is approximate (a
///   step can return more than it). A set repeats no member in what this
///   returns: a step's repeats of earlier members are dropped. A set's scan
///   ends when its cursor returns to `0`.
///
/// On the bounded-reply read ([`FanoutReplies::Bounded`]) a marked object's
/// `obj:chs` is counted (`SCARD`) in a window of its own after the first —
/// only once the header shows the object marked, so an unmarked object's
/// `obj:chs` is never read — and paged with the sets; and each page is also
/// bounded in bytes, its shares cut so that share × each index's per-member
/// bound ([`FanoutWindows::index_member_bound`]) sums within
/// `FANOUT_WINDOW_BYTES` ([`cap_shares_to_bytes`]). A set's step is charged
/// for its whole share, at least what it asks for, and at least twice that
/// for a share above one (a share cut to one asks `COUNT 1`); a set small
/// enough for the store's compact encoding can still answer more (its whole
/// membership, up to that encoding's limit of ids: see
/// `config::FANOUT_INDEX_WINDOW_MEMBERS`), and a hashtable-encoded set's step
/// can return a few members past its share (the store adds whole bucket
/// chains until it has `COUNT`), both of which the growth after the page
/// charges.
///
/// Every window goes through [`FanoutWindows::run_window`], so the test-only
/// seam counts these commands in issue order and every window logs its
/// `fan-out window` event. The members are kept, whole, for round 2, as the
/// whole-index reads kept them; memory is unchanged but for a set's de-dup
/// set, dropped here. The fields index members' names are taken into the
/// read's name tally as each page completes.
///
/// Read under a fence, nothing changes between windows. Read unfenced, a
/// commit landing between windows can tear the index read (a member missed
/// or read twice); the structural consumer's header proof (SR-4.3) rejects
/// such a read whole.
///
/// On a charged read the allowance grows the call's charge after the count
/// windows, before any member is read (WA1), and before and after every
/// page; `Ok(None)` when a growth was refused, the stop in
/// [`FanoutWindows::stopped`].
async fn fanout_round_one(
    windows: &mut FanoutWindows<'_>,
    keys: &KeySchema,
    object_id: &str,
) -> Result<Option<RoundOne>, ServiceError> {
    let mut header = HashMap::new();
    let mut counts: Vec<(SnapshotKeyFamily, u64)> = Vec::with_capacity(4);
    windows
        .run_window(
            vec![
                FanoutCommand::new(SnapshotKeyFamily::Header, keys.obj_header(object_id), ""),
                FanoutCommand::index_part(
                    SnapshotKeyFamily::SectionsIndex,
                    keys.obj_sections_index(object_id),
                    FanoutPage::Count,
                ),
                FanoutCommand::index_part(
                    SnapshotKeyFamily::DeletedSectionsIndex,
                    keys.obj_deleted_sections_index(object_id),
                    FanoutPage::Count,
                ),
                FanoutCommand::index_part(
                    SnapshotKeyFamily::FieldsIndex,
                    keys.obj_fields_index(object_id),
                    FanoutPage::Count,
                ),
                FanoutCommand::index_part(
                    SnapshotKeyFamily::TimeSamplesIndex,
                    keys.obj_time_samples_index(object_id),
                    FanoutPage::Count,
                ),
            ],
            WindowCall::Plain,
            &mut |_, command, reply| {
                match (command.family, reply) {
                    (SnapshotKeyFamily::Header, FanoutReply::Hash(h)) => header = h,
                    (family, FanoutReply::Count(n)) => counts.push((family, n)),
                    _ => return Err(fanout_shape_err(object_id, &command)),
                }
                Ok(())
            },
        )
        .await?;
    let count = |family: SnapshotKeyFamily| {
        counts
            .iter()
            .find(|(f, _)| *f == family)
            .map_or(0, |(_, n)| *n)
    };
    // The bounded-reply read of a marked object pages `obj:chs` here, so it
    // counts it first, in a window of its own: only once the header shows
    // the object marked, so an unmarked object's `obj:chs` is never read.
    let pages_children_index =
        windows.replies == FanoutReplies::Bounded && !header.is_empty() && chs_marked(&header);
    let mut children_index_count = 0u64;
    if pages_children_index {
        windows
            .run_window(
                vec![FanoutCommand::index_part(
                    SnapshotKeyFamily::ChildrenIndex,
                    keys.obj_children_index(object_id),
                    FanoutPage::Count,
                )],
                WindowCall::Plain,
                &mut |_, command, reply| match reply {
                    FanoutReply::Count(n) => {
                        children_index_count = n;
                        Ok(())
                    }
                    _ => Err(fanout_shape_err(object_id, &command)),
                },
            )
            .await?;
    }
    // WA1: the read's whole count-based need, before any member is read. An
    // absent header counts nothing, and its read ends below.
    if !header.is_empty()
        && !windows
            .allowance_counted([
                count(SnapshotKeyFamily::SectionsIndex),
                count(SnapshotKeyFamily::DeletedSectionsIndex),
                count(SnapshotKeyFamily::FieldsIndex),
                count(SnapshotKeyFamily::TimeSamplesIndex),
                children_index_count,
            ])
            .await
    {
        return Ok(None);
    }

    // An absent header ends the read before any page: the caller answers
    // `NotFound` from the empty header, as it did when round 1 was one window.
    let mut set_indices = vec![
        (
            SnapshotKeyFamily::SectionsIndex,
            keys.obj_sections_index(object_id),
            count(SnapshotKeyFamily::SectionsIndex),
        ),
        (
            SnapshotKeyFamily::DeletedSectionsIndex,
            keys.obj_deleted_sections_index(object_id),
            count(SnapshotKeyFamily::DeletedSectionsIndex),
        ),
    ];
    if pages_children_index {
        set_indices.push((
            SnapshotKeyFamily::ChildrenIndex,
            keys.obj_children_index(object_id),
            children_index_count,
        ));
    }
    let mut sets: Vec<SetPages> = set_indices
        .into_iter()
        .map(|(family, key, n)| SetPages {
            family,
            key,
            count: n,
            cursor: (n > 0 && !header.is_empty()).then(|| "0".to_owned()),
            members: Vec::new(),
            seen: std::collections::HashSet::new(),
        })
        .collect();
    let mut ranks = [
        (
            SnapshotKeyFamily::FieldsIndex,
            keys.obj_fields_index(object_id),
        ),
        (
            SnapshotKeyFamily::TimeSamplesIndex,
            keys.obj_time_samples_index(object_id),
        ),
    ]
    .map(|(family, key)| RankPages {
        family,
        key,
        count: if header.is_empty() { 0 } else { count(family) },
        next: 0,
        members: Vec::new(),
    });

    let budget = crate::config::FANOUT_INDEX_WINDOW_MEMBERS.max(2);
    loop {
        // What each index still asks for: a sorted index its unread ranks; a
        // set, while its scan is open, twice its whole count (its `SSCAN`
        // asks for half its share). Not just the members still to collect:
        // a step walks the set's table only until it has collected its
        // `COUNT` or walked ten buckets per member asked, so a step asked for
        // the few members left would crawl through the table's empty tail
        // one short window at a time. Asked for up to the whole count, the
        // step that collects the last members also reaches the scan's end.
        let demands: Vec<u64> = sets
            .iter()
            .map(|set| match set.cursor {
                Some(_) => set.count.max(1).saturating_mul(2),
                None => 0,
            })
            .chain(ranks.iter().map(|r| r.count.saturating_sub(r.next)))
            .collect();
        if demands.iter().all(|&d| d == 0) {
            break;
        }
        let mut shares = share_index_window(budget, &demands);
        if windows.replies == FanoutReplies::Bounded {
            // Within the byte window too, at each index's per-member bound;
            // then charged before the page is issued, for its bound and the
            // members it can add beside those collected so far.
            let bounds: Vec<u64> = sets
                .iter()
                .map(|set| set.family)
                .chain(ranks.iter().map(|r| r.family))
                .map(|family| windows.index_member_bound(family))
                .collect();
            shares = cap_shares_to_bytes(shares, &bounds, windows.max_bytes);
            let bound: u64 = shares
                .iter()
                .zip(&bounds)
                .map(|(&share, &member)| share.saturating_mul(member))
                .fold(0, u64::saturating_add);
            let items = collected_members(&sets, &ranks)
                .saturating_add(shares.iter().copied().fold(0, u64::saturating_add));
            // A page is plain commands, one per index with a share.
            let page_commands = shares.iter().filter(|&&share| share > 0).count() as u64;
            if !windows.allowance_grow(items, bound, page_commands).await {
                return Ok(None);
            }
            windows.window_bound = Some(bound);
        }
        let bytes_before = windows.bytes_seen;
        let fields_before = ranks[0].members.len();
        let mut window = Vec::new();
        for (set, &share) in sets.iter().zip(&shares) {
            if let (Some(cursor), true) = (&set.cursor, share > 0) {
                window.push(FanoutCommand::index_part(
                    set.family,
                    set.key.clone(),
                    FanoutPage::Scan {
                        cursor: cursor.clone(),
                        count: (share / 2).max(1),
                    },
                ));
            }
        }
        for (r, &share) in ranks.iter().zip(&shares[sets.len()..]) {
            if share > 0 {
                window.push(FanoutCommand::index_part(
                    r.family,
                    r.key.clone(),
                    FanoutPage::Ranks {
                        start: r.next,
                        stop: r.next + share - 1,
                    },
                ));
            }
        }
        windows
            .run_window(window, WindowCall::Plain, &mut |_, command, reply| {
                match (&command.page, reply) {
                    (FanoutPage::Scan { .. }, FanoutReply::Scan { cursor, members }) => {
                        let Some(set) = sets.iter_mut().find(|s| s.family == command.family) else {
                            return Err(fanout_shape_err(object_id, &command));
                        };
                        for member in members {
                            if set.seen.insert(member.clone()) {
                                set.members.push(member);
                            }
                        }
                        set.cursor = (cursor != "0").then_some(cursor);
                    }
                    (FanoutPage::Ranks { start, stop }, FanoutReply::List(members)) => {
                        let Some(r) = ranks.iter_mut().find(|r| r.family == command.family) else {
                            return Err(fanout_shape_err(object_id, &command));
                        };
                        let asked = stop - start + 1;
                        let short = (members.len() as u64) < asked;
                        r.members.extend(members);
                        // A short page: the index shrank under an unfenced
                        // read (a torn read the caller's proof rejects). Stop
                        // at what is there rather than ask past its end.
                        r.next = if short { r.count } else { stop + 1 };
                    }
                    _ => return Err(fanout_shape_err(object_id, &command)),
                }
                Ok(())
            })
            .await?;
        // The page's field names, then the charge for what it received and
        // the names retained, before the next page.
        windows
            .tally
            .names
            .field_members(&ranks[0].members[fields_before..]);
        let items = collected_members(&sets, &ranks);
        let received = windows.bytes_seen - bytes_before;
        if !windows.allowance_grow(items, received, 0).await {
            return Ok(None);
        }
    }

    let mut set_members = sets
        .into_iter()
        .map(|set| (set.family, set.members))
        .collect::<Vec<_>>();
    let mut take_set = |family: SnapshotKeyFamily| {
        set_members
            .iter_mut()
            .find(|(f, _)| *f == family)
            .map(|(_, members)| std::mem::take(members))
    };
    let sections = take_set(SnapshotKeyFamily::SectionsIndex).unwrap_or_default();
    let deleted_sections = take_set(SnapshotKeyFamily::DeletedSectionsIndex).unwrap_or_default();
    let children_index_ids = take_set(SnapshotKeyFamily::ChildrenIndex);
    let [fields, time_samples] = ranks.map(|r| r.members);
    Ok(Some(RoundOne {
        header,
        section_ids: sections,
        deleted_section_ids: deleted_sections,
        field_members: fields,
        time_sample_members: time_samples,
        children_index_ids,
    }))
}

/// The index members round 1 has collected so far, over every index.
fn collected_members(sets: &[SetPages], ranks: &[RankPages]) -> u64 {
    sets.iter()
        .map(|set| set.members.len() as u64)
        .chain(ranks.iter().map(|r| r.members.len() as u64))
        .fold(0, u64::saturating_add)
}

/// The unfenced bounded fan-out path of [`read_object_snapshot`] (see the
/// section comment above).
///
/// Returns the object's state with `path = Unfenced` and `cut` = the header
/// `seq` its first round read — unproven: a commit can land between its
/// rounds, and a caller that needs the state at one `seq` proves it itself.
/// [`AtomicSnapshot::structural`] carries the header's `chs_complete` column
/// and, on a marked object, the parents of the stored lists the read left
/// out. `NotFound` when the header is absent or holds no numeric `seq`. A
/// failed command fails the read, classified under
/// its own key family by [`classify_snapshot_read_err`] (transient →
/// `Aborted`, permanent → `Internal`, refusal counters ticked). Its replies
/// are bounded ([`FanoutReplies::Bounded`]): the structural update's read is
/// the only unfenced fan-out, and a consumer that takes the atomic path never
/// reaches it. With an `allowance` (the structural update's charged read only)
/// it can also stop before a window ([`FanoutOutcome::Stopped`]).
///
/// The pass runs on the `watch-read` thread that was least loaded when it
/// started ([`PassLane::Unfenced`];
/// [`FanoutThreads::run_pass`](crate::storage::fanout_thread::FanoutThreads::run_pass)),
/// with owned clones of `client`, `keys` and `object_id` — never on
/// `fanout-read`, where the fenced passes run, nor on a service worker.
/// `allowance` stays here, in the caller's task, and answers the pass's seams in the
/// order the pass reaches them — WA1's count charge, each growth before and
/// after a window or index page, the retained name bytes at the end
/// ([`answer_allowance`]) — the pass issuing nothing until the answer
/// arrives. Its progress goes to a local no-op, as it always has.
async fn read_object_snapshot_unfenced<S: SnapshotState>(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    mut allowance: Option<&mut FanoutAllowance<'_>>,
) -> Result<FanoutOutcome<S>, ServiceError> {
    let charged = allowance.is_some();
    let (pass_client, pass_keys, pass_object_id) =
        (client.clone(), keys.clone(), object_id.to_owned());
    client
        .fanout_threads()
        .run_pass(
            PassLane::Unfenced,
            move |link| async move {
                let mut no_progress = |_: u64| {};
                let allowance = if charged {
                    AllowanceSeam::Caller {
                        link: &link,
                        stopped: None,
                    }
                } else {
                    AllowanceSeam::None
                };
                fanout_snapshot_read::<S>(
                    &pass_client,
                    &pass_keys,
                    &pass_object_id,
                    FanoutRead::Unfenced,
                    ProgressSink::Local(&mut no_progress),
                    FanoutReplies::Bounded,
                    allowance,
                )
                .await
            },
            |request| answer_allowance(allowance.as_deref_mut(), request),
        )
        .await
}

/// The charged read's caller's answer to one of its pass's requests
/// ([`read_object_snapshot_unfenced`]): the seam applied to `allowance`
/// exactly as the pass once applied it inline. A refused growth answers the
/// stop with the target the allowance recorded. Every request goes on when
/// there is no allowance (an uncharged pass asks none).
fn answer_allowance(
    allowance: Option<&mut FanoutAllowance<'_>>,
    request: PassRequest,
) -> PassAnswer {
    let Some(allowance) = allowance else {
        return PassAnswer::GoOn;
    };
    let goes_on = match request {
        PassRequest::Counted {
            counts,
            max_commands,
            max_bytes,
        } => allowance.counted(counts, max_commands, max_bytes),
        PassRequest::Grow {
            items,
            window_bytes,
            name_bytes,
            plain_commands,
        } => allowance.grow(items, window_bytes, name_bytes, plain_commands),
        PassRequest::NameBytes { name_bytes } => {
            allowance.name_bytes = allowance.name_bytes.max(name_bytes);
            true
        }
        // The unfenced pass reports its progress locally.
        PassRequest::Progress { .. } => true,
    };
    if goes_on {
        PassAnswer::GoOn
    } else {
        PassAnswer::Stop {
            target: allowance.stopped.unwrap_or_default(),
        }
    }
}

/// How [`fanout_snapshot_read`] ended without an error.
enum FanoutOutcome<S: SnapshotState> {
    /// The whole object.
    Read(Box<AtomicSnapshotOf<S>>),
    /// The charged read stopped before a window ([`FanoutAllowance`]); only
    /// a read with an allowance answers it.
    Stopped { target: u64 },
}

/// One pass of the fenced read's fan-out (Spec SDD SR-5.3), for
/// `storage::snapshot_fence::read`, which holds the lease around it and alone
/// proves it.
///
/// The unfenced fan-out's commands, windows and decode on the snapshot lane,
/// returning the state with `path = Fenced` and `cut` = the header `seq` its
/// first round read. That `cut` is **not** proven here: the caller compares it
/// with the `seq` its fence acquire read and proves the pass with its release.
/// `NotFound` when the header is absent or has no numeric `seq`. Indexed
/// sections without a readable hash or with a non-decimal id are reported on
/// [`AtomicSnapshot::corrupt_section_index_members`], as on the atomic path.
/// A failed command fails the pass, classified as on the unfenced fan-out;
/// the test-only seam counts its commands as
/// [`ObjectSnapshotReader::FencedFanout`] (in test-fault-injection builds).
///
/// Runs on the `fanout-read` thread: the caller spawns it there through
/// [`FanoutThreads::run_pass`](crate::storage::fanout_thread::FanoutThreads::run_pass)
/// ([`PassLane::Fenced`]), with owned clones of the client, the key schema and the object id, and
/// `link` is its way back. The pass reports the reply bytes tallied so far
/// over `link` ([`PassRequest::Progress`]) — once after round 1 (the index
/// members) and once after every later window — ending with the pass's
/// `reply_bytes` when it completes, and issues nothing more until the caller
/// has applied each report. The caller's lease renewal runs beside it, in
/// the caller's task.
pub(crate) async fn read_object_snapshot_fenced_pass<S: SnapshotState>(
    client: ValkeyClient,
    keys: KeySchema,
    object_id: String,
    link: PassLink,
) -> Result<AtomicSnapshotOf<S>, ServiceError> {
    match fanout_snapshot_read::<S>(
        &client,
        &keys,
        &object_id,
        FanoutRead::Fenced,
        ProgressSink::Caller(&link),
        FanoutReplies::Whole,
        AllowanceSeam::None,
    )
    .await?
    {
        FanoutOutcome::Read(snapshot) => Ok(*snapshot),
        FanoutOutcome::Stopped { .. } => uncharged(&object_id, Err(0)),
    }
}

/// The bounded fan-out both [`read_object_snapshot_unfenced`] and
/// [`read_object_snapshot_fenced_pass`] run, as a pass on a fan-out thread
/// (the fenced pass on `fanout-read`, the unfenced on a `watch-read` thread);
/// `read` says which, and `progress` is where it reports the reply bytes
/// tallied so far ([`FanoutWindows::report_progress`]); `replies` its reply
/// shape ([`FanoutReplies`]). `allowance`, the structural update's charged
/// read's only, reaches the caller's [`FanoutAllowance`], which grows the
/// call's upload charge as the read goes and can stop it before a window;
/// after a whole read the caller's allowance holds the name bytes the read
/// retained.
async fn fanout_snapshot_read<'a, S: SnapshotState>(
    client: &'a ValkeyClient,
    keys: &'a KeySchema,
    object_id: &'a str,
    read: FanoutRead,
    progress: ProgressSink<'a>,
    replies: FanoutReplies,
    allowance: AllowanceSeam<'a>,
) -> Result<FanoutOutcome<S>, ServiceError> {
    let oid = object_id.parse::<u64>().map_err(|e| {
        ServiceError::Internal(format!(
            "read_object_snapshot called with non-numeric object_id '{object_id}': {e}"
        ))
    })?;
    // Held for the whole read: the in-use gauge counts snapshot reads in
    // flight on the lane.
    let (pool, _in_use) = client.acquire_snapshot();
    let mut windows = FanoutWindows::new(
        client,
        pool,
        keys,
        object_id,
        read.fault_cursor(),
        progress,
        replies,
        allowance,
    );

    // Round 1: the header and the four indices, collected: round 2's commands
    // are built from them. Bounded in index members per window
    // (`fanout_round_one`). A charged read can stop here, before the members
    // or before a later page.
    let Some(RoundOne {
        header,
        section_ids,
        deleted_section_ids,
        field_members,
        time_sample_members,
        children_index_ids,
    }) = fanout_round_one(&mut windows, keys, object_id).await?
    else {
        return Ok(stopped(&windows));
    };
    for members in [
        &section_ids,
        &deleted_section_ids,
        &field_members,
        &time_sample_members,
    ] {
        windows.tally.index_members(members);
    }
    // A marked object's `obj:chs`, paged in round 1 by the bounded-reply
    // read: tallied as the whole-row read's `SMEMBERS` would be.
    if let Some(members) = &children_index_ids {
        windows.tally.index_members(members);
    }
    windows.report_progress().await;

    if header.is_empty() {
        return Err(ServiceError::NotFound(format!(
            "object {object_id} not found"
        )));
    }
    let Some(cut) = header
        .get(crate::storage::layer::HEADER_FIELD_SEQ)
        .and_then(|s| s.parse::<u64>().ok())
    else {
        return Err(ServiceError::NotFound(format!(
            "object {object_id} header partial (missing or non-numeric seq); treating as \
             not-found"
        )));
    };
    let marked = chs_marked(&header);
    let bounded = windows.replies == FanoutReplies::Bounded;
    // The live parents: the root and every live section.
    let live: std::collections::HashSet<&str> = section_ids
        .iter()
        .map(String::as_str)
        .chain([ROOT_PARENT_ID])
        .collect();

    // Round 2: every section, tombstone, field and time-sample hash, then the
    // children: an unmarked object's lists for the root and every live section
    // (the atomic script's parent set for it), or a marked object's `obj:chs`
    // — on the bounded-reply read, the length of every list a marked
    // object's `obj:chs` (read in round 1) names under a live parent, or an
    // unmarked object's, and value rows by their two-command read.
    //
    // A marked object's lists under dangling parents stay out of the state
    // either way (SR-1.2). Their parents are reported instead, in
    // `StructuralSnapshotFacts::dangling_parents`: only an earlier build
    // stored such a list, and the structural commit path deletes it when a
    // commit creates its parent.
    let mut dangling_parents: Vec<u64> = Vec::new();
    let value_row_page = if bounded {
        FanoutPage::ValueRow
    } else {
        FanoutPage::Whole
    };
    let mut commands = Vec::with_capacity(
        section_ids.len()
            + deleted_section_ids.len()
            + field_members.len()
            + time_sample_members.len()
            + section_ids.len()
            + 1,
    );
    for sid in &section_ids {
        commands.push(FanoutCommand::new(
            SnapshotKeyFamily::Section,
            keys.obj_section(object_id, sid),
            sid,
        ));
    }
    for sid in &deleted_section_ids {
        commands.push(FanoutCommand::new(
            SnapshotKeyFamily::DeletedSection,
            keys.obj_deleted_section(object_id, sid),
            sid,
        ));
    }
    // Index members name their hash keys directly, as the atomic read builds
    // them (`fld:` / `ts:` + the member).
    for member in &field_members {
        commands.push(FanoutCommand::part(
            SnapshotKeyFamily::Field,
            keys.obj_field_from_member(object_id, member),
            member,
            value_row_page.clone(),
        ));
    }
    for member in &time_sample_members {
        commands.push(FanoutCommand::part(
            SnapshotKeyFamily::TimeSample,
            keys.obj_ts_from_member(object_id, member),
            member,
            value_row_page.clone(),
        ));
    }
    if let Some(children_index) = &children_index_ids {
        // The bounded-reply read of a marked object: the lengths of the
        // lists `obj:chs` names under a live parent.
        for pid in children_index {
            if live.contains(pid.as_str()) {
                commands.push(FanoutCommand::part(
                    SnapshotKeyFamily::Children,
                    keys.obj_children(object_id, pid),
                    pid,
                    FanoutPage::Length,
                ));
            } else if let Ok(parent_id) = pid.parse::<u64>() {
                dangling_parents.push(parent_id);
            }
        }
    } else if marked {
        commands.push(FanoutCommand::new(
            SnapshotKeyFamily::ChildrenIndex,
            keys.obj_children_index(object_id),
            "",
        ));
    } else {
        let list_page = if bounded {
            FanoutPage::Length
        } else {
            FanoutPage::Whole
        };
        for pid in section_ids
            .iter()
            .map(String::as_str)
            .chain([ROOT_PARENT_ID])
        {
            commands.push(FanoutCommand::part(
                SnapshotKeyFamily::Children,
                keys.obj_children(object_id, pid),
                pid,
                list_page.clone(),
            ));
        }
    }

    let mut entries = FanoutEntries::<S> {
        sections: HashMap::new(),
        tombstones: Vec::new(),
        value_rows: S::rows_with_capacity(field_members.len(), time_sample_members.len()),
        children: HashMap::new(),
        corrupt_section_index_members: read.records_corrupt_index().then(Vec::new),
        list_lengths: Vec::new(),
        #[cfg(any(test, feature = "test-support"))]
        value_row_bytes_received: 0,
    };
    let mut children_index: Vec<String> = Vec::new();
    windows
        .run(commands, |tally, command, reply| {
            if let Some(members) = entries.take_in(object_id, command, reply, tally)? {
                children_index = members;
            }
            Ok(())
        })
        .await?;
    if windows.stopped().is_some() {
        return Ok(stopped(&windows));
    }

    // Round 3 on the bounded-reply read: every list whose round-2 length was
    // not 0, read by the range that length names. A list that grew in
    // between is cut at that length, inside a read whose proof then fails.
    if bounded {
        let lists: Vec<FanoutCommand> = std::mem::take(&mut entries.list_lengths)
            .into_iter()
            .map(|(pid, len)| {
                FanoutCommand::part(
                    SnapshotKeyFamily::Children,
                    keys.obj_children(object_id, &pid),
                    &pid,
                    FanoutPage::Prefix { len },
                )
            })
            .collect();
        windows
            .run(lists, |tally, command, reply| {
                entries
                    .take_in(object_id, command, reply, tally)
                    .map(|_| ())
            })
            .await?;
        if windows.stopped().is_some() {
            return Ok(stopped(&windows));
        }
    }

    // Round 3 on the whole-row read, marked objects only: the lists `obj:chs`
    // names whose parent is the root or a live section. A marked object's
    // index is complete, so that is every list under a live parent; the
    // others' parents are reported as dangling (above).
    if marked && !bounded {
        let mut lists: Vec<FanoutCommand> = Vec::with_capacity(children_index.len());
        for pid in &children_index {
            if live.contains(pid.as_str()) {
                lists.push(FanoutCommand::new(
                    SnapshotKeyFamily::Children,
                    keys.obj_children(object_id, pid),
                    pid,
                ));
            } else if let Ok(parent_id) = pid.parse::<u64>() {
                dangling_parents.push(parent_id);
            }
        }
        windows
            .run(lists, |tally, command, reply| {
                entries
                    .take_in(object_id, command, reply, tally)
                    .map(|_| ())
            })
            .await?;
        if windows.stopped().is_some() {
            return Ok(stopped(&windows));
        }
    }

    // The charged read's allowance keeps the name bytes it retained, for the
    // attempt's later charge points; applied by the caller before the pass
    // assembles its state.
    let name_bytes = windows.tally.names.bytes;
    if let AllowanceSeam::Caller { link, .. } = &windows.allowance {
        link.ask(PassRequest::NameBytes { name_bytes }).await;
    }

    let state = S::assemble(
        sealed::SnapshotParts {
            object_id: oid,
            sequence_num: cut,
            first_version: parse_u64_warn(&header, "first_ver", object_id),
            source_format: header.get("src_fmt").cloned().unwrap_or_default(),
            next_set_order: parse_u64_warn(&header, "next_set_order", object_id),
            sections: entries.sections,
            children: entries.children,
            tombstones: entries.tombstones,
        },
        entries.value_rows,
        object_id,
    );
    Ok(FanoutOutcome::Read(Box::new(AtomicSnapshotOf {
        state,
        cut,
        last_server_ts: parse_u64_warn(&header, "last_server_ts", object_id),
        last_client_ts: parse_u64_warn(&header, "last_client_ts", object_id),
        reply_bytes: windows.tally.bytes,
        item_count: windows.tally.items,
        entry_count: windows.tally.entries,
        path: read.path(),
        corrupt_section_index_members: entries.corrupt_section_index_members.unwrap_or_default(),
        structural: Some(StructuralSnapshotFacts {
            commit_chain: header
                .get(crate::storage::layer::HEADER_FIELD_COMMIT_CHAIN)
                .cloned()
                .unwrap_or_default(),
            superseded: header
                .get(crate::storage::layer::HEADER_FIELD_SUPERSEDED)
                .cloned(),
            chs_complete: header
                .get(crate::storage::layer::HEADER_FIELD_CHS_COMPLETE)
                .cloned(),
            dangling_parents,
        }),
        memory: SnapshotMemory::default(),
        #[cfg(any(test, feature = "test-support"))]
        value_row_bytes_received: entries.value_row_bytes_received,
    })))
}

/// The charged read's stop, as [`fanout_snapshot_read`] answers it. Reached
/// only once [`FanoutWindows::stopped`] holds a target; 0 otherwise, which
/// no caller sees.
fn stopped<S: SnapshotState>(windows: &FanoutWindows<'_>) -> FanoutOutcome<S> {
    FanoutOutcome::Stopped {
        target: windows.stopped().unwrap_or_default(),
    }
}

// ── Startup self-test (Spec SDD SR-1.8) ─────────────────────────────────────

/// Why the snapshot-read self-test did not pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotSelfTestFailure {
    /// The store refused the script's read of a key it did not declare, in the
    /// declared key's slot. Every atomic snapshot read, and every script window
    /// of the structural WATCH read, would fail the same way, so startup must
    /// stop. Carries the store's error.
    UndeclaredKeyRefused(String),
    /// The store refused the script itself, permanently (an `EVALSHA` / `SCRIPT
    /// LOAD` it will not run, e.g. an ACL denial, or a reply this binary cannot
    /// read), on an error that is neither transient nor a temporary
    /// not-serving state. Startup must stop. Carries the error.
    ScriptRefused(String),
    /// The self-test could not run: a transient failure (timeout, IO, a
    /// connection lost), the store temporarily not serving (`-LOADING`,
    /// `-BUSY`, `-READONLY`, `-MISCONF`, `-NOREPLICAS`), the probe key's slot
    /// migrating or the cluster down, or the deadline passed. Not a verdict on
    /// the store.
    Unavailable(String),
}

/// Operator remediation for [`SnapshotSelfTestFailure::UndeclaredKeyRefused`].
pub const SNAPSHOT_SELF_TEST_REMEDIATION: &str =
    "live-edit's atomic snapshot read (object_snapshot_read.lua) declares only the object's \
     header key and reads the object's other keys, which all carry the same {object_id} hash \
     tag and so sit in the declared key's slot; the structural UpdateLayer read's per-window \
     script (fanout_window_read.lua) relies on the same capability. The backing store refused \
     such an undeclared same-slot read inside a script. Run live-edit against a store that \
     allows a script to read undeclared keys in the declared key's slot \
     (docs/service/docs/backing-store.md § Startup check)";

/// Run `object_snapshot_read.lua` once in self-test mode (Spec SDD SR-1.8):
/// read one undeclared same-slot key beside the header of
/// [`SNAPSHOT_SELF_TEST_OBJECT_ID`], on the snapshot lane.
///
/// One attempt, bounded by `deadline` as a whole (including the `SCRIPT LOAD`
/// a `NOSCRIPT` reply triggers), with no retry of its own: the caller passes
/// the per-command timeout, so the self-test adds at most one command timeout
/// to startup. A zero `deadline` means no bound, as a zero
/// `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` does for every command.
///
/// The verdict is [`SnapshotSelfTestFailure`]: only a refusal fails startup;
/// a transient failure, a store temporarily not serving and a slot or cluster
/// state are [`SnapshotSelfTestFailure::Unavailable`].
///
/// The probe's slot is fixed by the key prefix and
/// [`SNAPSHOT_SELF_TEST_OBJECT_ID`], so on a cluster every replica probes the
/// same one shard. That proves the store's policy, which is uniform across
/// shards, not each shard: per-shard acceptance is proven by the cluster-gated
/// `snapshot_read_cluster_test` integration tests, not at startup.
pub async fn run_snapshot_read_self_test(
    client: &ValkeyClient,
    keys: &KeySchema,
    deadline: std::time::Duration,
) -> Result<(), SnapshotSelfTestFailure> {
    let object_id = SNAPSHOT_SELF_TEST_OBJECT_ID;
    let header_key = keys.obj_header(object_id);
    let argv = object_snapshot_read_argv(
        keys,
        object_id,
        "selftest",
        client.snapshot_atomic_read_max_items,
        client.snapshot_atomic_read_max_bytes,
    );
    let read = eval_object_snapshot_read(client, &header_key, argv);
    let reply = if deadline.is_zero() {
        read.await
    } else {
        match tokio::time::timeout(deadline, read).await {
            Ok(reply) => reply,
            Err(_) => {
                return Err(SnapshotSelfTestFailure::Unavailable(format!(
                    "no reply within {} ms",
                    deadline.as_millis()
                )))
            }
        }
    };
    match reply {
        Ok(value) => {
            let parts = value.into_array();
            let format = parts
                .first()
                .and_then(|v| v.as_str().map(|s| s.to_string()));
            let status = parts.get(1).and_then(|v| v.as_str().map(|s| s.to_string()));
            if format.as_deref() == Some(OBJECT_SNAPSHOT_READ_FORMAT)
                && status.as_deref() == Some("SELFTEST")
            {
                Ok(())
            } else {
                Err(SnapshotSelfTestFailure::ScriptRefused(format!(
                    "unexpected self-test reply (format {format:?}, status {status:?})"
                )))
            }
        }
        Err(e) => {
            FredErrOrigin::client(client, POOL_LABEL_SNAPSHOT).observe_error(&e);
            let selftest_raw = matches!(e.kind(), fred::error::ErrorKind::Unknown)
                .then(|| e.details().strip_prefix(SNAPSHOT_SELFTEST_REFUSED_TAG))
                .flatten()
                .map(|rest| rest.split_once(' ').map_or(rest, |(_key, raw)| raw));
            match selftest_raw {
                // The probe key's slot migrating, or the cluster down, under
                // the self-test: a state of the slot, not a verdict on the
                // store's script policy.
                Some(raw) if is_script_cluster_state_refusal(raw) => {
                    Err(SnapshotSelfTestFailure::Unavailable(e.to_string()))
                }
                Some(_) => Err(SnapshotSelfTestFailure::UndeclaredKeyRefused(
                    e.details().to_string(),
                )),
                // A store temporarily not serving (`-LOADING` while a restarted
                // primary replays its data, `-BUSY` behind a long script, and
                // the write-gated three) answers `EVALSHA` this way although
                // `connect` succeeded; exiting on them would crash-loop the pod
                // until the store recovered. fred hands these over as
                // `ErrorKind::Unknown`. `is_transient_fred_error` calls the
                // first two transient (b.xd1) but not the write-gated three, so
                // `is_backend_unavailable_reply` is what keeps those off the
                // fatal arm.
                None if is_backend_unavailable_reply(&e)
                    || crate::storage::client::is_transient_fred_error(&e) =>
                {
                    Err(SnapshotSelfTestFailure::Unavailable(e.to_string()))
                }
                None => Err(SnapshotSelfTestFailure::ScriptRefused(e.to_string())),
            }
        }
    }
}

// ── build_diff_payload ──────────────────────────────────────────────────────

/// Build an OMPF-wrapped FlatBuffers Delta from a DiffPlan.
///
/// - `UpToDate` → empty Vec (total_size=0, no chunks per SDD §10.1 Case 3).
/// - `FullContent` → isDiff=true, baseVersion=0, sourceFormat from header.
/// - `Incremental` → isDiff=true, baseVersion=have_version.
///
/// External values are packed as OMPF Parts[1..N] with 1-based indices assigned
/// by the `ExternalValueScan`. Fields/time samples referencing external values
/// get their `extValueIndex` set to the correct part index.
pub fn build_diff_payload(
    plan: &DiffPlan,
    source_format: &str,
    have_version: u64,
    scan: &ExternalValueScan,
    ext_data: &HashMap<[u8; 32], Vec<u8>>,
) -> Result<Vec<u8>, ServiceError> {
    // b.kf1 H-stor-10: hard-error on a missing external value rather than
    // packing a zero-length blob. Pre-b.kf1 the lookup was
    // `.unwrap_or_default()`, which silently substituted an empty `Vec<u8>`
    // for any hash absent from `ext_data` — turning a GC race / corruption
    // into a deserialize-as-empty-success on the subscriber side. The
    // subscriber-facing read path now sees `INTERNAL` and can retry against
    // a healthy replica or escalate, as the contract for diff payloads
    // requires every referenced external blob to be present. Defense in
    // depth: upstream, `resolve_external_values` answers `Built::Missing`
    // for a probe that found a hash absent and `read_batch` returns
    // `MetaNotFound` / `DataNotFound`, so this branch is unreachable for the
    // production `ValkeyExternalValueStore` — the check is what makes that
    // unreachability load-bearing instead of a comment-only invariant.
    let mut ext_parts: Vec<Vec<u8>> = Vec::with_capacity(scan.ordered_hashes.len());
    for h in &scan.ordered_hashes {
        match ext_data.get(h) {
            Some(data) => ext_parts.push(data.clone()),
            None => {
                let hash_hex: String = h.iter().fold(String::with_capacity(64), |mut acc, b| {
                    use std::fmt::Write;
                    let _ = write!(acc, "{b:02x}");
                    acc
                });
                tracing::error!(
                    hash = %hash_hex,
                    "external value missing from resolved batch during diff build"
                );
                return Err(ServiceError::Internal(format!(
                    "external value missing: {hash_hex}"
                )));
            }
        }
    }
    let ext_refs: Vec<&[u8]> = ext_parts.iter().map(|v| v.as_slice()).collect();

    let bytes = match plan {
        DiffPlan::UpToDate => Vec::new(),

        DiffPlan::FullContent {
            sections,
            fields,
            time_samples,
        } => {
            let delta_bytes = build_delta(&DeltaBuildParams {
                is_diff: true,
                base_version: 0,
                source_format: Some(source_format),
                new_sections: sections,
                modified_sections: &[],
                deleted_sections: &[],
                fields,
                time_samples,
                hash_to_index: &scan.hash_to_index,
            })?;
            // serialize() only fails on num_parts > u32::MAX — unreachable in
            // practice: ext_refs holds one part per distinct external value of
            // the layer (scan.ordered_hashes), which no per-delta limit bounds,
            // and four billion distinct values would be far beyond any layer's
            // memory.
            OmpfContainer::serialize(&delta_bytes, &ext_refs).expect(
                "OMPF part count fits in u32 — a layer's distinct external values stay far below it",
            )
        }

        DiffPlan::Incremental {
            new_sections,
            modified_sections,
            deleted_sections,
            changed_fields,
            changed_time_samples,
        } => {
            let delta_bytes = build_delta(&DeltaBuildParams {
                is_diff: true,
                base_version: have_version,
                source_format: None,
                new_sections,
                modified_sections,
                deleted_sections,
                fields: changed_fields,
                time_samples: changed_time_samples,
                hash_to_index: &scan.hash_to_index,
            })?;
            // serialize() only fails on num_parts > u32::MAX — unreachable in
            // practice: ext_refs holds one part per distinct external value of
            // the layer (scan.ordered_hashes), which no per-delta limit bounds,
            // and four billion distinct values would be far beyond any layer's
            // memory.
            OmpfContainer::serialize(&delta_bytes, &ext_refs).expect(
                "OMPF part count fits in u32 — a layer's distinct external values stay far below it",
            )
        }
    };
    Ok(bytes)
}

/// Parameters for building a FlatBuffers Delta.
///
/// Visibility is `pub(crate)` so the checkpoint assembler (SR-2.3,
/// `src/checkpoint/assembler.rs::build_reset_ompf`) can construct the same
/// `is_diff=true, base_version=0` shape that the in-RPC diff path emits, without
/// duplicating the FFI marshalling.
pub(crate) struct DeltaBuildParams<'a> {
    pub(crate) is_diff: bool,
    pub(crate) base_version: u64,
    pub(crate) source_format: Option<&'a str>,
    pub(crate) new_sections: &'a [Section],
    pub(crate) modified_sections: &'a [Section],
    pub(crate) deleted_sections: &'a [(u64, u64, u64)],
    pub(crate) fields: &'a [Field],
    pub(crate) time_samples: &'a [crate::domain::types::TimeSample],
    /// Hash → 1-based OMPF part index for external values. Empty map = no external values.
    pub(crate) hash_to_index: &'a HashMap<[u8; 32], u32>,
}

/// Build an OODF delta buffer (with the "OODF" file identifier) from the
/// planned diff components.
///
/// Sections become `StructCommand::DiffChangeSection`, deleted sections become
/// `StructCommand::DiffDeleteSection`, fields become `SetField`, time samples
/// become `TimeSample`. Serialization is delegated to `oodf_ffi::build_delta`.
///
/// Visibility is `pub(crate)` so the checkpoint assembler (SR-2.3) can reuse
/// this helper to emit a reset-delta from a consistent snapshot — see
/// `src/checkpoint/assembler.rs::build_reset_ompf`.
pub(crate) fn build_delta(params: &DeltaBuildParams<'_>) -> Result<Vec<u8>, ServiceError> {
    let mut struct_commands: Vec<OodfStructCommand> = Vec::with_capacity(
        params.new_sections.len() + params.modified_sections.len() + params.deleted_sections.len(),
    );

    // New and modified sections → DiffChangeSection.
    for section in params
        .new_sections
        .iter()
        .chain(params.modified_sections.iter())
    {
        struct_commands.push(OodfStructCommand::DiffChangeSection(
            OodfDiffChangeSection {
                parent_id: section.parent_id,
                section_id: section.section_id,
                section_name: section.name.clone(),
                section_type: section.section_type,
                section_order: section.section_order,
            },
        ));
    }

    // Deleted sections → DiffDeleteSection.
    for &(section_id, _, _) in params.deleted_sections {
        struct_commands.push(OodfStructCommand::DiffDeleteSection(
            OodfDiffDeleteSection { section_id },
        ));
    }

    let set_fields: Vec<OodfSetField> = params
        .fields
        .iter()
        .map(|f| {
            let (value_bytes, ext_size, ext_index) =
                encode_field_value(&f.value, params.hash_to_index);
            OodfSetField {
                section_id: f.section_id,
                key_name: f.key_name.clone(),
                value_or_ext_hash: value_bytes,
                ext_value_size: ext_size,
                ext_value_index: ext_index,
                set_order: f.set_order,
            }
        })
        .collect();

    let time_samples: Vec<OodfTimeSample> = params
        .time_samples
        .iter()
        .map(|ts| {
            let (value_bytes, ext_size, ext_index) =
                encode_field_value(&ts.value, params.hash_to_index);
            OodfTimeSample {
                section_id: ts.section_id,
                time: ts.time.into_inner(),
                value_or_ext_hash: value_bytes,
                ext_value_size: ext_size,
                ext_value_index: ext_index,
            }
        })
        .collect();

    let owned = OwnedDelta {
        is_diff: params.is_diff,
        base_version: params.base_version,
        struct_commands,
        set_fields,
        time_samples,
        source_format: params.source_format.unwrap_or("").to_string(),
    };

    // `oodf_ffi::build_delta` is `Result<Vec<u8>, OodfError>` after b.vvs
    // item 8 removed the C++-side `std::abort()` on allocator / bad-tag
    // failures. The only reachable error variants in practice are
    // `OodfError::AllocationFailed` (the C++ FlatBufferBuilder ran out of
    // memory) and a never-fires `BuildUnknownCommandTag` (only a raw
    // bridge caller can smuggle an out-of-range tag, and `OwnedDelta`'s
    // idiomatic Rust enum makes that unrepresentable from this call site).
    // Propagate via `?` through `From<oodf_ffi::OodfError> for ServiceError`
    // so an allocator failure surfaces to the caller as a clean per-RPC
    // `ServiceError::Internal` (→ gRPC `INTERNAL`) rather than panicking the
    // tokio worker and taking down every concurrent in-flight RPC sharing
    // the process — that "blast-radius collapse" is exactly the win b.vvs
    // item 8 was set up to deliver.
    Ok(oodf_ffi::build_delta(&owned)?)
}

/// Encode a `FieldValue` into `(bytes, ext_value_size, ext_value_index)`.
///
/// For inline values, `ext_value_size=0` and `ext_value_index=0`; bytes are the
/// raw inline payload. For external values, bytes are the 32-byte hash,
/// `ext_value_size` is set, and `ext_value_index` is looked up from
/// `hash_to_index` (1-based OMPF part index).
fn encode_field_value(
    value: &FieldValue,
    hash_to_index: &HashMap<[u8; 32], u32>,
) -> (Vec<u8>, u64, u32) {
    match value {
        FieldValue::Inline(bytes) => (bytes.clone(), 0, 0),
        FieldValue::External { hash, size, .. } => {
            let ext_index = hash_to_index.get(hash).copied().unwrap_or(0);
            (hash.to_vec(), *size, ext_index)
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Parse a u64 from a hash map field.
fn parse_u64(map: &HashMap<String, String>, key: &str) -> Option<u64> {
    map.get(key).and_then(|s| s.parse::<u64>().ok())
}

/// Parse a u64 from a hash map field, defaulting to 0 and emitting a `warn`
/// log on missing-or-corrupt input so an on-wire corruption of the object
/// header is not silently masked (b.k1r medium).
///
/// `owner` is an opaque label (typically `object_id` or the raw key) used
/// in the log message so operators can correlate a warn line back to the
/// concrete object whose header was partial / malformed. We deliberately
/// stop short of returning an error — the snapshot readers and
/// `read_object_header` tolerate partial headers (SDD §7.2 only guarantees `seq` / `src_fmt` are
/// present once a write has committed). A louder log is the right middle
/// ground between "pure silent zero" (the prior behavior) and
/// "retry-the-entire-call-on-every-corrupt-header".
fn parse_u64_warn(map: &HashMap<String, String>, key: &str, owner: &str) -> u64 {
    match map.get(key) {
        None => {
            tracing::warn!(
                field = key,
                owner = owner,
                "object header missing expected u64 field; defaulting to 0"
            );
            0
        }
        Some(s) => match s.parse::<u64>() {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(
                    field = key,
                    value = %s,
                    owner = owner,
                    error = %e,
                    "object header field is not a valid u64; defaulting to 0"
                );
                0
            }
        },
    }
}

/// Per-field-hash sibling of [`parse_u64_warn`], AGGREGATED across one
/// snapshot read: parses one u64 column out of each field hash,
/// defaulting to 0 and tallying (rather than immediately logging) every miss,
/// so [`emit`](ColumnMissTally::emit) can close the pass with ONE `warn` line
/// per `(object_id, column)` (b.3pe).
///
/// Aggregation is the point. The columns this covers — `set_order` and
/// `version` — are written by BOTH commit paths on EVERY field write, so their
/// absence is not a per-field accident: it is all-or-nothing across the whole
/// object. Warning per field per column would emit two lines per field of the
/// object per snapshot read — millions on a large layer, whose field count no
/// per-delta limit bounds — and one runs per catch-up, per
/// CreateLayer-overwrite and per checkpoint tick. The aggregated line carries
/// the miss counts plus one exemplar `(section_id, key_name)` — enough to find
/// the offending rows without flooding the log.
///
/// Same middle-ground posture as the header helper otherwise: the snapshot-load
/// contract tolerates a partial row, so a louder log beats both a silent zero
/// and failing the whole catch-up on one corrupt hash. Columns whose absence is
/// EXPECTED (`authoring_set_order` on pre-release rows) keep using the quiet
/// [`parse_u64`] and are not tallied.
#[derive(Default)]
struct ColumnMissTally {
    /// Field hashes lacking the column entirely.
    missing: u64,
    /// Field hashes carrying a value that does not parse as a `u64`.
    malformed: u64,
    /// First offender seen: `(section_id, key_name, raw value if present)`.
    exemplar: Option<(u64, String, Option<String>)>,
}

impl ColumnMissTally {
    /// Parse `column` out of one field hash, defaulting to 0 and tallying a
    /// missing / malformed column instead of logging it.
    fn parse_or_tally(
        &mut self,
        map: &HashMap<String, String>,
        column: &str,
        section_id: u64,
        key_name: &str,
    ) -> u64 {
        match map.get(column) {
            None => {
                self.missing += 1;
                self.remember(section_id, key_name, None);
                0
            }
            Some(raw) => match raw.parse::<u64>() {
                Ok(v) => v,
                Err(_) => {
                    self.malformed += 1;
                    self.remember(section_id, key_name, Some(raw));
                    0
                }
            },
        }
    }

    fn remember(&mut self, section_id: u64, key_name: &str, raw: Option<&str>) {
        if self.exemplar.is_none() {
            self.exemplar = Some((section_id, key_name.to_string(), raw.map(|s| s.to_string())));
        }
    }

    /// Emit the single aggregated `warn` for this column, or nothing at all when
    /// every field hash parsed cleanly (the overwhelmingly common case).
    fn emit(&self, column: &str, object_id: &str) {
        let Some((section_id, key_name, raw)) = self.exemplar.as_ref() else {
            return;
        };
        tracing::warn!(
            column = column,
            object_id = object_id,
            missing = self.missing,
            malformed = self.malformed,
            exemplar_section_id = section_id,
            exemplar_key_name = %key_name,
            exemplar_value = raw.as_deref().unwrap_or("<absent>"),
            "field hash u64 column missing or not a valid u64; defaulted to 0"
        );
    }
}

/// Aggregated tally of value rows that carry BOTH shapes at once (b.zvd).
///
/// A value row (`obj:fld:*` / `obj:ts:*`) holds exactly one shape: `value` for
/// an inline value, `hash` + `size` for an external one. Both commit paths now
/// `HDEL` all three columns immediately before their `HSET`
/// (`storage::layer::VALUE_HFIELDS`, `scripts/update_layer_fast.lua` step 6),
/// so a row carrying `value` AND `hash` cannot be produced by this service any
/// more. Pre-b.zvd rows can still carry both — a field written inline and later
/// rewritten as an external value stranded the old `value` column, and
/// [`decode_field_value`] checks `value` first, so the row decoded as the STALE
/// inline bytes.
///
/// The reader is deliberately left preferring `value`. Staleness is symmetric —
/// an inline→external rewrite strands `value`, an external→inline rewrite
/// strands `hash` + `size` — and the row carries no ordering information that
/// says which column is the newer one, so no reader-side preference is correct
/// in both directions. Flipping the preference to `hash` would simply move the
/// corruption to the other rewrite direction. What the reader CAN do is make
/// the coexistence visible, which is what this tally is for.
///
/// Aggregated for the same reason [`ColumnMissTally`] is: the trigger is
/// object-wide, a snapshot read runs per catch-up / per
/// CreateLayer-overwrite / per checkpoint tick / per structural WATCH attempt
/// (whose script windows report such a row, `ValueRowReply::shape`), and a
/// per-row `warn` would emit a line per field of the object per pass, which
/// no per-delta limit bounds.
#[derive(Default)]
struct ValueShapeConflictTally {
    /// Rows carrying `value` and `hash` at the same time.
    conflicts: u64,
    /// First offender seen: `(section_id, row name)`.
    exemplar: Option<(u64, String)>,
}

impl ValueShapeConflictTally {
    /// Note one value row. `name` is called only for the first offender, so a
    /// clean pass allocates nothing.
    fn observe(
        &mut self,
        data: &HashMap<String, String>,
        section_id: u64,
        name: impl FnOnce() -> String,
    ) {
        if !(data.contains_key("value") && data.contains_key("hash")) {
            return;
        }
        self.note(section_id, name);
    }

    /// Note one row known to carry both shapes (a slim atomic read's `b`
    /// row, whose columns the reader never sees).
    fn note(&mut self, section_id: u64, name: impl FnOnce() -> String) {
        self.conflicts += 1;
        if self.exemplar.is_none() {
            self.exemplar = Some((section_id, name()));
        }
    }

    /// Emit the single aggregated `warn` for this pass, or nothing at all when
    /// every row held exactly one shape (the overwhelmingly common case).
    fn emit(&self, row_kind: &str, object_id: &str) {
        let Some((section_id, name)) = self.exemplar.as_ref() else {
            return;
        };
        tracing::warn!(
            row_kind = row_kind,
            object_id = object_id,
            conflicts = self.conflicts,
            exemplar_section_id = section_id,
            exemplar_name = %name,
            "value row carries both an inline `value` and an external `hash`; decoding the \
             inline one, which may be stale. Rows written by this service hold exactly one \
             shape, so these were written by an older release or out of band"
        );
    }
}

/// Decode a `FieldValue` from Valkey hash fields.
///
/// If the hash contains a "value" field, it's an inline base64-encoded value.
/// If it contains "hash" + "size", it's an external value reference.
///
/// `value` is checked FIRST and stays that way; [`ValueShapeConflictTally`]
/// explains why a coexistence cannot be resolved here and is reported instead.
///
/// # Errors
///
/// [`ServiceError::Internal`] when the `hash` column is not exactly 64 hex
/// characters — the strict [`hash_from_hex`] decode, never a padded or
/// truncated one, and never a panic on an odd-length or non-ASCII string. Every
/// writer stores a 32-byte hash's hex, so such a row is corrupt or was written
/// out of band; decoding it would hand subscribers a hash naming the wrong
/// blob (or none). The read fails closed instead, as the diff build does for a
/// referenced blob it cannot resolve. `row` names the row in the message.
fn decode_field_value(
    data: &HashMap<String, String>,
    object_id: &str,
    row: impl std::fmt::Display,
) -> Result<FieldValue, ServiceError> {
    if let Some(b64) = data.get("value") {
        let bytes = BASE64.decode(b64).unwrap_or_default();
        Ok(FieldValue::Inline(bytes))
    } else if let Some(hash_hex) = data.get("hash") {
        let hash = parse_external_hash(hash_hex, object_id, row)?;
        let size = data
            .get("size")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        Ok(FieldValue::External {
            hash,
            size,
            ext_value_index: 0,
        })
    } else {
        Ok(FieldValue::Inline(Vec::new()))
    }
}

/// The external hash of a value row, by [`decode_field_value`]'s rule without
/// decoding an inline value: `None` for an inline row (a `value` column, or
/// neither value column), the validated hash for an external one. The
/// compact decode's form, which keeps a row's hash and nothing else.
///
/// # Errors
///
/// As [`decode_field_value`], with its message.
fn decode_external_hash(
    data: &HashMap<String, String>,
    object_id: &str,
    row: impl std::fmt::Display,
) -> Result<Option<[u8; 32]>, ServiceError> {
    if data.contains_key("value") {
        return Ok(None);
    }
    data.get("hash")
        .map(|hash_hex| parse_external_hash(hash_hex, object_id, row))
        .transpose()
}

/// A value row's `hash` column, decoded by the strict [`hash_from_hex`] rule
/// [`decode_field_value`] documents; `row` names the row in the error.
fn parse_external_hash(
    hash_hex: &str,
    object_id: &str,
    row: impl std::fmt::Display,
) -> Result<[u8; 32], ServiceError> {
    hash_from_hex(hash_hex).ok_or_else(|| {
        ServiceError::Internal(format!(
            "object {object_id}: {row} has a malformed external-value `hash` column \
             ({} bytes, expected 64 hex characters)",
            hash_hex.len()
        ))
    })
}

/// Split a sorted set member into its two components at the first `:`.
///
/// Member format: `{section_id}:{hex_encoded_suffix}`
fn split_member(member: &str) -> (String, String) {
    if let Some(pos) = member.find(':') {
        (member[..pos].to_string(), member[pos + 1..].to_string())
    } else {
        (member.to_string(), String::new())
    }
}

// ── Test-support state reader ───────────────────────────────────────────────

/// Read an object's whole state for a test assertion, independently of the
/// service's own snapshot readers. **Test-support only**: compiled out of the
/// production binary, and no production path calls it.
///
/// It reads the whole object on the data pool in four pipelined round trips
/// with no bound, no fence, no proof and no fault seam: the header plus the
/// sections and deleted-sections indices; every section and tombstone hash;
/// the children list of the root and of every indexed section, plus the
/// fields and time-samples indices (`ZRANGE 0 -1`); then every field and
/// time-sample hash. That is the read shape of the loader the service used
/// before its reads became the atomic script and the bounded fan-out, kept
/// here as a reference those readers can be compared against: it shares
/// only the per-entry decoders with them, not the read plan, the windowing
/// or the parent-set selection of a marked object (it never reads
/// `obj:chs`; every indexed section's list is read).
///
/// Run it only on a layer no writer is committing to: with no proof, a commit
/// that lands mid-read leaves the returned state torn across two commits.
///
/// Returns [`ServiceError::NotFound`] when the header is absent. A failed
/// command is mapped by [`map_snapshot_load_err`] (transient → `Aborted`,
/// permanent → `Internal`) without ticking the snapshot-read refusal counter,
/// so a test reading state never moves a metric it may be asserting on.
/// `sections` leaves out an indexed section whose hash is empty, as the
/// service's readers do.
///
/// # Panics
///
/// If `object_id` is not a decimal `u64` — a test bug, not a server state.
#[cfg(any(test, feature = "test-support"))]
pub async fn read_object_state_for_test(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
) -> Result<ObjectSnapshot, ServiceError> {
    use crate::storage::client::POOL_LABEL_DATA;

    let map_err =
        |op: &str, e| map_snapshot_load_err(FredErrOrigin::client(client, POOL_LABEL_DATA), op, e);
    let oid = object_id.parse::<u64>().unwrap_or_else(|e| {
        panic!("read_object_state_for_test: object_id {object_id:?} is not a u64: {e}")
    });

    // Round 1: the header and the two section indices.
    let header_key = keys.obj_header(object_id);
    let (header, section_ids, deleted_section_ids): (
        HashMap<String, String>,
        Vec<String>,
        Vec<String>,
    ) = tokio::try_join!(
        async {
            client
                .pool
                .hgetall(&header_key)
                .await
                .map_err(|e| map_err("HGETALL obj header", e))
        },
        async {
            client
                .pool
                .smembers(keys.obj_sections_index(object_id))
                .await
                .map_err(|e| map_err("SMEMBERS sections index", e))
        },
        async {
            client
                .pool
                .smembers(keys.obj_deleted_sections_index(object_id))
                .await
                .map_err(|e| map_err("SMEMBERS deleted sections index", e))
        },
    )?;
    if header.is_empty() {
        return Err(ServiceError::NotFound(format!(
            "object {object_id} not found"
        )));
    }

    // Round 2: every section and tombstone hash.
    let section_reads = section_ids.iter().map(|sid| async move {
        let data: HashMap<String, String> = client
            .pool
            .hgetall(keys.obj_section(object_id, sid))
            .await
            .map_err(|e| map_err("HGETALL section", e))?;
        Ok::<_, ServiceError>((sid, data))
    });
    let tombstone_reads = deleted_section_ids.iter().map(|sid| async move {
        let data: HashMap<String, String> = client
            .pool
            .hgetall(keys.obj_deleted_section(object_id, sid))
            .await
            .map_err(|e| map_err("HGETALL deleted section", e))?;
        Ok::<_, ServiceError>((sid, data))
    });
    let (section_rows, tombstone_rows) = tokio::try_join!(
        futures::future::try_join_all(section_reads),
        futures::future::try_join_all(tombstone_reads),
    )?;
    let sections: HashMap<u64, Section> = section_rows
        .iter()
        .filter_map(|(sid, data)| decode_section(sid, data))
        .map(|section| (section.section_id, section))
        .collect();
    let tombstones: Vec<(u64, u64, u64)> = tombstone_rows
        .iter()
        .filter_map(|(sid, data)| decode_tombstone(sid, data))
        .collect();

    // Round 3: the children list of every indexed section and of the root,
    // and the two field-level indices.
    let parent_ids: Vec<String> = section_ids
        .iter()
        .cloned()
        .chain(std::iter::once(ROOT_PARENT_ID.to_string()))
        .collect();
    let children_reads = parent_ids.iter().map(|pid| async move {
        let child_ids: Vec<String> = client
            .pool
            .lrange(keys.obj_children(object_id, pid), 0, -1)
            .await
            .map_err(|e| map_err("LRANGE children list", e))?;
        Ok::<_, ServiceError>((pid, child_ids))
    });
    let (children_rows, field_members, ts_members): (_, Vec<String>, Vec<String>) = tokio::try_join!(
        futures::future::try_join_all(children_reads),
        async {
            client
                .pool
                .zrange(
                    keys.obj_fields_index(object_id),
                    0,
                    -1,
                    None,
                    false,
                    None,
                    false,
                )
                .await
                .map_err(|e| map_err("ZRANGE fields index", e))
        },
        async {
            client
                .pool
                .zrange(
                    keys.obj_time_samples_index(object_id),
                    0,
                    -1,
                    None,
                    false,
                    None,
                    false,
                )
                .await
                .map_err(|e| map_err("ZRANGE time samples index", e))
        },
    )?;
    let children: HashMap<u64, Vec<u64>> = children_rows
        .iter()
        .filter_map(|(pid, child_ids)| decode_children_list(pid, child_ids))
        .collect();

    // Round 4: every field and time-sample hash. Members are
    // `{section_id}:{hex(key_name)}` and `{section_id}:{hex_f64(time)}`.
    let field_reads = field_members.iter().map(|member| async move {
        let (sid, hex_key) = split_member(member);
        let key_name = keys::hex_decode(&hex_key);
        let data: HashMap<String, String> = client
            .pool
            .hgetall(keys.obj_field(object_id, &sid, &key_name))
            .await
            .map_err(|e| map_err("HGETALL field", e))?;
        Ok::<_, ServiceError>((sid, key_name, data))
    });
    let ts_reads = ts_members.iter().map(|member| async move {
        let (sid, time_hex) = split_member(member);
        let time = keys::hex_decode_f64(&time_hex);
        let data: HashMap<String, String> = client
            .pool
            .hgetall(keys.obj_time_sample(object_id, &sid, time))
            .await
            .map_err(|e| map_err("HGETALL time sample", e))?;
        Ok::<_, ServiceError>((sid, time, data))
    });
    let (field_rows, ts_rows) = tokio::try_join!(
        futures::future::try_join_all(field_reads),
        futures::future::try_join_all(ts_reads),
    )?;

    Ok(ObjectSnapshot {
        object_id: oid,
        sequence_num: parse_u64_warn(&header, "seq", object_id),
        first_version: parse_u64_warn(&header, "first_ver", object_id),
        source_format: header.get("src_fmt").cloned().unwrap_or_default(),
        next_set_order: parse_u64_warn(&header, "next_set_order", object_id),
        sections,
        children,
        fields: decode_field_rows(object_id, field_rows)?,
        time_samples: decode_time_sample_rows(object_id, ts_rows)?,
        tombstones,
    })
}

/// The compact projection of a full snapshot read: the structural cache's
/// contract oracle (structural-edits SDD SR-3.2). **Test-support only.**
///
/// The header scalars, sections and children lists as the snapshot holds
/// them, and per value row its key and external hash (`None` for an inline
/// row), each time-sample row keyed by its own stored time bits — so a `0.0`
/// row and a `-0.0` row stay two rows, as they are two stored keys, where
/// [`snapshot_to_object_state`] resolves them into one. Built over the rows
/// [`read_object_state_for_test`] reads (its `time_samples` is a `Vec`, one
/// element per stored row), it is what a cold start's slim read decodes the
/// same cut into.
///
/// Not normalized: a caller comparing it with a cache entry normalizes it
/// with `domain::replay::normalize_state` first, as the cold start does, and
/// keeps the parent ids that returns as the dangling parents. Tombstones are
/// dropped: no compact state holds them.
#[cfg(any(test, feature = "test-support"))]
pub fn compact_snapshot(snap: &ObjectSnapshot) -> CompactState {
    let external = |value: &FieldValue| match value {
        FieldValue::External { hash, .. } => Some(*hash),
        FieldValue::Inline(_) => None,
    };
    let mut fields = CompactFields::with_capacity(snap.fields.len());
    for field in &snap.fields {
        fields.insert(field.section_id, &field.key_name, external(&field.value));
    }
    let mut time_samples = CompactSamples::with_capacity(snap.time_samples.len());
    for sample in &snap.time_samples {
        time_samples.insert(
            sample.section_id,
            sample.time.into_inner().to_bits(),
            external(&sample.value),
        );
    }
    CompactState {
        object_id: snap.object_id,
        sequence_num: snap.sequence_num,
        first_version: snap.first_version,
        source_format: snap.source_format.clone(),
        next_set_order: snap.next_set_order,
        sections: snap.sections.clone(),
        children: snap.children.clone(),
        fields,
        time_samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guard 3 of the three named on [`SnapshotKeyFamily`]: the enum's labels
    /// and `metrics::SNAPSHOT_READ_KEY_FAMILIES` are the **same set**, checked
    /// in both directions.
    ///
    /// The `const` assertion beside `SnapshotKeyFamily::ALL` already pins the
    /// two cardinalities at compile time, so what is left for a test is the
    /// drift that keeps the count intact: a duplicated entry in the metrics
    /// array paired with an omitted one. A label registered but never emitted
    /// leaves an alert with a permanently-zero series; a label emitted but never
    /// registered defeats the zero pre-seeding the `increase(...) > 0` recipe
    /// depends on.
    ///
    /// Scope limit, shared with guard 2: this iterates `ALL`, so a variant that
    /// was labelled and called but never listed in `ALL` is invisible to it and
    /// passes. That is the residual hole named on `SnapshotKeyFamily`, and it is
    /// closed at review time rather than here — a test cannot enumerate the
    /// variants of an enum without a list, and `ALL` *is* the list.
    #[test]
    fn snapshot_key_family_all_matches_the_registered_label_set() {
        let from_enum: Vec<&'static str> = SnapshotKeyFamily::ALL
            .iter()
            .map(|f| f.label())
            .collect::<Vec<_>>();
        let registered: &[&str] = crate::observability::metrics::SNAPSHOT_READ_KEY_FAMILIES;

        // Neither side may repeat a value, or "same length + containment" would
        // stop implying "same set".
        for (i, label) in from_enum.iter().enumerate() {
            assert!(
                !from_enum[i + 1..].contains(label),
                "SnapshotKeyFamily::ALL yields the label {label:?} twice — two variants share a \
                 metric label, so the counter cannot distinguish them"
            );
        }
        for (i, label) in registered.iter().enumerate() {
            assert!(
                !registered[i + 1..].contains(label),
                "metrics::SNAPSHOT_READ_KEY_FAMILIES lists {label:?} twice"
            );
        }

        for label in &from_enum {
            assert!(
                registered.contains(label),
                "SnapshotKeyFamily emits {label:?} but metrics::SNAPSHOT_READ_KEY_FAMILIES does \
                 not register it, so that cell is never pre-seeded to zero"
            );
        }
        for label in registered {
            assert!(
                from_enum.contains(label),
                "metrics::SNAPSHOT_READ_KEY_FAMILIES registers {label:?} but no SnapshotKeyFamily \
                 variant emits it, so that cell can only ever read zero"
            );
        }

        // Redundant with the `const` assertion by construction, restated here so
        // a failure of this test reads as one story rather than two.
        assert_eq!(from_enum.len(), registered.len());
    }

    // ── b.muu: `snapshot_read_refused_total{kind}` vocabulary + producer ─────
    //
    // `SnapshotKeyFamily` gets a `const` cardinality guard because its labels
    // come from an enum this crate owns. The refusal vocabulary has no such
    // enum — its producer matches the third-party `fred::error::ErrorKind`
    // behind a `_` arm — so the only thing that can hold the constants, the
    // registered array and the producer's reachable output in step is a test
    // that actually runs the producer. The three below are that test, and the
    // doc on `metrics::SNAPSHOT_READ_REFUSAL_KINDS` names the middle one.
    //
    // As everywhere else in this repo that pins a fred contract, the errors are
    // hand-built: `pretty_error` lives in fred's private `mod protocol` and is
    // unreachable from this crate (`storage::client`'s unit block makes the same
    // concession, with the same consequence — these cannot detect fred changing
    // its kind mapping, only this crate drifting from what fred documents).

    /// Every `SNAPSHOT_READ_REFUSAL_KIND_*` constant is listed in
    /// `SNAPSHOT_READ_REFUSAL_KINDS`.
    ///
    /// Named individually rather than iterated, because naming them is the
    /// point: the array is what `Metrics::new` seeds from, so a constant that
    /// exists, is emitted by a match arm, and is missing here would be an
    /// unseeded cell — a series absent until its first tick, which is exactly
    /// the absent-metric semantics the `increase(...) > 0` recipe cannot
    /// evaluate. The set-equality test below catches the same drift through the
    /// producer; this one catches it a step earlier and says which constant.
    #[test]
    fn snapshot_read_refusal_kind_consts_are_all_registered() {
        use crate::observability::metrics as m;
        let registered: &[&str] = m::SNAPSHOT_READ_REFUSAL_KINDS;
        for label in [
            m::SNAPSHOT_READ_REFUSAL_KIND_PROTOCOL,
            m::SNAPSHOT_READ_REFUSAL_KIND_PARSE,
            m::SNAPSHOT_READ_REFUSAL_KIND_AUTH,
            m::SNAPSHOT_READ_REFUSAL_KIND_CONFIG,
            m::SNAPSHOT_READ_REFUSAL_KIND_TLS,
            m::SNAPSHOT_READ_REFUSAL_KIND_INVALID_ARGUMENT,
            m::SNAPSHOT_READ_REFUSAL_KIND_INVALID_COMMAND,
            m::SNAPSHOT_READ_REFUSAL_KIND_ROUTING,
            m::SNAPSHOT_READ_REFUSAL_KIND_NOT_FOUND,
            m::SNAPSHOT_READ_REFUSAL_KIND_BACKEND_UNAVAILABLE,
            m::SNAPSHOT_READ_REFUSAL_KIND_BACKEND_OUT_OF_MEMORY,
            m::SNAPSHOT_READ_REFUSAL_KIND_UNKNOWN,
        ] {
            assert!(
                registered.contains(&label),
                "SNAPSHOT_READ_REFUSAL_KIND_* constant {label:?} is not listed in \
                 metrics::SNAPSHOT_READ_REFUSAL_KINDS, so Metrics::new never seeds that cell to \
                 zero and the series appears only on its first tick"
            );
        }

        for (i, label) in registered.iter().enumerate() {
            assert!(
                !registered[i + 1..].contains(label),
                "metrics::SNAPSHOT_READ_REFUSAL_KINDS lists {label:?} twice — with a duplicate, \
                 'same length + containment' stops implying 'same set'"
            );
        }
    }

    /// The labels `snapshot_refusal_kind_label` can actually emit are exactly
    /// `metrics::SNAPSHOT_READ_REFUSAL_KINDS` — both directions.
    ///
    /// Forward: every label the producer emits is registered, or its cell is
    /// never pre-seeded. Reverse: every registered label is reachable from some
    /// input, or the array carries a cell that can only ever read zero and an
    /// operator's dashboard shows a row that means nothing.
    ///
    /// The reverse direction has one deliberate exception under the
    /// `--no-default-features` lint/build shape: `tls` is registered
    /// unconditionally (so the published series set does not depend on how the
    /// binary was compiled) while `fred::error::ErrorKind::Tls` does not exist
    /// unless fred is built with a TLS backend, so on that shape there is no
    /// input that can reach it. That is the design, not drift — see the
    /// constant's doc — and the `cfg` below encodes it rather than weakening
    /// the assertion to one direction.
    #[test]
    fn snapshot_refusal_kind_label_emits_exactly_the_registered_kind_set() {
        use crate::observability::metrics as m;
        use fred::error::{Error as FredError, ErrorKind as K};

        // One case per arm of the producer's match, including all three arms of
        // the `Unknown` split (its two guards and the fall-through), and one
        // input for the `_` arm that is not `Unknown` at all (`Url` — named on
        // the `unknown` constant's doc, and a kind fred mints client-side).
        let mut cases: Vec<(FredError, &'static str)> = vec![
            (
                FredError::new(K::Protocol, "unexpected frame"),
                m::SNAPSHOT_READ_REFUSAL_KIND_PROTOCOL,
            ),
            (
                FredError::new(K::Parse, "cannot convert to u64"),
                m::SNAPSHOT_READ_REFUSAL_KIND_PARSE,
            ),
            (
                FredError::new(K::Auth, "NOAUTH Authentication required."),
                m::SNAPSHOT_READ_REFUSAL_KIND_AUTH,
            ),
            (
                FredError::new(K::Config, "invalid client configuration"),
                m::SNAPSHOT_READ_REFUSAL_KIND_CONFIG,
            ),
            (
                // Not a `WRONGTYPE` reply: that case returns from
                // `count_object_key_refusal`'s first arm and never reaches the
                // producer, which is what makes the `invalid_argument` label
                // honest without saying "non-WRONGTYPE".
                FredError::new(K::InvalidArgument, "invalid argument"),
                m::SNAPSHOT_READ_REFUSAL_KIND_INVALID_ARGUMENT,
            ),
            (
                FredError::new(K::InvalidCommand, "cannot use a subscriber client for this"),
                m::SNAPSHOT_READ_REFUSAL_KIND_INVALID_COMMAND,
            ),
            (
                FredError::new(K::Routing, "Missing hash slot owner."),
                m::SNAPSHOT_READ_REFUSAL_KIND_ROUTING,
            ),
            (
                FredError::new(K::NotFound, "Cannot convert nil to u64"),
                m::SNAPSHOT_READ_REFUSAL_KIND_NOT_FOUND,
            ),
            (
                // The `Unknown` split, first guard: a real `-OOM` reply, the
                // one the orphan sweep's retirement-fence `HSET` draws under
                // the mandated `noeviction` policy.
                FredError::new(
                    K::Unknown,
                    "OOM command not allowed when used memory > 'maxmemory'.",
                ),
                m::SNAPSHOT_READ_REFUSAL_KIND_BACKEND_OUT_OF_MEMORY,
            ),
            (
                // Second guard: a real `-LOADING` reply.
                FredError::new(K::Unknown, "LOADING Redis is loading the dataset in memory"),
                m::SNAPSHOT_READ_REFUSAL_KIND_BACKEND_UNAVAILABLE,
            ),
            (
                // …and neither: `Unknown` that no guard claims.
                FredError::new(K::Unknown, "ERR unknown command 'FOO'"),
                m::SNAPSHOT_READ_REFUSAL_KIND_UNKNOWN,
            ),
        ];
        // Pushed rather than listed so the binding's `mut` is used on every
        // build shape (the `tls` push below is `cfg`-gated away on one).
        cases.push((
            // The `_` arm proper — a kind with no arm of its own.
            FredError::new(K::Url, "invalid url"),
            m::SNAPSHOT_READ_REFUSAL_KIND_UNKNOWN,
        ));
        // Built by hand: in this build (fred over rustls) only a TLS
        // configuration fault such as an invalid TLS server name has the
        // `Tls` kind. A certificate or handshake failure arrives as `IO`.
        #[cfg(feature = "valkey-tls")]
        cases.push((
            FredError::new(K::Tls, "invalid dns name"),
            m::SNAPSHOT_READ_REFUSAL_KIND_TLS,
        ));

        let registered: &[&str] = m::SNAPSHOT_READ_REFUSAL_KINDS;
        let mut emitted: Vec<&'static str> = Vec::new();
        for (e, expected) in &cases {
            let label = snapshot_refusal_kind_label(e);
            assert_eq!(
                label,
                *expected,
                "{:?} + {:?} must map to {expected:?}",
                e.kind(),
                e.details()
            );
            assert!(
                registered.contains(&label),
                "snapshot_refusal_kind_label emitted {label:?}, which \
                 metrics::SNAPSHOT_READ_REFUSAL_KINDS does not register — that cell is never \
                 seeded to zero"
            );
            if !emitted.contains(&label) {
                emitted.push(label);
            }
        }

        for label in registered {
            // Registered unconditionally, unreachable on this build shape — by
            // design, see the doc above. `cfg!` rather than a `#[cfg]` on the
            // statement so the expression type-checks identically on both
            // shapes.
            if !cfg!(feature = "valkey-tls") && *label == m::SNAPSHOT_READ_REFUSAL_KIND_TLS {
                continue;
            }
            assert!(
                emitted.contains(label),
                "metrics::SNAPSHOT_READ_REFUSAL_KINDS registers {label:?} but no input drives \
                 snapshot_refusal_kind_label to emit it, so that cell can only ever read zero — \
                 either the producer lost an arm or the array gained a label nothing mints"
            );
        }
    }

    /// `is_backend_unavailable_reply` is kind-gated and prefix-anchored, on
    /// each token with its trailing space, bare or after an `EXEC`'s
    /// `EXECABORT Transaction discarded because of: ` prefix.
    ///
    /// Both directions are load-bearing. A miss of one of the write-gated
    /// three (`-MISCONF`, `-READONLY`, `-NOREPLICAS` — the cell's live
    /// members, the orphan sweep's fence write refused) puts a condition that
    /// can clear without an operator into `unknown` and costs the eleven
    /// "should never happen" cells their `increase(...) > 0` alert; a miss of
    /// `-LOADING` / `-BUSY` loses the tripwire that would show their transient
    /// classification regressing (b.xd1). A false positive hides a real
    /// unclassified failure in the one cell nobody alerts on at `> 0`.
    #[test]
    fn is_backend_unavailable_reply_matches_the_five_prefixes_under_unknown_only() {
        use fred::error::{Error as FredError, ErrorKind as K};

        // Verbatim-shaped replies for the write-gated three, which fred's
        // `pretty_error` does not recognise, so all three arrive as `Unknown`.
        let write_gated = [
            "MISCONF Redis is configured to save RDB snapshots, but it's currently unable to \
             persist to disk",
            "READONLY You can't write against a read only replica.",
            "NOREPLICAS Not enough good replicas to write.",
        ];
        // The tripwire pair: every not-serving shape the classifier knows.
        let not_serving = crate::storage::client::STORE_NOT_SERVING_SHAPES;

        for reply in write_gated.into_iter().chain(not_serving) {
            for details in [
                reply.to_owned(),
                format!("EXECABORT Transaction discarded because of: {reply}"),
            ] {
                assert!(
                    is_backend_unavailable_reply(&FredError::new(K::Unknown, details.clone())),
                    "{details:?} is the backing store declining to serve; missing it leaves it \
                     in the `unknown` cell, which the eleven-cell `increase(...) > 0` recipe \
                     cannot tolerate"
                );
                // Kind gate: the same text under any other kind is that kind's
                // error. Mirrors `storage::client`'s `-UNBLOCKED` and router
                // predicates, which discriminate the same way.
                assert!(
                    !is_backend_unavailable_reply(&FredError::new(K::Protocol, details.clone())),
                    "a `Protocol` error is a malformed frame whatever its message says — \
                     matching {details:?} on the message alone would relabel a real protocol \
                     bug as the store declining to serve"
                );
            }
        }
        // The tripwire pair never reaches the counting arm while the
        // classifier calls it transient: the cell sees it only if that
        // regresses.
        for reply in not_serving {
            assert!(
                crate::storage::client::is_transient_fred_error(&FredError::new(K::Unknown, reply)),
                "{reply:?} is transient (b.xd1), so a healthy restart does not move the cell"
            );
        }

        // Prefix anchor, with the trailing space: the token is only meaningful
        // in the RESP error-code position, and `-BUSYGROUP` / `-BUSYKEY` (a
        // consumer-group or `RESTORE` conflict) are not the store declining to
        // serve.
        for details in [
            "Command failed: LOADING Redis is loading the dataset in memory",
            "ERR unknown command 'LOADING'",
            "BUSYGROUP Consumer Group name already exists",
            "BUSYKEY Target key name already exists.",
            "READONLYX something",
            "EXECABORT Transaction discarded because of previous errors.",
        ] {
            assert!(
                !is_backend_unavailable_reply(&FredError::new(K::Unknown, details)),
                "{details:?} does not START with one of the five tokens and its space — the \
                 match is a prefix on `details`, never a substring search"
            );
        }

        // The family split. `-OOM` belongs to `is_backend_out_of_memory_reply`
        // and its own cell, NOT to this family: every token above has a path
        // back that can close without an operator, and under the mandated
        // `noeviction` policy `-OOM` has none at all. A match here would route
        // an operator-action condition to this cell's WARN / sustained-rate arm
        // and make its documented contract false.
        for details in [
            "OOM command not allowed when used memory > 'maxmemory'.",
            "OOM allow-oom flag is not set on the script, can not run it when used memory > \
             'maxmemory'",
        ] {
            assert!(
                !is_backend_unavailable_reply(&FredError::new(K::Unknown, details)),
                "{details:?} is the store being FULL, not the store declining to serve something \
                 that has a path back which can close without an operator — matching it here \
                 costs `backend_out_of_memory` its ERROR level and its `increase(...) > 0` page"
            );
        }

        // The three other well-known `Unknown` residents this predicate shares
        // the kind with, each owned elsewhere: the `-UNBLOCKED` reply
        // (`storage::client::is_client_unblocked_error`) and the two
        // `Unknown`-kind router carve-outs
        // (`is_router_no_connection_error`, `is_router_retry_budget_exhausted_error`),
        // both of which are classified TRANSIENT and so never reach the
        // producer at all.
        for details in [
            "UNBLOCKED client unblocked via CLIENT UNBLOCK",
            "Failed to route command.",
            "Too many failed write attempts.",
        ] {
            assert!(
                !is_backend_unavailable_reply(&FredError::new(K::Unknown, details)),
                "{details:?} is not the backing store declining to serve — sharing \
                 `ErrorKind::Unknown` with the five tokens is exactly why this predicate matches \
                 a prefix rather than widening the kind"
            );
        }
    }

    /// `is_backend_out_of_memory_reply` is kind-gated and prefix-anchored, and
    /// claims the `-OOM` token alone.
    ///
    /// The sibling shape of the test above, for the cell that carries the one
    /// backing-store condition with no operator-free path back at all. A miss
    /// here puts a full store into `unknown` and describes it to the operator
    /// as a failure kind the service does not map; a false positive pages on
    /// somebody else's error.
    #[test]
    fn is_backend_out_of_memory_reply_matches_the_oom_prefix_under_unknown_only() {
        use fred::error::{Error as FredError, ErrorKind as K};

        // Both Valkey origins of the token. Only the first is reachable on the
        // counted touch set (its one script, the atomic snapshot read, has no
        // shebang, and the second origin refuses only flagged scripts), but the
        // predicate claims both so a future touch set that gains a flagged
        // script is already correct.
        for reply in [
            "OOM command not allowed when used memory > 'maxmemory'.",
            "OOM allow-oom flag is not set on the script, can not run it when used memory > \
             'maxmemory'",
        ] {
            assert!(
                is_backend_out_of_memory_reply(&FredError::new(K::Unknown, reply)),
                "{reply:?} is the backing store answering that it is full; missing it leaves a \
                 condition that does NOT self-clear under the mandated noeviction policy in the \
                 `unknown` cell"
            );
            // Kind gate, same as the sibling: the text under any other kind is
            // that kind's error, whatever its message says.
            assert!(
                !is_backend_out_of_memory_reply(&FredError::new(K::Protocol, reply)),
                "a `Protocol` error is a malformed frame whatever its message says — matching \
                 {reply:?} on the message alone would relabel a real protocol bug as a full store"
            );
        }

        // Prefix anchor: the token only means this in the RESP error-code
        // position. A message that merely quotes it is somebody else's error.
        assert!(
            !is_backend_out_of_memory_reply(&FredError::new(
                K::Unknown,
                "ERR the server reported OOM command not allowed"
            )),
            "a mid-string `OOM` is in the message body, not the RESP error-code position — the \
             match is a prefix on `details`, never a substring search"
        );
    }

    /// A snapshot read that times out on the snapshot lane is counted against
    /// the lane: `classify_snapshot_read_err`'s `pool_label` decides which
    /// `valkey_errors_total{pool, kind="command_timeout"}` cell ticks, and the
    /// row shows exactly one tick, on `snapshot`, with the data pool's cell
    /// untouched — a timeout counted on the wrong pool sends an operator to the
    /// wrong connection. The read is held behind a server-side `BLPOP` on every
    /// lane connection, so the `Timeout` is fred's real one. The returned
    /// variant is the transient one whatever the label.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn classify_snapshot_read_err_counts_a_lane_timeout_against_the_snapshot_pool() {
        use fred::interfaces::HashesInterface;

        use crate::storage::client::POOL_LABEL_SNAPSHOT;
        use crate::storage::test_support::{
            command_timeout_row, command_timeouts_only_on, release_all, TestValkey,
        };

        let tv = TestValkey::setup().await;
        let metrics = std::sync::Arc::new(crate::observability::metrics::metrics_scrape_only(
            "diff-read",
        ));
        tv.client()
            .install_observability_hooks(std::sync::Arc::clone(&metrics))
            .await;
        let key = format!("{}lane-timeout:hdr", tv.prefix());

        let stalls = tv.stall_snapshot_pool().await;
        let (lane, _in_use) = tv.client().acquire_snapshot();
        let read: Result<Option<String>, fred::error::Error> = lane.hget(&key, "seq").await;
        release_all(stalls).await;

        let e = read.expect_err("a read queued behind the stalled lane must time out");
        let err = classify_snapshot_read_err(
            tv.client(),
            POOL_LABEL_SNAPSHOT,
            "7",
            SnapshotKeyFamily::Header,
            &key,
            "HGET obj header seq",
            e,
        );
        assert!(
            matches!(err, ServiceError::Aborted(_)),
            "a timed-out snapshot read is transient; got {err:?}"
        );
        assert_eq!(
            command_timeout_row(&metrics),
            command_timeouts_only_on(POOL_LABEL_SNAPSHOT, 1),
            "the timed-out lane read must count once, on the snapshot pool"
        );
    }

    // ── The atomic read's in-script cluster refusals (Spec SDD SR-1.4) ──────
    //
    // `is_script_cluster_state_refusal` decides between "transient, retry"
    // and "permanent refusal, counted" for every tagged per-key error, and
    // between `Unavailable` and a fatal exit for the startup self-test. Both
    // directions are load-bearing: a miss fails every read of an object in a
    // migrating slot non-retryably for the whole resharding (and exits a pod
    // whose probe slot is migrating); a false positive retries a real refusal
    // forever and hides it from the refusal counters.

    /// The five texts — Valkey's four (`scriptVerifyClusterState`) and Azure
    /// Managed Redis's wording of the non-local-key one — each with the `ERR `
    /// code `luaPushErrorBuff` prepends before `redis.pcall` returns it and
    /// without it, and the rehashing text for several command names.
    #[test]
    fn is_script_cluster_state_refusal_matches_the_five_texts_with_or_without_err() {
        let texts = [
            "Script attempted to access a non local key in a cluster node".to_string(),
            "Script attempted to execute a command while the cluster is down".to_string(),
            "Script attempted to access a slot not served".to_string(),
            "Lua script attempted to access a non local key in a cluster node".to_string(),
            "Unable to execute command 'HGETALL' in script because undeclared keys were accessed \
             during rehashing of the slot"
                .to_string(),
            "Unable to execute command 'LRANGE' in script because undeclared keys were accessed \
             during rehashing of the slot"
                .to_string(),
            "Unable to execute command 'EXISTS' in script because undeclared keys were accessed \
             during rehashing of the slot"
                .to_string(),
        ];
        for text in texts {
            assert!(
                is_script_cluster_state_refusal(&text),
                "{text:?} is the slot's state refusing an undeclared key — transient"
            );
            assert!(
                is_script_cluster_state_refusal(&format!("ERR {text}")),
                "`redis.pcall` returns the text with Valkey's `ERR ` code: {text:?}"
            );
        }
    }

    /// Anchored and exact: anything that merely contains, extends or garbles
    /// one of the texts is somebody else's error and stays a refusal.
    #[test]
    fn is_script_cluster_state_refusal_rejects_substrings_decoys_and_malformed_command_names() {
        const REHASH_TAIL: &str =
            "in script because undeclared keys were accessed during rehashing of the slot";
        for raw in [
            "",
            "ERR",
            "ERR ",
            // A substring inside another error, at either end.
            "ERR wrapped: Script attempted to access a non local key in a cluster node",
            "Script attempted to access a non local key in a cluster node, and more",
            "Script attempted to access a slot not served yet",
            // The `ERR ` code is stripped once, in its exact spelling.
            "ERR ERR Script attempted to access a slot not served",
            "err Script attempted to access a slot not served",
            "ERRScript attempted to access a slot not served",
            // The cross-slot texts of the same Valkey function: every key the
            // script builds carries the object's tag, so these are bugs, never
            // a slot's state.
            "ERR Script attempted to access keys that do not hash to the same slot",
            "ERR Command attempted to access keys that do not hash to the same slot",
            // The ordinary per-key refusal.
            "WRONGTYPE Operation against a key holding the wrong kind of value",
            // AMR's wording, decorated the way an uncaught script error is:
            // with its location prefix, or with the Redis 7.x script suffix.
            // Neither is what `redis.pcall` hands the script on AMR.
            "@user_script: 30: Lua script attempted to access a non local key in a cluster node",
            "Lua script attempted to access a non local key in a cluster node script: \
             b08075fc7374844c5b04707915156ba3a0cd5466, on @user_script:30.",
            "ERR Lua script attempted to access a non local key in a cluster node script: \
             b08075fc7374844c5b04707915156ba3a0cd5466, on @user_script:30.",
        ] {
            assert!(
                !is_script_cluster_state_refusal(raw),
                "{raw:?} is not one of the five anchored texts"
            );
        }
        for command in ["", "HGET'ALL", "'"] {
            let raw = format!("ERR Unable to execute command '{command}' {REHASH_TAIL}");
            assert!(
                !is_script_cluster_state_refusal(&raw),
                "the rehashing text's command name must be non-empty and carry no quote: {raw:?}"
            );
        }
        assert!(
            !is_script_cluster_state_refusal(&format!(
                "ERR Unable to execute command 'HGETALL' {REHASH_TAIL} now"
            )),
            "the rehashing text is anchored at its end too"
        );
    }

    /// The tag is trusted only on a script's own error reply
    /// (`ErrorKind::Unknown`), and only with a known family label, a key and a
    /// raw error.
    #[test]
    fn parse_snapshot_key_refusal_reads_a_tagged_script_reply_only() {
        use fred::error::{Error as FredError, ErrorKind as K};

        let parsed = |kind: K, details: &str| {
            parse_snapshot_key_refusal(&FredError::new(kind, details.to_owned()))
                .map(|(family, key, raw)| (family, key.to_owned(), raw.to_owned()))
        };
        for family in SnapshotKeyFamily::ALL {
            let details = format!(
                "SNAPSHOT_KEY_REFUSED: {} le:{{7}}:obj:x WRONGTYPE Operation against a key",
                family.label()
            );
            assert_eq!(
                parsed(K::Unknown, &details),
                Some((
                    family,
                    "le:{7}:obj:x".to_owned(),
                    "WRONGTYPE Operation against a key".to_owned()
                )),
                "every family label round-trips through the tag: {details:?}"
            );
        }
        for (kind, details) in [
            // A non-script error carrying the text is not the script's tag.
            (
                K::InvalidArgument,
                "SNAPSHOT_KEY_REFUSED: section le:{7}:obj:sec:1 WRONGTYPE x",
            ),
            (K::Unknown, "SNAPSHOT_KEY_REFUSED: no_such_family k ERR x"),
            (K::Unknown, "SNAPSHOT_KEY_REFUSED: section"),
            (K::Unknown, "SNAPSHOT_KEY_REFUSED: section le:{7}:obj:sec:1"),
            (
                K::Unknown,
                "ERR SNAPSHOT_KEY_REFUSED: section k WRONGTYPE x",
            ),
        ] {
            let label = format!("{kind:?} {details:?}");
            assert_eq!(
                parsed(kind, details),
                None,
                "{label} is not a tagged refusal"
            );
        }
    }

    // ── object_snapshot_read.lua's key construction (Spec SDD SR-1.1) ───────
    //
    // "No key outside the `{oid}` tag" is asserted on the key set itself:
    // here over the script's source and the Rust side's `ARGV` / `KEYS`, and
    // in `tests/integration/storage/snapshot_read_atomic_test.rs` (single
    // node) and `snapshot_read_cluster_test.rs` (each shard) over a `MONITOR`
    // record of what the script actually touched.

    const OBJECT_SNAPSHOT_READ_LUA: &str = include_str!("scripts/object_snapshot_read.lua");

    /// The script's code lines with `--` comments removed.
    fn object_snapshot_read_code() -> Vec<&'static str> {
        OBJECT_SNAPSHOT_READ_LUA
            .lines()
            .map(|line| line.split("--").next().unwrap_or(""))
            .collect()
    }

    /// `(family literal, command, key expression)` for every call of the
    /// script's per-key `call(family, cmd, key, ...)` wrapper. The family is
    /// `None` where it is passed through a variable.
    fn object_snapshot_read_calls() -> Vec<(Option<String>, String, String)> {
        let mut out = Vec::new();
        for line in object_snapshot_read_code() {
            let mut rest = line;
            while let Some(at) = rest.find("call(") {
                let before = &rest[..at];
                let args = &rest[at + "call(".len()..];
                rest = args;
                // `redis.pcall(` and the wrapper's own definition are not calls
                // of the wrapper.
                if before.ends_with("function ")
                    || before
                        .chars()
                        .last()
                        .is_some_and(|c| c.is_alphanumeric() || c == '_')
                {
                    continue;
                }
                let mut depth = 0usize;
                let mut parts = vec![String::new()];
                for c in args.chars() {
                    match c {
                        '(' | '[' | '{' => depth += 1,
                        ')' | ']' | '}' if depth > 0 => depth -= 1,
                        ')' => break,
                        ',' if depth == 0 => {
                            parts.push(String::new());
                            continue;
                        }
                        _ => {}
                    }
                    parts.last_mut().expect("never empty").push(c);
                }
                let parts: Vec<String> = parts.iter().map(|p| p.trim().to_owned()).collect();
                assert!(
                    parts.len() >= 3,
                    "call({args}) must pass a family, a command and a key"
                );
                let literal = |s: &str| s.strip_prefix('\'')?.strip_suffix('\'').map(str::to_owned);
                out.push((
                    literal(&parts[0]),
                    literal(&parts[1]).unwrap_or_else(|| parts[1].clone()),
                    parts[2].clone(),
                ));
            }
        }
        out
    }

    /// The key suffix (after `{prefix}{oid}:obj:`) each family reads.
    fn family_key_suffix(family: SnapshotKeyFamily) -> &'static str {
        match family {
            SnapshotKeyFamily::Header => "hdr",
            SnapshotKeyFamily::SectionsIndex => "secs",
            SnapshotKeyFamily::DeletedSectionsIndex => "dels",
            SnapshotKeyFamily::Section => "sec:",
            SnapshotKeyFamily::DeletedSection => "del:",
            SnapshotKeyFamily::Children => "ch:",
            SnapshotKeyFamily::ChildrenIndex => "chs",
            SnapshotKeyFamily::FieldsIndex => "flds",
            SnapshotKeyFamily::TimeSamplesIndex => "tss",
            SnapshotKeyFamily::Field => "fld:",
            SnapshotKeyFamily::TimeSample => "ts:",
            SnapshotKeyFamily::ExtIndex => "ext",
        }
    }

    /// Every key the script reads is built from `base` — `ARGV[1] .. '{' ..
    /// ARGV[2] .. '}:obj:'` — or is the guarded `KEYS[1]`; each call's
    /// refusal label names the family whose key it reads; every snapshot
    /// family is read, and `ext_index` (`obj:ext`, external-value clean-up's
    /// family) never is; and each `base .. <suffix>` key is byte-for-byte the
    /// key `KeySchema` builds for that family, given the `ARGV` the Rust side
    /// passes. So every key carries the `{oid}` tag of the declared header.
    #[test]
    fn object_snapshot_read_builds_every_key_under_the_objects_hash_tag() {
        let code = object_snapshot_read_code().join("\n");
        assert!(
            code.contains("local base = ARGV[1] .. '{' .. ARGV[2] .. '}:obj:'"),
            "every undeclared key is built on `base`, which must be the prefix plus the object's \
             own hash tag"
        );
        assert!(
            code.contains("if hdr_key ~= base .. 'hdr' then"),
            "the script must refuse a KEYS[1] that is not the header `base` names, or the slot it \
             runs in need not be the tag's"
        );
        assert!(
            code.contains("local probe = base .. 'secs'"),
            "the self-test's probe must be built on `base` too"
        );

        let calls = object_snapshot_read_calls();
        assert!(
            calls.len() >= 10,
            "the call scan found too few calls: {calls:?}"
        );
        let mut families_read = std::collections::BTreeSet::new();
        for (family, command, key) in &calls {
            assert!(
                key == "hdr_key" || key.starts_with("base .. "),
                "call({family:?}, {command}, {key}) reads a key not built on `base`"
            );
            let Some(label) = family else { continue };
            let family = SnapshotKeyFamily::ALL
                .into_iter()
                .find(|f| f.label() == label)
                .unwrap_or_else(|| panic!("call label {label:?} is not a SnapshotKeyFamily label"));
            families_read.insert(family.label());
            if key == "hdr_key" {
                assert_eq!(
                    family,
                    SnapshotKeyFamily::Header,
                    "{label} reads the header"
                );
            } else {
                let suffix = key
                    .strip_prefix("base .. '")
                    .and_then(|s| s.split('\'').next())
                    .unwrap_or_else(|| panic!("{key} must start with a literal suffix"));
                assert_eq!(
                    suffix,
                    family_key_suffix(family),
                    "a refusal on {key} would be labelled {label}, which names another key family"
                );
            }
        }
        // `read_hashes('<family>', <ids>, '<suffix>')` passes the family and
        // the suffix through to a `call(family, 'HGETALL', base .. sub .. id)`.
        for line in object_snapshot_read_code() {
            let Some(args) = line.split("read_hashes(").nth(1) else {
                continue;
            };
            let args: Vec<&str> = args.split(',').map(str::trim).collect();
            let (Some(label), Some(suffix)) = (
                args[0]
                    .strip_prefix('\'')
                    .and_then(|s| s.strip_suffix('\'')),
                args.get(2)
                    .and_then(|s| s.strip_prefix('\''))
                    .and_then(|s| s.split('\'').next()),
            ) else {
                continue;
            };
            let family = SnapshotKeyFamily::ALL
                .into_iter()
                .find(|f| f.label() == label)
                .unwrap_or_else(|| panic!("read_hashes label {label:?} is not a family label"));
            assert_eq!(
                suffix,
                family_key_suffix(family),
                "read_hashes('{label}', …)"
            );
            families_read.insert(label);
        }
        // Every family but one: `ExtIndex` (`obj:ext`) is external-value
        // clean-up's family (`storage::retirement`'s size read and reference
        // walk), which no snapshot read touches — so the script must never
        // read it, under its label or by its key.
        let all: std::collections::BTreeSet<&str> = SnapshotKeyFamily::ALL
            .iter()
            .filter(|f| **f != SnapshotKeyFamily::ExtIndex)
            .map(|f| f.label())
            .collect();
        assert!(
            !families_read.contains(SnapshotKeyFamily::ExtIndex.label()),
            "the snapshot script must not read obj:ext: it is no part of an object's state"
        );
        assert!(
            !code.contains("base .. 'ext'"),
            "the snapshot script must not build an obj:ext key"
        );
        assert_eq!(
            families_read
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>(),
            all,
            "every snapshot key family must be read under its own label"
        );

        // The Rust side: ARGV[1..2] are the prefix and the object id, and
        // KEYS[1] is exactly the header `base` names — and every `base`
        // suffix the script uses builds the key `KeySchema` builds.
        let keys = KeySchema::new("test_tag:");
        let oid = "1234";
        let argv = object_snapshot_read_argv(&keys, oid, "read", 10, 20);
        let base = format!("{}{{{}}}:obj:", argv[0], argv[1]);
        assert_eq!(
            (argv[0].as_str(), argv[1].as_str(), argv[2].as_str()),
            ("test_tag:", oid, "read")
        );
        let member = KeySchema::field_member("5", "color");
        let ts_member = KeySchema::time_sample_member("5", 1.5);
        for (built, schema) in [
            (format!("{base}hdr"), keys.obj_header(oid)),
            (format!("{base}secs"), keys.obj_sections_index(oid)),
            (format!("{base}dels"), keys.obj_deleted_sections_index(oid)),
            (format!("{base}flds"), keys.obj_fields_index(oid)),
            (format!("{base}tss"), keys.obj_time_samples_index(oid)),
            (format!("{base}chs"), keys.obj_children_index(oid)),
            (format!("{base}sec:5"), keys.obj_section(oid, "5")),
            (format!("{base}del:5"), keys.obj_deleted_section(oid, "5")),
            (
                format!("{base}fld:{member}"),
                keys.obj_field(oid, "5", "color"),
            ),
            (
                format!("{base}ts:{ts_member}"),
                keys.obj_time_sample(oid, "5", 1.5),
            ),
            (format!("{base}ch:5"), keys.obj_children(oid, "5")),
            // Not read by the script (above), but `family_key_suffix` names
            // its suffix all the same, so pin it to the key it stands for.
            (format!("{base}ext"), keys.obj_ext_index(oid)),
        ] {
            assert_eq!(
                built, schema,
                "the script and KeySchema must build the same key"
            );
            assert!(built.starts_with(&format!("test_tag:{{{oid}}}:obj:")));
        }
    }

    /// Read-only, unflagged, and every read through `redis.pcall` (SR-1.1,
    /// SR-1.4): no write command, no `#!lua` shebang (so it runs in compat
    /// mode, SDD §7.5.1, which the write scripts' OOM classification relies on
    /// and in which the read keeps running on a store over `maxmemory`), and no
    /// `redis.call`, whose error would abort the script untagged instead of
    /// naming the key family.
    #[test]
    fn object_snapshot_read_script_is_read_only_unflagged_and_reads_through_pcall() {
        assert!(!OBJECT_SNAPSHOT_READ_LUA.starts_with("#!"), "no shebang");
        let code = object_snapshot_read_code().join("\n");
        assert!(
            !code.contains("redis.call("),
            "every read goes through redis.pcall"
        );
        const READS: [&str; 6] = ["HGETALL", "SCARD", "ZCARD", "SMEMBERS", "ZRANGE", "LRANGE"];
        for (family, command, key) in object_snapshot_read_calls() {
            assert!(
                READS.contains(&command.as_str()),
                "call({family:?}, {command}, {key}) is not a read"
            );
        }
        assert!(
            code.contains("redis.pcall('EXISTS', probe)"),
            "the self-test's one command is a read of its probe"
        );
    }

    /// The `slim` flag (`ARGV[13]`, b.mr6) reshapes the value rows only and
    /// reads nothing of its own: every `read_hashes` call passes
    /// `value_rows` true for exactly the field and time-sample families, the
    /// row is read by the same `HGETALL` and tallied before the flag is
    /// consulted, and the shape builder issues no command. So the item, byte
    /// and entry tallies of a slim read are a full read's (the integration
    /// twin compares the replies: `structural_cache_cold_start_test`'s
    /// `object_snapshot_read_slim_*`).
    #[test]
    fn object_snapshot_read_slim_flag_reshapes_only_the_value_rows_and_reads_nothing_more() {
        let code = object_snapshot_read_code();
        let mut value_row_families = std::collections::BTreeMap::new();
        for line in &code {
            let Some(args) = line.split("read_hashes(").nth(1) else {
                continue;
            };
            let args: Vec<&str> = args.split(',').map(str::trim).collect();
            let Some(label) = args[0]
                .strip_prefix('\'')
                .and_then(|s| s.strip_suffix('\''))
            else {
                continue;
            };
            let value_rows = args
                .get(3)
                .map(|a| a.trim_end_matches([')', ' ']))
                .unwrap_or_else(|| panic!("read_hashes('{label}', …) passes value_rows"));
            value_row_families.insert(label, value_rows == "true");
        }
        assert_eq!(
            value_row_families,
            std::collections::BTreeMap::from([
                ("deleted_section", false),
                ("field", true),
                ("section", false),
                ("time_sample", true),
            ]),
            "the slim flag applies to the value rows only"
        );

        let source = code.join("\n");
        let body = source
            .split("local function slim_row(")
            .nth(1)
            .and_then(|rest| rest.split("\n  end\n").next())
            .expect("the script defines slim_row");
        assert!(
            !body.contains("call(") && !body.contains("redis."),
            "slim_row only reshapes the row it is handed: {body}"
        );
        let read = source
            .split("local function read_hashes(")
            .nth(1)
            .expect("the script defines read_hashes");
        let (hgetall, tally, shape) = (
            read.find("'HGETALL'").expect("read_hashes reads each row"),
            read.find("add_hash(h)")
                .expect("read_hashes tallies each row"),
            read.find("slim_row(")
                .expect("read_hashes shapes a slim row"),
        );
        assert!(
            hgetall < tally && tally < shape,
            "each row is read whole and tallied before it is shaped"
        );
        assert!(
            source.contains("local slim = slim_arg == '1'"),
            "only '1' sets the flag"
        );
    }

    // ── The bounded fan-out's windows (Spec SDD SR-11.3) ────────────────────
    //
    // `fanout_window_len` is pure arithmetic, pinned row by row. Which average
    // it is handed — entry replies once one held bytes, every reply before —
    // is decided inside `FanoutWindows::run`, and the property the windows
    // exist for (no window is issued until the previous window's replies have
    // been decoded) is a matter of when commands reach the store, so those are
    // pinned against a real Valkey here, where the private `FanoutWindows` can
    // be built with chosen bounds. The integration twin — the whole read at a
    // one-command, one-byte window, equal to the atomic read — is in
    // `tests/integration/storage/snapshot_read_unfenced_test.rs`.
    //
    // The Valkey-backed tests observe window boundaries without a clock. Every
    // command `LRANGE`s one marker list, and the sink appends one marker to that
    // list for every reply it is handed. A command therefore reads exactly as
    // many markers as replies the sink had taken in before the store ran it:
    // under correct windowing, every command of a window reads the count of all
    // earlier windows' commands, and none of its own window's. A window issued
    // early, a window one command wider, or a sink called only at the end of the
    // round each changes those counts.

    /// Nothing observed yet: the byte target has no average to scale from, so
    /// the command window alone applies, capped by the commands left.
    #[test]
    fn fanout_window_len_with_nothing_observed_is_the_command_window_capped_by_what_remains() {
        assert_eq!(fanout_window_len(8, 100, 0, 0, 20), 8);
        assert_eq!(fanout_window_len(8, 100, 0, 0, 3), 3);
    }

    /// The byte target divided by the observed average, rounded down: 250 B
    /// over 10 replies is 25 B a reply, and a 100 B target admits 4; at 26 B a
    /// reply it admits 3.84, so 3.
    #[test]
    fn fanout_window_len_fits_the_byte_target_at_the_observed_average_rounding_down() {
        assert_eq!(fanout_window_len(64, 100, 10, 250, 1_000), 4);
        assert_eq!(fanout_window_len(64, 100, 10, 260, 1_000), 3);
    }

    /// The command window is the hard bound: 1 B a reply against a 1,000 B
    /// target would admit 1,000 commands.
    #[test]
    fn fanout_window_len_never_exceeds_the_command_window() {
        assert_eq!(fanout_window_len(8, 1_000, 10, 10, 1_000), 8);
    }

    /// One reply cannot be split, so an average above the byte target still
    /// admits one command — never zero, which would stop the read.
    #[test]
    fn fanout_window_len_is_one_command_once_the_average_exceeds_the_byte_target() {
        assert_eq!(fanout_window_len(64, 100, 1, 5_000, 1_000), 1);
    }

    /// Replies observed but none held a byte: there is still no size to scale
    /// from, so the command window applies.
    #[test]
    fn fanout_window_len_over_replies_that_held_no_bytes_is_the_command_window() {
        assert_eq!(fanout_window_len(8, 100, 12, 0, 1_000), 8);
    }

    /// `max_bytes × commands_seen` overflows `u64` at the extremes; the
    /// product is taken in `u128`, so the answer is still the bound that
    /// applies (here the commands left) rather than a panic or a wrapped value.
    #[test]
    fn fanout_window_len_does_not_overflow_at_the_extremes() {
        assert_eq!(fanout_window_len(u64::MAX, u64::MAX, u64::MAX, 1, 5), 5);
        assert_eq!(fanout_window_len(3, u64::MAX, u64::MAX, 1, usize::MAX), 3);
    }

    /// The entry basis is exactly the five per-entry families: one reply per
    /// section, tombstone, field or time-sample hash and per children list.
    /// The header and the five indices (`obj:chs` included) are left out — one
    /// index reply holds a member per entry, so averaging it in would size the
    /// entry windows many times too small.
    #[test]
    fn is_fanout_entry_is_true_for_exactly_the_per_entry_families() {
        const ENTRIES: [SnapshotKeyFamily; 5] = [
            SnapshotKeyFamily::Section,
            SnapshotKeyFamily::DeletedSection,
            SnapshotKeyFamily::Children,
            SnapshotKeyFamily::Field,
            SnapshotKeyFamily::TimeSample,
        ];
        for family in SnapshotKeyFamily::ALL {
            assert_eq!(
                is_fanout_entry(family),
                ENTRIES.contains(&family),
                "{family:?}: the entry basis must be exactly the per-entry families"
            );
        }
    }

    /// A [`FanoutWindows`] over `client`'s snapshot lane with the given bounds
    /// and nothing observed — the state [`FanoutWindows::new`] starts a read
    /// in, at bounds small enough to see.
    fn fanout_windows<'a>(
        client: &'a ValkeyClient,
        pool: &'a fred::clients::Pool,
        max_commands: u64,
        max_bytes: u64,
    ) -> FanoutWindows<'a> {
        FanoutWindows {
            client,
            pool,
            // The default schema, leaked like the sink below so it outlives
            // any borrow these tests take; a few bytes per call.
            keys: Box::leak(Box::new(KeySchema::default())),
            object_id: "7",
            transport: WindowTransport::Script,
            fell_back: false,
            max_commands,
            max_bytes,
            commands_seen: 0,
            bytes_seen: 0,
            entry_commands_seen: 0,
            entry_bytes_seen: 0,
            fault_cursor: SnapshotReadFaultCursor::unfenced_fanout(),
            tally: FanoutTally::default(),
            // A no-op sink that lives as long as any borrow these tests
            // take: a zero-sized closure, so leaking it allocates nothing.
            progress: ProgressSink::Local(Box::leak(Box::new(|_: u64| {}))),
            allowance: AllowanceSeam::None,
            replies: FanoutReplies::Whole,
            window_bound: None,
        }
    }

    /// `n` `LRANGE`s of `key`, as children-list entry commands.
    fn lranges_of(key: &str, n: usize) -> Vec<FanoutCommand> {
        (0..n)
            .map(|i| {
                FanoutCommand::new(SnapshotKeyFamily::Children, key.to_owned(), &i.to_string())
            })
            .collect()
    }

    /// The marker the sink appends: one byte, so a reply of `m` markers is
    /// `m` items and `m` bytes.
    const MARKER: &str = "m";

    /// Append one [`MARKER`] to `key` from inside a fan-out sink, which is
    /// synchronous: the push blocks the sink until the store has it, so the
    /// next window cannot be issued before it lands. Needs a multi-thread
    /// runtime: the blocking bridge is `test_support::rpush_blocking`, kept
    /// there because the blocking-bridge fence exempts that file only.
    fn push_marker(client: &ValkeyClient, key: &str) {
        crate::storage::test_support::rpush_blocking(client, key, MARKER);
    }

    /// The number of items in each reply the sink was handed, in the order it
    /// was handed them — `run`'s command order.
    type ReplyItems = Vec<(String, u64)>;

    /// Run `commands` through `windows` with a sink that records each reply's
    /// item count and then appends one marker to `marker_key`.
    async fn run_with_markers(
        windows: &mut FanoutWindows<'_>,
        commands: Vec<FanoutCommand>,
        client: &ValkeyClient,
        marker_key: &str,
    ) -> ReplyItems {
        let mut seen = Vec::new();
        windows
            .run(commands, |_, command, reply| {
                seen.push((command.id.clone(), reply.items()));
                push_marker(client, marker_key);
                Ok(())
            })
            .await
            .unwrap_or_else(|e| panic!("the fan-out round must succeed; got {e:?}"));
        seen
    }

    /// The bounded-memory property of the fan-out (SR-11.3): at most one
    /// window of commands is outstanding, and every reply is handed to the sink
    /// — decoded — before the next window is issued.
    ///
    /// Ten commands at a three-command window read 0, 0, 0, 3, 3, 3, 6, 6, 6
    /// and 9 markers (see the section comment for why). A reader that issued
    /// the whole round at once reads ten zeros; one that sized a window at four
    /// reads 3 markers' worth too few at the fourth command; one that called
    /// the sink only once the round was done reads ten zeros as well. The byte
    /// target is far out of reach, so the command window alone sizes every
    /// window.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fanout_windows_issue_each_window_only_after_the_previous_windows_replies_are_decoded()
    {
        use crate::storage::test_support::TestValkey;

        const WINDOW: u64 = 3;
        const COMMANDS: usize = 10;

        let tv = TestValkey::setup().await;
        let marker_key = format!("{}fanout-window:markers", tv.prefix());
        let (pool, _in_use) = tv.client().acquire_snapshot();
        let mut windows = fanout_windows(tv.client(), pool, WINDOW, u64::MAX / 2);

        let seen = run_with_markers(
            &mut windows,
            lranges_of(&marker_key, COMMANDS),
            tv.client(),
            &marker_key,
        )
        .await;

        let expected: ReplyItems = (0..COMMANDS)
            .map(|i| (i.to_string(), (i as u64 / WINDOW) * WINDOW))
            .collect();
        assert_eq!(
            seen, expected,
            "each command must read the markers of every earlier window and none of its own: at \
             most {WINDOW} commands may be outstanding, and a window's replies must all reach the \
             sink before the next window is issued (so no raw reply outlives its window)"
        );
    }

    /// The empty-entry fallback (SR-11.3): entry replies that held no bytes —
    /// indexed hashes deleted under the read — carry no size, so the next
    /// window is sized from every reply seen (the header and the indices),
    /// not from the full command window; once an entry reply holds bytes, the
    /// entry replies alone size the windows.
    ///
    /// The test seeds the observed basis with five non-entry replies, 50 bytes
    /// in all: the header and four empty indices, each read whole — the basis
    /// the paged round 1 leaves on such an object too, its first window being
    /// the header and four member counts (which hold no bytes) and an empty
    /// index having no page. With a 40-byte target and a 64-command window,
    /// round 2 is four `LRANGE`s of an empty list then thirty of the marker
    /// list, and the windows come out:
    ///
    /// | window | basis (commands, bytes) | size | each `LRANGE` of the marker list reads |
    /// |---|---|---|---|
    /// | 1 | all replies (5, 50) | ⌊40·5/50⌋ = 4 | (the empty list) |
    /// | 2 | all replies (9, 50) — the four entries held no bytes | ⌊40·9/50⌋ = 7 | 4 |
    /// | 3 | entries (11, 28) | ⌊40·11/28⌋ = 15 | 11 |
    /// | 4 | entries (26, 193) | ⌊40·26/193⌋ = 5 | 26 |
    /// | 5 | entries (31, 323) | ⌊40·31/323⌋ = 3 (all that remain) | 31 |
    ///
    /// A reader that took the empty entry basis at face value would size window
    /// 2 at the whole command window — all thirty marker reads would read 4; one
    /// that never switched to the entry basis would size window 3 from (16, 78)
    /// at 8 commands. Either fails the row-by-row comparison.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fanout_windows_size_past_empty_entry_replies_from_every_reply_seen() {
        use fred::interfaces::HashesInterface;

        use crate::storage::test_support::TestValkey;

        const MAX_COMMANDS: u64 = 64;
        const MAX_BYTES: u64 = 40;
        const EMPTY_ENTRIES: usize = 4;
        const MARKER_READS: usize = 30;

        let tv = TestValkey::setup().await;
        let base = format!("{}fanout-fallback:", tv.prefix());
        let header_key = format!("{base}hdr");
        let empty_key = format!("{base}empty");
        let marker_key = format!("{base}markers");
        // One field whose name and value come to 50 bytes: round 1's only bytes.
        let _: () = tv
            .client()
            .pool
            .hset(&header_key, [("f", "v".repeat(49))])
            .await
            .expect("HSET of the round-1 header must succeed");

        let (pool, _in_use) = tv.client().acquire_snapshot();
        let mut windows = fanout_windows(tv.client(), pool, MAX_COMMANDS, MAX_BYTES);

        let round_1 = vec![
            FanoutCommand::new(SnapshotKeyFamily::Header, header_key.clone(), ""),
            FanoutCommand::new(SnapshotKeyFamily::SectionsIndex, format!("{base}secs"), ""),
            FanoutCommand::new(
                SnapshotKeyFamily::DeletedSectionsIndex,
                format!("{base}dels"),
                "",
            ),
            FanoutCommand::new(SnapshotKeyFamily::FieldsIndex, format!("{base}flds"), ""),
            FanoutCommand::new(
                SnapshotKeyFamily::TimeSamplesIndex,
                format!("{base}tss"),
                "",
            ),
        ];
        let mut round_1_bytes = 0;
        windows
            .run(round_1, |_, _, reply| {
                round_1_bytes += reply.bytes();
                Ok(())
            })
            .await
            .unwrap_or_else(|e| panic!("round 1 must succeed; got {e:?}"));
        assert_eq!(
            (windows.commands_seen, windows.bytes_seen, round_1_bytes),
            (5, 50, 50),
            "fixture precondition: round 1 is five replies holding 50 bytes, all in the header"
        );

        let mut round_2 = lranges_of(&empty_key, EMPTY_ENTRIES);
        round_2.extend(lranges_of(&marker_key, MARKER_READS));
        let seen = run_with_markers(&mut windows, round_2, tv.client(), &marker_key).await;

        let marker_reads: Vec<u64> = seen[EMPTY_ENTRIES..].iter().map(|(_, n)| *n).collect();
        let expected: Vec<u64> = [(4, 7), (11, 15), (26, 5), (31, 3)]
            .into_iter()
            .flat_map(|(markers, window)| std::iter::repeat_n(markers, window))
            .collect();
        assert_eq!(
            seen[..EMPTY_ENTRIES]
                .iter()
                .map(|(_, n)| *n)
                .collect::<Vec<_>>(),
            vec![0; EMPTY_ENTRIES],
            "fixture precondition: the first four entry replies are empty"
        );
        assert_eq!(
            marker_reads, expected,
            "windows 2–5 must be 7, 15, 5 and 3 commands: past entry replies that held no bytes \
             the window is sized from every reply seen, and from the entry replies alone once one \
             holds bytes"
        );
    }

    /// An entry larger than the byte target still completes, and once the
    /// observed average is over the target every later window is one command:
    /// one reply cannot be split, and a window never admits fewer than one.
    /// Ten 10-byte elements (100 bytes) against a 10-byte target: the first
    /// read — a one-command round, so alone by construction — completes whole,
    /// and each of the next three reads one more marker than the last, each
    /// issued only after its predecessor's reply reached the sink.
    ///
    /// An over-target reply sharing a window sized from a smaller average is
    /// the next test's case.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fanout_windows_complete_an_over_target_entry_and_shrink_to_one_command_once_the_average_exceeds_the_target(
    ) {
        use fred::interfaces::ListInterface;

        use crate::storage::test_support::TestValkey;

        const MAX_BYTES: u64 = 10;
        const ELEMENTS: u64 = 10;

        let tv = TestValkey::setup().await;
        let big_key = format!("{}fanout-over-window:big", tv.prefix());
        let elements: Vec<String> = (0..ELEMENTS).map(|i| format!("element-{i:02}")).collect();
        let _: i64 = tv
            .client()
            .pool
            .rpush(&big_key, elements)
            .await
            .expect("RPUSH of the over-window list must succeed");

        let (pool, _in_use) = tv.client().acquire_snapshot();
        let mut windows = fanout_windows(tv.client(), pool, 64, MAX_BYTES);

        let mut first = Vec::new();
        windows
            .run(lranges_of(&big_key, 1), |_, _, reply| {
                first.push(reply.bytes());
                Ok(())
            })
            .await
            .unwrap_or_else(|e| {
                panic!("an entry over the byte target must still be read; got {e:?}")
            });
        assert_eq!(
            first,
            vec![ELEMENTS * 10],
            "the over-window entry must come back whole"
        );

        let seen =
            run_with_markers(&mut windows, lranges_of(&big_key, 3), tv.client(), &big_key).await;
        assert_eq!(
            seen.iter().map(|(_, n)| *n).collect::<Vec<_>>(),
            vec![ELEMENTS, ELEMENTS + 1, ELEMENTS + 2],
            "with the observed average over the byte target every window must be one command"
        );
    }

    /// defer-11 (SR-11.3): a window is sized before its own replies arrive, so
    /// a reply far over the byte target can land in a multi-command window
    /// sized from a smaller average — and the round still completes, every
    /// reply whole, after which the windows shrink to one command.
    ///
    /// A 4-command window and a 10-byte target. One read of a one-byte list
    /// sets the entry average at 1 byte, so the next window is
    /// min(4, ⌊10 · 1 / 1⌋) = 4 commands: the one-byte list, again, then a
    /// 100-byte list, then the one-byte list again — all issued together. The
    /// sink appends a marker to the one-byte list per reply, so every read of
    /// it in that one window sees 1 element; a read issued in a later window
    /// would see the markers. The average is then 104 bytes over 5 commands,
    /// over the target, so the round after it reads one command per window:
    /// 5 elements (the list and the four markers), then 6.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fanout_windows_complete_a_round_whose_over_target_reply_shares_a_window_sized_from_a_smaller_average(
    ) {
        use fred::interfaces::ListInterface;

        use crate::storage::test_support::TestValkey;

        const MAX_COMMANDS: u64 = 4;
        const MAX_BYTES: u64 = 10;
        const BIG_ELEMENTS: u64 = 10;

        let tv = TestValkey::setup().await;
        let small_key = format!("{}fanout-shared-window:small", tv.prefix());
        let big_key = format!("{}fanout-shared-window:big", tv.prefix());
        let _: i64 = tv
            .client()
            .pool
            .rpush(&small_key, vec!["s".to_string()])
            .await
            .expect("RPUSH of the one-byte list must succeed");
        let big: Vec<String> = (0..BIG_ELEMENTS)
            .map(|i| format!("element-{i:02}"))
            .collect();
        let _: i64 = tv
            .client()
            .pool
            .rpush(&big_key, big)
            .await
            .expect("RPUSH of the over-target list must succeed");

        let (pool, _in_use) = tv.client().acquire_snapshot();
        let mut windows = fanout_windows(tv.client(), pool, MAX_COMMANDS, MAX_BYTES);
        let read = |key: &str, id: &str| {
            FanoutCommand::new(SnapshotKeyFamily::Children, key.to_owned(), id)
        };

        // The first round's marker goes to a list nothing reads.
        let first = run_with_markers(
            &mut windows,
            vec![read(&small_key, "a")],
            tv.client(),
            &format!("{}fanout-shared-window:unread", tv.prefix()),
        )
        .await;
        assert_eq!(
            (first, windows.entry_commands_seen, windows.entry_bytes_seen),
            (vec![("a".to_string(), 1)], 1, 1),
            "fixture precondition: one one-byte entry sets the average the next window is sized \
             from"
        );

        let shared = run_with_markers(
            &mut windows,
            vec![
                read(&small_key, "b"),
                read(&small_key, "c"),
                read(&big_key, "big"),
                read(&small_key, "d"),
            ],
            tv.client(),
            &small_key,
        )
        .await;
        assert_eq!(
            shared,
            vec![
                ("b".to_string(), 1),
                ("c".to_string(), 1),
                ("big".to_string(), BIG_ELEMENTS),
                ("d".to_string(), 1),
            ],
            "the over-target reply must come back whole from a window it shared with three \
             commands sized from the one-byte average — and the round must complete"
        );

        let after = run_with_markers(
            &mut windows,
            vec![read(&small_key, "e"), read(&small_key, "f")],
            tv.client(),
            &small_key,
        )
        .await;
        assert_eq!(
            after,
            vec![("e".to_string(), 5), ("f".to_string(), 6)],
            "once the average is over the byte target every window is one command"
        );
    }

    /// [`FanoutWindows::new`] takes its bounds from the client's configured
    /// atomic-read limits — `FANOUT_WINDOW` from the item ceiling, the byte
    /// target from the byte ceiling — so an operator who lowers a ceiling
    /// narrows the windows with it. Distinct limits, so a swap is caught.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fanout_windows_take_their_bounds_from_the_clients_atomic_read_limits() {
        use crate::storage::test_support::TestValkey;

        let tv = TestValkey::setup().await;
        let mut client = tv.client().clone();
        client.snapshot_atomic_read_max_items = 100;
        client.snapshot_atomic_read_max_bytes = 4_096;
        let (pool, _in_use) = client.acquire_snapshot();

        let mut no_progress = |_: u64| {};
        let windows = FanoutWindows::new(
            &client,
            pool,
            tv.key_schema(),
            "7",
            SnapshotReadFaultCursor::unfenced_fanout(),
            ProgressSink::Local(&mut no_progress),
            FanoutReplies::Whole,
            AllowanceSeam::None,
        );

        assert_eq!(
            (windows.max_commands, windows.max_bytes),
            (18, 4_096),
            "a 100-item ceiling is a ⌊100 ÷ 5.266⌋ = 18-command window, and the byte target is \
             the byte ceiling"
        );
        assert_eq!(
            (
                windows.commands_seen,
                windows.bytes_seen,
                windows.entry_commands_seen,
                windows.entry_bytes_seen
            ),
            (0, 0, 0, 0),
            "a read starts with nothing observed"
        );
    }

    /// Capture sink for [`timing_lines_under`]: the tree-wide
    /// `CapturingWriter` idiom (`src/grpc/rpc_log.rs::tests`).
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
            self.0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The JSON lines one [`log_snapshot_read_timing`] call emits under a
    /// JSON subscriber filtered by the `EnvFilter` directives `filter`, as
    /// the service's log level is.
    fn timing_lines_under(filter: &str) -> Vec<serde_json::Value> {
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(std::sync::Arc::clone(&buf)))
            .with_env_filter(tracing_subscriber::EnvFilter::new(filter))
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            log_snapshot_read_timing(
                "42",
                SnapshotReadConsumer::Catchup,
                SnapshotReadPath::Atomic,
                metrics::SNAPSHOT_READ_OUTCOME_OK,
                std::time::Duration::from_micros(49_213),
            );
        });
        let raw = buf.lock().unwrap_or_else(|p| p.into_inner()).clone();
        String::from_utf8(raw)
            .expect("captured tracing output must be UTF-8")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("each captured line must be JSON"))
            .collect()
    }

    /// The per-read timing line is off at the service's default `info` and
    /// on under its own target's directive, and carries the read's object,
    /// the metric's consumer, path and outcome labels, and its duration in
    /// whole microseconds: the per-read durations a measurement takes its
    /// percentiles from.
    #[test]
    fn snapshot_read_timing_line_is_off_at_info_and_carries_each_reads_labels_and_duration() {
        assert!(
            timing_lines_under("info").is_empty(),
            "the timing line must stay off at the default log level"
        );
        let lines = timing_lines_under("info,live_edit::snapshot_read_timing=debug");
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["target"], SNAPSHOT_READ_TIMING_TARGET);
        assert_eq!(line["level"], "DEBUG");
        let fields = &line["fields"];
        assert_eq!(fields["message"], SNAPSHOT_READ_TIMING_MSG);
        assert_eq!(fields["object_id"], "42");
        assert_eq!(fields["consumer"], metrics::SNAPSHOT_READ_CONSUMER_CATCHUP);
        assert_eq!(fields["path"], metrics::SNAPSHOT_READ_PATH_ATOMIC);
        assert_eq!(fields["outcome"], metrics::SNAPSHOT_READ_OUTCOME_OK);
        assert_eq!(fields["elapsed_us"], 49_213);
    }

    // ── Round 1, bounded in index members (Spec SDD SR-11.3, SR-11.6 (i)) ──
    //
    // `share_index_window` is pure arithmetic, pinned row by row. Round 1
    // itself (`fanout_round_one`) is pinned here against a real Valkey on bare
    // index keys, where the members it collects can be compared with the
    // whole-index reads; the whole fan-out over a layer whose indices each
    // take several pages, its windows' member counts (from the `fan-out
    // window` event) and `SSCAN` repeats are pinned in
    // `tests/integration/storage/snapshot_read_{unfenced,fenced}_test.rs`.

    /// Every ask that fits is met in full; otherwise the smallest asks are met
    /// and the rest share the remainder equally (water-filling), every
    /// positive ask getting at least one member while the budget lasts; a
    /// zero ask gets nothing; the shares never sum past the budget.
    #[test]
    fn share_index_window_meets_small_asks_and_shares_the_rest_equally() {
        let rows: [(u64, &[u64], &[u64]); 8] = [
            (100, &[10, 20, 30, 0], &[10, 20, 30, 0]),
            (100, &[10, 200, 300, 0], &[10, 45, 45, 0]),
            // The level is 30: 10 + 3 × 30 = 100, so the 40-member ask is cut.
            (100, &[200, 10, 300, 40], &[30, 10, 30, 30]),
            (100, &[0, 0, 0, 0], &[0, 0, 0, 0]),
            (3, &[5, 5, 5, 5], &[1, 1, 1, 0]),
            (4, &[5, 5, 5, 5], &[1, 1, 1, 1]),
            (7, &[100, 100], &[3, 4]),
            (26_855, &[24_000, 0, 40_000, 5], &[13_425, 0, 13_425, 5]),
        ];
        for (budget, demands, want) in rows {
            let got = share_index_window(budget, demands);
            assert_eq!(got, want, "budget {budget}, demands {demands:?}");
            assert!(
                got.iter().sum::<u64>() <= budget,
                "the shares fit the budget: {got:?} of {budget}"
            );
        }
    }

    /// Bare round-1 keys of object "7" under `tv`'s prefix: a header, a
    /// sections set of `sections` numeric members (past the compact
    /// encodings, so `SSCAN` really pages), a three-member deleted-sections
    /// set, a fields sorted set of `fields` members and a five-member
    /// time-samples sorted set.
    async fn round_one_keys(
        tv: &crate::storage::test_support::TestValkey,
        sections: u64,
        fields: u64,
    ) -> KeySchema {
        use fred::interfaces::{HashesInterface, SetsInterface, SortedSetsInterface};

        let keys = tv.key_schema().clone();
        let pool = &tv.client().pool;
        let _: () = pool
            .hset(keys.obj_header("7"), [("seq", "1")])
            .await
            .expect("HSET of the header must succeed");
        let section_ids: Vec<String> = (1..=sections).map(|i| i.to_string()).collect();
        let _: i64 = pool
            .sadd(keys.obj_sections_index("7"), section_ids)
            .await
            .expect("SADD of the sections index must succeed");
        let _: i64 = pool
            .sadd(
                keys.obj_deleted_sections_index("7"),
                vec!["900001", "900002", "900003"],
            )
            .await
            .expect("SADD of the deleted-sections index must succeed");
        let field_members: Vec<(f64, String)> = (0..fields)
            .map(|i| {
                (
                    0.0,
                    KeySchema::field_member(&(i % 97).to_string(), &format!("f{i}")),
                )
            })
            .collect();
        let _: i64 = pool
            .zadd(
                keys.obj_fields_index("7"),
                None,
                None,
                false,
                false,
                field_members,
            )
            .await
            .expect("ZADD of the fields index must succeed");
        let samples: Vec<(f64, String)> = (0..5)
            .map(|i| (0.0, KeySchema::time_sample_member("3", f64::from(i) + 0.5)))
            .collect();
        let _: i64 = pool
            .zadd(
                keys.obj_time_samples_index("7"),
                None,
                None,
                false,
                false,
                samples,
            )
            .await
            .expect("ZADD of the time-samples index must succeed");
        keys
    }

    /// Round 1 over indices that each take more than one page — 12,000
    /// sections (`SSCAN` asked for about 6,700 a step) and 40,000 fields (at
    /// most 26,855 members a window, all indices together) — collects exactly
    /// what the whole-index reads return: each set's members once, each
    /// sorted index in rank order, the header whole; and it took more than
    /// one window of pages.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fanout_round_one_pages_collect_exactly_the_whole_indices() {
        use fred::interfaces::{SetsInterface, SortedSetsInterface};

        use crate::storage::test_support::TestValkey;

        const SECTIONS: u64 = 12_000;
        const FIELDS: u64 = 40_000;

        let tv = TestValkey::setup().await;
        let keys = round_one_keys(&tv, SECTIONS, FIELDS).await;
        let (pool, _in_use) = tv.client().acquire_snapshot();
        let mut windows = fanout_windows(tv.client(), pool, 2_141, u64::MAX / 2);

        let round = fanout_round_one(&mut windows, &keys, "7")
            .await
            .unwrap_or_else(|e| panic!("round 1 must succeed; got {e:?}"))
            .expect("an uncharged read never stops");

        let data = &tv.client().pool;
        let whole_set = |key: String| async move {
            let members: Vec<String> = data.smembers(key).await.unwrap();
            members
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
        };
        let whole_sorted = |key: String| async move {
            let members: Vec<String> = data
                .zrange(key, 0, -1, None, false, None, false)
                .await
                .unwrap();
            members
        };
        assert_eq!(
            round.header,
            HashMap::from([("seq".to_owned(), "1".to_owned())])
        );
        for (what, got, key) in [
            ("sections", &round.section_ids, keys.obj_sections_index("7")),
            (
                "deleted sections",
                &round.deleted_section_ids,
                keys.obj_deleted_sections_index("7"),
            ),
        ] {
            let collected: std::collections::BTreeSet<String> = got.iter().cloned().collect();
            assert_eq!(
                collected.len(),
                got.len(),
                "{what}: a set's members are collected once each"
            );
            assert_eq!(collected, whole_set(key).await, "{what}: every member");
        }
        assert_eq!(round.section_ids.len() as u64, SECTIONS);
        assert_eq!(
            round.field_members,
            whole_sorted(keys.obj_fields_index("7")).await,
            "fields: every member, in rank order"
        );
        assert_eq!(round.field_members.len() as u64, FIELDS);
        assert_eq!(
            round.time_sample_members,
            whole_sorted(keys.obj_time_samples_index("7")).await,
            "time samples: every member, in rank order"
        );
        // The counts window (5 commands), then at least two windows of pages:
        // 52,008 members do not fit one 26,855-member window.
        assert!(
            windows.commands_seen >= 5 + 2 * 2,
            "precondition: round 1 read its members in more than one window of pages; {} \
             commands",
            windows.commands_seen
        );
    }

    /// An absent header ends round 1 after its counts window: no page is
    /// read, however many members the indices hold (the caller answers
    /// `NotFound` from the empty header).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fanout_round_one_reads_no_page_without_a_header() {
        use fred::interfaces::KeysInterface;

        use crate::storage::test_support::TestValkey;

        let tv = TestValkey::setup().await;
        let keys = round_one_keys(&tv, 600, 100).await;
        let _: i64 = tv.client().pool.del(keys.obj_header("7")).await.unwrap();
        let (pool, _in_use) = tv.client().acquire_snapshot();
        let mut windows = fanout_windows(tv.client(), pool, 2_141, u64::MAX / 2);

        let round = fanout_round_one(&mut windows, &keys, "7")
            .await
            .unwrap_or_else(|e| panic!("round 1 must succeed; got {e:?}"))
            .expect("an uncharged read never stops");

        assert!(round.header.is_empty());
        assert!(
            round.section_ids.is_empty()
                && round.deleted_section_ids.is_empty()
                && round.field_members.is_empty()
                && round.time_sample_members.is_empty(),
            "no member is read without a header"
        );
        assert_eq!(
            windows.commands_seen, 5,
            "only the counts window: the header and the four counts"
        );
    }

    // ── b.26e Stage 2: WA1's byte window is the read's projected bound ───────
    //
    // WA1 charges `W_BYTE` × a byte window before any index member is read.
    // It is the smaller of `FANOUT_WINDOW_BYTES` and the projected reply
    // bound from the counts (`projected_read_reply_bound`) plus, since b.gvf,
    // the projected request bytes of the read's script windows
    // (`projected_read_request_bound`), so an empty or small object holds
    // kilobytes for it, and an object whose windows project past one window
    // holds the whole window.

    /// `FANOUT_WINDOW_BYTES` at the shipped atomic-read limit (604,619 B): the
    /// full byte window WA1 charged for every object before it was narrowed.
    fn full_window() -> u64 {
        crate::config::snapshot_fanout_window_bytes(
            crate::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
        )
    }

    /// `FANOUT_WINDOW` at the shipped atomic-read item limit.
    fn full_command_window() -> u64 {
        crate::config::snapshot_fanout_window(crate::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT)
    }

    /// The projection is the per-family reply bounds, at the 64-B name floor,
    /// summed over every reply the rest of the read receives. The figures are
    /// spelled out as numbers so the test does not re-run the function's own
    /// arithmetic: a set-index id is charged at twice its count × 20 B (its
    /// `SSCAN` step's share); a field-index member is 20 + 1 + 2 × 64 = 149 B;
    /// a time-sample member 37 B; a section row 112 + 64 = 176 B; a tombstone
    /// row 57 B; a field row 144 B; a time-sample row 104 B; and a children
    /// list 20 B per entry, for the larger of sections + 2 and `obj:chs`.
    #[test]
    fn projected_read_reply_bound_sums_each_familys_bound_at_the_name_floor_b26e() {
        let cases: [([u64; 5], u64, &str); 6] = [
            // Only the children lists' sections + 2 = 2 entries, × 20.
            ([0, 0, 0, 0, 0], 40, "an empty object"),
            // 40 (id) + 176 (row) + 3 list entries × 20.
            ([1, 0, 0, 0, 0], 276, "one section"),
            (
                [10, 3, 50, 7, 0],
                13 * 40 + 50 * 149 + 7 * 37 + 10 * 176 + 3 * 57 + 50 * 144 + 7 * 104 + 12 * 20,
                "every family",
            ),
            // `obj:chs` larger than sections + 2 sizes the lists.
            (
                [10, 0, 0, 0, 30],
                40 * 40 + 10 * 176 + 30 * 20,
                "a marked object",
            ),
            (
                [2_000, 0, 0, 0, 0],
                2_000 * 40 + 2_000 * 176 + 2_002 * 20,
                "2,000 sections",
            ),
            (
                [5_000, 0, 0, 0, 0],
                5_000 * 40 + 5_000 * 176 + 5_002 * 20,
                "5,000 sections",
            ),
        ];
        for (counts, expected, what) in cases {
            assert_eq!(
                projected_read_reply_bound(counts),
                expected,
                "{what} {counts:?}"
            );
        }
        assert!(
            projected_read_reply_bound([2_000, 0, 0, 0, 0]) < full_window()
                && projected_read_reply_bound([5_000, 0, 0, 0, 0]) > full_window(),
            "the reply bound alone straddles the window at these sizes"
        );
        assert_eq!(
            projected_read_reply_bound([u64::MAX, u64::MAX, u64::MAX, u64::MAX, u64::MAX]),
            u64::MAX,
            "saturates rather than wrapping"
        );
    }

    /// A [`SharedCommitCharge`] that never refuses, prices with the
    /// production pricing (`object_state_charge`) and records every need it
    /// is asked to price, grow-only like the call's real charge.
    #[derive(Default)]
    struct RecordingCharge {
        held: std::sync::Mutex<u64>,
        needs: std::sync::Mutex<Vec<ObjectStateNeed>>,
    }

    impl SharedCommitCharge for RecordingCharge {
        fn held(&self) -> u64 {
            *self.held.lock().unwrap()
        }

        fn target(&self, need: &ObjectStateNeed) -> u64 {
            self.needs.lock().unwrap().push(*need);
            crate::grpc::upload_admission::object_state_charge(need)
        }

        fn rewrite_allowance(
            &self,
            need: &ObjectStateNeed,
        ) -> crate::domain::delta::MutationAllowance {
            crate::grpc::upload_admission::object_state_rewrite_allowance(self.held(), need)
        }

        fn try_hold(&self, target: u64) -> bool {
            *self.held.lock().unwrap() = target;
            true
        }

        fn try_hold_at_least(&self, target: u64) -> bool {
            let mut held = self.held.lock().unwrap();
            *held = (*held).max(target);
            true
        }
    }

    /// WA1 through [`FanoutAllowance::counted`] at the shipped windows, on a
    /// fresh recording charge: the snapshot its one need carries, and what
    /// it left held.
    fn wa1_snapshot(counts: [u64; 5]) -> (crate::storage::commit_charge::WatchSnapshot, u64) {
        let charge = RecordingCharge::default();
        let mut allowance = FanoutAllowance::new(&charge);
        assert!(
            allowance.counted(counts, full_command_window(), full_window()),
            "a charge that never refuses lets the read go on"
        );
        let needs = charge.needs.lock().unwrap();
        let [ObjectStateNeed::Watch { snapshot, .. }] = needs.as_slice() else {
            panic!("WA1 prices one WATCH need; got {needs:?}");
        };
        (*snapshot, charge.held())
    }

    /// [`wa1_snapshot`]'s byte window, and what WA1 left held.
    fn wa1(counts: [u64; 5]) -> (u64, u64) {
        let (snapshot, held) = wa1_snapshot(counts);
        (snapshot.window_bytes, held)
    }

    /// WA1's narrowing is bounded both ways: on an empty base WA1 holds
    /// kilobytes, not megabytes; on a large object it holds the full window.
    /// Before the narrowing, WA1 held `W_BYTE` × the full window (about
    /// 4.1 MB) for every structural WATCH edit, an object of 7 entries
    /// included. Below the window the byte window is the projection: the
    /// replies' ([`projected_read_reply_bound`]) and, since b.gvf, the
    /// script windows' requests ([`projected_read_request_bound`]), each
    /// spelled out in its own test.
    #[test]
    fn wa1_holds_a_small_objects_projected_windows_and_a_large_objects_whole_window_b26e() {
        // The T7 empty-base fixture: one section, 7 entries.
        let (window, held) = wa1([1, 0, 0, 0, 0]);
        assert!(
            held < 1_000_000,
            "WA1 on an empty base holds kilobytes, not megabytes: {held} B"
        );
        assert_eq!(
            window,
            276 + 271,
            "an empty base's byte window is its projection: replies 276 B, requests 271 B"
        );
        let (window, held_empty_object) = wa1([0, 0, 0, 0, 0]);
        assert!(held_empty_object < 1_000_000, "{held_empty_object} B");
        assert_eq!(window, 40 + 188);

        // Below the window, the projection itself: 1,500 sections, replies
        // 1,500 × (40 + 176) + 1,502 × 20, requests 64 + 1,500 × 21 + 1,502
        // × 62.
        let (window, _) = wa1([1_500, 0, 0, 0, 0]);
        assert_eq!(window, 354_040 + 124_688);
        assert!(window < full_window(), "fixture: below the window");

        // Above it, the whole window, as before the narrowing.
        let (window, held_large) = wa1([5_000, 0, 0, 0, 0]);
        assert_eq!(
            window,
            full_window(),
            "a large object's byte window is the full window"
        );
        let full_window_term = (full_window()
            * crate::grpc::upload_admission::WATCH_SNAPSHOT_REPLY_BYTE_CHARGE_CENTI)
            .div_ceil(100);
        assert!(
            held_large > full_window_term,
            "a large object holds at least W_BYTE × the full window: {held_large} B \
             against {full_window_term} B"
        );
    }

    /// The script windows' request projection (b.gvf) is each entry's op
    /// byte and id at the name floor, summed over the read, plus one
    /// window's column lists. Spelled out as numbers so the test does not
    /// re-run the function's own arithmetic: the column lists are 47 B
    /// (`hash size set_order authoring_set_order version`) + 17 B (`hash size
    /// version`) = 64 B; a section or tombstone entry 1 + 20 = 21 B; a field
    /// 1 + 149 = 150 B (its index member at the floor); a time sample 1 + 37
    /// = 38 B; and each children list, for the larger of sections + 2 and
    /// `obj:chs`, its `LLEN` entry 21 B and its ranged entry 21 B + 20 length
    /// digits, 62 B.
    #[test]
    fn projected_read_request_bound_sums_each_entrys_op_and_id_at_the_name_floor_b_gvf() {
        assert_eq!(SCRIPT_WINDOW_COLUMN_ARG_BYTES, 47 + 17);
        assert_eq!(
            SCRIPT_WINDOW_COLUMN_ARG_BYTES,
            (column_arg(SnapshotKeyFamily::Field).len()
                + column_arg(SnapshotKeyFamily::TimeSample).len()) as u64,
            "the projection's column lists are the ones the script is sent"
        );
        let cases: [([u64; 5], u64, &str); 5] = [
            ([0, 0, 0, 0, 0], 64 + 2 * 62, "an empty object"),
            ([1, 0, 0, 0, 0], 64 + 21 + 3 * 62, "one section"),
            (
                [10, 3, 50, 7, 0],
                64 + 13 * 21 + 50 * 150 + 7 * 38 + 12 * 62,
                "every family",
            ),
            ([10, 0, 0, 0, 30], 64 + 10 * 21 + 30 * 62, "a marked object"),
            (
                [5_000, 0, 0, 0, 0],
                64 + 5_000 * 21 + 5_002 * 62,
                "5,000 sections",
            ),
        ];
        for (counts, expected, what) in cases {
            assert_eq!(
                projected_read_request_bound(counts),
                expected,
                "{what} {counts:?}"
            );
        }
        assert_eq!(
            projected_read_request_bound([u64::MAX; 5]),
            u64::MAX,
            "saturates rather than wrapping"
        );
    }

    /// Each command's share of a script window's request (b.gvf): its op
    /// byte and its id, and a ranged list's length in decimal digits; a
    /// command no script window carries (the header, an index part, a
    /// whole-row value row) adds none. The window adds its column lists once.
    #[test]
    fn script_request_bytes_are_the_op_the_id_and_a_ranged_lists_length_digits_b_gvf() {
        let part = |family, id: &str, page| FanoutCommand::part(family, "k".to_owned(), id, page);
        let cases = [
            (
                part(SnapshotKeyFamily::Field, "3:616263", FanoutPage::ValueRow),
                1 + 8,
            ),
            (
                part(
                    SnapshotKeyFamily::TimeSample,
                    "3:3ff8000000000000",
                    FanoutPage::ValueRow,
                ),
                1 + 18,
            ),
            (
                part(SnapshotKeyFamily::Section, "12345", FanoutPage::Whole),
                1 + 5,
            ),
            (
                part(SnapshotKeyFamily::DeletedSection, "7", FanoutPage::Whole),
                1 + 1,
            ),
            (
                part(SnapshotKeyFamily::Children, "42", FanoutPage::Length),
                1 + 2,
            ),
            (
                part(
                    SnapshotKeyFamily::Children,
                    "42",
                    FanoutPage::Prefix { len: 4_096 },
                ),
                1 + 2 + 4,
            ),
            (
                part(
                    SnapshotKeyFamily::Children,
                    "42",
                    FanoutPage::Prefix { len: 1 },
                ),
                1 + 2 + 1,
            ),
            (part(SnapshotKeyFamily::Header, "", FanoutPage::Whole), 0),
            (
                part(SnapshotKeyFamily::Field, "3:616263", FanoutPage::Whole),
                0,
            ),
        ];
        for (command, expected) in cases {
            assert_eq!(
                command.script_request_bytes(),
                expected,
                "{} ({:?})",
                command.op_label(),
                command.page
            );
        }
    }

    /// WA1 holds round 1's index pages as plain commands (b.gvf refinement
    /// 1): one plain command in flight per index with members, at most five
    /// (`WatchSnapshot::plain_in_flight`), each priced at `W_PLAIN_CMD`, so a
    /// read stopped at WA1 projects the pages it is about to issue and its
    /// one wait covers them.
    #[test]
    fn wa1_holds_round_ones_index_pages_as_plain_commands_b_gvf() {
        for (counts, pages) in [
            ([0, 0, 0, 0, 0], 0),
            ([1, 0, 0, 0, 0], 1),
            ([1, 0, 3, 0, 1], 3),
            ([2, 1, 3, 4, 5], 5),
        ] {
            let (snapshot, held) = wa1_snapshot(counts);
            assert_eq!(snapshot.plain_in_flight, pages, "{counts:?}");
            let without_pages =
                crate::grpc::upload_admission::object_state_charge(&ObjectStateNeed::Watch {
                    snapshot: crate::storage::commit_charge::WatchSnapshot {
                        plain_in_flight: 0,
                        ..snapshot
                    },
                    fixed: true,
                    rewrite: RewriteCount::default(),
                    queued: QueuedCommands::default(),
                });
            assert_eq!(
                held - without_pages,
                pages * crate::grpc::upload_admission::WATCH_SNAPSHOT_PLAIN_COMMAND_CHARGE_BYTES,
                "{counts:?}: W_PLAIN_CMD per page command"
            );
        }
    }

    // ── b.26e Stage 2: window sizing at the longest name seen ────────────────
    //
    // The bounded-reply read sizes a name-bearing reply at N, the longest name
    // the read has seen, at least `NAME_BYTES_SIZING_FLOOR`; each page and
    // window's summed bound at N stays within `FANOUT_WINDOW_BYTES`; and the
    // one page or window that brings names longer than any seen overshoots
    // its bound by at most R4's [code] figures at `MAX_NAME_BYTES`.

    /// N rises to the longest name seen — a field's key name (half its
    /// index member's hex) or a section's name — and never falls; the names'
    /// retained bytes are every member's bytes and every section name's.
    #[test]
    fn name_tally_tracks_the_longest_name_seen_and_never_lowers_it_b26e() {
        let member = |name_len: usize| format!("7:{}", "61".repeat(name_len));
        let mut names = NameTally::default();
        assert_eq!((names.longest, names.bytes), (0, 0));
        names.field_members(&[member(100), member(3)]);
        assert_eq!(
            names.longest, 100,
            "a field's key name, half its member's hex"
        );
        names.section_name(&"s".repeat(300));
        assert_eq!(names.longest, 300, "a longer section name raises it");
        names.section_name("short");
        names.field_members(&[member(10)]);
        assert_eq!(names.longest, 300, "a shorter name never lowers it");
        assert_eq!(
            names.bytes,
            (2 + 200) + (2 + 6) + 300 + 5 + (2 + 20),
            "every member's bytes and every section name's"
        );
    }

    /// A round-1 field page at the floor holds ⌊`FANOUT_WINDOW_BYTES` ÷ 149⌋
    /// = 4,057 members, within the byte window; the same page of names at
    /// `MAX_NAME_BYTES` (8,213 B each) arrives 32,715,522 B past it — R4's
    /// figure for a field page, the most one page overshoots [code]. Sized at
    /// that N, the page fits the window again.
    #[test]
    fn a_field_page_sized_at_n_fits_the_byte_window_and_overshoots_by_at_most_r4s_figure_b26e() {
        let window = full_window();
        let max_name = crate::domain::limits::MAX_NAME_BYTES as u64;
        let at_floor = cap_shares_to_bytes(
            vec![crate::config::FANOUT_INDEX_WINDOW_MEMBERS],
            &[field_member_reply_bound(NAME_BYTES_SIZING_FLOOR)],
            window,
        );
        assert_eq!(at_floor, vec![4_057]);
        assert!(at_floor[0] * field_member_reply_bound(NAME_BYTES_SIZING_FLOOR) <= window);
        assert_eq!(
            at_floor[0] * field_member_reply_bound(max_name) - window,
            32_715_522,
            "the field page residual at MAX_NAME_BYTES"
        );
        let at_max = cap_shares_to_bytes(
            vec![crate::config::FANOUT_INDEX_WINDOW_MEMBERS],
            &[field_member_reply_bound(max_name)],
            window,
        );
        assert!(
            at_max[0] >= 1 && at_max[0] * field_member_reply_bound(max_name) <= window,
            "sized at N = MAX_NAME_BYTES the page fits: {at_max:?}"
        );
    }

    /// Round-2 section windows on the bounded-reply read: sized at N, each
    /// window's summed bound is its rows × (112 + N) and stays within
    /// `FANOUT_WINDOW_BYTES` (and `FANOUT_WINDOW`), at the floor, past it and
    /// at `MAX_NAME_BYTES`. At the floor a window is 2,141 rows, whose names
    /// at `MAX_NAME_BYTES` arrive 8,404,709 B past the window — R4's figure
    /// for a section window [code]. A command whose bound alone passes the
    /// byte window (a list longer than the window holds ids) runs alone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bounded_windows_sized_at_n_stay_within_the_byte_window_b26e() {
        use crate::storage::test_support::TestValkey;

        let tv = TestValkey::setup().await;
        let client = tv.client().clone();
        let (pool, _in_use) = client.acquire_snapshot();
        let mut no_progress = |_: u64| {};
        let mut windows = FanoutWindows::new(
            &client,
            pool,
            tv.key_schema(),
            "7",
            SnapshotReadFaultCursor::unfenced_fanout(),
            ProgressSink::Local(&mut no_progress),
            FanoutReplies::Bounded,
            AllowanceSeam::None,
        );
        let max_name = crate::domain::limits::MAX_NAME_BYTES as u64;
        let sections = |n: u64| -> Vec<FanoutCommand> {
            (0..n)
                .map(|i| {
                    FanoutCommand::new(SnapshotKeyFamily::Section, format!("k{i}"), &i.to_string())
                })
                .collect()
        };
        for longest in [0, NAME_BYTES_SIZING_FLOOR, 1_000, max_name] {
            windows.tally.names.longest = longest;
            let n = longest.max(NAME_BYTES_SIZING_FLOOR);
            assert_eq!(windows.sizing_name_bytes(), n);
            let mut rest = sections(5_000).into_iter().peekable();
            let mut rows = 0;
            while rest.peek().is_some() {
                let BoundedWindow {
                    commands: window,
                    reply_bound: bound,
                    ..
                } = windows.bounded_window(&mut rest, WindowTransport::Plain);
                let len = window.len() as u64;
                assert_eq!(bound, len * (112 + n), "N {n}: a section row is 112 B + N");
                assert!(
                    bound <= windows.max_bytes && len <= windows.max_commands,
                    "N {n}: a window of {len} rows bound {bound} B"
                );
                rows += len;
            }
            assert_eq!(rows, 5_000, "N {n}: every command is windowed");
        }

        windows.tally.names.longest = 0;
        let window = windows
            .bounded_window(
                &mut sections(5_000).into_iter().peekable(),
                WindowTransport::Plain,
            )
            .commands;
        assert_eq!(
            window.len(),
            2_141,
            "at the floor a window is FANOUT_WINDOW rows"
        );
        assert_eq!(
            window.len() as u64 * section_row_reply_bound(max_name) - windows.max_bytes,
            8_404_709,
            "the section window residual at MAX_NAME_BYTES"
        );

        let long_list = || {
            FanoutCommand::part(
                SnapshotKeyFamily::Children,
                "list".to_owned(),
                "2",
                FanoutPage::Prefix {
                    len: windows.max_bytes / ID_REPLY_BOUND + 1,
                },
            )
        };
        let mut rest = vec![long_list(), long_list()].into_iter().peekable();
        let BoundedWindow {
            commands: window,
            reply_bound: bound,
            ..
        } = windows.bounded_window(&mut rest, WindowTransport::Plain);
        assert_eq!(window.len(), 1, "a command over the byte window runs alone");
        assert!(bound > windows.max_bytes);
    }

    // ── b.gvf: the script windows' cut (SDD §10.5) ───────────────────────────
    //
    // A script window of the bounded-reply read's rounds 2 and 3 holds
    // commands while their wire commands stay within `FANOUT_WINDOW`, their
    // reply bounds and the window's request bytes together within
    // `FANOUT_WINDOW_BYTES`, and their modelled shard time
    // (`config::fanout_script_window_modelled_shard_ns`) within τ
    // (`config::FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS`). A lone entry whose
    // modelled time alone exceeds τ — only a long children list can — is a
    // plain window of its own. Every expected figure below is computed from
    // the wired constants, or read from the window model's record, so a
    // re-fit of the window model's coefficients moves none of them by hand.

    /// A bounded-reply [`FanoutWindows`] on `tv`'s snapshot lane, its
    /// transport the production one: nothing observed, every name at the
    /// floor.
    fn bounded_windows<'a>(
        client: &'a ValkeyClient,
        pool: &'a fred::clients::Pool,
        keys: &'a KeySchema,
    ) -> FanoutWindows<'a> {
        FanoutWindows::new(
            client,
            pool,
            keys,
            "7",
            SnapshotReadFaultCursor::unfenced_fanout(),
            ProgressSink::Local(Box::leak(Box::new(|_: u64| {}))),
            FanoutReplies::Bounded,
            AllowanceSeam::None,
        )
    }

    /// The window kinds a script window carries, at the name floor: a
    /// command of each, and its wire commands and reply bound
    /// ([`FanoutWindows::reply_bound`] at [`NAME_BYTES_SIZING_FLOOR`]).
    fn script_window_kinds() -> [(&'static str, FanoutCommand, u64, u64); 5] {
        // A field's index member at the floor: `{section}:{hex(name)}`,
        // a 64-byte name.
        let field_id = format!("1:{}", "61".repeat(NAME_BYTES_SIZING_FLOOR as usize));
        let part = |family, id: &str, page| FanoutCommand::part(family, "k".to_owned(), id, page);
        [
            (
                "field",
                part(SnapshotKeyFamily::Field, &field_id, FanoutPage::ValueRow),
                2,
                FIELD_ROW_REPLY_BOUND,
            ),
            (
                "time_sample",
                part(
                    SnapshotKeyFamily::TimeSample,
                    "1:3ff8000000000000",
                    FanoutPage::ValueRow,
                ),
                2,
                TIME_SAMPLE_ROW_REPLY_BOUND,
            ),
            (
                "tombstone",
                part(
                    SnapshotKeyFamily::DeletedSection,
                    "12345",
                    FanoutPage::Whole,
                ),
                1,
                TOMBSTONE_ROW_REPLY_BOUND,
            ),
            (
                "section",
                part(SnapshotKeyFamily::Section, "12345", FanoutPage::Whole),
                1,
                section_row_reply_bound(NAME_BYTES_SIZING_FLOOR),
            ),
            (
                "lists_length",
                part(SnapshotKeyFamily::Children, "12345", FanoutPage::Length),
                1,
                0,
            ),
        ]
    }

    /// The most entries of `per_wire` wire commands and `per_reply` B of
    /// reply bound each that fit τ by the wired window model, list entries
    /// aside: `(τ − FIXED) ÷ (CMD × per_wire + BYTE × per_reply)`, unbounded
    /// when that per-entry cost is 0.
    fn entries_within_tau(per_wire: u64, per_reply: u64) -> u64 {
        use crate::config::{
            FANOUT_SCRIPT_COMMAND_NS, FANOUT_SCRIPT_REPLY_BYTE_NS, FANOUT_SCRIPT_WINDOW_FIXED_NS,
            FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS,
        };
        (FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS - FANOUT_SCRIPT_WINDOW_FIXED_NS)
            .checked_div(
                FANOUT_SCRIPT_COMMAND_NS * per_wire + FANOUT_SCRIPT_REPLY_BYTE_NS * per_reply,
            )
            .unwrap_or(u64::MAX)
    }

    /// The most entries one children list a lone script window ranges may
    /// hold within τ by the wired window model: `(τ − FIXED − CMD) ÷ (LIST +
    /// BYTE × 20)`, its `LRANGE` one wire command and 20 B of reply bound per
    /// entry; unbounded when that per-entry cost is 0.
    fn one_list_entries_within_tau() -> u64 {
        use crate::config::{
            FANOUT_SCRIPT_COMMAND_NS, FANOUT_SCRIPT_LIST_ENTRY_NS, FANOUT_SCRIPT_REPLY_BYTE_NS,
            FANOUT_SCRIPT_WINDOW_FIXED_NS, FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS,
        };
        (FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS
            - FANOUT_SCRIPT_WINDOW_FIXED_NS
            - FANOUT_SCRIPT_COMMAND_NS)
            .checked_div(FANOUT_SCRIPT_LIST_ENTRY_NS + FANOUT_SCRIPT_REPLY_BYTE_NS * ID_REPLY_BOUND)
            .unwrap_or(u64::MAX)
    }

    /// Every script window of a mixed round — each kind a script window
    /// carries, ranged lists among them, names at the floor — keeps the three
    /// rules (wire commands within `FANOUT_WINDOW`, reply bound and request
    /// bytes within `FANOUT_WINDOW_BYTES`, modelled shard time within τ), its
    /// recorded figures are its commands' (wire, reply bound, request bytes
    /// with the column lists, modelled time at the wired model), and it takes
    /// every command it can: the next command would break one of the rules.
    /// Every command is windowed, none as a plain window.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn script_windows_keep_the_cap_the_byte_window_and_tau_and_take_all_they_can_b_gvf() {
        use crate::config::{
            fanout_script_window_modelled_shard_ns, FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS,
        };
        use crate::storage::test_support::TestValkey;

        let tv = TestValkey::setup().await;
        let client = tv.client().clone();
        let (pool, _in_use) = client.acquire_snapshot();
        let windows = bounded_windows(&client, pool, tv.key_schema());
        let mut round = Vec::new();
        for (_, command, _, _) in script_window_kinds() {
            for i in 0..700 {
                round.push(FanoutCommand::part(
                    command.family,
                    format!("{}-{i}", command.key),
                    &command.id,
                    command.page.clone(),
                ));
            }
        }
        for i in 0..40u64 {
            round.push(FanoutCommand::part(
                SnapshotKeyFamily::Children,
                format!("list-{i}"),
                &i.to_string(),
                FanoutPage::Prefix { len: 10 + 13 * i },
            ));
        }
        let commands = round.len();
        let list_len = |command: &FanoutCommand| match command.page {
            FanoutPage::Prefix { len } => len,
            _ => 0,
        };

        let mut rest = round.into_iter().peekable();
        let mut windowed = 0;
        while rest.peek().is_some() {
            let window = windows.bounded_window(&mut rest, WindowTransport::Script);
            assert_eq!(
                window.transport,
                WindowTransport::Script,
                "no entry is over τ alone"
            );
            let wire: u64 = window
                .commands
                .iter()
                .map(FanoutCommand::wire_commands)
                .sum();
            let bound: u64 = window.commands.iter().map(|c| windows.reply_bound(c)).sum();
            let request: u64 = SCRIPT_WINDOW_COLUMN_ARG_BYTES
                + window
                    .commands
                    .iter()
                    .map(FanoutCommand::script_request_bytes)
                    .sum::<u64>();
            let lists: u64 = window.commands.iter().map(list_len).sum();
            let modelled = fanout_script_window_modelled_shard_ns(wire, lists, bound);
            assert_eq!(
                (window.wire, window.reply_bound, window.request_bytes),
                (wire, bound, request),
                "the window's recorded figures are its commands'"
            );
            assert_eq!(window.modelled_shard_ns, Some(modelled));
            assert!(wire <= windows.max_commands, "{wire} wire commands");
            assert!(
                bound + request <= windows.max_bytes,
                "{bound} B of replies and {request} B of request"
            );
            assert!(
                modelled <= FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS,
                "{modelled} ns modelled"
            );
            if let Some(next) = rest.peek() {
                let over_cap = wire + next.wire_commands() > windows.max_commands;
                let over_bytes =
                    bound + windows.reply_bound(next) + request + next.script_request_bytes()
                        > windows.max_bytes;
                let over_tau = fanout_script_window_modelled_shard_ns(
                    wire + next.wire_commands(),
                    lists + list_len(next),
                    bound + windows.reply_bound(next),
                ) > FANOUT_SCRIPT_WINDOW_SHARD_BUDGET_NS;
                assert!(
                    over_cap || over_bytes || over_tau,
                    "a window of {} commands left out {} ({}) though it fit every rule",
                    window.commands.len(),
                    next.op_label(),
                    next.key
                );
            }
            windowed += window.commands.len();
        }
        assert_eq!(windowed, commands, "every command is windowed");
    }

    /// The τ-cut composition at the name floor: a run of one kind cuts its
    /// first script window at the most entries every rule admits — the
    /// smallest of the τ cut by the wired window model
    /// ([`entries_within_tau`]), `FANOUT_WINDOW` over its wire commands, and
    /// the byte window over its reply bound and request bytes — and one
    /// children list's τ capacity is [`one_list_entries_within_tau`]. The τ
    /// cuts are the window model's record's (`raw/m1-fit.json`
    /// `tau_cut_at_the_name_floor`), which the record computes at the
    /// coefficients it fits, so a re-fit wired without its record, or the
    /// reverse, fails here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn script_windows_at_the_name_floor_cut_each_kind_at_the_wired_models_tau_cut_b_gvf() {
        use crate::storage::test_support::TestValkey;

        let tv = TestValkey::setup().await;
        let client = tv.client().clone();
        let (pool, _in_use) = client.acquire_snapshot();
        let windows = bounded_windows(&client, pool, tv.key_schema());
        let mut tau_cuts = Vec::new();
        for (kind, command, per_wire, per_reply) in script_window_kinds() {
            let per_request = command.script_request_bytes();
            let by_tau = entries_within_tau(per_wire, per_reply);
            let by_cap = windows.max_commands / per_wire;
            let by_bytes =
                (windows.max_bytes - SCRIPT_WINDOW_COLUMN_ARG_BYTES) / (per_reply + per_request);
            let expected = by_tau.min(by_cap).min(by_bytes);
            let run: Vec<FanoutCommand> = (0..expected + 10)
                .map(|i| {
                    FanoutCommand::part(
                        command.family,
                        format!("k{i}"),
                        &command.id,
                        command.page.clone(),
                    )
                })
                .collect();
            let window =
                windows.bounded_window(&mut run.into_iter().peekable(), WindowTransport::Script);
            assert_eq!(
                window.commands.len() as u64,
                expected,
                "{kind}: the first window holds the most entries every rule admits (τ {by_tau}, \
                 cap {by_cap}, bytes {by_bytes})"
            );
            tau_cuts.push((kind, by_tau));
        }

        let one_list = one_list_entries_within_tau();
        let list = |len: u64| {
            FanoutCommand::part(
                SnapshotKeyFamily::Children,
                "list".to_owned(),
                "2",
                FanoutPage::Prefix { len },
            )
        };
        let window = windows.bounded_window(
            &mut vec![list(one_list)].into_iter().peekable(),
            WindowTransport::Script,
        );
        assert_eq!(
            window.transport,
            WindowTransport::Script,
            "a list of the τ capacity is read by a script window"
        );

        // The record's τ cuts, where the repository carries it.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        if !root.join("docs-internal").is_dir() {
            eprintln!("skipped the record check: no docs-internal/ (published source tree)");
            return;
        }
        const M1_FIT: &str =
            "docs-internal/validation/watch-window-script-b-gvf-20261010/raw/m1-fit.json";
        let record: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(root.join(M1_FIT))
                .unwrap_or_else(|e| panic!("the committed window-model record {M1_FIT}: {e}")),
        )
        .expect("the window-model record is valid JSON");
        let recorded = |pointer: &str| {
            record
                .pointer(pointer)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or_else(|| panic!("{M1_FIT} carries {pointer}"))
        };
        for (kind, by_tau) in tau_cuts {
            assert_eq!(
                by_tau,
                recorded(&format!(
                    "/tau_cut_at_the_name_floor/{kind}/entries_within_tau"
                )),
                "{kind}: the τ cut at the wired constants is the record's"
            );
        }
        assert_eq!(
            one_list,
            recorded("/tau_cut_at_the_name_floor/lists_range/list_entries_within_tau_one_list"),
            "one list's τ capacity at the wired constants is the record's"
        );
    }

    /// The lone-entry rule on the children-list case (b.gvf refinement 3):
    /// a children list one entry longer than a script window may range within
    /// τ ([`one_list_entries_within_tau`]) is a plain window of its own — one
    /// `LRANGE`, no request bytes, no modelled time — whether it comes first or
    /// after other entries, which it closes into a script window before it;
    /// at exactly that capacity it is a script window. The plain transport
    /// has no τ rule: the same list shares a plain window, within the byte
    /// window.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_lone_children_list_over_tau_is_a_plain_window_of_its_own_b_gvf() {
        use crate::storage::test_support::TestValkey;

        let tv = TestValkey::setup().await;
        let client = tv.client().clone();
        let (pool, _in_use) = client.acquire_snapshot();
        let windows = bounded_windows(&client, pool, tv.key_schema());
        let capacity = one_list_entries_within_tau();
        let list = |len: u64| {
            FanoutCommand::part(
                SnapshotKeyFamily::Children,
                "list".to_owned(),
                "2",
                FanoutPage::Prefix { len },
            )
        };
        let length = || {
            FanoutCommand::part(
                SnapshotKeyFamily::Children,
                "other".to_owned(),
                "3",
                FanoutPage::Length,
            )
        };

        let within = windows.bounded_window(
            &mut vec![list(capacity)].into_iter().peekable(),
            WindowTransport::Script,
        );
        assert_eq!(
            (within.transport, within.commands.len()),
            (WindowTransport::Script, 1),
            "a list of the τ capacity is a script window"
        );

        let mut rest = vec![list(capacity + 1), length()].into_iter().peekable();
        let lone = windows.bounded_window(&mut rest, WindowTransport::Script);
        assert_eq!(
            (
                lone.transport,
                lone.commands.len(),
                lone.wire,
                lone.reply_bound,
                lone.request_bytes,
                lone.modelled_shard_ns
            ),
            (
                WindowTransport::Plain,
                1,
                1,
                (capacity + 1) * ID_REPLY_BOUND,
                0,
                None
            ),
            "a lone list over τ is one plain LRANGE"
        );
        let after = windows.bounded_window(&mut rest, WindowTransport::Script);
        assert_eq!(
            (after.transport, after.commands.len()),
            (WindowTransport::Script, 1),
            "the entry after it goes back to a script window"
        );

        let mut rest = vec![length(), list(capacity + 1)].into_iter().peekable();
        let before = windows.bounded_window(&mut rest, WindowTransport::Script);
        assert_eq!(
            (before.transport, before.commands.len()),
            (WindowTransport::Script, 1),
            "the entry before the long list closes into a script window"
        );
        let lone = windows.bounded_window(&mut rest, WindowTransport::Script);
        assert_eq!(
            (lone.transport, lone.commands.len()),
            (WindowTransport::Plain, 1)
        );

        let plain = windows.bounded_window(
            &mut vec![list(capacity + 1), length()].into_iter().peekable(),
            WindowTransport::Plain,
        );
        let shares = (capacity + 1) * ID_REPLY_BOUND <= windows.max_bytes;
        assert_eq!(
            (
                plain.transport,
                plain.commands.len(),
                plain.modelled_shard_ns
            ),
            (WindowTransport::Plain, if shares { 2 } else { 1 }, None),
            "the plain transport has no τ rule: the list shares its window within the byte window"
        );
    }

    /// A script window waits out the longest of the delays the seam queued
    /// for its commands, not their sum: the plain window's commands wait out
    /// theirs at once, and a script window is one call. Two delays (300 and
    /// 700 ms) on its first two commands and a permanent error on its third,
    /// so nothing is sent: the window fails after 700 ms (to the timer's
    /// 1 ms tick), not 1,000, on tokio's paused clock (virtual time,
    /// auto-advanced to the next timer when the runtime is idle:
    /// deterministic, no wall clock), every queued fault taken.
    #[tokio::test(flavor = "current_thread")]
    async fn a_script_window_waits_out_the_longest_of_its_delays_not_their_sum_b_gvf() {
        use crate::storage::test_support::TestValkey;

        let tv = TestValkey::setup().await;
        let client = tv.client();
        let (pool, _in_use) = client.acquire_snapshot();
        let injector = std::sync::Arc::new(ObjectSnapshotReadFaultInjector::new());
        let nth =
            |n| ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::UnfencedFanout, n);
        injector
            .push_delay_at(nth(0), Duration::from_millis(300))
            .push_delay_at(nth(1), Duration::from_millis(700))
            .push_at(
                nth(2),
                fred::error::Error::new(
                    fred::error::ErrorKind::Auth,
                    "NOAUTH Authentication required.",
                ),
            );
        client
            .install_object_snapshot_read_fault_injector_for_test(std::sync::Arc::clone(&injector));
        let window: Vec<FanoutCommand> = (0..3)
            .map(|i| {
                FanoutCommand::part(
                    SnapshotKeyFamily::Section,
                    format!("{}never-read:{i}", tv.prefix()),
                    &i.to_string(),
                    FanoutPage::Whole,
                )
            })
            .collect();
        let mut cursor = SnapshotReadFaultCursor::unfenced_fanout();

        tokio::time::pause();
        let started = tokio::time::Instant::now();
        let result =
            issue_script_window(client, pool, tv.key_schema(), "7", &mut cursor, &window).await;
        let waited = started.elapsed();

        assert!(
            matches!(&result, Err(ServiceError::Internal(m)) if m.contains("HGETALL section")),
            "the queued error fails the window as its command's failure; got {:?}",
            result.as_ref().map(|_| ())
        );
        assert_eq!(
            (injector.delays_remaining(), injector.remaining()),
            (0, 0),
            "every queued fault was taken"
        );
        // The paused clock rounds a sleep's deadline up to the timer wheel's
        // next 1 ms tick.
        assert!(
            waited >= Duration::from_millis(700) && waited < Duration::from_millis(1_000),
            "the window waited out the longest delay (700 ms), not their sum (1,000 ms): \
             waited {waited:?}"
        );
    }

    // ── b.gvf: the script window's reply decode ─────────────────────────────
    //
    // `decode_fanout_window_reply` turns the script's reply into the replies
    // the plain issuer returns, and refuses anything not of the script's
    // shape as `Internal`, attributed to the header's `EVALSHA`.

    /// The header key the decode tests attribute to.
    const DECODE_HEADER: &str = "pfx:{7}:obj:hdr";

    fn value_int(n: i64) -> fred::types::Value {
        fred::types::Value::Integer(n)
    }

    fn value_str(s: &str) -> fred::types::Value {
        fred::types::Value::String(s.into())
    }

    fn value_array(items: Vec<fred::types::Value>) -> fred::types::Value {
        fred::types::Value::Array(items)
    }

    /// A field row's element: value length, mask, the follow-up answers and
    /// the five columns (`hash size set_order authoring_set_order version`).
    fn field_row(
        value_len: i64,
        mask: i64,
        value_exists: i64,
        row_exists: i64,
        columns: [&str; 5],
    ) -> fred::types::Value {
        let mut parts = vec![
            value_int(value_len),
            value_int(mask),
            value_int(value_exists),
            value_int(row_exists),
        ];
        parts.extend(columns.into_iter().map(value_str));
        value_array(parts)
    }

    /// One window of each script op: a field row, a section, a list length
    /// and a ranged list.
    fn decode_window() -> (Vec<FanoutCommand>, Vec<ScriptOp>) {
        let part =
            |family, id: &str, page| FanoutCommand::part(family, format!("k:{id}"), id, page);
        let window = vec![
            part(SnapshotKeyFamily::Field, "1:6161", FanoutPage::ValueRow),
            part(SnapshotKeyFamily::Section, "1", FanoutPage::Whole),
            part(SnapshotKeyFamily::Children, "0", FanoutPage::Length),
            part(
                SnapshotKeyFamily::Children,
                "0",
                FanoutPage::Prefix { len: 2 },
            ),
        ];
        let ops = window
            .iter()
            .map(|c| c.script_op().expect("a script-window command"))
            .collect();
        (window, ops)
    }

    /// A well-formed reply for [`decode_window`], with `row` as the field
    /// row's element.
    fn decode_reply(row: fred::types::Value) -> fred::types::Value {
        value_array(vec![
            value_str(FANOUT_WINDOW_READ_FORMAT),
            value_str("OK"),
            row,
            value_array(vec![value_str("name"), value_str("World")]),
            value_int(2),
            value_array(vec![value_str("1"), value_str("2")]),
        ])
    }

    /// A well-formed reply decodes into the plain issuer's replies, in order
    /// — the value row as an atomic [`ValueRowReply`] with its columns and
    /// the follow-ups the rule asks, a section as its hash, a length as a
    /// count, a range as its ids — and a value row read by a script window
    /// holding a value beside a `hash` is the both-shapes row the plain read
    /// never reports.
    #[test]
    fn decode_fanout_window_reply_gives_the_plain_issuers_replies_marked_atomic_b_gvf() {
        let (window, ops) = decode_window();
        let hash = "ab".repeat(32);
        let replies = decode_fanout_window_reply(
            "7",
            DECODE_HEADER,
            &window,
            &ops,
            decode_reply(field_row(3, 0b1_0101, -1, -1, [&hash, "", "5", "", "1"])),
        )
        .expect("a well-formed reply decodes");
        assert_eq!(replies.len(), 4);
        let FanoutReply::ValueRow(row) = &replies[0] else {
            panic!(
                "the field row decodes as a value row; got {:?}",
                replies[0].items()
            );
        };
        assert_eq!(
            (row.value_len, row.atomic, row.value_exists, row.row_exists),
            (3, true, None, None)
        );
        assert_eq!(
            row.columns,
            vec![
                Some(hash.clone()),
                None,
                Some("5".to_owned()),
                None,
                Some("1".to_owned())
            ]
        );
        assert!(
            matches!(row.shape(), SlimRow::InlineBesideHash),
            "a value beside a hash, read atomically, holds both shapes"
        );
        let plain = ValueRowReply {
            family: row.family,
            value_len: row.value_len,
            columns: row.columns.clone(),
            value_exists: row.value_exists,
            row_exists: row.row_exists,
            atomic: false,
        };
        assert!(
            matches!(plain.shape(), SlimRow::Inline),
            "the same row read by the plain transport is inline, unreported"
        );
        assert!(
            matches!(&replies[1], FanoutReply::Hash(h) if h.get("name").map(String::as_str) == Some("World"))
        );
        assert!(matches!(replies[2], FanoutReply::Count(2)));
        assert!(matches!(&replies[3], FanoutReply::List(ids) if ids == &["1", "2"]));
    }

    /// Every reply not of the script's shape is refused `Internal`,
    /// attributed to the header's window `EVALSHA` and saying the reply is
    /// malformed: a wrong format tag or status, an arity other than 2 + N, a
    /// value row of the wrong arity, a mask with a bit past its columns, a
    /// column under an unset bit that carries a value, a follow-up answered
    /// where the rule does not ask it or missing where it does, a negative
    /// value length, and a list length or range of the wrong type.
    #[test]
    fn decode_fanout_window_reply_refuses_each_malformed_shape_internal_at_the_header_b_gvf() {
        let (window, ops) = decode_window();
        let hash = "ab".repeat(32);
        let good_row = || field_row(3, 0b1_0101, -1, -1, [&hash, "", "5", "", "1"]);
        let with = |i: usize, value: fred::types::Value| {
            let mut parts = decode_reply(good_row()).into_array();
            parts[i] = value;
            value_array(parts)
        };
        let cases: Vec<(&str, fred::types::Value)> = vec![
            ("format", with(0, value_str("fanout-window-read-v0"))),
            ("status", with(1, value_str("NOT-OK"))),
            ("arity", {
                let mut parts = decode_reply(good_row()).into_array();
                parts.pop();
                value_array(parts)
            }),
            (
                "value row arity",
                with(2, value_array(vec![value_int(3), value_int(0)])),
            ),
            (
                "mask past the columns",
                with(2, field_row(3, 1 << 5, -1, -1, ["", "", "", "", ""])),
            ),
            (
                "a value under an unset bit",
                with(2, field_row(3, 0, -1, -1, ["x", "", "", "", ""])),
            ),
            (
                "HEXISTS answered where the rule does not ask it",
                with(2, field_row(3, 0b1, 1, -1, [&hash, "", "", "", ""])),
            ),
            (
                "HEXISTS missing where the rule asks it",
                with(2, field_row(0, 0b1, -1, -1, [&hash, "", "", "", ""])),
            ),
            (
                "EXISTS missing where the rule asks it",
                with(2, field_row(0, 0, 0, -1, ["", "", "", "", ""])),
            ),
            (
                "a negative value length",
                with(2, field_row(-1, 0b10000, -1, -1, ["", "", "", "", "1"])),
            ),
            ("a length that is not an integer", with(4, value_str("2"))),
            ("a negative length", with(4, value_int(-2))),
            ("a range that is not an array", with(5, value_int(2))),
        ];
        for (what, reply) in cases {
            match decode_fanout_window_reply("7", DECODE_HEADER, &window, &ops, reply) {
                Err(ServiceError::Internal(msg)) => assert!(
                    msg.contains(FANOUT_WINDOW_READ_OP)
                        && msg.contains(DECODE_HEADER)
                        && msg.contains("is malformed"),
                    "{what}: attributed to the header's window EVALSHA; got {msg:?}"
                ),
                other => panic!(
                    "{what}: a malformed reply is Internal; got {:?}",
                    other.map(|replies| replies.len())
                ),
            }
        }
    }

    // ── b.26e Stage 2 (R1): one per-command issuer per reply shape ───────────
    //
    // Every command of a plain fan-out window holds its `issue_window_command`
    // future at once, so that future's size is the window's per-command
    // transient. The whole-row read (the fenced read's pass, whose window the
    // catch-up budget's `X` funds) issues through
    // `issue_fanout_command`, which has no value-row arm; the bounded-reply
    // read's plain windows (the structural WATCH read's round 1, a lone
    // children list over τ, and every window after a script fallback) through
    // `issue_bounded_fanout_command`, which keeps the value-row read inline.
    // Each of that read's other windows is one script window holding one
    // `issue_script_window` future (b.gvf). The size pins are debug-build
    // sizes of a `cfg(test)` build; the routing pins hold in every build.

    /// Debug builds only: the ceiling is a debug-build size. A release build
    /// lays an async fn's suspended states over one another where a debug
    /// build gives each its own slot, so its sizes are smaller and not
    /// comparable with these; what both builds share is the structure the
    /// pins guard — the whole-row issuer has no value-row arm, and the
    /// bounded issuer holds the value-row read inline.
    #[cfg(debug_assertions)]
    mod issuer_future_size {
        use super::*;

        /// The size of the future `f` returns, without calling `f`: only its
        /// type is measured, so no client, pool or command is needed.
        fn future_size_of<F: std::future::Future>(_f: impl FnOnce() -> F) -> usize {
            std::mem::size_of::<F>()
        }

        /// A value of any type for an argument of a function whose future is
        /// measured but never created: [`future_size_of`] never calls the
        /// closure this appears in.
        fn never<T>() -> T {
            unreachable!("never called: only the future's type is measured")
        }

        /// The whole-row read's per-command future: [`issue_window_command`]
        /// over [`issue_fanout_command`].
        fn whole_row_command_future() -> usize {
            future_size_of(|| {
                issue_window_command(
                    never::<&'static ValkeyClient>(),
                    never::<&'static fred::clients::Pool>(),
                    never::<&'static str>(),
                    never::<FanoutCommandFault>(),
                    never::<&'static FanoutCommand>(),
                    issue_fanout_command,
                )
            })
        }

        /// The bounded-reply read's per-command future: [`issue_window_command`]
        /// over [`issue_bounded_fanout_command`].
        fn bounded_command_future() -> usize {
            future_size_of(|| {
                issue_window_command(
                    never::<&'static ValkeyClient>(),
                    never::<&'static fred::clients::Pool>(),
                    never::<&'static str>(),
                    never::<FanoutCommandFault>(),
                    never::<&'static FanoutCommand>(),
                    issue_bounded_fanout_command,
                )
            })
        }

        /// One script window's future (b.gvf): [`issue_script_window`], the
        /// whole window's one `EVALSHA` and its decode.
        fn script_window_future() -> usize {
            future_size_of(|| {
                issue_script_window(
                    never::<&'static ValkeyClient>(),
                    never::<&'static fred::clients::Pool>(),
                    never::<&'static KeySchema>(),
                    never::<&'static str>(),
                    never::<&'static mut SnapshotReadFaultCursor>(),
                    never::<&'static [FanoutCommand]>(),
                )
            })
        }

        /// [`value_row_read`]'s future: its two joined fred calls.
        fn value_row_read_future() -> usize {
            future_size_of(|| {
                value_row_read(
                    never::<&'static fred::clients::Pool>(),
                    never::<&'static FanoutCommand>(),
                )
            })
        }

        /// The ceiling on the whole-row read's per-command future in a debug
        /// `cfg(test)` build: 8 KiB. It measured 7,216 B with its own issuer;
        /// with the value-row read as an arm of the one issuer both reads
        /// shared (the shape before R1) it measured 13,768 B, because an async
        /// fn's future is as large as its largest arm. Above the ceiling the
        /// whole-row read again carries a value-row-sized state on every
        /// command of every window.
        const WHOLE_ROW_COMMAND_FUTURE_CEILING: usize = 8 * 1024;

        /// R1: the whole-row read's per-command future stays under
        /// [`WHOLE_ROW_COMMAND_FUTURE_CEILING`], below the value-row read's
        /// own future — so no value-row arm rides on it.
        #[test]
        fn whole_row_command_future_stays_under_its_ceiling_and_carries_no_value_row_read_b26e() {
            let (whole, value_row) = (whole_row_command_future(), value_row_read_future());
            eprintln!(
                "per-command futures (debug, bytes): whole-row {whole}, bounded {}, value-row read \
                 {value_row}",
                bounded_command_future()
            );
            assert!(
                whole <= WHOLE_ROW_COMMAND_FUTURE_CEILING,
                "the whole-row read's per-command future is {whole} B in a debug build, above its \
                 {WHOLE_ROW_COMMAND_FUTURE_CEILING} B ceiling: every command of every window of \
                 the fenced read's pass holds it at once (R1). Keep the value-row read out of \
                 issue_fanout_command"
            );
            assert!(
                whole < value_row,
                "the whole-row read's per-command future ({whole} B) must be smaller than the \
                 value-row read's own ({value_row} B): it is the size of one fred call, and holds \
                 no value-row read"
            );
        }

        /// R1: the bounded-reply read's plain per-command future holds the
        /// value-row read inline — the single-allocation shape
        /// `W_PLAIN_CMD`
        /// (`grpc::upload_admission::WATCH_SNAPSHOT_PLAIN_COMMAND_CHARGE_BYTES`)
        /// prices its plain windows at, the figure fitted on it, one heap node
        /// per command. Boxing the value-row read would shrink this future
        /// below the value-row read's own and add a second allocation per
        /// value-row command that the fit never saw.
        #[test]
        fn bounded_command_future_holds_the_value_row_read_inline_b26e() {
            let (bounded, value_row) = (bounded_command_future(), value_row_read_future());
            assert!(
                bounded >= value_row,
                "the bounded-reply read's per-command future ({bounded} B) must hold the value-row \
                 read ({value_row} B) inline, not boxed: W_PLAIN_CMD's figure was fitted on that \
                 shape"
            );
            assert!(
                bounded > whole_row_command_future(),
                "the bounded issuer's future is the larger of the two: it alone carries the \
                 value-row read"
            );
        }

        /// The ceiling on one script window's future in a debug `cfg(test)`
        /// build: 8 KiB. It measured 7,584 B (b.gvf), the size `W_CMD`
        /// (`grpc::upload_admission::WATCH_SNAPSHOT_COMMAND_CHARGE_BYTES`)
        /// was refitted with on the script transport: the read holds one such
        /// future per script window, whatever its entries.
        const SCRIPT_WINDOW_FUTURE_CEILING: usize = 8 * 1024;

        /// b.gvf: a script window's future — one per window, beside its
        /// argument list and its replies — stays under
        /// [`SCRIPT_WINDOW_FUTURE_CEILING`], and below one plain value-row
        /// command's future, which a plain window holds per command. A future
        /// past the ceiling is memory per window the `W_CMD` refit never saw:
        /// re-fit it, or move the frame that grew off the future.
        #[test]
        fn script_window_future_stays_under_its_ceiling_and_below_one_plain_command_b_gvf() {
            let (script, bounded) = (script_window_future(), bounded_command_future());
            eprintln!(
                "script window future (debug, bytes): {script}; plain bounded per-command: \
                 {bounded}"
            );
            assert!(
                script <= SCRIPT_WINDOW_FUTURE_CEILING,
                "the script window's future is {script} B in a debug build, above its \
                 {SCRIPT_WINDOW_FUTURE_CEILING} B ceiling: W_CMD's refit on the script transport \
                 never saw it"
            );
            assert!(
                script < bounded,
                "a whole script window's future ({script} B) must be smaller than one plain \
                 value-row command's ({bounded} B)"
            );
        }
    }

    /// A string key of `tv`'s, which every hash command answers `WRONGTYPE`.
    async fn wrong_type_key(tv: &crate::storage::test_support::TestValkey, name: &str) -> String {
        use fred::interfaces::KeysInterface;
        let key = format!("{}issuer-routing:{name}", tv.prefix());
        let _: () = tv
            .client()
            .pool
            .set(&key, "not a hash", None, None, false)
            .await
            .expect("fixture: SET a string key");
        key
    }

    /// R1 routing, the bounded-reply read: a field or time-sample value row
    /// ([`FanoutPage::ValueRow`]) reaches [`value_row_read`] through
    /// [`issue_bounded_fanout_command`] — a row reads as a value row's reply,
    /// and a failed row's error carries the wire command that met it
    /// (`failed_op`), so it is logged and classified under that command. The
    /// whole-row issuer, [`issue_fanout_command`], reads no value row: it
    /// refuses one without touching `failed_op`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bounded_issuer_routes_value_rows_to_the_value_row_read_and_keeps_its_failed_op_b26e() {
        use fred::interfaces::HashesInterface;
        let tv = crate::storage::test_support::TestValkey::setup().await;
        let (pool, _in_use) = tv.client().acquire_snapshot();
        for family in [SnapshotKeyFamily::Field, SnapshotKeyFamily::TimeSample] {
            let row = format!("{}issuer-routing:row:{}", tv.prefix(), family.label());
            let _: () = tv
                .client()
                .pool
                .hset(&row, [("value", "abc"), ("version", "1")])
                .await
                .expect("fixture: HSET a value row");
            let command = FanoutCommand::part(family, row, "1", FanoutPage::ValueRow);
            match issue_bounded_fanout_command(pool, &command).await {
                Ok(FanoutReply::ValueRow(reply)) => assert_eq!(
                    (reply.family, reply.value_len),
                    (family, 3),
                    "{family:?}: the row is read by its two-command read: its value's length"
                ),
                Ok(_) => panic!("{family:?}: a value row reads as a value row's reply"),
                Err(e) => panic!("{family:?}: the value row's read succeeds: {e:?}"),
            }
            assert!(command.failed_op.get().is_none(), "{family:?}: no failure");

            let broken = FanoutCommand::part(
                family,
                wrong_type_key(&tv, family.label()).await,
                "1",
                FanoutPage::ValueRow,
            );
            let Err(err) = issue_bounded_fanout_command(pool, &broken).await else {
                panic!("{family:?}: a row that is not a hash fails its read");
            };
            assert_eq!(
                broken.failed_op.get().copied(),
                Some("HSTRLEN"),
                "{family:?}: the value-row read records the wire command that failed: {err:?}"
            );
            assert_eq!(
                broken.op_label(),
                format!("HSTRLEN {}", value_row_label(family)),
                "{family:?}"
            );

            let whole = FanoutCommand::part(
                family,
                wrong_type_key(&tv, &format!("whole-{}", family.label())).await,
                "1",
                FanoutPage::ValueRow,
            );
            let Err(err) = issue_fanout_command(pool, &whole).await else {
                panic!("{family:?}: the whole-row issuer has no value-row read");
            };
            assert!(
                err.details().contains("cannot read part ValueRow"),
                "{family:?}: the whole-row issuer refuses a value row, issuing nothing: {err:?}"
            );
            assert!(
                whole.failed_op.get().is_none(),
                "{family:?}: and records no wire command"
            );
        }
    }

    /// R1 routing, the whole-row read: it never builds a value row's
    /// two-command read ([`FanoutPage::ValueRow`]), which its issuer cannot
    /// read. The fenced read's pass — whole rows ([`FanoutReplies::Whole`]),
    /// the one whole-row fan-out since b.fsz MR-D4 removed the unfenced
    /// ext-GC read — of a layer of large inline field values above the
    /// atomic-read ceilings completes, and carries every value whole.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn whole_row_read_reads_value_rows_whole_never_by_the_value_row_read_b26e() {
        use crate::storage::snapshot_fence::read::CappedFenceWait;
        use crate::storage::snapshot_fence::LayerSize;
        use crate::storage::test_support::fenced_read_progress::{
            lowered_client, posed_rig_layer, read_fenced, RIG_SECTIONS, RIG_VALUE_BYTES,
        };
        let tv = crate::storage::test_support::TestValkey::setup().await;
        let client = lowered_client(&tv, None);
        let (oid, answer) = Box::pin(posed_rig_layer(&tv, &client, "whole-row-routing")).await;
        let consumer = SnapshotReadConsumer::Checkpoint;
        let snapshot = read_fenced(
            &client,
            &tv,
            oid,
            consumer,
            LayerSize::from_too_large(&answer, None),
            &mut CappedFenceWait::new(&client, oid, consumer),
        )
        .await;
        assert_eq!(
            snapshot.path,
            SnapshotReadPath::Fenced,
            "fixture: the fenced pass"
        );
        let fields = &snapshot.state.fields;
        assert_eq!(fields.len() as u64, RIG_SECTIONS, "every field row is read");
        for field in fields {
            assert!(
                matches!(&field.value, crate::domain::types::FieldValue::Inline(v) if v.len() == RIG_VALUE_BYTES),
                "field of section {} is read whole, its inline value carried",
                field.section_id
            );
        }
    }
}
