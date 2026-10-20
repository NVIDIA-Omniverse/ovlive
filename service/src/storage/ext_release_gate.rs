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

//! The external-value release budget's upload gate (b.yuo).
//!
//! Released external-value blobs stay in the backing store until the GC loop
//! collects them: from the GC age floor after their release, or, for the
//! excess over half the release budget, at any age — except a release a live
//! catch-up read hold keeps (`GcPolicy`). The release
//! ledger accounts their bytes in `le:ext:{gc}:pending:total`
//! (`KeySchema::ext_pending_total`), and the release budget `H`
//! (`LIVE_EDIT_EXT_RELEASE_MAX_BYTES`; when unset, derived from the store's
//! own `maxmemory`, else from `LIVE_EDIT_VALKEY_MEMORY_BYTES`, the store's
//! total memory, by its cluster map, else the fallback —
//! `storage::store_memory`, which re-reads it while the service runs) gates
//! on it: an upload carrying external values
//! is held back while the total is at or above `H`. That holds back only those
//! uploads, so other sources can take the total past `H`: the releases of
//! commits admitted before it crossed, `DeleteLayer`, overwrites and edits that
//! drop references (inline-only ones included, which this gate never holds),
//! the reference-count reconciler's repairs, uploads let through when the
//! total cannot be read (below), and the waiters admitted together on the
//! first reading below `H`.
//!
//! **Where it applies.** `CreateLayer` and `UpdateLayer`, just before the
//! external-value pre-store, and only when the upload's container has external
//! parts — the uploads that can create more garbage. Never gated: inline-only
//! uploads, `DeleteLayer` (which only releases), the DR-restore pre-store and
//! the read path's fetch-on-miss (which restore what a reader needs), and the
//! `UpdateLayer` checkpoint-recovery re-issue's second pre-store of the same
//! values. That RPC already passed this gate, and the second store writes new
//! blob bytes only where GC deleted a blob since the first, freeing the same
//! bytes (otherwise it only increments); a second wait would also make the
//! recovering RPC likelier to be refused right after its own restore raised
//! the total, with a refusal promising "applied nothing" that is false
//! whenever its first call landed.
//!
//! **How it waits.** [`ExtReleaseGate::admit`] reads the total once, with an
//! `MGET` on the data pool that also reads the upload hold (below). Below `H`,
//! with no hold standing, the upload proceeds. At or above it the
//! call waits up to what is left of its own deadline minus
//! [`EXT_RELEASE_GATE_DEADLINE_MARGIN`], racing the graceful shutdown's
//! `begun` token (`GracefulShutdown::begun`), as the upload-admission waits
//! do: the handlers pass it straight in, and the shutdown-token guard lists
//! this wait (`refusal_and_shutdown_guards_test.rs`).
//! "What is left" is the request-path deadline every deadline-aware wait in
//! this service reads — `grpc::deadline::remaining`, which already folds in a
//! client `grpc-timeout` shorter than `LIVE_EDIT_RPC_DEADLINE_MS` — capped at
//! `LIVE_EDIT_RPC_DEADLINE_MS`, which is also the bound when no deadline is in
//! scope (the same derivation as the structural fence wait and the object-id
//! counter repair's wait budget). A remaining deadline at or below the margin
//! refuses at once a call that found the budget full. The wait does not use
//! `LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS`:
//! that wait is short because it runs before the upload is read and holds the
//! connection's flow-control window, while this one runs after the upload is
//! read and parsed and holds only memory the upload budget already bounds, so
//! it can afford the call's whole deadline and make a refusal rare. The cost,
//! accepted with that choice: a waiting upload keeps its upload-budget
//! reservation — and possibly the replica's single overflow slot — for up to
//! about 25 s at the default deadline, so while the release budget is full,
//! inline-only uploads, which this gate never holds itself, can be refused at
//! upload admission or wait in the processing queue until their deadline.
//! While any call waits, one shared poller per replica re-reads the total
//! every [`EXT_RELEASE_GATE_POLL_INTERVAL`] and wakes every waiter, so N
//! waiters cost one read per interval, not N. The poller exits when the last
//! waiter leaves.
//!
//! **The out-of-memory collection's upload hold (b.pa1).** While the GC
//! leader runs an out-of-memory collection (`storage::external_values`, module
//! header) it sets `le:ext:{gc}:oom:upload_hold`
//! (`KeySchema::ext_oom_upload_hold`; `PX` 15 s, deleted when the collection
//! ends), so uploads carrying external values do not refill the store while it
//! frees released ones. The gate reads the hold in the same `MGET` as the
//! total, at the first read and at every poll, and a call that finds it
//! standing waits — in the one wait described above, whatever the total —
//! until a reading shows it gone and the total below `H`. The hold never
//! refuses an upload on its own: a wait that ends with the hold still standing
//! and the total below `H` lets the upload through, and one that ends with the
//! total at or above `H` is refused exactly as without a hold. The leader never
//! waits on uploads or on this gate, and the hold lapses by its TTL when the
//! leader dies, so the hold cannot deadlock. No gate (a release budget of 0),
//! no hold.
//!
//! **Outcomes.** A wait that sees the total drop below `H` proceeds
//! (`live_edit_ext_release_gate_total{outcome="passed_after_wait"}`); a wait
//! for the upload hold alone, with the total below `H` throughout, proceeds
//! uncounted. A wait
//! that ends with the budget still full — or a call with no more than the
//! margin of its deadline left — refuses the upload `UNAVAILABLE` with
//! nothing stored, incremented, mirrored or committed (`outcome="refused"`),
//! carrying the retry hint `EXT_RELEASE_BUDGET_FULL`
//! ([`ExtReleaseRefusal::into_service_error`]); the refusal is logged once,
//! by the RPC failure path. A failed read — at the
//! first check or by the poller — lets the upload through
//! (`outcome="read_failed"`): the gate is a load-shed for a store under
//! pressure, and an unreadable total must not turn into a write outage of its
//! own. An upload admitted at once is not counted. The gate does not tick
//! `live_edit_store_write_refused_total`: nothing reached the store.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use fred::interfaces::KeysInterface;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::error::{refusal_reason, RefusalDetail, ServiceError, CAPACITY_RETRY_DELAY};
use crate::observability::metrics::{
    SharedMetrics, EXT_RELEASE_GATE_OUTCOME_PASSED_AFTER_WAIT,
    EXT_RELEASE_GATE_OUTCOME_READ_FAILED, EXT_RELEASE_GATE_OUTCOME_REFUSED,
};
use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use crate::storage::keys::KeySchema;

/// How often the shared poller re-reads the release total while uploads wait
/// at the gate. Short against the wait (up to the call's remaining deadline
/// minus [`EXT_RELEASE_GATE_DEADLINE_MARGIN`]), so a waiter proceeds within a
/// tenth of a second of the GC bringing the total below the budget; one `GET`
/// per interval per replica while anything waits.
pub const EXT_RELEASE_GATE_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// How much of a call's deadline the gate leaves unspent: it waits at most
/// the call's remaining deadline minus this, and refuses at once when no more
/// than this is left.
///
/// Five seconds covers two things. First, the refusal itself must reach the
/// client as `UNAVAILABLE` "nothing applied" — a status it can act on — rather
/// than the call dying as a bare `DEADLINE_EXCEEDED`; and an upload admitted
/// at the end of a long wait still has to store its external values and
/// commit inside the deadline. The gate's own first read of the release
/// total is paid from the wait, not the margin: [`ExtReleaseGate::admit`]
/// times the wait from before that `GET`, so a slow read shortens the wait,
/// and a read that finds the budget full after outlasting the whole wait is
/// refused as soon as it returns, only its overrun past the wait coming out
/// of the margin; such a refusal reaches the client unless the read took
/// longer than the call's whole remaining deadline, which is possible only
/// when the store's command timeout (`LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`)
/// exceeds what is left of the call. A first read that finds the total below
/// `H`, or that fails, lets the upload through however long it took: the
/// store and commit then run against whatever deadline is left, as after any
/// slow store command, and can still end in `DEADLINE_EXCEEDED`. Second, at
/// the default `LIVE_EDIT_RPC_DEADLINE_MS` of 30000 the wait stays at or
/// below 25 s, under the deadline of a client that sends no `grpc-timeout`
/// and gives up after 30 s, whose clock starts before the server's (the
/// request headers' transit), so the refusal arrives while that client still
/// waits for it.
pub const EXT_RELEASE_GATE_DEADLINE_MARGIN: Duration = Duration::from_secs(5);

/// The client-facing message of a release-budget refusal.
pub const EXT_RELEASE_BUDGET_FULL_MESSAGE: &str =
    "the backing store's external-value release budget (LIVE_EDIT_EXT_RELEASE_MAX_BYTES) is \
     full: released external values awaiting garbage collection stayed at or above it for as \
     long as the call's deadline allowed; nothing was applied, retry";

/// Why [`ExtReleaseGate::admit`] did not admit an upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtReleaseRefusal {
    /// The release total stayed at or above the budget for as long as the
    /// call's deadline allowed (or no more than
    /// [`EXT_RELEASE_GATE_DEADLINE_MARGIN`] of it was left).
    Full,
    /// The service is shutting down.
    ShuttingDown,
}

impl ExtReleaseRefusal {
    /// The error a refusal answers: always `UNAVAILABLE`, never
    /// `RESOURCE_EXHAUSTED` — this call applied nothing; a re-send is safe
    /// unless an earlier attempt of the same write is still in doubt — then
    /// use `operation_id`. A budget refusal carries the retry hint
    /// [`EXT_RELEASE_BUDGET_FULL`](refusal_reason::EXT_RELEASE_BUDGET_FULL)
    /// (no metadata, a delay drawn from [`CAPACITY_RETRY_DELAY`]) with
    /// [`EXT_RELEASE_BUDGET_FULL_MESSAGE`]: the gate runs before the
    /// external-value store and the commit, so the hint's "applied nothing"
    /// holds. A shutdown refusal is the shared shutdown status, with its
    /// `SERVER_SHUTTING_DOWN` detail, as on `UploadRefusal`. The handlers
    /// answer a shutdown refusal with `grpc::transport::server_shutting_down`
    /// instead, which also marks the completion line shutdown-caused.
    pub fn into_service_error(self) -> ServiceError {
        match self {
            ExtReleaseRefusal::Full => RefusalDetail::new(
                refusal_reason::EXT_RELEASE_BUDGET_FULL,
                Some(CAPACITY_RETRY_DELAY),
            )
            .into_unavailable_error(EXT_RELEASE_BUDGET_FULL_MESSAGE),
            ExtReleaseRefusal::ShuttingDown => {
                RefusalDetail::server_shutting_down().into_unavailable_error("server shutting down")
            }
        }
    }
}

/// The shared poller's latest reading of the release total and the upload
/// hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reading {
    /// No reading yet.
    None,
    /// What one read found.
    Levels(Levels),
    /// The read failed.
    Failed,
}

/// One read of the gate's two `{gc}` keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Levels {
    /// The release total, in bytes (absent reads 0; a negative total reads 0).
    total: u64,
    /// An out-of-memory collection's upload hold stands
    /// (`le:ext:{gc}:oom:upload_hold` exists; b.pa1).
    hold: bool,
}

#[derive(Debug, Default)]
struct PollerState {
    /// Calls waiting at the gate right now.
    waiters: usize,
    /// The shared poller task is running.
    running: bool,
}

/// A test-only replacement for the gate's read of the release total: each
/// call is one read, answering the total in bytes, or `None` for a failed read.
/// The future may sleep, so a test under paused tokio time can place the read's
/// return at an exact virtual instant. See
/// [`ExtReleaseGate::with_total_reader_for_test`].
#[cfg(any(test, feature = "test-support"))]
pub type TotalReaderForTest =
    Arc<dyn Fn() -> futures::future::BoxFuture<'static, Option<u64>> + Send + Sync>;

/// A test-only replacement for the gate's read of the release total and the
/// out-of-memory collection's upload hold (b.pa1): each call is one read,
/// answering `(total, hold)` — the total in bytes and whether the hold
/// stands — or `None` for a failed read. The future may sleep, as with
/// [`TotalReaderForTest`]. See [`ExtReleaseGate::with_levels_reader_for_test`].
#[cfg(any(test, feature = "test-support"))]
pub type LevelsReaderForTest =
    Arc<dyn Fn() -> futures::future::BoxFuture<'static, Option<(u64, bool)>> + Send + Sync>;

/// Where the gate reads the release total and the upload hold from.
// One per replica, built once at startup, so the size gap between the store
// variant and the test-only readers costs nothing worth a box on the
// production variant.
#[allow(clippy::large_enum_variant)]
enum TotalSource {
    /// `MGET <total_key> <hold_key>` on `client`'s data pool: the only source
    /// in production.
    Store {
        client: ValkeyClient,
        key: String,
        hold_key: String,
    },
    /// A test's total reader ([`ExtReleaseGate::with_total_reader_for_test`]);
    /// the hold never stands.
    #[cfg(any(test, feature = "test-support"))]
    Injected(TotalReaderForTest),
    /// A test's total-and-hold reader
    /// ([`ExtReleaseGate::with_levels_reader_for_test`]).
    #[cfg(any(test, feature = "test-support"))]
    InjectedLevels(LevelsReaderForTest),
}

/// The per-replica upload gate at the external-value release budget `H`; see
/// the module docs.
pub struct ExtReleaseGate {
    source: TotalSource,
    /// `H`, in bytes; never 0. Moves only when a budget derived from the
    /// store's memory is re-read ([`Self::set_budget_bytes`]); every admission and every GC
    /// pass reads the current value.
    budget: AtomicU64,
    rpc_deadline: Duration,
    metrics: Option<SharedMetrics>,
    poller: Mutex<PollerState>,
    reading: watch::Sender<Reading>,
}

impl std::fmt::Debug for ExtReleaseGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_key = match &self.source {
            TotalSource::Store { key, .. } => key.as_str(),
            #[cfg(any(test, feature = "test-support"))]
            TotalSource::Injected(_) | TotalSource::InjectedLevels(_) => "<injected for test>",
        };
        f.debug_struct("ExtReleaseGate")
            .field("total_key", &total_key)
            .field("budget", &self.budget_bytes())
            .field("rpc_deadline", &self.rpc_deadline)
            .finish()
    }
}

/// Holds one waiter's place for as long as its wait runs, on every exit
/// (proceeds, refused, shutdown, or the handler future dropped at its
/// deadline), so the poller stops when the last waiter leaves.
struct WaiterGuard<'a> {
    gate: &'a ExtReleaseGate,
}

impl Drop for WaiterGuard<'_> {
    fn drop(&mut self) {
        let mut st = self.gate.lock();
        st.waiters = st.waiters.saturating_sub(1);
    }
}

impl ExtReleaseGate {
    /// A gate at `budget_bytes` over `keys`' release total, counting on
    /// `metrics`. `rpc_deadline` is `LIVE_EDIT_RPC_DEADLINE_MS`: the cap on a
    /// call's remaining deadline, and the deadline assumed when none is in
    /// scope (see [`Self::wait_for`]).
    ///
    /// # Panics
    ///
    /// When `budget_bytes` is 0: a zero budget disables the gate, and the
    /// caller attaches no gate at all for it.
    pub fn new(
        client: ValkeyClient,
        keys: &KeySchema,
        budget_bytes: u64,
        rpc_deadline: Duration,
        metrics: Option<SharedMetrics>,
    ) -> Self {
        Self::from_source(
            TotalSource::Store {
                client,
                key: keys.ext_pending_total(),
                hold_key: keys.ext_oom_upload_hold(),
            },
            budget_bytes,
            rpc_deadline,
            metrics,
        )
    }

    /// **Test-only.** A gate exactly like [`Self::new`]'s, except that every
    /// read of the release total — the first read in [`Self::admit`] and each
    /// read of the shared poller — is a call of `reader`, so a test can choose
    /// each total, fail a read, or delay it under paused tokio time, with no
    /// store. The out-of-memory collection's upload hold never stands
    /// ([`Self::with_levels_reader_for_test`] scripts it too). Compiled only
    /// for tests and the `test-support` feature; production gates always read
    /// the store.
    ///
    /// # Panics
    ///
    /// When `budget_bytes` is 0, as [`Self::new`].
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_total_reader_for_test(
        reader: TotalReaderForTest,
        budget_bytes: u64,
        rpc_deadline: Duration,
        metrics: Option<SharedMetrics>,
    ) -> Self {
        Self::from_source(
            TotalSource::Injected(reader),
            budget_bytes,
            rpc_deadline,
            metrics,
        )
    }

    /// **Test-only.** [`Self::with_total_reader_for_test`], with a reader that
    /// also answers whether the out-of-memory collection's upload hold stands
    /// (b.pa1): each read is a call of `reader`, answering `(total, hold)`, or
    /// `None` for a failed read. Compiled only for tests and the
    /// `test-support` feature.
    ///
    /// # Panics
    ///
    /// When `budget_bytes` is 0, as [`Self::new`].
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_levels_reader_for_test(
        reader: LevelsReaderForTest,
        budget_bytes: u64,
        rpc_deadline: Duration,
        metrics: Option<SharedMetrics>,
    ) -> Self {
        Self::from_source(
            TotalSource::InjectedLevels(reader),
            budget_bytes,
            rpc_deadline,
            metrics,
        )
    }

    fn from_source(
        source: TotalSource,
        budget_bytes: u64,
        rpc_deadline: Duration,
        metrics: Option<SharedMetrics>,
    ) -> Self {
        assert!(
            budget_bytes > 0,
            "a zero release budget disables the gate; attach none instead"
        );
        let (reading, _) = watch::channel(Reading::None);
        Self {
            source,
            budget: AtomicU64::new(budget_bytes),
            rpc_deadline,
            metrics,
            poller: Mutex::new(PollerState::default()),
            reading,
        }
    }

    /// `H`, in bytes: the value in force now. The GC loop's pressure
    /// threshold (`H / 2`) is read from here on every pass, so it follows a
    /// re-read budget too.
    pub fn budget_bytes(&self) -> u64 {
        self.budget.load(Ordering::Relaxed)
    }

    /// Move `H` to `budget_bytes`, for a budget derived from the store's
    /// memory — its own `maxmemory`, or `LIVE_EDIT_VALKEY_MEMORY_BYTES` by its
    /// cluster map — and re-read while the service runs
    /// (`storage::store_memory`). Takes effect at the next admission check and
    /// the next GC pass; a call already waiting compares each later reading
    /// against the new value.
    ///
    /// # Panics
    ///
    /// When `budget_bytes` is 0: a re-read budget never disables the gate.
    pub fn set_budget_bytes(&self, budget_bytes: u64) {
        assert!(
            budget_bytes > 0,
            "a re-read release budget is never 0; a zero budget disables the gate at startup only"
        );
        self.budget.store(budget_bytes, Ordering::Relaxed);
    }

    /// How long a call whose deadline has `remaining` left may wait at the
    /// gate: `min(LIVE_EDIT_RPC_DEADLINE_MS, remaining)` — the RPC deadline
    /// alone when `remaining` is `None` (no deadline in scope, which bounds
    /// nothing by itself) — minus [`EXT_RELEASE_GATE_DEADLINE_MARGIN`]. Zero
    /// when no more than the margin is left: the call is refused at once.
    ///
    /// `remaining` is `grpc::deadline::remaining`, read by the handler (this
    /// module is in `storage`, which must not depend on `grpc`); the cap and
    /// the `None` arm are the derivation the structural fence wait and the
    /// object-id counter repair's wait budget use.
    pub fn wait_for(&self, remaining: Option<Duration>) -> Duration {
        let effective = match remaining {
            Some(remaining) => self.rpc_deadline.min(remaining),
            None => self.rpc_deadline,
        };
        effective.saturating_sub(EXT_RELEASE_GATE_DEADLINE_MARGIN)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PollerState> {
        // Nothing panics while holding the lock; recover the state rather than
        // wedging every upload if something ever did.
        self.poller.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Admit one upload carrying external values, whose call has `remaining`
    /// of its deadline left (`grpc::deadline::remaining`): `Ok` while the
    /// release total is below the budget and no out-of-memory collection's
    /// upload hold stands, or once both hold within the wait
    /// [`Self::wait_for`] allows, or when the levels cannot be read;
    /// [`ExtReleaseRefusal::Full`] when the total stays at or above the budget
    /// for that whole wait, at once when the wait is zero or the first read
    /// outlasted it (the wait is timed from before that read);
    /// [`ExtReleaseRefusal::ShuttingDown`] when `shutdown` fires first.
    ///
    /// The upload hold (b.pa1) shares the one wait and never refuses on its
    /// own: a wait that ends with the hold still standing and the total below
    /// the budget lets the upload through. See the module docs for the
    /// counters; a wait for the hold alone is not counted. `shutdown` is the
    /// graceful shutdown's `begun` token; it is only observed, never kept or
    /// derived from (the shutdown-token guard reads this module for that).
    pub async fn admit(
        self: &Arc<Self>,
        remaining: Option<Duration>,
        shutdown: &CancellationToken,
    ) -> Result<(), ExtReleaseRefusal> {
        if shutdown.is_cancelled() {
            return Err(ExtReleaseRefusal::ShuttingDown);
        }
        // The wait is timed from here, before the first read: `remaining` was
        // read before it, so a slow read comes out of the wait, not out of the
        // margin the call keeps after it.
        let started = tokio::time::Instant::now();
        let first = match self.read_levels().await {
            Some(levels) if !levels.hold && levels.total < self.budget_bytes() => return Ok(()),
            Some(levels) => levels,
            None => {
                self.count(EXT_RELEASE_GATE_OUTCOME_READ_FAILED);
                return Ok(());
            }
        };
        // Whether the latest reading found the budget full (against the budget
        // in force when it was read), and whether any reading of this call
        // did: only a wait that saw it full is counted, so a wait for the
        // upload hold alone is not.
        let mut full_now = first.total >= self.budget_bytes();
        let mut budget_full = full_now;
        let deadline = started + self.wait_for(remaining);
        if tokio::time::Instant::now() >= deadline {
            // No wait allowed (no more than the margin was left), or the first
            // read took all of it.
            return self.wait_ran_out(full_now, budget_full);
        }

        // Subscribe before joining, so a reading the poller publishes after
        // this point is seen.
        let mut readings = self.reading.subscribe();
        let _waiter = self.join_waiters();
        loop {
            tokio::select! {
                biased;
                () = shutdown.cancelled() => return Err(ExtReleaseRefusal::ShuttingDown),
                changed = readings.changed() => {
                    if changed.is_err() {
                        // The sender lives as long as the gate; unreachable.
                        return Ok(());
                    }
                    let reading = *readings.borrow_and_update();
                    match reading {
                        Reading::Levels(levels) => {
                            full_now = levels.total >= self.budget_bytes();
                            budget_full |= full_now;
                            if !full_now && !levels.hold {
                                if budget_full {
                                    self.count(EXT_RELEASE_GATE_OUTCOME_PASSED_AFTER_WAIT);
                                }
                                return Ok(());
                            }
                        }
                        Reading::Failed => {
                            self.count(EXT_RELEASE_GATE_OUTCOME_READ_FAILED);
                            return Ok(());
                        }
                        Reading::None => {}
                    }
                }
                () = tokio::time::sleep_until(deadline) => {
                    return self.wait_ran_out(full_now, budget_full);
                }
            }
        }
    }

    /// The end of a wait that ran out: [`ExtReleaseRefusal::Full`] when the
    /// latest reading found the budget full (`full_now`); otherwise only the
    /// upload hold was still standing, which never refuses an upload on its
    /// own, so the upload proceeds — counted as `passed_after_wait` when an
    /// earlier reading of the call found the budget full (`budget_full`),
    /// uncounted when it waited for the hold alone.
    fn wait_ran_out(&self, full_now: bool, budget_full: bool) -> Result<(), ExtReleaseRefusal> {
        if full_now {
            self.count(EXT_RELEASE_GATE_OUTCOME_REFUSED);
            return Err(ExtReleaseRefusal::Full);
        }
        if budget_full {
            self.count(EXT_RELEASE_GATE_OUTCOME_PASSED_AFTER_WAIT);
        }
        Ok(())
    }

    /// Register one waiter, starting the shared poller if it is not running.
    fn join_waiters(self: &Arc<Self>) -> WaiterGuard<'_> {
        let mut st = self.lock();
        st.waiters += 1;
        if !st.running {
            st.running = true;
            let gate = Arc::clone(self);
            tokio::spawn(async move { gate.poll_while_waited().await });
        }
        WaiterGuard { gate: self }
    }

    /// The shared poller: read the total and the upload hold, publish them,
    /// sleep, and stop once no call waits. The waiter count is checked under
    /// the same lock a joining waiter takes, so a waiter never joins a poller
    /// that is about to stop.
    async fn poll_while_waited(self: Arc<Self>) {
        loop {
            let reading = match self.read_levels().await {
                Some(levels) => Reading::Levels(levels),
                None => Reading::Failed,
            };
            self.reading.send_replace(reading);
            tokio::time::sleep(EXT_RELEASE_GATE_POLL_INTERVAL).await;
            let mut st = self.lock();
            if st.waiters == 0 {
                st.running = false;
                return;
            }
        }
    }

    /// `MGET le:ext:{gc}:pending:total le:ext:{gc}:oom:upload_hold` on the
    /// data pool — one round trip, both keys on the `{gc}` slot: `None` when
    /// the read fails (logged at WARN); an absent or negative total reads 0,
    /// and the hold stands while its key exists. `MGET` answers nil for a key
    /// that is not a string, so a wrong-typed total reads 0 (and a
    /// wrong-typed hold key as no hold) rather than failing the read; the GC
    /// pass's own read of the total (`gc_select.lua`) still fails on it.
    async fn read_levels(&self) -> Option<Levels> {
        let (client, key, hold_key) = match &self.source {
            TotalSource::Store {
                client,
                key,
                hold_key,
            } => (client, key, hold_key),
            #[cfg(any(test, feature = "test-support"))]
            TotalSource::Injected(reader) => {
                return reader().await.map(|total| Levels { total, hold: false });
            }
            #[cfg(any(test, feature = "test-support"))]
            TotalSource::InjectedLevels(reader) => {
                return reader().await.map(|(total, hold)| Levels { total, hold });
            }
        };
        let read: Result<(Option<i64>, Option<String>), fred::error::Error> = client
            .pool
            .mget(vec![key.as_str(), hold_key.as_str()])
            .await;
        match read {
            Ok((total, hold)) => Some(Levels {
                total: u64::try_from(total.unwrap_or(0)).unwrap_or(0),
                hold: hold.is_some(),
            }),
            Err(e) => {
                FredErrOrigin::client(client, POOL_LABEL_DATA).observe_error(&e);
                tracing::warn!(
                    error = %e,
                    "could not read the external-value release total; letting the upload through \
                     the release-budget gate"
                );
                None
            }
        }
    }

    fn count(&self, outcome: &'static str) {
        if let Some(m) = self.metrics.as_ref() {
            m.inc_ext_release_gate(outcome);
        }
    }
}

#[cfg(test)]
mod tests {
    //! The gate's wait bound, its end points, its outcome counters, and the
    //! out-of-memory collection's upload hold (b.pa1).
    //!
    //! The end points run under paused tokio time against a scripted total
    //! ([`ExtReleaseGate::with_total_reader_for_test`]), or a scripted total
    //! and hold ([`ExtReleaseGate::with_levels_reader_for_test`]): each read
    //! answers what the test sets, or fails, after a delay the test chooses,
    //! so a slow first read, the total's drop and the hold's clearing land at
    //! exact virtual instants, and each test asserts the exact instant the
    //! call ends — no wall-clock bound. Four tests read a real Valkey instead,
    //! for what only the store source does: an absent total reads 0, a
    //! non-integer total fails the read and lets the upload through, a
    //! wrong-typed total reads 0, and the hold key is read beside the total.

    use std::collections::VecDeque;

    use super::*;
    use fred::interfaces::HashesInterface;

    use crate::observability::metrics::test_util::{render, sample_value};
    use crate::observability::metrics::{metrics_scrape_only, Metrics};
    use crate::storage::test_support::TestValkey;

    /// The release budget every gate here runs at.
    const BUDGET: u64 = 1_000;

    /// The default `LIVE_EDIT_RPC_DEADLINE_MS`.
    const RPC_DEADLINE: Duration = Duration::from_secs(30);

    /// The default `LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS` at the default RPC
    /// deadline (`grpc::upload_admission::UPLOAD_ADMISSION_WAIT_MS_DEFAULT`,
    /// restated: `storage` does not depend on `grpc`). The gate's wait must
    /// not be bounded by it.
    const UPLOAD_ADMISSION_WAIT_DEFAULT: Duration = Duration::from_millis(2_000);

    fn metrics() -> Arc<Metrics> {
        Arc::new(metrics_scrape_only("ext-release-gate-test"))
    }

    /// A gate over `tv`'s real release total.
    fn store_gate(tv: &TestValkey) -> (Arc<ExtReleaseGate>, Arc<Metrics>) {
        let metrics = metrics();
        let gate = ExtReleaseGate::new(
            tv.client().clone(),
            tv.key_schema(),
            BUDGET,
            RPC_DEADLINE,
            Some(Arc::clone(&metrics)),
        );
        (Arc::new(gate), metrics)
    }

    /// What the scripted reader answers.
    struct Script {
        /// The total each read answers; `None` is a failed read.
        total: Option<u64>,
        /// Whether the out-of-memory collection's upload hold stands at each
        /// read (b.pa1). Read only by [`scripted_levels_gate`]'s reader; the
        /// total-only reader of [`scripted_gate`] never reports a hold.
        hold: bool,
        /// The delay of each next read, in order; a read with none queued
        /// returns at once.
        delays: VecDeque<Duration>,
        /// Reads so far.
        reads: usize,
    }

    type Scripted = Arc<Mutex<Script>>;

    /// A gate whose every read of the total follows the returned script: the
    /// read waits its queued delay (on the tokio clock), then answers the
    /// script's total at that moment.
    fn scripted_gate(total: Option<u64>) -> (Arc<ExtReleaseGate>, Arc<Metrics>, Scripted) {
        let script: Scripted = Arc::new(Mutex::new(Script {
            total,
            hold: false,
            delays: VecDeque::new(),
            reads: 0,
        }));
        let reader: TotalReaderForTest = {
            let script = Arc::clone(&script);
            Arc::new(
                move || -> futures::future::BoxFuture<'static, Option<u64>> {
                    let script = Arc::clone(&script);
                    Box::pin(async move {
                        let delay = {
                            let mut s = script.lock().unwrap();
                            s.reads += 1;
                            s.delays.pop_front()
                        };
                        if let Some(delay) = delay {
                            tokio::time::sleep(delay).await;
                        }
                        script.lock().unwrap().total
                    })
                },
            )
        };
        let metrics = metrics();
        let gate = ExtReleaseGate::with_total_reader_for_test(
            reader,
            BUDGET,
            RPC_DEADLINE,
            Some(Arc::clone(&metrics)),
        );
        (Arc::new(gate), metrics, script)
    }

    /// [`scripted_gate`] through [`ExtReleaseGate::with_levels_reader_for_test`]:
    /// every read also answers whether the out-of-memory collection's upload
    /// hold stands (b.pa1), as the script says at that moment.
    fn scripted_levels_gate(
        total: Option<u64>,
        hold: bool,
    ) -> (Arc<ExtReleaseGate>, Arc<Metrics>, Scripted) {
        let script: Scripted = Arc::new(Mutex::new(Script {
            total,
            hold,
            delays: VecDeque::new(),
            reads: 0,
        }));
        let reader: LevelsReaderForTest = {
            let script = Arc::clone(&script);
            Arc::new(
                move || -> futures::future::BoxFuture<'static, Option<(u64, bool)>> {
                    let script = Arc::clone(&script);
                    Box::pin(async move {
                        let delay = {
                            let mut s = script.lock().unwrap();
                            s.reads += 1;
                            s.delays.pop_front()
                        };
                        if let Some(delay) = delay {
                            tokio::time::sleep(delay).await;
                        }
                        let s = script.lock().unwrap();
                        s.total.map(|total| (total, s.hold))
                    })
                },
            )
        };
        let metrics = metrics();
        let gate = ExtReleaseGate::with_levels_reader_for_test(
            reader,
            BUDGET,
            RPC_DEADLINE,
            Some(Arc::clone(&metrics)),
        );
        (Arc::new(gate), metrics, script)
    }

    fn set_scripted_total(script: &Scripted, total: Option<u64>) {
        script.lock().unwrap().total = total;
    }

    fn set_scripted_hold(script: &Scripted, hold: bool) {
        script.lock().unwrap().hold = hold;
    }

    fn reads(script: &Scripted) -> usize {
        script.lock().unwrap().reads
    }

    async fn set_total(tv: &TestValkey, total: u64) {
        let _: () = tv
            .client()
            .pool
            .set(
                tv.key_schema().ext_pending_total(),
                total.to_string(),
                None,
                None,
                false,
            )
            .await
            .expect("SET pending:total must succeed");
    }

    /// `(passed_after_wait, refused, read_failed)` of
    /// `live_edit_ext_release_gate_total`.
    fn cells(metrics: &Metrics) -> (u64, u64, u64) {
        let text = render(metrics);
        let cell = |outcome: &str| {
            sample_value(
                &text,
                "live_edit_ext_release_gate_total",
                &[("outcome", outcome)],
            )
            .unwrap_or_else(|| panic!("outcome={outcome} is zero-seeded")) as u64
        };
        (
            cell(EXT_RELEASE_GATE_OUTCOME_PASSED_AFTER_WAIT),
            cell(EXT_RELEASE_GATE_OUTCOME_REFUSED),
            cell(EXT_RELEASE_GATE_OUTCOME_READ_FAILED),
        )
    }

    /// A remaining deadline of `margin + wait`, so [`ExtReleaseGate::wait_for`]
    /// allows exactly `wait` under a larger RPC deadline.
    fn remaining_for_wait(wait: Duration) -> Option<Duration> {
        Some(EXT_RELEASE_GATE_DEADLINE_MARGIN + wait)
    }

    /// Spawn one `admit` call that answers its result and the virtual time
    /// it took.
    fn spawn_admit(
        gate: &Arc<ExtReleaseGate>,
        remaining: Option<Duration>,
        shutdown: &CancellationToken,
    ) -> tokio::task::JoinHandle<(Result<(), ExtReleaseRefusal>, Duration)> {
        let gate = Arc::clone(gate);
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            let started = tokio::time::Instant::now();
            let result = gate.admit(remaining, &shutdown).await;
            (result, started.elapsed())
        })
    }

    /// The wait is the call's remaining deadline, capped at the RPC deadline,
    /// less the margin; the RPC deadline alone when no deadline is in scope;
    /// and zero at or below the margin.
    #[test]
    fn wait_for_is_the_capped_remaining_deadline_less_the_margin() {
        let (gate, _, _) = scripted_gate(Some(0));
        let margin = EXT_RELEASE_GATE_DEADLINE_MARGIN;
        assert_eq!(margin, Duration::from_secs(5), "the margin the docs state");

        assert_eq!(
            gate.wait_for(None),
            RPC_DEADLINE - margin,
            "no deadline in scope: the RPC deadline bounds the wait"
        );
        assert_eq!(
            gate.wait_for(Some(Duration::from_secs(60))),
            RPC_DEADLINE - margin,
            "a remaining deadline above the RPC deadline is capped at it"
        );
        assert_eq!(
            gate.wait_for(Some(Duration::from_secs(12))),
            Duration::from_secs(7),
            "a shorter remaining deadline (a client grpc-timeout) bounds the wait"
        );
        assert_eq!(
            gate.wait_for(Some(margin + Duration::from_millis(1))),
            Duration::from_millis(1)
        );
        assert_eq!(gate.wait_for(Some(margin)), Duration::ZERO);
        assert_eq!(gate.wait_for(Some(Duration::from_secs(1))), Duration::ZERO);
        assert_eq!(gate.wait_for(Some(Duration::ZERO)), Duration::ZERO);
        assert!(
            gate.wait_for(None) <= Duration::from_secs(25),
            "at the default RPC deadline the wait stays at or below 25 s, under a 30 s \
             client deadline"
        );
    }

    /// Below the budget an upload proceeds at once and nothing is counted. An
    /// absent total — the store's ledger before any release — reads 0.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_upload_below_the_budget_is_admitted_at_once_and_counts_nothing() {
        let tv = TestValkey::setup().await;
        let (gate, metrics) = store_gate(&tv);
        let shutdown = CancellationToken::new();

        gate.admit(None, &shutdown)
            .await
            .expect("an absent total is below any budget");
        set_total(&tv, BUDGET - 1).await;
        gate.admit(None, &shutdown)
            .await
            .expect("one byte below the budget is admitted");
        assert_eq!(cells(&metrics), (0, 0, 0));
    }

    /// With no more than the margin of its deadline left, a call at a full
    /// budget is refused at the instant it reads the total — no wait, no
    /// poller — and counted `refused`.
    #[tokio::test(start_paused = true)]
    async fn a_remaining_deadline_at_or_below_the_margin_is_refused_at_once() {
        let (gate, metrics, script) = scripted_gate(Some(BUDGET));
        let shutdown = CancellationToken::new();

        for remaining in [EXT_RELEASE_GATE_DEADLINE_MARGIN, Duration::from_millis(10)] {
            let (result, took) = spawn_admit(&gate, Some(remaining), &shutdown)
                .await
                .expect("the call must not panic");
            assert_eq!(
                result,
                Err(ExtReleaseRefusal::Full),
                "remaining {remaining:?}: a full budget with no wait left must refuse"
            );
            assert_eq!(took, Duration::ZERO, "remaining {remaining:?}: no wait");
        }
        assert_eq!(cells(&metrics), (0, 2, 0), "each refusal counts once");
        assert_eq!(reads(&script), 2, "one read per call, and no poller reads");
        let poller = gate.lock();
        assert!(
            !poller.running && poller.waiters == 0,
            "an immediate refusal joins no waiters and starts no poller"
        );
    }

    /// A waiter at a full budget proceeds at the first poll that sees the
    /// total below it, and is counted `passed_after_wait`. The total drops at
    /// 2 450 ms, after the default upload-admission wait — the gate's wait is
    /// bounded by the call's deadline, not by `LIVE_EDIT_UPLOAD_ADMISSION_WAIT_MS`
    /// — and the shared poller, which reads at the waiter's start and every
    /// 100 ms after, sees it at 2 500 ms.
    #[tokio::test(start_paused = true)]
    async fn a_waiter_proceeds_when_the_total_drops_after_longer_than_the_admission_wait() {
        let (gate, metrics, script) = scripted_gate(Some(BUDGET + 5));
        let shutdown = CancellationToken::new();

        let waiter = spawn_admit(&gate, remaining_for_wait(Duration::from_secs(8)), &shutdown);
        let drop_at = UPLOAD_ADMISSION_WAIT_DEFAULT + Duration::from_millis(450);
        tokio::time::sleep(drop_at).await;
        assert!(
            !waiter.is_finished(),
            "the waiter must still be waiting past the admission wait"
        );
        set_scripted_total(&script, Some(BUDGET - 1));

        let (admitted, took) = waiter.await.expect("the waiter must not panic");
        assert_eq!(admitted, Ok(()));
        assert_eq!(
            took,
            drop_at + Duration::from_millis(50),
            "admitted at the first poll after the drop"
        );
        assert_eq!(
            took.as_millis() % EXT_RELEASE_GATE_POLL_INTERVAL.as_millis(),
            0
        );
        assert_eq!(cells(&metrics), (1, 0, 0));
    }

    /// A wait that never sees the total drop ends at exactly `started +
    /// wait_for`, refused `Full` and counted `refused`, and leaves no waiter
    /// behind.
    #[tokio::test(start_paused = true)]
    async fn a_wait_that_stays_full_is_refused_at_its_end_point() {
        let (gate, metrics, _) = scripted_gate(Some(BUDGET));
        let shutdown = CancellationToken::new();
        let wait = Duration::from_millis(600);

        let (refusal, took) = spawn_admit(&gate, remaining_for_wait(wait), &shutdown)
            .await
            .expect("the call must not panic");
        assert_eq!(refusal, Err(ExtReleaseRefusal::Full));
        assert_eq!(took, wait, "the refusal comes at the end of the wait");
        assert_eq!(cells(&metrics), (0, 1, 0));
        assert_eq!(gate.lock().waiters, 0, "the waiter left on refusal");
    }

    /// The wait is timed from BEFORE the first read of the total: a first read
    /// that takes 500 ms comes out of the 800 ms wait, so the refusal lands at
    /// exactly 800 ms — timed from after the read it would land at 1 300 ms.
    #[tokio::test(start_paused = true)]
    async fn a_waiters_end_point_is_measured_from_before_its_first_read() {
        let (gate, metrics, script) = scripted_gate(Some(BUDGET));
        let shutdown = CancellationToken::new();
        let wait = Duration::from_millis(800);
        let first_read = Duration::from_millis(500);
        script.lock().unwrap().delays.push_back(first_read);

        let (refusal, took) = spawn_admit(&gate, remaining_for_wait(wait), &shutdown)
            .await
            .expect("the call must not panic");
        assert_eq!(refusal, Err(ExtReleaseRefusal::Full));
        assert_eq!(
            took, wait,
            "the refusal must land at started + wait, the first read's {first_read:?} paid \
             from the wait"
        );
        assert_eq!(cells(&metrics), (0, 1, 0));
    }

    /// A first read that outlasts the whole wait and finds the budget full is
    /// refused at the instant it returns — counted `refused`, never
    /// `passed_after_wait`, and with no poller started.
    #[tokio::test(start_paused = true)]
    async fn a_first_read_slower_than_the_wait_is_refused_as_soon_as_it_returns() {
        let (gate, metrics, script) = scripted_gate(Some(BUDGET));
        let shutdown = CancellationToken::new();
        let wait = Duration::from_millis(200);
        let first_read = Duration::from_millis(700);
        script.lock().unwrap().delays.push_back(first_read);

        let (refusal, took) = spawn_admit(&gate, remaining_for_wait(wait), &shutdown)
            .await
            .expect("the call must not panic");
        assert_eq!(refusal, Err(ExtReleaseRefusal::Full));
        assert_eq!(
            took, first_read,
            "the refusal must follow the read's return with no further wait"
        );
        assert_eq!(cells(&metrics), (0, 1, 0));
        assert_eq!(reads(&script), 1, "no poller read");
        let poller = gate.lock();
        assert!(!poller.running && poller.waiters == 0);
    }

    /// A total the gate cannot read lets the upload through and is counted
    /// `read_failed` (policy 6: the gate is a load-shed, never an outage of
    /// its own). A total that is a string but not an integer fails the
    /// decode of the store's `MGET` reply.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unreadable_total_lets_the_upload_through_and_counts_read_failed() {
        let tv = TestValkey::setup().await;
        let (gate, metrics) = store_gate(&tv);
        let _: () = tv
            .client()
            .pool
            .set(
                tv.key_schema().ext_pending_total(),
                "not-an-integer",
                None,
                None,
                false,
            )
            .await
            .expect("SET pending:total must succeed");
        let shutdown = CancellationToken::new();

        gate.admit(None, &shutdown)
            .await
            .expect("an unreadable total must let the upload through");
        assert_eq!(cells(&metrics), (0, 0, 1));
    }

    /// A total key of the wrong type reads as no total at all: `MGET` answers
    /// nil for a key that is not a string (b.pa1), so the upload is admitted
    /// at once and nothing is counted — unlike a non-integer string, which
    /// fails the read. The GC pass's own read of the total still fails on it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_wrong_typed_total_reads_as_zero_and_admits_uncounted() {
        let tv = TestValkey::setup().await;
        let (gate, metrics) = store_gate(&tv);
        let _: () = tv
            .client()
            .pool
            .hset(tv.key_schema().ext_pending_total(), [("not", "a-string")])
            .await
            .expect("HSET must succeed");
        let shutdown = CancellationToken::new();

        gate.admit(None, &shutdown)
            .await
            .expect("a wrong-typed total reads 0, below any budget");
        assert_eq!(cells(&metrics), (0, 0, 0), "nothing counted");
    }

    /// The store source reads the out-of-memory collection's upload hold in
    /// the same `MGET` as the total (b.pa1): with the hold key set and no
    /// total at all, a call waits, and proceeds — uncounted — once the key is
    /// deleted. Without the hold in the read the call would be admitted at
    /// once, and the probe for a waiting call would run out.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_store_gate_holds_an_upload_while_the_upload_hold_key_exists() {
        let tv = TestValkey::setup().await;
        let (gate, metrics) = store_gate(&tv);
        let hold_key = tv.key_schema().ext_oom_upload_hold();
        let _: () = tv
            .client()
            .pool
            .set(&hold_key, "collection-token", None, None, false)
            .await
            .expect("SET the upload hold must succeed");
        let shutdown = CancellationToken::new();

        let waiter = spawn_admit(&gate, None, &shutdown);
        tokio::time::timeout(Duration::from_secs(5), async {
            while gate.lock().waiters != 1 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("the call must wait while the upload hold key exists");
        assert!(!waiter.is_finished(), "precondition: the call waits");
        let _: i64 = tv
            .client()
            .pool
            .del(&hold_key)
            .await
            .expect("DEL the upload hold must succeed");

        let (admitted, _) = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("the call must end once the hold is gone")
            .expect("the waiter must not panic");
        assert_eq!(admitted, Ok(()));
        assert_eq!(
            cells(&metrics),
            (0, 0, 0),
            "a wait for the hold alone counts nothing"
        );
    }

    /// GT1 (b.pa1): a call that finds the upload hold standing, with the
    /// total below the budget, waits — past the default upload-admission
    /// wait — and proceeds at the first poll that finds the hold gone,
    /// uncounted: the wait was for the hold alone.
    #[tokio::test(start_paused = true)]
    async fn an_upload_waits_while_the_upload_hold_stands_and_proceeds_when_it_clears() {
        let (gate, metrics, script) = scripted_levels_gate(Some(BUDGET - 1), true);
        let shutdown = CancellationToken::new();

        let waiter = spawn_admit(&gate, remaining_for_wait(Duration::from_secs(8)), &shutdown);
        let clear_at = UPLOAD_ADMISSION_WAIT_DEFAULT + Duration::from_millis(450);
        tokio::time::sleep(clear_at).await;
        assert!(
            !waiter.is_finished(),
            "the call must wait while the hold stands, whatever the total"
        );
        set_scripted_hold(&script, false);

        let (admitted, took) = waiter.await.expect("the waiter must not panic");
        assert_eq!(admitted, Ok(()));
        assert_eq!(
            took,
            clear_at + Duration::from_millis(50),
            "admitted at the first poll after the hold cleared"
        );
        assert_eq!(
            cells(&metrics),
            (0, 0, 0),
            "a wait for the hold alone is not counted"
        );
    }

    /// GT2 (b.pa1): the upload hold never refuses an upload on its own. A
    /// call whose whole wait finds the hold standing and the total below the
    /// budget proceeds at the end of the wait — not `Full` — uncounted, and
    /// leaves no waiter behind.
    #[tokio::test(start_paused = true)]
    async fn the_upload_hold_never_refuses_an_upload_on_its_own() {
        let (gate, metrics, _) = scripted_levels_gate(Some(BUDGET - 1), true);
        let shutdown = CancellationToken::new();
        let wait = Duration::from_millis(600);

        let (admitted, took) = spawn_admit(&gate, remaining_for_wait(wait), &shutdown)
            .await
            .expect("the call must not panic");
        assert_eq!(
            admitted,
            Ok(()),
            "a wait that ends with only the hold standing must let the upload through"
        );
        assert_eq!(took, wait, "it proceeds at the end of the wait");
        assert_eq!(cells(&metrics), (0, 0, 0));
        assert_eq!(gate.lock().waiters, 0, "the waiter left");
    }

    /// GT3 (b.pa1): the upload hold and a full budget share ONE wait, timed
    /// from before the first read. The hold stands at first (total below the
    /// budget); at 250 ms it clears just as the total reaches the budget, which
    /// the 300 ms poll sees. The call is refused `Full` at `started + wait`
    /// (800 ms) — a wait restarted when the hold cleared would end at
    /// 1 100 ms.
    #[tokio::test(start_paused = true)]
    async fn the_upload_hold_and_a_full_budget_share_one_wait() {
        let (gate, metrics, script) = scripted_levels_gate(Some(BUDGET - 1), true);
        let shutdown = CancellationToken::new();
        let wait = Duration::from_millis(800);

        let waiter = spawn_admit(&gate, remaining_for_wait(wait), &shutdown);
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(
            !waiter.is_finished(),
            "precondition: the call waits on the hold"
        );
        {
            let mut s = script.lock().unwrap();
            s.hold = false;
            s.total = Some(BUDGET);
        }

        let (refusal, took) = waiter.await.expect("the waiter must not panic");
        assert_eq!(refusal, Err(ExtReleaseRefusal::Full));
        assert_eq!(
            took, wait,
            "the refusal lands at started + wait: the hold's part of the wait is not given back"
        );
        assert_eq!(cells(&metrics), (0, 1, 0), "counted refused once");
    }

    /// A wait that found the budget full and ends with only the upload hold
    /// still standing (the total back below the budget) proceeds at its end,
    /// counted `passed_after_wait`: the call did wait at a full budget
    /// (b.pa1).
    #[tokio::test(start_paused = true)]
    async fn a_wait_that_saw_the_budget_full_and_ends_on_the_hold_alone_counts_passed_after_wait() {
        let (gate, metrics, script) = scripted_levels_gate(Some(BUDGET), false);
        let shutdown = CancellationToken::new();
        let wait = Duration::from_millis(800);

        let waiter = spawn_admit(&gate, remaining_for_wait(wait), &shutdown);
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(
            !waiter.is_finished(),
            "precondition: the call waits on the budget"
        );
        {
            let mut s = script.lock().unwrap();
            s.hold = true;
            s.total = Some(BUDGET - 1);
        }

        let (admitted, took) = waiter.await.expect("the waiter must not panic");
        assert_eq!(admitted, Ok(()), "the hold alone never refuses");
        assert_eq!(took, wait, "it proceeds at the end of the one wait");
        assert_eq!(cells(&metrics), (1, 0, 0));
    }

    /// GT4 (b.pa1): a graceful shutdown ends a wait for the upload hold at
    /// its instant, answered `ShuttingDown` and not counted.
    #[tokio::test(start_paused = true)]
    async fn a_shutdown_ends_an_upload_hold_wait() {
        let (gate, metrics, _) = scripted_levels_gate(Some(BUDGET - 1), true);
        let shutdown = CancellationToken::new();

        let waiter = spawn_admit(&gate, None, &shutdown);
        let at = Duration::from_millis(300);
        tokio::time::sleep(at).await;
        assert!(
            !waiter.is_finished(),
            "precondition: the call waits on the hold"
        );
        shutdown.cancel();
        let (ended, took) = waiter.await.expect("the waiter must not panic");
        assert_eq!(ended, Err(ExtReleaseRefusal::ShuttingDown));
        assert_eq!(took, at, "the wait ends at the shutdown");
        assert_eq!(cells(&metrics), (0, 0, 0));
        assert_eq!(gate.lock().waiters, 0, "the waiter left on shutdown");
    }

    /// A read that fails while a call waits — the shared poller's — also lets
    /// the upload through, at that poll, counted `read_failed` and never
    /// `refused`.
    #[tokio::test(start_paused = true)]
    async fn a_poller_read_that_fails_lets_the_waiter_through() {
        let (gate, metrics, script) = scripted_gate(Some(BUDGET));
        let shutdown = CancellationToken::new();

        let waiter = spawn_admit(&gate, None, &shutdown);
        tokio::time::sleep(Duration::from_millis(350)).await;
        assert!(!waiter.is_finished(), "precondition: the call waits");
        set_scripted_total(&script, None);

        let (result, took) = waiter.await.expect("the waiter must not panic");
        assert_eq!(result, Ok(()), "a failed read lets the upload through");
        assert_eq!(
            took,
            Duration::from_millis(400),
            "at the first poll after the failure"
        );
        assert_eq!(cells(&metrics), (0, 0, 1));
    }

    /// A shutdown before or during the wait answers `ShuttingDown` — during
    /// it, at the instant of the shutdown — and is not counted as a budget
    /// refusal.
    #[tokio::test(start_paused = true)]
    async fn a_shutdown_ends_the_wait_without_counting_a_refusal() {
        let (gate, metrics, _) = scripted_gate(Some(BUDGET));

        let already = CancellationToken::new();
        already.cancel();
        assert_eq!(
            gate.admit(None, &already).await,
            Err(ExtReleaseRefusal::ShuttingDown)
        );

        let shutdown = CancellationToken::new();
        let waiter = spawn_admit(&gate, None, &shutdown);
        let at = Duration::from_millis(300);
        tokio::time::sleep(at).await;
        assert!(!waiter.is_finished(), "precondition: the call waits");
        shutdown.cancel();
        let (ended, took) = waiter.await.expect("the waiter must not panic");
        assert_eq!(ended, Err(ExtReleaseRefusal::ShuttingDown));
        assert_eq!(took, at, "the wait ends at the shutdown");
        assert_eq!(cells(&metrics), (0, 0, 0));
        assert_eq!(gate.lock().waiters, 0, "the waiter left on shutdown");
    }

    /// A budget refusal is `UNAVAILABLE` "nothing applied, retry", naming the
    /// setting, and carries the retry hint `EXT_RELEASE_BUDGET_FULL`: no
    /// metadata, a delay in `CAPACITY_RETRY_DELAY`, in the service's refusal
    /// domain — read back off the status its detail builds
    /// (`RefusalDetail::unavailable`). The `ServiceError` → `Status` mapping a
    /// client sees is checked end to end in `ext_release_gate_test`, and the
    /// reason's spelling and the delay range in `error.rs`. A shutdown refusal
    /// is the shutdown status, with its `SERVER_SHUTTING_DOWN` detail.
    #[test]
    fn a_refusal_is_unavailable_and_says_nothing_was_applied() {
        use tonic_types::StatusExt;

        match ExtReleaseRefusal::Full.into_service_error() {
            ServiceError::UnavailableWithDetail { message, detail } => {
                assert_eq!(message, EXT_RELEASE_BUDGET_FULL_MESSAGE);
                for fragment in [
                    "LIVE_EDIT_EXT_RELEASE_MAX_BYTES",
                    "nothing was applied",
                    "retry",
                ] {
                    assert!(
                        message.contains(fragment),
                        "{fragment:?} missing from {message:?}"
                    );
                }
                assert_eq!(
                    *detail,
                    RefusalDetail::new(
                        refusal_reason::EXT_RELEASE_BUDGET_FULL,
                        Some(CAPACITY_RETRY_DELAY)
                    ),
                    "the budget refusal's reason, no metadata, and the capacity delay range"
                );
                let status = detail.unavailable(message.clone());
                assert_eq!(status.code(), tonic::Code::Unavailable);
                assert_eq!(status.message(), message, "the message is unchanged");
                let info = status
                    .get_details_error_info()
                    .unwrap_or_else(|| panic!("no ErrorInfo on {status:?}"));
                assert_eq!(
                    (info.reason.as_str(), info.domain.as_str()),
                    (
                        refusal_reason::EXT_RELEASE_BUDGET_FULL,
                        crate::error::REFUSAL_DETAIL_DOMAIN
                    ),
                );
                assert!(info.metadata.is_empty(), "no metadata: {:?}", info.metadata);
                let delay = status
                    .get_details_retry_info()
                    .and_then(|retry| retry.retry_delay)
                    .unwrap_or_else(|| panic!("no RetryInfo delay on {status:?}"));
                assert!(
                    CAPACITY_RETRY_DELAY.min <= delay && delay < CAPACITY_RETRY_DELAY.max,
                    "retry_delay {delay:?} outside {CAPACITY_RETRY_DELAY:?}"
                );
            }
            other => panic!("a budget refusal must be UnavailableWithDetail; got {other:?}"),
        }
        assert!(matches!(
            ExtReleaseRefusal::ShuttingDown.into_service_error(),
            ServiceError::UnavailableWithDetail { message, detail }
                if message == "server shutting down"
                    && *detail == RefusalDetail::server_shutting_down()
        ));
    }
}
