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

//! `SubscribeReadLayer` admission: three gates and the budget that bounds
//! waiting for them.
//!
//! - [`SubscriberAdmissionGate`] caps concurrent subscriptions per replica at
//!   the subscription ceiling, derived from the pod memory limit
//!   (`config::memory_plan`; 4096 at the chart's 5 GiB), with a bounded,
//!   cancellable wait. A gate of 0 admits without bound; the configuration
//!   never builds one, and the same holds for the catch-up gate's 0.
//!   Subscriptions hold no Valkey connection
//!   (they share the stream multiplexer's readers), so an established one costs
//!   only memory; the ceiling bounds that memory.
//! - [`CatchUpGate`] bounds catch-ups running at once
//!   (`LIVE_EDIT_SUBSCRIBER_MAX_CONCURRENT_CATCHUPS`). A catch-up holds a slot
//!   only across its backing-store reads and its build — where a subscribe's
//!   memory is spent — and, in rollback mode
//!   (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`) on a same-object reconnect,
//!   the cluster stream-reader wait; never across a send, so a join storm queues here
//!   instead of allocating every catch-up at once, and a slow client cannot
//!   hold a slot. A shared fenced-read flight (`storage::snapshot_flights`)
//!   is one unit of catch-up work: it holds one slot from its read, through
//!   its fence sleeps and its permit and memory waits, to its attached
//!   catch-ups' last build (they hold none), waits for it with
//!   [`no_deadline`], and takes one more per extra build with
//!   [`CatchUpGate::try_acquire`].
//! - [`CatchUpReadGate`] bounds the catch-up snapshot reads of one permit
//!   class running at once: one instance for atomic reads
//!   (`LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY`), one for fenced reads of
//!   layers above the atomic-read ceilings
//!   (`LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY`, taken only by a flight's
//!   read stage). A read permit is taken inside a catch-up slot (a catch-up's
//!   own, or its flight's) and nothing waits for a slot while holding one.
//! - [`CapacityBudget`] is one subscription's single capacity-wait budget,
//!   `LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS`: the admission wait, the cluster
//!   stream-reader wait, every catch-up slot wait and every read-permit wait
//!   (atomic or fenced class) draw on it, and so does every capacity wait of a
//!   shared fenced-read flight a catch-up is attached to (its slot, its fenced
//!   read permit, its memory reservation; `storage::snapshot_flights`), so a
//!   subscribe waits for capacity at most that long in total (a re-bootstrap
//!   onto a new object starts a fresh one).

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;

use crate::observability::metrics::{
    SharedMetrics, ADMISSION_RESULT_ADMITTED, ADMISSION_RESULT_CANCELLED,
    ADMISSION_RESULT_REFUSED_SATURATED, ADMISSION_RESULT_REFUSED_WAITER_CAP,
    CATCHUP_WAIT_RESULT_ACQUIRED, CATCHUP_WAIT_RESULT_CANCELLED, CATCHUP_WAIT_RESULT_TIMED_OUT,
};
use crate::storage::client::POOL_LABEL_SUBSCRIBER;
use crate::storage::diff_read::SnapshotConsumer;

/// One subscription's capacity-wait budget: how much longer it may spend
/// waiting for capacity, across every wait that draws on it.
///
/// Its full size is `LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS`. The handler
/// creates it after admission and charges it the admission wait, so it starts
/// at the configured wait minus the time admission took. After that it is
/// charged only for time spent waiting for capacity: the cluster stream-reader
/// wait (`StreamMux::consume`), each catch-up slot wait
/// ([`CatchUpGate::acquire`]) and each read-permit wait
/// ([`CatchUpReadGate::acquire`]), each given [`Self::deadline`] as its
/// deadline. A catch-up attached to a shared fenced-read flight
/// (`storage::snapshot_flights`) is charged instead for the time it is
/// attached while the flight waits for capacity — its slot, its fenced read
/// permit or its memory reservation (`storage::memory_budget`) — since the
/// flight's own waits carry no deadline and serve every attached catch-up.
/// Reads, builds and sends are never charged, nor are a flight's fence
/// sleeps, its wait for a predecessor flight or a wait for a build turn. A
/// re-bootstrap onto a new object
/// [`Self::refill`]s it, so every catch-up entered that way gets the full
/// budget again.
///
/// A zero budget means "never wait for capacity": each wait takes a free
/// reader, slot or permit if there is one and is refused otherwise.
///
/// Interior-mutable (an atomic), so the subscription's shared context can hand
/// it out by `&` and stay `Sync`; one subscription never charges it from two
/// tasks at once.
#[derive(Debug)]
pub struct CapacityBudget {
    total: Duration,
    /// Remaining nanoseconds, saturating at zero.
    remaining_nanos: AtomicU64,
}

impl CapacityBudget {
    /// A full budget of `total`.
    pub fn new(total: Duration) -> Self {
        Self {
            total,
            remaining_nanos: AtomicU64::new(duration_nanos(total)),
        }
    }

    /// The full size this budget was created with, and refills to.
    pub fn total(&self) -> Duration {
        self.total
    }

    /// What is left to wait.
    pub fn remaining(&self) -> Duration {
        Duration::from_nanos(self.remaining_nanos.load(Ordering::Acquire))
    }

    /// Spend `waited` of the budget, saturating at zero.
    pub fn charge(&self, waited: Duration) {
        let waited = duration_nanos(waited);
        let _ = self
            .remaining_nanos
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |left| {
                Some(left.saturating_sub(waited))
            });
    }

    /// Restore the full budget: a catch-up entered through a re-bootstrap
    /// waits as long as the first one could.
    pub fn refill(&self) {
        self.remaining_nanos
            .store(duration_nanos(self.total), Ordering::Release);
    }

    /// The deadline for a wait starting now: now plus what is left. A budget
    /// too large to add to the clock yields a deadline decades away rather
    /// than panicking.
    pub fn deadline(&self) -> tokio::time::Instant {
        let now = tokio::time::Instant::now();
        now.checked_add(self.remaining())
            .unwrap_or_else(|| now + FAR_FUTURE)
    }
}

/// Stand-in deadline for a budget whose end the clock cannot represent (about
/// 30 years, as tokio's own `sleep` uses).
const FAR_FUTURE: Duration = Duration::from_secs(86_400 * 365 * 30);

/// A deadline no wait reaches: [`FAR_FUTURE`] from now. For a deadline-taking
/// wait ([`CatchUpGate::acquire`], [`CatchUpReadGate::acquire`]) whose caller
/// has no deadline of its own — a shared fenced-read flight, whose capacity
/// waits end only when its last attached catch-up leaves
/// (`storage::snapshot_flights`). Such a wait can therefore end only acquired,
/// closed or dropped, never timed out.
pub(crate) fn no_deadline() -> tokio::time::Instant {
    tokio::time::Instant::now() + FAR_FUTURE
}

/// `d` in nanoseconds, saturating at `u64::MAX` (about 584 years).
fn duration_nanos(d: Duration) -> u64 {
    u64::try_from(d.as_nanos()).unwrap_or(u64::MAX)
}

/// One admitted subscription; the slot returns to the gate on drop.
///
/// Counted on `live_edit_subscriber_admitted` from creation to drop, so the
/// gauge follows the RPC on every exit path, a task abort included.
pub struct SubscriberAdmission {
    _permit: OwnedSemaphorePermit,
    metrics: Option<SharedMetrics>,
}

impl SubscriberAdmission {
    fn new(permit: OwnedSemaphorePermit, metrics: Option<SharedMetrics>) -> Self {
        if let Some(m) = &metrics {
            m.add_subscriber_admitted(1);
        }
        Self {
            _permit: permit,
            metrics,
        }
    }
}

impl Drop for SubscriberAdmission {
    fn drop(&mut self) {
        if let Some(m) = &self.metrics {
            m.add_subscriber_admitted(-1);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmissionRefusal {
    /// Every slot stayed taken for the whole bounded wait (or the wait is 0).
    Saturated,
    /// The wait queue is at `LIVE_EDIT_SUBSCRIBER_ADMISSION_MAX_WAITERS`.
    WaiterCap,
    /// The session was cancelled (deleted, or missing from the live store).
    Cancelled,
    /// The service is shutting down: the subscription's enrolled shutdown
    /// token fired (born cancelled once graceful shutdown's paced stream
    /// ending has begun, or cancelled in its batch of it), or
    /// `ValkeyClient::quit` closed the gate. The client retries elsewhere.
    ShuttingDown,
}

impl AdmissionRefusal {
    /// `result` label on `live_edit_subscriber_admission_total`.
    pub fn metric_result(self) -> &'static str {
        match self {
            AdmissionRefusal::Saturated => ADMISSION_RESULT_REFUSED_SATURATED,
            AdmissionRefusal::WaiterCap => ADMISSION_RESULT_REFUSED_WAITER_CAP,
            AdmissionRefusal::ShuttingDown | AdmissionRefusal::Cancelled => {
                ADMISSION_RESULT_CANCELLED
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AdmissionPolicy {
    pub wait: Duration,
    pub max_waiters: usize,
}

pub struct SubscriberAdmissionGate {
    permits: Arc<Semaphore>,
    capacity: Option<usize>,
    policy: AdmissionPolicy,
    waiters: AtomicUsize,
    metrics: Arc<StdMutex<Option<SharedMetrics>>>,
}

impl std::fmt::Debug for SubscriberAdmissionGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubscriberAdmissionGate")
            .field("capacity", &self.capacity)
            .field("available_permits", &self.available_permits())
            .field("waiters", &self.waiters())
            .field("policy", &self.policy)
            .finish()
    }
}

impl SubscriberAdmissionGate {
    /// `capacity: None` admits without limit (closing and cancellation still
    /// refuse).
    pub fn new(
        capacity: Option<usize>,
        policy: AdmissionPolicy,
        metrics: Arc<StdMutex<Option<SharedMetrics>>>,
    ) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(capacity.unwrap_or(Semaphore::MAX_PERMITS))),
            capacity,
            policy,
            waiters: AtomicUsize::new(0),
            metrics,
        }
    }

    pub fn capacity(&self) -> Option<usize> {
        self.capacity
    }

    pub fn policy(&self) -> AdmissionPolicy {
        self.policy
    }

    pub fn available_permits(&self) -> usize {
        self.permits.available_permits()
    }

    pub fn waiters(&self) -> usize {
        self.waiters.load(Ordering::Acquire)
    }

    /// Refuses every future admission and releases parked waiters as `Cancelled`.
    pub fn close(&self) {
        self.permits.close();
    }

    pub fn is_closed(&self) -> bool {
        self.permits.is_closed()
    }

    /// Admits one subscription: immediately when a slot is free, otherwise
    /// after a wait bounded by the policy and raced against `cancel`.
    pub async fn admit(
        &self,
        cancel: Option<&CancellationToken>,
    ) -> Result<SubscriberAdmission, AdmissionRefusal> {
        self.admit_until_shutdown(None, cancel).await
    }

    /// [`Self::admit`], also racing `shutdown`. Shutdown is checked before
    /// `cancel` on entry and while queued, so a request that sees both is
    /// refused [`AdmissionRefusal::ShuttingDown`], the refusal a client
    /// retries.
    pub async fn admit_until_shutdown(
        &self,
        shutdown: Option<&CancellationToken>,
        cancel: Option<&CancellationToken>,
    ) -> Result<SubscriberAdmission, AdmissionRefusal> {
        let started = Instant::now();
        let refuse = |refusal: AdmissionRefusal| {
            self.observe(refusal.metric_result(), started.elapsed());
            Err(refusal)
        };

        if self.is_closed() || shutdown.is_some_and(CancellationToken::is_cancelled) {
            return refuse(AdmissionRefusal::ShuttingDown);
        }
        if cancel.is_some_and(CancellationToken::is_cancelled) {
            return refuse(AdmissionRefusal::Cancelled);
        }
        match self.permits.clone().try_acquire_owned() {
            Ok(permit) => {
                self.observe(ADMISSION_RESULT_ADMITTED, started.elapsed());
                return Ok(SubscriberAdmission::new(permit, self.metrics()));
            }
            Err(tokio::sync::TryAcquireError::Closed) => {
                return refuse(AdmissionRefusal::ShuttingDown);
            }
            Err(tokio::sync::TryAcquireError::NoPermits) => {}
        }
        if self.policy.wait.is_zero() {
            return refuse(AdmissionRefusal::Saturated);
        }
        let Some(_waiter) = WaiterSlot::join(self) else {
            return refuse(AdmissionRefusal::WaiterCap);
        };

        let outcome = tokio::select! {
            biased;
            () = cancelled_or_never(shutdown) => Err(AdmissionRefusal::ShuttingDown),
            () = cancelled_or_never(cancel) => Err(AdmissionRefusal::Cancelled),
            permit = self.permits.clone().acquire_owned() => {
                permit.map_err(|_closed| AdmissionRefusal::ShuttingDown)
            }
            () = tokio::time::sleep(self.policy.wait) => Err(AdmissionRefusal::Saturated),
        };
        match outcome {
            Ok(permit) => {
                self.observe(ADMISSION_RESULT_ADMITTED, started.elapsed());
                Ok(SubscriberAdmission::new(permit, self.metrics()))
            }
            Err(refusal) => refuse(refusal),
        }
    }

    fn metrics(&self) -> Option<SharedMetrics> {
        self.metrics.lock().ok().and_then(|g| g.clone())
    }

    fn observe(&self, result: &'static str, elapsed: Duration) {
        if let Some(m) = self.metrics() {
            m.observe_subscriber_admission(result, elapsed);
        }
    }
}

/// Resolves when `token` is cancelled; never, when there is no token. Lets one
/// `biased` `select!` race an optional token.
pub(crate) async fn cancelled_or_never(token: Option<&CancellationToken>) {
    match token {
        Some(token) => token.cancelled().await,
        None => std::future::pending().await,
    }
}

struct WaiterSlot<'a> {
    gate: &'a SubscriberAdmissionGate,
}

impl<'a> WaiterSlot<'a> {
    fn join(gate: &'a SubscriberAdmissionGate) -> Option<Self> {
        let joined = gate.waiters.fetch_add(1, Ordering::AcqRel) + 1;
        if joined > gate.policy.max_waiters {
            gate.waiters.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        if let Some(m) = gate.metrics() {
            m.inc_valkey_pool_waiters(POOL_LABEL_SUBSCRIBER);
        }
        Some(Self { gate })
    }
}

impl Drop for WaiterSlot<'_> {
    fn drop(&mut self) {
        self.gate.waiters.fetch_sub(1, Ordering::AcqRel);
        if let Some(m) = self.gate.metrics() {
            m.dec_valkey_pool_waiters(POOL_LABEL_SUBSCRIBER);
        }
    }
}

/// Bounds concurrent catch-ups; see the module docs.
pub struct CatchUpGate {
    permits: Arc<Semaphore>,
    limit: Option<usize>,
    metrics: Arc<StdMutex<Option<SharedMetrics>>>,
}

/// Why [`CatchUpGate::acquire`] returned without a slot, a
/// [`CatchUpReadGate`] acquisition without a read permit, or a
/// `MemoryBudget` reservation (`storage::memory_budget`) without memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatchUpRefusal {
    /// The gate or budget is closed: `ValkeyClient::quit` ran at shutdown.
    Closed,
    /// No slot, permit or memory freed before the deadline (or the deadline
    /// had already passed and none was free).
    TimedOut,
}

/// One catch-up's slot; it returns to the gate on drop.
pub struct CatchUpPermit {
    _permit: OwnedSemaphorePermit,
    metrics: Option<SharedMetrics>,
}

impl Drop for CatchUpPermit {
    fn drop(&mut self) {
        if let Some(m) = &self.metrics {
            m.add_subscribe_catchups_inflight(-1);
        }
    }
}

impl std::fmt::Debug for CatchUpGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatchUpGate")
            .field("limit", &self.limit)
            .field("available", &self.available())
            .finish()
    }
}

impl CatchUpGate {
    /// `limit: None` never waits.
    pub fn new(limit: Option<usize>, metrics: Arc<StdMutex<Option<SharedMetrics>>>) -> Self {
        Self {
            permits: Arc::new(Semaphore::new(limit.unwrap_or(Semaphore::MAX_PERMITS))),
            limit,
            metrics,
        }
    }

    pub fn limit(&self) -> Option<usize> {
        self.limit
    }

    pub fn available(&self) -> usize {
        self.permits.available_permits()
    }

    /// Takes a catch-up slot: immediately when one is free, otherwise after a
    /// FIFO wait that ends at `deadline` — the subscription's
    /// [`CapacityBudget::deadline`] for a catch-up of its own, or
    /// [`no_deadline`] for a shared fenced-read flight, whose attached
    /// catch-ups are charged for the wait one by one. A deadline already
    /// passed never joins the queue. Callers race this against their own
    /// cancellation.
    ///
    /// Refuses [`CatchUpRefusal::Closed`] once [`Self::close`] ran, and
    /// [`CatchUpRefusal::TimedOut`] when no slot freed in time.
    ///
    /// Records one `live_edit_subscribe_catchup_wait_seconds` observation per
    /// call, whatever the outcome: `acquired`; `timed_out`; or `cancelled` when
    /// the gate is closed or the caller drops the wait.
    pub async fn acquire(
        &self,
        deadline: tokio::time::Instant,
    ) -> Result<CatchUpPermit, CatchUpRefusal> {
        let metrics = self.metrics.lock().ok().and_then(|g| g.clone());
        let mut wait = CatchUpWaitObservation::start(metrics.clone());
        let permit = match self.permits.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(tokio::sync::TryAcquireError::Closed) => {
                wait.finish(CATCHUP_WAIT_RESULT_CANCELLED);
                return Err(CatchUpRefusal::Closed);
            }
            Err(tokio::sync::TryAcquireError::NoPermits) => {
                if deadline <= tokio::time::Instant::now() {
                    wait.finish(CATCHUP_WAIT_RESULT_TIMED_OUT);
                    return Err(CatchUpRefusal::TimedOut);
                }
                let _waiting = CatchUpWaiter::join(metrics.clone());
                match tokio::time::timeout_at(deadline, self.permits.clone().acquire_owned()).await
                {
                    Ok(Ok(p)) => p,
                    Ok(Err(_closed)) => {
                        wait.finish(CATCHUP_WAIT_RESULT_CANCELLED);
                        return Err(CatchUpRefusal::Closed);
                    }
                    Err(_elapsed) => {
                        wait.finish(CATCHUP_WAIT_RESULT_TIMED_OUT);
                        return Err(CatchUpRefusal::TimedOut);
                    }
                }
            }
        };
        wait.finish(CATCHUP_WAIT_RESULT_ACQUIRED);
        if let Some(m) = &metrics {
            m.add_subscribe_catchups_inflight(1);
        }
        Ok(CatchUpPermit {
            _permit: permit,
            metrics,
        })
    }

    /// Takes a free catch-up slot at once, or returns `None`: never waits and
    /// never joins the queue, so it cannot jump ahead of a waiting catch-up
    /// (a free slot with waiters queued is handed to them first). `None` also
    /// once [`Self::close`] ran.
    ///
    /// For a shared fenced-read flight's extra build (`storage::snapshot_flights`),
    /// which builds only when a slot is free at that moment. Records no
    /// `live_edit_subscribe_catchup_wait_seconds` observation, since nothing
    /// waited; the slot counts on `live_edit_subscribe_catchups_inflight` while
    /// held, like any other.
    pub fn try_acquire(&self) -> Option<CatchUpPermit> {
        let permit = self.permits.clone().try_acquire_owned().ok()?;
        let metrics = self.metrics.lock().ok().and_then(|g| g.clone());
        if let Some(m) = &metrics {
            m.add_subscribe_catchups_inflight(1);
        }
        Some(CatchUpPermit {
            _permit: permit,
            metrics,
        })
    }

    pub fn close(&self) {
        self.permits.close();
    }
}

/// One [`CatchUpGate::acquire`] call's `live_edit_subscribe_catchup_wait_seconds`
/// observation: the result the call finished with (`acquired`, `timed_out`, or
/// `cancelled` for a closed gate), or `cancelled` when the caller dropped the
/// wait before it finished.
struct CatchUpWaitObservation {
    started: Instant,
    metrics: Option<SharedMetrics>,
    recorded: bool,
}

impl CatchUpWaitObservation {
    fn start(metrics: Option<SharedMetrics>) -> Self {
        Self {
            started: Instant::now(),
            metrics,
            recorded: false,
        }
    }

    fn finish(&mut self, result: &'static str) {
        if self.recorded {
            return;
        }
        self.recorded = true;
        if let Some(m) = &self.metrics {
            m.observe_subscribe_catchup_wait(result, self.started.elapsed());
        }
    }
}

impl Drop for CatchUpWaitObservation {
    fn drop(&mut self) {
        self.finish(CATCHUP_WAIT_RESULT_CANCELLED);
    }
}

/// Bounds the catch-up snapshot reads of one permit class running at once per
/// replica (Spec SDD SR-7.1). `ValkeyClient` builds one gate per class:
///
/// - **atomic**, `LIVE_EDIT_CATCHUP_SNAPSHOT_READ_CONCURRENCY` (`P_a`), held
///   across each catch-up atomic read
///   (`storage::diff_read::read_object_snapshot`'s `EVALSHA`) and nothing
///   else;
/// - **fenced**, `LIVE_EDIT_CATCHUP_FENCED_READ_CONCURRENCY` (`P_f`), for the
///   fenced read of a layer above the atomic-read ceilings, so a multi-second
///   fenced read never holds a permit the atomic reads need. Taken by a shared
///   fenced-read flight's read stage (`storage::snapshot_flights`) per fence
///   acquire attempt, with no deadline, and held from before the acquire
///   through the fan-out, the fence's release and the memory adjust after it;
///   released during every `RETRY_AFTER` sleep. The structural cache's
///   fenced cold start (`storage::structural_cache_fenced`) takes one the
///   same way, outside any catch-up slot: after the cache's one-at-a-time
///   and plain cold-start permits, with a deadline at its waiting cap; and
///   so does the cache's sampled audit, with a deadline already passed, so
///   it takes a free permit or none and never queues.
///
/// **Nesting with the catch-up slot.** On the catch-up path a read permit is
/// taken only while holding a [`CatchUpGate`] slot (a flight's own slot, for
/// the fenced class), and nothing waits for a slot while holding a permit, so
/// the two never form a wait cycle. The slot stays the CPU bound; the permits
/// narrow the shard-blocking snapshot reads inside it. The steady state never
/// reads layer state, so it takes no permit. The cold start holds no catch-up
/// slot and waits for no slot, and no catch-up waits for the cache's
/// cold-start permits, so its fenced permit adds no wait cycle either; the
/// one-at-a-time permit it takes first keeps at most one cold start per
/// replica waiting for, or holding, a fenced permit.
///
/// FIFO, like the slot: a released permit goes to the longest waiter.
pub struct CatchUpReadGate {
    permits: Arc<Semaphore>,
    limit: usize,
    /// The `class` label of `live_edit_catchup_snapshot_reads_in_flight` this
    /// gate's permits count under.
    class: &'static str,
    metrics: Arc<StdMutex<Option<SharedMetrics>>>,
}

/// One snapshot read's permit; it returns to its [`CatchUpReadGate`] on drop,
/// on every exit: completion, error, cancellation of the future holding it,
/// or a task abort.
pub struct CatchUpReadPermit {
    _permit: OwnedSemaphorePermit,
    class: &'static str,
    metrics: Option<SharedMetrics>,
}

impl Drop for CatchUpReadPermit {
    fn drop(&mut self) {
        if let Some(m) = &self.metrics {
            m.add_catchup_snapshot_reads_in_flight(self.class, -1);
        }
    }
}

impl std::fmt::Debug for CatchUpReadGate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatchUpReadGate")
            .field("class", &self.class)
            .field("limit", &self.limit)
            .field("available", &self.available())
            .finish()
    }
}

impl CatchUpReadGate {
    /// A gate of `limit` permits counted under `class` (a
    /// `metrics::CATCHUP_SNAPSHOT_READ_CLASSES` value). `limit` is at least 1:
    /// configuration rejects 0 (Spec SDD SR-7.1), and a gate built with 0
    /// anyway is raised to 1, because a gate with no permit would refuse every
    /// read rather than bound them.
    pub fn new(
        limit: usize,
        class: &'static str,
        metrics: Arc<StdMutex<Option<SharedMetrics>>>,
    ) -> Self {
        let limit = limit.max(1);
        Self {
            permits: Arc::new(Semaphore::new(limit)),
            limit,
            class,
            metrics,
        }
    }

    pub fn limit(&self) -> usize {
        self.limit
    }

    pub fn available(&self) -> usize {
        self.permits.available_permits()
    }

    /// Takes a read permit for a `consumer` read: at once when one is free,
    /// otherwise after a FIFO wait that ends at `deadline` — the
    /// subscription's [`CapacityBudget::deadline`] for an atomic read, or
    /// [`no_deadline`] for a shared fenced-read flight, whose attached
    /// catch-ups are charged for the wait one by one. A deadline already
    /// passed never joins the queue, so a zero budget takes a free permit or is
    /// refused at once. The caller races this against its own endings.
    ///
    /// Refuses [`CatchUpRefusal::Closed`] once [`Self::close`] ran, and
    /// [`CatchUpRefusal::TimedOut`] when no permit freed in time.
    ///
    /// Records one `live_edit_catchup_snapshot_read_wait_seconds{consumer,
    /// class}` observation per call, whatever the outcome, a dropped wait
    /// included, under this gate's class: a fenced wait never mixes with the
    /// atomic ones.
    pub async fn acquire(
        &self,
        deadline: tokio::time::Instant,
        consumer: impl SnapshotConsumer,
    ) -> Result<CatchUpReadPermit, CatchUpRefusal> {
        let metrics = self.metrics();
        let _wait = ReadPermitWaitObservation::start(consumer.label(), self.class, metrics.clone());
        let permit = match self.permits.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(tokio::sync::TryAcquireError::Closed) => return Err(CatchUpRefusal::Closed),
            Err(tokio::sync::TryAcquireError::NoPermits) => {
                if deadline <= tokio::time::Instant::now() {
                    return Err(CatchUpRefusal::TimedOut);
                }
                match tokio::time::timeout_at(deadline, self.permits.clone().acquire_owned()).await
                {
                    Ok(Ok(p)) => p,
                    Ok(Err(_closed)) => return Err(CatchUpRefusal::Closed),
                    Err(_elapsed) => return Err(CatchUpRefusal::TimedOut),
                }
            }
        };
        Ok(self.granted(permit, metrics))
    }

    /// Refuses every later acquisition and releases parked waiters as
    /// [`CatchUpRefusal::Closed`]. Permits already held stay valid until
    /// dropped. `ValkeyClient::quit` calls it at shutdown.
    pub fn close(&self) {
        self.permits.close();
    }

    fn granted(
        &self,
        permit: OwnedSemaphorePermit,
        metrics: Option<SharedMetrics>,
    ) -> CatchUpReadPermit {
        if let Some(m) = &metrics {
            m.add_catchup_snapshot_reads_in_flight(self.class, 1);
        }
        CatchUpReadPermit {
            _permit: permit,
            class: self.class,
            metrics,
        }
    }

    fn metrics(&self) -> Option<SharedMetrics> {
        self.metrics.lock().ok().and_then(|g| g.clone())
    }
}

/// One [`CatchUpReadGate`] acquisition's
/// `live_edit_catchup_snapshot_read_wait_seconds{consumer, class}`
/// observation, recorded when the acquisition ends — returned, refused, or
/// dropped by its caller.
struct ReadPermitWaitObservation {
    started: Instant,
    /// The consumer's metric label.
    consumer: &'static str,
    class: &'static str,
    metrics: Option<SharedMetrics>,
}

impl ReadPermitWaitObservation {
    fn start(consumer: &'static str, class: &'static str, metrics: Option<SharedMetrics>) -> Self {
        Self {
            started: Instant::now(),
            consumer,
            class,
            metrics,
        }
    }
}

impl Drop for ReadPermitWaitObservation {
    fn drop(&mut self) {
        if let Some(m) = &self.metrics {
            m.observe_catchup_snapshot_read_wait(self.consumer, self.class, self.started.elapsed());
        }
    }
}

/// Counts a parked catch-up on `live_edit_subscribe_catchups_waiting`, including
/// when the wait is dropped by a cancelled caller.
struct CatchUpWaiter(Option<SharedMetrics>);

impl CatchUpWaiter {
    fn join(metrics: Option<SharedMetrics>) -> Self {
        if let Some(m) = &metrics {
            m.add_subscribe_catchups_waiting(1);
        }
        Self(metrics)
    }
}

impl Drop for CatchUpWaiter {
    fn drop(&mut self) {
        if let Some(m) = &self.0 {
            m.add_subscribe_catchups_waiting(-1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::metrics::test_util::{render, sample_value};
    use crate::storage::diff_read::SnapshotReadConsumer;
    use std::future::Future;
    use std::task::Poll;

    const SETTLE: Duration = Duration::from_secs(5);

    /// A catch-up slot deadline no test here reaches: these tests exercise
    /// ordering, closing and dropping, not the deadline.
    fn no_deadline() -> tokio::time::Instant {
        tokio::time::Instant::now() + Duration::from_secs(3_600)
    }

    fn gate(capacity: usize, wait_ms: u64, max_waiters: usize) -> SubscriberAdmissionGate {
        SubscriberAdmissionGate::new(
            Some(capacity),
            AdmissionPolicy {
                wait: Duration::from_millis(wait_ms),
                max_waiters,
            },
            Arc::new(StdMutex::new(None)),
        )
    }

    fn gate_with_metrics(
        capacity: usize,
        wait_ms: u64,
        max_waiters: usize,
    ) -> (SubscriberAdmissionGate, SharedMetrics) {
        let metrics: SharedMetrics =
            Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let g = SubscriberAdmissionGate::new(
            Some(capacity),
            AdmissionPolicy {
                wait: Duration::from_millis(wait_ms),
                max_waiters,
            },
            Arc::new(StdMutex::new(Some(metrics.clone()))),
        );
        (g, metrics)
    }

    fn read_admission(metrics: &SharedMetrics, result: &str) -> u64 {
        sample_value(
            &render(metrics),
            "live_edit_subscriber_admission_total",
            &[("result", result)],
        )
        .expect("pre-initialised") as u64
    }

    fn read_waiters(metrics: &SharedMetrics) -> i64 {
        sample_value(
            &render(metrics),
            "live_edit_valkey_pool_waiters",
            &[("pool", POOL_LABEL_SUBSCRIBER)],
        )
        .expect("pre-initialised") as i64
    }

    fn refusal<T>(r: Result<T, AdmissionRefusal>) -> AdmissionRefusal {
        match r {
            Err(r) => r,
            Ok(_) => panic!("expected a refusal"),
        }
    }

    async fn poll_once<F: Future>(fut: std::pin::Pin<&mut F>) -> Poll<F::Output> {
        let mut fut = Some(fut);
        std::future::poll_fn(move |cx| Poll::Ready(fut.take().expect("once").poll(cx))).await
    }

    #[tokio::test]
    async fn capacity_is_independent_of_connections_and_admits_up_to_it() {
        let g = gate(3, 0, 1);
        let held: Vec<_> = futures::future::join_all((0..3).map(|_| g.admit(None)))
            .await
            .into_iter()
            .map(|r| r.expect("within capacity"))
            .collect();
        assert_eq!(refusal(g.admit(None).await), AdmissionRefusal::Saturated);
        drop(held);
        assert_eq!(g.available_permits(), 3, "permits return on drop");
    }

    #[tokio::test]
    async fn a_zero_wait_refuses_on_the_first_poll_without_queueing() {
        let g = gate(1, 0, 1);
        let _held = g.admit(None).await.unwrap();
        let mut second = std::pin::pin!(g.admit(None));
        match poll_once(second.as_mut()).await {
            Poll::Ready(r) => assert_eq!(refusal(r), AdmissionRefusal::Saturated),
            Poll::Pending => panic!("a zero wait must not queue"),
        }
        assert_eq!(g.waiters(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn at_the_waiter_cap_the_refusal_is_waiter_cap() {
        let g = Arc::new(gate(1, 60_000, 1));
        let _held = g.admit(None).await.unwrap();
        let queued = {
            let g = g.clone();
            tokio::spawn(async move { g.admit(None).await.map(|_| ()) })
        };
        let deadline = Instant::now() + SETTLE;
        while g.waiters() < 1 {
            assert!(Instant::now() < deadline, "the first waiter never queued");
            tokio::task::yield_now().await;
        }
        let mut third = std::pin::pin!(g.admit(None));
        match poll_once(third.as_mut()).await {
            Poll::Ready(r) => assert_eq!(refusal(r), AdmissionRefusal::WaiterCap),
            Poll::Pending => panic!("joined a queue already at its cap"),
        }
        queued.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_wait_is_bounded_and_ends_saturated() {
        let g = gate(1, 50, 4);
        let _held = g.admit(None).await.unwrap();
        let r = tokio::time::timeout(SETTLE, g.admit(None))
            .await
            .expect("bounded wait");
        assert_eq!(refusal(r), AdmissionRefusal::Saturated);
        assert_eq!(g.waiters(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancel_mid_wait_refuses_cancelled_not_saturated() {
        let (g, metrics) = gate_with_metrics(1, 60_000, 4);
        let _held = g.admit(None).await.unwrap();
        let cancel = CancellationToken::new();
        let mut waiting = std::pin::pin!(g.admit(Some(&cancel)));
        assert!(poll_once(waiting.as_mut()).await.is_pending());
        cancel.cancel();
        let r = tokio::time::timeout(SETTLE, waiting)
            .await
            .expect("cancel wakes");
        assert_eq!(refusal(r), AdmissionRefusal::Cancelled);
        assert_eq!(read_admission(&metrics, ADMISSION_RESULT_CANCELLED), 1);
        assert_eq!(
            read_admission(&metrics, ADMISSION_RESULT_REFUSED_SATURATED),
            0
        );
    }

    #[tokio::test]
    async fn an_already_cancelled_token_refuses_on_entry_even_with_free_slots() {
        let (g, metrics) = gate_with_metrics(2, 60_000, 4);
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert_eq!(
            refusal(g.admit(Some(&cancel)).await),
            AdmissionRefusal::Cancelled
        );
        assert_eq!(read_admission(&metrics, ADMISSION_RESULT_ADMITTED), 0);
        assert_eq!(g.available_permits(), 2);
        assert_eq!(g.waiters(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn close_releases_a_parked_waiter_as_shutting_down() {
        let (g, metrics) = gate_with_metrics(1, 60_000, 4);
        let _held = g.admit(None).await.unwrap();
        let mut parked = std::pin::pin!(g.admit(None));
        assert!(poll_once(parked.as_mut()).await.is_pending());
        g.close();
        let r = tokio::time::timeout(SETTLE, parked)
            .await
            .expect("close wakes");
        assert_eq!(refusal(r), AdmissionRefusal::ShuttingDown);
        assert_eq!(
            read_admission(&metrics, ADMISSION_RESULT_REFUSED_SATURATED),
            0
        );
        assert_eq!(refusal(g.admit(None).await), AdmissionRefusal::ShuttingDown);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_waiter_gauge_rises_while_parked_and_falls_after() {
        let (g, metrics) = gate_with_metrics(1, 60_000, 4);
        let held = g.admit(None).await.unwrap();
        assert_eq!(read_waiters(&metrics), 0, "the fast path does not queue");
        let mut parked = std::pin::pin!(g.admit(None));
        assert!(poll_once(parked.as_mut()).await.is_pending());
        assert_eq!(read_waiters(&metrics), 1);
        drop(held);
        let admitted = tokio::time::timeout(SETTLE, parked)
            .await
            .expect("admitted");
        assert!(admitted.is_ok());
        assert_eq!(read_waiters(&metrics), 0);
    }

    #[tokio::test]
    async fn every_decision_records_its_wait() {
        const NAME: &str = "live_edit_subscriber_admission_wait_seconds_count";
        let (g, metrics) = gate_with_metrics(1, 0, 1);
        assert_eq!(sample_value(&render(&metrics), NAME, &[]), None);
        let _held = g.admit(None).await.unwrap();
        assert_eq!(sample_value(&render(&metrics), NAME, &[]), Some(1.0));
        let _ = g.admit(None).await;
        assert_eq!(sample_value(&render(&metrics), NAME, &[]), Some(2.0));
    }

    /// Rounds for the queued both-tokens case. Its outcome under an unbiased
    /// `select!` would be a coin flip per round, so one round would let a
    /// dropped `biased;` pass half the time; 64 rounds miss it with
    /// probability 2^-64.
    const BIASED_ROUNDS: usize = 64;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_wins_over_session_cancel_on_entry_and_while_queued() {
        let g = gate(1, 60_000, 4);
        let _held = g.admit(None).await.unwrap();

        for round in 0..BIASED_ROUNDS {
            let shutdown = CancellationToken::new();
            let session = CancellationToken::new();
            let mut parked =
                std::pin::pin!(g.admit_until_shutdown(Some(&shutdown), Some(&session)));
            assert!(poll_once(parked.as_mut()).await.is_pending());
            // Both fire before the parked wait is polled again, so both of its
            // arms are ready at once and only the `select!`'s order decides.
            shutdown.cancel();
            session.cancel();
            let r = tokio::time::timeout(SETTLE, parked)
                .await
                .expect("shutdown wakes");
            assert_eq!(
                refusal(r),
                AdmissionRefusal::ShuttingDown,
                "round {round}: a queued request that sees both tokens is a shutdown"
            );
        }

        let shutdown = CancellationToken::new();
        let session = CancellationToken::new();
        shutdown.cancel();
        session.cancel();
        assert_eq!(
            refusal(
                g.admit_until_shutdown(Some(&shutdown), Some(&session))
                    .await
            ),
            AdmissionRefusal::ShuttingDown,
            "both already cancelled on entry: shutdown is reported"
        );
        assert_eq!(
            refusal(g.admit_until_shutdown(None, Some(&session)).await),
            AdmissionRefusal::Cancelled
        );
    }

    #[test]
    fn each_refusal_maps_to_its_own_metric_label() {
        let mut labels = [
            AdmissionRefusal::Saturated.metric_result(),
            AdmissionRefusal::WaiterCap.metric_result(),
            AdmissionRefusal::Cancelled.metric_result(),
        ];
        assert_eq!(labels[0], ADMISSION_RESULT_REFUSED_SATURATED);
        assert_eq!(labels[1], ADMISSION_RESULT_REFUSED_WAITER_CAP);
        assert_eq!(labels[2], ADMISSION_RESULT_CANCELLED);
        assert_eq!(
            AdmissionRefusal::ShuttingDown.metric_result(),
            ADMISSION_RESULT_CANCELLED,
            "a shutdown refusal records `cancelled`: the request went away, not the capacity"
        );
        labels.sort_unstable();
        assert!(labels.windows(2).all(|w| w[0] != w[1]));
    }

    #[tokio::test]
    async fn an_unlimited_gate_admits_without_bound_but_still_honours_close_and_cancel() {
        let g = SubscriberAdmissionGate::new(
            None,
            AdmissionPolicy {
                wait: Duration::ZERO,
                max_waiters: 1,
            },
            Arc::new(StdMutex::new(None)),
        );
        let held: Vec<_> = futures::future::join_all((0..10_000).map(|_| g.admit(None)))
            .await
            .into_iter()
            .map(|r| r.expect("no cap"))
            .collect();
        assert_eq!(held.len(), 10_000);
        let cancel = CancellationToken::new();
        cancel.cancel();
        assert_eq!(
            refusal(g.admit(Some(&cancel)).await),
            AdmissionRefusal::Cancelled
        );
        g.close();
        assert_eq!(refusal(g.admit(None).await), AdmissionRefusal::ShuttingDown);
    }

    fn read_gauge(metrics: &SharedMetrics, name: &str) -> i64 {
        sample_value(&render(metrics), name, &[]).expect("pre-initialised") as i64
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_catch_up_gate_queues_past_its_limit_in_arrival_order() {
        let metrics: SharedMetrics =
            Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let g = CatchUpGate::new(Some(2), Arc::new(StdMutex::new(Some(metrics.clone()))));
        let a = g.acquire(no_deadline()).await.unwrap();
        let b = g.acquire(no_deadline()).await.unwrap();
        assert_eq!(
            read_gauge(&metrics, "live_edit_subscribe_catchups_inflight"),
            2
        );

        // Three waiters park in this order.
        let mut first = std::pin::pin!(g.acquire(no_deadline()));
        let mut second = std::pin::pin!(g.acquire(no_deadline()));
        let mut third = std::pin::pin!(g.acquire(no_deadline()));
        for (name, w) in [
            ("first", poll_once(first.as_mut()).await.is_pending()),
            ("second", poll_once(second.as_mut()).await.is_pending()),
            ("third", poll_once(third.as_mut()).await.is_pending()),
        ] {
            assert!(w, "past the limit the {name} waiter waits");
        }
        assert_eq!(
            read_gauge(&metrics, "live_edit_subscribe_catchups_waiting"),
            3
        );

        // Each release admits the longest-waiting catch-up and no other: the
        // later waiters are polled first and must still be pending.
        drop(a);
        assert!(poll_once(third.as_mut()).await.is_pending());
        assert!(poll_once(second.as_mut()).await.is_pending());
        let c1 = tokio::time::timeout(SETTLE, first.as_mut())
            .await
            .expect("the first release admits the first waiter");
        assert!(c1.is_ok());

        drop(b);
        assert!(poll_once(third.as_mut()).await.is_pending());
        let c2 = tokio::time::timeout(SETTLE, second.as_mut())
            .await
            .expect("the second release admits the second waiter");
        assert!(c2.is_ok());

        drop(c1);
        let c3 = tokio::time::timeout(SETTLE, third.as_mut())
            .await
            .expect("the third release admits the third waiter");
        assert!(c3.is_ok());

        assert_eq!(
            read_gauge(&metrics, "live_edit_subscribe_catchups_waiting"),
            0
        );
        assert_eq!(
            read_gauge(&metrics, "live_edit_subscribe_catchups_inflight"),
            2
        );
        drop(c2);
        drop(c3);
        assert_eq!(
            read_gauge(&metrics, "live_edit_subscribe_catchups_inflight"),
            0
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_dropped_catch_up_wait_leaves_no_waiter_and_close_releases_the_rest() {
        let metrics: SharedMetrics =
            Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let g = CatchUpGate::new(Some(1), Arc::new(StdMutex::new(Some(metrics.clone()))));
        let _held = g.acquire(no_deadline()).await.unwrap();
        {
            let mut dropped = std::pin::pin!(g.acquire(no_deadline()));
            assert!(poll_once(dropped.as_mut()).await.is_pending());
        }
        assert_eq!(
            read_gauge(&metrics, "live_edit_subscribe_catchups_waiting"),
            0
        );
        let mut parked = std::pin::pin!(g.acquire(no_deadline()));
        assert!(poll_once(parked.as_mut()).await.is_pending());
        g.close();
        assert!(matches!(
            tokio::time::timeout(SETTLE, parked).await.unwrap(),
            Err(CatchUpRefusal::Closed)
        ));
        assert!(matches!(
            g.acquire(no_deadline()).await,
            Err(CatchUpRefusal::Closed)
        ));
    }

    #[tokio::test]
    async fn an_unlimited_catch_up_gate_never_waits() {
        let g = CatchUpGate::new(None, Arc::new(StdMutex::new(None)));
        let held: Vec<_> =
            futures::future::join_all((0..10_000).map(|_| g.acquire(no_deadline()))).await;
        assert!(held.iter().all(Result::is_ok));
        assert_eq!(g.limit(), None);
    }

    /// `try_acquire` (a shared fenced-read flight's extra build) takes a free
    /// slot at once, counted on the in-flight gauge while held and recording
    /// no wait observation; it never waits and never joins the queue, so with
    /// no slot free — or a slot freed while a catch-up is queued, which goes
    /// to that catch-up — it returns `None`, and `None` once the gate closed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_catch_up_gate_try_acquire_takes_only_a_free_slot_and_never_jumps_the_queue() {
        let metrics: SharedMetrics =
            Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let g = CatchUpGate::new(Some(1), Arc::new(StdMutex::new(Some(metrics.clone()))));
        let wait_count = || {
            sample_value(
                &render(&metrics),
                "live_edit_subscribe_catchup_wait_seconds_count",
                &[],
            )
        };

        let extra = g.try_acquire().expect("a free slot is taken at once");
        assert_eq!(g.available(), 0);
        assert_eq!(
            read_gauge(&metrics, "live_edit_subscribe_catchups_inflight"),
            1
        );
        assert_eq!(wait_count(), None, "nothing waited, so nothing is observed");
        assert!(g.try_acquire().is_none(), "no slot free: never waits");
        assert_eq!(
            read_gauge(&metrics, "live_edit_subscribe_catchups_waiting"),
            0,
            "and never joins the queue"
        );

        let mut queued = std::pin::pin!(g.acquire(no_deadline()));
        assert!(poll_once(queued.as_mut()).await.is_pending());
        drop(extra);
        assert!(
            g.try_acquire().is_none(),
            "a slot freed while a catch-up is queued goes to that catch-up"
        );
        let held = tokio::time::timeout(SETTLE, queued)
            .await
            .expect("the release admits the queued catch-up")
            .expect("a slot");
        drop(held);
        assert_eq!(
            read_gauge(&metrics, "live_edit_subscribe_catchups_inflight"),
            0
        );

        g.close();
        assert!(g.try_acquire().is_none(), "a closed gate grants nothing");
    }

    // ── `live_edit_subscriber_admitted` ──────────────────────────────────────

    const ADMITTED: &str = "live_edit_subscriber_admitted";

    /// The gauge exports at zero before the first admission, counts every
    /// admission — the fast path and a queued one — and falls back to zero
    /// when each admission is dropped. A refusal never counts.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_admitted_gauge_counts_each_admission_until_it_is_dropped() {
        let (g, metrics) = gate_with_metrics(1, 60_000, 4);
        assert_eq!(read_gauge(&metrics, ADMITTED), 0, "pre-registered at zero");

        let first = g.admit(None).await.expect("a free slot admits");
        assert_eq!(read_gauge(&metrics, ADMITTED), 1);

        let mut queued = std::pin::pin!(g.admit(None));
        assert!(poll_once(queued.as_mut()).await.is_pending());
        assert_eq!(
            read_gauge(&metrics, ADMITTED),
            1,
            "a queued request is not admitted yet"
        );
        drop(first);
        let second = tokio::time::timeout(SETTLE, queued)
            .await
            .expect("the release admits the queued request")
            .expect("admitted");
        assert_eq!(read_gauge(&metrics, ADMITTED), 1, "one out, one in");

        let cancel = CancellationToken::new();
        cancel.cancel();
        assert_eq!(
            refusal(g.admit(Some(&cancel)).await),
            AdmissionRefusal::Cancelled
        );
        assert_eq!(read_gauge(&metrics, ADMITTED), 1, "a refusal never counts");

        drop(second);
        assert_eq!(read_gauge(&metrics, ADMITTED), 0);
    }

    /// An admission held by a task that is aborted — the one exit path that
    /// runs no code of the subscription's own — still leaves the gauge.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_admitted_gauge_falls_when_the_holding_task_is_aborted() {
        let (g, metrics) = gate_with_metrics(1, 60_000, 4);
        let g = Arc::new(g);
        let holder = {
            let g = Arc::clone(&g);
            tokio::spawn(async move {
                let _admission = g.admit(None).await.expect("a free slot admits");
                std::future::pending::<()>().await;
            })
        };
        let deadline = Instant::now() + SETTLE;
        while read_gauge(&metrics, ADMITTED) != 1 {
            assert!(
                Instant::now() < deadline,
                "the holder never took its admission"
            );
            tokio::task::yield_now().await;
        }
        holder.abort();
        let _ = holder.await;
        assert_eq!(read_gauge(&metrics, ADMITTED), 0);
        assert_eq!(g.available_permits(), 1, "and the slot returned");
    }

    // ── `live_edit_subscribe_catchup_wait_seconds{result}` ───────────────────

    fn catchup_wait_count(metrics: &SharedMetrics, result: &str) -> Option<u64> {
        sample_value(
            &render(metrics),
            "live_edit_subscribe_catchup_wait_seconds_count",
            &[("result", result)],
        )
        .map(|v| v as u64)
    }

    fn catch_up_gate_with_metrics(limit: usize) -> (CatchUpGate, SharedMetrics) {
        let metrics: SharedMetrics =
            Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let g = CatchUpGate::new(Some(limit), Arc::new(StdMutex::new(Some(metrics.clone()))));
        (g, metrics)
    }

    /// Both ways of getting a slot record `acquired`: the uncontended fast
    /// path and a wait that a release ends.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_catch_up_that_gets_its_slot_records_one_acquired_wait() {
        let (g, metrics) = catch_up_gate_with_metrics(1);
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_ACQUIRED),
            None,
            "a histogram series appears with its first observation"
        );
        let held = g.acquire(no_deadline()).await.expect("fast path");
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_ACQUIRED),
            Some(1)
        );

        let mut queued = std::pin::pin!(g.acquire(no_deadline()));
        assert!(poll_once(queued.as_mut()).await.is_pending());
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_ACQUIRED),
            Some(1),
            "a wait still in progress records nothing yet"
        );
        drop(held);
        let _second = tokio::time::timeout(SETTLE, queued)
            .await
            .expect("the release admits it")
            .expect("a slot");
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_ACQUIRED),
            Some(2)
        );
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_CANCELLED),
            None
        );
    }

    /// Every way of ending without a slot records exactly one `cancelled`
    /// wait, and never `acquired`: the gate already closed on entry, the gate
    /// closed under a parked wait, and a caller that drops its wait.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_catch_up_wait_that_ends_without_a_slot_records_one_cancelled_wait() {
        let (g, metrics) = catch_up_gate_with_metrics(1);
        let _held = g.acquire(no_deadline()).await.expect("fast path");
        let acquired_before = catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_ACQUIRED);

        {
            let mut dropped = std::pin::pin!(g.acquire(no_deadline()));
            assert!(poll_once(dropped.as_mut()).await.is_pending());
        }
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_CANCELLED),
            Some(1),
            "a dropped wait records `cancelled`"
        );

        let mut parked = std::pin::pin!(g.acquire(no_deadline()));
        assert!(poll_once(parked.as_mut()).await.is_pending());
        g.close();
        assert!(matches!(
            tokio::time::timeout(SETTLE, parked)
                .await
                .expect("close wakes the parked wait"),
            Err(CatchUpRefusal::Closed)
        ));
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_CANCELLED),
            Some(2),
            "a wait the close ends records `cancelled` once"
        );

        assert!(matches!(
            g.acquire(no_deadline()).await,
            Err(CatchUpRefusal::Closed)
        ));
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_CANCELLED),
            Some(3),
            "an acquire on a closed gate records `cancelled`"
        );
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_ACQUIRED),
            acquired_before,
            "no ending without a slot may count as `acquired`"
        );
    }

    /// A wait that reaches its deadline with no slot freed is refused
    /// `TimedOut`, records exactly one `timed_out` observation, leaves the
    /// queue, and takes nothing from the slot that is still held.
    ///
    /// Paused clock: the runtime auto-advances to the deadline once every task
    /// is idle, so the wait runs out without any wall-clock sleep.
    #[tokio::test(start_paused = true)]
    async fn a_catch_up_wait_that_reaches_its_deadline_is_refused_timed_out() {
        const BUDGET: Duration = Duration::from_millis(300);
        let (g, metrics) = catch_up_gate_with_metrics(1);
        let held = g.acquire(no_deadline()).await.expect("fast path");

        let started = tokio::time::Instant::now();
        let refused = g.acquire(started + BUDGET).await;
        assert!(
            matches!(refused, Err(CatchUpRefusal::TimedOut)),
            "no slot freed before the deadline, so the wait must be refused TimedOut"
        );
        assert!(
            tokio::time::Instant::now() >= started + BUDGET,
            "the refusal must come at the deadline, not before it"
        );
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_TIMED_OUT),
            Some(1),
            "a wait that ran out records one `timed_out` observation"
        );
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_CANCELLED),
            None,
            "running out is not a cancellation"
        );
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_ACQUIRED),
            Some(1),
            "only the holder's fast path counts as `acquired`"
        );
        assert_eq!(
            read_gauge(&metrics, "live_edit_subscribe_catchups_waiting"),
            0,
            "the timed-out wait left the queue"
        );
        assert_eq!(g.available(), 0, "the held slot is still held");

        // The queue position it left is not leaked: the holder's release
        // admits the next caller at once.
        drop(held);
        assert!(g.acquire(no_deadline()).await.is_ok());
    }

    /// A deadline already passed never joins the queue: with no free slot it
    /// is refused `TimedOut` at once, and with a free slot it still takes it.
    /// This is the zero-budget semantics ("never wait for capacity").
    #[tokio::test]
    async fn a_passed_deadline_takes_a_free_slot_and_is_otherwise_refused_without_queueing() {
        let (g, metrics) = catch_up_gate_with_metrics(1);
        let passed = tokio::time::Instant::now();

        let held = g
            .acquire(passed)
            .await
            .expect("a free slot is taken even with no budget left");

        let mut refused = std::pin::pin!(g.acquire(passed));
        match poll_once(refused.as_mut()).await {
            Poll::Ready(Err(CatchUpRefusal::TimedOut)) => {}
            Poll::Ready(other) => panic!(
                "a passed deadline with no free slot must be refused TimedOut, got {:?}",
                other.map(|_| ())
            ),
            Poll::Pending => {
                panic!("a passed deadline must be refused on its first poll, never queued")
            }
        }
        assert_eq!(
            read_gauge(&metrics, "live_edit_subscribe_catchups_waiting"),
            0,
            "the refusal never joined the queue"
        );
        assert_eq!(
            catchup_wait_count(&metrics, CATCHUP_WAIT_RESULT_TIMED_OUT),
            Some(1)
        );
        drop(held);
    }

    // ── CatchUpReadGate (Spec SDD SR-7.1, SR-7.3, SR-7.4, SR-8.5) ──────────────

    use crate::observability::metrics::{
        CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC, CATCHUP_SNAPSHOT_READ_CLASS_FENCED,
    };

    const READS_IN_FLIGHT: &str = "live_edit_catchup_snapshot_reads_in_flight";
    const READ_WAIT_COUNT: &str = "live_edit_catchup_snapshot_read_wait_seconds_count";

    fn read_gate_with_metrics(limit: usize) -> (CatchUpReadGate, SharedMetrics) {
        let metrics: SharedMetrics =
            Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let g = CatchUpReadGate::new(
            limit,
            CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC,
            Arc::new(StdMutex::new(Some(metrics.clone()))),
        );
        (g, metrics)
    }

    fn reads_in_flight(metrics: &SharedMetrics, class: &str) -> i64 {
        sample_value(&render(metrics), READS_IN_FLIGHT, &[("class", class)])
            .expect("pre-registered") as i64
    }

    fn read_waits(metrics: &SharedMetrics, consumer: SnapshotReadConsumer) -> Option<u64> {
        sample_value(
            &render(metrics),
            READ_WAIT_COUNT,
            &[("consumer", consumer.label())],
        )
        .map(|v| v as u64)
    }

    /// [`read_waits`] restricted to one permit `class`.
    fn read_waits_in_class(
        metrics: &SharedMetrics,
        consumer: SnapshotReadConsumer,
        class: &str,
    ) -> Option<u64> {
        sample_value(
            &render(metrics),
            READ_WAIT_COUNT,
            &[("consumer", consumer.label()), ("class", class)],
        )
        .map(|v| v as u64)
    }

    const CATCHUP: SnapshotReadConsumer = SnapshotReadConsumer::Catchup;

    /// At most `limit` permits are held at once; past it the waiters queue,
    /// and each release admits the longest-waiting one and no other.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_read_gate_grants_its_limit_and_queues_the_rest_in_arrival_order() {
        let (g, metrics) = read_gate_with_metrics(2);
        assert_eq!(g.limit(), 2);
        let a = g.acquire(no_deadline(), CATCHUP).await.unwrap();
        let b = g.acquire(no_deadline(), CATCHUP).await.unwrap();
        assert_eq!(g.available(), 0);
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            2
        );

        let mut first = std::pin::pin!(g.acquire(no_deadline(), CATCHUP));
        let mut second = std::pin::pin!(g.acquire(no_deadline(), CATCHUP));
        let mut third = std::pin::pin!(g.acquire(no_deadline(), CATCHUP));
        assert!(poll_once(first.as_mut()).await.is_pending());
        assert!(poll_once(second.as_mut()).await.is_pending());
        assert!(poll_once(third.as_mut()).await.is_pending());

        drop(a);
        assert!(poll_once(third.as_mut()).await.is_pending());
        assert!(poll_once(second.as_mut()).await.is_pending());
        let p1 = tokio::time::timeout(SETTLE, first.as_mut())
            .await
            .expect("the first release admits the first waiter")
            .expect("a permit");

        drop(b);
        assert!(poll_once(third.as_mut()).await.is_pending());
        let p2 = tokio::time::timeout(SETTLE, second.as_mut())
            .await
            .expect("the second release admits the second waiter")
            .expect("a permit");

        drop(p1);
        let p3 = tokio::time::timeout(SETTLE, third.as_mut())
            .await
            .expect("the third release admits the third waiter")
            .expect("a permit");
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            2,
            "never more than the limit in flight"
        );
        drop(p2);
        drop(p3);
        assert_eq!(g.available(), 2, "back to full capacity");
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            0
        );
    }

    /// A wait that reaches its deadline with no permit freed is refused
    /// `TimedOut` at the deadline, not before, and leaves the held permit
    /// held and the queue clean.
    #[tokio::test(start_paused = true)]
    async fn a_read_permit_wait_that_reaches_its_deadline_is_refused_timed_out() {
        const BUDGET: Duration = Duration::from_millis(300);
        let (g, metrics) = read_gate_with_metrics(1);
        let held = g.acquire(no_deadline(), CATCHUP).await.unwrap();

        let started = tokio::time::Instant::now();
        let refused = g.acquire(started + BUDGET, CATCHUP).await;
        assert!(matches!(refused, Err(CatchUpRefusal::TimedOut)));
        assert!(
            tokio::time::Instant::now() >= started + BUDGET,
            "the refusal comes at the deadline"
        );
        assert_eq!(g.available(), 0, "the held permit is still held");
        assert_eq!(
            read_waits(&metrics, CATCHUP),
            Some(2),
            "the holder's acquisition and the timed-out wait are both observed"
        );

        drop(held);
        assert!(
            g.acquire(no_deadline(), CATCHUP).await.is_ok(),
            "the timed-out wait left no queue position behind"
        );
    }

    /// A deadline already passed — a zero or spent capacity budget — takes a
    /// free permit, and with none free is refused on its first poll without
    /// joining the queue.
    #[tokio::test]
    async fn a_zero_budget_takes_a_free_read_permit_and_is_otherwise_refused_at_once() {
        let (g, _metrics) = read_gate_with_metrics(1);
        let passed = tokio::time::Instant::now();
        let held = g
            .acquire(passed, CATCHUP)
            .await
            .expect("a free permit is taken with no budget left");

        let mut refused = std::pin::pin!(g.acquire(passed, CATCHUP));
        match poll_once(refused.as_mut()).await {
            Poll::Ready(Err(CatchUpRefusal::TimedOut)) => {}
            Poll::Ready(other) => panic!(
                "a passed deadline with no free permit must be refused TimedOut, got {:?}",
                other.map(|_| ())
            ),
            Poll::Pending => panic!("a passed deadline must be refused at once, never queued"),
        }
        drop(held);
        assert_eq!(g.available(), 1);
    }

    /// Every way a permit or a wait can end returns the gate to full capacity:
    /// a dropped permit, a dropped wait, and an aborted task holding a permit.
    /// The in-flight gauge follows.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_read_gate_returns_to_full_capacity_on_drop_cancellation_and_abort() {
        let (g, metrics) = read_gate_with_metrics(1);
        let g = Arc::new(g);
        let held = g.acquire(no_deadline(), CATCHUP).await.unwrap();
        {
            let mut dropped = std::pin::pin!(g.acquire(no_deadline(), CATCHUP));
            assert!(poll_once(dropped.as_mut()).await.is_pending());
        }
        drop(held);
        assert_eq!(g.available(), 1, "the dropped wait took nothing");
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            0
        );

        let holder = {
            let g = Arc::clone(&g);
            tokio::spawn(async move {
                let _permit = g.acquire(no_deadline(), CATCHUP).await.unwrap();
                std::future::pending::<()>().await;
            })
        };
        let deadline = Instant::now() + SETTLE;
        while g.available() != 0 {
            assert!(
                Instant::now() < deadline,
                "the holder never took its permit"
            );
            tokio::task::yield_now().await;
        }
        holder.abort();
        let _ = holder.await;
        assert_eq!(g.available(), 1, "an aborted holder's permit returns");
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            0
        );
    }

    /// Closing the gate (shutdown) releases every parked waiter as `Closed`,
    /// refuses every later acquisition, and leaves a held permit valid until
    /// it is dropped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn closing_the_read_gate_releases_its_waiters_and_refuses_later_acquisitions() {
        let (g, _metrics) = read_gate_with_metrics(1);
        let held = g.acquire(no_deadline(), CATCHUP).await.unwrap();
        let mut parked = std::pin::pin!(g.acquire(no_deadline(), CATCHUP));
        assert!(poll_once(parked.as_mut()).await.is_pending());
        g.close();
        assert!(matches!(
            tokio::time::timeout(SETTLE, parked).await.unwrap(),
            Err(CatchUpRefusal::Closed)
        ));
        assert!(matches!(
            g.acquire(tokio::time::Instant::now(), CATCHUP).await,
            Err(CatchUpRefusal::Closed)
        ));
        drop(held);
    }

    /// The gauge is seeded at zero for both classes and counts under the
    /// gate's own class; each acquisition is observed once on the wait
    /// histogram under its consumer and the gate's class.
    #[tokio::test]
    async fn the_read_gate_counts_under_its_class_and_observes_each_wait_by_consumer() {
        let (g, metrics) = read_gate_with_metrics(2);
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            0
        );
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_FENCED),
            0,
            "the fenced class is pre-registered"
        );
        assert_eq!(read_waits(&metrics, CATCHUP), None);

        let a = g.acquire(no_deadline(), CATCHUP).await.unwrap();
        let b = g
            .acquire(no_deadline(), SnapshotReadConsumer::CatchupLegacyPerCommit)
            .await
            .unwrap();
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            2
        );
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_FENCED),
            0
        );
        assert_eq!(read_waits(&metrics, CATCHUP), Some(1));
        assert_eq!(
            read_waits(&metrics, SnapshotReadConsumer::CatchupLegacyPerCommit),
            Some(1)
        );
        assert_eq!(
            read_waits_in_class(&metrics, CATCHUP, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            Some(1),
            "an atomic gate's waits carry class=\"atomic\""
        );
        assert_eq!(
            read_waits_in_class(&metrics, CATCHUP, CATCHUP_SNAPSHOT_READ_CLASS_FENCED),
            None
        );
        drop((a, b));
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            0
        );
    }

    /// The fenced class (Spec SDD SR-7.1) is a gate of its own: built beside
    /// an atomic gate on the same metrics slot, as `ValkeyClient` builds them,
    /// its permits count under `class="fenced"` only, its waits are observed
    /// under `class="fenced"` only (never mixed with the atomic reads' waits),
    /// and with every fenced permit held an atomic acquisition is still
    /// granted at once.
    #[tokio::test]
    async fn a_fenced_class_read_gate_counts_under_fenced_and_never_delays_an_atomic_read() {
        let metrics: SharedMetrics =
            Arc::new(crate::observability::metrics::metrics_scrape_only("test"));
        let slot = Arc::new(StdMutex::new(Some(metrics.clone())));
        let atomic = CatchUpReadGate::new(1, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC, slot.clone());
        let fenced = CatchUpReadGate::new(2, CATCHUP_SNAPSHOT_READ_CLASS_FENCED, slot);

        let f1 = fenced.acquire(no_deadline(), CATCHUP).await.unwrap();
        let f2 = fenced.acquire(no_deadline(), CATCHUP).await.unwrap();
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_FENCED),
            2
        );
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            0,
            "a fenced permit is not counted as an atomic read"
        );
        let mut parked = std::pin::pin!(fenced.acquire(no_deadline(), CATCHUP));
        assert!(
            poll_once(parked.as_mut()).await.is_pending(),
            "the fenced gate is exhausted"
        );

        let mut atomic_read = std::pin::pin!(atomic.acquire(no_deadline(), CATCHUP));
        let a = match poll_once(atomic_read.as_mut()).await {
            Poll::Ready(Ok(p)) => p,
            Poll::Ready(Err(e)) => panic!("the atomic read was refused: {e:?}"),
            Poll::Pending => panic!("an exhausted fenced gate must not delay an atomic read"),
        };
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            1
        );
        drop((a, f1, f2));
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            0
        );
        let f3 = tokio::time::timeout(SETTLE, parked)
            .await
            .expect("a fenced release admits the fenced waiter")
            .expect("a permit");
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_FENCED),
            1
        );
        drop(f3);
        assert_eq!(
            reads_in_flight(&metrics, CATCHUP_SNAPSHOT_READ_CLASS_FENCED),
            0
        );
        assert_eq!(
            (
                read_waits_in_class(&metrics, CATCHUP, CATCHUP_SNAPSHOT_READ_CLASS_FENCED),
                read_waits_in_class(&metrics, CATCHUP, CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC),
            ),
            (Some(3), Some(1)),
            "the three fenced waits (the parked one included) under class=\"fenced\", the one \
             atomic wait under class=\"atomic\""
        );
    }

    /// A zero limit cannot reach the gate through configuration (it is
    /// rejected at startup); a gate built with one anyway still bounds reads
    /// at one rather than refusing them all.
    #[tokio::test]
    async fn a_read_gate_built_with_a_zero_limit_holds_one_permit() {
        let g = CatchUpReadGate::new(
            0,
            CATCHUP_SNAPSHOT_READ_CLASS_ATOMIC,
            Arc::new(StdMutex::new(None)),
        );
        assert_eq!(g.limit(), 1);
        let _held = g.acquire(no_deadline(), CATCHUP).await.unwrap();
        assert_eq!(g.available(), 0);
    }

    // ── CapacityBudget ───────────────────────────────────────────────────────

    /// A new budget holds its full size, which charges draw down additively
    /// and `refill` restores.
    #[test]
    fn capacity_budget_charges_accumulate_and_refill_restores_the_total() {
        let budget = CapacityBudget::new(Duration::from_millis(2_000));
        assert_eq!(budget.total(), Duration::from_millis(2_000));
        assert_eq!(budget.remaining(), Duration::from_millis(2_000));

        budget.charge(Duration::from_millis(300));
        budget.charge(Duration::from_millis(450));
        assert_eq!(
            budget.remaining(),
            Duration::from_millis(1_250),
            "every capacity wait the subscription makes draws on the one budget"
        );
        assert_eq!(
            budget.total(),
            Duration::from_millis(2_000),
            "charges leave the total"
        );

        budget.refill();
        assert_eq!(
            budget.remaining(),
            Duration::from_millis(2_000),
            "a re-bootstrap's refill restores the full budget"
        );
    }

    /// Charging more than is left saturates at zero instead of wrapping, and a
    /// refill still restores the full size afterwards.
    #[test]
    fn capacity_budget_overcharge_saturates_at_zero_and_refills_from_empty() {
        let budget = CapacityBudget::new(Duration::from_millis(100));
        budget.charge(Duration::from_millis(250));
        assert_eq!(budget.remaining(), Duration::ZERO);
        budget.charge(Duration::from_secs(10));
        assert_eq!(
            budget.remaining(),
            Duration::ZERO,
            "still zero, never wraps"
        );
        budget.refill();
        assert_eq!(budget.remaining(), Duration::from_millis(100));
    }

    /// The deadline for a wait starting now is now plus what is left: it
    /// shrinks with every charge, and an empty (or zero-sized) budget's
    /// deadline is now — a wait that never queues.
    #[tokio::test(start_paused = true)]
    async fn capacity_budget_deadline_is_now_plus_what_is_left() {
        let budget = CapacityBudget::new(Duration::from_millis(2_000));
        let now = tokio::time::Instant::now();
        assert_eq!(budget.deadline(), now + Duration::from_millis(2_000));

        budget.charge(Duration::from_millis(1_500));
        assert_eq!(budget.deadline(), now + Duration::from_millis(500));

        budget.charge(Duration::from_millis(1_500));
        assert_eq!(
            budget.deadline(),
            now,
            "an exhausted budget waits for nothing"
        );

        let zero = CapacityBudget::new(Duration::ZERO);
        assert_eq!(
            zero.deadline(),
            now,
            "LIVE_EDIT_SUBSCRIBER_ADMISSION_WAIT_MS=0 never waits for capacity"
        );
    }

    /// A budget too large for the clock neither panics nor overflows: its
    /// remaining time saturates and its deadline lands decades away.
    #[tokio::test(start_paused = true)]
    async fn capacity_budget_larger_than_the_clock_can_represent_yields_a_far_deadline() {
        let budget = CapacityBudget::new(Duration::MAX);
        assert_eq!(budget.total(), Duration::MAX);
        assert_eq!(
            budget.remaining(),
            Duration::from_nanos(u64::MAX),
            "remaining saturates at u64::MAX nanoseconds"
        );
        let now = tokio::time::Instant::now();
        assert!(
            budget.deadline() >= now + Duration::from_secs(86_400 * 365 * 29),
            "an unrepresentable budget is a deadline decades away, not a panic"
        );
        budget.charge(Duration::MAX);
        assert_eq!(budget.remaining(), Duration::ZERO);
    }
}
