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

//! The chain gate: where a chained `UpdateLayer`
//! (`UpdateLayerRequest.chained_header`) waits for its predecessor.
//!
//! A chained call names its predecessor, `previous_operation_id`: the same
//! `client_id`'s previous write to the object (empty for a chain's first). It
//! may commit only while the latest operation recorded for its
//! `(object_id, client_id)` — the `le:{oid}:obj:wtok` field the commit path
//! keeps ([`KeySchema::obj_write_token`](crate::storage::keys::KeySchema::obj_write_token))
//! — is that predecessor, or is its own operation (it then answers
//! `ALREADY_COMMITTED`). The `UpdateLayer` handler runs the gate after the
//! call's content is read and before its parse, so a waiting call holds only
//! its upload reservation (its fixed charge plus its content): no processing
//! charge or overflow slot, no external-value reference, no per-object writer
//! permit, no pool slot. A predecessor therefore never waits for anything a
//! waiting successor holds, except room in the upload budget `B`
//! (`grpc::upload_admission`). A predecessor that finds it full at admission
//! waits at most the admission wait and is then refused `UPLOAD_CAPACITY`,
//! applying nothing (a receive-growth charge that does not fit is refused at
//! once). Past admission, a predecessor's processing step is never refused
//! for capacity: it waits, up to its RPC deadline, for room in `B` (which
//! waiting successors hold) or for the overflow slot. That slot's holder may
//! itself be waiting at the external-value release budget (SDD §5.4; the
//! `UpdateLayer` and `CreateLayer` handlers await it while holding their
//! processing charge, and so the slot if they run in it), so the predecessor,
//! and the chain waiting behind it, can wait up to that holder's
//! release-budget wait. It is still deadlock-free: there the slot's holder
//! waits only on GC — the GC loop bringing the release total below the
//! budget — never on a chained call.
//!
//! # No call tracking
//!
//! The gate reads the record and nothing else. The service does not track the
//! predecessor's call, and never refuses a waiting call because that call
//! failed: the client learns of the failure first, from the predecessor's own
//! status, and cancels the calls waiting behind it. A cancel drops the
//! handler at the gate's await — the gate runs inside the request's own
//! future and spawns nothing — so a cancelled waiter applies nothing, holds
//! nothing and leaves nothing that can commit later. A waiter that is never
//! cancelled is refused `ABORTED` `PREDECESSOR_NOT_COMMITTED` at its margin
//! before its deadline (the deadline cap, below); with `no_wait` it is
//! refused at once, as soon as a probe reads the record.
//!
//! # Probe, classification, exits
//!
//! One probe is one pipelined round trip on the object's slot, through the
//! data pool, sent at most once ([`CHAIN_PROBE_MAX_ATTEMPTS`]):
//! `HGET le:{oid}:obj:wtok <client_id>` and `HMGET le:{oid}:obj:hdr seq
//! superseded`. A probe is advisory: the commit decides. [`classify`] reads it:
//!
//! - **ready** — the record is this call's operation or its predecessor's
//!   (`op.<id>:…`), or it is absent and the call starts a chain: the call goes
//!   on to its parse and commit;
//! - **replaced** — the object's header is gone or `superseded`: the handler
//!   re-resolves the layer (`FAILED_PRECONDITION` / `NOT_FOUND`);
//! - **pending** — anything else: the call waits, or with `no_wait` is refused
//!   at once. There is no exit for a record that "moved": a record moving to
//!   an earlier write of the chain is ordinary pipelining.
//!
//! A probe that fails counts `error` and the wait goes on, `no_wait` or not
//! (see the deadline cap below); a wrong-typed key ends the wait as `ready`,
//! and the commit refuses it as a keyspace type error (`WRONGTYPE_KEY` for
//! the record key; the store's raw `WRONGTYPE` for the header).
//! Graceful shutdown is raced by the handler, outside this module, and ends
//! the wait `UNAVAILABLE "server shutting down"`.
//!
//! # Schedule and ceiling
//!
//! The `k`-th sleep (from 0) is `rtt99·2^k`, capped at
//! [`CHAIN_PROBE_MAX_INTERVAL`] (20 ms), drawn uniformly in `[d/2, d]`, and
//! never shorter than `n` / [`CHAIN_PROBE_CEILING`] seconds, `n` being this
//! replica's waiting chained calls ([`chain_waiters`]): the probes sent after
//! a sleep stay at or below 4,000 per second per replica, however many wait,
//! so a flood of waiters slows detection rather than loading the store. The
//! ceiling paces only those. Each call's first probe on arrival is not paced,
//! and neither is a final probe pulled forward to the deadline cap (a sleep
//! cut short at the call's margin before its deadline), so the replica's
//! total can exceed 4,000 per second by the arrival rate of chained calls.
//! `rtt99` is the recorded small-call round trip
//! ([`FenceCoefficients::rtt99_ms`](crate::storage::snapshot_fence::FenceCoefficients::rtt99_ms));
//! it sets the schedule and nothing else.
//! Correctness depends on neither the rate nor the replica count: a probe can
//! be stale, never wrong, because the commit re-checks the record.
//!
//! **The deadline cap.** The `UpdateLayer` handler reads the call's deadline
//! budget `T` once, at its entry, before its first await
//! ([`grpc::deadline::remaining`](crate::grpc::deadline::remaining)): the
//! sooner of `LIVE_EDIT_RPC_DEADLINE_MS` and the client's `grpc-timeout`,
//! counted from the arrival of the request's headers, or
//! `LIVE_EDIT_RPC_DEADLINE_MS` when no deadline is in scope. The call's
//! deadline `D` is that instant plus `T`; both are passed to
//! [`ChainGate::enter`]. The gate's margin is `M = max(T / 10, 50 ms)`
//! ([`chain_deadline_margin`]).
//!
//! - No sleep ends later than `D − M`, and a probe still unanswered then is
//!   given up and counts as failed (`error`).
//! - The final probe is the first that starts at or after `D − M`. That
//!   includes a call's first probe when the call reaches the gate with no
//!   more than `M` left: such a call is checked once and does not wait —
//!   once more after a same-object re-resolve, below.
//! - The final probe is given up halfway between its start and `D`
//!   ([`final_probe_limit`]; `D − M/2` on the normal path). After a
//!   same-object re-resolve ([`ChainGate::keep_waiting`]) it is given up no
//!   later than `D − M/2` either ([`reentered_final_probe_limit`]). It
//!   answers `ready`, `replaced`, or — still pending — `ABORTED`
//!   `PREDECESSOR_NOT_COMMITTED` naming the freshest record read (outcome
//!   `timeout`); when no probe of the call ever read the record,
//!   `UNAVAILABLE` `CHAIN_CHECK_UNAVAILABLE` (outcome `unavailable`).
//!
//! So the gate's refusal (`ABORTED` or `CHAIN_CHECK_UNAVAILABLE`) of a call
//! that is never cancelled leaves, to within the timer's precision:
//!
//! - for a call that waited until its margin, at least `M/2` before `D`
//!   (to within the timer's precision: the final probe starts when its sleep
//!   wakes, at or just after `D − M`), normally about `M` before it;
//! - for a late arrival, at least half of what was left when it was checked;
//! - the exception to both: after a same-object re-resolve the call is probed
//!   again, a late arrival too. A waiter keeps the `M/2` floor only if the
//!   re-resolve finished before `D − M/2`; a late arrival keeps only the
//!   last bound below, counted from that later probe; a re-resolve that
//!   finishes after `D − M/2` leaves the call answered once it finishes (its
//!   next probe's limit has passed), and one that runs past `D` is not
//!   refused: the call ends unrefused at `D` — `MetricsLayer` ends it
//!   `DEADLINE_EXCEEDED` when `LIVE_EDIT_RPC_DEADLINE_MS` set `D`, and when a
//!   shorter client `grpc-timeout` did, tonic's `GrpcTimeout` drops it and
//!   `RecoverError` answers `CANCELLED "Timeout expired"`; either way the
//!   client normally reads its own `DEADLINE_EXCEEDED` or `CANCELLED` first,
//!   and the call is unsettled;
//! - on every path, at least the lesser of `M/2` and half of what was left
//!   when the call's last probe started.
//!
//! The margin does not bound the re-resolve's own answer: the handler sends
//! its `FAILED_PRECONDITION` / `NOT_FOUND`, or an error it met, whenever it
//! finishes.
//!
//! A `ready` at the final probe goes on with at least half of what was left
//! when that probe started. The margin is a fraction of the call's own
//! deadline, not the store's round trip, because the client's deadline timer
//! starts before `D`'s clock does — before the request is sent, and grpc-core
//! sends a header up to about 4% longer than the time the client has left —
//! so the answer must leave early by the client's round trip, its delay
//! before sending and that excess, none of which the service can observe. It has no cap and no
//! environment variable (SDD §7.5.7).
//!
//! # The commit-time guard
//!
//! The gate's `ready` is advisory: the record can move between its probe and
//! the commit, and a commit handed to the store before its caller was dropped
//! can run at any later time. The commit decides. A chained call's operation
//! carries its predecessor as its writer guard
//! ([`Operation::with_predecessor`](crate::storage::layer::Operation::with_predecessor)),
//! and every commit route, on every backend, checks it against the record
//! after the own-key match and the superseded guard and before the snapshot
//! fence: the update script's PREDECESSOR step (after its step 1c), the WATCH
//! path after its superseded guard, and the structural read side after its
//! superseded check, before planning. A record that is neither the call's
//! own operation nor its predecessor (for a chain's first write: any record)
//! refuses the attempt, writing nothing:
//!
//! - on the call's first attempt, `ABORTED` `PREDECESSOR_NOT_COMMITTED`
//!   naming the record the commit read, as the gate does — from the commit,
//!   so the gate's deadline margin does not bound when it is sent;
//! - after a counted retry, a plain `UNAVAILABLE`, outcome unknown: an
//!   earlier attempt of the call may have committed with its reply lost.
//!
//! The same reading as [`record_is_ready`], once the own-key match has been
//! answered. A chained call is never answered overtaken and never ticks the
//! overtaken counter: the start-record guard is unchained calls' alone, and a
//! chained call reads no start record.
//!
//! # Metrics
//!
//! `live_edit_update_layer_chain_wait_seconds{outcome}` once per call at its
//! exit, `live_edit_update_layer_chain_probes_total{result}` per probe, the
//! `live_edit_update_layer_chain_waiters` and
//! `live_edit_update_layer_chain_waiting_bytes` gauges while it waits, and
//! `live_edit_update_layer_chain_detect_lag_seconds` on a `ready` exit after
//! a sleep. A dropped waiter (client cancel, lost connection, the RPC
//! deadline) records `cancelled` from its drop guard, which also returns both
//! gauges.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use fred::interfaces::{ClientLike, HashesInterface};
use fred::types::Value;
use tokio::time::Instant;

use crate::error::{LastRecord, RefusalDetail, ServiceError};
use crate::observability::metrics::{
    SharedMetrics, CHAIN_OUTCOME_CANCELLED, CHAIN_OUTCOME_READY, CHAIN_OUTCOME_REFUSED,
    CHAIN_OUTCOME_REPLACED, CHAIN_OUTCOME_SHUTDOWN, CHAIN_OUTCOME_TIMEOUT,
    CHAIN_OUTCOME_UNAVAILABLE, CHAIN_PROBE_ERROR, CHAIN_PROBE_OK,
};
use crate::storage::client::{FredErrOrigin, POOL_LABEL_DATA};
use crate::storage::layer::{
    LayerStore, CALL_APPLIED_NOTHING, HEADER_FIELD_SEQ, HEADER_FIELD_SUPERSEDED,
    OPERATION_KEY_PREFIX,
};

/// The longest sleep between two probes of one waiting call: 20 ms. An
/// internal choice, measured with the chained load generator at the product
/// range (50 chained writers on one layer, about one edit per second each,
/// up to 16 in flight per writer): the gate's added wait (its detection lag)
/// was p99 ≤ 25 ms against the ≤ ~50 ms target, so the cap holds (SDD
/// §7.5.7).
pub const CHAIN_PROBE_MAX_INTERVAL: Duration = Duration::from_millis(20);

/// The most probes per second the waiting chained calls of one replica send
/// together after a sleep: 4,000. Every sleep is at least `n` / this many
/// seconds, `n` being [`chain_waiters`]; a call's first probe and a final
/// probe pulled forward to the deadline cap are not paced (see the module
/// docs). An internal choice. At the product range measured for
/// [`CHAIN_PROBE_MAX_INTERVAL`], the chained calls' probes ran at about 115
/// per second in all, first probes included, so the paced share stays far
/// under the ceiling (SDD §7.5.7).
pub const CHAIN_PROBE_CEILING: u64 = 4_000;

/// The divisor of the gate's deadline margin: a waiting call is answered once
/// only a tenth of its deadline budget `T` is left, or
/// [`CHAIN_DEADLINE_MARGIN_MIN`] when that is more ([`chain_deadline_margin`];
/// the deadline cap in the module docs). A tenth scales the room left for the
/// answer's trip to the client with the deadline the client chose.
pub const CHAIN_DEADLINE_MARGIN_DIVISOR: u32 = 10;

/// The floor of the gate's deadline margin: 50 ms, so a short deadline still
/// leaves the answer room to reach its client ([`chain_deadline_margin`]).
pub const CHAIN_DEADLINE_MARGIN_MIN: Duration = Duration::from_millis(50);

/// The gate's margin `M` for a call whose deadline budget is `budget` (`T`):
/// `max(T / CHAIN_DEADLINE_MARGIN_DIVISOR, CHAIN_DEADLINE_MARGIN_MIN)`, with
/// no cap. No sleep of the call ends later than `M` before its deadline, and
/// its final probe is the first to start at or after that point.
pub fn chain_deadline_margin(budget: Duration) -> Duration {
    (budget / CHAIN_DEADLINE_MARGIN_DIVISOR).max(CHAIN_DEADLINE_MARGIN_MIN)
}

/// When a final probe that starts at `start` is given up: halfway between
/// `start` and the call's deadline `deadline`, so the answer always keeps the
/// other half for its trip to the client. `D − M/2` for a final probe that
/// starts at `D − M`; `start` itself when `start` is at or after `deadline`.
pub fn final_probe_limit(start: Instant, deadline: Instant) -> Instant {
    start + deadline.saturating_duration_since(start) / 2
}

/// When a final probe that starts at `start` is given up once the call has
/// re-entered the gate ([`ChainGate::keep_waiting`]: a probe found the layer
/// replaced, and resolving it again named the same object):
/// [`final_probe_limit`]`(start, deadline)`, but never later than
/// `deadline − margin/2` (`D − M/2`, the normal path's limit). So the
/// re-resolve keeps the waiter's `M/2` floor whenever it finishes before
/// `D − M/2`. A probe that starts after that point has a limit already
/// passed: it is given up at once, and the call is answered from the record
/// it last read.
pub fn reentered_final_probe_limit(start: Instant, deadline: Instant, margin: Duration) -> Instant {
    let limit = final_probe_limit(start, deadline);
    match deadline.checked_sub(margin / 2) {
        Some(floor) => limit.min(floor),
        // `D − M/2` before the clock's range is already past.
        None => limit.min(start),
    }
}

/// The attempt budget of one probe: written to the wire at most once, so fred
/// never replays it after a reconnect. A failed probe is the gate's to retry,
/// on its own schedule.
pub const CHAIN_PROBE_MAX_ATTEMPTS: u32 = 1;

/// This replica's chained calls waiting in the gate right now: the `n` of the
/// probe ceiling. One per process; a process serves one replica.
static CHAIN_WAITERS: AtomicU64 = AtomicU64::new(0);

/// This replica's chained calls waiting in the gate right now.
pub fn chain_waiters() -> u64 {
    CHAIN_WAITERS.load(Ordering::Relaxed)
}

/// What a chained call names: its own `operation_id` (non-empty), its
/// predecessor's (empty: the call starts a chain) and its `no_wait`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainLink<'a> {
    pub operation_id: &'a str,
    pub previous_operation_id: &'a str,
    pub no_wait: bool,
}

/// What a probe read of the object's header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderState {
    /// The header holds a `seq` and no `superseded` of `1`.
    Live,
    /// No `seq`: the header is gone (the object was deleted, or replaced and
    /// cleaned up).
    Gone,
    /// `superseded` is `1`: a `CreateLayer` overwrite retired the object. `1`
    /// is the only value the overwrite writes and is what the WATCH-path
    /// guards test; the script's step 1c tests presence, which differs only
    /// for a value nothing writes.
    Superseded,
}

/// One successful probe: the writer's record and the object's header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainObservation {
    /// `le:{oid}:obj:wtok[client_id]`, verbatim; `None` when absent. Carries a
    /// commit token: never logged above DEBUG.
    pub record: Option<String>,
    pub header: HeaderState,
}

/// How a probe classifies ([`classify`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainState {
    Ready,
    Replaced,
    Pending,
}

/// Does `record` (`op.<id>:…`, or a no-operation record) carry
/// `operation_id`? Only a record written under an operation id can: its match
/// key is `op.` plus the id, up to the first `:`, which an id cannot contain.
pub fn record_names_operation(record: &str, operation_id: &str) -> bool {
    !operation_id.is_empty()
        && record
            .strip_prefix(OPERATION_KEY_PREFIX)
            .and_then(|rest| rest.strip_prefix(operation_id))
            .is_some_and(|rest| rest.starts_with(':'))
}

/// Is the writer's record one a chained call may commit after: its own
/// operation's, its predecessor's, or — for a chain's first write — none?
pub fn record_is_ready(record: Option<&str>, link: &ChainLink<'_>) -> bool {
    match record {
        None => link.previous_operation_id.is_empty(),
        Some(record) => {
            record_names_operation(record, link.operation_id)
                || record_names_operation(record, link.previous_operation_id)
        }
    }
}

/// Classify one probe for `link`. `ready` first: a call whose record is ready
/// goes on to its commit, which decides — its own-key answer precedes the
/// commit's superseded guard; a gone header is answered by the commit's
/// `NOT_FOUND` (the handler's recovery).
pub fn classify(observation: &ChainObservation, link: &ChainLink<'_>) -> ChainState {
    if record_is_ready(observation.record.as_deref(), link) {
        ChainState::Ready
    } else if observation.header != HeaderState::Live {
        ChainState::Replaced
    } else {
        ChainState::Pending
    }
}

/// The `last_record` a refusal reports for `record`.
pub fn last_record(record: Option<&str>) -> LastRecord<'_> {
    match record {
        None => LastRecord::Absent,
        Some(record) => match record.strip_prefix(OPERATION_KEY_PREFIX) {
            Some(rest) => LastRecord::Operation(rest.split(':').next().unwrap_or(rest)),
            None => LastRecord::NoOperationId,
        },
    }
}

/// The server timestamp (µs) of `record` when it is the record of
/// `previous_operation_id`: its third `:`-field
/// (`op.<id>:<seq>:<server_ts>:…`).
fn predecessor_server_ts(record: &str, previous_operation_id: &str) -> Option<u64> {
    if !record_names_operation(record, previous_operation_id) {
        return None;
    }
    record.split(':').nth(2)?.parse().ok()
}

/// The `k`-th sleep (`sleeps` = k, from 0) of a call while `waiters` calls wait
/// on this replica: `d = min(rtt99·2^k, CHAIN_PROBE_MAX_INTERVAL)`, drawn in
/// `[d/2, d]` by `jitter` in `[0, 1]`, and at least `waiters` /
/// [`CHAIN_PROBE_CEILING`] seconds.
pub fn probe_sleep(sleeps: u32, waiters: u64, rtt99: Duration, jitter: f64) -> Duration {
    let doubled = 1u32
        .checked_shl(sleeps)
        .and_then(|factor| rtt99.checked_mul(factor))
        .unwrap_or(CHAIN_PROBE_MAX_INTERVAL);
    let d = doubled.min(CHAIN_PROBE_MAX_INTERVAL);
    let half = d / 2;
    let jitter = if jitter.is_finite() {
        jitter.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let drawn = half + half.mul_f64(jitter);
    let floor = Duration::from_secs_f64(waiters as f64 / CHAIN_PROBE_CEILING as f64);
    drawn.max(floor)
}

/// How a call left the gate.
#[derive(Debug)]
pub enum GateExit {
    /// Go on to the parse and commit. Advisory: the record can move before
    /// the commit, whose predecessor guard decides (the commit-time guard in
    /// the module docs).
    Ready,
    /// The object's header was gone or superseded: re-resolve the layer. The
    /// gate is not finished: the handler calls [`ChainGate::finish_replaced`]
    /// or, when the layer still names this object, [`ChainGate::keep_waiting`].
    Replaced,
    /// Refused: `ABORTED` `PREDECESSOR_NOT_COMMITTED` (`no_wait`, or at the
    /// deadline cap), or `UNAVAILABLE` `CHAIN_CHECK_UNAVAILABLE`.
    Refused(ServiceError),
}

/// What one probe returned.
enum ProbeReading {
    Read(ChainObservation),
    /// A key held a foreign type.
    WrongType,
    Failed,
}

/// A chained call's place in the gate: the gauges it holds, and the outcome
/// its exit records. Dropped unfinished — the handler future dropped while it
/// waits — it records `cancelled`.
struct ChainWaiter {
    metrics: Option<SharedMetrics>,
    bytes: i64,
    entered: Instant,
    finished: bool,
}

impl ChainWaiter {
    fn enter(metrics: Option<SharedMetrics>, bytes: u64) -> Self {
        let bytes = i64::try_from(bytes).unwrap_or(i64::MAX);
        CHAIN_WAITERS.fetch_add(1, Ordering::Relaxed);
        if let Some(m) = &metrics {
            m.add_update_layer_chain_waiters(1, bytes);
        }
        Self {
            metrics,
            bytes,
            entered: Instant::now(),
            finished: false,
        }
    }

    fn waited(&self) -> Duration {
        self.entered.elapsed()
    }

    /// Record `outcome` and leave the gate's gauges. Once.
    fn finish(&mut self, outcome: &'static str) {
        if std::mem::replace(&mut self.finished, true) {
            return;
        }
        CHAIN_WAITERS.fetch_sub(1, Ordering::Relaxed);
        if let Some(m) = &self.metrics {
            m.add_update_layer_chain_waiters(-1, -self.bytes);
            m.observe_update_layer_chain_wait(outcome, self.waited());
        }
    }
}

impl Drop for ChainWaiter {
    fn drop(&mut self) {
        self.finish(CHAIN_OUTCOME_CANCELLED);
    }
}

/// One chained call's wait for its predecessor (see the module docs).
pub struct ChainGate<'a> {
    store: &'a LayerStore,
    object_id: u64,
    client_id: &'a str,
    link: ChainLink<'a>,
    /// `D`: when the call's deadline elapses.
    deadline: Instant,
    /// `M`: the call's margin before `deadline` ([`chain_deadline_margin`]).
    margin: Duration,
    /// The probe schedule's base step; nothing else reads it.
    rtt99: Duration,
    /// Sleeps taken so far: the schedule's `k`, and whether a `ready` exit
    /// comes after a sleep (the detection lag is observed only then).
    sleeps: u32,
    /// The handler re-resolved after a `replaced` probe and the layer still
    /// names this object ([`Self::keep_waiting`]).
    resolved_after_replaced: bool,
    /// Sleep before the next probe ([`Self::keep_waiting`]).
    sleep_first: bool,
    freshest: Option<ChainObservation>,
    probes_failed: u64,
    waiter: ChainWaiter,
}

impl<'a> ChainGate<'a> {
    /// Enter the gate for `link` on `object_id` under `client_id`. The call
    /// holds `held_bytes` of the upload budget while it waits (its
    /// reservation at entry).
    ///
    /// `budget` is the call's deadline budget `T`, read once at the handler's
    /// entry, and `deadline` is `D`, the instant of that read plus `T` (the
    /// deadline cap in the module docs). The gate's margin is
    /// [`chain_deadline_margin`]`(budget)`; no sleep ends later than that
    /// margin before `deadline`.
    // One parameter per fact the gate needs of its call; bundling them would
    // only move the list into a struct built at its one production caller.
    #[allow(clippy::too_many_arguments)]
    pub fn enter(
        store: &'a LayerStore,
        metrics: Option<SharedMetrics>,
        object_id: u64,
        client_id: &'a str,
        link: ChainLink<'a>,
        deadline: Instant,
        budget: Duration,
        held_bytes: u64,
    ) -> Self {
        let rtt99 = crate::storage::snapshot_fence::ms_to_duration(
            crate::storage::snapshot_fence::coefficients(store.client()).rtt99_ms,
        );
        Self {
            store,
            object_id,
            client_id,
            link,
            deadline,
            margin: chain_deadline_margin(budget),
            rtt99,
            sleeps: 0,
            resolved_after_replaced: false,
            sleep_first: false,
            freshest: None,
            probes_failed: 0,
            waiter: ChainWaiter::enter(metrics, held_bytes),
        }
    }

    /// Wait until the call can go on, is refused, or its object turns out
    /// replaced. Cancel-safe: dropping the future mid-wait leaves the gate
    /// unfinished, and dropping the gate then records `cancelled`.
    pub async fn wait(&mut self) -> GateExit {
        // `D − M`: no sleep ends later, and the first probe that starts at or
        // after it is the final one — a call's first probe too, when it
        // arrives with no more than its margin left, so it is checked once
        // per pass and does not wait (a same-object re-resolve makes another
        // pass). (An instant before the clock's range is already past: now
        // stands in for it.)
        let final_at = self
            .deadline
            .checked_sub(self.margin)
            .unwrap_or_else(Instant::now);
        if std::mem::take(&mut self.sleep_first) {
            self.sleep(final_at).await;
        }
        loop {
            let start = Instant::now();
            let is_final = start >= final_at;
            // Each probe has its own time limit, so a stalled store cannot
            // hold the call past the point where it can still answer: an
            // earlier probe is given up at `final_at`, leaving the final probe
            // its turn, and the final probe halfway between its start and the
            // deadline, leaving the other half for the answer's trip — after
            // a same-object re-resolve, no later than `D − M/2`.
            let limit = if !is_final {
                final_at
            } else if self.resolved_after_replaced {
                reentered_final_probe_limit(start, self.deadline, self.margin)
            } else {
                final_probe_limit(start, self.deadline)
            };
            match self.probe(limit).await {
                ProbeReading::Read(observation) => {
                    let state = classify(&observation, &self.link);
                    self.freshest = Some(observation);
                    match state {
                        ChainState::Ready => return self.ready(),
                        // The layer still named this object when the handler
                        // re-resolved: wait on it as on a pending record, and
                        // answer as one when the call cannot wait.
                        ChainState::Replaced
                            if self.resolved_after_replaced && (self.link.no_wait || is_final) =>
                        {
                            return self.refuse(if is_final && !self.link.no_wait {
                                CHAIN_OUTCOME_TIMEOUT
                            } else {
                                CHAIN_OUTCOME_REFUSED
                            });
                        }
                        ChainState::Replaced => return GateExit::Replaced,
                        ChainState::Pending if self.link.no_wait => {
                            return self.refuse(CHAIN_OUTCOME_REFUSED);
                        }
                        ChainState::Pending => {}
                    }
                }
                ProbeReading::WrongType => {
                    // Proceed: every commit route reads the record key and the
                    // header before it compares anything, and refuses a
                    // foreign type as a keyspace type error (`WRONGTYPE_KEY`
                    // for the record key; the store's raw `WRONGTYPE` for the
                    // header). The client sees `INTERNAL` either way.
                    self.waiter.finish(CHAIN_OUTCOME_READY);
                    return GateExit::Ready;
                }
                ProbeReading::Failed => self.probes_failed += 1,
            }
            if is_final {
                return match self.freshest {
                    Some(_) => self.refuse(CHAIN_OUTCOME_TIMEOUT),
                    None => self.unavailable(),
                };
            }
            self.sleep(final_at).await;
        }
    }

    /// After [`GateExit::Replaced`]: the handler re-resolved the layer and it
    /// still names this object (a `CreateLayer` overwrite between setting
    /// `superseded` and moving the mapping, a header lost under a live
    /// mapping, or a checkpoint restore that brought the object back). The
    /// call waits on, as for a pending record: the next [`Self::wait`] sleeps
    /// first (a `no_wait` call does not, and a call already past `D − M`, a
    /// late arrival included, has no sleep left and is probed again at once),
    /// and a `replaced` probe that it cannot wait past answers the predecessor
    /// refusal. From here on, a final probe is given up no later than
    /// `D − M/2` ([`reentered_final_probe_limit`]), so a re-resolve that
    /// finished after that point leaves the call refused at once; the
    /// re-resolve's own time is not bounded by the gate.
    pub fn keep_waiting(&mut self) {
        self.resolved_after_replaced = true;
        self.sleep_first = !self.link.no_wait;
    }

    /// After [`GateExit::Replaced`]: the call ends on the re-resolve's answer
    /// (the layer names another object, or none). Records `replaced`.
    pub fn finish_replaced(&mut self) {
        self.waiter.finish(CHAIN_OUTCOME_REPLACED);
    }

    /// Graceful shutdown ended the wait. Records `shutdown`.
    pub fn finish_shutdown(&mut self) {
        self.waiter.finish(CHAIN_OUTCOME_SHUTDOWN);
    }

    async fn sleep(&mut self, final_at: Instant) {
        let sleep = probe_sleep(
            self.sleeps,
            chain_waiters(),
            self.rtt99,
            rand::random::<f64>(),
        );
        self.sleeps = self.sleeps.saturating_add(1);
        tokio::time::sleep_until((Instant::now() + sleep).min(final_at)).await;
    }

    fn ready(&mut self) -> GateExit {
        if self.sleeps > 0 {
            let record = self.freshest.as_ref().and_then(|o| o.record.as_deref());
            if let (Some(m), Some(ts)) = (
                &self.waiter.metrics,
                record.and_then(|r| predecessor_server_ts(r, self.link.previous_operation_id)),
            ) {
                let lag_us = crate::domain::time::now_micros().saturating_sub(ts);
                m.observe_update_layer_chain_detect_lag(Duration::from_micros(lag_us));
            }
        }
        self.waiter.finish(CHAIN_OUTCOME_READY);
        GateExit::Ready
    }

    /// `ABORTED` `PREDECESSOR_NOT_COMMITTED`, naming the freshest record.
    fn refuse(&mut self, outcome: &'static str) -> GateExit {
        let record = self.freshest.as_ref().and_then(|o| o.record.as_deref());
        let latest = describe_record(record);
        let predecessor = describe_predecessor(self.link.previous_operation_id);
        let waited_ms = u64::try_from(self.waiter.waited().as_millis()).unwrap_or(u64::MAX);
        let message = if outcome == CHAIN_OUTCOME_REFUSED {
            format!(
                "no_wait: the latest operation recorded for this client_id on object {} is {}, \
                 not this call's predecessor ({predecessor}){CALL_APPLIED_NOTHING}",
                self.object_id, latest
            )
        } else {
            format!(
                "this call's predecessor ({predecessor}) did not become the latest operation \
                 recorded for this client_id on object {} within the call's deadline (waited \
                 {waited_ms} ms; latest: {latest}){CALL_APPLIED_NOTHING}",
                self.object_id
            )
        };
        let error = ServiceError::predecessor_not_committed(message, last_record(record));
        self.waiter.finish(outcome);
        GateExit::Refused(error)
    }

    /// `UNAVAILABLE` `CHAIN_CHECK_UNAVAILABLE`: no probe read the record.
    fn unavailable(&mut self) -> GateExit {
        let waited_ms = u64::try_from(self.waiter.waited().as_millis()).unwrap_or(u64::MAX);
        let message = format!(
            "could not read this client_id's commit record on object {} within the call's \
             deadline (waited {waited_ms} ms, {} failed probes){CALL_APPLIED_NOTHING}",
            self.object_id, self.probes_failed
        );
        self.waiter.finish(CHAIN_OUTCOME_UNAVAILABLE);
        GateExit::Refused(RefusalDetail::chain_check_unavailable().into_unavailable_error(message))
    }

    /// One probe: `HGET le:{oid}:obj:wtok <client_id>` and `HMGET
    /// le:{oid}:obj:hdr seq superseded`, pipelined in one round trip on the
    /// object's slot through the data pool, and given up as
    /// [`ProbeReading::Failed`] at `limit`.
    async fn probe(&self, limit: Instant) -> ProbeReading {
        let client = self.store.client();
        let origin = FredErrOrigin::client(client, POOL_LABEL_DATA);
        let object_id = self.object_id.to_string();
        let wtok_key = self.store.keys().obj_write_token(&object_id);
        let header_key = self.store.keys().obj_header(&object_id);
        let options = fred::types::config::Options {
            max_attempts: Some(CHAIN_PROBE_MAX_ATTEMPTS),
            ..Default::default()
        };
        let round_trip = async {
            let pipeline = client.pool.next().pipeline().with_options(&options);
            let _: () = pipeline.hget(wtok_key.as_str(), self.client_id).await?;
            let _: () = pipeline
                .hmget(
                    header_key.as_str(),
                    vec![HEADER_FIELD_SEQ, HEADER_FIELD_SUPERSEDED],
                )
                .await?;
            Ok::<_, fred::error::Error>(pipeline.try_all::<Value>().await)
        };
        // A probe still unanswered at `limit` is dropped and counts as failed:
        // the gate's own cut, not a store command timeout, so it ticks no
        // `valkey_errors_total` cell.
        let Ok(reading) = tokio::time::timeout_at(limit, round_trip).await else {
            if let Some(m) = self.waiter.metrics.as_deref() {
                m.inc_update_layer_chain_probe(CHAIN_PROBE_ERROR);
            }
            tracing::debug!(
                object_id = self.object_id,
                "chain gate probe timed out at its time limit; waiting on"
            );
            return ProbeReading::Failed;
        };
        let reading = match reading {
            Ok(mut replies) if replies.len() == 2 => {
                let header = replies.pop().expect("length checked");
                let record = replies.pop().expect("length checked");
                classify_replies(record, header)
            }
            Ok(replies) => Err(replies
                .into_iter()
                .find_map(Result::err)
                .unwrap_or_else(|| {
                    fred::error::Error::new(
                        fred::error::ErrorKind::Protocol,
                        "chain probe returned a short reply set",
                    )
                })),
            Err(e) => Err(e),
        };
        let metrics = self.waiter.metrics.as_deref();
        match reading {
            Ok(reading) => {
                if let Some(m) = metrics {
                    m.inc_update_layer_chain_probe(match reading {
                        ProbeReading::Read(_) => CHAIN_PROBE_OK,
                        _ => CHAIN_PROBE_ERROR,
                    });
                }
                reading
            }
            Err(e) => {
                origin.observe_error(&e);
                if let Some(m) = metrics {
                    m.inc_update_layer_chain_probe(CHAIN_PROBE_ERROR);
                }
                tracing::debug!(
                    object_id = self.object_id,
                    error = %e,
                    "chain gate probe failed; waiting on"
                );
                ProbeReading::Failed
            }
        }
    }
}

/// Is `e` a `WRONGTYPE` reply (a key holding a foreign type)?
fn is_wrong_type(e: &fred::error::Error) -> bool {
    e.details().starts_with("WRONGTYPE")
}

/// One probe's two replies, read. `Err` for a failure that is not a
/// wrong-typed key.
fn classify_replies(
    record: Result<Value, fred::error::Error>,
    header: Result<Value, fred::error::Error>,
) -> Result<ProbeReading, fred::error::Error> {
    let record = match record.and_then(|v| v.convert::<Option<String>>()) {
        Ok(record) => Ok(record),
        Err(e) if is_wrong_type(&e) => {
            return Ok(ProbeReading::WrongType);
        }
        Err(e) => Err(e),
    }?;
    let fields = match header.and_then(|v| v.convert::<Vec<Option<String>>>()) {
        Ok(fields) => fields,
        Err(e) if is_wrong_type(&e) => return Ok(ProbeReading::WrongType),
        Err(e) => return Err(e),
    };
    let header = match fields.as_slice() {
        [None, _] => HeaderState::Gone,
        // `1` is the only value the overwrite writes (`LayerStore`'s
        // `CreateLayer` overwrite) and is what the WATCH-path guards test; the
        // script's step 1c tests presence, which differs only for a value
        // nothing writes.
        [Some(_), Some(superseded)] if superseded == "1" => HeaderState::Superseded,
        [Some(_), _] => HeaderState::Live,
        _ => {
            return Err(fred::error::Error::new(
                fred::error::ErrorKind::Protocol,
                "chain probe: HMGET returned an unexpected reply",
            ))
        }
    };
    Ok(ProbeReading::Read(ChainObservation { record, header }))
}

/// A refusal message's name for the record `record`. Never the record
/// itself, which carries a commit token.
fn describe_record(record: Option<&str>) -> String {
    match last_record(record) {
        LastRecord::Absent => "no record".to_owned(),
        LastRecord::Operation(operation_id) => format!("operation {operation_id}"),
        LastRecord::NoOperationId => "a commit without an operation_id".to_owned(),
    }
}

/// A refusal message's name for the predecessor.
fn describe_predecessor(previous_operation_id: &str) -> String {
    if previous_operation_id.is_empty() {
        "no record: the first write of a chain".to_owned()
    } else {
        format!("operation {previous_operation_id}")
    }
}

/// The message of a chained call's predecessor refusal made by its commit
/// rather than by the gate (the commit-time guard in the module docs): on
/// `object_id`, the record `record` (`None`: absent) does not name the call's
/// predecessor `previous_operation_id`. Names the record the commit read,
/// never the record itself, and ends [`CALL_APPLIED_NOTHING`], which
/// [`outcome_unknown_after_retry`](crate::storage::layer::outcome_unknown_after_retry)
/// strips when the refusal came after a counted retry.
pub(crate) fn commit_predecessor_message(
    object_id: u64,
    previous_operation_id: &str,
    record: Option<&str>,
) -> String {
    format!(
        "this call's predecessor ({}) is not the latest operation recorded for this client_id \
         on object {object_id} at its commit (latest: {}){CALL_APPLIED_NOTHING}",
        describe_predecessor(previous_operation_id),
        describe_record(record)
    )
}

#[cfg(test)]
mod tests {
    //! The gate's pure parts: matching a record to an operation, classifying
    //! a probe, the refusal's `last_record`, the detection lag's timestamp,
    //! the probe schedule (b.pav group 21) and the deadline cap's margin math
    //! (group 26: the margin, the final probe's time limit, and that limit
    //! after a same-object re-resolve). The gate's exits against a store —
    //! every outcome, the deadline cap, the probe's time limit — are pinned in
    //! `tests/integration/grpc/chained_writes_test.rs`.

    use super::*;

    /// The recorded `rtt99` (1.3915 ms), rounded to whole microseconds so
    /// every expected sleep below is exact.
    const RTT99: Duration = Duration::from_micros(1_392);

    fn link<'a>(operation_id: &'a str, previous_operation_id: &'a str) -> ChainLink<'a> {
        ChainLink {
            operation_id,
            previous_operation_id,
            no_wait: false,
        }
    }

    fn observed(record: Option<&str>, header: HeaderState) -> ChainObservation {
        ChainObservation {
            record: record.map(str::to_owned),
            header,
        }
    }

    // ── Records ──────────────────────────────────────────────────────────────

    /// A record names an operation exactly when it is `op.` + the whole id +
    /// `:` — never a longer id the name is a prefix of, never a commit made
    /// without an operation id, and never the empty id.
    #[test]
    fn record_names_operation_matches_the_whole_id_and_nothing_else() {
        assert!(record_names_operation("op.w1:7:1700000000:tok", "w1"));
        assert!(
            !record_names_operation("op.w12:7:1700000000:tok", "w1"),
            "an id that is a prefix of the recorded one"
        );
        assert!(
            !record_names_operation("op.w1", "w1"),
            "no `:` after the id: not a record of it"
        );
        assert!(
            !record_names_operation("w1:7:1700000000", "w1"),
            "a commit without an operation id"
        );
        assert!(
            !record_names_operation("op.:7:1700000000:tok", ""),
            "the empty id names no record"
        );
    }

    /// `ready` on the predecessor's record or the call's own — whatever the
    /// header says, since the commit decides: its own-key answer precedes its
    /// superseded guard, and a gone header is answered by its `NOT_FOUND`
    /// (the handler's recovery) — or, for a chain's first write, on no
    /// record; otherwise `replaced` when the header is gone or superseded,
    /// else `pending`.
    #[test]
    fn classify_matrix() {
        let next = link("w2", "w1");
        let first = link("w1", "");
        let rows: &[(Option<&str>, HeaderState, &ChainLink<'_>, ChainState)] = &[
            (
                Some("op.w1:2:10:t"),
                HeaderState::Live,
                &next,
                ChainState::Ready,
            ),
            (
                Some("op.w2:3:10:t"),
                HeaderState::Live,
                &next,
                ChainState::Ready,
            ),
            (
                Some("op.w2:3:10:t"),
                HeaderState::Superseded,
                &next,
                ChainState::Ready,
            ),
            (
                Some("op.w1:2:10:t"),
                HeaderState::Gone,
                &next,
                ChainState::Ready,
            ),
            (None, HeaderState::Live, &first, ChainState::Ready),
            (None, HeaderState::Live, &next, ChainState::Pending),
            (
                Some("op.w0:1:10:t"),
                HeaderState::Live,
                &next,
                ChainState::Pending,
            ),
            (
                Some("tok:1:10"),
                HeaderState::Live,
                &next,
                ChainState::Pending,
            ),
            (
                Some("op.w9:9:10:t"),
                HeaderState::Live,
                &first,
                ChainState::Pending,
            ),
            (
                Some("op.w0:1:10:t"),
                HeaderState::Superseded,
                &next,
                ChainState::Replaced,
            ),
            (
                Some("op.w0:1:10:t"),
                HeaderState::Gone,
                &next,
                ChainState::Replaced,
            ),
            (None, HeaderState::Gone, &next, ChainState::Replaced),
        ];
        for (record, header, link, want) in rows {
            assert_eq!(
                classify(&observed(*record, *header), link),
                *want,
                "record {record:?}, header {header:?}, link {link:?}"
            );
        }
    }

    /// A refusal reports what it found: no record (`absent`), an operation's
    /// record by its id, or a commit made without an operation id.
    #[test]
    fn last_record_reports_absent_the_operation_or_no_operation_id() {
        assert_eq!(last_record(None), LastRecord::Absent);
        assert_eq!(
            last_record(Some("op.w1:7:1700000000:tok")),
            LastRecord::Operation("w1")
        );
        assert_eq!(
            last_record(Some("tok:7:1700000000")),
            LastRecord::NoOperationId
        );
    }

    /// The detection lag is measured from the predecessor's commit, the
    /// server timestamp in its record's third field — and only from the
    /// predecessor's record: a `ready` on the call's own record observes none.
    #[test]
    fn predecessor_server_ts_reads_only_the_predecessors_record() {
        assert_eq!(
            predecessor_server_ts("op.w1:7:1700000000:tok", "w1"),
            Some(1_700_000_000)
        );
        assert_eq!(predecessor_server_ts("op.w2:8:1700000000:tok", "w1"), None);
        assert_eq!(predecessor_server_ts("op.w1:7:soon:tok", "w1"), None);
        assert_eq!(predecessor_server_ts("op.w1:7:1700000000:tok", ""), None);
    }

    // ── The schedule ─────────────────────────────────────────────────────────

    /// The `k`-th sleep is `rtt99·2^k`, drawn in its upper half: `jitter` 0
    /// gives `d/2`, 1 gives `d`, anything between lands between.
    #[test]
    fn probe_sleep_doubles_from_rtt99_and_is_drawn_in_its_upper_half() {
        for k in 0..4u32 {
            let d = RTT99 * 2u32.pow(k);
            assert_eq!(probe_sleep(k, 0, RTT99, 0.0), d / 2, "k={k}, jitter 0");
            assert_eq!(probe_sleep(k, 0, RTT99, 1.0), d, "k={k}, jitter 1");
            let mid = probe_sleep(k, 0, RTT99, 0.5);
            assert!(d / 2 < mid && mid < d, "k={k}, jitter 0.5: {mid:?}");
        }
    }

    /// The doubling stops at [`CHAIN_PROBE_MAX_INTERVAL`] (20 ms), drawn in
    /// `[10 ms, 20 ms]`, however many sleeps came before — the shift that
    /// would overflow included.
    #[test]
    fn probe_sleep_is_capped_at_the_maximum_interval_for_every_sleep_count() {
        for k in [4, 5, 10, 31, 32, 64, u32::MAX] {
            assert_eq!(
                probe_sleep(k, 0, RTT99, 0.0),
                CHAIN_PROBE_MAX_INTERVAL / 2,
                "k={k}, jitter 0"
            );
            assert_eq!(
                probe_sleep(k, 0, RTT99, 1.0),
                CHAIN_PROBE_MAX_INTERVAL,
                "k={k}, jitter 1"
            );
        }
        assert_eq!(CHAIN_PROBE_MAX_INTERVAL, Duration::from_millis(20));
    }

    /// The ceiling: every sleep is at least `waiters` / 4,000 s, so the
    /// replica's waiting calls probe after their sleeps at most 4,000 times a
    /// second together, however many wait.
    #[test]
    fn probe_sleep_keeps_every_waiter_under_the_replica_ceiling() {
        assert_eq!(CHAIN_PROBE_CEILING, 4_000);
        for waiters in [0u64, 1, 40, 80, 4_000, 10_000] {
            let floor = Duration::from_secs_f64(waiters as f64 / CHAIN_PROBE_CEILING as f64);
            for k in [0u32, 1, 3, 10] {
                for jitter in [0.0, 0.5, 1.0] {
                    let sleep = probe_sleep(k, waiters, RTT99, jitter);
                    assert!(
                        sleep >= floor,
                        "waiters {waiters}, k={k}, jitter {jitter}: {sleep:?} < {floor:?}"
                    );
                }
            }
        }
        assert_eq!(
            probe_sleep(10, 4_000, RTT99, 1.0),
            Duration::from_secs(1),
            "4,000 waiters: one second each, 4,000 probes a second"
        );
        assert_eq!(
            probe_sleep(0, 0, RTT99, 1.0),
            RTT99,
            "no other waiter: the floor is zero"
        );
    }

    /// A `jitter` outside `[0, 1]` is clamped to it, and one that is not a
    /// finite number counts as 0: the draw never leaves `[d/2, d]`.
    #[test]
    fn probe_sleep_bounds_an_out_of_range_jitter() {
        assert_eq!(probe_sleep(0, 0, RTT99, -1.0), RTT99 / 2);
        assert_eq!(probe_sleep(0, 0, RTT99, 2.0), RTT99);
        assert_eq!(probe_sleep(0, 0, RTT99, f64::NAN), RTT99 / 2);
        assert_eq!(probe_sleep(0, 0, RTT99, f64::INFINITY), RTT99 / 2);
    }

    // ── The deadline cap's margin (b.pav group 26) ───────────────────────────

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// `M = max(T / 10, 50 ms)` with no cap: the 50 ms floor up to a 500 ms
    /// deadline, a tenth from there on, however long the deadline.
    #[test]
    fn chain_deadline_margin_is_a_tenth_of_the_deadline_floored_at_50_ms_and_never_capped() {
        assert_eq!(CHAIN_DEADLINE_MARGIN_DIVISOR, 10);
        assert_eq!(CHAIN_DEADLINE_MARGIN_MIN, ms(50));
        let rows = [
            (Duration::ZERO, ms(50)),
            (ms(1), ms(50)),
            (ms(499), ms(50)),
            (ms(500), ms(50)),
            (ms(600), ms(60)),
            (ms(1_500), ms(150)),
            (ms(3_000), ms(300)),
            (ms(30_000), ms(3_000)),
            (ms(300_000), ms(30_000)),
        ];
        for (budget, margin) in rows {
            assert_eq!(
                chain_deadline_margin(budget),
                margin,
                "deadline budget {budget:?}"
            );
        }
    }

    /// A final probe is given up halfway between its start and the deadline:
    /// `D − M/2` for one that starts at `D − M`; at the deadline, or after it,
    /// at its own start — never later than the deadline, never before its
    /// start.
    #[test]
    fn final_probe_limit_is_halfway_from_its_start_to_the_deadline() {
        let d = Instant::now() + Duration::from_secs(60);
        let m = chain_deadline_margin(Duration::from_secs(3));
        assert_eq!(final_probe_limit(d - m, d), d - m / 2, "start at D − M");
        assert_eq!(
            final_probe_limit(d - ms(400), d),
            d - ms(200),
            "start at D − 400 ms"
        );
        assert_eq!(final_probe_limit(d, d), d, "start at D");
        let late = d + ms(25);
        assert_eq!(final_probe_limit(late, d), late, "start after D");
    }

    /// After a same-object re-resolve the final probe's limit is
    /// [`final_probe_limit`] capped at `D − M/2`: so a waiter whose re-resolve
    /// finished before `D − M/2` keeps its floor; one that finished after it
    /// meets a limit already passed and is answered at once; one past `D` is
    /// never given more than its own start.
    #[test]
    fn reentered_final_probe_limit_never_gives_the_probe_past_d_minus_half_the_margin() {
        let d = Instant::now() + Duration::from_secs(60);
        let m = ms(1_000);
        let floor = d - m / 2;
        assert_eq!(
            reentered_final_probe_limit(d - m, d, m),
            floor,
            "start at D − M: the normal path's limit, unchanged"
        );
        let start = d - m.mul_f64(0.8);
        assert_eq!(
            final_probe_limit(start, d),
            d - m.mul_f64(0.4),
            "precondition: uncapped, a probe starting at D − 0.8M would run to D − 0.4M"
        );
        assert_eq!(
            reentered_final_probe_limit(start, d, m),
            floor,
            "start at D − 0.8M: capped at D − M/2"
        );
        let after_floor = d - m.mul_f64(0.3);
        let capped = reentered_final_probe_limit(after_floor, d, m);
        assert_eq!(capped, floor, "start after D − M/2: the cap itself");
        assert!(
            capped < after_floor,
            "start after D − M/2: a limit before its start, so the probe is given up at once"
        );
        for start in [d, d + ms(10)] {
            assert!(
                reentered_final_probe_limit(start, d, m) <= start,
                "start at or after D ({:?} past it): no later than its start",
                start.saturating_duration_since(d)
            );
        }
    }

    /// When `D − M/2` falls before the clock's range, that point is already
    /// past: the limit is the probe's start, so it is given up at once rather
    /// than handed the uncapped half of what is left. (The clock's range
    /// reaches far enough back that only an instant itself near its start can
    /// underflow, so the probe here starts 2^62 s before now.)
    #[test]
    fn reentered_final_probe_limit_takes_the_start_when_d_minus_half_the_margin_underflows() {
        let start = Instant::now()
            .checked_sub(Duration::from_secs(1 << 62))
            .expect("precondition: the clock reaches 2^62 s back");
        let d = start + Duration::from_secs(60);
        let margin = Duration::MAX;
        assert!(
            d.checked_sub(margin / 2).is_none(),
            "precondition: D − M/2 underflows the clock"
        );
        assert_eq!(
            final_probe_limit(start, d),
            start + Duration::from_secs(30),
            "precondition: uncapped, the probe would have half of what is left"
        );
        assert_eq!(reentered_final_probe_limit(start, d, margin), start);
    }
}
