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

//! Startup counter-seed recovery (SDD SR-5.3, Epic `t1.fgj.mh` / Task
//! `t2.fgj.mh.4u`).
//!
//! [`seed_object_id_counter`] is the AOF-off recovery primitive `main.rs`
//! calls once during service startup, **before** any RPC handler accepts
//! traffic. It reads the singleton counters-sidecar blob produced by the
//! Epic-6 periodic counters-sync tick (SR-5.1), parses it back into
//! [`CountersSidecar`], adds the configured safety margin, and asks
//! [`CounterStore::seed_object_id_if_below`] to conditionally advance
//! `le:counter:object_id` past that value.
//!
//! ## Why a dedicated module
//!
//! The routine is pure orchestration over three already-existing
//! primitives:
//!
//! - [`CheckpointStore::get_counters_sidecar`] — adapter-side bytes fetch.
//! - [`CounterStore::seed_object_id_if_below`] — atomic conditional `SET`
//!   via Lua (sibling Task 2, `t2.fgj.mh.b6`).
//! - [`CountersSidecar`] — shared producer/consumer wire shape (sibling
//!   Task 3, `t2.fgj.mh.fc`).
//!
//! Living in its own module lets integration tests target the seeding
//! logic against a real Valkey + adapter without standing up the full
//! `main.rs` startup. The downstream main.rs-wiring Subtask consumes this
//! routine through a pure async function call with no shared state beyond
//! the explicit args.
//!
//! ## Two call sites, one body
//!
//! Since b.b2j the same seeding body also runs at **runtime**, from
//! [`CheckpointObjectIdCounterRepair`], when a `CreateLayer` finds
//! `le:counter:object_id` gone. The shared body is
//! [`seed_object_id_counter_inner`]; [`seed_object_id_counter`] is the
//! startup-flavoured wrapper. A [`SeedTrigger`] discriminator rides through
//! so the two phases stay separable in logs and metrics — a repair-time
//! sample landing in a histogram documented as "the startup routine" is a
//! lie an operator debugging startup latency would act on, and the startup
//! `Absent` WARN's AOF-durability framing is simply the wrong question at
//! `CreateLayer` time.
//!
//! ## Histogram observation discipline
//!
//! The routine observes `live_edit_checkpoint_counter_seed_secs{phase}` on
//! every exit path **except** the [`SeedError::Store`] arm — the adapter
//! fetch itself failed and the operator signal is the readiness gate (the
//! main.rs-wiring Subtask) or the repair's own outcome counter, not a
//! histogram sample. Observation is gated by an inline `impl Drop` guard so
//! the metric still fires on the `ParseFailed` error path (where the
//! duration is operationally meaningful: an operator can correlate the
//! elapsed time against an oversized / malformed sidecar) and on the
//! [`SeedError::Seed`] arm (where the EVAL round-trip actually happened).

use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::watch;

use crate::checkpoint::counters::CountersSidecar;
use crate::checkpoint::store::{CheckpointError, CheckpointStore};
use crate::config::Config;
use crate::error::ServiceError;
use crate::observability::metrics::SharedMetrics;
use crate::storage::counters::{CounterStore, ObjectIdCounterRepair};

/// Which call site is running the shared seeding body.
///
/// Threaded through [`seed_object_id_counter_inner`] purely for
/// observability: it selects the `phase` label on
/// `live_edit_checkpoint_counter_seed_secs` and decides whether the
/// `Absent`-arm WARN — whose text is about AOF durability at startup, a
/// question that is already settled by the time a `CreateLayer` runs — is
/// emitted at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeedTrigger {
    /// The one-shot `main.rs` startup routine (SR-5.3).
    Startup,
    /// The lazy runtime repair reached from `CreateLayer` (b.b2j).
    Repair,
}

impl SeedTrigger {
    /// `phase` label value on `live_edit_checkpoint_counter_seed_secs`.
    pub fn label(self) -> &'static str {
        match self {
            SeedTrigger::Startup => "startup",
            SeedTrigger::Repair => "repair",
        }
    }
}

/// Outcome of a single [`seed_object_id_counter`] invocation.
///
/// `Absent` — the store answered that no sidecar exists: normally because
/// none has ever been written, the pre-first-checkpoint startup path; the
/// caller's readiness gate may decide whether this is fatal (AOF-off
/// durability mode) or benign (AOF-on durability mode). That answer may be
/// stale for up to the storage cache TTL (see
/// `CheckpointStore::get_counters_sidecar`), and this seed takes it at its
/// word; that residual is carried as t1.tv3.gk defer-12.
///
/// `Seeded` — the sidecar existed, parsed, and the conditional Lua
/// `seed_object_id_if_below` advanced `le:counter:object_id` from a value
/// strictly below `seed` up to `seed` itself. `new_counter` equals `seed`.
///
/// `NoOp` — the sidecar existed and parsed, but `le:counter:object_id`
/// was already at-or-above `seed` (a previous replica's writes survived,
/// or some other replica won the startup race first). The counter is
/// unchanged; `existing` is its observed value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeedOutcome {
    /// The configured checkpoint adapter reported no counters-sidecar blob.
    /// Pre-first-checkpoint state, or a stale "absent" (see the enum doc).
    Absent,
    /// Sidecar present and parsed; the Lua `SEED_OBJECT_ID_IF_BELOW`
    /// advanced the counter from a value below `seed` up to `seed`.
    Seeded {
        /// Computed seed = `sidecar.max_object_id_observed + safety_margin`
        /// (saturating-add).
        seed: u64,
        /// Post-script counter value (equal to `seed` on this arm).
        new_counter: u64,
    },
    /// Sidecar present and parsed; the counter was already at-or-above
    /// `seed`, so the conditional Lua left it untouched.
    NoOp {
        /// Existing counter value, returned unchanged by the Lua script.
        existing: u64,
    },
}

/// Failure modes of [`seed_object_id_counter`].
///
/// `Store` — the adapter fetch returned a classified [`CheckpointError`].
/// Histogram is NOT observed (the fetch itself failed; the duration sample
/// would describe adapter-error latency, not the seeding routine's work).
///
/// `ParseFailed` — the sidecar bytes were present but
/// `serde_json::from_slice::<CountersSidecar>` failed. The routine
/// deliberately does NOT advance the counter on this arm — the operator
/// signal is the readiness gate not flipping, surfaced via this `Err` from
/// the caller in the main.rs-wiring Subtask.
///
/// `Seed` — the [`CounterStore::seed_object_id_if_below`] EVAL itself
/// failed (Valkey transport / Lua reply parse).
#[derive(Debug, thiserror::Error)]
pub enum SeedError {
    /// Adapter-side bytes fetch failed.
    #[error("checkpoint store error during counters sidecar fetch: {0}")]
    Store(#[from] CheckpointError),
    /// Sidecar bytes present but `serde_json::from_slice::<CountersSidecar>`
    /// failed. The counter is NOT advanced on this arm — see the variant
    /// docstring.
    #[error("counters sidecar JSON parse failure: {0}")]
    ParseFailed(serde_json::Error),
    /// `CounterStore::seed_object_id_if_below` EVAL failed.
    #[error("counter store error while seeding: {0}")]
    Seed(#[from] ServiceError),
}

/// RAII guard that observes the `live_edit_checkpoint_counter_seed_secs`
/// histogram on drop iff `armed` is `true`. Set `armed = false` on the
/// [`SeedError::Store`] exit path; every other exit path leaves the guard
/// armed and the histogram fires on scope exit.
///
/// `phase` is [`SeedTrigger::label`] — the label that keeps startup samples
/// and runtime-repair samples in separate series.
struct ObserveOnDrop<'m> {
    metrics: &'m SharedMetrics,
    start: Instant,
    phase: &'static str,
    armed: bool,
}

impl Drop for ObserveOnDrop<'_> {
    fn drop(&mut self) {
        if self.armed {
            self.metrics
                .observe_checkpoint_counter_seed(self.start.elapsed().as_secs_f64(), self.phase);
        }
    }
}

/// Run the SDD SR-5.3 startup counter-seed recovery routine.
///
/// Fetches the counters-sidecar blob via
/// [`CheckpointStore::get_counters_sidecar`], parses it into a
/// [`CountersSidecar`], computes `seed = max_object_id_observed +
/// safety_margin` (saturating-add — an absurdly large sidecar value
/// downgrades to `Seeded { seed: u64::MAX, ... }` and preserves the
/// no-collision invariant rather than overflowing), and asks
/// [`CounterStore::seed_object_id_if_below`] to conditionally advance
/// `le:counter:object_id` past that value.
///
/// The routine observes `live_edit_checkpoint_counter_seed_secs` on every
/// exit path except [`SeedError::Store`] (the fetch itself failed; the
/// duration is not meaningful).
///
/// See the module-level docstring for the rationale on every design
/// choice — error variants, histogram-observation discipline, the
/// `Seeded` vs `NoOp` distinction made at the call site of the conditional
/// Lua script.
pub async fn seed_object_id_counter(
    store: Arc<dyn CheckpointStore>,
    counters: Arc<CounterStore>,
    safety_margin: u64,
    metrics: SharedMetrics,
) -> Result<SeedOutcome, SeedError> {
    seed_object_id_counter_inner(
        store,
        counters,
        safety_margin,
        metrics,
        SeedTrigger::Startup,
        // No preemptive bound at startup: `seed_object_id_counter_with_retry`
        // owns the startup budget (attempt-shaped, see
        // `COUNTER_SEED_RETRY_MAX_ATTEMPTS`), and nothing is waiting on an RPC
        // clock here.
        None,
    )
    .await
}

/// The shared seeding body behind [`seed_object_id_counter`] (startup) and
/// [`CheckpointObjectIdCounterRepair`] (runtime repair).
///
/// `trigger` is observability-only — see [`SeedTrigger`].
///
/// `fetch_budget` bounds **only** the adapter read
/// ([`CheckpointStore::get_counters_sidecar`]), never the conditional Lua seed
/// that follows it. That split is load-bearing. The adapter read is a pure
/// remote read whose enforced per-call ceiling (180 s on StorageApi at the
/// defaults) is far longer than any caller waits, and cancelling it leaves no
/// partial durable state — so a preemptive timeout there is free. The seed is a
/// single-round-trip `EVAL` against Valkey that WRITES: it is already bounded
/// by `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`, and wrapping it in a second,
/// caller-derived timeout would let a budget sized for the adapter abandon a
/// write mid-flight — which does not cancel it server-side, so the caller would
/// report a failed repair against a counter that in fact got seeded. An elapse
/// lands on [`SeedError::Store`], the same arm a genuinely unreachable adapter
/// takes, so no caller needs a third error shape.
pub(crate) async fn seed_object_id_counter_inner(
    store: Arc<dyn CheckpointStore>,
    counters: Arc<CounterStore>,
    safety_margin: u64,
    metrics: SharedMetrics,
    trigger: SeedTrigger,
    fetch_budget: Option<Duration>,
) -> Result<SeedOutcome, SeedError> {
    let mut observer = ObserveOnDrop {
        metrics: &metrics,
        start: Instant::now(),
        phase: trigger.label(),
        armed: true,
    };

    let fetch = match fetch_budget {
        None => store.get_counters_sidecar().await,
        Some(budget) => match tokio::time::timeout(budget, store.get_counters_sidecar()).await {
            Ok(result) => result,
            Err(_elapsed) => {
                tracing::warn!(
                    phase = trigger.label(),
                    budget_ms = budget.as_millis() as u64,
                    "counter-seed abandoned its counters-sidecar fetch at the configured fetch \
                     budget; the adapter's own per-call ceiling is far longer than any caller \
                     waits"
                );
                Err(CheckpointError::Transient {
                    source: format!(
                        "get_counters_sidecar exceeded the {}ms counter-seed fetch budget",
                        budget.as_millis()
                    )
                    .into(),
                })
            }
        },
    };

    let bytes = match fetch {
        Ok(opt) => opt,
        Err(e) => {
            // Adapter fetch failed — duration sample would describe
            // adapter-error latency, not seeding work. Disarm the guard.
            observer.armed = false;
            return Err(SeedError::Store(e));
        }
    };

    let Some(bytes) = bytes else {
        // Phase-appropriate, because the two phases are asking different
        // questions. At startup an absent sidecar means "nothing durable has
        // ever recorded a counter, so durability must be coming from AOF" —
        // the readiness-gate framing SR-5.3 wants. At repair time that
        // question is long settled and the operative verdict ("no durable
        // checkpoint claims any object id, so re-establishing at 0 is safe")
        // is logged by `CheckpointObjectIdCounterRepair`'s own `Absent` arm at
        // the level its evidence earns. Emitting the startup text here too
        // would put a false AOF claim in front of an operator on every repair.
        if trigger == SeedTrigger::Startup {
            tracing::warn!("counters sidecar absent; assuming AOF is providing durability");
        }
        // b.b2j: establish `le:counter:object_id` at 0 so the key exists from
        // first boot. `next_object_id` fails closed on an absent counter, and
        // that verdict is only unambiguous ("the keyspace was lost") if the
        // key is otherwise guaranteed to exist. Nothing in the service ever
        // deletes it, so this one `SET NX` is what makes absence mean loss.
        //
        // `NX` keeps a racing peer's real seed intact — it is never stomped
        // back to 0. A failure here is logged and swallowed rather than
        // propagated: this arm's `Ok(SeedOutcome::Absent)` is what flips the
        // SR-5.4 readiness gate, and the `Seed` error arm is not retried by
        // `seed_object_id_counter_with_retry`, so propagating would wedge
        // readiness permanently on a transient blip. The lazy repair path
        // (`ObjectIdCounterRepair`) reaches exactly this same
        // initialize-to-zero conclusion on the first CreateLayer, so nothing
        // is lost by deferring.
        //
        // The swallow is startup-scoped in effect: under `SeedTrigger::Repair`
        // the caller re-runs the same `SET NX` and DOES propagate a failure,
        // because "the counter key exists" is that path's postcondition rather
        // than a deferrable nicety.
        if let Err(e) = counters.initialize_object_id_counter_if_absent().await {
            tracing::error!(
                error = %e,
                "counters sidecar absent and establishing le:counter:object_id at 0 failed; \
                 the first CreateLayer will repair it lazily"
            );
        }
        return Ok(SeedOutcome::Absent);
    };

    let sidecar: CountersSidecar = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!(
                error = %e,
                "counters sidecar present but JSON parse failed; cannot seed counter — operator must investigate"
            );
            return Err(SeedError::ParseFailed(e));
        }
    };

    // `saturating_add` — an absurdly large sidecar value downgrades a u64
    // overflow to `Seeded { seed: u64::MAX, ... }` rather than wrapping
    // and producing a tiny seed that would collide with the live counter.
    let seed = sidecar.max_object_id_observed.saturating_add(safety_margin);

    let new_counter = counters.seed_object_id_if_below(seed).await?;

    // The Lua script returns `seed` iff it advanced (current was strictly
    // below seed); otherwise it returns the unchanged current value. The
    // two arms are distinguished by `new_counter == seed`: equality means
    // we advanced (the routine raced the counter from <seed up to seed);
    // inequality means the counter was already at-or-above seed.
    //
    // The boundary case `new_counter == seed` AND the counter was already
    // exactly at `seed` before this call (a NoOp masquerading as a
    // Seeded) is harmless: the post-script state is identical either way,
    // and operator-facing semantics — "counter is now at-or-above
    // sidecar + safety_margin" — hold in both interpretations. Logging
    // and dashboards never need to distinguish these two; the routine
    // returns `Seeded` for consistency with the "advanced" reading.
    if new_counter == seed {
        Ok(SeedOutcome::Seeded { seed, new_counter })
    } else {
        Ok(SeedOutcome::NoOp {
            existing: new_counter,
        })
    }
}

/// Default maximum attempts for [`seed_object_id_counter_with_retry`].
///
/// # The startup bound, corrected (b.gg7)
///
/// This used to read "`30 × 3 s ≈ 90 s` — comfortably inside the Helm
/// per-release `--wait` timeout (5 min)". That counted only the *sleeps*. Each
/// attempt also makes an adapter call (`get_counters_sidecar`), and on the
/// StorageApi adapter that call is bounded — enforced since b.gg7 — at
/// `LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS × MAX_RETRY_ATTEMPTS ×
/// AUTH_ATTEMPTS`, i.e. 180 s at the defaults. The real worst case is therefore
/// `30 × (180 s + 3 s) ≈ 91 minutes`, not 90 seconds — an order of magnitude
/// past the 5-minute `--wait`, which would time the release out while readiness
/// legitimately waited on this gate.
///
/// The common case is unaffected: the failure this retry exists for is a
/// sidecar that has not written its first token yet, which surfaces as a fast
/// 401/`None`, not as a stalled request — so the observed startup wait really is
/// the sleep sum. The corrected number is the *tail*, reached only when the
/// storage backend is itself slow or browning out.
///
/// Only the doc is corrected here. Re-sizing the attempt budget (or making it
/// deadline-shaped rather than attempt-shaped, so the bound is a wall-clock
/// number an operator can align with `--wait`) is tracked as its own ticket,
/// b.8m9 — deliberately not folded into b.gg7, which bounds calls rather than
/// re-tuning the routines that make them.
pub const COUNTER_SEED_RETRY_MAX_ATTEMPTS: u32 = 30;

/// Fixed delay between [`seed_object_id_counter_with_retry`] attempts.
pub const COUNTER_SEED_RETRY_DELAY: Duration = Duration::from_secs(3);

/// Bounded-retry wrapper around [`seed_object_id_counter`] for the startup
/// service-token race (SDD §14.6 / SR-5.3).
///
/// The `StorageApi` checkpoint adapter authenticates its counters-sidecar
/// fetch with a Bearer token written to disk by a credential-provisioner
/// sidecar. At pod startup the sidecar has typically not written the first
/// token yet, so the fetch fails for the first few seconds and surfaces as
/// [`SeedError::Store`]. Because the seed routine is a one-shot readiness
/// gate whose only documented recovery was a liveness restart — and the
/// liveness probe targets the bare gRPC server (already `SERVING`), not the
/// `LiveService` readiness gate — a one-shot failure wedged readiness at
/// `NOT_SERVING` permanently and timed out `helm upgrade --wait`. This
/// wrapper retries the `Store` arm with a bounded budget so the seed
/// completes once the sidecar token lands.
///
/// Only [`SeedError::Store`] is retried (the adapter fetch — which is where
/// the transient token-unavailable failure occurs; the token-file-missing
/// case is classified `Permanent` by the adapter, so this wrapper retries
/// the `Store` arm regardless of the inner `CheckpointError` classification
/// rather than keying off transient-vs-permanent). [`SeedError::ParseFailed`]
/// (corrupt sidecar) is genuinely permanent and returns immediately;
/// [`SeedError::Seed`] (counter-store EVAL) returns immediately because the
/// Valkey-dependency readiness check already covers Valkey health. The seed
/// routine is idempotent — `get_counters_sidecar` is a read and
/// `seed_object_id_if_below` is a conditional `SET` — so re-running across
/// attempts is safe.
pub async fn seed_object_id_counter_with_retry(
    store: Arc<dyn CheckpointStore>,
    counters: Arc<CounterStore>,
    safety_margin: u64,
    metrics: SharedMetrics,
    max_attempts: u32,
    retry_delay: Duration,
) -> Result<SeedOutcome, SeedError> {
    let mut attempt: u32 = 1;
    loop {
        match seed_object_id_counter(
            Arc::clone(&store),
            Arc::clone(&counters),
            safety_margin,
            metrics.clone(),
        )
        .await
        {
            Ok(outcome) => return Ok(outcome),
            Err(SeedError::Store(e)) if attempt < max_attempts => {
                tracing::warn!(
                    attempt,
                    max_attempts,
                    error = %e,
                    "counter-seed counters-sidecar fetch failed (likely the service-identity sidecar has not written the token file yet); retrying after delay"
                );
                attempt += 1;
                tokio::time::sleep(retry_delay).await;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Outcome of one detached repair attempt, as seen by the callers waiting on
/// it.
///
/// `Ok(())` means the counter key exists and allocation may proceed. `Err` is
/// the `UNAVAILABLE` each waiter returns ([`AttemptFailure`]); the attempt
/// itself has already recorded the metric and the ERROR log, so its message
/// is a caller-side description, not the diagnosis.
type AttemptVerdict = Result<(), AttemptFailure>;

/// A failed repair attempt, as each waiter answers it.
///
/// Clonable (a message rather than a `ServiceError`) because every waiter on
/// the same attempt receives its own copy through a `watch` channel. It
/// carries no retry hint: the repair runs at a `CreateLayer`'s commit stage,
/// where no refusal carries one — a store write refusal of a repair write
/// answers the plain `create_commit` `UNAVAILABLE`
/// (`storage::client::store_write_refusal`), so this message is all there is
/// to keep.
#[derive(Debug, Clone)]
struct AttemptFailure {
    /// The operator-facing message of the `UNAVAILABLE`.
    message: String,
}

impl AttemptFailure {
    fn new(message: String) -> Self {
        Self { message }
    }

    /// The `UNAVAILABLE` a waiter on this attempt answers.
    fn into_service_error(self) -> ServiceError {
        ServiceError::Unavailable(self.message)
    }
}

/// Reads how much of the calling RPC's deadline is left, or `None` when no
/// deadline is in scope (a deadline-exempt streaming RPC, a background task, a
/// direct unit-test call).
///
/// Injected as a closure rather than read directly from
/// `crate::grpc::deadline` **because of the module dependency rules in
/// CLAUDE.md**: `checkpoint → {domain, storage, auth}`, and `grpc` is not on
/// that list — reading the task-local here would add a `checkpoint → grpc`
/// edge, and it would be a *backwards* one (`grpc → checkpoint` already
/// exists). The closure keeps the dependency where it belongs: `main.rs`, the
/// composition root, hands in `crate::grpc::deadline::remaining` and this
/// module names nothing outside its allowed set.
pub type RemainingDeadlineFn = Box<dyn Fn() -> Option<Duration> + Send + Sync>;

/// How [`CheckpointObjectIdCounterRepair`] sizes each caller's wait.
enum WaitBudget {
    /// A value fixed at construction — `Config::object_id_counter_repair_wait_budget`
    /// in production wiring before b.hka (f), and a hand-picked duration in
    /// tests that drive the give-up path directly.
    Fixed(Duration),
    /// Re-derived per call from what is actually left of the calling RPC's
    /// deadline. See
    /// [`CheckpointObjectIdCounterRepair::with_deadline_aware_wait_budget`].
    DeadlineAware {
        config: Arc<Config>,
        remaining: RemainingDeadlineFn,
    },
}

/// Production [`ObjectIdCounterRepair`]: re-seed `le:counter:object_id` from
/// the checkpoint-store counters sidecar when a `CreateLayer` finds the
/// counter gone (b.b2j).
///
/// # Why a runtime repair exists at all
///
/// The startup seed is one-shot, but the failure it protects against is not.
/// A Valkey restart that loses the keyspace does **not** restart the pods: the
/// liveness probe targets the bare gRPC health service, which is `SERVING`
/// from startup until shutdown, and readiness recovers the moment Valkey
/// answers pings again — against an empty keyspace just as happily as a full
/// one. So a surviving process can be asked to allocate an object id against a
/// counter that no longer exists, and a bare `INCR` there returns `1`, colliding
/// with ids that surviving checkpoint blobs still claim.
///
/// # Why the repair is lazy rather than a watchdog
///
/// A periodic `EXISTS` poll races the first post-wipe `CreateLayer` and buys
/// nothing over repairing at the allocation itself, which is the only moment
/// the answer is load-bearing. And a readiness gate cannot help: readiness is
/// advisory — it removes a pod from load-balancer rotation but neither closes
/// established gRPC streams nor stops a client that resolved the pod directly —
/// so it structurally cannot enforce "must not mint a colliding id". Only the
/// allocator can, which is why [`CounterStore::next_object_id`] fails closed
/// and this hook is what un-blocks it.
///
/// # Why the repair runs on a detached task
///
/// `CreateLayer` is **not** in `EXEMPT_FROM_DEADLINE`
/// (`src/grpc/metrics_layer.rs` exempts exactly `JoinSession` and
/// `SubscribeReadLayer`), so the tower deadline layer bounds the whole handler
/// future at `LIVE_EDIT_RPC_DEADLINE_MS` and **drops** it on elapse. An earlier
/// revision ran the sidecar fetch inline under
/// `Config::checkpoint_recovery_fetch_budget`, which is `>= rpc_deadline_ms` by
/// construction — so on a browning-out checkpoint store the handler future was
/// always dropped first and the inline budget could never elapse. Dropping the
/// future cancelled the fetch mid-flight, which meant: no outcome metric, no
/// ERROR log, a bare `DEADLINE_EXCEEDED` at the client, and the next request
/// restarting the same doomed fetch from zero. The failure mode the repair
/// exists to make legible was the one it reported nothing about.
///
/// So the attempt runs on a `tokio::spawn`ed task that no RPC cancellation can
/// reach, and [`Self::repair_object_id_counter`] merely *waits* on it under a
/// budget strictly below the RPC deadline
/// (`Config::object_id_counter_repair_wait_budget`, or — once
/// [`Self::with_deadline_aware_wait_budget`] is wired —
/// `Config::object_id_counter_repair_wait_budget_within` measured against what
/// is actually left of *this* request's deadline). A caller that gives up
/// returns `UNAVAILABLE` — honest and retriable — while the attempt keeps
/// running to completion and records its own outcome. The next retry then joins
/// that still-running attempt rather than starting a second one (see
/// *Singleflight* below for what "still running" means, and for the one shape —
/// an attempt that died without reporting — where the next caller must spawn
/// instead).
///
/// The task is deliberately NOT tracked for shutdown: its work is a read plus
/// an idempotent conditional `SET`, both individually bounded (adapter fetch
/// budget; `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`), and a process exiting
/// mid-attempt loses only the log line — the next process's startup seed
/// reaches the same conclusion.
///
/// # Singleflight
///
/// A `CreateLayer` burst arriving right after the wipe would otherwise issue
/// one sidecar fetch per request. The private `inflight` field holds the
/// `watch` receiver
/// of the attempt currently running; a caller that finds one there joins it
/// instead of spawning, so one keyspace-loss event costs exactly one sidecar
/// fetch no matter how many requests race into it. A caller arriving after the
/// attempt finished starts a fresh one — and that attempt's first act is the
/// `current_object_id() > 0` re-check, which short-circuits to
/// `already_repaired` without touching the adapter.
///
/// A slot is joinable only while its attempt is genuinely *alive*, which is two
/// conditions rather than one: its verdict is still `None` **and** its sender
/// still exists. The second half is what recovers from an attempt that died
/// without reporting. `tokio::spawn` swallows a panic in the spawned future, so
/// a panicking `run_repair_attempt` drops `tx` with no send: the slot's value
/// stays `None` forever. Under a `None`-only join predicate that dead receiver
/// would be handed to every subsequent caller, no further attempt would ever be
/// spawned for the process lifetime, and `CreateLayer` would stay wedged behind
/// a repair that can no longer run — with nothing to restart the pod, for the
/// same reason the runtime repair exists at all (liveness targets the bare gRPC
/// health service). So a caller that finds a dead slot replaces it with a fresh
/// attempt. Waiters already joined to the dead attempt are not stranded: their
/// `rx.changed()` resolves `Err`, which is the [`ObjectIdCounterRepair`] impl's
/// "ended without reporting" arm — `failed` + ERROR + `UNAVAILABLE`, the
/// correct verdict for a caller whose attempt died.
///
/// # Log levels
///
/// ERROR is reserved for outcomes that carry *evidence* of keyspace loss —
/// `repaired_seeded` / `repaired_noop` (a counters sidecar exists, so durable
/// checkpoint state claims object ids the missing counter was protecting) and
/// `failed`. `repaired_initialized` (no sidecar at all) and `already_repaired`
/// log at WARN: the former is the expected first-CreateLayer shape on a fresh
/// store with an adapter configured, where nothing durable exists to collide
/// with, and alarming on it trained operators to ignore the family.
/// `wait_timeout` — the caller-side give-up, which is not itself a verdict on
/// the counter — logs at WARN.
pub struct CheckpointObjectIdCounterRepair {
    store: Arc<dyn CheckpointStore>,
    counters: Arc<CounterStore>,
    safety_margin: u64,
    metrics: SharedMetrics,
    /// Bound on the sidecar fetch **inside the detached task**, where nothing
    /// is on an RPC clock. Same value and same rationale as the recovery hooks'
    /// `checkpoint_recovery_fetch_budget`: the adapter's own per-call ceiling
    /// (180 s on StorageApi at the defaults) is far longer than any operator
    /// wants a repair to sit on, and abandoning a pure read costs nothing.
    fetch_budget: Duration,
    /// Bound on how long a *caller* waits for the in-flight attempt before
    /// returning `UNAVAILABLE`. Strictly below `LIVE_EDIT_RPC_DEADLINE_MS`
    /// (`Config::object_id_counter_repair_wait_budget`) so this — not the tower
    /// deadline layer — is what ends the wait, which is the whole point: a
    /// caller that returns on its own budget leaves the attempt running and its
    /// outcome recorded, where a dropped handler future would take both with
    /// it. See [`WaitBudget`] for the two ways it is sourced.
    wait_budget: WaitBudget,
    /// The attempt currently in flight, if any. `None` before the first
    /// attempt. A receiver is joinable only while its attempt is *alive*: its
    /// value is still `None` **and** at least one sender still exists. Two
    /// other shapes can sit here and both mean "start a fresh attempt" — a
    /// receiver holding `Some(verdict)` is a *finished* attempt, and a receiver
    /// still reading `None` whose senders are all gone is a *dead* one (the
    /// spawned task panicked — `tokio::spawn` swallows that — or the runtime is
    /// tearing down).
    ///
    /// Nothing ever clears this slot from the task side: the detached attempt
    /// never reaches back here, so there is no completion race to lose. The
    /// reader decides what the slot means, and the dead-channel half of that
    /// decision is what keeps a single panicking attempt from wedging every
    /// later `CreateLayer` for the process lifetime.
    ///
    /// A `std::sync::Mutex` (never held across an `await`) rather than a
    /// `tokio::sync::Mutex`: the critical section is a slot read plus a
    /// `tokio::spawn`, and a caller cancelled while acquiring an async mutex
    /// would be one more way for a dropped `CreateLayer` future to matter.
    inflight: StdMutex<Option<watch::Receiver<Option<AttemptVerdict>>>>,
}

impl CheckpointObjectIdCounterRepair {
    pub fn new(
        store: Arc<dyn CheckpointStore>,
        counters: Arc<CounterStore>,
        safety_margin: u64,
        metrics: SharedMetrics,
        fetch_budget: Duration,
        wait_budget: Duration,
    ) -> Self {
        Self {
            store,
            counters,
            safety_margin,
            metrics,
            fetch_budget,
            wait_budget: WaitBudget::Fixed(wait_budget),
            inflight: StdMutex::new(None),
        }
    }

    /// Re-derive the caller-side wait budget on every call from what is
    /// actually left of the calling RPC's deadline, instead of using the fixed
    /// value handed to [`Self::new`] (b.hka (f)).
    ///
    /// `remaining` is normally `crate::grpc::deadline::remaining`, wired from
    /// `main.rs` — see [`RemainingDeadlineFn`] for why it arrives as a closure
    /// rather than a direct call.
    ///
    /// **This is a strict refinement, not a different policy.** The budget
    /// becomes `Config::object_id_counter_repair_wait_budget_within(remaining)`,
    /// which collapses to exactly
    /// `Config::object_id_counter_repair_wait_budget` — the value `new` was
    /// given — whenever no deadline is in scope. It can only ever shrink the
    /// wait, and it shrinks it precisely when the fixed budget would have
    /// outlasted the deadline and let the tower layer drop the handler future
    /// mid-wait (bare `DEADLINE_EXCEEDED`, no `wait_timeout` tick, nothing said
    /// about the pending repair).
    ///
    /// The `config` MUST be the one the fixed budget was derived from;
    /// disagreeing values would make the `remaining = None` path silently
    /// change behaviour. `debug_assert`ed rather than enforced, because the
    /// only production caller is the composition root that computed both.
    ///
    /// Deliberately a builder rather than a `new` parameter: the detached
    /// attempt is unaffected (it keeps `fetch_budget`, off any RPC clock), and
    /// every existing construction site keeps working unchanged and
    /// unambiguously fixed-budget.
    pub fn with_deadline_aware_wait_budget(
        mut self,
        config: Arc<Config>,
        remaining: RemainingDeadlineFn,
    ) -> Self {
        debug_assert!(
            matches!(
                self.wait_budget,
                WaitBudget::Fixed(fixed) if fixed == config.object_id_counter_repair_wait_budget()
            ),
            "with_deadline_aware_wait_budget: `config` must be the Config the fixed wait budget \
             was derived from, otherwise the no-deadline-in-scope path changes behaviour"
        );
        self.wait_budget = WaitBudget::DeadlineAware { config, remaining };
        self
    }

    /// The wait budget for the caller running right now.
    ///
    /// Under [`WaitBudget::Fixed`] this is a constant; under
    /// [`WaitBudget::DeadlineAware`] it is re-read per call, so two concurrent
    /// `CreateLayer`s joined to the SAME attempt can legitimately give up at
    /// different times — each on its own deadline. That is the intended shape:
    /// the budget describes the caller, not the attempt.
    fn wait_budget_now(&self) -> Duration {
        match &self.wait_budget {
            WaitBudget::Fixed(fixed) => *fixed,
            WaitBudget::DeadlineAware { config, remaining } => {
                config.object_id_counter_repair_wait_budget_within(remaining())
            }
        }
    }

    /// Return a receiver for the in-flight attempt, spawning one if none is
    /// running.
    ///
    /// The whole singleflight decision lives in this one critical section, held
    /// under a sync mutex with no `await` inside it, so "is an attempt running"
    /// and "start an attempt" cannot interleave.
    ///
    /// "Running" is two conditions, not one: the slot's verdict is still `None`
    /// **and** the attempt's sender is still alive.
    /// [`watch::Receiver::has_changed`] answers the second — it returns `Err`
    /// exactly when every sender has been dropped, which for this channel means
    /// the spawned task ended without sending (a panic, which `tokio::spawn`
    /// swallows, or runtime teardown). Without that half one panicking attempt
    /// would poison the slot for the process lifetime: its value stays `None`,
    /// every later caller joins the same dead receiver, no further attempt is
    /// ever spawned, and `CreateLayer` stays permanently wedged behind a repair
    /// that will never run again. A dead slot is therefore replaced here with a
    /// fresh attempt, exactly like a finished one.
    fn join_or_spawn_attempt(&self) -> watch::Receiver<Option<AttemptVerdict>> {
        let mut slot = self
            .inflight
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // A receiver whose value is still `None` AND whose sender is still
        // alive belongs to an attempt that has not reported yet — join it.
        // `has_changed()` is `Err` iff every sender has been dropped, so it is
        // what distinguishes a running attempt from one that died without
        // sending (panic swallowed by `tokio::spawn`, or runtime teardown).
        // Anything else — no attempt, a finished one, or a dead one — means
        // this caller starts the next attempt.
        if let Some(rx) = slot.as_ref() {
            if rx.borrow().is_none() && rx.has_changed().is_ok() {
                return rx.clone();
            }
        }

        let (tx, rx) = watch::channel(None);
        *slot = Some(rx.clone());

        let store = Arc::clone(&self.store);
        let counters = Arc::clone(&self.counters);
        let metrics = Arc::clone(&self.metrics);
        let safety_margin = self.safety_margin;
        let fetch_budget = self.fetch_budget;
        tokio::spawn(async move {
            let verdict =
                run_repair_attempt(store, counters, safety_margin, metrics, fetch_budget).await;
            // Send failure means every waiter went away — the attempt still
            // ran and still recorded its outcome, which is the point.
            let _ = tx.send(Some(verdict));
        });

        rx
    }
}

/// One repair attempt, run to completion on a detached task.
///
/// Every exit path increments exactly one
/// `live_edit_object_id_counter_repair_total{outcome}` cell and logs at the
/// level that outcome's evidence earns. That total accounting is the reason
/// this body is off the RPC clock: an attempt that is merely *abandoned* by its
/// caller still lands here.
async fn run_repair_attempt(
    store: Arc<dyn CheckpointStore>,
    counters: Arc<CounterStore>,
    safety_margin: u64,
    metrics: SharedMetrics,
    fetch_budget: Duration,
) -> AttemptVerdict {
    // Re-check. A peer task — or a peer *replica* — that repaired while this
    // attempt was being scheduled has already re-seeded, so the counter is back
    // above zero and there is nothing left to do. (A repair that concluded
    // "initialize at 0" leaves the counter AT zero and is not detected here;
    // re-running the sidecar read in that case is idempotent and costs one
    // adapter round trip, which is the honest trade for not tracking repair
    // epochs.)
    match counters.current_object_id().await {
        Ok(existing) if existing > 0 => {
            metrics.inc_object_id_counter_repair("already_repaired");
            tracing::warn!(
                existing,
                "object-id counter was repaired by a concurrent task while this CreateLayer \
                 waited; proceeding without a checkpoint-store read"
            );
            return Ok(());
        }
        Ok(_) => {}
        Err(e) => {
            // Reading `le:counter:object_id` is the attempt's first Valkey
            // round trip, so a failure here is almost always the backing store
            // being unreachable — the same condition that produced the absent
            // verdict upstream. Booked as `failed` like every other
            // non-completion: an attempt that returns without repairing must
            // never be invisible in the outcome vector, which is exactly what
            // this arm used to be.
            metrics.inc_object_id_counter_repair("failed");
            tracing::error!(
                error = %e,
                "object-id counter repair could not read le:counter:object_id to decide whether \
                 a peer had already repaired it; CreateLayer will fail with UNAVAILABLE rather \
                 than allocate a possibly-colliding id"
            );
            return Err(AttemptFailure::new(format!(
                "object-id counter repair could not read the counter: {e}"
            )));
        }
    }

    // WARN, not ERROR: at this point the verdict is not yet known. An absent
    // counter is keyspace loss *only* if something durable exists that could
    // collide with a fresh one, and the counters sidecar is what answers that —
    // a store with no sidecar has never checkpointed a counter, which is the
    // ordinary first-boot shape of a deployment that has an adapter configured
    // with the periodic loop still off. The sidecar read below decides, and each
    // outcome arm logs at the level its verdict earns (ERROR on evidence of
    // loss, WARN on a fresh store).
    tracing::warn!(
        "object-id counter le:counter:object_id is ABSENT at CreateLayer; refusing to \
         allocate from an empty counter (that would re-mint object ids any surviving \
         checkpoint still claims) and consulting the checkpoint counters sidecar to \
         determine whether durable state exists to protect"
    );

    // `fetch_budget` bounds ONLY the adapter read inside — the conditional Lua
    // seed that follows it runs under the Valkey command timeout, because
    // abandoning a write mid-flight does not cancel it server-side and would
    // make this attempt report a failure against a counter that got seeded. See
    // `seed_object_id_counter_inner`.
    let outcome = seed_object_id_counter_inner(
        store,
        Arc::clone(&counters),
        safety_margin,
        metrics.clone(),
        SeedTrigger::Repair,
        Some(fetch_budget),
    )
    .await;

    match outcome {
        Ok(SeedOutcome::Seeded { seed, new_counter }) => {
            metrics.inc_object_id_counter_repair("repaired_seeded");
            // ERROR: a sidecar exists, so durable checkpoint state claims
            // object ids — an absent counter alongside it is unambiguous
            // evidence the Valkey keyspace was lost underneath this process.
            tracing::error!(
                seed,
                new_counter,
                safety_margin,
                "object-id counter re-seeded from the checkpoint counters sidecar after \
                 keyspace loss; allocation resumes above every id a durable checkpoint \
                 can claim. Investigate the backing store: a restart that dropped the \
                 keyspace also dropped every live layer, session, and presence record"
            );
            Ok(())
        }
        Ok(SeedOutcome::NoOp { existing }) => {
            metrics.inc_object_id_counter_repair("repaired_noop");
            // ERROR for the same reason as the `Seeded` arm: a sidecar was
            // found, so the absence this repair observed was real data loss
            // — a peer replica simply won the race to re-seed.
            tracing::error!(
                existing,
                "object-id counter was already at or above the sidecar-derived seed by the \
                 time the repair ran (a peer replica re-seeded first) after keyspace loss. \
                 Investigate the backing store: a restart that dropped the keyspace also \
                 dropped every live layer, session, and presence record"
            );
            Ok(())
        }
        Ok(SeedOutcome::Absent) => {
            // The store answered that no sidecar exists, taken to mean none has
            // ever been written, so there is no durable object id anywhere
            // that a fresh counter could collide with. The answer can be a
            // stale cached "absent" (`SeedOutcome::Absent`'s doc), the residual
            // carried as t1.tv3.gk defer-12.
            // `seed_object_id_counter_inner`'s own `Absent` arm already ran the
            // `SET NX 0` best-effort; repeat it here so this attempt's
            // postcondition ("the counter key exists on Ok") does not depend on
            // that arm's swallowed error handling.
            if let Err(e) = counters.initialize_object_id_counter_if_absent().await {
                // Same accounting argument as the `current_object_id` arm
                // above: an attempt that returns without leaving a counter
                // behind is a failed repair and must be booked as one, even
                // though the checkpoint store answered perfectly.
                metrics.inc_object_id_counter_repair("failed");
                tracing::error!(
                    error = %e,
                    "object-id counter repair found no counters sidecar but could not establish \
                     le:counter:object_id at 0; CreateLayer will fail with UNAVAILABLE rather \
                     than allocate a possibly-colliding id"
                );
                return Err(AttemptFailure::new(format!(
                    "object-id counter repair could not establish the counter: {e}"
                )));
            }
            metrics.inc_object_id_counter_repair("repaired_initialized");
            // WARN, not ERROR — and deliberately NOT a keyspace-loss claim.
            // Reaching here means the checkpoint store holds no counters
            // sidecar at all, so nothing durable has ever recorded an object
            // id and there is nothing a counter starting at 0 could collide
            // with. That is the expected once-per-deployment shape when an
            // adapter is configured but the periodic counters-sync loop has
            // not run (the `LIVE_EDIT_CHECKPOINT_ENABLED=false` +
            // adapter-configured combination): the first CreateLayer finds
            // no counter and no sidecar, and establishes the key here.
            // Logging that at ERROR alongside "the keyspace was lost" was a
            // false alarm on a perfectly healthy fresh store.
            tracing::warn!(
                "object-id counter established at 0 against a checkpoint store that holds \
                 no counters sidecar; no durable checkpoint claims any object id, so \
                 allocation can safely start from 1. Expected once on a fresh deployment. \
                 If this store is NOT fresh — a counters sidecar has been written before — \
                 treat it as checkpoint-store data loss and investigate"
            );
            Ok(())
        }
        Err(e) => {
            metrics.inc_object_id_counter_repair("failed");
            tracing::error!(
                error = %e,
                "object-id counter repair failed; CreateLayer will fail with UNAVAILABLE \
                 rather than allocate a possibly-colliding id"
            );
            Err(AttemptFailure::new(format!(
                "object-id counter repair failed: {e}"
            )))
        }
    }
}

#[async_trait]
impl ObjectIdCounterRepair for CheckpointObjectIdCounterRepair {
    async fn repair_object_id_counter(&self) -> Result<(), ServiceError> {
        let mut rx = self.join_or_spawn_attempt();

        // Resolved once, here, so the value that bounds the wait is the same
        // one the give-up log and error message report (b.hka (f) — under
        // `WaitBudget::DeadlineAware` it is re-read per call and would
        // otherwise have shrunk by the time the timeout fires).
        let wait_budget = self.wait_budget_now();

        let waited = tokio::time::timeout(wait_budget, async {
            loop {
                // Scoped so the `Ref` guard is dropped before the `await`.
                let seen = rx.borrow_and_update().clone();
                if let Some(verdict) = seen {
                    return Some(verdict);
                }
                if rx.changed().await.is_err() {
                    // The sender was dropped without reporting — only reachable
                    // if the detached task panicked or the runtime is shutting
                    // down. Neither is a repaired counter. The dead attempt does
                    // not outlive this call: `join_or_spawn_attempt` treats a
                    // sender-less slot as replaceable, so the next `CreateLayer`
                    // spawns a fresh attempt rather than re-joining this one.
                    return None;
                }
            }
        })
        .await;

        match waited {
            Ok(Some(Ok(()))) => Ok(()),
            Ok(Some(Err(failure))) => {
                // The attempt already incremented its own outcome cell and
                // logged the diagnosis; this caller just carries the verdict
                // out as a retriable status.
                Err(failure.into_service_error())
            }
            Ok(None) => {
                self.metrics.inc_object_id_counter_repair("failed");
                tracing::error!(
                    "object-id counter repair task ended without reporting an outcome (panic or \
                     runtime shutdown); CreateLayer will fail with UNAVAILABLE rather than \
                     allocate a possibly-colliding id"
                );
                Err(ServiceError::Unavailable(
                    "object-id counter repair task ended without reporting an outcome".to_string(),
                ))
            }
            Err(_elapsed) => {
                // The caller gives up; the attempt does NOT. It keeps running
                // on its detached task and will record its own outcome, and the
                // client's retry will join that same attempt rather than
                // starting a second one. WARN rather than ERROR: this is a
                // statement about how long the wait budget is, not a verdict on
                // the counter — the verdict is still pending.
                self.metrics.inc_object_id_counter_repair("wait_timeout");
                tracing::warn!(
                    // The EFFECTIVE budget this caller actually waited, which
                    // under deadline-aware wiring is ⅔ of what was left of its
                    // own deadline — not the configured ⅔ of
                    // `LIVE_EDIT_RPC_DEADLINE_MS`. Reporting the configured
                    // value would send an operator hunting a timeout that
                    // never ran.
                    wait_budget_ms = wait_budget.as_millis() as u64,
                    "CreateLayer stopped waiting on the in-flight object-id counter repair at its \
                     wait budget and will fail with UNAVAILABLE; the repair keeps running on its \
                     own task and will log its outcome, and a retry will join it rather than \
                     starting a second attempt"
                );
                Err(ServiceError::Unavailable(format!(
                    "object-id counter repair did not complete within the {}ms wait budget; \
                     it is still running — retry",
                    wait_budget.as_millis()
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit coverage for the caller-side wait budget (b.hka (f)).
    //!
    //! Scoped deliberately narrowly: the arithmetic itself belongs to
    //! `Config::object_id_counter_repair_wait_budget_within` and is pinned in
    //! `src/config.rs`'s tests, and the composed payoff — a `CreateLayer` on a
    //! short client deadline returning a deliberate `UNAVAILABLE` instead of a
    //! bare `DEADLINE_EXCEEDED` — is pinned in
    //! `tests/integration/checkpoint/counter_repair_test.rs`. What is left, and
    //! what only these tests cover, is the WIRING: which arm of [`WaitBudget`]
    //! is in force, and whether the injected closure is consulted per call.
    //!
    //! A `TestValkey` appears only because `CounterStore` holds a real
    //! `ValkeyClient`; no test here issues a command against it.
    use super::*;
    use crate::checkpoint::adapters::InMemoryCheckpointStore;
    use crate::config::PodMemorySource;
    use crate::observability::metrics::metrics_scrape_only;
    use crate::storage::test_support::TestValkey;

    /// The RPC deadline every case below is derived against, and the fixed
    /// budget that follows from it (⅔ × 30 s).
    const RPC_DEADLINE_MS: &str = "30000";
    const FIXED_BUDGET: Duration = Duration::from_secs(20);

    /// A `Config` parsed from a minimal env, so the budget under test is the
    /// one `from_env` actually produces rather than a hand-set field.
    fn test_config() -> Arc<Config> {
        temp_env::with_vars(
            vec![
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                ("LIVE_EDIT_RPC_DEADLINE_MS", Some(RPC_DEADLINE_MS)),
            ],
            || {
                Arc::new(
                    Config::from_env_with(PodMemorySource::Reference)
                        .expect("the minimal test env must parse"),
                )
            },
        )
    }

    /// Construct a repair with the fixed budget `config` implies — the shape
    /// `main.rs` builds before it applies the deadline-aware builder.
    fn repair(tv: &TestValkey, config: &Config) -> CheckpointObjectIdCounterRepair {
        CheckpointObjectIdCounterRepair::new(
            Arc::new(InMemoryCheckpointStore::new()),
            Arc::new(CounterStore::new(
                tv.client().clone(),
                tv.key_schema().clone(),
            )),
            1000,
            Arc::new(metrics_scrape_only("counter-seed-unit-test")),
            Duration::from_secs(10),
            config.object_id_counter_repair_wait_budget(),
        )
    }

    /// Without the builder, the budget is the constant handed to `new` — every
    /// call, forever.
    ///
    /// This is the arm every existing construction site (tests, tools, and
    /// `main.rs` before the builder runs) is on, so it is the one that must not
    /// change: `with_deadline_aware_wait_budget` is a strict refinement, and a
    /// refinement that also moved the un-refined default would be a silent
    /// behaviour change for callers that never opted in.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wait_budget_now_is_the_constructed_constant_without_the_builder() {
        let tv = TestValkey::setup().await;
        let config = test_config();
        let repair = repair(&tv, &config);

        assert_eq!(
            repair.wait_budget_now(),
            FIXED_BUDGET,
            "an un-refined repair MUST answer the budget it was constructed with (⅔ × \
             {RPC_DEADLINE_MS}ms); anything else means the builder's arm leaked into the default"
        );
    }

    /// With the builder wired, the budget is re-derived per call from whatever
    /// the injected closure answers.
    ///
    /// Three claims in one test because they are one behaviour observed at
    /// three inputs, and separating them would only duplicate the fixture:
    ///
    /// - `None` (no ambient deadline — a streaming RPC, a background task)
    ///   collapses to exactly the fixed budget. This is what makes the wiring
    ///   in `main.rs` unconditional-and-safe.
    /// - A remainder *below* the deadline shrinks the budget to ⅔ of it. This
    ///   is the case the change exists for.
    /// - The closure is consulted on EVERY call, not memoised. Two concurrent
    ///   `CreateLayer`s can be joined to the same attempt while sitting on very
    ///   different remaining deadlines, and the budget describes the caller,
    ///   not the attempt — a cached first answer would give the second caller
    ///   the first one's deadline.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wait_budget_now_re_reads_the_injected_remaining_deadline_on_every_call() {
        let tv = TestValkey::setup().await;
        let config = test_config();

        // Mutable cell the closure reads, so successive calls can answer
        // differently — the only way to tell "re-read" from "read once".
        let answer: Arc<StdMutex<Option<Duration>>> = Arc::new(StdMutex::new(None));
        let answer_for_closure = Arc::clone(&answer);
        let repair = repair(&tv, &config).with_deadline_aware_wait_budget(
            Arc::clone(&config),
            Box::new(move || {
                *answer_for_closure
                    .lock()
                    .expect("the remaining-deadline cell must not be poisoned")
            }),
        );

        assert_eq!(
            repair.wait_budget_now(),
            FIXED_BUDGET,
            "`None` means no ambient deadline is in scope, which MUST collapse to exactly the \
             fixed budget — `main.rs` installs this wiring unconditionally, so a divergence \
             here changes the wait for every caller that never had a deadline"
        );

        *answer.lock().expect("cell must not be poisoned") = Some(Duration::from_millis(900));
        assert_eq!(
            repair.wait_budget_now(),
            Duration::from_millis(600),
            "with 900ms of the request's deadline left, the wait MUST shrink to ⅔ of THAT, not \
             stay at ⅔ of the configured {RPC_DEADLINE_MS}ms. The fixed budget would outlast \
             the remainder and the tower deadline layer would end the wait by dropping the \
             handler future — a bare DEADLINE_EXCEEDED with no `wait_timeout` tick"
        );

        *answer.lock().expect("cell must not be poisoned") = Some(Duration::from_millis(300));
        assert_eq!(
            repair.wait_budget_now(),
            Duration::from_millis(200),
            "and the closure MUST be consulted again rather than memoised: two callers joined \
             to one in-flight attempt legitimately give up at different times, each on its own \
             remaining deadline"
        );
    }
}
