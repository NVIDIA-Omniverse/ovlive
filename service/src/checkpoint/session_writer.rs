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

//! Async-off-hot-path session-checkpoint writer (SR-2.7, Task `t2.fgj.7v.p9`).
//!
//! [`SessionCheckpointWriter`] is the dispatcher the gRPC lifecycle handlers
//! (CreateSession / CreateLayer / DeleteLayer / DeleteSession) call to persist
//! a fresh session-checkpoint after a successful lifecycle write. The dispatch
//! API is **synchronous** (`fn`, not `async fn`) and **fire-and-forget**: the
//! lifecycle RPC has already returned success to the client by the time these
//! are reached, so the spawned task runs to completion (or failure) on its own
//! and the RPC's tail latency is unaffected.
//!
//! ## Ordering: why dispatch is serialized per session (b.443)
//!
//! Session checkpoints are the ROOT OF REACHABILITY for disaster recovery.
//! `apply_checkpoint`'s SR-2.4 guard refuses to restore any layer that has no
//! live `layer_name → object_id` mapping, and the session applier's SET loop is
//! the only thing that re-creates those mappings after state loss. A durable
//! session blob that is missing one mapping therefore makes that layer's own
//! checkpoint permanently unrestorable — silently, with the layer blob sitting
//! right there.
//!
//! `put_session_checkpoint` is last-writer-wins to one fixed path per session,
//! so unordered dispatch produced exactly that outcome: two lifecycle events
//! spawn two independent tasks, the older build's PUT lands last, and the newer
//! layer's mapping is gone from the durable blob until some future lifecycle
//! event happens to rewrite it. Two mechanisms close it:
//!
//! 1. **Per-session serialized dispatch with coalescing** (this module). At
//!    most one in-flight adapter operation per `session_id`; further enqueues
//!    set a pending op that the running task picks up when it finishes. A
//!    pending DELETE supersedes a pending WRITE and never the reverse — a
//!    session that has been deleted must not be resurrected by a rewrite.
//!    Coalescing also collapses a burst of N lifecycle events into ~2 adapter
//!    calls instead of N.
//! 2. **A Valkey-side publish watermark** (`session_checkpoint_publish.lua`).
//!    Serialization inside one process is not enough: there is no session
//!    affinity, so concurrent lifecycle RPCs for one session routinely land on
//!    DIFFERENT replicas. Every build mints a monotone stamp atomically with
//!    its read, and after a successful adapter call the writer publishes that
//!    stamp, paired with the key's incarnation token. A refused publish means
//!    either that this operation clobbered a strictly fresher one — the writer
//!    repairs by rebuilding (which mints a higher stamp, so the repair wins) or
//!    re-issuing the DELETE — or that the coordination key was lost and
//!    re-created between the mint and the publish, so the stamp is not
//!    comparable at all and the writer repairs by rebuilding regardless of
//!    kind.
//!
//! The adapter cannot arbitrate this itself: `CheckpointMeta` is discarded by
//! LocalDisk and ignored by StorageApi, SR-1.5 forbids parsing the body, and
//! the object surface has no CAS. Valkey is the one thing every replica shares.
//!
//! ## Why a shared `TaskTracker`
//!
//! The spawned task is owned by a [`TaskTracker`] so graceful shutdown can
//! `wait().await` for in-flight session-checkpoint writes inside the existing
//! `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS` budget. The tracker is plumbed in by the
//! parent Task's wire-up Subtask (`t2.fgj.7v.p9.6f` / Subtask E); this module
//! is concerned only with spawning into it.
//!
//! **The dispatch task drains its pending work in an internal loop rather than
//! re-spawning, and that is load-bearing for shutdown.** `main.rs` calls
//! `close()` and then `wait()` on the tracker; a task spawned after `close()`
//! is not covered by a `wait()` that has already returned, so a re-spawn shape
//! could drop a queued write on the floor at shutdown. Looping inside the
//! already-tracked task keeps every coalesced operation inside the drain.
//!
//! ## Disabled adapters short-circuit at the enqueue
//!
//! The dispatcher is constructed with an `enabled` flag (adapter is not
//! `Disabled`). When it is `false`, both enqueue entry points return
//! immediately: no build, no Valkey traffic, no slot bookkeeping. Previously a
//! `Disabled` deployment paid a full multi-round-trip session read on every
//! lifecycle event only to throw the payload away at the adapter's
//! `NotImplemented`.
//!
//! ## Failure-mode contract
//!
//! - `Ok(_)` from the adapter put, **once its publish has resolved** →
//!   `inc_checkpoint_write("success", "session")`. The tick sits behind the
//!   publish deliberately: an operation whose bytes landed but whose ordering
//!   was never established has not succeeded in any sense an operator cares
//!   about, and reporting it as `success` would hide the one outcome where a
//!   durable clobber goes unrepaired. A publish that resolves to "you
//!   clobbered something fresher" still counts as success here — the
//!   operation completed and returned a verdict; the verdict itself is
//!   carried by `checkpoint_session_publish_superseded_total`.
//! - `Ok(None)` from [`build_session_checkpoint`] (session vanished between
//!   the lifecycle event and our read) → **no metric tick, no log**. The
//!   absence is not a write attempt; the metric reads as zero traffic for
//!   that session, which is the truth.
//! - `Err(CheckpointError::NotImplemented { .. })` — the `Disabled` adapter's
//!   `NoopCheckpointStore` returns this on every method (see
//!   `src/checkpoint/builder.rs`). This is the "checkpointing is off"
//!   pathway; **no metric tick, no log**. Surfacing it as `error` would tick
//!   the counter on every lifecycle event in a `Disabled` deployment — an
//!   operator-visible noise floor that does not represent a real failure.
//!   No `Config` plumbing is required: the `Disabled` selector wires
//!   `NoopCheckpointStore` at startup and the `NotImplemented` return is
//!   already the gate.
//! - Every other classified failure (`Transient`, `Permanent`, `NotFound`,
//!   `Conflict`) → `inc_checkpoint_write("error", "session")` plus a WARN log
//!   that carries `is_transient` / `is_permanent` flags so an operator can
//!   triage transient-vs-permanent from logs even though the metric folds
//!   them into a single `error` cell. Adapter-phase failures tick once per
//!   failed attempt; a landed operation whose publish could not be confirmed
//!   ticks `error` exactly once, with a WARN that says the bytes are durable
//!   but unordered.
//!
//! ### Asymmetry with Epic 1's `result="skipped"` cell
//!
//! Epic 1's periodic layer-checkpoint loop ticks `result="skipped"` when no
//! commits have happened since the last checkpoint. Session writes have no
//! equivalent skip case: every lifecycle event represents a real state
//! mutation, so emitting `skipped` would have no meaningful semantics. The
//! closest analog — `NotImplemented` on `Disabled` — is correctly silent
//! rather than `skipped` because a `Disabled` deployment is opted out of
//! checkpoint observability entirely.
//!
//! ## Logging redaction
//!
//! - `session_id` is logged verbatim (matches `SessionStore` posture).
//! - Bytes and the deserialized `SessionCheckpoint` content are **never**
//!   logged, on either the success or error path. Lifecycle payloads can
//!   carry user-supplied strings; the metric and the error variant carry
//!   enough triage signal without exposing them.

use std::sync::{Arc, LazyLock};
use std::time::Duration;

use dashmap::DashMap;
use fred::types::scripts::Script;
use fred::types::Value;
use tokio_util::task::TaskTracker;

use crate::checkpoint::now_unix_ms;
use crate::checkpoint::session::{
    build_session_checkpoint, mint_session_checkpoint_stamp, SessionBuild,
    SESSION_CHECKPOINT_STATE_TTL_MS,
};
use crate::checkpoint::store::{CheckpointError, CheckpointMeta, CheckpointStore};
use crate::observability::metrics::SharedMetrics;
use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use crate::storage::keys::KeySchema;

/// Compiled Lua script backing the publish watermark (b.443). See the
/// script's own header for the protocol and the soundness argument; the
/// handle is declared next to its only caller, mirroring
/// `SESSION_CHECKPOINT_READ_SCRIPT` in `src/checkpoint/session.rs`.
static SESSION_CHECKPOINT_PUBLISH_SCRIPT: LazyLock<Script> = LazyLock::new(|| {
    Script::from_lua(include_str!(
        "../storage/scripts/session_checkpoint_publish.lua"
    ))
});

/// # The retry ladder
///
/// One coalesced operation runs in two phases with **separate** retry
/// budgets, and the boundary between them is a correctness boundary, not a
/// tuning knob:
///
/// - **Adapter phase** — mint/build and the `put` / `delete` call. Nothing is
///   durable yet, so a transient failure retries the WHOLE operation and
///   re-mints a fresh stamp. Each failed attempt ticks
///   `checkpoint_writes_total{result="error"}`.
/// - **Publish phase** — begins the instant the adapter call LANDS. From
///   there the operation is never re-run: only [`publish_stamp`] is retried,
///   with the SAME stamp the landed bytes were built under.
///
/// Retrying the whole operation after a landed adapter call is what makes
/// the resurrection bug possible: if a `DeleteSession` commits between the
/// two attempts, attempt 2's build sees an absent session, returns `Ok(None)`
/// and settles silently — so the blob that attempt 1 wrote ON TOP of the
/// tombstone is never detected, never repaired, and
/// `SessionRecoveryRunner` re-materializes the deleted session on the next
/// cold lookup. Carrying the stamp forward removes that path entirely: the
/// publish either confirms the landed bytes are newest, or reports the
/// clobbered operation's kind so it can be repaired.
///
/// Re-publishing an identical stamp is safe by design — the script's
/// equal-stamp/same-kind arm accepts it — which matters because the common
/// publish failure is a timeout on a call the server actually applied.
///
/// # Attempts per adapter-phase operation.
///
/// Small on purpose. This is a best-effort DR path whose failure mode is "the
/// durable blob is one lifecycle event stale", and the retries run inside the
/// shutdown drain budget, so a long backoff ladder would trade a real
/// shutdown-latency cost for a marginal durability gain. Before b.443 there
/// was no retry at all — a single transient Valkey or adapter blip silently
/// dropped the write.
const MAX_ADAPTER_ATTEMPTS: u32 = 3;

/// Attempts to confirm a landed operation's stamp before giving up on the
/// ordering.
///
/// Deliberately its own constant rather than a reuse of
/// [`MAX_ADAPTER_ATTEMPTS`]: the two phases fail for different reasons and
/// have different stakes. An adapter-phase give-up leaves nothing durable; a
/// publish-phase give-up leaves durable bytes whose ordering was never
/// established, which is the more serious exposure — so if these ever need
/// separate tuning, the publish budget is the one to raise.
const MAX_PUBLISH_ATTEMPTS: u32 = 3;

/// Delay before the second attempt in either phase; quadrupled for the
/// third. Total added latency for a fully-failing phase is ~250 ms,
/// comfortably inside `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS` (default 5 s).
const RETRY_INITIAL_DELAY: Duration = Duration::from_millis(50);

/// Maximum publish-supersession repairs one dispatch task will perform in a
/// single chain.
///
/// Each repair mints a strictly higher stamp, so under any bounded amount of
/// concurrency the chain terminates on its own; the budget exists to stop a
/// pathological write storm on one session from pinning a task indefinitely.
/// Exhausting it ticks `result="exhausted"` and leaves the blob stale until
/// the next lifecycle event.
///
/// The budget is per CHAIN, not per task: it is restored whenever an
/// operation ends the chain — settled, abandoned, or repair-exhausted alike.
/// A long-lived task on a hot session must not be permanently barred from
/// repairing because it once hit a three-deep supersession chain, which
/// would turn a self-healing condition into a continuously-firing alert.
const MAX_PUBLISH_REPAIRS: u32 = 3;

/// The two adapter operations this dispatcher performs, and the unit of
/// coalescing in a [`SessionWriteSlot`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingOp {
    /// Rebuild the session checkpoint from live state and PUT it.
    Write,
    /// Tombstone the session checkpoint (DeleteSession).
    Delete,
}

impl PendingOp {
    /// Wire spelling used for the `pub_kind` field and the supersession
    /// metric's `result` label. MUST match the script's vocabulary.
    fn kind(self) -> &'static str {
        match self {
            PendingOp::Write => "write",
            PendingOp::Delete => "delete",
        }
    }

    fn from_kind(kind: &str) -> Option<Self> {
        match kind {
            "write" => Some(PendingOp::Write),
            "delete" => Some(PendingOp::Delete),
            _ => None,
        }
    }

    /// Coalesce two operations queued for the same session.
    ///
    /// **A DELETE absorbs a WRITE; a WRITE never absorbs a DELETE.** The
    /// asymmetry is the point: dropping a queued rewrite in favour of a
    /// tombstone loses nothing (the session is gone), while the reverse
    /// resurrects a deleted session's checkpoint — which recovery would then
    /// happily restore.
    fn merge(self, other: Self) -> Self {
        if self == PendingOp::Delete || other == PendingOp::Delete {
            PendingOp::Delete
        } else {
            PendingOp::Write
        }
    }
}

/// Sentinel `session_checkpoint_publish.lua` answers when the operation's
/// incarnation is not the one currently on `le:{sid}:ckpt`, and the matching
/// `result` label on `checkpoint_session_publish_superseded_total`.
///
/// Deliberately OUTSIDE the `pub_kind` vocabulary (`"write"` / `"delete"`) so
/// a decoder that only knows the two kinds cannot mistake it for one — it is
/// matched by name before [`PendingOp::from_kind`] is consulted.
///
/// An **alias**, not a second literal: the spelling is owned by
/// [`crate::observability::metrics::CHECKPOINT_SESSION_PUBLISH_STALE_INCARNATION`],
/// which is also what
/// [`crate::observability::metrics::CHECKPOINT_SESSION_PUBLISH_SUPERSEDED_RESULTS`]
/// seeds the metric cell from, so decode and label cannot drift apart.
pub(crate) const STALE_INCARNATION_RESULT: &str =
    crate::observability::metrics::CHECKPOINT_SESSION_PUBLISH_STALE_INCARNATION;

/// Why `session_checkpoint_publish.lua` refused to record this operation.
///
/// Both arms mean the same thing operationally — the bytes this operation made
/// durable are not the ones that should be there, and a repair is owed — but
/// they are distinct conditions with distinct repairs and distinct metric
/// cells, so the caller is told which.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PublishRefusal {
    /// A strictly fresher operation of this kind already landed and this one
    /// clobbered it. Repair by re-running that kind.
    Fresher(PendingOp),
    /// The `le:{sid}:ckpt` key this operation's stamp was minted against is
    /// gone or has been re-created under a new incarnation, so the stamp is
    /// not comparable to the live watermark at all — nothing was written and
    /// nothing can be concluded about what else has landed.
    StaleIncarnation,
}

/// Per-`session_id` dispatch state. Lives in a [`DashMap`] keyed by session
/// id; entries are created on first enqueue and retired by the dispatch task
/// when it finds no pending work (see `dispatch_loop`).
#[derive(Debug, Default)]
struct SessionWriteSlot {
    /// True while a dispatch task owns this session. Enqueues observing it
    /// only set `pending` — the running task will pick the work up.
    in_flight: bool,
    /// The next operation to run, coalesced via [`PendingOp::merge`].
    pending: Option<PendingOp>,
}

/// Everything a dispatch task needs, cloned once per spawn.
#[derive(Clone)]
struct DispatchCtx {
    store: Arc<dyn CheckpointStore>,
    client: ValkeyClient,
    keys: KeySchema,
    metrics: SharedMetrics,
    creator_replica_id: String,
    slots: Arc<DashMap<String, SessionWriteSlot>>,
}

/// Result of one coalesced operation, after both phases of its retry ladder.
enum OpOutcome {
    /// Landed (or there was nothing to do) and the publish was accepted.
    Settled,
    /// Landed, but the publish was refused; the clobbered state must be
    /// repaired by re-running `repair`. `result` is the metric label naming
    /// WHY (the fresher operation's kind, or
    /// [`STALE_INCARNATION_RESULT`]) — the two are not always the same
    /// string, so they travel separately.
    Superseded {
        repair: PendingOp,
        result: &'static str,
    },
    /// Gave up in either phase. Already logged and metered at the point of
    /// failure, because the two phases carry different exposures and the
    /// distinction belongs in the log, not in a variant the caller would
    /// treat identically.
    Abandoned,
}

// ───────────────────────────────────────────────────────────────────────────
// Publish fault-injection seam (test-only)
// ───────────────────────────────────────────────────────────────────────────

/// What an injected publish fault makes [`publish_stamp`] do.
///
/// **Test-only.** Production code never names this enum; release builds
/// compile without `test-checkpoint-fault-injection`, leaving the whole seam
/// absent from the binary. Mirrors the `ApplyStep` /
/// `APPLY_STEP_FAULT_INJECTORS` precedent in `src/checkpoint/session.rs`.
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
#[derive(Debug)]
pub enum PublishFault {
    /// Fail the publish with this classified error, without issuing the
    /// `EVALSHA`. A queue of `Transient`s followed by nothing exercises the
    /// retry-then-succeed path; a queue of at least `MAX_PUBLISH_ATTEMPTS`
    /// forces publish-phase exhaustion.
    Fail(CheckpointError),
    /// Answer as though the script had refused this operation, without issuing
    /// the `EVALSHA`. Drives the repair path deterministically; queue at least
    /// `MAX_PUBLISH_REPAIRS + 1` to force repair-budget exhaustion.
    ///
    /// The string is the script's raw refusal reply and is decoded by the same
    /// `classify_refusal` the live `EVALSHA` path uses — so a test driven
    /// through this seam exercises the real decoder, including its WARNs, and
    /// not a copy of it. The full refusal vocabulary is therefore reachable
    /// here: the two `pub_kind` spellings (`"write"` / `"delete"`) yield
    /// `PublishRefusal::Fresher`, the `STALE_INCARNATION_RESULT` sentinel
    /// (`"stale_incarnation"`) yields `PublishRefusal::StaleIncarnation`, and
    /// anything else exercises the unrecognised-kind arm.
    Refuse(String),
}

/// Process-global queue consulted at the top of [`publish_stamp`]. Each call
/// pops one entry; an empty queue lets the real publish run.
///
/// The publish path is a free function, so the seam must be process-global
/// rather than instance-bound — same reasoning as
/// `APPLY_STEP_FAULT_INJECTORS`. Under `cargo test`'s in-process parallelism
/// a test that installs faults must serialize itself against peers that also
/// dispatch session checkpoints; `cargo nextest`'s process-per-test isolation
/// (canonical here) makes that moot.
///
/// **Test-only.**
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
static PUBLISH_FAULT_INJECTOR: std::sync::OnceLock<
    std::sync::Mutex<std::collections::VecDeque<PublishFault>>,
> = std::sync::OnceLock::new();

/// Install a queue of publish faults, consulted one per [`publish_stamp`]
/// call. REPLACES any previous queue.
///
/// **Test-only.** Gated behind
/// `#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]`.
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
pub fn set_publish_fault_injector_for_test(queue: std::collections::VecDeque<PublishFault>) {
    let cell = PUBLISH_FAULT_INJECTOR
        .get_or_init(|| std::sync::Mutex::new(std::collections::VecDeque::new()));
    *cell.lock().expect("PUBLISH_FAULT_INJECTOR lock poisoned") = queue;
}

/// Drain the publish-fault queue. Recommended at the top of any test that
/// installs one, so a previous test's leftovers cannot bleed across.
///
/// **Test-only.**
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
pub fn clear_publish_fault_injector_for_test() {
    if let Some(cell) = PUBLISH_FAULT_INJECTOR.get() {
        cell.lock()
            .expect("PUBLISH_FAULT_INJECTOR lock poisoned")
            .clear();
    }
}

/// Pop one injected publish fault, if any.
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
fn pop_publish_fault() -> Option<PublishFault> {
    PUBLISH_FAULT_INJECTOR
        .get()?
        .lock()
        .expect("PUBLISH_FAULT_INJECTOR lock poisoned")
        .pop_front()
}

/// Async-off-hot-path session-checkpoint writer.
///
/// One instance is constructed at service startup (Subtask E wire-up) and
/// shared across the gRPC lifecycle handlers via `Arc<SessionCheckpointWriter>`.
/// Cheaply cloneable — every field is itself cheaply cloneable / `Arc`-wrapped
/// — so callers may stash a clone alongside their own service state without
/// thinking about it. (We do not derive `Clone` here because callers do not
/// need it today: the wire-up Subtask hands the value through an `Arc`.)
pub struct SessionCheckpointWriter {
    /// Adapter selected by `LIVE_EDIT_CHECKPOINT_ADAPTER`. May be the
    /// `NoopCheckpointStore` for `Disabled`, in which case every spawned
    /// task short-circuits silently on `NotImplemented`.
    store: Arc<dyn CheckpointStore>,
    /// Read-side Valkey handle handed to [`build_session_checkpoint`] so it
    /// can issue the four-family read pattern (`HGETALL` / `SMEMBERS` /
    /// fan-out GET / `SISMEMBER`) against the data pool.
    client: ValkeyClient,
    /// Same key schema the rest of the service uses.
    keys: KeySchema,
    /// Shared tracker that owns the spawned background task. Graceful
    /// shutdown's `wait().await` drains every in-flight session-checkpoint
    /// write inside the configured budget.
    tracker: TaskTracker,
    /// Metrics handle for `inc_checkpoint_write(outcome, "session")`.
    metrics: Arc<crate::observability::metrics::Metrics>,
    /// Service / replica identifier persisted as `CheckpointMeta::creator_replica_id`.
    creator_replica_id: String,
    /// `false` when the selected adapter is `Disabled`. Both enqueue entry
    /// points short-circuit on it before doing any work — see the
    /// module-level "Disabled adapters short-circuit at the enqueue" note.
    enabled: bool,
    /// Per-`session_id` dispatch state (b.443). Shared with every spawned
    /// dispatch task; entries exist only while a session has work in flight
    /// or pending, so the map's size tracks concurrent session activity
    /// rather than the number of sessions ever seen.
    slots: Arc<DashMap<String, SessionWriteSlot>>,
}

impl SessionCheckpointWriter {
    /// Construct a writer. The `creator_replica_id` is typically
    /// `LIVE_EDIT_SERVICE_LABEL` (falling back to `$HOSTNAME` / `unknown`)
    /// and is stamped into every persisted [`CheckpointMeta`] so an
    /// operator can attribute a checkpoint blob to its origin replica.
    ///
    /// `enabled` gates the whole dispatch path and should be
    /// `!matches!(config.checkpoint_adapter, CheckpointAdapter::Disabled)` —
    /// the same form the sibling adapter-touching wirings in `main.rs` use.
    /// It is deliberately NOT `config.checkpoint_enabled`: that flag gates
    /// the periodic layer-checkpoint loop, and — as with the SR-3.2
    /// ext-value recovery wiring — once a real adapter is configured, session
    /// checkpoints are written even on a replica that runs no checkpoint
    /// loop.
    pub fn new(
        store: Arc<dyn CheckpointStore>,
        client: ValkeyClient,
        keys: KeySchema,
        tracker: TaskTracker,
        metrics: SharedMetrics,
        creator_replica_id: String,
        enabled: bool,
    ) -> Self {
        Self {
            store,
            client,
            keys,
            tracker,
            metrics,
            creator_replica_id,
            enabled,
            slots: Arc::new(DashMap::new()),
        }
    }

    /// Enqueue a session-checkpoint write for `session_id`.
    ///
    /// **Synchronous** — returns as soon as the operation has been recorded
    /// on the session's dispatch slot. The lifecycle-event RPC returns to the
    /// client without blocking on the build / serialize / put chain.
    ///
    /// The dispatch task, for each coalesced write:
    ///
    /// 1. Calls [`build_session_checkpoint`], which mints the build stamp
    ///    atomically with the read; on `Ok(None)` (session gone), exits
    ///    silently with no metric tick.
    /// 2. Serializes via `SessionCheckpoint::to_bytes`.
    /// 3. Constructs [`CheckpointMeta`] with the configured
    ///    `creator_replica_id`, `now_unix_ms()` as `written_at_unix_ms`,
    ///    `snap_seq = None`, and `session_id` populated; every other meta
    ///    field is left at `Default` (session checkpoints share the
    ///    `CheckpointMeta` shape with layer checkpoints but only consult
    ///    the four named fields, per `CheckpointMeta` doc-comments).
    /// 4. Calls [`CheckpointStore::put_session_checkpoint`] and dispatches
    ///    the outcome through [`handle_put_outcome`].
    /// 5. Publishes the build stamp, and repairs if the publish is refused.
    pub fn enqueue_session_checkpoint_write(&self, session_id: String) {
        self.enqueue(session_id, PendingOp::Write);
    }

    /// Enqueue a session-checkpoint delete for `session_id`.
    ///
    /// **Synchronous** — same fire-and-forget shape as
    /// [`Self::enqueue_session_checkpoint_write`]. The lifecycle handler for
    /// `DeleteSession` calls this so a tombstone is durably persisted
    /// off the hot path; failure ticks `result="error"` and a WARN log but
    /// is swallowed. Nothing retries an abandoned delete, so the checkpoint
    /// stays in place. `DeleteSession`'s `le:{sid}:sess:deleted` marker (7-day
    /// TTL) makes session recovery refuse it until the marker expires or is
    /// lost with the keyspace; after that, a lookup of the session id could
    /// restore it.
    ///
    /// A delete queued behind a pending write REPLACES it — see
    /// [`PendingOp::merge`].
    pub fn enqueue_session_checkpoint_delete(&self, session_id: String) {
        self.enqueue(session_id, PendingOp::Delete);
    }

    /// Record `op` on `session_id`'s dispatch slot, spawning a dispatch task
    /// only if none is already running for that session.
    fn enqueue(&self, session_id: String, op: PendingOp) {
        if !self.enabled {
            // Disabled adapter: no build, no Valkey traffic, no slot entry.
            return;
        }

        // Scope the DashMap guard tightly: it is a shard write-lock, and the
        // spawn below can start the task on another worker thread
        // immediately.
        let spawn_needed = {
            let mut slot = self.slots.entry(session_id.clone()).or_default();
            slot.pending = Some(match slot.pending {
                Some(existing) => existing.merge(op),
                None => op,
            });
            if slot.in_flight {
                false
            } else {
                slot.in_flight = true;
                true
            }
        };
        if !spawn_needed {
            return;
        }

        let ctx = DispatchCtx {
            store: Arc::clone(&self.store),
            client: self.client.clone(),
            keys: self.keys.clone(),
            metrics: Arc::clone(&self.metrics),
            creator_replica_id: self.creator_replica_id.clone(),
            slots: Arc::clone(&self.slots),
        };
        self.tracker.spawn(dispatch_loop(ctx, session_id));
    }
}

/// RAII owner of a session's `in_flight` claim.
///
/// The dispatch task retires its slot itself on the normal path and
/// [`disarm`](Self::disarm)s this guard. An armed drop therefore means the
/// task unwound — a panic anywhere in the loop, the adapter, or a decoder.
/// Without the guard that unwind would leave `in_flight = true` with no task
/// running, and EVERY subsequent enqueue for that session would silently
/// record a pending op that nobody ever services: permanent per-session
/// dispatch death from one panic, where the pre-b.443 shape lost a single
/// write.
///
/// On an armed drop the slot is retired if idle, or has its claim released
/// if an op is pending, so the next enqueue spawns a fresh task and picks up
/// the work. We deliberately do NOT re-spawn from `Drop`: a panic loop would
/// respawn forever, and a spawn during shutdown unwinding would land after
/// the tracker's `close()` where nothing drains it.
struct SlotGuard {
    slots: Arc<DashMap<String, SessionWriteSlot>>,
    session_id: String,
    armed: bool,
}

impl SlotGuard {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // Retire the slot outright when nothing is queued; otherwise just
        // release the claim. Both arms take the shard lock, so an enqueuer
        // either observes `in_flight = false` (and spawns) or has already
        // written its pending op under the same lock (and the `remove_if`
        // predicate sees it).
        if self
            .slots
            .remove_if(&self.session_id, |_, slot| slot.pending.is_none())
            .is_none()
        {
            if let Some(mut slot) = self.slots.get_mut(&self.session_id) {
                slot.in_flight = false;
            }
        }
        tracing::error!(
            session_id = %self.session_id,
            "session-checkpoint dispatch task unwound; released its per-session claim so a \
             later lifecycle event can resume dispatch. Any operation queued at the moment of \
             the unwind is deferred to that event"
        );
    }
}

/// Drain every operation queued for `session_id`, one at a time, then retire
/// the slot.
///
/// Looping here rather than re-spawning per operation is load-bearing for
/// graceful shutdown — see the module-level `TaskTracker` note.
async fn dispatch_loop(ctx: DispatchCtx, session_id: String) {
    let mut guard = SlotGuard {
        slots: Arc::clone(&ctx.slots),
        session_id: session_id.clone(),
        armed: true,
    };
    let mut repairs_remaining = MAX_PUBLISH_REPAIRS;

    loop {
        // Take the pending op, if any. The DashMap guard is confined to this
        // statement so it is never held across an await.
        let next = match ctx.slots.get_mut(&session_id) {
            Some(mut slot) => slot.pending.take(),
            None => {
                // No slot at all. Unreachable: this task holds the entry's
                // `in_flight` claim and is the only thing that retires it.
                // Returning (rather than falling through to the retire arm)
                // closes a spin: `remove_if` on an absent key also answers
                // `None`, so a `continue` here would loop without ever
                // awaiting and peg a worker thread.
                debug_assert!(
                    false,
                    "dispatch slot for {session_id} vanished while a dispatch task held it"
                );
                guard.disarm();
                return;
            }
        };

        let Some(op) = next else {
            // Retire the slot — but only if nothing was enqueued between the
            // take above and now. `remove_if` evaluates its predicate under
            // the same shard lock `enqueue` takes, so the check and the
            // removal are atomic with respect to an enqueuer: either we
            // remove (and the next enqueue starts a fresh task) or we see the
            // new pending op and loop. There is no window where an enqueue
            // lands on a slot nobody will service.
            if ctx
                .slots
                .remove_if(&session_id, |_, slot| slot.pending.is_none())
                .is_some()
            {
                guard.disarm();
                return;
            }
            continue;
        };

        match run_operation(&ctx, &session_id, op).await {
            OpOutcome::Settled | OpOutcome::Abandoned => {
                repairs_remaining = MAX_PUBLISH_REPAIRS;
            }
            OpOutcome::Superseded { repair, result } => {
                if repairs_remaining == 0 {
                    ctx.metrics
                        .inc_checkpoint_session_publish_superseded("exhausted");
                    tracing::warn!(
                        session_id = %session_id,
                        refusal = result,
                        clobbered_kind = repair.kind(),
                        max_repairs = MAX_PUBLISH_REPAIRS,
                        "session-checkpoint publish superseded repeatedly; repair budget \
                         exhausted. The durable session checkpoint may be stale until the next \
                         lifecycle event on this session"
                    );
                    // Exhaustion ends the chain exactly as settling does.
                    // Leaving the budget at zero would bar this task from
                    // ever repairing again — on a hot session that means an
                    // `exhausted` tick on every subsequent supersession,
                    // i.e. a permanently-firing alert for a condition that
                    // heals itself the moment the write storm subsides.
                    repairs_remaining = MAX_PUBLISH_REPAIRS;
                    continue;
                }
                repairs_remaining -= 1;
                ctx.metrics
                    .inc_checkpoint_session_publish_superseded(result);
                tracing::warn!(
                    session_id = %session_id,
                    refusal = result,
                    clobbered_kind = repair.kind(),
                    repairs_remaining,
                    "session-checkpoint operation landed but its publish was refused; repairing"
                );
                let mut slot = ctx.slots.entry(session_id.clone()).or_default();
                slot.pending = Some(match slot.pending {
                    Some(existing) => existing.merge(repair),
                    None => repair,
                });
            }
        }
    }
}

/// Run one coalesced operation through both phases of the retry ladder — see
/// the ladder's design note above [`MAX_ADAPTER_ATTEMPTS`].
///
/// # `checkpoint_writes_total` accounting
///
/// Exactly one cell per operation once it lands:
///
/// - adapter attempt failed → `result="error"`, once per failed attempt
///   (the counter's unit is the adapter call, and a retried operation
///   genuinely made several);
/// - adapter landed, publish resolved (accepted OR refused) →
///   `result="success"`, once — the ticks moved behind the publish so a
///   landed-but-unconfirmed operation cannot report success;
/// - adapter landed, publish exhausted → `result="error"`, once. The bytes
///   ARE durable, so this cell is a deliberate choice between two half-true
///   options: the operation did not complete (its ordering was never
///   established, and any clobber it caused will go unrepaired), and "did
///   not complete" is exactly what this cell has always meant. The
///   accompanying WARN carries the distinction an operator needs — durable
///   but unordered, versus not written at all.
/// - session gone before the build (`Ok(None)`) → no tick at all. Absence is
///   not a write attempt.
async fn run_operation(ctx: &DispatchCtx, session_id: &str, op: PendingOp) -> OpOutcome {
    // ── Phase 1: adapter. Nothing durable yet, so retry the whole op. ──────
    let mut delay = RETRY_INITIAL_DELAY;
    let mut attempt: u32 = 0;
    let (stamp, incarnation) = loop {
        attempt += 1;
        let err = match run_adapter_phase(ctx, session_id, op).await {
            // Session vanished between the lifecycle event and our read — no
            // metric, no log, no publish. Nothing landed at the adapter.
            Ok(None) => return OpOutcome::Settled,
            Ok(Some(landed)) => break landed,
            Err(err) => err,
        };

        handle_put_outcome(&ctx.metrics, session_id, Err(&err));

        // Only `Transient` is worth another attempt. `Permanent` (corrupt
        // keyspace, serialization failure), `NotFound`, `Conflict` and the
        // `Disabled` adapter's `NotImplemented` all reproduce identically.
        if !err.is_transient() || attempt >= MAX_ADAPTER_ATTEMPTS {
            return OpOutcome::Abandoned;
        }
        tokio::time::sleep(delay).await;
        delay *= 4;
    };

    // ── Phase 2: publish. The operation LANDED; it is never re-run. ────────
    // Retries carry the same stamp AND the same incarnation forward, which is
    // both necessary (a rebuild could observe a since-deleted session and
    // settle silently, hiding a resurrection) and safe (the script accepts a
    // re-published identical stamp under the same incarnation, so a timeout on
    // a call the server actually applied resolves cleanly on the next
    // attempt). Re-minting the incarnation between attempts would break that:
    // a key lost mid-ladder would be re-adopted rather than detected.
    let mut delay = RETRY_INITIAL_DELAY;
    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        let err = match publish_stamp(ctx, session_id, stamp, &incarnation, op).await {
            Ok(None) => {
                handle_put_outcome(&ctx.metrics, session_id, Ok(()));
                return OpOutcome::Settled;
            }
            Ok(Some(refusal)) => {
                // The operation itself completed — it landed and its
                // ordering verdict came back. The verdict happens to be a
                // refusal, which the caller repairs and the supersession
                // counter records separately.
                handle_put_outcome(&ctx.metrics, session_id, Ok(()));
                return match refusal {
                    PublishRefusal::Fresher(fresher) => OpOutcome::Superseded {
                        repair: fresher,
                        result: fresher.kind(),
                    },
                    // Repair as a WRITE for BOTH kinds, deliberately — the
                    // same "arm that cannot destroy information" rationale
                    // the script uses for a missing `pub_kind`. A stale
                    // incarnation tells us nothing about what else has
                    // landed, so a DELETE repair could tombstone a live
                    // session's checkpoint. On a live session the rebuild
                    // mints a fresh stamp against the NEW incarnation and
                    // overwrites the stale bytes; on a session that no
                    // longer exists `build_session_checkpoint` returns
                    // `Ok(None)` and the chain exits, leaving exactly the
                    // bytes DR wants.
                    PublishRefusal::StaleIncarnation => OpOutcome::Superseded {
                        repair: PendingOp::Write,
                        result: STALE_INCARNATION_RESULT,
                    },
                };
            }
            Err(err) => err,
        };

        if !err.is_transient() || attempt >= MAX_PUBLISH_ATTEMPTS {
            ctx.metrics.inc_checkpoint_write("error", "session");
            tracing::warn!(
                session_id = %session_id,
                kind = op.kind(),
                stamp,
                attempts = attempt,
                error = %err,
                "session-checkpoint operation landed at the adapter but its publish could not \
                 be confirmed; the durable blob's ordering is unestablished. If it clobbered a \
                 fresher operation, that clobber will go unrepaired until the next lifecycle \
                 event on this session. NOT repairing blind: a rebuild could resurrect a \
                 session deleted in the meantime, and a delete could destroy a live checkpoint"
            );
            return OpOutcome::Abandoned;
        }
        // Intermediate publish failures are logged at debug and do NOT tick
        // the counter — the operation is still in progress, and ticking per
        // attempt here would report multiple outcomes for one landed write.
        tracing::debug!(
            session_id = %session_id,
            kind = op.kind(),
            stamp,
            attempt,
            error = %err,
            "session-checkpoint publish failed transiently; retrying with the same stamp"
        );
        tokio::time::sleep(delay).await;
        delay *= 4;
    }
}

/// Run the adapter phase of one operation: build/mint, then the adapter
/// call.
///
/// `Ok(Some((stamp, incarnation)))` — the adapter call landed under `stamp`,
/// which is only comparable within `incarnation` (see
/// [`SessionBuild::incarnation`]).
/// `Ok(None)` — nothing to do (the session is gone); nothing landed.
/// `Err(_)` — the attempt failed before anything landed.
async fn run_adapter_phase(
    ctx: &DispatchCtx,
    session_id: &str,
    op: PendingOp,
) -> Result<Option<(u64, String)>, CheckpointError> {
    let landed = match op {
        PendingOp::Write => {
            let Some(SessionBuild {
                checkpoint,
                stamp,
                incarnation,
            }) = build_session_checkpoint(&ctx.client, &ctx.keys, session_id).await?
            else {
                // Session vanished between the lifecycle event and our read —
                // no metric, no log. Absence is not a write attempt, and it is
                // deliberately NOT escalated to a delete: an empty
                // `le:{sid}:sess` is also what selective key loss looks like,
                // and tombstoning on it would destroy the artifact recovery
                // needs. No publish either — nothing landed at the adapter.
                return Ok(None);
            };
            let bytes = checkpoint.to_bytes()?;
            let meta = CheckpointMeta {
                creator_replica_id: ctx.creator_replica_id.clone(),
                written_at_unix_ms: now_unix_ms(),
                snap_seq: None,
                session_id: session_id.to_string(),
                ..CheckpointMeta::default()
            };
            ctx.store
                .put_session_checkpoint(session_id, bytes, meta)
                .await?;
            (stamp, incarnation)
        }
        PendingOp::Delete => {
            // Mint BEFORE the adapter call, exactly as the write path does:
            // the stamp has to order this delete against writes whose builds
            // are already in flight elsewhere.
            let (stamp, incarnation) =
                mint_session_checkpoint_stamp(&ctx.client, &ctx.keys, session_id).await?;
            ctx.store.delete_session_checkpoint(session_id).await?;
            (stamp, incarnation)
        }
    };

    Ok(Some(landed))
}

/// Decode one refusal string from `session_checkpoint_publish.lua` into a
/// [`PublishRefusal`], emitting the arm's WARN where one is owed.
///
/// **The single decision point for the refusal vocabulary**, called by the live
/// `EVALSHA` decode *and* by the [`PublishFault::Refuse`] injection seam. Those
/// were two separate `match`es in different branch orders until b.hka: a test
/// driven through the seam exercised the seam's copy, so a drift in the live
/// decoder — a renamed sentinel, a reordered arm — was structurally invisible
/// to it. One function means the seam can no longer agree with a decoder that
/// is wrong.
///
/// Branch order is load-bearing and matches [`STALE_INCARNATION_RESULT`]'s
/// contract: the sentinel is matched by name FIRST, before
/// [`PendingOp::from_kind`] is consulted, so it can never be read as a kind.
/// The residual arm — a string in neither vocabulary — means the script and
/// this decoder have diverged, and repairs as a write because rebuilding from
/// live state is the arm that cannot destroy information.
fn classify_refusal(raw: &str, session_id: &str, op: PendingOp, stamp: u64) -> PublishRefusal {
    if raw == STALE_INCARNATION_RESULT {
        // Not a supersession by a fresher operation — the key our stamp came
        // from is gone or was re-created, so the stamp is not comparable to
        // the live watermark and the script wrote nothing at all.
        tracing::warn!(
            session_id = %session_id,
            kind = op.kind(),
            stamp,
            "session-checkpoint publish refused: the le:{{sid}}:ckpt incarnation this build was \
             stamped against is gone or has been re-created (keyspace loss, failover to a lagging \
             replica, or TTL expiry). The durable blob may predate that loss; repairing by \
             rebuilding from live state"
        );
        return PublishRefusal::StaleIncarnation;
    }
    match PendingOp::from_kind(raw) {
        Some(fresher) => PublishRefusal::Fresher(fresher),
        None => {
            tracing::warn!(
                session_id = %session_id,
                pub_kind = %raw,
                "session-checkpoint publish returned an unrecognised pub_kind; \
                 repairing as a rebuild"
            );
            PublishRefusal::Fresher(PendingOp::Write)
        }
    }
}

/// Advance the publish watermark for `session_id`.
///
/// `Ok(None)` — accepted; this operation is the newest one to land.
/// `Ok(Some(refusal))` — refused; see [`PublishRefusal`].
///
/// Callers retry this with an UNCHANGED `stamp` AND an unchanged
/// `incarnation` — see the retry-ladder note above [`MAX_ADAPTER_ATTEMPTS`].
/// Idempotent under that discipline: the script's equal-stamp/same-kind arm
/// accepts a re-publish, so a timeout on a call the server actually applied
/// resolves cleanly rather than looking like a self-supersession. Re-reading
/// the incarnation between attempts would defeat the gate — the point of
/// carrying it is that it names the key lifetime the STAMP was drawn from.
async fn publish_stamp(
    ctx: &DispatchCtx,
    session_id: &str,
    stamp: u64,
    incarnation: &str,
    op: PendingOp,
) -> Result<Option<PublishRefusal>, CheckpointError> {
    #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
    match pop_publish_fault() {
        Some(PublishFault::Fail(err)) => return Err(err),
        Some(PublishFault::Refuse(kind)) => {
            return Ok(Some(classify_refusal(&kind, session_id, op, stamp)))
        }
        None => {}
    }

    let reply: Value = SESSION_CHECKPOINT_PUBLISH_SCRIPT
        .evalsha_with_reload(
            ctx.client.pool.next(),
            vec![ctx.keys.session_checkpoint_state(session_id)],
            vec![
                stamp.to_string(),
                op.kind().to_string(),
                SESSION_CHECKPOINT_STATE_TTL_MS.to_string(),
                incarnation.to_string(),
            ],
        )
        .await
        .inspect_err(|e| FredErrOrigin::client(&ctx.client, POOL_LABEL_DATA).observe_error(e))
        .map_err(|e| CheckpointError::Transient {
            source: format!("EVALSHA session_checkpoint_publish failed for {session_id}: {e}")
                .into(),
        })?;

    if reply.is_null() {
        return Ok(None);
    }

    let raw_kind = reply.as_string().unwrap_or_default();
    Ok(Some(classify_refusal(&raw_kind, session_id, op, stamp)))
}

/// Shared metric + log emission for both the write and delete dispatch
/// paths.
///
/// - `Ok(())` → `inc_checkpoint_write("success", "session")`, no log.
/// - `Err(CheckpointError::NotImplemented { .. })` → silent. The `Disabled`
///   adapter's no-op pathway is the only producer; surfacing it on the
///   `error` cell would tick the counter on every lifecycle event in
///   `Disabled` deployments and bury the real-error signal.
/// - Every other `Err(_)` (`Transient`, `Permanent`, `NotFound`,
///   `Conflict`) → `inc_checkpoint_write("error", "session")` and a WARN
///   log with the `session_id`, `is_transient` / `is_permanent` flags, and
///   the error's `Display` rendering. Bytes and `SessionCheckpoint`
///   content NEVER appear on the log.
fn handle_put_outcome(
    metrics: &SharedMetrics,
    session_id: &str,
    outcome: Result<(), &CheckpointError>,
) {
    match outcome {
        Ok(()) => {
            metrics.inc_checkpoint_write("success", "session");
        }
        Err(CheckpointError::NotImplemented { .. }) => {
            // Disabled adapter — silently drop. No metric, no log.
        }
        Err(err) => {
            metrics.inc_checkpoint_write("error", "session");
            tracing::warn!(
                session_id = %session_id,
                is_transient = err.is_transient(),
                is_permanent = err.is_permanent(),
                error = %err,
                "session-checkpoint adapter operation failed; transient failures are retried a \
                 bounded number of times, and anything still unwritten is picked up by the next \
                 lifecycle event on this session"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit + slot-level coverage for the dispatch surface.
    //!
    //! Three groups:
    //!
    //! 1. **API shape and `handle_put_outcome`'s decision matrix** — the
    //!    original smoke tests.
    //! 2. **Coalescing and slot lifecycle (b.443)** — driven through the
    //!    real `enqueue` → `dispatch_loop` path against a gate store that
    //!    lets a test hold one adapter call open. These live here rather
    //!    than in `tests/` because the unit under test (`PendingOp`, the
    //!    `SessionWriteSlot` map, the internal drain loop) is private.
    //! 3. **The publish-watermark script contract (b.443)** — each arm of
    //!    `session_checkpoint_publish.lua` driven directly through
    //!    `publish_stamp` against a live Valkey, so the script's accept /
    //!    refuse / equal-stamp behaviour is pinned independently of the
    //!    end-to-end repair tests in
    //!    `tests/integration/checkpoint/lifecycle_session_publish_watermark_test.rs`.
    //!    When a repair test fails, these say whether the script or the
    //!    dispatcher is at fault.
    //!
    //! End-to-end coverage — real lifecycle hooks, real adapters, the
    //! cross-replica repair — lives in `tests/integration/checkpoint/`.
    use super::*;
    use crate::checkpoint::adapters::InMemoryCheckpointStore;
    use crate::checkpoint::store::{CheckpointBlob, CheckpointHandle};
    use crate::observability::metrics::test_util::{render, sample_value};
    use crate::observability::metrics::{metrics_scrape_only, Metrics};
    use crate::storage::session::SessionStore;
    use crate::storage::test_support::TestValkey;
    use async_trait::async_trait;
    use bytes::Bytes;
    use fred::interfaces::{HashesInterface, KeysInterface};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use tokio::sync::oneshot;

    /// Read a `live_edit_checkpoint_writes_total{result,kind}` cell from the
    /// rendered scrape exposition. The `(result, kind)` grid is pre-initialised
    /// at zero in `Metrics::new`, so every cell is present from the first
    /// render.
    fn writes(metrics: &Metrics, result: &str, kind: &str) -> f64 {
        sample_value(
            &render(metrics),
            "live_edit_checkpoint_writes_total",
            &[("result", result), ("kind", kind)],
        )
        .unwrap_or(0.0)
    }

    /// Compile-time witness that the dispatch entry-points are NOT
    /// `async fn`. If a future change makes either one async, the
    /// coercion from `fn(&_, _) -> ()` fails to typecheck and the test
    /// (and the build) breaks here, surfacing the API-shape regression
    /// at the closest possible call site instead of at a downstream
    /// lifecycle handler.
    #[test]
    fn enqueue_entry_points_are_synchronous_fn() {
        let _w: fn(&SessionCheckpointWriter, String) =
            SessionCheckpointWriter::enqueue_session_checkpoint_write;
        let _d: fn(&SessionCheckpointWriter, String) =
            SessionCheckpointWriter::enqueue_session_checkpoint_delete;
    }

    /// Silent-on-NotImplemented branch coverage. `handle_put_outcome` is
    /// the central decision point for the metric + log emission; an
    /// explicit `NotImplemented` input MUST leave the counter at zero.
    /// We exercise it directly (rather than through the full enqueue
    /// path) so this remains a unit-scoped test that does not require a
    /// `TestValkey` harness.
    #[test]
    fn handle_put_outcome_silent_on_not_implemented() {
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only("unit-test"));

        let err = CheckpointError::NotImplemented {
            method: "put_session_checkpoint",
        };
        handle_put_outcome(&metrics, "sess-noop", Err(&err));

        // Sample the counter for every (result, kind) cell touching the
        // session kind. None of them should have ticked.
        for result in ["success", "error", "skipped"] {
            let v = writes(&metrics, result, "session");
            assert_eq!(
                v, 0.0,
                "NotImplemented must not tick checkpoint_writes_total{{result={result:?}, kind=\"session\"}}; got {v}"
            );
        }
    }

    /// Success-path branch coverage. `Ok(())` MUST tick the success cell
    /// and only that cell.
    #[test]
    fn handle_put_outcome_ticks_success_on_ok() {
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only("unit-test"));

        handle_put_outcome(&metrics, "sess-ok", Ok(()));

        let success = writes(&metrics, "success", "session");
        assert_eq!(success, 1.0, "Ok must tick the success cell exactly once");

        let error = writes(&metrics, "error", "session");
        assert_eq!(error, 0.0, "Ok must not tick the error cell");
    }

    /// Classified-failure branch coverage. Every non-`NotImplemented`
    /// variant MUST fold into the `error` cell. Cycling through the
    /// four classifications in one test keeps the cross-variant
    /// invariant visible — a future variant addition that forgets to
    /// route through `error` would have to be added here explicitly.
    #[test]
    fn handle_put_outcome_ticks_error_on_classified_failures() {
        let metrics: SharedMetrics = Arc::new(metrics_scrape_only("unit-test"));

        let cases: Vec<CheckpointError> = vec![
            CheckpointError::Transient {
                source: "io blip".into(),
            },
            CheckpointError::Permanent {
                source: "schema mismatch".into(),
            },
            CheckpointError::NotFound,
            CheckpointError::Conflict,
        ];

        for case in &cases {
            handle_put_outcome(&metrics, "sess-err", Err(case));
        }

        let error = writes(&metrics, "error", "session");
        assert_eq!(
            error,
            cases.len() as f64,
            "every classified failure must tick the error cell once"
        );

        let success = writes(&metrics, "success", "session");
        assert_eq!(
            success, 0.0,
            "classified failures must not tick the success cell"
        );
    }

    // =====================================================================
    // Group 2 — coalescing and slot lifecycle (b.443).
    // =====================================================================

    /// `PendingOp::merge`'s asymmetry, exhaustively.
    ///
    /// **A DELETE absorbs a WRITE; a WRITE never absorbs a DELETE.**
    /// Dropping a queued rewrite in favour of a tombstone loses nothing —
    /// the session is gone. The reverse resurrects a deleted session's
    /// checkpoint, which recovery would then happily restore, re-creating a
    /// session the operator deleted. All four combinations are enumerated so
    /// a future third variant cannot be added without confronting this
    /// table.
    #[test]
    fn pending_op_merge_lets_delete_absorb_write_never_the_reverse() {
        use PendingOp::{Delete, Write};
        assert_eq!(Write.merge(Write), Write);
        assert_eq!(
            Write.merge(Delete),
            Delete,
            "a DELETE queued behind a pending WRITE must win"
        );
        assert_eq!(
            Delete.merge(Write),
            Delete,
            "a WRITE queued behind a pending DELETE must NOT downgrade it — that is the \
             resurrection bug"
        );
        assert_eq!(Delete.merge(Delete), Delete);
    }

    /// The wire vocabulary is closed and round-trips.
    ///
    /// `kind()` feeds the script's `pub_kind` field AND the supersession
    /// metric's `result` label, and `from_kind` decodes the script's reply.
    /// Drift between the two halves would silently route every repair down
    /// `publish_stamp`'s unrecognised-kind fallback.
    #[test]
    fn pending_op_kind_round_trips_and_rejects_unknown_spellings() {
        for op in [PendingOp::Write, PendingOp::Delete] {
            assert_eq!(
                PendingOp::from_kind(op.kind()),
                Some(op),
                "{op:?} must round-trip through its wire spelling"
            );
        }
        assert_eq!(PendingOp::from_kind("exhausted"), None);
        assert_eq!(PendingOp::from_kind(""), None);
    }

    /// A `CheckpointStore` that counts the two session methods and can hold
    /// ONE chosen `put_session_checkpoint` open until the test releases it.
    ///
    /// Holding a call open gives the slot tests a *deterministic* in-flight
    /// window to enqueue into, instead of racing sleeps against the dispatch
    /// task.
    ///
    /// Every method the dispatcher must never call is `unreachable!()`
    /// rather than a silent no-op — the dispatcher's surface on this trait
    /// is exactly two methods, and that is worth pinning too.
    struct GateStore {
        inner: InMemoryCheckpointStore,
        gate_on_put: usize,
        /// 1-based index of a `put_session_checkpoint` call that panics
        /// INSTEAD of delegating, after any gating on the same index has
        /// been released. `usize::MAX` disables it. Drives the
        /// [`SlotGuard`] unwind tests: a panic here is the closest
        /// reachable stand-in for the real trigger (a panicking adapter or
        /// decoder), and it unwinds through exactly the frames the guard
        /// exists to protect.
        panic_on_put: usize,
        put_calls: AtomicUsize,
        delete_calls: AtomicUsize,
        reached_tx: Mutex<Option<oneshot::Sender<()>>>,
        release_rx: Mutex<Option<oneshot::Receiver<()>>>,
    }

    /// Test-side handles for the gated call.
    struct Gate {
        reached: oneshot::Receiver<()>,
        release: oneshot::Sender<()>,
    }

    impl GateStore {
        fn new(gate_on_put: usize) -> (Arc<Self>, Gate) {
            Self::with_options(gate_on_put, usize::MAX)
        }

        /// A store that panics on the `panic_on_put`th
        /// `put_session_checkpoint`, optionally parking on the same call
        /// first so the test can queue work before the unwind.
        fn with_options(gate_on_put: usize, panic_on_put: usize) -> (Arc<Self>, Gate) {
            let (reached_tx, reached) = oneshot::channel();
            let (release, release_rx) = oneshot::channel();
            let store = Arc::new(Self {
                inner: InMemoryCheckpointStore::new(),
                gate_on_put,
                panic_on_put,
                put_calls: AtomicUsize::new(0),
                delete_calls: AtomicUsize::new(0),
                reached_tx: Mutex::new(Some(reached_tx)),
                release_rx: Mutex::new(Some(release_rx)),
            });
            (store, Gate { reached, release })
        }

        fn put_calls(&self) -> usize {
            self.put_calls.load(Ordering::Acquire)
        }

        fn delete_calls(&self) -> usize {
            self.delete_calls.load(Ordering::Acquire)
        }
    }

    #[async_trait]
    impl CheckpointStore for GateStore {
        async fn put_layer_checkpoint(
            &self,
            _object_id: u64,
            _snap_seq: u64,
            _ompf: Bytes,
            _meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            unreachable!("the session dispatcher must never touch the layer-checkpoint surface")
        }

        async fn get_latest_layer_checkpoint(
            &self,
            _object_id: u64,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            unreachable!("the session dispatcher must never read layer checkpoints")
        }

        // `CheckpointStore::get_layer_checkpoint`'s MUST is to forward
        // "wherever the other layer methods are forwarded" — this fixture
        // forwards NONE of them, it asserts the session dispatcher never
        // reaches the layer surface at all. Inheriting the `NotImplemented`
        // default would return a classified error instead of tripping that
        // assertion, so the arm matches its siblings rather than the default.
        async fn get_layer_checkpoint(
            &self,
            _handle: &CheckpointHandle,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            unreachable!("the session dispatcher must never read layer checkpoints")
        }

        async fn list_layer_checkpoints(
            &self,
            _object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            unreachable!("the session dispatcher must never list layer checkpoints")
        }

        async fn delete_layer_checkpoint(
            &self,
            _handle: CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            unreachable!("the session dispatcher must never delete layer checkpoints")
        }

        async fn put_session_checkpoint(
            &self,
            session_id: &str,
            body: Bytes,
            meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            let call = self.put_calls.fetch_add(1, Ordering::AcqRel) + 1;
            if call == self.gate_on_put {
                if let Some(tx) = self
                    .reached_tx
                    .lock()
                    .expect("reached_tx mutex poisoned")
                    .take()
                {
                    let _ = tx.send(());
                }
                let rx = self
                    .release_rx
                    .lock()
                    .expect("release_rx mutex poisoned")
                    .take();
                if let Some(rx) = rx {
                    let _ = rx.await;
                }
            }
            if call == self.panic_on_put {
                panic!("GateStore: synthetic adapter panic on put_session_checkpoint call {call}");
            }
            self.inner
                .put_session_checkpoint(session_id, body, meta)
                .await
        }

        async fn get_session_checkpoint(
            &self,
            session_id: &str,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            self.inner.get_session_checkpoint(session_id).await
        }

        async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
            self.delete_calls.fetch_add(1, Ordering::AcqRel);
            self.inner.delete_session_checkpoint(session_id).await
        }

        async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
            self.inner.list_session_checkpoint_ids().await
        }

        async fn put_ext_blob(
            &self,
            _hash_hex: &str,
            _bytes: Bytes,
        ) -> Result<(), CheckpointError> {
            unreachable!("the session dispatcher must never touch ext blobs")
        }

        async fn get_ext_blob(&self, _hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
            unreachable!("the session dispatcher must never touch ext blobs")
        }

        async fn ext_blob_exists(&self, _hash_hex: &str) -> Result<bool, CheckpointError> {
            unreachable!("the session dispatcher must never touch ext blobs")
        }

        async fn delete_ext_blob(&self, _hash_hex: &str) -> Result<(), CheckpointError> {
            unreachable!("the session dispatcher must never touch ext blobs")
        }

        async fn put_counters_sidecar(&self, _body: Bytes) -> Result<(), CheckpointError> {
            unreachable!("the session dispatcher must never touch the counters sidecar")
        }

        async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
            unreachable!("the session dispatcher must never touch the counters sidecar")
        }

        async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
            unreachable!("the session dispatcher must never enumerate object ids")
        }

        async fn list_referenced_ext_hashes(
            &self,
        ) -> Result<std::collections::HashSet<String>, CheckpointError> {
            unreachable!("the session dispatcher must never enumerate ext hashes")
        }
    }

    /// Create a session plus a writer wired to `store` — everything a
    /// slot-level test needs.
    async fn slot_harness(
        tv: &TestValkey,
        store: Arc<dyn CheckpointStore>,
    ) -> (String, SessionCheckpointWriter, TaskTracker) {
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create("slot-stage", "/stage/slot", "slot-user")
            .await
            .expect("session creation must succeed");
        let tracker = TaskTracker::new();
        let writer = SessionCheckpointWriter::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            tracker.clone(),
            Arc::new(metrics_scrape_only("slot-test")),
            "slot-replica".to_string(),
            true,
        );
        (session_id, writer, tracker)
    }

    /// `close()` + `wait()` under a timeout. A hang is a dispatcher bug —
    /// most plausibly an unbounded repair chain — not a test-only flake.
    async fn drain(tracker: &TaskTracker) {
        tracker.close();
        tokio::time::timeout(Duration::from_secs(20), tracker.wait())
            .await
            .expect("dispatch tracker must drain inside 20s");
    }

    /// Await the gated adapter call's arrival signal.
    async fn await_gate(reached: oneshot::Receiver<()>) {
        tokio::time::timeout(Duration::from_secs(20), reached)
            .await
            .expect("the gated put must be reached inside 20s")
            .expect("the gated put must signal arrival before parking");
    }

    /// **Coalescing.** A burst of N enqueues landing while one operation is
    /// in flight collapses into ONE follow-up operation, not N.
    ///
    /// Without it, a session under rapid lifecycle churn spawns one full
    /// build-plus-PUT per event — every one of which is superseded by the
    /// next before it is worth anything, and all of them unordered relative
    /// to each other, which is the b.443 clobber. Coalescing removes the
    /// redundant work; the publish watermark orders what remains across
    /// replicas.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn burst_of_enqueues_coalesces_into_one_follow_up_operation() {
        const BURST: usize = 9;

        let tv = TestValkey::setup().await;
        let (gate_store, gate) = GateStore::new(1);
        let (session_id, writer, tracker) =
            slot_harness(&tv, Arc::clone(&gate_store) as Arc<dyn CheckpointStore>).await;

        writer.enqueue_session_checkpoint_write(session_id.clone());
        await_gate(gate.reached).await;

        for _ in 0..BURST {
            writer.enqueue_session_checkpoint_write(session_id.clone());
        }
        gate.release.send(()).expect("the gated put must be parked");
        drain(&tracker).await;

        let puts = gate_store.put_calls();
        assert_eq!(
            puts,
            2,
            "{BURST} enqueues arriving while one operation was in flight MUST coalesce into a \
             single follow-up — 2 adapter calls total, not {}. A count near {} means the \
             coalescing slot is gone and every lifecycle event spawns its own unordered build, \
             which is both wasted work and the b.443 clobber window; got {puts}",
            BURST + 1,
            BURST + 1
        );
        assert_eq!(
            gate_store.delete_calls(),
            0,
            "no DELETE was enqueued in this test"
        );
    }

    /// **A DELETE queued behind pending WRITEs supersedes them, and a WRITE
    /// queued behind it does not resurrect them.**
    ///
    /// The enqueue order is deliberately `Write, Write, Delete, Write` while
    /// the first operation is parked: the trailing `Write` is the one that
    /// matters. If `merge` were symmetric, that last enqueue would downgrade
    /// the pending DELETE back to a WRITE, the dispatcher would rebuild from
    /// live state, and a session the operator deleted would keep a durable
    /// checkpoint that recovery restores.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pending_delete_absorbs_queued_writes_and_is_not_downgraded() {
        let tv = TestValkey::setup().await;
        let (gate_store, gate) = GateStore::new(1);
        let (session_id, writer, tracker) =
            slot_harness(&tv, Arc::clone(&gate_store) as Arc<dyn CheckpointStore>).await;

        writer.enqueue_session_checkpoint_write(session_id.clone());
        await_gate(gate.reached).await;

        writer.enqueue_session_checkpoint_write(session_id.clone());
        writer.enqueue_session_checkpoint_write(session_id.clone());
        writer.enqueue_session_checkpoint_delete(session_id.clone());
        // The trailing WRITE: must NOT downgrade the pending DELETE.
        writer.enqueue_session_checkpoint_write(session_id.clone());

        gate.release.send(()).expect("the gated put must be parked");
        drain(&tracker).await;

        assert_eq!(
            gate_store.put_calls(),
            1,
            "only the already-in-flight WRITE may reach the adapter — the queued WRITEs were \
             absorbed by the DELETE"
        );
        assert_eq!(
            gate_store.delete_calls(),
            1,
            "the pending DELETE MUST run, and exactly once. Zero means the trailing WRITE \
             downgraded it, and the deleted session keeps a durable checkpoint that recovery \
             would restore"
        );
    }

    /// **Shutdown drains coalesced work.** An operation queued while one is
    /// in flight still runs after the tracker is closed.
    ///
    /// This is why the dispatch task drains its pending work in an internal
    /// loop instead of re-spawning per operation: `main.rs` calls `close()`
    /// then `wait()`, and a task spawned after `close()` is not covered by a
    /// `wait()` that has already returned. A re-spawn shape would drop the
    /// queued write on the floor at shutdown — silently, and precisely when
    /// the durable copy matters most.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pending_operation_survives_tracker_close_and_lands() {
        let tv = TestValkey::setup().await;
        let (gate_store, gate) = GateStore::new(1);
        let (session_id, writer, tracker) =
            slot_harness(&tv, Arc::clone(&gate_store) as Arc<dyn CheckpointStore>).await;

        writer.enqueue_session_checkpoint_write(session_id.clone());
        await_gate(gate.reached).await;

        // Queue a second operation, THEN close the tracker, THEN let the
        // first one finish. The second has to survive the close.
        writer.enqueue_session_checkpoint_write(session_id.clone());
        tracker.close();
        gate.release.send(()).expect("the gated put must be parked");
        tokio::time::timeout(Duration::from_secs(20), tracker.wait())
            .await
            .expect("dispatch tracker must drain inside 20s");

        assert_eq!(
            gate_store.put_calls(),
            2,
            "the operation queued before `close()` MUST still land. A count of 1 means the \
             dispatcher re-spawns per operation and the post-`close()` spawn was dropped — the \
             queued session checkpoint is lost exactly at shutdown"
        );
    }

    /// Poll until the tracker owns no live tasks, or fail.
    ///
    /// Used where the test must observe the aftermath of a task that ENDED
    /// while keeping the tracker open for a later enqueue, so `drain` (which
    /// closes) is not usable. `TaskTracker` drops a task from its count only
    /// after the task's future has been dropped — which, on the unwind path,
    /// is after [`SlotGuard::drop`] has run. So an empty tracker is a
    /// sufficient happens-before for reading the slot map.
    async fn await_tracker_idle(tracker: &TaskTracker) {
        tokio::time::timeout(Duration::from_secs(20), async {
            while !tracker.is_empty() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("the dispatch task must finish inside 20s");
    }

    /// **A dispatch task that unwinds does not kill dispatch for its
    /// session.** (`SlotGuard`, retire arm.)
    ///
    /// The task sets `in_flight = true` before it runs and is the only thing
    /// that clears it. Without the RAII guard, a panic anywhere in the loop —
    /// the adapter, a decoder, the publish decode — leaves the claim set with
    /// no task behind it, and every subsequent enqueue for that session
    /// silently records a pending op nobody ever services. That converts a
    /// one-write loss into permanent per-session dispatch death, which for a
    /// hot session means its durable checkpoint is frozen at the pre-panic
    /// state for the lifetime of the replica.
    ///
    /// The panic here is synthetic (a `GateStore` that panics instead of
    /// delegating) but the unwind path is the real one: it tears down
    /// `dispatch_loop`'s frame, dropping the guard.
    ///
    /// A panic backtrace on stderr during this test is EXPECTED output, not
    /// a failure.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_panicking_adapter_call_releases_the_slot_for_the_next_enqueue() {
        let tv = TestValkey::setup().await;
        let (store, _gate) = GateStore::with_options(usize::MAX, 1);
        let (session_id, writer, tracker) =
            slot_harness(&tv, Arc::clone(&store) as Arc<dyn CheckpointStore>).await;

        writer.enqueue_session_checkpoint_write(session_id.clone());
        await_tracker_idle(&tracker).await;

        assert_eq!(
            store.put_calls(),
            1,
            "precondition: the first operation must have reached the panicking adapter call"
        );
        assert!(
            writer.slots.get(&session_id).is_none(),
            "an unwinding dispatch task MUST retire its slot when nothing is queued. A surviving \
             entry means the `in_flight` claim outlived the task that held it, and every later \
             enqueue for this session records a pending op with nobody to service it"
        );

        // The real assertion: dispatch still works afterwards.
        writer.enqueue_session_checkpoint_write(session_id.clone());
        drain(&tracker).await;

        assert_eq!(
            store.put_calls(),
            2,
            "the enqueue AFTER the unwind MUST dispatch. A count of 1 means the panic left the \
             session's dispatch permanently dead"
        );
    }

    /// **An operation queued at the moment of an unwind is deferred, not
    /// lost.** (`SlotGuard`, release arm.)
    ///
    /// The guard's two arms differ: with nothing pending it retires the slot
    /// outright; with an op pending it must leave the entry in place and only
    /// drop the claim, so the pending op is still there for the next task to
    /// pick up. Getting that backwards (retiring unconditionally) would
    /// discard the queued operation silently.
    ///
    /// Deliberately NOT a re-spawn from `Drop`: a panic loop would respawn
    /// forever, and a spawn during shutdown unwinding lands after the
    /// tracker's `close()` where nothing drains it. So the contract is
    /// "deferred to the next lifecycle event", and this test drives that
    /// event explicitly.
    ///
    /// A panic backtrace on stderr during this test is EXPECTED output.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn an_operation_queued_when_a_dispatch_task_unwinds_is_deferred_not_lost() {
        let tv = TestValkey::setup().await;
        // Park on put #1, then panic on the same call — so the test can queue
        // a second operation while the first is provably still in flight.
        let (store, gate) = GateStore::with_options(1, 1);
        let (session_id, writer, tracker) =
            slot_harness(&tv, Arc::clone(&store) as Arc<dyn CheckpointStore>).await;

        writer.enqueue_session_checkpoint_write(session_id.clone());
        await_gate(gate.reached).await;

        writer.enqueue_session_checkpoint_write(session_id.clone());
        gate.release.send(()).expect("the gated put must be parked");
        await_tracker_idle(&tracker).await;

        let slot_pending = writer
            .slots
            .get(&session_id)
            .map(|slot| (slot.in_flight, slot.pending));
        assert_eq!(
            slot_pending,
            Some((false, Some(PendingOp::Write))),
            "the queued WRITE MUST survive the unwind with the claim released: `None` means the \
             guard retired a slot that still had work on it (the operation is silently lost), and \
             `in_flight = true` means the claim outlived its task (dispatch is dead)"
        );

        // The next lifecycle event resumes dispatch and drains BOTH the
        // deferred op and its own.
        writer.enqueue_session_checkpoint_write(session_id.clone());
        drain(&tracker).await;

        assert_eq!(
            store.put_calls(),
            2,
            "put #1 panicked; the next enqueue must spawn a fresh task that services the deferred \
             work — exactly one further adapter call, since the deferred WRITE and the new one \
             coalesce"
        );
    }

    // =====================================================================
    // Group 3 — the publish-watermark script contract (b.443).
    // =====================================================================

    /// The incarnation every watermark test below publishes under.
    ///
    /// b.hka (a) put an incarnation gate in front of the watermark
    /// comparison: a publish whose token is not the one currently on
    /// `le:{sid}:ckpt` is refused with `"stale_incarnation"` having written
    /// nothing. These tests exist to pin the watermark arms *behind* that
    /// gate, so each one seeds the key with [`seed_incarnation`] first —
    /// otherwise every one of them would be refused before reaching the arm
    /// it names, and would be green for the wrong reason if an arm regressed.
    ///
    /// A hand-written token rather than one minted through
    /// `session_checkpoint_read.lua`: the mint also `HINCRBY`s `seq`, and
    /// `building_advances_seq_but_leaves_the_publish_watermark_alone` counts
    /// stamps. `HSET`ting `inc` is the only seeding that leaves `seq`
    /// untouched. The gate compares for equality and never interprets the
    /// value, so a fixed string is as faithful as a random one — and the
    /// *mismatch* arm is driven separately by
    /// `stale_incarnation_refusal_writes_nothing_and_does_not_refresh_the_ttl`.
    const TEST_INCARNATION: &str = "test-incarnation";

    /// Install [`TEST_INCARNATION`] as `le:{sid}:ckpt`'s incarnation.
    ///
    /// `HSET`, not `HSETNX`: a couple of arms seed a hand-written watermark
    /// first, and the point is to make the key's token match what the test
    /// publishes under, whatever else is on the key.
    async fn seed_incarnation(tv: &TestValkey, session_id: &str) {
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().session_checkpoint_state(session_id),
                vec![("inc", TEST_INCARNATION.to_string())],
            )
            .await
            .expect("HSET inc must succeed");
    }

    /// Build a `DispatchCtx` for driving `publish_stamp` directly. The store
    /// is never touched on this path — only the Valkey handle and the key
    /// schema matter.
    fn publish_ctx(tv: &TestValkey) -> DispatchCtx {
        DispatchCtx {
            store: Arc::new(InMemoryCheckpointStore::new()),
            client: tv.client().clone(),
            keys: tv.key_schema().clone(),
            metrics: Arc::new(metrics_scrape_only("publish-test")),
            creator_replica_id: "publish-replica".to_string(),
            slots: Arc::new(DashMap::new()),
        }
    }

    /// Read `pub` / `pub_kind` off `le:{sid}:ckpt`.
    async fn watermark(tv: &TestValkey, session_id: &str) -> (Option<String>, Option<String>) {
        let key = tv.key_schema().session_checkpoint_state(session_id);
        let published: Option<String> = tv
            .client()
            .pool
            .hget(&key, "pub")
            .await
            .expect("HGET pub must succeed");
        let kind: Option<String> = tv
            .client()
            .pool
            .hget(&key, "pub_kind")
            .await
            .expect("HGET pub_kind must succeed");
        (published, kind)
    }

    /// **Accept arm.** A stamp above the current watermark is accepted and
    /// becomes the new watermark, along with its kind.
    ///
    /// Recording the KIND — not just the number — is what lets a later,
    /// stale operation choose the right repair. A watermark storing only the
    /// stamp would leave the loser knowing it lost but not what to do.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn publish_accepts_a_higher_stamp_and_records_its_kind() {
        let tv = TestValkey::setup().await;
        let ctx = publish_ctx(&tv);
        let sid = "publish-accept";
        seed_incarnation(&tv, sid).await;

        let refused = publish_stamp(&ctx, sid, 5, TEST_INCARNATION, PendingOp::Write)
            .await
            .expect("publish must not error against a healthy Valkey");
        assert_eq!(
            refused, None,
            "a publish above the (absent, i.e. zero) watermark MUST be accepted"
        );
        assert_eq!(
            watermark(&tv, sid).await,
            (Some("5".to_string()), Some("write".to_string())),
            "the accepted stamp and its kind MUST both be recorded"
        );
    }

    /// **Refuse arm, WRITE.** A stamp below the watermark is refused, the
    /// fresher operation's kind is reported back, and the watermark is left
    /// untouched.
    ///
    /// Leaving `pub` untouched is the load-bearing half: lowering it to the
    /// loser's stamp would re-open the window for the NEXT stale write to
    /// land undetected.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn publish_refuses_a_lower_stamp_and_reports_the_fresher_write() {
        let tv = TestValkey::setup().await;
        let ctx = publish_ctx(&tv);
        let sid = "publish-refuse-write";
        seed_incarnation(&tv, sid).await;

        publish_stamp(&ctx, sid, 5, TEST_INCARNATION, PendingOp::Write)
            .await
            .expect("the first publish must succeed");
        let refused = publish_stamp(&ctx, sid, 3, TEST_INCARNATION, PendingOp::Write)
            .await
            .expect("publish must not error");

        assert_eq!(
            refused,
            Some(PublishRefusal::Fresher(PendingOp::Write)),
            "stamp 3 landed on top of stamp 5 and MUST be told the fresher operation was a \
             WRITE, so it repairs by rebuilding"
        );
        assert_eq!(
            watermark(&tv, sid).await,
            (Some("5".to_string()), Some("write".to_string())),
            "a refused publish MUST leave the watermark at the fresher stamp"
        );
    }

    /// **Refuse arm, DELETE.** The reported kind follows the fresher
    /// operation, not the caller's.
    ///
    /// A stale WRITE that clobbered a DELETE must repair as a DELETE.
    /// Repairing as a rebuild would not work: `build_session_checkpoint`
    /// returns `Ok(None)` for the (now absent) session and exits silently,
    /// leaving the resurrected blob durable.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn publish_reports_a_fresher_delete_to_a_stale_write() {
        let tv = TestValkey::setup().await;
        let ctx = publish_ctx(&tv);
        let sid = "publish-refuse-delete";
        seed_incarnation(&tv, sid).await;

        publish_stamp(&ctx, sid, 9, TEST_INCARNATION, PendingOp::Delete)
            .await
            .expect("the first publish must succeed");
        let refused = publish_stamp(&ctx, sid, 4, TEST_INCARNATION, PendingOp::Write)
            .await
            .expect("publish must not error");

        assert_eq!(
            refused,
            Some(PublishRefusal::Fresher(PendingOp::Delete)),
            "the kind reported MUST be the FRESHER operation's (delete), not the caller's \
             (write) — otherwise the repair rebuilds, exits silently on the absent session, and \
             the resurrected checkpoint stays durable"
        );
    }

    /// **Equal stamp, same kind → accepted.** The self-repair guard.
    ///
    /// Distinct operations never share a stamp (each mints its own), so an
    /// equal stamp with an equal kind can only be one operation publishing
    /// twice — a retry that re-ran the adapter call but reused its stamp.
    /// Reporting a supersession there would send the caller into a repair
    /// against itself, which is a livelock, not a fix.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn publish_accepts_a_repeat_of_the_same_stamp_and_kind() {
        let tv = TestValkey::setup().await;
        let ctx = publish_ctx(&tv);
        let sid = "publish-equal-same";
        seed_incarnation(&tv, sid).await;

        publish_stamp(&ctx, sid, 7, TEST_INCARNATION, PendingOp::Write)
            .await
            .expect("the first publish must succeed");
        let repeat = publish_stamp(&ctx, sid, 7, TEST_INCARNATION, PendingOp::Write)
            .await
            .expect("publish must not error");

        assert_eq!(
            repeat, None,
            "re-publishing the SAME stamp with the SAME kind is one operation reporting twice \
             (a retry reusing its stamp) and MUST be accepted — treating it as a supersession \
             sends the task into a repair against itself"
        );
    }

    /// **Equal stamp, different kind → refused.** The guard above is scoped
    /// to same-kind repeats only.
    ///
    /// Two different kinds at one stamp cannot be the same operation
    /// reporting twice, so the safe reading is that something genuinely
    /// landed on top of something else and the caller is told to repair.
    /// Pinned separately from the same-kind case because widening the guard
    /// to ignore `kind` is an easy-looking simplification that would
    /// silently drop this repair.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn publish_refuses_an_equal_stamp_with_a_different_kind() {
        let tv = TestValkey::setup().await;
        let ctx = publish_ctx(&tv);
        let sid = "publish-equal-diff";
        seed_incarnation(&tv, sid).await;

        publish_stamp(&ctx, sid, 7, TEST_INCARNATION, PendingOp::Write)
            .await
            .expect("the first publish must succeed");
        let refused = publish_stamp(&ctx, sid, 7, TEST_INCARNATION, PendingOp::Delete)
            .await
            .expect("publish must not error");

        assert_eq!(
            refused,
            Some(PublishRefusal::Fresher(PendingOp::Write)),
            "an equal stamp with a DIFFERENT kind cannot be the same operation reporting twice, \
             so it MUST be refused and the recorded kind reported"
        );
    }

    /// **A publish refused for being STALE still refreshes the TTL.**
    ///
    /// `le:{sid}:ckpt` is deliberately absent from `SessionStore::delete`'s
    /// teardown (a repair reads the watermark after the DELETE has landed),
    /// so a TTL is the only thing that reaps it — and it must not expire
    /// while operations are still arriving. A session hot enough to be
    /// producing supersessions is precisely the one whose ordering anchor
    /// must survive: if the key expired, `seq` would restart from 1 and a
    /// stale build's stamp could outrank a fresh one under a fresh
    /// incarnation.
    ///
    /// **Scoped to the fresher-operation arm on purpose (b.hka (a)).** The
    /// incarnation-mismatch arm added by (a) deliberately does the
    /// opposite — it writes *nothing*, TTL included, because a `PEXPIRE`
    /// there would resurrect a key the live incarnation never asked for. The
    /// incarnation is therefore seeded to match before anything else happens,
    /// so this test reaches the arm it names;
    /// `stale_incarnation_refusal_writes_nothing_and_does_not_refresh_the_ttl`
    /// pins the other one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_refused_publish_refreshes_the_ttl() {
        let tv = TestValkey::setup().await;
        let ctx = publish_ctx(&tv);
        let sid = "publish-ttl-refresh";
        let key = tv.key_schema().session_checkpoint_state(sid);
        seed_incarnation(&tv, sid).await;

        publish_stamp(&ctx, sid, 5, TEST_INCARNATION, PendingOp::Write)
            .await
            .expect("the first publish must succeed");

        // Squeeze the TTL far below the configured value.
        let _: () = tv
            .client()
            .pool
            .pexpire(&key, 1_000, None)
            .await
            .expect("PEXPIRE must succeed");
        let squeezed: i64 = tv
            .client()
            .pool
            .pttl(&key)
            .await
            .expect("PTTL must succeed");
        assert!(
            squeezed > 0 && (squeezed as u64) < SESSION_CHECKPOINT_STATE_TTL_MS,
            "precondition: the TTL must actually be squeezed; PTTL={squeezed}"
        );

        let refused = publish_stamp(&ctx, sid, 3, TEST_INCARNATION, PendingOp::Write)
            .await
            .expect("publish must not error");
        assert_eq!(
            refused,
            Some(PublishRefusal::Fresher(PendingOp::Write)),
            "precondition: the second publish must be refused"
        );

        let refreshed: i64 = tv
            .client()
            .pool
            .pttl(&key)
            .await
            .expect("PTTL must succeed");
        assert!(
            refreshed as u64 > SESSION_CHECKPOINT_STATE_TTL_MS / 2,
            "a REFUSED publish must still refresh the TTL — a hot session is exactly the one \
             whose ordering anchor must not expire, and losing it restarts `seq` from 1 so a \
             stale build could outrank a fresh one; PTTL={refreshed}"
        );
    }

    /// **Builds advance `seq` and never touch `pub`.**
    ///
    /// The two fields share one key on purpose — they always reset together,
    /// so an expiry cannot leave a high watermark guarding a restarted
    /// counter. But they must not be written by the same operation: if a
    /// build advanced `pub`, an operation would be published before its
    /// adapter call had landed, and a concurrent write that clobbered it
    /// would see a watermark indistinguishable from a real one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn building_advances_seq_but_leaves_the_publish_watermark_alone() {
        let tv = TestValkey::setup().await;
        let ctx = publish_ctx(&tv);
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create("pub-untouched-stage", "/stage/pub-untouched", "pub-user")
            .await
            .expect("session creation must succeed");
        // Seeding with `HSET` (not a mint through the read script) is what
        // keeps the `seq == 3` assertion below meaningful: a mint would
        // itself `HINCRBY seq`, and the count would stop being "one stamp
        // per build".
        seed_incarnation(&tv, &session_id).await;

        publish_stamp(&ctx, &session_id, 5, TEST_INCARNATION, PendingOp::Write)
            .await
            .expect("publish must succeed");

        for _ in 0..3 {
            crate::checkpoint::session::build_session_checkpoint(
                tv.client(),
                tv.key_schema(),
                &session_id,
            )
            .await
            .expect("assembler must succeed")
            .expect("session must exist");
        }

        assert_eq!(
            watermark(&tv, &session_id).await,
            (Some("5".to_string()), Some("write".to_string())),
            "reads MUST NOT advance `pub` — publishing before the adapter call landed would let \
             a clobbering write mistake an unlanded operation for a durable one"
        );
        let seq: Option<String> = tv
            .client()
            .pool
            .hget(tv.key_schema().session_checkpoint_state(&session_id), "seq")
            .await
            .expect("HGET seq must succeed");
        assert_eq!(
            seq.as_deref(),
            Some("3"),
            "each build MUST mint exactly one stamp"
        );
    }

    /// **A watermark with a missing or unrecognised `pub_kind` degrades to a
    /// rebuild rather than to silence.**
    ///
    /// Neither state is reachable through the normal protocol — `pub` and
    /// `pub_kind` are written by one `HSET`, and the kind vocabulary is
    /// closed — so both are defensive arms, one in the script and one in
    /// `publish_stamp`'s decoder. They are worth pinning anyway because the
    /// *choice of fallback* is a correctness decision, not a formality: the
    /// caller must be told to repair, and told to repair as a WRITE.
    /// Rebuilding from live state is the arm that cannot destroy
    /// information (worst case it re-writes what is already there), whereas
    /// falling back to `delete`, or to `Ok(None)`, would respectively
    /// tombstone a live session's checkpoint or leave a known clobber
    /// unrepaired.
    ///
    /// Both arms are driven by writing the corrupt state into
    /// `le:{sid}:ckpt` directly, which is the only way to reach them.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn publish_falls_back_to_a_rebuild_when_the_recorded_kind_is_unusable() {
        let tv = TestValkey::setup().await;
        let ctx = publish_ctx(&tv);

        // Arm 1 — `pub` set, `pub_kind` absent. Handled inside the script.
        let missing_kind_sid = "publish-kind-missing";
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().session_checkpoint_state(missing_kind_sid),
                // `inc` alongside the hand-written watermark: b.hka (a)'s gate
                // runs first and would otherwise refuse this publish with
                // `stale_incarnation`, never reaching the missing-kind arm.
                vec![
                    ("pub", "100".to_string()),
                    ("inc", TEST_INCARNATION.to_string()),
                ],
            )
            .await
            .expect("HSET must succeed");
        assert_eq!(
            publish_stamp(
                &ctx,
                missing_kind_sid,
                1,
                TEST_INCARNATION,
                PendingOp::Delete
            )
            .await
            .expect("publish must not error"),
            Some(PublishRefusal::Fresher(PendingOp::Write)),
            "a watermark with no recorded kind MUST still refuse the stale publish and ask for a \
             rebuild — answering `delete` would tombstone a session that may well be live, and \
             accepting would leave a known clobber unrepaired"
        );

        // Arm 2 — `pub_kind` present but outside the closed vocabulary.
        // Handled by `publish_stamp`'s decoder.
        let bogus_kind_sid = "publish-kind-bogus";
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().session_checkpoint_state(bogus_kind_sid),
                vec![
                    ("pub", "100".to_string()),
                    ("pub_kind", "purge".to_string()),
                    ("inc", TEST_INCARNATION.to_string()),
                ],
            )
            .await
            .expect("HSET must succeed");
        assert_eq!(
            publish_stamp(&ctx, bogus_kind_sid, 1, TEST_INCARNATION, PendingOp::Delete)
                .await
                .expect("publish must not error"),
            Some(PublishRefusal::Fresher(PendingOp::Write)),
            "an unrecognised `pub_kind` means the script and this decoder have diverged; the \
             caller MUST still be sent down the information-preserving rebuild arm rather than \
             having the supersession dropped"
        );
    }

    // =====================================================================
    // Group 4 — the incarnation gate in front of the watermark (b.hka (a)).
    //
    // The watermark orders operations only while `le:{sid}:ckpt` survives.
    // Lose the key and `seq` restarts at 1, so a build minted before the loss
    // still holds a stamp (say 5) that outranks the fresh watermark (1) and
    // would durably reinstate pre-loss bytes. The gate closes that by scoping
    // every stamp to the key's LIFETIME: `session_checkpoint_read.lua`
    // `HSETNX`es a random token, every publish carries the token its build
    // read, and `session_checkpoint_publish.lua` refuses outright when the
    // token on the key is absent or different.
    //
    // These tests pin the SCRIPT half of that contract. The end-to-end
    // keyspace-loss repair — the thing that actually fails without the fix —
    // is in `tests/integration/checkpoint/
    // lifecycle_session_publish_watermark_test.rs`.
    // =====================================================================

    /// **The token a real mint hands back is the one publish is judged
    /// against.**
    ///
    /// The positive control for the whole group, and the only test here that
    /// uses a genuinely minted token rather than [`TEST_INCARNATION`]: it
    /// pins that `session_checkpoint_read.lua`'s reply element 7 and
    /// `session_checkpoint_publish.lua`'s `ARGV[4]` are the same value, so
    /// the gate is transparent on the healthy path. If the two ever disagreed
    /// — a reply-element renumbering, an ARGV shift — every publish in the
    /// fleet would be refused as stale and every session checkpoint would
    /// enter a permanent rebuild loop, which is a far worse failure than the
    /// one the gate exists to prevent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_publish_under_the_minted_incarnation_is_accepted() {
        let tv = TestValkey::setup().await;
        let ctx = publish_ctx(&tv);
        let sid = "publish-minted-inc";

        let (stamp, incarnation) = mint_session_checkpoint_stamp(tv.client(), tv.key_schema(), sid)
            .await
            .expect("the mint must succeed against a healthy Valkey");
        assert!(
            !incarnation.is_empty(),
            "the mint MUST return a non-empty incarnation — an empty token would make the gate \
             vacuous"
        );

        let refused = publish_stamp(&ctx, sid, stamp, &incarnation, PendingOp::Write)
            .await
            .expect("publish must not error against a healthy Valkey");
        assert_eq!(
            refused, None,
            "a publish carrying the incarnation its own mint returned MUST be accepted — the \
             gate is only ever meant to fire on a key that was lost and re-created"
        );
        assert_eq!(
            watermark(&tv, sid).await,
            (Some(stamp.to_string()), Some("write".to_string())),
            "and it MUST advance the watermark exactly as it did before the gate existed"
        );
    }

    /// **A publish under a foreign incarnation is refused, and the script
    /// writes NOTHING — not the watermark, not even the TTL.**
    ///
    /// Both halves are load-bearing:
    ///
    /// - Refusing is the fix: this is the pre-loss build whose stamp (5)
    ///   outranks the post-loss watermark (3) and would otherwise be accepted,
    ///   durably reinstating pre-loss bytes and poisoning the next two
    ///   legitimate builds.
    /// - Writing nothing is what keeps the refusal from doing damage of its
    ///   own. A bare `PEXPIRE` from an operation that does not own the key
    ///   would extend the life of a key the live incarnation never asked for;
    ///   an `HSET` would hand the stale build the watermark it was just
    ///   refused.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn stale_incarnation_refusal_writes_nothing_and_does_not_refresh_the_ttl() {
        /// The squeezed TTL, in milliseconds. Any refresh would restore
        /// `SESSION_CHECKPOINT_STATE_TTL_MS` (24 h), which is four orders of
        /// magnitude away — so the comparison needs no tolerance.
        const SQUEEZED_TTL_MS: i64 = 1_000;

        let tv = TestValkey::setup().await;
        let ctx = publish_ctx(&tv);
        let sid = "publish-foreign-inc";
        let key = tv.key_schema().session_checkpoint_state(sid);
        seed_incarnation(&tv, sid).await;

        // Arrange: a live incarnation with a watermark at 3, then a squeezed
        // TTL so a refresh is unmistakable.
        publish_stamp(&ctx, sid, 3, TEST_INCARNATION, PendingOp::Delete)
            .await
            .expect("the seeding publish must succeed");
        let _: () = tv
            .client()
            .pool
            .pexpire(&key, SQUEEZED_TTL_MS, None)
            .await
            .expect("PEXPIRE must succeed");

        // Act: the pre-loss build lands, holding a HIGHER stamp under a token
        // the key no longer carries.
        let refused = publish_stamp(&ctx, sid, 5, "some-other-incarnation", PendingOp::Write)
            .await
            .expect("publish must not error");

        assert_eq!(
            refused,
            Some(PublishRefusal::StaleIncarnation),
            "a stamp minted against a DIFFERENT incarnation is not comparable to the live \
             watermark at all, so it MUST be refused as `StaleIncarnation` — reporting \
             `Fresher(..)` would tell the caller a race it can reason about happened, and \
             accepting it (stamp 5 > watermark 3) is exactly the b.hka (a) silent overwrite"
        );
        assert_eq!(
            watermark(&tv, sid).await,
            (Some("3".to_string()), Some("delete".to_string())),
            "the refusal MUST leave the live incarnation's watermark and kind untouched"
        );
        let pttl: i64 = tv
            .client()
            .pool
            .pttl(&key)
            .await
            .expect("PTTL must succeed");
        assert!(
            pttl > 0 && pttl <= SQUEEZED_TTL_MS,
            "the refusal MUST NOT refresh the TTL: the operation does not own this key, and \
             extending a key on behalf of a dead incarnation keeps a husk alive that the next \
             mint would adopt. PTTL={pttl}, expected <= {SQUEEZED_TTL_MS}"
        );
    }

    /// **A publish against an ABSENT `le:{sid}:ckpt` is refused without
    /// creating the key.**
    ///
    /// This is the shape keyspace loss actually takes: the key is simply gone
    /// when the pre-loss build's publish arrives. The refusal must not be the
    /// thing that brings it back, because a key created by `PEXPIRE` alone has
    /// no `inc` and no `seq` — a husk that the next mint would `HSETNX` its
    /// own token onto and then count from 1 against, silently adopting a key
    /// that no build ever established.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn publish_against_an_absent_ckpt_key_is_refused_without_creating_it() {
        let tv = TestValkey::setup().await;
        let ctx = publish_ctx(&tv);
        let sid = "publish-absent-key";
        let key = tv.key_schema().session_checkpoint_state(sid);

        let refused = publish_stamp(&ctx, sid, 42, TEST_INCARNATION, PendingOp::Write)
            .await
            .expect("publish must not error");
        assert_eq!(
            refused,
            Some(PublishRefusal::StaleIncarnation),
            "an absent key carries no incarnation, so nothing this operation holds can be \
             comparable to it — that is a `StaleIncarnation` refusal, not an accept"
        );

        let exists: i64 = tv
            .client()
            .pool
            .exists(&key)
            .await
            .expect("EXISTS must succeed");
        assert_eq!(
            exists, 0,
            "the refusal MUST NOT create {key}: a key holding only a TTL has no `inc` and no \
             `seq`, and the next mint would adopt that husk as its own incarnation"
        );
    }

    /// **Both kinds route the stale-incarnation refusal to a WRITE repair, and
    /// both label it `stale_incarnation`.**
    ///
    /// The kind-preserving repair the watermark uses for a genuine race would
    /// be wrong here. A stale incarnation says nothing about what else landed
    /// — the key that recorded it is gone — so repairing a stale DELETE *as* a
    /// delete could tombstone the checkpoint of a session that is very much
    /// alive. Rebuilding is the arm that cannot destroy information: against a
    /// live session it overwrites the stale bytes under the new incarnation,
    /// and against a deleted one `build_session_checkpoint` returns `Ok(None)`
    /// and the chain exits, leaving exactly the bytes DR wants.
    ///
    /// The `result` label travels separately from the repair kind for the same
    /// reason: the repair is `write`, but booking it under `result="write"`
    /// would file a keyspace-integrity event as an ordinary race.
    ///
    /// Driven through the publish fault seam rather than the script because
    /// the mapping under test lives in `run_operation`, *after* the decode: a
    /// real script refusal would have to be arranged between the build's mint
    /// and the publish, which needs a gated adapter (that shape is covered
    /// end-to-end in
    /// `tests/integration/checkpoint/lifecycle_session_publish_watermark_test.rs`).
    /// The seam is process-global, so the queue is cleared on entry — see
    /// `PUBLISH_FAULT_INJECTOR`.
    ///
    /// One caveat to read this test with: the seam re-derives the refusal
    /// classification rather than running the live decoder, so what is pinned
    /// here is the `run_operation` mapping (refusal → WRITE repair, labelled
    /// `stale_incarnation`) and not the decode that produces the refusal. That
    /// the real `session_checkpoint_publish.lua` reply decodes to the stale arm
    /// at all is covered by the two script-driven tests above —
    /// `stale_incarnation_refusal_writes_nothing_and_does_not_refresh_the_ttl`
    /// and `publish_against_an_absent_ckpt_key_is_refused_without_creating_it`
    /// — so no arm of the path is left resting on the seam's re-derivation
    /// alone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stale_incarnation_repairs_as_a_write_for_both_kinds() {
        let tv = TestValkey::setup().await;
        let ctx = publish_ctx(&tv);
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create("stale-inc-stage", "/stage/stale-inc", "stale-inc-user")
            .await
            .expect("session creation must succeed");

        for op in [PendingOp::Write, PendingOp::Delete] {
            clear_publish_fault_injector_for_test();
            set_publish_fault_injector_for_test(std::collections::VecDeque::from(vec![
                PublishFault::Refuse(STALE_INCARNATION_RESULT.to_string()),
            ]));

            let outcome = run_operation(&ctx, &session_id, op).await;
            clear_publish_fault_injector_for_test();

            let OpOutcome::Superseded { repair, result } = outcome else {
                panic!(
                    "a {kind} whose publish was refused for a stale incarnation must report \
                     `Superseded`; the operation LANDED at the adapter, so reporting `Settled` \
                     would leave stale bytes durable and `Abandoned` would misreport a completed \
                     adapter call as a failure",
                    kind = op.kind()
                );
            };
            assert_eq!(
                repair,
                PendingOp::Write,
                "a stale-incarnation refusal on a {kind} MUST repair as a WRITE — a DELETE \
                 repair would tombstone the checkpoint of a session the lost key can no longer \
                 vouch for",
                kind = op.kind()
            );
            assert_eq!(
                result, STALE_INCARNATION_RESULT,
                "and it MUST be booked under its own metric label, not under the repair's kind: \
                 `write` means the ordering protocol caught a live race, `stale_incarnation` \
                 means the coordination key was lost"
            );
        }
    }
}
