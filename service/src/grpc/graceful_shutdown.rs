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

//! The graceful-shutdown handle the streaming RPCs and the uploads read
//! (SDD §5.6), the paced ending of the live streams, and the hold ending.
//!
//! **SIGTERM is the one *shutdown* signal.** It alone closes the registry,
//! cancels `begun` and the parent, and starts the bound after GOAWAY
//! ([`end_streams_paced`], [`GracefulShutdown::cancel`]); those steps are
//! terminal. A disaster-recovery hold (SDD SR-6.3) is a separate,
//! non-terminal ending ([`GracefulShutdown::end_streams_for_hold`]): it ends
//! every enrolled stream at once with a per-stream [`HoldCause`], and touches
//! neither the registry's `closed` flag, nor `begun`, nor the parent, so the
//! server keeps answering and a later SIGTERM still runs the normal paced
//! shutdown.
//!
//! [`GracefulShutdown`] owns three tokens' worth of state:
//!
//! - **the parent** — the process-wide service shutdown token. It is the only
//!   signal graceful shutdown sends the streaming RPCs, cancelled once (after
//!   the last batch of the paced ending, or by [`GracefulShutdown::cancel`])
//!   and never reset. Shutdown never cancels a per-session token.
//! - **`begun`** — a child of the parent, cancelled when the paced ending
//!   starts (t0: the drain's end plus
//!   [`STREAM_END_DELAY_AFTER_GOAWAY`](super::transport::STREAM_END_DELAY_AFTER_GOAWAY)).
//!   The uploads' waits race it (admission, a processing step, the
//!   external-value release budget's, and an `UpdateLayer` commit's recharge
//!   between attempts, a processing step at stage `storage`), as does a
//!   chained `UpdateLayer`'s chain gate (`storage::chain_gate`), and
//!   JoinSession's handler checks it, so a call arriving or waiting at t0 is
//!   refused with nothing applied — except a recharge after a counted retry,
//!   which answers the outcome-unknown `UNAVAILABLE` (SDD §18.1), since an
//!   earlier attempt's outcome is not known.
//! - **the enrollment registry** — every live `SubscribeReadLayer` and
//!   `JoinSession` gets its shutdown token only from
//!   [`GracefulShutdown::enroll_stream`]: a [`StreamShutdown`], a fresh child
//!   of the parent with the stream's own cause slot, entered in the registry
//!   until the returned [`StreamEnrollment`] drops. Closing the registry,
//!   cancelling `begun` and taking the snapshot happen under one lock, so
//!   every stream is either in the snapshot or born cancelled. The hold
//!   ending's snapshot and its `held` flag are taken under the same lock, so
//!   the same holds for a hold.
//!
//! [`end_streams_paced`] ends the snapshot of N streams in random order over a
//! window of at most [`StreamPacing::max_window`] (W_cap): at
//! [`StreamPacing::rate_per_sec`] (R) per second, or at N / W_cap per second
//! once the cap binds (N > R × W_cap) — max(R, N / W_cap) — so the streams'
//! reconnects reach the other replicas as a ramp rather than a step, then
//! cancels the parent.
//!
//! **A stream's ending follows its own cause.** A stream that observes its
//! token cancelled reads [`StreamShutdown::hold_cause`]: none — SIGTERM, the
//! only other canceller — ends it `UNAVAILABLE "server shutting down"`;
//! [`HoldCause::Held`] ends it `SERVICE_HELD`, and [`HoldCause::StaleVerdict`]
//! `UPSTREAM_UNAVAILABLE` ([`HoldCause::status`]). The hold records the cause
//! before it cancels, so a stream woken by the hold never reads an empty cause,
//! and a stream the hold ended keeps its cause through a later SIGTERM. No
//! ending here ever cancels a per-session token (SDD §7.8).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use rand::seq::SliceRandom;
use tokio_util::sync::CancellationToken;
use tonic::Status;

use crate::error::RefusalDetail;

/// The interval between two batches of the paced ending (Δ).
pub const STREAM_END_BATCH_INTERVAL: Duration = Duration::from_millis(100);

/// How fast, and over how long at most, graceful shutdown ends the live
/// streams (`LIVE_EDIT_SHUTDOWN_STREAM_END_RATE`,
/// `LIVE_EDIT_SHUTDOWN_STREAM_END_MAX_WINDOW_SECS`).
///
/// N streams are ended over a window W = min(N / `rate_per_sec`,
/// `max_window`), in T = max(1, ⌈W / Δ⌉) batches, one every
/// [`STREAM_END_BATCH_INTERVAL`] (Δ). The ending rate is therefore
/// max(`rate_per_sec`, N / `max_window`): `rate_per_sec` is not a ceiling,
/// because above `rate_per_sec` × `max_window` streams the cap governs and
/// the rate rises with N. A `rate_per_sec` of 0 or a `max_window` of zero
/// ends every stream in one batch, at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamPacing {
    /// Endings per second (R) while the window is under its cap; above
    /// R × W_cap streams the rate is N / W_cap. 0 = all at once.
    pub rate_per_sec: u32,
    /// The window's cap (W_cap). Zero = all at once.
    pub max_window: Duration,
}

impl StreamPacing {
    /// The production defaults: 25 endings per second, over at most 20 s.
    pub const DEFAULT: Self = Self {
        rate_per_sec: 25,
        max_window: Duration::from_secs(20),
    };

    /// Every stream at once: one batch at t0.
    pub const ALL_AT_ONCE: Self = Self {
        rate_per_sec: 0,
        max_window: Duration::ZERO,
    };

    /// The window over which `streams` endings are spread:
    /// min(`streams` / `rate_per_sec`, `max_window`); zero when either switch
    /// is 0.
    pub fn window(&self, streams: usize) -> Duration {
        if self.rate_per_sec == 0 || self.max_window.is_zero() {
            return Duration::ZERO;
        }
        let nanos = (streams as u128).saturating_mul(1_000_000_000) / u128::from(self.rate_per_sec);
        let at_rate = Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX));
        at_rate.min(self.max_window)
    }

    /// The number of batches (T) `streams` endings are split into:
    /// max(1, ⌈W / Δ⌉).
    pub fn batches(&self, streams: usize) -> usize {
        let window = self.window(streams).as_nanos();
        let interval = STREAM_END_BATCH_INTERVAL.as_nanos();
        let ticks = window.div_ceil(interval);
        usize::try_from(ticks).unwrap_or(usize::MAX).max(1)
    }
}

impl Default for StreamPacing {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Why the hold ending ended a stream (SDD SR-6.3, SR-6.5). Recorded on the
/// stream's [`StreamShutdown`] before its token is cancelled; a stream whose
/// token is cancelled with no cause recorded was ended by SIGTERM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HoldCause {
    /// The fleet is held for disaster recovery: the stream ends
    /// `SERVICE_HELD`. `incident` is the hold marker's, `None` when the marker
    /// names none.
    Held {
        /// The incident the hold marker names, if any.
        incident: Option<String>,
    },
    /// This replica has had no successful hold-poll verdict for the stale
    /// threshold, and fails closed on its own: the stream ends
    /// `UPSTREAM_UNAVAILABLE`. No incident: there is no fleet hold.
    StaleVerdict,
}

impl HoldCause {
    /// The status a stream ended for this cause ends with:
    /// [`RefusalDetail::service_held_status`] for [`Self::Held`], with its
    /// incident; [`hold_verdict_stale`](super::transport::hold_verdict_stale)
    /// (`UPSTREAM_UNAVAILABLE`) for [`Self::StaleVerdict`].
    pub fn status(&self) -> Status {
        match self {
            Self::Held { incident } => RefusalDetail::service_held_status(incident.as_deref()),
            Self::StaleVerdict => super::transport::hold_verdict_stale(),
        }
    }
}

/// One live stream's shutdown token, with the cause slot the hold ending
/// writes. Handed out only by [`GracefulShutdown::enroll_stream`].
///
/// Derefs to the token, so `cancelled()` / `is_cancelled()` and every API
/// that races a bare token read as before. A stream that observes the token
/// cancelled reads [`Self::hold_cause`] to choose its ending
/// ([`Self::ending_status`]).
#[derive(Clone, Debug)]
pub struct StreamShutdown {
    token: CancellationToken,
    cause: Arc<OnceLock<HoldCause>>,
}

impl StreamShutdown {
    /// A token already cancelled, so the stream is refused at its first
    /// check, carrying `cause` when a hold refused it. Not a child of the
    /// parent: nothing will ever need to cancel it.
    fn born_cancelled(cause: Option<HoldCause>) -> Self {
        let stream = Self {
            token: CancellationToken::new(),
            cause: Arc::default(),
        };
        stream.end(cause);
        stream
    }

    /// Record `cause` if one is given and none is recorded yet, then cancel.
    /// The record is written before the cancel, so a stream that observes the
    /// cancel reads the cause (the token's cancel synchronizes with its
    /// waiters).
    fn end(&self, cause: Option<HoldCause>) {
        if let Some(cause) = cause {
            let _ = self.cause.set(cause);
        }
        self.token.cancel();
    }

    /// Why the hold ended this stream; `None` when it did not — the token is
    /// uncancelled, or SIGTERM cancelled it. Meaningful once the token is
    /// cancelled. The first cause recorded stays: a SIGTERM after the hold
    /// changes nothing.
    pub fn hold_cause(&self) -> Option<&HoldCause> {
        self.cause.get()
    }

    /// The status a stream that observed this token cancelled ends with:
    /// [`HoldCause::status`] for a hold's cause,
    /// [`server_shutting_down`](super::transport::server_shutting_down)
    /// otherwise.
    pub fn ending_status(&self) -> Status {
        match self.hold_cause() {
            Some(cause) => cause.status(),
            None => super::transport::server_shutting_down(),
        }
    }
}

impl std::ops::Deref for StreamShutdown {
    type Target = CancellationToken;

    fn deref(&self) -> &CancellationToken {
        &self.token
    }
}

/// The registry of live streams, under [`Shared::registry`]'s lock.
#[derive(Default)]
struct Registry {
    /// Set when the paced ending starts, or by [`GracefulShutdown::cancel`];
    /// never cleared. A closed registry enrolls nothing. Wins over `held`: a
    /// stream that enrolls once both are set is born cancelled with no cause,
    /// so it ends `server shutting down`.
    closed: bool,
    /// Set by the hold ending ([`GracefulShutdown::end_streams_for_hold`]) to
    /// its cause; never cleared — a held process exits rather than reopen
    /// (SDD SR-6.4). While set, a stream that enrolls is born cancelled
    /// carrying that cause, and nothing is entered.
    held: Option<HoldCause>,
    next_id: u64,
    entries: HashMap<u64, StreamShutdown>,
}

struct Shared {
    parent: CancellationToken,
    begun: CancellationToken,
    registry: Mutex<Registry>,
}

impl Shared {
    fn registry(&self) -> MutexGuard<'_, Registry> {
        // A panic while the lock was held leaves the map consistent: every
        // operation under it is a single insert, remove, flag set or drain.
        self.registry.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// The service's graceful-shutdown handle (SDD §5.6). Cloning shares it.
///
/// Created with the service in `main.rs` (and in the test builders). The
/// full-API server's shutdown signal ends the streams through it
/// ([`end_streams_after_drain`](super::transport::end_streams_after_drain)),
/// and the post-server sequence calls [`Self::cancel`] once the server has
/// returned, or its bound after GOAWAY has fired.
///
/// SIGTERM is the one shutdown signal: only it runs the paced ending and
/// [`Self::cancel`]. A disaster-recovery hold ends the streams through
/// [`Self::end_streams_for_hold`], which is not a shutdown and starts none of
/// its steps.
///
/// Who reads what:
///
/// * `SubscribeReadLayer` and `JoinSession` take their token only from
///   [`Self::enroll_stream`] — the subscription before its admission, the
///   join at its task-start check — and read that per-stream child everywhere
///   after: admission, catch-up waits, sends, the idle wait, the membership
///   replay, the bidirectional loop and the lease keeper. Each ending reads
///   the stream's own cause ([`StreamShutdown::ending_status`]): `UNAVAILABLE
///   "server shutting down"` for SIGTERM, `SERVICE_HELD` or
///   `UPSTREAM_UNAVAILABLE` for a hold. A per-stream child keeps the hot
///   paths off the shared tokens' locks: polling a token locks that token's
///   own mutex and waiter list.
/// * JoinSession's handler refusal reads [`Self::begun`].
/// * `CreateLayer` and `UpdateLayer` race [`Self::begun`] in their admission
///   and processing waits and at the external-value release budget
///   (`ExtReleaseGate::admit`), and an `UpdateLayer` commit races it in its
///   recharge between attempts (`UploadCommitCharge`'s `CommitCharge::hold`).
/// * A chained `UpdateLayer` races [`Self::begun`] in its chain-gate wait
///   (`storage::chain_gate`).
#[derive(Clone)]
pub struct GracefulShutdown {
    shared: Arc<Shared>,
}

impl Default for GracefulShutdown {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for GracefulShutdown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GracefulShutdown")
            .field("begun", &self.shared.begun.is_cancelled())
            .field("cancelled", &self.shared.parent.is_cancelled())
            .finish_non_exhaustive()
    }
}

impl GracefulShutdown {
    /// A handle with nothing begun, nothing cancelled and nothing enrolled.
    pub fn new() -> Self {
        let parent = CancellationToken::new();
        let begun = parent.child_token();
        Self {
            shared: Arc::new(Shared {
                parent,
                begun,
                registry: Mutex::new(Registry::default()),
            }),
        }
    }

    /// Enroll one live stream: its shutdown token — a fresh child of the
    /// parent, with the stream's own cause slot — and the guard that keeps it
    /// in the registry until dropped.
    ///
    /// Once the paced ending has started (or [`Self::cancel`] has run) the
    /// registry is closed: the token comes back already cancelled with no
    /// cause and nothing is enrolled, so the stream is refused at its first
    /// check, `server shutting down`. Once the hold ending has run
    /// ([`Self::end_streams_for_hold`]) the token likewise comes back already
    /// cancelled, carrying the hold's cause, so the stream ends as the hold
    /// ended every other one. When both are set, `closed` wins.
    pub fn enroll_stream(&self) -> (StreamShutdown, StreamEnrollment) {
        let mut registry = self.shared.registry();
        if registry.closed || registry.held.is_some() {
            let cause = if registry.closed {
                None
            } else {
                registry.held.clone()
            };
            drop(registry);
            return (
                StreamShutdown::born_cancelled(cause),
                StreamEnrollment {
                    shared: Arc::clone(&self.shared),
                    id: None,
                },
            );
        }
        let stream = StreamShutdown {
            token: self.shared.parent.child_token(),
            cause: Arc::default(),
        };
        let id = registry.next_id;
        registry.next_id = registry.next_id.wrapping_add(1);
        registry.entries.insert(id, stream.clone());
        drop(registry);
        (
            stream,
            StreamEnrollment {
                shared: Arc::clone(&self.shared),
                id: Some(id),
            },
        )
    }

    /// The hold ending (SDD SR-6.3): end every enrolled stream at once with
    /// `cause`, and refuse every later enrollment with it.
    ///
    /// Under the registry lock it sets the never-cleared `held` flag to
    /// `cause` (a later call replaces it, so late enrollments carry the latest
    /// cause), takes every enrolled stream out of the registry, and for each
    /// records `cause`, then cancels its token — so a stream woken by the
    /// cancel always reads the cause. A stream is therefore either ended here
    /// or born cancelled with the cause.
    ///
    /// Not a shutdown, and terminal for nothing: it never closes the registry
    /// and never touches `begun` or the parent, so the server keeps serving
    /// health checks, the uploads' `begun()` races and the scaler's
    /// `shutting_down` probe are unaffected, and a later SIGTERM still runs
    /// the normal paced shutdown. A stream this ended keeps its cause through
    /// that shutdown; one that enrolls after [`Self::cancel`] is born
    /// cancelled `server shutting down`. It cancels no per-session token.
    ///
    /// Once the paced ending has started the registry is already empty, so
    /// this ends nothing: the streams in that snapshot end in their batches,
    /// `server shutting down`.
    ///
    /// Logs one INFO line with the number of streams ended and the cause.
    pub fn end_streams_for_hold(&self, cause: HoldCause) {
        let mut registry = self.shared.registry();
        registry.held = Some(cause.clone());
        let ended: Vec<StreamShutdown> = registry.entries.drain().map(|(_, s)| s).collect();
        for stream in &ended {
            stream.end(Some(cause.clone()));
        }
        drop(registry);
        let (cause_label, incident) = match &cause {
            HoldCause::Held { incident } => ("held", incident.as_deref()),
            HoldCause::StaleVerdict => ("stale_verdict", None),
        };
        tracing::info!(
            streams = ended.len(),
            cause = cause_label,
            incident,
            "hold: ended every live SubscribeReadLayer and JoinSession stream at once; later \
             ones are refused as they enroll"
        );
    }

    /// Cancelled when the paced ending starts (t0), or by [`Self::cancel`].
    /// From then on new streaming RPCs are refused, and an upload waiting in
    /// admission, for processing room, at the external-value release budget or
    /// in a commit's recharge between attempts, or a chained `UpdateLayer`
    /// waiting in its chain gate (`storage::chain_gate`), is refused — with
    /// nothing applied, except a recharge after a counted retry, which answers
    /// the outcome-unknown `UNAVAILABLE` (SDD §18.1).
    ///
    /// Never a stream's ending: a stream that selected on it would end at t0
    /// together with every other such stream, outside the paced ending. Every
    /// production read of it is pinned, by file and by kind, in the
    /// shutdown-token guard (`refusal_and_shutdown_guards_test.rs`), so a new
    /// one has to be justified there.
    ///
    /// A shutdown signal only: SIGTERM is the one shutdown signal, and the
    /// hold ending ([`Self::end_streams_for_hold`]), a separate non-terminal
    /// ending, never cancels it.
    pub fn begun(&self) -> &CancellationToken {
        &self.shared.begun
    }

    /// Close the registry and cancel the parent: every stream still enrolled
    /// ends at once, `begun` is cancelled, and later enrollments are born
    /// cancelled with no cause (`server shutting down`, even after a hold).
    /// Streams the hold ending already ended are out of the registry, so they
    /// are not cancelled again and keep their hold cause. Idempotent.
    pub fn cancel(&self) {
        let mut registry = self.shared.registry();
        registry.closed = true;
        registry.entries.clear();
        // Under the lock, so no enrollment can observe an open registry after
        // this returns.
        self.shared.parent.cancel();
    }

    /// Whether the parent is cancelled: the paced ending has finished, or
    /// [`Self::cancel`] has run.
    ///
    /// Test builds only: a stream that waited on the parent this way would be
    /// a bare child of it — in no paced-ending snapshot, ending only with the
    /// last batch and not refused after t0. Production reads the parent only
    /// through [`Self::service_shutdown_token`], whose production reads the
    /// shutdown-token guard pins by file (today one: the bound after GOAWAY).
    #[cfg(any(test, feature = "test-support"))]
    pub fn is_cancelled(&self) -> bool {
        self.shared.parent.is_cancelled()
    }

    /// Resolves once the parent is cancelled. Test builds only, as
    /// [`Self::is_cancelled`].
    #[cfg(any(test, feature = "test-support"))]
    pub async fn cancelled(&self) {
        self.shared.parent.cancelled().await;
    }

    /// The number of streams enrolled now: neither ended by the hold nor
    /// taken by the paced ending's snapshot, and not yet dropped.
    pub fn enrolled(&self) -> usize {
        self.shared.registry().entries.len()
    }

    /// The parent token, for the bound after GOAWAY
    /// ([`serve_with_post_goaway_bound`](super::transport::serve_with_post_goaway_bound)),
    /// which waits for the paced ending to finish. Never handed to a stream:
    /// streams take theirs from [`Self::enroll_stream`].
    pub(crate) fn service_shutdown_token(&self) -> &CancellationToken {
        &self.shared.parent
    }

    /// Start the paced ending: under one lock, close the registry, cancel
    /// `begun` and take every enrolled stream's token. A stream is therefore
    /// either in the returned snapshot or born cancelled. A stream the hold
    /// ending already ended is out of the registry, so it is in no snapshot.
    fn begin(&self) -> Vec<CancellationToken> {
        let mut registry = self.shared.registry();
        registry.closed = true;
        self.shared.begun.cancel();
        registry
            .entries
            .drain()
            .map(|(_, stream)| stream.token)
            .collect()
    }
}

/// Keeps one stream in the [`GracefulShutdown`] registry; dropping it removes
/// the stream (a no-op once the paced ending or the hold ending has taken it
/// out). Held by the stream's task for the task's whole life, so a stream that
/// ended on its own before the paced ending starts is not in its snapshot.
#[must_use = "dropping the enrollment removes the stream from the registry at once"]
pub struct StreamEnrollment {
    shared: Arc<Shared>,
    /// `None` for a stream born cancelled, which was never entered.
    id: Option<u64>,
}

impl Drop for StreamEnrollment {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            self.shared.registry().entries.remove(&id);
        }
    }
}

impl std::fmt::Debug for StreamEnrollment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamEnrollment")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

/// End every enrolled stream, paced, then cancel the parent. Call it at t0.
///
/// At once: close the registry, cancel `begun` and snapshot the N enrolled
/// streams (`GracefulShutdown::begin`), and shuffle the snapshot with `rng`
/// (joins and subscriptions together, not grouped by client). Then, for batch
/// k of T ([`StreamPacing::batches`]), at t0 + k·Δ, cancel entries
/// ⌊kN/T⌋..⌊(k+1)N/T⌋. After the last batch, cancel the parent, which ends
/// anything left (nothing, unless [`GracefulShutdown::cancel`] ran first).
///
/// Every ending is the stream's own shutdown observation, so each still ends
/// `UNAVAILABLE "server shutting down"` (no hold cause is recorded on these
/// streams), and a stream not yet reached that observes its session's deletion
/// first ends as a deletion.
///
/// Logs one INFO line at the start (`streams`, `window_ms`, `rate_per_sec`)
/// and one at the end.
pub async fn end_streams_paced<R>(shutdown: &GracefulShutdown, pacing: StreamPacing, rng: &mut R)
where
    R: rand::Rng + ?Sized,
{
    let t0 = tokio::time::Instant::now();
    let mut snapshot = shutdown.begin();
    snapshot.shuffle(rng);
    let streams = snapshot.len();
    let window = pacing.window(streams);
    let batches = pacing.batches(streams);
    tracing::info!(
        streams,
        window_ms = u64::try_from(window.as_millis()).unwrap_or(u64::MAX),
        rate_per_sec = pacing.rate_per_sec,
        "drain complete and the server's GOAWAY under way; ending live SubscribeReadLayer and \
         JoinSession streams with UNAVAILABLE, paced"
    );
    // ⌊kN/T⌋, in u128 so no product of a batch index and a stream count can
    // overflow; never above N.
    let boundary = |k: usize| -> usize {
        let at = (k as u128).saturating_mul(streams as u128) / (batches as u128);
        usize::try_from(at).unwrap_or(streams).min(streams)
    };
    for k in 0..batches {
        let at =
            t0 + STREAM_END_BATCH_INTERVAL.saturating_mul(u32::try_from(k).unwrap_or(u32::MAX));
        tokio::time::sleep_until(at).await;
        for token in &snapshot[boundary(k)..boundary(k + 1)] {
            token.cancel();
        }
    }
    shutdown.shared.parent.cancel();
    tracing::info!(
        streams,
        elapsed_ms = u64::try_from(t0.elapsed().as_millis()).unwrap_or(u64::MAX),
        "every live SubscribeReadLayer and JoinSession stream has been told to end"
    );
}

#[cfg(test)]
mod tests {
    //! The enrollment registry, the window arithmetic and the order of the
    //! paced ending, on paused time: every wait is a virtual-clock deadline, so
    //! each instant is asserted exactly. The hold ending — cause before
    //! cancel, the `held` flag, `closed` winning over it, and what it leaves
    //! untouched — is pinned at the end; its endings through the real streaming
    //! handlers are `tests/integration/grpc/hold_ending_test.rs`. The schedule
    //! as `main.rs` runs it —
    //! drain, GOAWAY head start, batch instants and sizes, the rollback
    //! switches — is pinned through `end_streams_after_drain` in
    //! `transport.rs`'s tests; the real-server ramp is
    //! `tests/integration/grpc/shutdown_ending_test.rs`.

    use super::*;
    use rand::SeedableRng;
    use tokio::time::Instant;

    const DELTA: Duration = STREAM_END_BATCH_INTERVAL;

    /// The virtual instant `token` is cancelled at.
    fn cancelled_at(token: &CancellationToken) -> tokio::task::JoinHandle<Instant> {
        let token = token.clone();
        tokio::spawn(async move {
            token.cancelled().await;
            Instant::now()
        })
    }

    /// The virtual instant `shutdown`'s parent is cancelled at.
    fn parent_cancelled_at(shutdown: &GracefulShutdown) -> tokio::task::JoinHandle<Instant> {
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            shutdown.cancelled().await;
            Instant::now()
        })
    }

    fn seeded() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(7)
    }

    /// Sink for [`capture_json_events`] (the tree-wide `CapturingWriter`
    /// idiom, as in `transport.rs`'s tests).
    #[derive(Clone)]
    struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

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

    /// Drive `f` under a JSON capture and return every event it emitted.
    async fn capture_json_events<F, T>(f: F) -> (T, Vec<serde_json::Value>)
    where
        F: std::future::Future<Output = T>,
    {
        use tracing::instrument::WithSubscriber;
        let buf = Arc::new(Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(Arc::clone(&buf)))
            .with_max_level(tracing::Level::TRACE)
            .with_ansi(false)
            .finish();
        let value = f.with_subscriber(tracing::Dispatch::new(subscriber)).await;
        let raw = buf.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let events = String::from_utf8(raw)
            .expect("captured tracing output must be UTF-8")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("each captured line must be JSON"))
            .collect();
        (value, events)
    }

    const START_MSG: &str = "drain complete and the server's GOAWAY under way; ending live \
                             SubscribeReadLayer and JoinSession streams with UNAVAILABLE, paced";
    const END_MSG: &str =
        "every live SubscribeReadLayer and JoinSession stream has been told to end";

    // ── The window arithmetic ────────────────────────────────────────────────

    /// W = min(N / R, W_cap) and T = max(1, ⌈W / Δ⌉): at the defaults (25/s,
    /// 20 s) one stream ends in one batch, 500 streams fill the 20 s cap in
    /// 200 batches, 1,000 streams are capped at the same 20 s, and an empty
    /// registry is still one batch (the parent is cancelled at t0). Either
    /// switch at 0 is a zero window — every stream at once — and no stream
    /// count overflows the arithmetic.
    #[test]
    fn stream_pacing_window_is_streams_over_rate_capped_and_either_switch_ends_all_at_once() {
        let d = StreamPacing::DEFAULT;
        assert_eq!(
            d,
            StreamPacing {
                rate_per_sec: 25,
                max_window: Duration::from_secs(20)
            },
            "the production defaults are 25 endings per second over at most 20 s"
        );
        assert_eq!(StreamPacing::default(), d);
        for (streams, window, batches) in [
            (0, Duration::ZERO, 1),
            (1, Duration::from_millis(40), 1),
            (3, Duration::from_millis(120), 2),
            (500, Duration::from_secs(20), 200),
            (1_000, Duration::from_secs(20), 200),
            (usize::MAX, Duration::from_secs(20), 200),
        ] {
            assert_eq!(d.window(streams), window, "{streams} streams");
            assert_eq!(d.batches(streams), batches, "{streams} streams");
        }
        for off in [
            StreamPacing::ALL_AT_ONCE,
            StreamPacing {
                rate_per_sec: 0,
                max_window: Duration::from_secs(20),
            },
            StreamPacing {
                rate_per_sec: 25,
                max_window: Duration::ZERO,
            },
        ] {
            assert_eq!(off.window(1_000), Duration::ZERO, "{off:?}");
            assert_eq!(off.batches(1_000), 1, "{off:?}");
        }
    }

    // ── The enrollment registry ──────────────────────────────────────────────

    /// A stream that enrolls once the paced ending has started gets a token
    /// that is already cancelled — so it is refused at its first check — and
    /// is not entered in the registry, while the stream enrolled before the
    /// start keeps serving until its own batch. Dropping the born-cancelled
    /// enrollment removes nothing.
    #[tokio::test(start_paused = true)]
    async fn enroll_stream_after_the_paced_ending_started_is_born_cancelled_and_not_entered() {
        let shutdown = GracefulShutdown::new();
        let (early, _early_enrollment) = shutdown.enroll_stream();
        assert!(!early.is_cancelled());
        assert_eq!(shutdown.enrolled(), 1);

        // One stream at 1/s: its batch is the last of ten, at t0 + 900 ms.
        let pacing = StreamPacing {
            rate_per_sec: 1,
            max_window: Duration::from_secs(20),
        };
        let ending = tokio::spawn({
            let shutdown = shutdown.clone();
            async move { end_streams_paced(&shutdown, pacing, &mut seeded()).await }
        });
        tokio::time::timeout(Duration::from_secs(1), shutdown.begun().cancelled())
            .await
            .expect("`begun` is cancelled as the paced ending starts");
        assert!(
            !shutdown.is_cancelled(),
            "the parent is cancelled only after the last batch"
        );
        assert_eq!(
            shutdown.enrolled(),
            0,
            "the snapshot took every enrolled stream out of the registry"
        );

        let (late, late_enrollment) = shutdown.enroll_stream();
        assert!(
            late.is_cancelled(),
            "a stream enrolling once the paced ending has started must be born cancelled, so \
             it is refused at its first check"
        );
        assert_eq!(
            shutdown.enrolled(),
            0,
            "a born-cancelled stream is not entered, so the paced ending does not count it"
        );
        assert!(
            !early.is_cancelled(),
            "the stream enrolled before the start is not yet reached: it keeps serving"
        );
        drop(late_enrollment);
        assert_eq!(shutdown.enrolled(), 0);

        ending.await.expect("the paced ending must not panic");
        assert!(early.is_cancelled() && shutdown.is_cancelled());
    }

    /// A stream that ended on its own before the start — its enrollment
    /// dropped — is out of the registry and out of the snapshot: the paced
    /// ending counts two streams, not three, so its two batches end at t0 and
    /// t0 + Δ and the parent is cancelled at t0 + Δ (three would put the
    /// parent at t0 + 2Δ).
    #[tokio::test(start_paused = true)]
    async fn a_dropped_enrollment_leaves_the_registry_and_the_snapshot() {
        let shutdown = GracefulShutdown::new();
        let (kept_a, _kept_a) = shutdown.enroll_stream();
        let (gone, gone_enrollment) = shutdown.enroll_stream();
        let (kept_b, _kept_b) = shutdown.enroll_stream();
        assert_eq!(shutdown.enrolled(), 3);
        drop(gone_enrollment);
        assert_eq!(
            shutdown.enrolled(),
            2,
            "dropping an enrollment removes exactly its stream"
        );

        let kept = [cancelled_at(&kept_a), cancelled_at(&kept_b)];
        let gone_at = cancelled_at(&gone);
        let parent_at = parent_cancelled_at(&shutdown);
        // 10/s: two streams are a 200 ms window, two batches.
        let pacing = StreamPacing {
            rate_per_sec: 10,
            max_window: Duration::from_secs(20),
        };
        let t0 = Instant::now();
        let ((), events) =
            capture_json_events(end_streams_paced(&shutdown, pacing, &mut seeded())).await;

        let starts: Vec<_> = events
            .iter()
            .filter(|e| e["fields"]["message"] == START_MSG)
            .collect();
        assert_eq!(starts.len(), 1, "one start line; got {events:#?}");
        assert_eq!(
            starts[0]["fields"]["streams"], 2,
            "the snapshot holds the two live streams, not the dropped one"
        );
        let mut kept_at = Vec::new();
        for at in kept {
            kept_at.push(at.await.expect("watcher") - t0);
        }
        kept_at.sort();
        assert_eq!(kept_at, vec![Duration::ZERO, DELTA], "one per batch");
        assert_eq!(parent_at.await.expect("watcher") - t0, DELTA);
        assert_eq!(
            gone_at.await.expect("watcher") - t0,
            DELTA,
            "the dropped stream's token is still the parent's child, so the parent's cancel \
             reaches it; no batch did"
        );
    }

    /// `cancel()` ends every stream still enrolled at once — here in the
    /// middle of a paced ending, so the streams not yet reached end with it —
    /// cancels `begun` and the parent, and refuses every later enrollment.
    /// Idempotent.
    #[tokio::test(start_paused = true)]
    async fn cancel_ends_every_remaining_stream_at_once_and_refuses_later_enrollment() {
        let shutdown = GracefulShutdown::new();
        let mut tokens = Vec::new();
        let mut enrollments = Vec::new();
        for _ in 0..10 {
            let (token, enrollment) = shutdown.enroll_stream();
            tokens.push(token);
            enrollments.push(enrollment);
        }
        let watchers: Vec<_> = tokens.iter().map(|token| cancelled_at(token)).collect();
        // 1/s: ten streams over 10 s, one every second.
        let pacing = StreamPacing {
            rate_per_sec: 1,
            max_window: Duration::from_secs(20),
        };
        let t0 = Instant::now();
        let ending = tokio::spawn({
            let shutdown = shutdown.clone();
            async move { end_streams_paced(&shutdown, pacing, &mut seeded()).await }
        });
        tokio::time::sleep(Duration::from_millis(1_550)).await;
        let reached = tokens.iter().filter(|t| t.is_cancelled()).count();
        assert_eq!(
            reached, 1,
            "1.55 s in, only the first second's batch has run"
        );

        shutdown.cancel();
        for (i, at) in watchers.into_iter().enumerate() {
            let at = at.await.expect("watcher") - t0;
            assert!(
                at <= Duration::from_millis(1_550),
                "stream {i} ended at {at:?}, after cancel(): cancel() must end every stream \
                 still enrolled at once"
            );
        }
        assert!(shutdown.begun().is_cancelled() && shutdown.is_cancelled());
        assert_eq!(shutdown.enrolled(), 0);
        let (late, _late) = shutdown.enroll_stream();
        assert!(late.is_cancelled(), "cancel() closes the registry");
        assert_eq!(shutdown.enrolled(), 0);
        shutdown.cancel();
        assert!(shutdown.is_cancelled(), "a second cancel() is a no-op");
        ending.await.expect("the paced ending must not panic");
    }

    /// `cancel()` before any paced ending — the post-server `cancel()` on a
    /// shutdown whose server returned early, and the tests' direct cancel —
    /// ends every enrolled stream and `begun` at once, and a later enrollment
    /// is born cancelled.
    #[tokio::test]
    async fn cancel_without_a_paced_ending_ends_everything_and_closes_the_registry() {
        let shutdown = GracefulShutdown::new();
        assert!(!shutdown.begun().is_cancelled() && !shutdown.is_cancelled());
        let (a, _a) = shutdown.enroll_stream();
        let (b, _b) = shutdown.enroll_stream();
        shutdown.cancel();
        assert!(a.is_cancelled() && b.is_cancelled());
        assert!(shutdown.begun().is_cancelled() && shutdown.is_cancelled());
        assert_eq!(shutdown.enrolled(), 0);
        let (late, _late) = shutdown.enroll_stream();
        assert!(late.is_cancelled());
        assert_eq!(shutdown.enrolled(), 0);
        tokio::time::timeout(Duration::from_secs(1), shutdown.cancelled())
            .await
            .expect("cancelled() resolves once cancel() has run");
    }

    // ── The order ────────────────────────────────────────────────────────────

    /// An `RngCore` that counts how often it is drawn from.
    struct CountingRng {
        inner: rand::rngs::StdRng,
        draws: usize,
    }

    impl rand::RngCore for CountingRng {
        fn next_u32(&mut self) -> u32 {
            self.draws += 1;
            self.inner.next_u32()
        }
        fn next_u64(&mut self) -> u64 {
            self.draws += 1;
            self.inner.next_u64()
        }
        fn fill_bytes(&mut self, dst: &mut [u8]) {
            self.draws += 1;
            self.inner.fill_bytes(dst);
        }
    }

    /// The paced ending ends every enrolled stream exactly once, in an order
    /// the shuffle draws from the given RNG — not the order the streams
    /// enrolled in. Sixteen streams at 10/s are sixteen batches of one, so
    /// the end order is the order of the cancel instants.
    #[tokio::test(start_paused = true)]
    async fn end_streams_paced_ends_a_shuffled_permutation_of_the_enrolled_streams() {
        const STREAMS: usize = 16;
        let shutdown = GracefulShutdown::new();
        let mut enrollments = Vec::new();
        let mut watchers = Vec::new();
        for _ in 0..STREAMS {
            let (token, enrollment) = shutdown.enroll_stream();
            watchers.push(cancelled_at(&token));
            enrollments.push(enrollment);
        }
        let pacing = StreamPacing {
            rate_per_sec: 10,
            max_window: Duration::from_secs(20),
        };
        let mut rng = CountingRng {
            inner: seeded(),
            draws: 0,
        };
        let t0 = Instant::now();
        end_streams_paced(&shutdown, pacing, &mut rng).await;

        let mut ended: Vec<(Duration, usize)> = Vec::new();
        for (enrolled_as, at) in watchers.into_iter().enumerate() {
            ended.push((at.await.expect("watcher") - t0, enrolled_as));
        }
        ended.sort();
        let instants: Vec<Duration> = ended.iter().map(|(at, _)| *at).collect();
        let expected: Vec<Duration> = (0..STREAMS as u32).map(|k| DELTA * k).collect();
        assert_eq!(
            instants, expected,
            "one stream per batch, one batch every {DELTA:?}"
        );
        let order: Vec<usize> = ended.iter().map(|(_, i)| *i).collect();
        let mut seen = order.clone();
        seen.sort_unstable();
        assert_eq!(
            seen,
            (0..STREAMS).collect::<Vec<_>>(),
            "every enrolled stream ends exactly once"
        );
        assert_ne!(
            order,
            (0..STREAMS).collect::<Vec<_>>(),
            "the streams must not end in the order they enrolled"
        );
        // rand's shuffle packs several indices into one draw, so only "drawn
        // from at all" is stable across rand versions.
        assert!(
            rng.draws > 0,
            "the end order must be shuffled with the given RNG; it was never drawn from"
        );
    }

    /// The operator's two lines: one INFO when the paced ending starts, with
    /// the stream count, the window and the rate, and one INFO when every
    /// stream has been told to end.
    #[tokio::test(start_paused = true)]
    async fn end_streams_paced_logs_one_start_and_one_end_info_line() {
        let shutdown = GracefulShutdown::new();
        let mut enrollments = Vec::new();
        for _ in 0..5 {
            enrollments.push(shutdown.enroll_stream());
        }
        let pacing = StreamPacing {
            rate_per_sec: 10,
            max_window: Duration::from_secs(20),
        };
        let ((), events) =
            capture_json_events(end_streams_paced(&shutdown, pacing, &mut seeded())).await;

        let with = |msg: &str| -> Vec<&serde_json::Value> {
            events
                .iter()
                .filter(|e| e["fields"]["message"] == msg)
                .collect()
        };
        let start = with(START_MSG);
        assert_eq!(start.len(), 1, "one start line; got {events:#?}");
        assert_eq!(start[0]["level"], "INFO", "{}", start[0]);
        assert_eq!(start[0]["fields"]["streams"], 5, "{}", start[0]);
        assert_eq!(start[0]["fields"]["window_ms"], 500, "{}", start[0]);
        assert_eq!(start[0]["fields"]["rate_per_sec"], 10, "{}", start[0]);
        let end = with(END_MSG);
        assert_eq!(end.len(), 1, "one end line; got {events:#?}");
        assert_eq!(end[0]["level"], "INFO", "{}", end[0]);
        assert_eq!(end[0]["fields"]["streams"], 5, "{}", end[0]);
        assert_eq!(
            end[0]["fields"]["elapsed_ms"], 400,
            "the last of five batches runs at t0 + 4Δ; {}",
            end[0]
        );
    }

    // ── The hold ending (SDD SR-6.3) ─────────────────────────────────────────

    const INCIDENT: &str = "20261013t1708-a3f9";

    fn held(incident: &str) -> HoldCause {
        HoldCause::Held {
            incident: Some(incident.to_owned()),
        }
    }

    /// The `ErrorInfo` reason of `status`, as a client decodes it.
    fn reason_of(status: &Status) -> String {
        use tonic_types::StatusExt;
        status
            .get_details_error_info()
            .unwrap_or_else(|| panic!("the ending must carry an ErrorInfo: {status:?}"))
            .reason
    }

    /// A waker that records, at the instant it is woken, the cause its stream
    /// carries. `CancellationToken::cancel` wakes the token's waiters
    /// synchronously, inside the cancel, so what this records is what a
    /// stream woken by that cancel reads first — before anything that runs
    /// after the cancel returns.
    struct CauseAtWake {
        stream: StreamShutdown,
        seen: Mutex<Vec<Option<HoldCause>>>,
    }

    impl std::task::Wake for CauseAtWake {
        fn wake(self: Arc<Self>) {
            self.wake_by_ref();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            let cause = self.stream.hold_cause().cloned();
            self.seen
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .push(cause);
        }
    }

    /// **Cause before cancel.** Every enrolled stream's waiter, woken by the
    /// hold's cancel of its token, already reads the hold's cause at the
    /// moment it is woken — so no stream can end `server shutting down` for a
    /// hold. The probe is a waker called synchronously inside the cancel; a
    /// hold that cancelled first and recorded after would be seen with no
    /// cause.
    #[test]
    fn end_streams_for_hold_records_each_cause_before_it_cancels_so_a_woken_stream_reads_it() {
        use std::future::Future as _;

        let shutdown = GracefulShutdown::new();
        let mut streams = Vec::new();
        let mut enrollments = Vec::new();
        for _ in 0..3 {
            let (stream, enrollment) = shutdown.enroll_stream();
            streams.push(stream);
            enrollments.push(enrollment);
        }
        let probes: Vec<Arc<CauseAtWake>> = streams
            .iter()
            .map(|stream| {
                Arc::new(CauseAtWake {
                    stream: stream.clone(),
                    seen: Mutex::default(),
                })
            })
            .collect();
        let mut waits: Vec<_> = streams
            .iter()
            .map(|stream| Box::pin(stream.cancelled()))
            .collect();
        for (wait, probe) in waits.iter_mut().zip(&probes) {
            let waker = std::task::Waker::from(Arc::clone(probe));
            let mut cx = std::task::Context::from_waker(&waker);
            assert!(
                wait.as_mut().poll(&mut cx).is_pending(),
                "precondition: the stream's wait is parked on its uncancelled token"
            );
        }

        shutdown.end_streams_for_hold(held(INCIDENT));

        for (i, probe) in probes.iter().enumerate() {
            let seen = probe.seen.lock().unwrap_or_else(|p| p.into_inner()).clone();
            assert_eq!(
                seen.first(),
                Some(&Some(held(INCIDENT))),
                "stream {i}: woken by the hold's cancel, it must already read the hold's cause; \
                 it read {seen:?} (None is a cancel that came before its cause)"
            );
        }
    }

    /// The hold ends every enrolled stream at once — each cancelled, each
    /// carrying the cause, each ending `SERVICE_HELD` naming the incident —
    /// and takes them out of the registry, so dropping their enrollments
    /// afterwards removes nothing.
    #[test]
    fn end_streams_for_hold_ends_every_enrolled_stream_with_its_cause_and_drains_the_registry() {
        let shutdown = GracefulShutdown::new();
        let mut streams = Vec::new();
        let mut enrollments = Vec::new();
        for _ in 0..3 {
            let (stream, enrollment) = shutdown.enroll_stream();
            streams.push(stream);
            enrollments.push(enrollment);
        }
        assert_eq!(shutdown.enrolled(), 3);

        shutdown.end_streams_for_hold(held(INCIDENT));

        assert_eq!(
            shutdown.enrolled(),
            0,
            "the hold takes every stream it ended out of the registry"
        );
        for (i, stream) in streams.iter().enumerate() {
            assert!(stream.is_cancelled(), "stream {i} must be ended at once");
            assert_eq!(stream.hold_cause(), Some(&held(INCIDENT)), "stream {i}");
            let status = stream.ending_status();
            assert_eq!(status.code(), tonic::Code::Unavailable, "stream {i}");
            assert_eq!(reason_of(&status), "SERVICE_HELD", "stream {i}");
            assert!(
                status.message().contains(INCIDENT),
                "stream {i}: the ending names the incident; got {status:?}"
            );
        }
        drop(enrollments);
        assert_eq!(shutdown.enrolled(), 0);
    }

    /// **Not a shutdown.** The hold never closes the registry, never cancels
    /// `begun` (the uploads' and the join handler's shutdown signal) and never
    /// cancels the parent: the server keeps answering, and a later SIGTERM
    /// still runs the normal shutdown.
    #[test]
    fn end_streams_for_hold_leaves_begun_the_parent_and_the_registry_open() {
        let shutdown = GracefulShutdown::new();
        let (_stream, _enrollment) = shutdown.enroll_stream();

        shutdown.end_streams_for_hold(held(INCIDENT));

        assert!(
            !shutdown.begun().is_cancelled(),
            "a hold is not a shutdown: `begun` must stay uncancelled"
        );
        assert!(
            !shutdown.is_cancelled(),
            "a hold is not a shutdown: the parent must stay uncancelled"
        );
        assert!(
            !shutdown.shared.registry().closed,
            "a hold must not close the registry: only SIGTERM does"
        );
    }

    /// **The `held` flag.** Once the hold has run, a stream that enrolls is
    /// born cancelled, carrying the hold's cause, and is not entered — so a
    /// stream that passed a hold gate just before the hold ends as every
    /// other stream did. For each cause, including a hold whose marker names
    /// no incident.
    #[test]
    fn enroll_stream_after_a_hold_is_born_cancelled_with_the_hold_cause_and_not_entered() {
        for cause in [
            held(INCIDENT),
            HoldCause::Held { incident: None },
            HoldCause::StaleVerdict,
        ] {
            let shutdown = GracefulShutdown::new();
            shutdown.end_streams_for_hold(cause.clone());

            let (late, late_enrollment) = shutdown.enroll_stream();
            assert!(
                late.is_cancelled(),
                "{cause:?}: a stream enrolling after the hold must be born cancelled"
            );
            assert_eq!(late.hold_cause(), Some(&cause), "{cause:?}");
            assert_eq!(
                shutdown.enrolled(),
                0,
                "{cause:?}: a born-cancelled stream is not entered"
            );
            drop(late_enrollment);
            assert_eq!(shutdown.enrolled(), 0, "{cause:?}");

            let status = late.ending_status();
            assert_eq!(status.code(), tonic::Code::Unavailable, "{cause:?}");
            match &cause {
                HoldCause::Held { incident } => {
                    assert_eq!(reason_of(&status), "SERVICE_HELD", "{cause:?}");
                    let named = incident.as_deref().unwrap_or("unknown");
                    assert_eq!(
                        status.message(),
                        format!("service is offline for disaster recovery (incident {named})"),
                        "{cause:?}"
                    );
                }
                HoldCause::StaleVerdict => {
                    assert_eq!(reason_of(&status), "UPSTREAM_UNAVAILABLE", "{cause:?}");
                    assert_eq!(
                        status.message(),
                        super::super::transport::HOLD_VERDICT_STALE_MSG
                    );
                }
            }
        }
    }

    /// **`closed` wins over `held`**, in either order: once SIGTERM's
    /// `cancel()` has closed the registry, a stream that enrolls is born
    /// cancelled with no cause and ends `server shutting down`, even though a
    /// hold ran — the replica is going away, and the retry belongs on another.
    #[test]
    fn enroll_stream_with_the_registry_closed_and_held_is_born_cancelled_with_no_cause() {
        for hold_first in [true, false] {
            let shutdown = GracefulShutdown::new();
            if hold_first {
                shutdown.end_streams_for_hold(held(INCIDENT));
                shutdown.cancel();
            } else {
                shutdown.cancel();
                shutdown.end_streams_for_hold(held(INCIDENT));
            }
            let (late, _late) = shutdown.enroll_stream();
            assert!(late.is_cancelled(), "hold_first={hold_first}");
            assert_eq!(
                late.hold_cause(),
                None,
                "hold_first={hold_first}: with the registry closed, `closed` wins over `held`"
            );
            let status = late.ending_status();
            assert_eq!(
                status.message(),
                "server shutting down",
                "hold_first={hold_first}"
            );
            assert_eq!(
                reason_of(&status),
                "SERVER_SHUTTING_DOWN",
                "hold_first={hold_first}"
            );
        }
    }

    /// **SIGTERM after a hold.** A stream the hold ended keeps its
    /// `SERVICE_HELD` ending through the later `cancel()` (its cause was
    /// recorded first, and it is out of the registry, so it is not cancelled
    /// again), and `cancel()` still does everything a shutdown does: `begun`
    /// and the parent cancelled.
    #[test]
    fn cancel_after_a_hold_keeps_each_ended_streams_hold_cause_and_still_shuts_down() {
        let shutdown = GracefulShutdown::new();
        let (stream, _enrollment) = shutdown.enroll_stream();
        shutdown.end_streams_for_hold(held(INCIDENT));

        shutdown.cancel();

        assert_eq!(
            stream.hold_cause(),
            Some(&held(INCIDENT)),
            "a stream the hold ended keeps its cause through SIGTERM"
        );
        assert_eq!(reason_of(&stream.ending_status()), "SERVICE_HELD");
        assert!(
            shutdown.begun().is_cancelled() && shutdown.is_cancelled(),
            "SIGTERM after a hold is a full shutdown"
        );
    }

    /// A second hold — the verdict changing from a fleet hold to a stale one —
    /// replaces the cause later enrollments are born with, while a stream the
    /// first hold ended keeps the first cause.
    #[test]
    fn a_second_hold_changes_the_late_enrolment_cause_and_each_ended_stream_keeps_its_first() {
        let shutdown = GracefulShutdown::new();
        let (first, _first) = shutdown.enroll_stream();
        shutdown.end_streams_for_hold(held(INCIDENT));
        shutdown.end_streams_for_hold(HoldCause::StaleVerdict);

        assert_eq!(first.hold_cause(), Some(&held(INCIDENT)));
        let (late, _late) = shutdown.enroll_stream();
        assert_eq!(
            late.hold_cause(),
            Some(&HoldCause::StaleVerdict),
            "late enrollments carry the latest hold's cause"
        );
    }

    /// SIGTERM's paced ending after a hold: the streams the hold ended are in
    /// no snapshot — the paced ending counts none and so cancels no stream a
    /// second time — and the parent is still cancelled at the end. The ended
    /// streams keep their hold cause.
    #[tokio::test(start_paused = true)]
    async fn end_streams_paced_after_a_hold_finds_an_empty_snapshot_and_still_cancels_the_parent() {
        let shutdown = GracefulShutdown::new();
        let mut streams = Vec::new();
        let mut enrollments = Vec::new();
        for _ in 0..3 {
            let (stream, enrollment) = shutdown.enroll_stream();
            streams.push(stream);
            enrollments.push(enrollment);
        }
        shutdown.end_streams_for_hold(held(INCIDENT));
        let pacing = StreamPacing {
            rate_per_sec: 10,
            max_window: Duration::from_secs(20),
        };

        let ((), events) =
            capture_json_events(end_streams_paced(&shutdown, pacing, &mut seeded())).await;

        let starts: Vec<_> = events
            .iter()
            .filter(|e| e["fields"]["message"] == START_MSG)
            .collect();
        assert_eq!(starts.len(), 1, "one start line; got {events:#?}");
        assert_eq!(
            starts[0]["fields"]["streams"], 0,
            "the streams the hold ended are out of the registry, so in no snapshot"
        );
        assert!(shutdown.is_cancelled() && shutdown.begun().is_cancelled());
        for (i, stream) in streams.iter().enumerate() {
            assert_eq!(stream.hold_cause(), Some(&held(INCIDENT)), "stream {i}");
        }
    }

    /// A hold once the paced ending has started finds the registry already
    /// empty, so it ends nothing: the snapshot's stream ends in its batch,
    /// with no cause — `server shutting down`, as SIGTERM ends it.
    #[tokio::test(start_paused = true)]
    async fn a_hold_after_the_paced_ending_started_ends_nothing_and_the_snapshot_ends_as_a_shutdown(
    ) {
        let shutdown = GracefulShutdown::new();
        let (early, _early_enrollment) = shutdown.enroll_stream();
        // One stream at 1/s: its batch is the last of ten, at t0 + 900 ms.
        let pacing = StreamPacing {
            rate_per_sec: 1,
            max_window: Duration::from_secs(20),
        };
        let ending = tokio::spawn({
            let shutdown = shutdown.clone();
            async move { end_streams_paced(&shutdown, pacing, &mut seeded()).await }
        });
        tokio::time::timeout(Duration::from_secs(1), shutdown.begun().cancelled())
            .await
            .expect("`begun` is cancelled as the paced ending starts");

        shutdown.end_streams_for_hold(held(INCIDENT));
        assert!(
            !early.is_cancelled(),
            "the hold must not reach a stream the paced ending's snapshot holds"
        );

        ending.await.expect("the paced ending must not panic");
        assert!(early.is_cancelled(), "its batch ended it");
        assert_eq!(
            early.hold_cause(),
            None,
            "a stream the paced ending ended carries no hold cause"
        );
        assert_eq!(early.ending_status().message(), "server shutting down");
    }

    /// The operator's line: one INFO per hold, with the number of streams it
    /// ended, the cause, and — for a fleet hold — the incident.
    #[tokio::test]
    async fn end_streams_for_hold_logs_one_info_line_with_the_stream_count_and_the_cause() {
        for (cause, label, incident) in [
            (held(INCIDENT), "held", Some(INCIDENT)),
            (HoldCause::StaleVerdict, "stale_verdict", None),
        ] {
            let shutdown = GracefulShutdown::new();
            let (_a, _a_enrollment) = shutdown.enroll_stream();
            let (_b, _b_enrollment) = shutdown.enroll_stream();

            let ((), events) =
                capture_json_events(async { shutdown.end_streams_for_hold(cause.clone()) }).await;

            assert_eq!(
                events.len(),
                1,
                "{label}: one line per hold; got {events:#?}"
            );
            let line = &events[0];
            assert_eq!(line["level"], "INFO", "{label}: {line}");
            assert_eq!(line["fields"]["streams"], 2, "{label}: {line}");
            assert_eq!(line["fields"]["cause"], label, "{label}: {line}");
            match incident {
                Some(incident) => assert_eq!(line["fields"]["incident"], incident, "{line}"),
                None => assert!(
                    line["fields"].get("incident").is_none(),
                    "a stale verdict has no incident; got {line}"
                ),
            }
        }
    }

    /// The status each cause ends a stream with, and the no-cause ending:
    /// `SERVICE_HELD` for a fleet hold, `UPSTREAM_UNAVAILABLE` for a stale
    /// verdict, `SERVER_SHUTTING_DOWN` for SIGTERM — all `UNAVAILABLE`, so
    /// every client retries.
    #[test]
    fn hold_cause_status_and_ending_status_map_each_cause_to_its_reason() {
        assert_eq!(reason_of(&held(INCIDENT).status()), "SERVICE_HELD");
        assert_eq!(
            reason_of(&HoldCause::StaleVerdict.status()),
            "UPSTREAM_UNAVAILABLE"
        );
        let shutdown = GracefulShutdown::new();
        let (stream, _enrollment) = shutdown.enroll_stream();
        shutdown.cancel();
        let status = stream.ending_status();
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert_eq!(status.message(), "server shutting down");
        assert_eq!(reason_of(&status), "SERVER_SHUTTING_DOWN");
        for cause in [held(INCIDENT), HoldCause::StaleVerdict] {
            assert_eq!(cause.status().code(), tonic::Code::Unavailable, "{cause:?}");
        }
    }
}
