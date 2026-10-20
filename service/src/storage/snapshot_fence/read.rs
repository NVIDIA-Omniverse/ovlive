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

//! The fenced snapshot read (Spec SDD SR-5.3): the proven read of a layer
//! above the atomic-read ceiling.
//!
//! [`read_object_snapshot_fenced`] runs, per attempt:
//!
//! 1. **acquire** — [`acquire_fence`] until it grants, with the caller's
//!    [`FencedReadHook`] deciding how to wait between `RETRY_AFTER`s;
//! 2. **the fan-out under the fence** — the unfenced read's bounded fan-out
//!    on the snapshot lane (`storage::diff_read`'s windows, marked/unmarked
//!    children rule and decode), run as a pass on the `fanout-read` thread
//!    (`storage::fanout_thread`), with a **renewal timer** beside it in this
//!    read's own task, on the data pool, renewing the lease every
//!    `lease(L) / 4` for the whole hold: a window is bounded in commands,
//!    not bytes, so renewing only between windows could let one slow window
//!    outlast the lease. A `LOST` renewal stops the fan-out;
//! 3. **release** — [`FenceLease::release`], which proves the read in the same
//!    script that gives the lease up: `VERIFIED` returns the state at the
//!    cut (the `seq` acquire read), `BROKEN` retries the whole read, up to
//!    [`FENCE_READ_ATTEMPTS`], and `NOT_FOUND` is `NotFound`.
//!
//! It never returns an unverified state: three broken attempts are a
//! transient error (`ServiceError::Unavailable`).
//!
//! # The caller decides how to wait
//!
//! This module knows nothing of catch-up flights, capacity budgets or
//! checkpoint leases. The caller hands in a [`FencedReadHook`], told of each
//! step: before each attempt (take capacity), on `RETRY_AFTER` (release
//! capacity, sleep, re-take — or refuse), on grant, as the fan-out reads (its
//! progress, synchronously, between windows: the pass waits until the hook
//! has applied each report before it issues its next window), before
//! release, after release. Every step runs in this read's own task; only the
//! fan-out pass runs on `fanout-read`.
//! A catch-up flight's hook takes its fenced read permit and memory
//! reservation, tops the reservation up as the read's progress shows it
//! heavier, and publishes its phase; the checkpoint's takes no capacity and
//! caps its wait. [`CappedFenceWait`] is the plain capped wait.
//!
//! # Every guard released on every exit
//!
//! The lease is a [`FenceLease`] guard owned by this future; the renewal timer
//! is a future polled inside it; the fan-out pass, spawned on `fanout-read`,
//! is held by an abort-on-drop guard inside it; whatever capacity the hook
//! holds is the hook's, owned by its caller. So dropping the read at any
//! await — the hook's own, acquire, a fan-out window, a renewal, the release
//! — drops the timer, aborts the pass at its next await, and drops the lease
//! guard, which spawns a best-effort release; the lease's deadline is the
//! backstop. A lease that acquire granted but whose reply the cancelled read
//! never saw lapses at its deadline. A release that fails leaves its guard
//! armed, and that guard's drop spawns one more release. A panic in the pass
//! is re-raised here, in the read's own task.
//!
//! # Metrics
//!
//! Per attempt: `live_edit_snapshot_read_total{path="fenced"}` and
//! `live_edit_snapshot_read_duration_seconds` (Spec SDD SR-8.1, SR-8.2; the
//! size histograms on `ok`), `live_edit_snapshot_fence_reader_wait_seconds`
//! and `live_edit_snapshot_fence_hold_seconds`; per acquire reply,
//! `live_edit_snapshot_fence_acquire_total` (SR-8.3).

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use tokio::time::Instant;

use super::{
    acquire_fence, coefficients, ms_to_duration, AcquireVerdict, FenceCoefficients, FenceLease,
    GrantKind, LayerSize, ReleaseVerdict, RenewVerdict, RetryReason, FENCE_READ_ATTEMPTS,
};
use crate::error::ServiceError;
use crate::observability::metrics::{self, SharedMetrics};
use crate::storage::client::ValkeyClient;
use crate::storage::diff_read::{
    read_object_snapshot_fenced_pass, AtomicSnapshotOf, SnapshotConsumer, SnapshotReadPath,
};
use crate::storage::fanout_thread::{PassAnswer, PassLane, PassRequest};
use crate::storage::keys::KeySchema;

/// What the fenced read tells its [`FencedReadHook`] before an attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FenceAttempt {
    /// Which attempt this is, from 1 to [`FENCE_READ_ATTEMPTS`]. Every attempt
    /// after the first follows a broken attempt
    /// ([`FenceAttemptOutcome::Broken`]).
    pub attempt: u32,
    /// The layer size acquire will be asked under: the caller's estimate
    /// first, then the best size known — the size a `RETRY_AFTER` or the grant
    /// named (the fence key's stored size when it has one), or the size the
    /// broken attempt measured. The size a hook sizes its capacity from.
    pub size: LayerSize,
}

impl FenceAttempt {
    /// Whether a broken attempt ([`FenceAttemptOutcome::Broken`]) came before
    /// this one.
    pub fn follows_broken_read(&self) -> bool {
        self.attempt > 1
    }
}

/// What the fenced read tells its [`FencedReadHook`] on a `RETRY_AFTER`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FenceRetry {
    /// How long acquire asked the reader to wait before calling again.
    pub hint: Duration,
    /// Why (Spec SDD SR-5.2).
    pub reason: RetryReason,
    /// The layer size the script computed its limits from (the fence key's
    /// stored size when it has one); what the next acquire is asked under.
    pub size: LayerSize,
}

/// What the fenced read tells its [`FencedReadHook`] on a grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FenceGrant {
    /// The header `seq` acquire read: the cut the read is proven at if its
    /// release verifies it. No commit lands on the layer from here until the
    /// release.
    pub cut: u64,
    /// Whether the grant started an epoch or joined one.
    pub kind: GrantKind,
    /// The layer size the grant computed its limits from.
    pub size: LayerSize,
}

/// How a fenced read attempt ended, as [`FenceReleased::outcome`] tells its
/// [`FencedReadHook`]. An attempt that ends [`Self::Verified`] or
/// [`Self::Ended`] is the read's last.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceAttemptOutcome {
    /// The read returns this attempt's snapshot: the release answered
    /// `VERIFIED` and the fan-out read the cut. The one attempt on which the
    /// read's once-per-read work runs, such as adjusting a memory reservation
    /// to the measured weight.
    Verified,
    /// A broken attempt, not served. The release answered `VERIFIED` or
    /// `BROKEN`, and the fan-out completed but the release answered `BROKEN`
    /// or the fan-out read a header other than the cut's, or a lost lease
    /// stopped the fan-out, or the fan-out found no header (which the release
    /// then found). A fan-out that failed otherwise, or a release that
    /// answered `NOT_FOUND` or failed, ends the read instead ([`Self::Ended`]).
    /// The read retries with a new attempt, or after [`FENCE_READ_ATTEMPTS`]
    /// broken attempts ends `ServiceError::Unavailable`.
    Broken,
    /// The read ends with an error (`NotFound` included).
    Ended,
}

/// What the fenced read tells its [`FencedReadHook`] after a release.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FenceReleased {
    /// What the release script answered; `None` when the release call itself
    /// failed. The lease guard then spawns one more best-effort release, and
    /// the read ends with an error: the fan-out's own error when the fan-out
    /// failed with anything but `NotFound`, and otherwise the release's. This
    /// is the script's raw answer, not how the attempt ended: a `VERIFIED`
    /// release can still end a broken attempt (see [`Self::outcome`]).
    pub verdict: Option<ReleaseVerdict>,
    /// How the attempt ended, decided before the hook is told. Work a hook
    /// does once per read belongs on [`FenceAttemptOutcome::Verified`] only —
    /// a catch-up flight's adjust of its memory grant and held top-up to the
    /// measured weight (`storage::memory_budget::MemoryGrant::adjust_with_top_up`),
    /// since `measured` is also set on broken attempts, whose retry still
    /// needs the memory a broken read held.
    pub outcome: FenceAttemptOutcome,
    /// The layer's size as the read measured it
    /// ([`AtomicSnapshotOf::measured_size`]), when the fan-out completed —
    /// on broken attempts too; the weight a memory reservation is adjusted
    /// to.
    pub measured: Option<LayerSize>,
}

/// How a caller of [`read_object_snapshot_fenced`] waits (Spec SDD SR-5.6's
/// acquire cycle), told of every step of every attempt, in this order:
///
/// 1. [`Self::before_attempt`] — before the attempt's first acquire;
/// 2. [`Self::on_retry_after`] — after each acquire that answered
///    `RETRY_AFTER`, before the next acquire;
/// 3. [`Self::on_grant`] — once acquire granted the lease, before the
///    fan-out's first command;
///    - [`Self::on_read_progress`] — while the fan-out reads: after its
///      first round and after each later window, with the reply bytes read
///      so far (none when a lost lease stopped the read before it began);
/// 4. [`Self::before_release`] — once the fan-out ended (completed, failed,
///    or stopped by a lost lease), before the release script is sent;
/// 5. [`Self::after_release`] — once the release replied or failed, and the
///    read decided how the attempt ended ([`FenceAttemptOutcome`]).
///
/// A broken attempt then starts the next attempt at step 1, unless it was the
/// [`FENCE_READ_ATTEMPTS`]th (the read then ends `Unavailable`). An attempt that
/// ends at acquire (the object gone, or an error) calls neither of steps 3–5.
/// Steps 3–5 run on every attempt that got a lease, and run to completion:
/// the renewal timer keeps renewing the lease while they run, and a lease lost
/// meanwhile is not a reason to cut them short.
///
/// The two fallible events end the read with their error, which the read
/// returns as it is: that is how a hook refuses (a capacity wait that ran out,
/// a wait cap). Whatever capacity the hook holds when the read returns is
/// still the hook's, for its owner to keep or drop.
///
/// Implementations are owned by the future that drives the read, so a
/// cancelled read drops nothing the hook holds; its owner does. `Send`, and
/// every event future `Send`, so the read can run in a spawned or shared
/// future.
pub trait FencedReadHook: Send {
    /// Before an attempt's first acquire: take whatever capacity the attempt
    /// needs and is not already held. After a broken attempt
    /// ([`FenceAttempt::follows_broken_read`]) this also reopens whatever the
    /// caller closed in [`Self::before_release`]. An error ends the read.
    fn before_attempt(
        &mut self,
        attempt: FenceAttempt,
    ) -> impl Future<Output = Result<(), ServiceError>> + Send;

    /// Acquire answered `RETRY_AFTER`: release capacity held idle, wait
    /// `retry.hint`, and re-take it before returning; the next acquire follows
    /// at once. An error ends the read.
    fn on_retry_after(
        &mut self,
        retry: FenceRetry,
    ) -> impl Future<Output = Result<(), ServiceError>> + Send;

    /// Acquire granted the lease; the cut is fixed.
    fn on_grant(&mut self, grant: FenceGrant) -> impl Future<Output = ()> + Send;

    /// The fan-out has read `bytes_read` reply bytes so far, cumulative over
    /// this attempt: called after its first round (the index members) and
    /// after each later window, the last call carrying the attempt's
    /// `reply_bytes` ([`AtomicSnapshotOf::measured_size`]'s bytes) when the
    /// fan-out completes. Each attempt counts from zero again.
    ///
    /// Synchronous, and must not wait: it runs between the fan-out's windows,
    /// beside the lease's renewal timer. A hook that reserved memory for the
    /// read tops it up here, never waiting, to what the bytes read so far
    /// hold resident. Does nothing by default.
    fn on_read_progress(&mut self, bytes_read: u64) {
        let _ = bytes_read;
    }

    /// The fan-out ended and the release script is about to be sent: the last
    /// moment at which the layer's `seq` is known to equal the cut.
    fn before_release(&mut self) -> impl Future<Output = ()> + Send;

    /// The release replied (or failed); `released.outcome` says how the
    /// attempt ended. Once-per-read work — a catch-up flight's memory-grant
    /// adjust to the measured weight — runs only when it is
    /// [`FenceAttemptOutcome::Verified`]: every other attempt that completed
    /// its fan-out reports a `measured` size too, and a broken one is
    /// retried, or, when it was the
    /// [`FENCE_READ_ATTEMPTS`]th, ends the read `ServiceError::Unavailable`.
    fn after_release(&mut self, released: FenceReleased) -> impl Future<Output = ()> + Send;
}

/// Read `object_id`'s state above the atomic-read ceiling, proven at a cut
/// (Spec SDD SR-5.3); see the module docs.
///
/// `estimate` is the caller's size for the layer (Spec SDD SR-11.1,
/// [`LayerSize::from_too_large`]); the fence scripts use the fence key's
/// stored size whenever it has one. `consumer` labels the metrics and is one
/// of the consumers with an above-ceiling path
/// ([`SnapshotConsumer::takes_atomic_path`]); its type decides the state's
/// form — a `SnapshotReadConsumer`'s read returns the full
/// `ObjectSnapshot`, a `SlimSnapshotReadConsumer`'s the compact
/// `CompactState`, from the same fan-out. `hook` decides how to wait
/// ([`FencedReadHook`]).
///
/// Returns:
///
/// - the state, with `path = Fenced`, `cut` = the `seq` acquire read, and the
///   read's `reply_bytes`, `item_count` and `entry_count`
///   ([`AtomicSnapshotOf::measured_size`]) — only after a `VERIFIED` release;
/// - `ServiceError::NotFound` when the header is absent or has no numeric
///   `seq` at acquire or at release — a release that finds no header ends
///   the read `NotFound` whatever the fan-out answered, its failure included
///   (Spec SDD SR-5.3). A fan-out that found no header but a release that
///   did is a restore under the read: a broken attempt, retried;
/// - `ServiceError::Unavailable` after [`FENCE_READ_ATTEMPTS`] broken attempts
///   (the lease lost to a stall, or a restore or cleanup during the read);
/// - the hook's error, when it refused;
/// - a classified error otherwise (transient → `Aborted`, permanent →
///   `Internal`): a fence script's failure as `storage::snapshot_fence`
///   classifies it, a fan-out command's as the unfenced fan-out does.
pub async fn read_object_snapshot_fenced<C: SnapshotConsumer, H: FencedReadHook>(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    consumer: C,
    estimate: LayerSize,
    hook: &mut H,
) -> Result<AtomicSnapshotOf<C::State>, ServiceError> {
    debug_assert!(
        consumer.takes_atomic_path(),
        "{consumer:?} has no above-ceiling fenced path (Spec SDD SR-1.9)"
    );
    let metrics = client.metrics_snapshot();
    let read = FencedRead {
        client,
        keys,
        object_id,
        object_id_str: object_id.to_string(),
        consumer,
        metrics,
    };
    let mut size = estimate;
    for attempt in 1..=FENCE_READ_ATTEMPTS {
        hook.before_attempt(FenceAttempt { attempt, size }).await?;
        let lease = read.wait_for_lease(&mut size, hook).await?;
        match read.hold(lease, hook).await {
            AttemptEnd::Verified(snapshot) => return Ok(*snapshot),
            AttemptEnd::Broken { measured } => {
                if let Some(measured) = measured {
                    size = measured;
                }
                tracing::debug!(
                    object_id,
                    consumer = consumer.label(),
                    attempt,
                    max_attempts = FENCE_READ_ATTEMPTS,
                    "fenced snapshot read broken (its lease was lost, or the object was \
                     restored or removed during the read); not serving it"
                );
            }
            AttemptEnd::Ended(error) => return Err(error),
        }
    }
    Err(ServiceError::Unavailable(format!(
        "fenced snapshot read of object {object_id} was broken {FENCE_READ_ATTEMPTS} times in a \
         row (its lease lost, or the object restored or removed, during each read); an \
         unverified snapshot is never served"
    )))
}

/// How one attempt's hold ended.
enum AttemptEnd<S> {
    /// The release verified the read. Boxed so the other variants do not carry
    /// the snapshot's size.
    Verified(Box<AtomicSnapshotOf<S>>),
    /// The release answered `BROKEN`, a lost lease stopped the read, the
    /// read's header was not at the cut, or the read found no header that the
    /// release then found: retry. `measured` when the fan-out completed.
    Broken { measured: Option<LayerSize> },
    /// The read ends with this error (`NotFound` included).
    Ended(ServiceError),
}

impl<S> AttemptEnd<S> {
    /// What the hook is told of this end.
    fn outcome(&self) -> FenceAttemptOutcome {
        match self {
            Self::Verified(_) => FenceAttemptOutcome::Verified,
            Self::Broken { .. } => FenceAttemptOutcome::Broken,
            Self::Ended(_) => FenceAttemptOutcome::Ended,
        }
    }
}

/// One fenced read's fixed inputs.
struct FencedRead<'a, C> {
    client: &'a ValkeyClient,
    keys: &'a KeySchema,
    object_id: u64,
    object_id_str: String,
    consumer: C,
    metrics: Option<SharedMetrics>,
}

impl<C: SnapshotConsumer> FencedRead<'_, C> {
    /// Acquire until granted (one attempt's wait), telling `hook` of each
    /// `RETRY_AFTER`. `size` follows what the replies name. An attempt that
    /// ends here — the object gone, a failed acquire — is counted as a read
    /// (with that acquire's round trip as its duration); one the hook ends is
    /// not, since no read started.
    async fn wait_for_lease<H: FencedReadHook>(
        &self,
        size: &mut LayerSize,
        hook: &mut H,
    ) -> Result<Box<FenceLease>, ServiceError> {
        let mut wait = Span::start(
            self.metrics.clone(),
            self.consumer.label(),
            SpanKind::ReaderWait,
        );
        loop {
            let called = Instant::now();
            match acquire_fence(self.client, self.keys, self.object_id, *size).await {
                Err(error) => {
                    self.record_read(read_outcome(&error), called.elapsed());
                    return Err(error);
                }
                Ok(AcquireVerdict::NotFound) => {
                    self.record_read(metrics::SNAPSHOT_READ_OUTCOME_NOT_FOUND, called.elapsed());
                    return Err(self.not_found());
                }
                Ok(AcquireVerdict::RetryAfter {
                    hint,
                    reason,
                    size: named,
                }) => {
                    self.count_acquire(retry_label(reason));
                    *size = named;
                    hook.on_retry_after(FenceRetry {
                        hint,
                        reason,
                        size: named,
                    })
                    .await?;
                }
                Ok(AcquireVerdict::Granted(lease)) => {
                    wait.finish();
                    self.count_acquire(grant_label(lease.kind()));
                    *size = lease.size();
                    return Ok(lease);
                }
            }
        }
    }

    /// Hold `lease` for one attempt: tell `hook` of the grant, run the fan-out
    /// beside the renewal timer, routing its progress reports to `hook`, tell
    /// `hook` the release is next, release, decide how the attempt ended, and
    /// tell `hook` both.
    async fn hold<H: FencedReadHook>(
        &self,
        lease: Box<FenceLease>,
        hook: &mut H,
    ) -> AttemptEnd<C::State> {
        let mut hold = Span::start(self.metrics.clone(), self.consumer.label(), SpanKind::Hold);
        let cut = lease.cut();
        let grant = FenceGrant {
            cut,
            kind: lease.kind(),
            size: lease.size(),
        };
        let mut lost = false;
        let mut read_elapsed = Duration::ZERO;
        let read = {
            // The timer borrows the lease, so it is dropped at the end of this
            // block, before the release consumes the lease.
            let renewal = renew_until_lost(self.client, &lease);
            tokio::pin!(renewal);
            beside_renewal(hook.on_grant(grant), renewal.as_mut(), &mut lost, false).await;
            let read = if lost {
                None
            } else {
                let started = Instant::now();
                // The pass runs on the `fanout-read` thread
                // (`storage::fanout_thread`, `PassLane::Fenced`), so its
                // windows allocate from that thread's one allocator arena
                // whatever the worker count, and it never queues behind an
                // unfenced WATCH or ext-GC read, which run on the
                // `watch-read` threads; it owns clones of what it reads. Its progress
                // reports come back here, to the hook, which is borrowed by
                // nothing else meanwhile: each is applied before the pass
                // issues its next window. Dropping this future (a lost lease
                // below, or the read's own cancellation) aborts the pass.
                let (client, keys, object_id) = (
                    self.client.clone(),
                    self.keys.clone(),
                    self.object_id_str.clone(),
                );
                // Boxed: this future is held by catch-up flights.
                let pass = Box::pin(self.client.fanout_threads().run_pass(
                    PassLane::Fenced,
                    move |link| {
                        read_object_snapshot_fenced_pass::<C::State>(client, keys, object_id, link)
                    },
                    |request| {
                        if let PassRequest::Progress { bytes_read } = request {
                            hook.on_read_progress(bytes_read);
                        }
                        PassAnswer::GoOn
                    },
                ));
                let read = beside_renewal(pass, renewal.as_mut(), &mut lost, true).await;
                read_elapsed = started.elapsed();
                read
            };
            beside_renewal(hook.before_release(), renewal.as_mut(), &mut lost, false).await;
            read
        };
        let measured = match &read {
            Some(Ok(snapshot)) => Some(snapshot.measured_size()),
            _ => None,
        };
        let release = lease.release(measured).await;
        hold.finish();
        let verdict = release.as_ref().ok().copied();

        let end = match (read, release) {
            // The pass found no live header, but the release — which answers
            // `BROKEN` or `VERIFIED` only on a present header — found one: a
            // non-commit mutator (a restore materializing the object again)
            // ran under the read. The object exists, so this is a broken
            // attempt to retry, not `NotFound` (Spec SDD SR-5.3).
            (
                Some(Err(ServiceError::NotFound(_))),
                Ok(ReleaseVerdict::Broken | ReleaseVerdict::Verified),
            ) => AttemptEnd::Broken { measured: None },
            // The pass found no live header and the release could not say
            // whether one is back: end with the release's (classified,
            // usually transient) error rather than claim the object is gone.
            (Some(Err(ServiceError::NotFound(_))), Err(error)) => AttemptEnd::Ended(error),
            // The release found no header: the object is gone, whatever the
            // fan-out failed with (Spec SDD SR-5.3).
            (Some(Err(_)), Ok(ReleaseVerdict::NotFound)) => AttemptEnd::Ended(self.not_found()),
            // Otherwise the fan-out's own failure (a failed command) is the
            // first cause, whatever the release then answered.
            (Some(Err(error)), _) => AttemptEnd::Ended(error),
            (_, Err(error)) => AttemptEnd::Ended(error),
            (_, Ok(ReleaseVerdict::NotFound)) => AttemptEnd::Ended(self.not_found()),
            // A pass whose header was not at the cut read something other
            // than the cut's state, whatever the release says.
            (Some(Ok(snapshot)), Ok(ReleaseVerdict::Verified)) if snapshot.cut == cut => {
                AttemptEnd::Verified(Box::new(snapshot))
            }
            (Some(Ok(_)), Ok(_)) => AttemptEnd::Broken { measured },
            // A lost lease stopped the fan-out: nothing to prove.
            (None, Ok(_)) => AttemptEnd::Broken { measured: None },
        };
        hook.after_release(FenceReleased {
            verdict,
            outcome: end.outcome(),
            measured,
        })
        .await;
        let outcome = match &end {
            AttemptEnd::Verified(_) => metrics::SNAPSHOT_READ_OUTCOME_OK,
            AttemptEnd::Broken { .. } => metrics::SNAPSHOT_READ_OUTCOME_FENCE_BROKEN,
            AttemptEnd::Ended(error) => read_outcome(error),
        };
        self.record_read(outcome, read_elapsed);
        if let (AttemptEnd::Verified(snapshot), Some(m)) = (&end, &self.metrics) {
            debug_assert_eq!(snapshot.path, SnapshotReadPath::Fenced);
            m.observe_snapshot_read_size(
                self.consumer.label(),
                snapshot.reply_bytes,
                snapshot.entry_count,
            );
        }
        end
    }

    fn not_found(&self) -> ServiceError {
        ServiceError::NotFound(format!("object {} not found", self.object_id))
    }

    fn record_read(&self, outcome: &'static str, duration: Duration) {
        if let Some(m) = &self.metrics {
            m.observe_snapshot_read(
                self.consumer.label(),
                SnapshotReadPath::Fenced.label(),
                outcome,
                duration,
            );
        }
    }

    fn count_acquire(&self, result: &'static str) {
        if let Some(m) = &self.metrics {
            m.inc_snapshot_fence_acquire(result);
        }
    }
}

/// Poll `fut` to completion beside the renewal timer. A renewal that answers
/// `LOST` sets `lost` (the timer is not polled again) and, when
/// `stop_on_lost`, drops `fut` unfinished and returns `None`; otherwise `fut`
/// runs on to its end.
async fn beside_renewal<F: Future>(
    fut: F,
    mut renewal: Pin<&mut impl Future<Output = ()>>,
    lost: &mut bool,
    stop_on_lost: bool,
) -> Option<F::Output> {
    tokio::pin!(fut);
    loop {
        tokio::select! {
            biased;
            out = &mut fut => return Some(out),
            () = renewal.as_mut(), if !*lost => {
                *lost = true;
                if stop_on_lost {
                    return None;
                }
            }
        }
    }
}

/// The renewal timer (Spec SDD SR-5.3): renew `lease` every `lease(L) / 4`,
/// on the data pool, until a renewal answers `LOST`. A failed renewal is
/// logged and retried at the next period — the lease may still be live, and
/// if it lapses the release says so. Polled only inside the read's future,
/// so dropping the read stops it.
async fn renew_until_lost(client: &ValkeyClient, lease: &FenceLease) {
    let period = lease.renew_interval();
    loop {
        tokio::time::sleep(period).await;
        if client.object_snapshot_read_faults.fence_renewals_stalled() {
            continue;
        }
        match lease.renew().await {
            Ok(RenewVerdict::Renewed) => {}
            Ok(RenewVerdict::Lost) => return,
            Err(error) => tracing::warn!(
                lease = ?lease,
                error = %error,
                "snapshot-fence lease renewal failed; retrying at the next period (if the lease \
                 lapses meanwhile, the read's release answers BROKEN and the read is retried)"
            ),
        }
    }
}

/// The `live_edit_snapshot_read_total` outcome of an error that ended a
/// fenced read attempt.
fn read_outcome(error: &ServiceError) -> &'static str {
    match error {
        ServiceError::NotFound(_) => metrics::SNAPSHOT_READ_OUTCOME_NOT_FOUND,
        ServiceError::Aborted(_) | ServiceError::Unavailable(_) => {
            metrics::SNAPSHOT_READ_OUTCOME_TRANSIENT_ERROR
        }
        _ => metrics::SNAPSHOT_READ_OUTCOME_PERMANENT_ERROR,
    }
}

/// The `live_edit_snapshot_fence_acquire_total` result of a grant.
fn grant_label(kind: GrantKind) -> &'static str {
    match kind {
        GrantKind::StartedEpoch => metrics::SNAPSHOT_FENCE_ACQUIRE_STARTED_EPOCH,
        GrantKind::JoinedEpoch => metrics::SNAPSHOT_FENCE_ACQUIRE_JOINED_EPOCH,
    }
}

/// The `live_edit_snapshot_fence_acquire_total` result of a `RETRY_AFTER`.
fn retry_label(reason: RetryReason) -> &'static str {
    match reason {
        RetryReason::Epoch => metrics::SNAPSHOT_FENCE_ACQUIRE_RETRY_AFTER_EPOCH,
        RetryReason::Gap => metrics::SNAPSHOT_FENCE_ACQUIRE_RETRY_AFTER_GAP,
        RetryReason::Intent => metrics::SNAPSHOT_FENCE_ACQUIRE_RETRY_AFTER_INTENT,
    }
}

#[derive(Clone, Copy)]
enum SpanKind {
    /// `live_edit_snapshot_fence_reader_wait_seconds`.
    ReaderWait,
    /// `live_edit_snapshot_fence_hold_seconds`.
    Hold,
}

/// One timed span of a fenced read attempt, recorded once: at
/// [`Self::finish`], or when dropped unfinished (an error return, or the read
/// cancelled), up to that moment.
struct Span {
    started: Instant,
    metrics: Option<SharedMetrics>,
    /// The consumer's metric label.
    consumer: &'static str,
    kind: SpanKind,
    recorded: bool,
}

impl Span {
    fn start(metrics: Option<SharedMetrics>, consumer: &'static str, kind: SpanKind) -> Self {
        Self {
            started: Instant::now(),
            metrics,
            consumer,
            kind,
            recorded: false,
        }
    }

    fn finish(&mut self) {
        if self.recorded {
            return;
        }
        self.recorded = true;
        let Some(m) = &self.metrics else {
            return;
        };
        let elapsed = self.started.elapsed();
        match self.kind {
            SpanKind::ReaderWait => m.observe_snapshot_fence_reader_wait(self.consumer, elapsed),
            SpanKind::Hold => m.observe_snapshot_fence_hold(self.consumer, elapsed),
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        self.finish();
    }
}

/// A [`FencedReadHook`] that takes no capacity and caps the read's wait at
/// `reader_wait_max(L)` (Spec SDD SR-5.5, SR-11.4, SR-11.5): the wait a
/// checkpoint's fenced read is allowed, and a plain hook for any caller that
/// holds no capacity.
///
/// Its clock starts at the first attempt and counts everything after it — the
/// `RETRY_AFTER` sleeps and the reads themselves, broken ones included — as
/// SR-11.4's cap does. `L` is the latest size the read was told of, so the cap
/// follows the fence key's stored size once a reply names it. On a
/// `RETRY_AFTER` it sleeps the hint, unless the sleep would end past the cap,
/// and before a retry after a broken attempt it checks the cap again; past it,
/// it refuses with a WARN and `ServiceError::Unavailable` (retryable), and
/// the read ends there.
#[derive(Debug)]
pub struct CappedFenceWait {
    coefficients: FenceCoefficients,
    object_id: u64,
    /// The consumer's metric label, for the refusal's WARN.
    consumer: &'static str,
    started: Option<Instant>,
}

impl CappedFenceWait {
    /// A capped wait for one fenced read of `object_id` by `consumer`, with
    /// the cap computed from `client`'s fence coefficients.
    pub fn new(client: &ValkeyClient, object_id: u64, consumer: impl SnapshotConsumer) -> Self {
        Self {
            coefficients: coefficients(client),
            object_id,
            consumer: consumer.label(),
            started: None,
        }
    }

    /// `reader_wait_max(L)` for a layer of `size`.
    pub fn cap(&self, size: LayerSize) -> Duration {
        ms_to_duration(self.coefficients.timing(size).reader_wait_max_ms)
    }

    /// When the clock started (at the first attempt).
    fn started(&mut self) -> Instant {
        *self.started.get_or_insert_with(Instant::now)
    }

    fn refuse(&self, size: LayerSize, waited: Duration, would_wait: Duration) -> ServiceError {
        let cap = self.cap(size);
        tracing::warn!(
            object_id = self.object_id,
            consumer = self.consumer,
            waited_ms = waited.as_millis() as u64,
            would_wait_ms = would_wait.as_millis() as u64,
            reader_wait_max_ms = cap.as_millis() as u64,
            size_items = size.items,
            size_bytes = size.bytes,
            "fenced snapshot read gave up waiting for the snapshot fence: its wait would pass \
             reader_wait_max(L)"
        );
        ServiceError::Unavailable(format!(
            "fenced snapshot read of object {} would wait past reader_wait_max(L) = {} ms for the \
             snapshot fence (waited {} ms)",
            self.object_id,
            cap.as_millis(),
            waited.as_millis()
        ))
    }
}

impl FencedReadHook for CappedFenceWait {
    async fn before_attempt(&mut self, attempt: FenceAttempt) -> Result<(), ServiceError> {
        let waited = self.started().elapsed();
        if attempt.follows_broken_read() && waited > self.cap(attempt.size) {
            return Err(self.refuse(attempt.size, waited, waited));
        }
        Ok(())
    }

    async fn on_retry_after(&mut self, retry: FenceRetry) -> Result<(), ServiceError> {
        let waited = self.started().elapsed();
        let would_wait = waited.saturating_add(retry.hint);
        if would_wait > self.cap(retry.size) {
            return Err(self.refuse(retry.size, waited, would_wait));
        }
        tokio::time::sleep(retry.hint).await;
        Ok(())
    }

    async fn on_grant(&mut self, _grant: FenceGrant) {}

    async fn before_release(&mut self) {}

    async fn after_release(&mut self, _released: FenceReleased) {}
}
