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
//! (SDD §5.6), and the paced ending of the live streams.
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
//!   [`GracefulShutdown::enroll_stream`]: a fresh child of the parent, entered
//!   in the registry until the returned [`StreamEnrollment`] drops. Closing the
//!   registry, cancelling `begun` and taking the snapshot happen under one
//!   lock, so every stream is either in the snapshot or born cancelled.
//!
//! [`end_streams_paced`] ends the snapshot of N streams in random order over a
//! window of at most [`StreamPacing::max_window`] (W_cap): at
//! [`StreamPacing::rate_per_sec`] (R) per second, or at N / W_cap per second
//! once the cap binds (N > R × W_cap) — max(R, N / W_cap) — so the streams'
//! reconnects reach the other replicas as a ramp rather than a step, then
//! cancels the parent.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use rand::seq::SliceRandom;
use tokio_util::sync::CancellationToken;

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

/// The registry of live streams, under [`Shared::registry`]'s lock.
#[derive(Default)]
struct Registry {
    /// Set when the paced ending starts, or by [`GracefulShutdown::cancel`];
    /// never cleared. A closed registry enrolls nothing.
    closed: bool,
    next_id: u64,
    entries: HashMap<u64, CancellationToken>,
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
/// Who reads what:
///
/// * `SubscribeReadLayer` and `JoinSession` take their token only from
///   [`Self::enroll_stream`] — the subscription before its admission, the
///   join at its task-start check — and read that per-stream child everywhere
///   after: admission, catch-up waits, sends, the idle wait, the membership
///   replay, the bidirectional loop and the lease keeper. Each ends the RPC
///   `UNAVAILABLE "server shutting down"`. A per-stream child keeps the hot
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

    /// Enroll one live stream: its shutdown token, a fresh child of the
    /// parent, and the guard that keeps it in the registry until dropped.
    ///
    /// Once the paced ending has started (or [`Self::cancel`] has run) the
    /// registry is closed: the token comes back already cancelled and nothing
    /// is enrolled, so the stream is refused at its first check.
    pub fn enroll_stream(&self) -> (CancellationToken, StreamEnrollment) {
        let mut registry = self.shared.registry();
        if registry.closed {
            drop(registry);
            let token = CancellationToken::new();
            token.cancel();
            return (
                token,
                StreamEnrollment {
                    shared: Arc::clone(&self.shared),
                    id: None,
                },
            );
        }
        let token = self.shared.parent.child_token();
        let id = registry.next_id;
        registry.next_id = registry.next_id.wrapping_add(1);
        registry.entries.insert(id, token.clone());
        drop(registry);
        (
            token,
            StreamEnrollment {
                shared: Arc::clone(&self.shared),
                id: Some(id),
            },
        )
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
    pub fn begun(&self) -> &CancellationToken {
        &self.shared.begun
    }

    /// Close the registry and cancel the parent: every stream still enrolled
    /// ends at once, `begun` is cancelled, and later enrollments are born
    /// cancelled. Idempotent.
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

    /// The number of streams enrolled now.
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
    /// either in the returned snapshot or born cancelled.
    fn begin(&self) -> Vec<CancellationToken> {
        let mut registry = self.shared.registry();
        registry.closed = true;
        self.shared.begun.cancel();
        registry.entries.drain().map(|(_, token)| token).collect()
    }
}

/// Keeps one stream in the [`GracefulShutdown`] registry; dropping it removes
/// the stream. Held by the stream's task for the task's whole life, so a
/// stream that ended on its own before the paced ending starts is not in its
/// snapshot.
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
/// `UNAVAILABLE "server shutting down"`, and a stream not yet reached that
/// observes its session's deletion first ends as a deletion.
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
    //! each instant is asserted exactly. The schedule as `main.rs` runs it —
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
        let watchers: Vec<_> = tokens.iter().map(cancelled_at).collect();
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
}
