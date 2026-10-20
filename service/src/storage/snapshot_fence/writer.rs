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

//! The writer side of the snapshot fence (Spec SDD SR-5.4): the state one
//! `update_layer` call carries across its attempts, a WATCH-path attempt's
//! intent renewal, and the clock that caps a call's fence waits.
//!
//! `LayerStore::update_layer_with_operation` drives all three, and its rustdoc
//! ("Writers honour the snapshot fence") describes the protocol end to end.
//! In short:
//!
//! - A commit that finds a live fence — the update script's `{"FENCED", ms}`
//!   reply in either mode (the fast path, the structural route), the WATCH
//!   path's probe beside its `WATCH`ed header read, an intent registration
//!   refused `FENCED` at step 2b (after the header guards), or an `EXEC`
//!   conflict whose re-probe finds a live fence or a fence key changed since
//!   the attempt's own probe (an epoch that began, or began and ended, after
//!   the `WATCH`) — waits and re-runs the attempt without spending a retry
//!   attempt. (A registration refused before
//!   `WATCH` is not a wait: the attempt falls through to its guards.) Each
//!   sleep is [`FenceWaitClock::next_sleep`]'s; the call's total fence wait is
//!   capped there.
//! - A WATCH-path attempt that reads a snapshot, on a layer known to be above
//!   the atomic-read ceiling ([`WriterFenceClass`]), registers a writer intent,
//!   renewed by [`IntentRenewal`] while the attempt runs and cleared on every
//!   exit. The update script's paths register none (structural-edits SR-8.2).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::Instant;

use super::{
    ms_to_duration, FenceCoefficients, FenceProbe, IntentRenewVerdict, LayerSize, WriterIntent,
};

/// Reads how much of the calling RPC's deadline is left, or `None` when no
/// deadline is in scope (a background task, a direct test call).
///
/// Injected at the composition root (`main.rs` hands in
/// `grpc::deadline::remaining`) because `storage` must not depend on `grpc`
/// (CLAUDE.md module dependency rules); the same shape as the object-id
/// counter repair's `RemainingDeadlineFn`. `Arc` rather than `Box` because
/// `LayerStore` is `Clone`.
pub type RemainingDeadlineFn = Arc<dyn Fn() -> Option<Duration> + Send + Sync>;

/// What one `update_layer` call has learned about its layer's size, carried
/// from attempt to attempt (Spec SDD SR-5.4 classification).
///
/// A WATCH-path attempt that reads a snapshot registers a writer intent only
/// when its layer is
/// known to be above the atomic-read ceiling: a previous attempt of the same
/// call (or this attempt's own probe) found the fence key — which exists only
/// for layers fenced within the last `key_linger(L)` — or a previous attempt's
/// unfenced fan-out read tallied more items or bytes than the atomic ceilings.
/// A layer never classified this way registers no intent and costs no extra
/// round trip.
///
/// Interior mutability so the attempts, which borrow it shared, can record
/// into it. Monotone: once above the ceiling, the call stays classified, and
/// the recorded size only grows.
#[derive(Debug, Default)]
pub struct WriterFenceClass {
    above_ceiling: Mutex<Option<LayerSize>>,
}

impl WriterFenceClass {
    /// The size class an intent should be registered under, or `None` while
    /// the layer is not known to be above the ceiling.
    pub fn above_ceiling(&self) -> Option<LayerSize> {
        *self.above_ceiling.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Record that the layer is above the ceiling at (at least) `size`.
    pub fn note_above_ceiling(&self, size: LayerSize) {
        let mut slot = self.above_ceiling.lock().unwrap_or_else(|p| p.into_inner());
        *slot = Some(match *slot {
            Some(known) => LayerSize {
                items: known.items.max(size.items),
                bytes: known.bytes.max(size.bytes),
            },
            None => size,
        });
    }

    /// Record what a fence probe saw. A present fence key classifies the
    /// layer: its stored size when it has one, else `ceiling` — the layer was
    /// fenced, so it is at least ceiling-sized.
    pub fn note_probe(&self, probe: &FenceProbe, ceiling: LayerSize) {
        if let FenceProbe::Present { stored_size, .. } = probe {
            self.note_above_ceiling(stored_size.unwrap_or(ceiling));
        }
    }

    /// Record an unfenced fan-out read's tallies: above either atomic
    /// ceiling classifies the layer at the tallied size.
    pub fn note_tallies(&self, tallied: LayerSize, ceiling: LayerSize) {
        if tallied.items > ceiling.items || tallied.bytes > ceiling.bytes {
            self.note_above_ceiling(tallied);
        }
    }
}

/// The clock one `update_layer` call's fence waits run on (Spec SDD SR-5.4).
///
/// Each wait sleeps `min(retry_after, poll)` plus a jitter of up to `rtt99` —
/// both from the cost model ([`FenceCoefficients::poll_ms`], `rtt99_ms`), so
/// no step is a hand-picked constant (SR-5.7). `poll` is the readers' own
/// polling step, so a writer behind a reader's lease re-checks at the pace a
/// reader would. The jitter spreads the writers that the same lease end
/// releases across one round trip, instead of firing them in the same
/// millisecond.
///
/// **What counts as fence wait.** The span from the start of the first
/// attempt a fence refused or failed — the `since` the call's first
/// [`Self::next_sleep`] is handed — to the end of its last fence sleep
/// ([`Self::waited`]): that attempt, the sleeps, and the attempts between them
/// that a fence refused or failed again. So a WATCH-path attempt an epoch
/// interrupted counts whole, not only the sleep after its failed `EXEC`. And
/// because the span starts where that first attempt started, it includes
/// every step the attempt took before it met the fence, which under writer
/// contention is time the fence did not add. For example: on the fast path
/// nothing, the update script's step 1d being both the probe and the
/// attempt's only round trip; for a WATCH-path step-2b refusal its steps
/// before its probe — its txn-pool checkout wait, the `WATCH` round trip and,
/// on a layer known to be above the ceiling, the before-`WATCH` intent
/// registration; for the cached structural route's `FENCED` reply (its
/// `structural`, `structural_full` and `reset` commits) its read — the header
/// repair at least and, on a cold start or a reload, a whole-layer read under
/// the cache's cold-start permits — and its plan; and for a structural
/// attempt that falls back to the WATCH path within the same attempt (its
/// read answered uncacheable, its reset was over the size gate before its
/// cold start, its plan was over the size gate — after its read, a cold start
/// included, and its whole plan — or its commit was refused `CACHE_DIVERGED`
/// a second time; `storage::structural_size_gate`) and is then refused at
/// step 2b, the structural steps it took first as well as the WATCH path's
/// pre-probe steps. It stops at the end of
/// the last sleep, so the attempt that finally commits — for an above-ceiling
/// structural writer on the WATCH path, `S(e, b)`, on the AMR model about
/// 134 ms at the shipped item ceiling, 0.57 s at four times it and about 4 s
/// at 200k entries — and whatever the call does after it are not fence wait:
/// the span is the latency the fence added, plus whatever the first refused
/// or failed attempt did before it met the fence. That span is what
/// `live_edit_update_layer_fence_wait_seconds` records (SR-9.1's writer
/// pause, reported beside the fence hold), once per call that met a fence,
/// on every exit: a call whose future is dropped mid-call (the RPC deadline,
/// a disconnect, shutdown) records the span up to the drop
/// ([`Self::waited_by`]).
///
/// **The cap.** The same span is what `cap` (`LIVE_EDIT_RPC_DEADLINE_MS`)
/// bounds: [`Self::next_sleep`] takes a sleep only if the span it would
/// extend to — the first refused or failed attempt's start to the end of that
/// sleep — stays under `cap`, and only if the sleep ends before what is left
/// of the calling RPC's deadline, re-read before every sleep. So a commit's
/// recorded fence wait never exceeds the cap, and the cap is not spent by the
/// commit's own attempt. A sleep
/// that would reach either bound is not taken: `next_sleep` answers `None`,
/// and the caller fails the commit `UNAVAILABLE` while the RPC still has time
/// to say so, rather than sleeping into the tower deadline's bare
/// `DEADLINE_EXCEEDED`. Two `update_layer` calls in one handler share the
/// RPC's deadline through that re-read.
pub struct FenceWaitClock {
    cap: Duration,
    remaining: Option<RemainingDeadlineFn>,
    poll: Duration,
    jitter: Duration,
    /// When the first attempt a fence refused or failed began: the `since`
    /// of the call's first [`Self::next_sleep`].
    started: Option<Instant>,
    /// When the call's last fence sleep ends (or ended).
    wait_end: Option<Instant>,
}

impl FenceWaitClock {
    /// A clock for one call, with nothing waited yet.
    pub fn new(
        cap: Duration,
        remaining: Option<RemainingDeadlineFn>,
        coefficients: &FenceCoefficients,
    ) -> Self {
        FenceWaitClock {
            cap,
            remaining,
            poll: ms_to_duration(coefficients.poll_ms()),
            jitter: ms_to_duration(coefficients.rtt99_ms),
            started: None,
            wait_end: None,
        }
    }

    /// The next sleep for a fence held for another `retry_after`, or `None`
    /// when taking it would pass the cap (see the type's docs).
    ///
    /// `since` is when the attempt the fence just refused or failed began.
    /// The first call fixes the span's start there; later calls' `since` is
    /// ignored, because the span already covers their attempts.
    pub fn next_sleep(&mut self, retry_after: Duration, since: Instant) -> Option<Duration> {
        let now = Instant::now();
        let started = *self.started.get_or_insert(since.min(now));
        let jitter_ns = self.jitter.as_nanos().min(u128::from(u64::MAX)) as u64;
        let jitter = {
            use rand::Rng;
            Duration::from_nanos(rand::rng().random_range(0..=jitter_ns))
        };
        let sleep = retry_after.min(self.poll) + jitter;
        let end = now + sleep;
        let within_cap = end.saturating_duration_since(started) < self.cap;
        let within_deadline = self
            .remaining
            .as_ref()
            .and_then(|f| f())
            .is_none_or(|remaining| sleep < remaining);
        if !(within_cap && within_deadline) {
            return None;
        }
        self.wait_end = Some(end);
        Some(sleep)
    }

    /// How long this call waited on fences: from the start of the first
    /// attempt a fence refused or failed to the end of its last fence sleep
    /// (zero when it met a fence and gave up without sleeping), or `None` when
    /// it never met a fence.
    pub fn waited(&self) -> Option<Duration> {
        let started = self.started?;
        Some(
            self.wait_end
                .map_or(Duration::ZERO, |end| end.saturating_duration_since(started)),
        )
    }

    /// [`Self::waited`] as of `now`, for a call that ends at `now` without
    /// returning (its future dropped): a sleep still in progress counts only
    /// up to `now`, not to its planned end. Equal to `waited()` once the last
    /// sleep has ended.
    pub fn waited_by(&self, now: Instant) -> Option<Duration> {
        let started = self.started?;
        Some(self.wait_end.map_or(Duration::ZERO, |end| {
            end.min(now).saturating_duration_since(started)
        }))
    }
}

/// A registered writer intent and the task that renews it (Spec SDD SR-5.4).
///
/// [`Self::start`] hands the intent to a spawned task that renews it every
/// `intent_ttl(L) / 4` until told to stop, then clears it (`HDEL`). A
/// `LOST` renewal stops renewing (the intent lapsed; nothing revives it) and
/// a failed one is logged and retried at the next period — the intent lapses
/// on its own deadline if renewals keep failing.
///
/// Cleared on every exit: [`Self::clear`] stops the task and waits for its
/// `HDEL`; dropping the handle unstopped (a cancelled or panicking attempt)
/// closes the stop channel, which the task treats the same way, so the `HDEL`
/// still runs, after the attempt is gone.
///
/// The task is tracked on the intent's client from its start
/// (`ValkeyClient::track_fence_guard_task`), like a dropped guard's cleanup,
/// so `ValkeyClient::quit` waits for its `HDEL` before it closes the pools:
/// an attempt that graceful shutdown drops mid-commit does not leave its
/// intent holding other replicas' fenced readers off until it lapses.
pub struct IntentRenewal {
    stop: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for IntentRenewal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntentRenewal")
            .field("running", &self.task.is_some())
            .finish()
    }
}

impl IntentRenewal {
    /// Start renewing `intent` on the current tokio runtime.
    pub fn start(intent: Box<WriterIntent>) -> Self {
        let (stop, mut stopped) = oneshot::channel::<()>();
        let client = intent.client.clone();
        let task = tokio::spawn(client.track_fence_guard_task(async move {
            let period = intent.renew_interval();
            let mut renewing = true;
            loop {
                tokio::select! {
                    biased;
                    // A sent stop and a dropped sender both end the attempt.
                    _ = &mut stopped => break,
                    () = tokio::time::sleep(period), if renewing => {
                        match intent.renew().await {
                            Ok(IntentRenewVerdict::Renewed) => {}
                            Ok(IntentRenewVerdict::Lost) => {
                                renewing = false;
                                tracing::debug!(
                                    intent = ?intent,
                                    "writer intent lapsed before the attempt ended; no longer \
                                     renewing it (readers stop honouring it, and an epoch that \
                                     starts aborts the attempt as an uncounted fence wait)"
                                );
                            }
                            Err(error) => tracing::warn!(
                                intent = ?intent,
                                error = %error,
                                "writer intent renewal failed; retrying at the next period (the \
                                 intent lapses at its deadline if renewals keep failing)"
                            ),
                        }
                    }
                }
            }
            if let Err(error) = (*intent).clear().await {
                tracing::warn!(
                    error = %error,
                    "clearing a writer intent failed; the intent lapses at its deadline, so \
                     readers of this layer may wait up to one intent lifetime longer"
                );
            }
        }));
        IntentRenewal {
            stop: Some(stop),
            task: Some(task),
        }
    }

    /// Stop renewing and clear the intent, waiting for the `HDEL`. A failed
    /// clear is logged by the task, not returned: the commit's outcome does
    /// not depend on it.
    pub async fn clear(mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(task) = self.task.take() {
            if let Err(error) = task.await {
                tracing::warn!(
                    error = %error,
                    "the writer-intent renewal task ended abnormally; the intent lapses at its \
                     deadline"
                );
            }
        }
    }
}

impl Drop for IntentRenewal {
    fn drop(&mut self) {
        // Dropping `stop` closes the channel: the task clears the intent and
        // exits on its own. The task handle is detached, not aborted, so that
        // clear still runs.
        self.stop.take();
        self.task.take();
    }
}

#[cfg(test)]
mod tests {
    //! `FenceWaitClock` and `WriterFenceClass` on their own, with no store:
    //! the clock on tokio's paused clock, so every span is exact. The protocol
    //! they serve — fenced commits waiting, uncounted, then committing — is
    //! driven against a live store in
    //! `tests/integration/storage/snapshot_fence_writer_test.rs`.

    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use super::super::FenceMarker;
    use super::*;

    /// The default atomic-read item ceiling.
    const CEILING_ITEMS: u64 = 37_554;

    /// The classification tests' atomic-read ceilings, and a layer size
    /// exactly at them: the default item ceiling, and 449,938 B, the default
    /// byte ceiling until it was raised to 604,619. Only its own values
    /// matter to the tests, never the current defaults.
    const CEILING: LayerSize = LayerSize {
        items: CEILING_ITEMS,
        bytes: 449_938,
    };

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// The recorded coefficients with `rtt99` zeroed, so the jitter is zero
    /// and every sleep is exactly `min(retry_after, poll)`.
    fn jitterless() -> FenceCoefficients {
        let mut c = FenceCoefficients::recorded(CEILING_ITEMS);
        c.rtt99_ms = 0.0;
        c
    }

    fn clock(cap: Duration, remaining: Option<RemainingDeadlineFn>) -> FenceWaitClock {
        FenceWaitClock::new(cap, remaining, &jitterless())
    }

    /// Each sleep is `min(retry_after, poll)` plus a jitter in `[0, rtt99]`,
    /// with `poll` and `rtt99` from the cost model.
    #[tokio::test(start_paused = true)]
    async fn next_sleep_is_the_shorter_of_retry_after_and_poll_plus_at_most_rtt99() {
        let c = FenceCoefficients::recorded(CEILING_ITEMS);
        let poll = ms_to_duration(c.poll_ms());
        let rtt99 = ms_to_duration(c.rtt99_ms);
        assert!(
            rtt99 > Duration::ZERO && poll > ms(100),
            "precondition: a real jitter span and a poll above the short hint"
        );
        let mut clock = FenceWaitClock::new(Duration::from_secs(3_600), None, &c);
        for _ in 0..64 {
            let short = clock
                .next_sleep(ms(100), Instant::now())
                .expect("far inside the cap");
            assert!(
                (ms(100)..=ms(100) + rtt99).contains(&short),
                "a hint below poll sleeps the hint plus ≤ rtt99; got {short:?}"
            );
            let long = clock
                .next_sleep(Duration::from_secs(600), Instant::now())
                .expect("inside the cap");
            assert!(
                (poll..=poll + rtt99).contains(&long),
                "a hint above poll sleeps poll ({poll:?}) plus ≤ rtt99; got {long:?}"
            );
        }
    }

    /// No fence met: `None`. A fence met whose first sleep is refused: zero —
    /// the call met a fence but waited for none of it.
    #[tokio::test(start_paused = true)]
    async fn waited_is_none_until_a_fence_and_zero_when_the_first_sleep_is_refused() {
        let untouched = clock(ms(1_000), None);
        assert_eq!(untouched.waited(), None, "a call that met no fence");

        let mut refused = clock(Duration::ZERO, None);
        assert_eq!(
            refused.next_sleep(ms(100), Instant::now()),
            None,
            "a zero cap allows no sleep"
        );
        assert_eq!(refused.waited(), Some(Duration::ZERO));
    }

    /// `waited()` runs from the start of the first attempt a fence refused or
    /// failed — the `since` the first sleep is handed — to the end of the last
    /// sleep taken: that attempt and the attempts between sleeps count, the
    /// attempt after the last sleep (the one that commits) and anything later
    /// do not. A later call's `since` moves nothing: the span already covers
    /// its attempt.
    #[tokio::test(start_paused = true)]
    async fn waited_spans_the_first_refused_attempts_start_to_the_end_of_the_last_sleep() {
        let mut clock = clock(ms(60_000), None);

        // The first attempt runs 30 ms before the fence refuses (or an epoch
        // fails) it.
        let first_attempt = Instant::now();
        tokio::time::advance(ms(30)).await;
        let first = clock
            .next_sleep(ms(100), first_attempt)
            .expect("inside the cap");
        assert_eq!(first, ms(100));
        assert_eq!(
            clock.waited(),
            Some(ms(130)),
            "the refused attempt's 30 ms + the 100 ms sleep just taken"
        );
        tokio::time::advance(first).await;
        // A re-run attempt the fence refuses again, handed a `since` of its
        // own, which the span already covers.
        let rerun = Instant::now();
        tokio::time::advance(ms(40)).await;
        let second = clock.next_sleep(ms(100), rerun).expect("inside the cap");
        assert_eq!(second, ms(100));
        assert_eq!(
            clock.waited(),
            Some(ms(270)),
            "30 ms attempt + 100 ms sleep + 40 ms refused re-run + the 100 ms sleep just taken"
        );
        tokio::time::advance(second).await;
        // A `since` before the first attempt does not move the start back.
        let earlier = first_attempt
            .checked_sub(ms(1_000))
            .expect("the clock is past its first second");
        let third = clock.next_sleep(ms(100), earlier).expect("inside the cap");
        assert_eq!(
            clock.waited(),
            Some(ms(270) + third),
            "a later call's since never moves the span's start"
        );

        tokio::time::advance(third).await;
        // The committing attempt (seconds, for an above-ceiling structural
        // writer) is not fence wait.
        tokio::time::advance(ms(5_000)).await;
        assert_eq!(clock.waited(), Some(ms(370)));
    }

    /// The cap bounds the span from `since`, not from the first sleep: with a
    /// 200 ms cap, a 100 ms sleep after a 100 ms attempt would end exactly at
    /// the cap and is refused — the call met a fence and waited nothing — and
    /// after a 99 ms attempt it ends at 199 ms and is taken.
    #[tokio::test(start_paused = true)]
    async fn the_cap_counts_the_refused_attempt_from_since() {
        let mut refused = clock(ms(200), None);
        let attempt = Instant::now();
        tokio::time::advance(ms(100)).await;
        assert_eq!(
            refused.next_sleep(ms(100), attempt),
            None,
            "100 ms attempt + 100 ms sleep reaches the 200 ms cap: refused"
        );
        assert_eq!(refused.waited(), Some(Duration::ZERO));

        let mut taken = clock(ms(200), None);
        let attempt = Instant::now();
        tokio::time::advance(ms(99)).await;
        assert_eq!(
            taken.next_sleep(ms(100), attempt),
            Some(ms(100)),
            "99 ms attempt + 100 ms sleep ends at 199 < 200: taken"
        );
        assert_eq!(taken.waited(), Some(ms(199)));
    }

    /// A `since` after `now` (no attempt starts in the future) starts the span
    /// at the call, so the span is never shorter than the sleeps it holds.
    #[tokio::test(start_paused = true)]
    async fn a_since_after_now_starts_the_span_at_the_call() {
        let mut clock = clock(ms(60_000), None);
        let sleep = clock
            .next_sleep(ms(100), Instant::now() + ms(1_000))
            .expect("inside the cap");
        assert_eq!(sleep, ms(100));
        assert_eq!(clock.waited(), Some(ms(100)));
    }

    /// A sleep is taken only if the span it extends to — the first refused
    /// attempt's start (here each first call's own instant: `since` = now) to
    /// the end of that sleep — stays strictly under the cap. The refused sleep
    /// does not move `waited()`.
    #[tokio::test(start_paused = true)]
    async fn the_cap_refuses_a_sleep_whose_end_would_reach_it() {
        let mut clock = clock(ms(200), None);

        assert_eq!(
            clock.next_sleep(ms(100), Instant::now()),
            Some(ms(100)),
            "ends at 100 < 200"
        );
        tokio::time::advance(ms(100)).await;
        assert_eq!(
            clock.next_sleep(ms(100), Instant::now()),
            None,
            "would end exactly at the 200 ms cap: refused"
        );
        assert_eq!(clock.waited(), Some(ms(100)), "only the sleep taken counts");

        let mut clock = clock_with_cap(ms(201));
        assert_eq!(clock.next_sleep(ms(100), Instant::now()), Some(ms(100)));
        tokio::time::advance(ms(100)).await;
        assert_eq!(
            clock.next_sleep(ms(100), Instant::now()),
            Some(ms(100)),
            "ends at 200 < 201: taken"
        );
        tokio::time::advance(ms(100)).await;
        assert_eq!(
            clock.next_sleep(ms(1), Instant::now()),
            None,
            "ends at 201: refused"
        );
        assert_eq!(clock.waited(), Some(ms(200)));
    }

    fn clock_with_cap(cap: Duration) -> FenceWaitClock {
        clock(cap, None)
    }

    /// The remaining RPC deadline is re-read before every sleep, and a sleep
    /// that would not end strictly before it is refused, whatever the cap.
    #[tokio::test(start_paused = true)]
    async fn the_remaining_deadline_is_reread_before_every_sleep_and_bounds_it() {
        let remaining_ms = Arc::new(AtomicU64::new(1_000));
        let reads = Arc::new(AtomicUsize::new(0));
        let remaining: RemainingDeadlineFn = {
            let remaining_ms = Arc::clone(&remaining_ms);
            let reads = Arc::clone(&reads);
            Arc::new(move || {
                reads.fetch_add(1, Ordering::SeqCst);
                Some(ms(remaining_ms.load(Ordering::SeqCst)))
            })
        };
        let mut clock = clock(Duration::from_secs(3_600), Some(remaining));

        assert_eq!(
            clock.next_sleep(ms(100), Instant::now()),
            Some(ms(100)),
            "100 < 1,000 left"
        );
        remaining_ms.store(100, Ordering::SeqCst);
        assert_eq!(
            clock.next_sleep(ms(100), Instant::now()),
            None,
            "100 is not strictly under the 100 ms left: refused"
        );
        remaining_ms.store(101, Ordering::SeqCst);
        assert_eq!(
            clock.next_sleep(ms(100), Instant::now()),
            Some(ms(100)),
            "100 < 101 left"
        );
        assert_eq!(reads.load(Ordering::SeqCst), 3, "re-read on every sleep");
    }

    /// A remaining-deadline reader that answers `None` (no RPC deadline in
    /// scope) leaves the cap as the only bound.
    #[tokio::test(start_paused = true)]
    async fn no_remaining_deadline_in_scope_leaves_only_the_cap() {
        let mut clock = clock(ms(150), Some(Arc::new(|| None)));
        assert_eq!(clock.next_sleep(ms(100), Instant::now()), Some(ms(100)));
        tokio::time::advance(ms(100)).await;
        assert_eq!(
            clock.next_sleep(ms(100), Instant::now()),
            None,
            "the cap still applies"
        );
    }

    /// A fresh call knows nothing; a probe that found no fence key or a
    /// wrong-typed one teaches it nothing.
    #[test]
    fn a_fresh_class_is_not_above_the_ceiling_and_absent_probes_keep_it_so() {
        let class = WriterFenceClass::default();
        assert_eq!(class.above_ceiling(), None);
        class.note_probe(&FenceProbe::Absent, CEILING);
        class.note_probe(
            &FenceProbe::WrongType {
                detail: "WRONGTYPE_KEY: k holds a string, expected hash".to_owned(),
            },
            CEILING,
        );
        assert_eq!(class.above_ceiling(), None);
    }

    /// A present fence key — live or not — classifies the layer: at its stored
    /// size when it has one, else at the ceiling.
    #[test]
    fn a_present_fence_key_classifies_the_layer_at_its_stored_size_or_the_ceiling() {
        let stored = LayerSize {
            items: 150_216,
            bytes: 1_799_752,
        };
        let with_size = WriterFenceClass::default();
        with_size.note_probe(
            &FenceProbe::Present {
                live_retry_after: None,
                stored_size: Some(stored),
                stored_bpi: stored.bytes_per_item(),
                marker: FenceMarker::default(),
            },
            CEILING,
        );
        assert_eq!(with_size.above_ceiling(), Some(stored));

        let without_size = WriterFenceClass::default();
        without_size.note_probe(
            &FenceProbe::Present {
                live_retry_after: Some(ms(5)),
                stored_size: None,
                stored_bpi: None,
                marker: FenceMarker::default(),
            },
            CEILING,
        );
        assert_eq!(without_size.above_ceiling(), Some(CEILING));
    }

    /// Fan-out tallies classify the layer only when they pass either ceiling,
    /// strictly; at the ceiling it is still atomic-readable.
    #[test]
    fn tallies_classify_only_past_either_ceiling() {
        let class = WriterFenceClass::default();
        class.note_tallies(CEILING, CEILING);
        class.note_tallies(
            LayerSize {
                items: 10,
                bytes: 10,
            },
            CEILING,
        );
        assert_eq!(class.above_ceiling(), None, "at or below both: not above");

        let by_items = LayerSize {
            items: CEILING.items + 1,
            bytes: 1,
        };
        class.note_tallies(by_items, CEILING);
        assert_eq!(class.above_ceiling(), Some(by_items));

        let by_bytes = LayerSize {
            items: 1,
            bytes: CEILING.bytes + 1,
        };
        let class = WriterFenceClass::default();
        class.note_tallies(by_bytes, CEILING);
        assert_eq!(class.above_ceiling(), Some(by_bytes));
    }

    /// Monotone: once above the ceiling a call stays so, and each component of
    /// the recorded size only grows.
    #[test]
    fn the_class_is_monotone_per_component() {
        let class = WriterFenceClass::default();
        class.note_above_ceiling(LayerSize {
            items: 100_000,
            bytes: 500_000,
        });
        class.note_above_ceiling(LayerSize {
            items: 50_000,
            bytes: 900_000,
        });
        class.note_probe(&FenceProbe::Absent, CEILING);
        assert_eq!(
            class.above_ceiling(),
            Some(LayerSize {
                items: 100_000,
                bytes: 900_000,
            })
        );
    }
}
