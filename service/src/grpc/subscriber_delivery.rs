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

//! How far behind its commits a `SubscribeReadLayer` live feed runs, measured
//! where the service can see it.
//!
//! A relayed commit passes through four queues between the writer's commit and
//! the client:
//!
//! 1. **The subscription's stream-multiplexer backlog** — the unbounded queue
//!    the stream reader fills and the subscription task drains
//!    (`storage::stream_mux`, counted in entries and bytes by
//!    [`LayerStreamConsumer::backlog`](crate::storage::LayerStreamConsumer::backlog)).
//!    Measured by `live_edit_subscriber_backlog_entries` and
//!    `live_edit_subscriber_backlog_bytes`, sampled by the subscription task as
//!    each live delivery finishes enqueuing, so the reader's offer path is
//!    untouched.
//! 2. **The response channel** — 32 data slots between the subscription task
//!    and tonic (`grpc::subscribe`; a 33rd slot is reserved for the trailer).
//!    The `enqueued` stage of `live_edit_subscriber_delivery_lag_seconds` is
//!    the commit's age when its last message goes onto this channel. A send
//!    that finds every data slot full is a stall ([`SendStall`]):
//!    `live_edit_subscriber_send_stalls_total` counts it and
//!    `live_edit_subscriber_send_wait_seconds` records how long it waited.
//! 3. **The service's HTTP/2 send capacity.** hyper holds the message it last
//!    polled while it waits for stream-window capacity, and polls the response
//!    stream again only once it has some; so a peer whose window is used up
//!    stops tonic draining the channel, and the channel fills. The `transport`
//!    stage is the commit's age when tonic's encoder polls the commit's last
//!    message out of the response stream ([`TransportLagStream`]).
//! 4. **Past the service** — a proxy's buffers and window, then the client.
//!    Nothing here sees them: behind a proxy with a large stream window a
//!    client can fall far behind while every signal above stays low.
//!
//! What is measured is exactly what `live_subscriber_updates_total` counts:
//! steady-state live deliveries whose every message was enqueued. Catch-up
//! content (it has no single commit time), acks, self-echo, duplicates and an
//! incomplete send are not. The lag is the reading replica's clock minus the
//! writing replica's commit stamp ([`commit_lag`]), so cross-replica clock
//! skew enters it.
//!
//! **Window maxima, not per-delivery records.** Every instrument is shared by
//! all subscriber tasks and fanned out to two metric readers, so a record per
//! delivery would contend under fan-out. Each subscription instead keeps the
//! largest value of a window and publishes that one value when the window
//! closes: at [`DELIVERY_TALLY_FLUSH_AT`] deliveries, about every
//! [`DELIVERY_TALLY_FLUSH_INTERVAL`], before a re-bootstrap, and when the
//! stream ends. The histograms therefore show the worst delivery per window,
//! not a per-delivery percentile.
//!
//! Which stream is lagging is never a label. A subscription whose `enqueued`
//! lag, counted only from when it last entered steady state, reaches
//! [`LAGGING_LOG_THRESHOLD`] logs [`LAGGING_LOG_MESSAGE`] at INFO with its
//! identity, at most once per [`LAGGING_LOG_INTERVAL`]. The histograms keep
//! measuring from the commit stamp, so commits that waited out a catch-up show
//! there but do not trigger the line.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::{Instant, Sleep};
use tonic::Status;

use super::proto::SubscribeReadLayerResponse;
use crate::domain::time::now_micros;
use crate::observability::metrics::{
    Metrics, SharedMetrics, DELIVERY_LAG_STAGE_ENQUEUED, DELIVERY_LAG_STAGE_TRANSPORT,
    SEND_PHASE_CATCHUP, SEND_PHASE_LIVE,
};
use crate::storage::notifications::BacklogDepth;

/// Observations a window holds before it is published at once: the live
/// deliveries a [`DeliveryTally`] holds, and the stamped messages a
/// [`TransportLagStream`] holds.
pub const DELIVERY_TALLY_FLUSH_AT: u64 = 64;

/// How long a window may stay open before it is published. The
/// [`DeliveryTally`] is flushed on this period only by `run_steady_state`'s
/// idle wait, so a window can stay open longer while the loop is busy
/// elsewhere (see [`DeliveryTally`]); a [`TransportLagStream`] runs its own
/// timer.
pub const DELIVERY_TALLY_FLUSH_INTERVAL: Duration = Duration::from_secs(1);

/// The steady-state lag at which a live delivery counts as lagging and may log
/// [`LAGGING_LOG_MESSAGE`]: the `enqueued` lag, measured from the later of the
/// commit's stamp and the moment the subscription last entered steady state
/// (its caught-up ack enqueued, at subscribe or after a re-bootstrap). Lag
/// alone is the trigger: an entry that waited in the backlog during steady
/// state carries that wait in its lag, while time spent catching up is not
/// counted, so a slow catch-up does not read as a lagging stream.
pub const LAGGING_LOG_THRESHOLD: Duration = Duration::from_secs(1);

/// The shortest gap between two [`LAGGING_LOG_MESSAGE`] lines of one
/// subscription. The first lagging delivery logs at once.
pub const LAGGING_LOG_INTERVAL: Duration = Duration::from_secs(60);

/// The message of the rate-limited INFO line a lagging subscription logs. It
/// carries `session_id`, `layer_name`, `client_id`, `object_id`,
/// `sequence_num`, `lag_ms`, `max_lag_ms`, `lagging_deliveries`,
/// `backlog_entries` and `backlog_bytes`; `max_lag_ms` and
/// `lagging_deliveries` cover this delivery and every lagging delivery whose
/// line was suppressed since the subscription's previous line. `lag_ms` and
/// `max_lag_ms` are steady-state lags ([`LAGGING_LOG_THRESHOLD`]), which can
/// be shorter than the commit lag the `enqueued` histogram records.
pub const LAGGING_LOG_MESSAGE: &str = "subscriber delivery is lagging";

/// A commit's age: `now_us` minus the commit stamp `server_ts_us`, both in
/// microseconds since the UNIX epoch.
///
/// `None` when `server_ts_us <= 0`: the commit time is unknown, because
/// [`now_micros`] returns `0` on a clock before 1970. A stamp ahead of `now_us`
/// — a writer replica whose clock runs ahead of this one — gives zero, never a
/// negative lag.
pub fn commit_lag(now_us: u64, server_ts_us: i64) -> Option<Duration> {
    if server_ts_us <= 0 {
        return None;
    }
    Some(Duration::from_micros(
        now_us.saturating_sub(server_ts_us as u64),
    ))
}

/// One item on a `SubscribeReadLayer` response channel: the response, and the
/// commit stamp the [`TransportLagStream`] measures it against.
///
/// Only the last message of a live delivery carries a stamp: `commit_end` is
/// that commit's `server_ts`. Every other message — catch-up content, acks, a
/// live delivery's earlier chunks, the terminal trailer — carries `None`. A
/// stamped item still queued when the client disconnects is dropped
/// unmeasured.
#[derive(Debug)]
pub struct Outbound {
    /// What the client receives.
    pub item: Result<SubscribeReadLayerResponse, Status>,
    /// The commit stamp (microseconds since the UNIX epoch) of the live
    /// delivery this item ends, or `None`.
    pub commit_end: Option<i64>,
}

impl Outbound {
    /// An item no lag is measured for.
    pub fn unstamped(item: Result<SubscribeReadLayerResponse, Status>) -> Self {
        Self {
            item,
            commit_end: None,
        }
    }
}

/// What a send to the response channel carries, for the stall accounting's
/// `phase` and for the [`Outbound`] stamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SendKind {
    /// Catch-up content or an ack, including a re-bootstrap's and a
    /// rollback-mode in-window replay's. Never stamped.
    CatchUp,
    /// A steady-state live delivery. `commit_end` is the stamp of the message
    /// that ends the delivery, and `None` on its earlier chunks.
    Live { commit_end: Option<i64> },
}

impl SendKind {
    /// The `phase` label value ([`SEND_PHASE_CATCHUP`] | [`SEND_PHASE_LIVE`]).
    pub(crate) fn phase(self) -> &'static str {
        match self {
            SendKind::CatchUp => SEND_PHASE_CATCHUP,
            SendKind::Live { .. } => SEND_PHASE_LIVE,
        }
    }

    /// The stamp the [`Outbound`] carries.
    pub(crate) fn commit_end(self) -> Option<i64> {
        match self {
            SendKind::CatchUp => None,
            SendKind::Live { commit_end } => commit_end,
        }
    }

    /// The kind of one chunk of a delivery of this kind: a live delivery's
    /// stamp goes on its last chunk only.
    pub(crate) fn for_chunk(self, is_last: bool) -> Self {
        match self {
            SendKind::Live { .. } if !is_last => SendKind::Live { commit_end: None },
            kind => kind,
        }
    }
}

/// A send that found every data slot of the response channel full: counted on
/// `live_edit_subscriber_send_stalls_total{phase}` when it begins, and its wait
/// recorded on `live_edit_subscriber_send_wait_seconds{phase}` when it drops,
/// whatever ended the send — sent, the slow-reader deadline, shutdown, session
/// cancel, disconnect, or the task being dropped mid-send.
pub(crate) struct SendStall<'a> {
    metrics: &'a Metrics,
    phase: &'static str,
    started: Instant,
}

impl<'a> SendStall<'a> {
    /// `Some` only when `tx` has no free permit. On the response channel that
    /// means its 32 data slots are full: the trailer's slot was reserved when
    /// the channel was created, and `Sender::capacity` counts only permits
    /// still available.
    pub(crate) fn begin<T>(
        tx: &mpsc::Sender<T>,
        metrics: &'a Metrics,
        phase: &'static str,
    ) -> Option<Self> {
        if tx.capacity() != 0 {
            return None;
        }
        metrics.inc_subscriber_send_stall(phase);
        Some(Self {
            metrics,
            phase,
            started: Instant::now(),
        })
    }
}

impl Drop for SendStall<'_> {
    fn drop(&mut self) {
        self.metrics
            .observe_subscriber_send_wait(self.phase, self.started.elapsed());
    }
}

/// The response stream handed to tonic: the response channel's items,
/// unchanged and in order, with the `transport` stage of
/// `live_edit_subscriber_delivery_lag_seconds` measured as tonic's encoder
/// polls each stamped item ([`Outbound::commit_end`]).
///
/// The lags go into a window that keeps only the largest, published as one
/// observation at [`DELIVERY_TALLY_FLUSH_AT`] observations, on the stream's own
/// timer about [`DELIVERY_TALLY_FLUSH_INTERVAL`] after the window opened (armed
/// only while a window is open), at end of stream, and on `Drop`. A timer that
/// fires while hyper waits for HTTP/2 capacity is not polled through until
/// hyper polls the stream again; the window is published then. Nothing is
/// allocated per message.
pub struct TransportLagStream {
    rx: mpsc::Receiver<Outbound>,
    metrics: SharedMetrics,
    window: MaxWindow,
    /// Created on the first window and re-armed for each later one.
    timer: Option<Pin<Box<Sleep>>>,
}

impl TransportLagStream {
    /// Wrap the response channel's receiver.
    pub fn new(rx: mpsc::Receiver<Outbound>, metrics: SharedMetrics) -> Self {
        Self {
            rx,
            metrics,
            window: MaxWindow::default(),
            timer: None,
        }
    }

    fn observe(&mut self, lag: Duration, cx: &mut Context<'_>) {
        self.window.observe(lag);
        if self.window.count >= DELIVERY_TALLY_FLUSH_AT {
            self.flush();
            return;
        }
        if self.window.count == 1 {
            // A window opened: arm the timer, and poll it once so it is
            // registered and wakes this task even if the stream is not polled
            // to `Pending` before it fires.
            let deadline = Instant::now() + DELIVERY_TALLY_FLUSH_INTERVAL;
            match self.timer.as_mut() {
                Some(timer) => timer.as_mut().reset(deadline),
                None => self.timer = Some(Box::pin(tokio::time::sleep_until(deadline))),
            }
            if self
                .timer
                .as_mut()
                .is_some_and(|timer| timer.as_mut().poll(cx).is_ready())
            {
                self.flush();
            }
        } else if self.timer.as_ref().is_some_and(|timer| timer.is_elapsed()) {
            self.flush();
        }
    }

    /// Publish the open window, if any. The timer stays as it is: an open
    /// window is what arms it.
    fn flush(&mut self) {
        if let Some(max) = self.window.take() {
            self.metrics
                .observe_subscriber_delivery_lag(DELIVERY_LAG_STAGE_TRANSPORT, max);
        }
    }
}

impl futures::Stream for TransportLagStream {
    type Item = Result<SubscribeReadLayerResponse, Status>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        match this.rx.poll_recv(cx) {
            Poll::Ready(Some(Outbound { item, commit_end })) => {
                if let Some(lag) = commit_end.and_then(|ts| commit_lag(now_micros(), ts)) {
                    this.observe(lag, cx);
                }
                Poll::Ready(Some(item))
            }
            Poll::Ready(None) => {
                this.flush();
                Poll::Ready(None)
            }
            Poll::Pending => {
                if this.window.count > 0 {
                    if let Some(timer) = this.timer.as_mut() {
                        if timer.as_mut().poll(cx).is_ready() {
                            this.flush();
                        }
                    }
                }
                Poll::Pending
            }
        }
    }
}

impl Drop for TransportLagStream {
    fn drop(&mut self) {
        self.flush();
    }
}

/// The largest of a window's observations, and how many there were.
#[derive(Debug, Default)]
struct MaxWindow {
    count: u64,
    max: Duration,
}

impl MaxWindow {
    fn observe(&mut self, value: Duration) {
        self.count += 1;
        self.max = self.max.max(value);
    }

    /// The window's maximum, or `None` when it is empty; empties it.
    fn take(&mut self) -> Option<Duration> {
        let window = std::mem::take(self);
        (window.count > 0).then_some(window.max)
    }
}

/// One steady-state live delivery, every message of which was enqueued.
#[derive(Debug, Clone, Copy)]
pub(crate) struct LiveDelivery {
    pub object_id: u64,
    pub sequence_num: u64,
    /// The commit stamp (microseconds since the UNIX epoch).
    pub server_ts: i64,
    /// The subscription's stream-multiplexer backlog, sampled after the
    /// delivery was enqueued.
    pub backlog: BacklogDepth,
}

/// One subscription's pending live-delivery metrics: the
/// `live_subscriber_updates_total` count, and the window maxima of the
/// `enqueued` lag and of the backlog depth in entries and bytes. Also the
/// lagging-stream log gate.
///
/// Every instrument is shared by all subscriber tasks, so a record per
/// delivery would contend one cell across them under fan-out; batching removes
/// that. (For the counter, the interleaved comparison against the SDD §21.2
/// fan-out shape showed the batched build at parity or better with the
/// uncounted baseline.) The tally accumulates locally and publishes one
/// `add(n)` and one maximum per window when [`DELIVERY_TALLY_FLUSH_AT`]
/// deliveries are pending, when `run_steady_state`'s flush timer fires in its
/// idle wait (every [`DELIVERY_TALLY_FLUSH_INTERVAL`]), before a `Deleted`
/// re-bootstrap, and on `Drop` — every exit from the loop, errors included.
/// Every delivery is counted exactly once; only when it reaches the counter is
/// batched. A delivery therefore reaches the counter normally within about a
/// second, and all of them by the time the stream ends; one waiting behind
/// blocked sends (for a slow reader, each of the delivery's messages — a
/// header and at least one chunk — may wait up to `OUTER_SEND_DEADLINE`, so a
/// multiple of it) reaches it once the loop next goes idle.
///
/// A delivery whose **steady-state lag** reaches [`LAGGING_LOG_THRESHOLD`]
/// logs [`LAGGING_LOG_MESSAGE`] through the tally's [`LaggingLogGate`]. The
/// steady-state lag is measured from the later of the commit's stamp and the
/// moment the subscription entered steady state
/// ([`Self::enter_steady_state`]), so a commit that waited in the backlog
/// while a catch-up ran is not counted as lagging for the catch-up's time;
/// the `enqueued` histogram still measures it from the commit stamp.
/// Crossings the gate suppressed and has not yet reported when the tally
/// drops are discarded.
pub(crate) struct DeliveryTally<'a> {
    metrics: &'a Metrics,
    session_id: &'a str,
    layer_name: &'a str,
    client_id: &'a str,
    /// When the subscription last entered steady state, in microseconds since
    /// the UNIX epoch on this replica's clock; `0` until
    /// [`Self::enter_steady_state`] runs, which measures the lagging trigger
    /// from the commit stamp alone.
    steady_since_us: u64,
    pending: u64,
    /// The largest `enqueued` lag of the window; `None` when no delivery in it
    /// had a commit stamp.
    max_lag: Option<Duration>,
    /// The deepest backlog sampled in the window, entries and bytes each.
    max_backlog: BacklogDepth,
    lagging: LaggingLogGate,
}

impl<'a> DeliveryTally<'a> {
    pub(crate) fn new(
        metrics: &'a Metrics,
        session_id: &'a str,
        layer_name: &'a str,
        client_id: &'a str,
    ) -> Self {
        DeliveryTally {
            metrics,
            session_id,
            layer_name,
            client_id,
            steady_since_us: 0,
            pending: 0,
            max_lag: None,
            max_backlog: BacklogDepth::default(),
            lagging: LaggingLogGate::default(),
        }
    }

    /// The subscription enters steady state now: its caught-up ack is
    /// enqueued, so everything before this instant was catch-up. Called once
    /// the first caught-up ack is enqueued, as `run_steady_state` starts, and
    /// again after each re-bootstrap's ack, so a re-bootstrap's catch-up is not
    /// counted as lagging either.
    pub(crate) fn enter_steady_state(&mut self) {
        self.enter_steady_state_at(now_micros());
    }

    /// [`Self::enter_steady_state`] at an explicit `now_us` (microseconds since
    /// the UNIX epoch), so the lagging trigger is testable.
    pub(crate) fn enter_steady_state_at(&mut self, now_us: u64) {
        self.steady_since_us = now_us;
    }

    /// One live delivery to this subscriber, every message enqueued.
    pub(crate) fn record(&mut self, delivery: LiveDelivery) {
        self.record_at(delivery, now_micros());
    }

    /// [`Self::record`] at an explicit `now_us`, so the lag is testable.
    ///
    /// The lagging line's `lag_ms` and `max_lag_ms` report the steady-state
    /// lag that triggers it (see [`DeliveryTally`]), not the commit lag the
    /// histogram records; the two differ only for a commit stamped before the
    /// subscription entered steady state.
    pub(crate) fn record_at(&mut self, delivery: LiveDelivery, now_us: u64) {
        self.pending += 1;
        let lag = commit_lag(now_us, delivery.server_ts);
        if let Some(lag) = lag {
            self.max_lag = Some(self.max_lag.map_or(lag, |max| max.max(lag)));
        }
        self.max_backlog.entries = self.max_backlog.entries.max(delivery.backlog.entries);
        self.max_backlog.bytes = self.max_backlog.bytes.max(delivery.backlog.bytes);
        // Measured from max(commit stamp, steady-state entry): the smaller of
        // the commit lag and the time since steady state began.
        let steady_lag = lag.map(|lag| {
            lag.min(Duration::from_micros(
                now_us.saturating_sub(self.steady_since_us),
            ))
        });
        if let Some(lag) = steady_lag.filter(|lag| *lag >= LAGGING_LOG_THRESHOLD) {
            if let Some(line) = self.lagging.cross(Instant::now(), lag) {
                tracing::info!(
                    session_id = %self.session_id,
                    layer_name = %self.layer_name,
                    client_id = %self.client_id,
                    object_id = delivery.object_id,
                    sequence_num = delivery.sequence_num,
                    lag_ms = lag.as_millis() as u64,
                    max_lag_ms = line.max_lag.as_millis() as u64,
                    lagging_deliveries = line.lagging_deliveries,
                    backlog_entries = delivery.backlog.entries,
                    backlog_bytes = delivery.backlog.bytes,
                    "{}",
                    LAGGING_LOG_MESSAGE
                );
            }
        }
        if self.pending >= DELIVERY_TALLY_FLUSH_AT {
            self.flush();
        }
    }

    pub(crate) fn has_pending(&self) -> bool {
        self.pending > 0
    }

    /// Publish the window: the delivery count, the largest `enqueued` lag (if
    /// any delivery had a commit stamp) and the deepest backlog. An empty
    /// window publishes nothing.
    pub(crate) fn flush(&mut self) {
        if self.pending == 0 {
            return;
        }
        self.metrics.add_subscriber_updates(self.pending);
        if let Some(lag) = self.max_lag.take() {
            self.metrics
                .observe_subscriber_delivery_lag(DELIVERY_LAG_STAGE_ENQUEUED, lag);
        }
        let backlog = std::mem::take(&mut self.max_backlog);
        self.metrics
            .observe_subscriber_backlog(backlog.entries, backlog.bytes);
        self.pending = 0;
    }
}

impl Drop for DeliveryTally<'_> {
    fn drop(&mut self) {
        self.flush();
    }
}

/// Rate limit of one subscription's [`LAGGING_LOG_MESSAGE`] lines: the first
/// lagging delivery logs at once, and after a line the next lagging delivery
/// logs only once [`LAGGING_LOG_INTERVAL`] has passed. Timed on
/// `tokio::time::Instant`.
#[derive(Debug, Default)]
pub(crate) struct LaggingLogGate {
    last_logged: Option<Instant>,
    /// The largest lag among the crossings since the previous line.
    max_lag: Duration,
    /// Crossings since the previous line.
    crossings: u64,
}

/// What one lagging line reports beyond its own delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LaggingLine {
    /// The largest lag of this crossing and of those suppressed since the
    /// previous line.
    pub max_lag: Duration,
    /// This crossing and those suppressed since the previous line.
    pub lagging_deliveries: u64,
}

impl LaggingLogGate {
    /// One delivery lagged by `lag` at `now`. `Some` when it may log.
    pub(crate) fn cross(&mut self, now: Instant, lag: Duration) -> Option<LaggingLine> {
        self.crossings += 1;
        self.max_lag = self.max_lag.max(lag);
        let due = self
            .last_logged
            .is_none_or(|last| now.saturating_duration_since(last) >= LAGGING_LOG_INTERVAL);
        if !due {
            return None;
        }
        self.last_logged = Some(now);
        let line = LaggingLine {
            max_lag: std::mem::take(&mut self.max_lag),
            lagging_deliveries: std::mem::take(&mut self.crossings),
        };
        Some(line)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::observability::metrics::test_util::histogram_bucket;

    // ── DeliveryTally ───────────────────────────────────────────────────
    //
    // The tally is what turns "one counter add per recipient per commit" into
    // one add per batch. Batching may delay a delivery's arrival on the
    // counter; it must never lose one or count one twice. These pin both
    // halves at the unit level; `grpc::delivery_metrics_test` pins them end to
    // end through a real subscription.

    fn subscriber_updates(metrics: &crate::observability::metrics::Metrics) -> f64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_subscriber_updates_total",
            &[("endpoint", "SubscribeReadLayer")],
        )
        .expect("the SubscribeReadLayer cell is seeded in Metrics::new")
    }

    /// A delivery with no commit stamp and an empty backlog: these tests are
    /// about the count alone.
    fn delivery() -> LiveDelivery {
        LiveDelivery {
            object_id: 1,
            sequence_num: 2,
            server_ts: 0,
            backlog: BacklogDepth::default(),
        }
    }

    fn tally(metrics: &crate::observability::metrics::Metrics) -> DeliveryTally<'_> {
        DeliveryTally::new(metrics, "session", "layer", "client")
    }

    /// Below the batch threshold nothing reaches the counter; the delivery
    /// that reaches it publishes the whole batch in one add.
    #[test]
    fn delivery_tally_publishes_a_full_batch_at_the_threshold() {
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        let mut tally = tally(&metrics);
        for _ in 0..DELIVERY_TALLY_FLUSH_AT - 1 {
            tally.record(delivery());
        }
        assert!(tally.has_pending());
        assert_eq!(
            subscriber_updates(&metrics),
            0.0,
            "below the threshold the deliveries are still pending"
        );
        tally.record(delivery());
        assert_eq!(
            subscriber_updates(&metrics),
            DELIVERY_TALLY_FLUSH_AT as f64,
            "the threshold delivery flushes the whole batch"
        );
        assert!(!tally.has_pending(), "a flush empties the tally");
    }

    /// An explicit flush publishes what is pending and a second flush adds
    /// nothing: the flush before a re-bootstrap followed by the `Drop` flush at
    /// stream end must not count those deliveries twice. `Drop` is the flush on
    /// every exit from `run_steady_state`, errors included, so the deliveries
    /// still pending when the tally drops must be published, not lost.
    #[test]
    fn delivery_tally_flush_then_drop_counts_each_delivery_once() {
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        {
            let mut tally = tally(&metrics);
            for _ in 0..5 {
                tally.record(delivery());
            }
            tally.flush();
            assert_eq!(subscriber_updates(&metrics), 5.0);
            tally.flush();
            assert_eq!(
                subscriber_updates(&metrics),
                5.0,
                "an empty flush adds nothing"
            );
            tally.record(delivery());
            tally.record(delivery());
        }
        assert_eq!(
            subscriber_updates(&metrics),
            7.0,
            "Drop must publish the deliveries recorded after the last flush, and only those"
        );
    }

    // ── Readers for the delivery-lag, stall and backlog instruments ─────
    //
    // None of these histograms is pre-registered (SDD §17.1), so an absent
    // series reads as zero. Every test that asserts a zero through one of them
    // also drives the same reader to a non-zero value somewhere in this module,
    // so a misspelled family or label cannot pass unnoticed.

    const LAG_FAMILY: &str = "live_edit_subscriber_delivery_lag_seconds";
    const SEND_WAIT_FAMILY: &str = "live_edit_subscriber_send_wait_seconds";

    fn sample_or_zero(metrics: &Metrics, name: &str, labels: &[(&str, &str)]) -> f64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(&render(metrics), name, labels).unwrap_or(0.0)
    }

    /// Observations recorded on the `stage` cell of the delivery-lag histogram.
    fn lag_count(metrics: &Metrics, stage: &str) -> u64 {
        sample_or_zero(metrics, &format!("{LAG_FAMILY}_count"), &[("stage", stage)]) as u64
    }

    /// Sum of the observations on the `stage` cell, in seconds.
    fn lag_sum(metrics: &Metrics, stage: &str) -> f64 {
        sample_or_zero(metrics, &format!("{LAG_FAMILY}_sum"), &[("stage", stage)])
    }

    /// Cumulative count of the `stage` cell's bucket `le`.
    fn lag_bucket(metrics: &Metrics, stage: &str, le: &str) -> u64 {
        use crate::observability::metrics::test_util::render;
        histogram_bucket(&render(metrics), LAG_FAMILY, le, &[("stage", stage)]).unwrap_or(0.0)
            as u64
    }

    /// `live_edit_subscriber_send_stalls_total{phase}`. Strict: both cells are
    /// seeded at zero in `Metrics::new`.
    fn send_stalls(metrics: &Metrics, phase: &str) -> u64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_edit_subscriber_send_stalls_total",
            &[("phase", phase)],
        )
        .expect("both send_stalls_total phase cells are seeded at zero") as u64
    }

    fn send_wait_count(metrics: &Metrics, phase: &str) -> u64 {
        sample_or_zero(
            metrics,
            &format!("{SEND_WAIT_FAMILY}_count"),
            &[("phase", phase)],
        ) as u64
    }

    fn send_wait_bucket(metrics: &Metrics, phase: &str, le: &str) -> u64 {
        use crate::observability::metrics::test_util::render;
        histogram_bucket(&render(metrics), SEND_WAIT_FAMILY, le, &[("phase", phase)]).unwrap_or(0.0)
            as u64
    }

    /// `(count, sum)` of an unlabelled backlog histogram (`_entries` or
    /// `_bytes`).
    fn backlog(metrics: &Metrics, family: &str) -> (u64, f64) {
        (
            sample_or_zero(metrics, &format!("{family}_count"), &[]) as u64,
            sample_or_zero(metrics, &format!("{family}_sum"), &[]),
        )
    }

    /// An arbitrary reading-replica clock, in microseconds since the epoch, so
    /// a tally test can state each lag exactly through [`DeliveryTally::record_at`].
    const NOW_US: u64 = 1_800_000_000_000_000;

    /// A live delivery committed `lag` before [`NOW_US`], sampled with the
    /// given backlog.
    fn lagged(lag: Duration, entries: u64, bytes: u64) -> LiveDelivery {
        LiveDelivery {
            object_id: 1,
            sequence_num: 2,
            server_ts: (NOW_US - lag.as_micros() as u64) as i64,
            backlog: BacklogDepth { entries, bytes },
        }
    }

    // ── commit_lag ──────────────────────────────────────────────────────

    #[test]
    fn commit_lag_of_a_past_stamp_is_the_difference() {
        assert_eq!(
            commit_lag(10_000_000, 7_500_000),
            Some(Duration::from_micros(2_500_000))
        );
        assert_eq!(commit_lag(10_000_000, 10_000_000), Some(Duration::ZERO));
    }

    /// A writer replica whose clock runs ahead of the reader's stamps commits
    /// in the reader's future; the lag clamps to zero rather than going
    /// negative or wrapping.
    #[test]
    fn commit_lag_of_a_stamp_ahead_of_now_clamps_to_zero() {
        assert_eq!(commit_lag(10_000_000, 10_000_001), Some(Duration::ZERO));
        assert_eq!(commit_lag(10_000_000, i64::MAX), Some(Duration::ZERO));
    }

    /// `0` is what `now_micros` returns on a clock before 1970, so a zero or
    /// negative stamp means "commit time unknown", never "57 years ago".
    #[test]
    fn commit_lag_of_a_zero_or_negative_stamp_is_none() {
        for stamp in [0, -1, i64::MIN] {
            assert_eq!(commit_lag(10_000_000, stamp), None, "stamp {stamp}");
        }
    }

    // ── SendKind ────────────────────────────────────────────────────────

    /// A live delivery's commit stamp rides on its last chunk only, so the
    /// transport stage is measured once per delivery, when its final message
    /// is polled; catch-up content is never stamped on any chunk.
    #[test]
    fn send_kind_stamps_only_the_last_chunk_of_a_live_delivery() {
        let live = SendKind::Live {
            commit_end: Some(42),
        };
        assert_eq!(live.for_chunk(false).commit_end(), None);
        assert_eq!(live.for_chunk(true).commit_end(), Some(42));
        assert_eq!(live.for_chunk(false).phase(), SEND_PHASE_LIVE);
        assert_eq!(SendKind::CatchUp.for_chunk(true).commit_end(), None);
        assert_eq!(
            SendKind::CatchUp.for_chunk(true).phase(),
            SEND_PHASE_CATCHUP
        );
    }

    // ── DeliveryTally: window maxima ────────────────────────────────────

    /// A window publishes ONE `enqueued` observation, the largest lag in it —
    /// not one per delivery, and not the last or the first — and nothing at all
    /// until it closes. Every lag here is under the lagging-log threshold.
    #[test]
    fn delivery_tally_window_publishes_one_lag_observation_equal_to_its_maximum() {
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        let mut tally = tally(&metrics);
        for lag_ms in [200, 700, 300] {
            tally.record_at(lagged(Duration::from_millis(lag_ms), 0, 0), NOW_US);
        }
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_ENQUEUED),
            0,
            "an open window publishes nothing"
        );

        tally.flush();
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_ENQUEUED),
            1,
            "three deliveries in one window are one observation"
        );
        assert!(
            (lag_sum(&metrics, DELIVERY_LAG_STAGE_ENQUEUED) - 0.7).abs() < 1e-9,
            "the observation is the window's largest lag (0.7 s); got a sum of {}",
            lag_sum(&metrics, DELIVERY_LAG_STAGE_ENQUEUED)
        );
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_TRANSPORT),
            0,
            "the tally records the enqueued stage only"
        );
    }

    /// The backlog window keeps the deepest sample of entries and of bytes
    /// independently: the deepest entry count and the deepest byte count may
    /// come from different deliveries.
    #[test]
    fn delivery_tally_window_publishes_the_deepest_backlog_in_entries_and_bytes() {
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        let mut tally = tally(&metrics);
        for (entries, bytes) in [(3, 100), (9, 50), (5, 400)] {
            tally.record_at(lagged(Duration::from_millis(10), entries, bytes), NOW_US);
        }
        tally.flush();

        assert_eq!(
            backlog(&metrics, "live_edit_subscriber_backlog_entries"),
            (1, 9.0),
            "one entries observation, the deepest sample"
        );
        assert_eq!(
            backlog(&metrics, "live_edit_subscriber_backlog_bytes"),
            (1, 400.0),
            "one bytes observation, the deepest sample — from a different delivery than the \
             deepest entry count"
        );
    }

    /// An empty flush records nothing, and a second flush after a real one
    /// records nothing more: the re-bootstrap flush followed by the `Drop`
    /// flush must not observe a window twice.
    #[test]
    fn delivery_tally_empty_flush_records_no_observation() {
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        let mut tally = tally(&metrics);
        tally.flush();
        assert_eq!(lag_count(&metrics, DELIVERY_LAG_STAGE_ENQUEUED), 0);
        assert_eq!(
            backlog(&metrics, "live_edit_subscriber_backlog_entries").0,
            0
        );

        tally.record_at(lagged(Duration::from_millis(5), 1, 1), NOW_US);
        tally.flush();
        tally.flush();
        assert_eq!(lag_count(&metrics, DELIVERY_LAG_STAGE_ENQUEUED), 1);
        assert_eq!(
            backlog(&metrics, "live_edit_subscriber_backlog_entries").0,
            1
        );
    }

    /// A window whose deliveries carry no commit stamp is counted and its
    /// backlog published, but it has no lag to observe.
    #[test]
    fn delivery_tally_window_without_a_commit_stamp_publishes_no_lag() {
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        let mut tally = tally(&metrics);
        tally.record_at(delivery(), NOW_US);
        tally.record_at(delivery(), NOW_US);
        tally.flush();

        assert_eq!(subscriber_updates(&metrics), 2.0);
        assert_eq!(
            backlog(&metrics, "live_edit_subscriber_backlog_entries").0,
            1,
            "the window was published"
        );
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_ENQUEUED),
            0,
            "no delivery had a commit time, so there is no lag to observe"
        );
    }

    /// `Drop` closes the open window: `run_steady_state` returns through it on
    /// every exit, errors included.
    #[test]
    fn delivery_tally_drop_publishes_the_open_window() {
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        {
            let mut tally = tally(&metrics);
            tally.record_at(lagged(Duration::from_millis(400), 2, 64), NOW_US);
        }
        assert_eq!(lag_count(&metrics, DELIVERY_LAG_STAGE_ENQUEUED), 1);
        assert!((lag_sum(&metrics, DELIVERY_LAG_STAGE_ENQUEUED) - 0.4).abs() < 1e-9);
        assert_eq!(
            backlog(&metrics, "live_edit_subscriber_backlog_entries"),
            (1, 2.0)
        );
    }

    /// The window closes on its own at [`DELIVERY_TALLY_FLUSH_AT`] deliveries,
    /// publishing the maximum of all of them.
    #[test]
    fn delivery_tally_window_closes_at_the_batch_threshold() {
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        let mut tally = tally(&metrics);
        for i in 0..DELIVERY_TALLY_FLUSH_AT - 1 {
            // The 11th delivery is the window's worst: 0.9 s, against 1–63 ms.
            let lag_ms = if i == 10 { 900 } else { i + 1 };
            tally.record_at(lagged(Duration::from_millis(lag_ms), 0, 0), NOW_US);
        }
        assert_eq!(lag_count(&metrics, DELIVERY_LAG_STAGE_ENQUEUED), 0);

        tally.record_at(lagged(Duration::from_millis(1), 0, 0), NOW_US);
        assert_eq!(lag_count(&metrics, DELIVERY_LAG_STAGE_ENQUEUED), 1);
        assert!(
            (lag_sum(&metrics, DELIVERY_LAG_STAGE_ENQUEUED) - 0.9).abs() < 1e-9,
            "the threshold flush publishes the whole window's maximum"
        );
    }

    // ── LaggingLogGate ──────────────────────────────────────────────────

    /// The first crossing logs at once; crossings within
    /// [`LAGGING_LOG_INTERVAL`] are suppressed but counted; the next line,
    /// once the interval has passed, carries how many crossings it covers and
    /// the largest lag among them; and both then reset.
    #[test]
    fn lagging_log_gate_logs_the_first_crossing_then_at_most_once_per_interval() {
        let start = Instant::now();
        let secs = Duration::from_secs;
        let mut gate = LaggingLogGate::default();

        assert_eq!(
            gate.cross(start, secs(2)),
            Some(LaggingLine {
                max_lag: secs(2),
                lagging_deliveries: 1,
            }),
            "the first crossing logs at once"
        );
        assert_eq!(gate.cross(start + secs(1), secs(5)), None);
        assert_eq!(
            gate.cross(
                start + LAGGING_LOG_INTERVAL - Duration::from_millis(1),
                secs(3)
            ),
            None,
            "a crossing just inside the interval is still suppressed"
        );
        assert_eq!(
            gate.cross(start + LAGGING_LOG_INTERVAL, Duration::from_millis(1_500)),
            Some(LaggingLine {
                max_lag: secs(5),
                lagging_deliveries: 3,
            }),
            "the next line covers itself and the two suppressed crossings, with their maximum"
        );
        assert_eq!(
            gate.cross(start + LAGGING_LOG_INTERVAL * 2, secs(1)),
            Some(LaggingLine {
                max_lag: secs(1),
                lagging_deliveries: 1,
            }),
            "the count and maximum reset after each line"
        );
    }

    /// Capture sink for [`capture_json`]: the tree-wide `CapturingWriter`
    /// idiom (`grpc::rpc_log`'s tests).
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

    /// A thread-local JSON capture at INFO, live until the guard drops. Only
    /// sound on a current-thread runtime, where everything the test awaits is
    /// polled on the thread the guard is installed on.
    fn start_json_capture() -> (
        tracing::subscriber::DefaultGuard,
        std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    ) {
        let buf = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(std::sync::Arc::clone(&buf)))
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .finish();
        (tracing::subscriber::set_default(subscriber), buf)
    }

    /// Every event a [`start_json_capture`] buffer holds, parsed.
    fn captured_events(buf: &std::sync::Mutex<Vec<u8>>) -> Vec<serde_json::Value> {
        let raw = buf.lock().unwrap_or_else(|p| p.into_inner()).clone();
        String::from_utf8(raw)
            .expect("captured tracing output must be UTF-8")
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| serde_json::from_str(line).expect("each captured line must be JSON"))
            .collect()
    }

    /// The tally's lagging line end to end, on paused tokio time: a lag just
    /// under [`LAGGING_LOG_THRESHOLD`] never logs, a lag at it does, a second
    /// lagging delivery 30 s later is suppressed, and the line due once
    /// [`LAGGING_LOG_INTERVAL`] has passed reports both crossings and their
    /// maximum. Only the line is asserted; the window's metrics, published
    /// when the tally drops, are covered by the window tests above. This tally
    /// never enters steady state, so its trigger is the commit lag itself; the
    /// steady-state reference has its own tests below.
    #[tokio::test(start_paused = true)]
    async fn delivery_tally_logs_a_lagging_delivery_at_most_once_per_interval() {
        assert_eq!(
            tokio::runtime::Handle::current().runtime_flavor(),
            tokio::runtime::RuntimeFlavor::CurrentThread,
            "the thread-local capture only sees this test's events on a current-thread runtime"
        );
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        let (capture, buf) = start_json_capture();
        {
            let mut tally = tally(&metrics);
            let at = |sequence_num, lag: Duration| LiveDelivery {
                sequence_num,
                ..lagged(lag, 7, 4_096)
            };
            tally.record_at(
                at(1, LAGGING_LOG_THRESHOLD - Duration::from_millis(1)),
                NOW_US,
            );
            tally.record_at(at(2, LAGGING_LOG_THRESHOLD), NOW_US);
            // `record_at` reads the gate's clock through tokio's `Instant`,
            // which `advance` moves on a paused runtime.
            tokio::time::advance(Duration::from_secs(30)).await;
            tally.record_at(at(3, Duration::from_secs(5)), NOW_US);
            tally.record_at(at(4, Duration::from_millis(400)), NOW_US);
            tokio::time::advance(Duration::from_secs(30)).await;
            tally.record_at(at(5, Duration::from_secs(2)), NOW_US);
        }
        drop(capture);
        let events = captured_events(&buf);

        let lines: Vec<_> = events
            .iter()
            .filter(|e| e["fields"]["message"] == LAGGING_LOG_MESSAGE)
            .collect();
        assert_eq!(
            lines.len(),
            2,
            "one line for the first crossing (seq 2) and one once the interval passed (seq 5); \
             the 999 ms and 400 ms deliveries are not lagging and seq 3 was suppressed: {events:?}"
        );

        let first = lines[0];
        assert_eq!(first["level"], "INFO");
        assert_eq!(first["fields"]["session_id"], "session");
        assert_eq!(first["fields"]["layer_name"], "layer");
        assert_eq!(first["fields"]["client_id"], "client");
        assert_eq!(first["fields"]["object_id"], 1);
        assert_eq!(first["fields"]["sequence_num"], 2);
        assert_eq!(first["fields"]["lag_ms"], 1_000);
        assert_eq!(first["fields"]["max_lag_ms"], 1_000);
        assert_eq!(first["fields"]["lagging_deliveries"], 1);
        assert_eq!(first["fields"]["backlog_entries"], 7);
        assert_eq!(first["fields"]["backlog_bytes"], 4_096);

        let second = lines[1];
        assert_eq!(second["fields"]["sequence_num"], 5);
        assert_eq!(second["fields"]["lag_ms"], 2_000);
        assert_eq!(
            second["fields"]["max_lag_ms"], 5_000,
            "the suppressed seq 3's 5 s lag is the maximum the line reports"
        );
        assert_eq!(
            second["fields"]["lagging_deliveries"], 2,
            "seq 3 (suppressed) and seq 5; the 400 ms seq 4 was not a crossing"
        );
    }

    // ── DeliveryTally: the lagging trigger's steady-state reference ─────
    //
    // The lagging line triggers on the lag measured from the later of the
    // commit stamp and the moment the subscription entered steady state, so
    // time a commit spent waiting out a catch-up is not "lagging". The
    // `enqueued` histogram keeps measuring from the commit stamp. Every clock
    // here is explicit (`enter_steady_state_at` / `record_at`), and each test
    // uses a fresh tally, so the gate's rate limit never suppresses a line.

    /// When this subscription entered steady state (microseconds since the
    /// epoch), and a commit stamp 5 s before it.
    const STEADY_SINCE_US: u64 = NOW_US;
    const STAMPED_BEFORE_STEADY: i64 = (STEADY_SINCE_US - 5_000_000) as i64;

    /// A live delivery stamped at `server_ts` (microseconds since the epoch).
    fn stamped_delivery(server_ts: i64) -> LiveDelivery {
        LiveDelivery {
            server_ts,
            ..delivery()
        }
    }

    /// Drive a fresh tally through `steps` under a JSON capture, drop it (which
    /// publishes its window), and return the lagging lines it logged.
    fn lagging_lines(
        metrics: &Metrics,
        steps: impl FnOnce(&mut DeliveryTally<'_>),
    ) -> Vec<serde_json::Value> {
        let (capture, buf) = start_json_capture();
        {
            let mut tally = tally(metrics);
            steps(&mut tally);
        }
        drop(capture);
        captured_events(&buf)
            .into_iter()
            .filter(|e| e["fields"]["message"] == LAGGING_LOG_MESSAGE)
            .collect()
    }

    /// (a) A commit stamped 5 s before steady state began, delivered 0.5 s
    /// into steady state, has a 5.5 s commit lag but only a 0.5 s steady-state
    /// lag: it does not log, and the `enqueued` histogram still records the
    /// full 5.5 s.
    #[test]
    fn delivery_tally_does_not_log_a_commit_whose_lag_is_mostly_catch_up_time() {
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        let lines = lagging_lines(&metrics, |tally| {
            tally.enter_steady_state_at(STEADY_SINCE_US);
            tally.record_at(
                stamped_delivery(STAMPED_BEFORE_STEADY),
                STEADY_SINCE_US + 500_000,
            );
        });

        assert!(
            lines.is_empty(),
            "0.5 s of steady state is under the 1 s trigger, whatever the commit's age: {lines:?}"
        );
        assert_eq!(lag_count(&metrics, DELIVERY_LAG_STAGE_ENQUEUED), 1);
        assert!(
            (lag_sum(&metrics, DELIVERY_LAG_STAGE_ENQUEUED) - 5.5).abs() < 1e-9,
            "the histogram measures from the commit stamp: 5.5 s; got {}",
            lag_sum(&metrics, DELIVERY_LAG_STAGE_ENQUEUED)
        );
    }

    /// (b) The same commit delivered 1.5 s into steady state logs, and the
    /// line reports the 1.5 s steady-state lag, not the 6.5 s commit lag.
    #[test]
    fn delivery_tally_logs_a_commit_once_steady_state_has_lasted_the_threshold() {
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        let lines = lagging_lines(&metrics, |tally| {
            tally.enter_steady_state_at(STEADY_SINCE_US);
            tally.record_at(
                stamped_delivery(STAMPED_BEFORE_STEADY),
                STEADY_SINCE_US + 1_500_000,
            );
        });

        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0]["fields"]["lag_ms"], 1_500);
        assert_eq!(lines[0]["fields"]["max_lag_ms"], 1_500);
        assert!(
            (lag_sum(&metrics, DELIVERY_LAG_STAGE_ENQUEUED) - 6.5).abs() < 1e-9,
            "the histogram still records the 6.5 s commit lag"
        );
    }

    /// (c) A commit stamped after steady state began is measured from its own
    /// stamp, as before: 2 s after a stamp 1 s into steady state logs 2 s, and
    /// 0.5 s after a stamp 2.5 s in does not log.
    #[test]
    fn delivery_tally_measures_a_commit_stamped_in_steady_state_from_its_stamp() {
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        let at_3s = STEADY_SINCE_US + 3_000_000;
        let lines = lagging_lines(&metrics, |tally| {
            tally.enter_steady_state_at(STEADY_SINCE_US);
            tally.record_at(
                stamped_delivery((STEADY_SINCE_US + 2_500_000) as i64),
                at_3s,
            );
            tally.record_at(
                stamped_delivery((STEADY_SINCE_US + 1_000_000) as i64),
                at_3s,
            );
        });

        assert_eq!(
            lines.len(),
            1,
            "only the commit 2 s old logs; the one 0.5 s old is under the trigger even 3 s into \
             steady state: {lines:?}"
        );
        assert_eq!(lines[0]["fields"]["lag_ms"], 2_000);
    }

    /// (d) Re-entering steady state — after a re-bootstrap's ack — moves the
    /// reference: a commit 10.5 s old, delivered 0.5 s after the re-entry, does
    /// not log, although it is 10.5 s past the first entry too.
    #[test]
    fn delivery_tally_reentering_steady_state_resets_the_trigger_reference() {
        let metrics = crate::observability::metrics::metrics_scrape_only("tally");
        let reentered = STEADY_SINCE_US + 10_000_000;
        let lines = lagging_lines(&metrics, |tally| {
            tally.enter_steady_state_at(STEADY_SINCE_US);
            tally.enter_steady_state_at(reentered);
            tally.record_at(
                stamped_delivery(STEADY_SINCE_US as i64),
                reentered + 500_000,
            );
        });

        assert!(
            lines.is_empty(),
            "0.5 s since the re-entry is under the trigger: {lines:?}"
        );
        assert!(
            (lag_sum(&metrics, DELIVERY_LAG_STAGE_ENQUEUED) - 10.5).abs() < 1e-9,
            "the histogram still records the 10.5 s commit lag"
        );
    }

    // ── SendStall ───────────────────────────────────────────────────────

    /// A channel shaped like the response channel: 33 slots, one reserved for
    /// the trailer before anything is sent, and `queued` data messages in it.
    async fn response_channel(
        queued: usize,
    ) -> (mpsc::Sender<u8>, mpsc::Receiver<u8>, mpsc::OwnedPermit<u8>) {
        let (tx, rx) = mpsc::channel(33);
        let trailer = tx
            .clone()
            .reserve_owned()
            .await
            .expect("the receiver is alive");
        for _ in 0..queued {
            tx.try_send(0).expect("a data slot is free");
        }
        (tx, rx, trailer)
    }

    /// With all 32 data slots full, a send is a stall: counted the moment it
    /// begins, under its own phase only, and its wait recorded when it ends —
    /// as the time since it began.
    #[tokio::test(start_paused = true)]
    async fn send_stall_on_a_full_response_channel_counts_at_begin_and_records_its_wait_on_drop() {
        let metrics = crate::observability::metrics::metrics_scrape_only("stall");
        let (tx, _rx, _trailer) = response_channel(32).await;

        let stall = SendStall::begin(&tx, &metrics, SEND_PHASE_LIVE)
            .expect("every data slot is full, so this send is a stall");
        assert_eq!(
            send_stalls(&metrics, SEND_PHASE_LIVE),
            1,
            "counted at begin"
        );
        assert_eq!(
            send_wait_count(&metrics, SEND_PHASE_LIVE),
            0,
            "the wait is recorded when the send ends, not when it begins"
        );

        tokio::time::advance(Duration::from_secs(3)).await;
        drop(stall);
        assert_eq!(send_wait_count(&metrics, SEND_PHASE_LIVE), 1);
        assert_eq!(
            send_wait_bucket(&metrics, SEND_PHASE_LIVE, "2.5"),
            0,
            "the recorded wait is the 3 s since the stall began"
        );
        assert_eq!(
            (
                send_stalls(&metrics, SEND_PHASE_CATCHUP),
                send_wait_count(&metrics, SEND_PHASE_CATCHUP)
            ),
            (0, 0),
            "a live stall records nothing under the catchup phase"
        );
    }

    /// One free data slot means the send will not wait: nothing is counted or
    /// recorded, under either phase.
    #[tokio::test]
    async fn send_stall_with_a_free_data_slot_records_nothing() {
        let metrics = crate::observability::metrics::metrics_scrape_only("stall");
        let (tx, _rx, _trailer) = response_channel(31).await;

        for phase in [SEND_PHASE_CATCHUP, SEND_PHASE_LIVE] {
            assert!(SendStall::begin(&tx, &metrics, phase).is_none(), "{phase}");
            assert_eq!(send_stalls(&metrics, phase), 0, "{phase}");
            assert_eq!(send_wait_count(&metrics, phase), 0, "{phase}");
        }
    }

    // ── TransportLagStream ──────────────────────────────────────────────

    fn header(sequence_num: u64) -> SubscribeReadLayerResponse {
        use super::super::proto::{subscribe_read_layer_response, SubscribeReadLayerHeader};
        SubscribeReadLayerResponse {
            response_type: Some(subscribe_read_layer_response::ResponseType::Header(
                SubscribeReadLayerHeader {
                    sequence_num,
                    ..Default::default()
                },
            )),
        }
    }

    fn stamped(sequence_num: u64, commit_end: i64) -> Outbound {
        Outbound {
            item: Ok(header(sequence_num)),
            commit_end: Some(commit_end),
        }
    }

    fn transport_stream(
        capacity: usize,
    ) -> (Arc<Metrics>, mpsc::Sender<Outbound>, TransportLagStream) {
        let metrics = Arc::new(crate::observability::metrics::metrics_scrape_only(
            "transport",
        ));
        let (tx, rx) = mpsc::channel(capacity);
        let stream = TransportLagStream::new(rx, Arc::clone(&metrics));
        (metrics, tx, stream)
    }

    /// A stamped item is measured against its commit stamp when it is polled:
    /// a commit stamped 5 s before it was queued lands above `le="2.5"`. The
    /// lower bound comes from the synthetic stamp, not from timing.
    #[tokio::test]
    async fn transport_lag_stream_measures_a_stamped_item_against_its_commit_stamp() {
        use futures::StreamExt;
        let (metrics, tx, mut stream) = transport_stream(4);
        let five_seconds_ago = now_micros() as i64 - 5_000_000;
        tx.send(stamped(7, five_seconds_ago))
            .await
            .expect("the stream holds the receiver");
        drop(tx);

        let item = stream.next().await.expect("the item is delivered");
        assert_eq!(
            item.expect("an Ok item"),
            header(7),
            "passed through unchanged"
        );
        assert!(stream.next().await.is_none(), "the channel closed");

        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_TRANSPORT),
            1,
            "end of stream publishes the open window"
        );
        assert_eq!(
            lag_bucket(&metrics, DELIVERY_LAG_STAGE_TRANSPORT, "2.5"),
            0,
            "a commit stamped 5 s before it was queued is at least 5 s old when polled"
        );
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_ENQUEUED),
            0,
            "the stream records the transport stage only"
        );
    }

    /// Unstamped items, a stamp with no commit time, and the `Err` trailer
    /// pass through unchanged and in order, and record nothing; the stream ends
    /// when the channel does.
    #[tokio::test]
    async fn transport_lag_stream_passes_unmeasured_items_and_the_trailer_through_unchanged() {
        use futures::StreamExt;
        let (metrics, tx, stream) = transport_stream(4);
        tx.send(Outbound::unstamped(Ok(header(1)))).await.unwrap();
        tx.send(stamped(2, 0)).await.unwrap();
        tx.send(Outbound::unstamped(Err(Status::unavailable("trailer"))))
            .await
            .unwrap();
        drop(tx);

        let items: Vec<_> = stream.collect().await;
        assert_eq!(items.len(), 3, "every item, then end of stream");
        assert_eq!(items[0].as_ref().expect("Ok"), &header(1));
        assert_eq!(items[1].as_ref().expect("Ok"), &header(2));
        let trailer = items[2].as_ref().expect_err("the trailer is the Err item");
        assert_eq!(
            (trailer.code(), trailer.message()),
            (tonic::Code::Unavailable, "trailer")
        );
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_TRANSPORT),
            0,
            "nothing stamped with a commit time passed, so nothing is observed"
        );
    }

    /// A window closes on its own at [`DELIVERY_TALLY_FLUSH_AT`] stamped items
    /// and publishes their maximum. The other 63 are stamped in the future, so
    /// each is a zero lag; the one 30 s-old item must be what is published.
    /// Paused time keeps the window's timer from closing it first.
    #[tokio::test(start_paused = true)]
    async fn transport_lag_stream_publishes_a_window_at_the_batch_threshold() {
        use futures::StreamExt;
        let (metrics, tx, mut stream) = transport_stream(DELIVERY_TALLY_FLUSH_AT as usize);
        let future = now_micros() as i64 + 3_600_000_000;
        let thirty_seconds_ago = now_micros() as i64 - 30_000_000;
        for seq in 0..DELIVERY_TALLY_FLUSH_AT {
            let stamp = if seq == 10 {
                thirty_seconds_ago
            } else {
                future
            };
            tx.send(stamped(seq, stamp)).await.unwrap();
        }

        for _ in 0..DELIVERY_TALLY_FLUSH_AT - 1 {
            stream.next().await.expect("queued").expect("Ok");
        }
        assert_eq!(lag_count(&metrics, DELIVERY_LAG_STAGE_TRANSPORT), 0);

        stream.next().await.expect("queued").expect("Ok");
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_TRANSPORT),
            1,
            "the 64th stamped item closes the window"
        );
        assert_eq!(
            lag_bucket(&metrics, DELIVERY_LAG_STAGE_TRANSPORT, "20"),
            0,
            "the window's maximum, the 30 s-old item, is what is published"
        );
        drop(tx);
    }

    /// An open window is published on the stream's own timer while the
    /// channel is idle and the client still connected — not only at the
    /// threshold or end of stream. On paused time, waiting past
    /// [`DELIVERY_TALLY_FLUSH_INTERVAL`] fires the timer the window armed.
    #[tokio::test(start_paused = true)]
    async fn transport_lag_stream_publishes_an_open_window_on_its_timer() {
        use futures::StreamExt;
        let (metrics, tx, mut stream) = transport_stream(4);
        tx.send(stamped(1, now_micros() as i64 - 5_000_000))
            .await
            .unwrap();
        stream.next().await.expect("queued").expect("Ok");
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_TRANSPORT),
            0,
            "the window is open"
        );

        let idle = tokio::time::timeout(DELIVERY_TALLY_FLUSH_INTERVAL * 2, stream.next()).await;
        assert!(idle.is_err(), "nothing more was queued");
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_TRANSPORT),
            1,
            "the window's timer published it while the stream sat idle"
        );
        drop(tx);
    }

    /// Each window gets its own full interval: once a window has closed on
    /// the timer, the next window's first item re-arms it. Two items polled
    /// straight after are one open window, not two — a timer left at the old,
    /// already-elapsed deadline would close the new window on its first item.
    #[tokio::test(start_paused = true)]
    async fn transport_lag_stream_rearms_its_timer_for_each_later_window() {
        use futures::StreamExt;
        let (metrics, tx, mut stream) = transport_stream(4);
        tx.send(stamped(1, now_micros() as i64 - 5_000_000))
            .await
            .unwrap();
        stream.next().await.expect("queued").expect("Ok");
        let idle = tokio::time::timeout(DELIVERY_TALLY_FLUSH_INTERVAL * 2, stream.next()).await;
        assert!(idle.is_err(), "nothing more was queued");
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_TRANSPORT),
            1,
            "precondition: the first window closed on its timer"
        );

        for seq in [2, 3] {
            tx.send(stamped(seq, now_micros() as i64 - 5_000_000))
                .await
                .unwrap();
            stream.next().await.expect("queued").expect("Ok");
        }
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_TRANSPORT),
            1,
            "the second window is still open: its timer was re-armed a full interval from its \
             first item, so neither of its two items closed it"
        );
        drop(tx);
    }

    /// A window whose interval passes while the stream is never left idle is
    /// still published: the next stamped item finds the timer elapsed and
    /// closes the window, without waiting for a `Pending` poll. Paused time
    /// moves past the deadline while the stream is not polled at all.
    #[tokio::test(start_paused = true)]
    async fn transport_lag_stream_publishes_an_overdue_window_on_its_next_item() {
        use futures::StreamExt;
        let (metrics, tx, mut stream) = transport_stream(4);
        tx.send(stamped(1, now_micros() as i64 - 5_000_000))
            .await
            .unwrap();
        stream.next().await.expect("queued").expect("Ok");
        tokio::time::advance(DELIVERY_TALLY_FLUSH_INTERVAL * 2).await;
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_TRANSPORT),
            0,
            "precondition: the stream was not polled, so the overdue window is still open"
        );

        tx.send(stamped(2, now_micros() as i64 - 5_000_000))
            .await
            .unwrap();
        stream.next().await.expect("queued").expect("Ok");
        assert_eq!(
            lag_count(&metrics, DELIVERY_LAG_STAGE_TRANSPORT),
            1,
            "the item polled after the deadline closes the overdue window"
        );
        drop(tx);
    }

    /// Dropping the stream — tonic does when the client goes away — publishes
    /// the open window.
    #[tokio::test]
    async fn transport_lag_stream_drop_publishes_the_open_window() {
        use futures::StreamExt;
        let (metrics, tx, mut stream) = transport_stream(4);
        tx.send(stamped(1, now_micros() as i64 - 5_000_000))
            .await
            .unwrap();
        stream.next().await.expect("queued").expect("Ok");
        drop(stream);
        assert_eq!(lag_count(&metrics, DELIVERY_LAG_STAGE_TRANSPORT), 1);
        drop(tx);
    }
}
