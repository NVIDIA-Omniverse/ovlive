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

//! Stream multiplexer reader failure handling, driven by injected reader
//! faults (`MuxFaultSite::Xread` and `MuxFaultSite::Wake`): which errors a
//! reader rides out, where it gives up, that every give-up ends its
//! subscriptions with a recorded `Unavailable` reason, how a whole-reader
//! failure heals and backs off, and how wakes that fail or hang are handled.
//! Also, through the same test knobs: each subscription's backlog bounds (in
//! entries and in bytes; hard caps from registration, the soft watchdog only
//! once the consumer drains, and the `bound` each ending is counted under),
//! and the byte-sized `COUNT` of every read.
//!
//! Compiled only under `test-fault-injection` (see `storage/mod.rs`).
//!
//! Waits are deadlines (`SETTLE`), not measurements; the soft-watchdog tests
//! also sleep past their shortened watchdog, and the failed-liveness-wake test
//! past its shortened liveness age, because each is a timer.
//! The few negative checks —
//! "no forced reconnect happened", "no read was issued" — hold for a bounded
//! observation window ([`QUIET`]) sized to many multiples of the shortened
//! timer they watch; such a window can only let a regression through under
//! extreme load, never fail a correct build.

use std::time::Duration;

use fred::error::ErrorKind;
use fred::interfaces::KeysInterface;
use fred::types::Value;

use live_edit::observability::metrics::{
    CONSUMER_PHASE_PERMANENT, CONSUMER_PHASE_PROBE, CONSUMER_PHASE_WAKE_ACK, CONSUMER_PHASE_XREAD,
    STREAM_EXIT_ERROR, STREAM_EXIT_INNER_WATCHDOG, STREAM_EXIT_SHUTDOWN, WAKE_OUTCOME_ERROR,
    WAKE_OUTCOME_OK,
};
use live_edit::storage::client::{
    BUDGET_EXHAUSTED_SHAPES, POOL_LABEL_SUBSCRIBER, STORE_NOT_SERVING_SHAPES,
};
use live_edit::storage::notifications::READER_ENDED_WITHOUT_REASON_MSG;
use live_edit::storage::stream_mux::{MuxFault, MuxFaultSite, StreamMuxKnobs};
use live_edit::storage::test_support::{
    command_timeout_row, command_timeouts_only_on, release_all, ConnStall, TestValkey,
};
use live_edit::storage::LayerStreamConsumer;

use super::mux_test_helpers::{
    assert_ends_with, assert_quiet, await_faults_consumed, await_lines, await_mux,
    await_readers_parked, await_series, consume, fast_backoff_knobs, fault, forced_reconnect_row,
    forced_reconnects_only_on, install_metrics, push_faults, push_timeout_faults,
    read_permanent_exits, read_series, read_unblocked_retries, read_wakes, recv_seqs, xadd_update,
    QUIET, SETTLE, STREAMS, UNBLOCKED_REPLY, WAKE_ACK_WEDGE_MSG, WHOLE_READER_BACKOFF_MSG,
};
use crate::grpc::log_capture::{assert_field, capture_debug, purge_keys, Captured, ThreadCapture};

/// The WARN a wake task logs when every attempt of a coverage wake — one for a
/// joining subscription the in-flight read does not cover — failed.
const WAKE_FAILED_COVERAGE_MSG: &str =
    "stream multiplexer wake failed; the new subscription's first live update may wait until \
     the in-flight XREAD returns (up to XREAD_BLOCK_MS)";

/// The WARN a wake task logs when every attempt of a liveness wake — one for a
/// covered join while the reader's connection has not answered recently —
/// failed. It shares [`WAKE_FAILED_COVERAGE_MSG`]'s prefix, so the tests
/// below tell the two apart by the full message.
const WAKE_FAILED_LIVENESS_MSG: &str =
    "stream multiplexer wake failed; the joining subscription's delivery is not delayed, but it \
     gets no connection check from this wake (the next join sends another; the XREAD's own \
     timeout still applies)";

/// Waits until the reader's latest XREAD names `keys` stream keys.
async fn await_read_over(tv: &TestValkey, keys: usize) {
    await_mux(tv.client(), "the reader reads every key", |s| {
        s.readers[0].keys == keys
    })
    .await;
}

// ── Transient failures: ridden out ───────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transient_failures_are_ridden_out_without_losing_entries() {
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client()
        .stream_mux
        .set_knobs_for_test(fast_backoff_knobs(tv.client()));
    push_faults(tv.client(), 3, ErrorKind::IO, "synthetic connection reset");
    for seq in 1..=3 {
        xadd_update(&tv, 1, seq).await;
    }
    let c = consume(&tv, 1).await;
    await_faults_consumed(tv.client()).await;
    xadd_update(&tv, 1, 4).await;
    assert_eq!(recv_seqs(&c, 4).await, [1, 2, 3, 4]);
    assert!(c.take_terminal_exit().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn tryagain_during_slot_migration_is_retried() {
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client()
        .stream_mux
        .set_knobs_for_test(fast_backoff_knobs(tv.client()));
    push_faults(
        tv.client(),
        2,
        ErrorKind::Unknown,
        "TRYAGAIN Multiple keys request during rehashing of slot",
    );
    let c = consume(&tv, 2).await;
    await_faults_consumed(tv.client()).await;
    xadd_update(&tv, 2, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
}

/// A router miss and every redirect-budget-exhausted shape on the reader's
/// XREAD — what a resharding or a router rebuild surfaces to a
/// `max_attempts = 1` read — are ridden out: delivery resumes from the
/// durable cursor, nothing ends, nothing counts as permanent, and nothing
/// forces a reconnect (fred is already healing underneath; b.c7a, b.cv4).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_router_miss_or_a_spent_redirect_budget_on_the_xread_is_ridden_out_without_a_reconnect() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client()
        .stream_mux
        .set_knobs_for_test(fast_backoff_knobs(tv.client()));
    let mut faults = vec![fault(ErrorKind::Unknown, "Failed to route command.")];
    faults.extend(
        BUDGET_EXHAUSTED_SHAPES
            .iter()
            .map(|(kind, message)| fault(kind.clone(), message)),
    );
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Xread, faults);

    let c = consume(&tv, 3).await;
    await_faults_consumed(tv.client()).await;
    xadd_update(&tv, 3, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    assert!(c.take_terminal_exit().is_none());
    assert_eq!(read_permanent_exits(&metrics, CONSUMER_PHASE_XREAD), 0);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_XREAD, 0),
        "a routing miss or a spent redirect budget is backoff-only"
    );
}

/// The store not serving — every `STORE_NOT_SERVING_SHAPES` row: `-LOADING`
/// while a node replays its data, `-BUSY` behind a long script — on the
/// reader's idle→active probe and then on its XREAD is ridden out: delivery
/// resumes from the durable cursor, nothing ends, nothing counts as
/// permanent, and nothing forces a reconnect (the connection is healthy and
/// the node answering; a reconnect would meet the same refusal; b.xd1).
/// Pre-b.xd1 the first one was a permanent failure that ended every
/// subscription on the reader.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_stores_not_serving_replies_on_the_probe_and_the_xread_are_ridden_out_without_a_reconnect(
) {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client()
        .stream_mux
        .set_knobs_for_test(fast_backoff_knobs(tv.client()));
    let not_serving = || {
        STORE_NOT_SERVING_SHAPES
            .iter()
            .map(|reply| fault(ErrorKind::Unknown, reply))
            .collect::<Vec<_>>()
    };
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Probe, not_serving());
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Xread, not_serving());

    let c = consume(&tv, 24).await;
    await_faults_consumed(tv.client()).await;
    assert_eq!(
        tv.client()
            .stream_mux
            .pending_faults_for_test(MuxFaultSite::Probe),
        0,
        "every probe fault was taken before the reads began"
    );
    xadd_update(&tv, 24, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    assert!(c.take_terminal_exit().is_none());
    assert_eq!(read_permanent_exits(&metrics, CONSUMER_PHASE_PROBE), 0);
    assert_eq!(read_permanent_exits(&metrics, CONSUMER_PHASE_XREAD), 0);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_XREAD, 0),
        "the store not serving is backoff-only"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_timeout_streak_forces_a_rate_limited_reconnect_and_keeps_subscriptions() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client()
        .stream_mux
        .set_knobs_for_test(fast_backoff_knobs(tv.client()));
    push_timeout_faults(tv.client(), 3);
    let c = consume(&tv, 7).await;
    await_faults_consumed(tv.client()).await;
    xadd_update(&tv, 7, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_XREAD, 1),
        "the first timeout of a streak forces a reconnect, under `xread`; the next ones are \
         rate-limited"
    );
    assert_eq!(
        command_timeout_row(&metrics),
        command_timeouts_only_on(POOL_LABEL_SUBSCRIBER, 3),
        "each reader timeout counts once, on the reader pool and no other"
    );
}

// ── `-UNBLOCKED`: the cheap allowance ────────────────────────────────────────

/// Past `MAX_CONSECUTIVE_UNBLOCK_RETRIES` (3) cheap re-issues, a further
/// `-UNBLOCKED` is charged to the transient budget like any retryable
/// failure. With a budget of 0 that ends the subscription.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unblock_storm_past_the_cheap_allowance_is_charged_to_the_budget() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        max_transient_failures: 0,
        ..fast_backoff_knobs(tv.client())
    });
    push_faults(tv.client(), 4, ErrorKind::Unknown, UNBLOCKED_REPLY);
    let c = consume(&tv, 20).await;
    let message = assert_ends_with(&c, STREAM_EXIT_ERROR).await;
    assert!(
        message.contains("gave up after 1 consecutive transient"),
        "the fourth `-UNBLOCKED` is the first charged failure; got {message:?}"
    );
    assert_eq!(read_unblocked_retries(&metrics), 3);
    assert_eq!(read_permanent_exits(&metrics, CONSUMER_PHASE_XREAD), 0);
}

/// The allowance is per burst: a successful read between two bursts of three
/// resets it, so neither burst is charged — with a budget of 0, one charge
/// would end the subscription.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_unblock_allowance_resets_after_a_successful_read() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        max_transient_failures: 0,
        // The `Pass` read between the bursts returns empty (a success) after
        // this short BLOCK.
        block_ms: 50,
        ..fast_backoff_knobs(tv.client())
    });
    let burst = || (0..3).map(|_| fault(ErrorKind::Unknown, UNBLOCKED_REPLY));
    let faults: Vec<MuxFault> = burst()
        .chain(std::iter::once(MuxFault::Pass))
        .chain(burst())
        .collect();
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Xread, faults);

    let c = consume(&tv, 21).await;
    await_faults_consumed(tv.client()).await;
    xadd_update(&tv, 21, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    assert!(
        c.take_terminal_exit().is_none(),
        "the second burst must not be charged"
    );
    assert_eq!(read_unblocked_retries(&metrics), 6);
}

// ── Budget exhaustion and permanent failures ─────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_exhausted_transient_budget_ends_every_subscription_on_the_reader() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        max_transient_failures: 3,
        ..fast_backoff_knobs(tv.client())
    });
    push_faults(tv.client(), 4, ErrorKind::IO, "synthetic outage");
    let a = consume(&tv, 3).await;
    let b = consume(&tv, 4).await;
    assert_ends_with(&a, STREAM_EXIT_ERROR).await;
    assert_ends_with(&b, STREAM_EXIT_ERROR).await;
    assert_eq!(
        read_permanent_exits(&metrics, CONSUMER_PHASE_XREAD),
        0,
        "budget exhaustion is inside the transient taxonomy"
    );

    let fresh = consume(&tv, 3).await;
    xadd_update(&tv, 3, 1).await;
    assert_eq!(
        recv_seqs(&fresh, 1).await,
        [1],
        "the reader serves new subscriptions"
    );
}

/// `CROSSSLOT` means the endpoint is a cluster addressed by a non-cluster URL:
/// no single key explains it, so every subscription ends without probing.
/// The key is a string here, so a `TYPE` probe would have blamed it — and
/// counted a poison key — had the reader probed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn crossslot_ends_every_subscription_without_probing_the_keys() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client()
        .stream_mux
        .set_knobs_for_test(fast_backoff_knobs(tv.client()));
    let _: () = tv
        .client()
        .pool
        .set(
            tv.key_schema().layer_stream(22),
            "not a stream",
            None,
            None,
            false,
        )
        .await
        .unwrap();
    push_faults(
        tv.client(),
        1,
        ErrorKind::Unknown,
        "CROSSSLOT Keys in request don't hash to the same slot",
    );
    let c = consume(&tv, 22).await;
    let message = assert_ends_with(&c, STREAM_EXIT_ERROR).await;
    assert!(message.contains("CROSSSLOT"), "{message}");
    assert_eq!(read_permanent_exits(&metrics, CONSUMER_PHASE_XREAD), 1);
    assert_eq!(
        read_series(&metrics, "live_edit_stream_mux_poison_keys_total", &[]),
        0,
        "CROSSSLOT must not probe the reader's keys"
    );
}

// ── Whole-reader failures heal and back off ──────────────────────────────────

/// A whole-reader failure — one that ended every subscription on the reader
/// with a retryable `Unavailable`, and counts one permanent exit under
/// `phase="xread"` — force-reconnects the reader's connection under
/// `phase="permanent"` (and no other phase) and backs off before the next
/// read, one backoff window of the first failure. While it backs off, a new
/// subscription waits: no read is issued for it.
///
/// Current-thread, so the reader task's DEBUG backoff line is captured
/// (`ThreadCapture`). The backoff is an hour, so the observation window below
/// is inside it unless the full-jitter draw lands under it.
async fn a_whole_reader_failure_heals_and_backs_off(fault: MuxFault) {
    const HOUR: Duration = Duration::from_secs(3_600);
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        backoff_initial: HOUR,
        backoff_cap: HOUR,
        ..tv.client().stream_mux.knobs()
    });
    let capture = ThreadCapture::start_debug();
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Xread, [fault]);

    let c = consume(&tv, 30).await;
    assert_ends_with(&c, STREAM_EXIT_ERROR).await;
    assert_eq!(read_permanent_exits(&metrics, CONSUMER_PHASE_XREAD), 1);
    let line = await_lines(&capture, WHOLE_READER_BACKOFF_MSG, 1)
        .await
        .remove(0);
    let fields = &line["fields"];
    assert_eq!(fields["consecutive_failures"], 1, "{line}");
    assert_eq!(
        fields["backoff_window_ms"],
        HOUR.as_millis() as u64,
        "{line}"
    );
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_PERMANENT, 1)
    );

    let sleep_ms = fields["sleep_ms"].as_u64().expect("sleep_ms is a number");
    if sleep_ms > 4 * QUIET.as_millis() as u64 {
        let xreads = tv.client().stream_mux.snapshot().xreads();
        let fresh = consume(&tv, 31).await;
        xadd_update(&tv, 31, 1).await;
        assert_quiet(
            &fresh,
            QUIET,
            "no read is issued while the reader backs off",
        )
        .await;
        assert_eq!(tv.client().stream_mux.snapshot().xreads(), xreads);
    } else {
        eprintln!(
            "the full-jitter backoff drew {sleep_ms} ms, inside the observation window; the \
             no-read check is skipped this run"
        );
    }
    drop(capture);
    purge_keys(&tv).await;
}

#[tokio::test]
async fn a_permanent_xread_error_heals_the_reader_and_backs_off() {
    a_whole_reader_failure_heals_and_backs_off(fault(
        ErrorKind::Auth,
        "synthetic permanent failure",
    ))
    .await;
}

#[tokio::test]
async fn crossslot_heals_the_reader_and_backs_off() {
    a_whole_reader_failure_heals_and_backs_off(fault(
        ErrorKind::Unknown,
        "CROSSSLOT Keys in request don't hash to the same slot",
    ))
    .await;
}

#[tokio::test]
async fn an_unconvertible_reply_heals_the_reader_and_backs_off() {
    a_whole_reader_failure_heals_and_backs_off(MuxFault::Reply(Value::String(
        "not a reply".into(),
    )))
    .await;
}

/// Only a successful read resets the budget, so repeated whole-reader failures
/// back off exponentially: the second one's window is twice the first.
#[tokio::test]
async fn repeated_whole_reader_failures_back_off_exponentially() {
    const INITIAL: Duration = Duration::from_millis(20);
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        backoff_initial: INITIAL,
        backoff_cap: Duration::from_secs(5),
        ..tv.client().stream_mux.knobs()
    });
    let capture = ThreadCapture::start_debug();
    push_faults(
        tv.client(),
        2,
        ErrorKind::Auth,
        "synthetic permanent failure",
    );

    let first = consume(&tv, 32).await;
    assert_ends_with(&first, STREAM_EXIT_ERROR).await;
    let second = consume(&tv, 33).await;
    assert_ends_with(&second, STREAM_EXIT_ERROR).await;

    let lines = await_lines(&capture, WHOLE_READER_BACKOFF_MSG, 2).await;
    let windows: Vec<(u64, u64)> = lines
        .iter()
        .map(|l| {
            (
                l["fields"]["consecutive_failures"].as_u64().unwrap(),
                l["fields"]["backoff_window_ms"].as_u64().unwrap(),
            )
        })
        .collect();
    let initial = INITIAL.as_millis() as u64;
    assert_eq!(windows, [(1, initial), (2, 2 * initial)]);
    drop(capture);
    purge_keys(&tv).await;
}

/// A whole-reader failure is charged to the same budget as transient ones: a
/// transient failure right after one is the budget's second, and a budget of
/// 1 ends the subscription it hits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_whole_reader_failure_is_charged_to_the_transient_budget() {
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        max_transient_failures: 1,
        ..fast_backoff_knobs(tv.client())
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [
            fault(ErrorKind::Auth, "synthetic permanent failure"),
            fault(ErrorKind::IO, "synthetic connection reset"),
        ],
    );
    let first = consume(&tv, 34).await;
    assert_ends_with(&first, STREAM_EXIT_ERROR).await;
    let second = consume(&tv, 35).await;
    let message = assert_ends_with(&second, STREAM_EXIT_ERROR).await;
    assert!(
        message.contains("gave up after 2 consecutive transient XREAD failures"),
        "the transient failure is the budget's second; got {message:?}"
    );
}

// ── Poison-key isolation against a degraded backing store ────────────────────

/// Keys the isolation tests subscribe to on their one reader.
const ISOLATION_KEYS: u64 = 8;

/// The data pool's stalled connections. Released on drop too, so a failing
/// assertion does not leave `TestValkey`'s cleanup — which runs on the data
/// pool — queued behind the stalls until their 60 s `BLPOP` backstop.
struct DataPoolStalls {
    stalls: Option<Vec<ConnStall>>,
    observer: fred::clients::Pool,
    keys: Vec<String>,
}

impl DataPoolStalls {
    async fn stall(tv: &TestValkey) -> Self {
        let stalls = tv.stall_data_pool().await;
        // `TestValkey::stall_data_pool`'s per-connection keys.
        let keys = (0..stalls.len())
            .map(|i| format!("{}conn-stall:data:{i}", tv.prefix()))
            .collect();
        Self {
            stalls: Some(stalls),
            observer: tv.client().maintenance_pool.clone(),
            keys,
        }
    }

    /// Releases every stall, asserting each held until now.
    async fn release(mut self) {
        release_all(self.stalls.take().expect("released once")).await;
    }
}

impl Drop for DataPoolStalls {
    fn drop(&mut self) {
        use fred::interfaces::ListInterface;
        if self.stalls.take().is_none() {
            return;
        }
        // Only reached when the test failed first: release without asserting,
        // since a second panic while unwinding would abort the run.
        let (observer, keys) = (self.observer.clone(), std::mem::take(&mut self.keys));
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                for key in keys {
                    let _: Result<i64, _> = observer.lpush(key, "release").await;
                }
            });
        });
    }
}

/// Subscribes [`ISOLATION_KEYS`] streams on one reader cycling a 100 ms BLOCK,
/// stalls every data-pool connection (where the `TYPE` probes go), then turns
/// the first key into a string through the maintenance pool so the next real
/// XREAD fails `WRONGTYPE`. Returns once that failure has been counted, i.e.
/// while the probes are running against the stalled pool.
async fn stall_the_probes_of_a_permanent_failure(
    tv: &TestValkey,
) -> (
    Vec<LayerStreamConsumer>,
    DataPoolStalls,
    std::sync::Arc<live_edit::observability::metrics::Metrics>,
) {
    let metrics = install_metrics(tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: 100,
        // A wake still in flight when the pool stalls must not be scored as a
        // wedge.
        wake_ack_timeout: Duration::from_secs(60),
        ..fast_backoff_knobs(tv.client())
    });
    let mut subs = Vec::new();
    for oid in 1..=ISOLATION_KEYS {
        subs.push(consume(tv, 40 + oid).await);
    }
    await_read_over(tv, ISOLATION_KEYS as usize).await;
    let stalls = DataPoolStalls::stall(tv).await;
    let _: () = tv
        .client()
        .maintenance_pool
        .set(
            tv.key_schema().layer_stream(41),
            "not a stream",
            None,
            None,
            false,
        )
        .await
        .unwrap();
    await_series(
        &metrics,
        "live_edit_valkey_consumer_permanent_exits_total",
        &[("phase", CONSUMER_PHASE_XREAD)],
        1,
    )
    .await;
    (subs, stalls, metrics)
}

/// With every probe stuck behind a stalled connection, isolation finishes
/// within ONE probe timeout (2 s) however many keys the reader holds — the
/// probes run concurrently under one deadline — and a probe that does not
/// answer marks no key poisoned, so the failure is the whole reader's and every
/// subscription ends. Sequential probes would take 16 s and outlast `SETTLE`.
///
/// A subscription registered while the probes run is queued, not lost, and
/// not ended by an isolation it was not part of: it is served once the reader
/// resumes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stalled_probes_end_the_whole_reader_within_one_timeout_and_keep_a_queued_joiner() {
    let tv = TestValkey::setup_with_readers(1).await;
    let (subs, stalls, metrics) = stall_the_probes_of_a_permanent_failure(&tv).await;
    let joiner = consume(&tv, 99).await;

    for sub in &subs {
        let message = assert_ends_with(sub, STREAM_EXIT_ERROR).await;
        assert!(
            message.contains("permanent XREAD error"),
            "an unanswered probe blames no key, so the whole reader fails; got {message:?}"
        );
    }
    assert_eq!(
        read_series(&metrics, "live_edit_stream_mux_poison_keys_total", &[]),
        0,
        "a probe that did not answer must not mark its key poisoned"
    );

    stalls.release().await;
    xadd_update(&tv, 99, 1).await;
    assert_eq!(recv_seqs(&joiner, 1).await, [1]);
    assert!(joiner.take_terminal_exit().is_none());
}

/// Shutdown during the stalled probes ends the reader at once — the probes
/// race the reader's cancellation — so every subscription ends `shutdown`, not
/// `error`, and the multiplexer stops inside its grace. A subscription whose
/// `Add` was still queued behind the cancel ends `shutdown` too: shutdown
/// applies what is queued before it ends the reader's subscriptions.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_during_stalled_probes_ends_every_subscription_and_a_queued_joiner_shutdown() {
    let tv = TestValkey::setup_with_readers(1).await;
    let (subs, stalls, _metrics) = stall_the_probes_of_a_permanent_failure(&tv).await;
    let joiner = consume(&tv, 99).await;

    let ((), captured) =
        capture_debug(tv.client().stream_mux.shutdown(Duration::from_secs(1))).await;
    assert!(
        captured
            .with_message("stream multiplexer readers did not stop within the shutdown grace")
            .is_empty(),
        "the reader must stop inside the grace, not after its 2 s probes; full capture:\n{}",
        captured.text
    );
    for sub in &subs {
        assert_ends_with(sub, STREAM_EXIT_SHUTDOWN).await;
    }
    assert_ends_with(&joiner, STREAM_EXIT_SHUTDOWN).await;
    stalls.release().await;
}

// ── Shutdown while backing off ───────────────────────────────────────────────

/// `StreamMux::shutdown` reaches a reader that is sleeping out a transient
/// backoff (a minute here): the reader stops inside the 1 s grace and its
/// subscription ends `shutdown`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_while_a_reader_backs_off_ends_its_subscriptions_shutdown_inside_the_grace() {
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        backoff_initial: Duration::from_secs(60),
        backoff_cap: Duration::from_secs(60),
        ..tv.client().stream_mux.knobs()
    });
    push_faults(tv.client(), 1, ErrorKind::IO, "synthetic connection reset");
    let c = consume(&tv, 50).await;
    await_faults_consumed(tv.client()).await;

    let ((), captured) =
        capture_debug(tv.client().stream_mux.shutdown(Duration::from_secs(1))).await;
    assert!(
        captured
            .with_message("stream multiplexer readers did not stop within the shutdown grace")
            .is_empty(),
        "a reader in backoff must observe the shutdown; full capture:\n{}",
        captured.text
    );
    assert_eq!(
        captured
            .with_message("woke blocked stream reads before the reader pool QUIT")
            .len(),
        1,
        "the readers stopped inside the grace; full capture:\n{}",
        captured.text
    );
    assert_ends_with(&c, STREAM_EXIT_SHUTDOWN).await;
}

// ── Wakes ────────────────────────────────────────────────────────────────────

/// A read that does not return within the wake-ack timeout of its wake
/// LANDING is a wedge: the reader force-reconnects under `phase="wake_ack"`,
/// the hung read ends on that reconnect (`MuxFault::Hang`), and the retried
/// read delivers for both subscriptions.
///
/// The verdict also counts toward the timeout streak: the `Timeout` right
/// after it is the streak's second, which the rate limit lets pass without a
/// further reconnect. Had the wake-ack verdict not counted, that `Timeout`
/// would be a first and force an `xread` reconnect.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_read_that_outlives_its_landed_wake_is_reconnected_under_wake_ack_and_then_delivers() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        wake_ack_timeout: Duration::from_millis(200),
        ..fast_backoff_knobs(tv.client())
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [
            MuxFault::Hang,
            fault(ErrorKind::Timeout, "synthetic wedged-connection timeout"),
        ],
    );

    let a = consume(&tv, 60).await;
    await_mux(tv.client(), "the hung read is in flight", |s| {
        s.xreads() >= 1
    })
    .await;
    // A new key: its join wakes the hung read, and the wake lands.
    let b = consume(&tv, 61).await;
    await_faults_consumed(tv.client()).await;

    xadd_update(&tv, 60, 1).await;
    xadd_update(&tv, 61, 1).await;
    assert_eq!(recv_seqs(&a, 1).await, [1]);
    assert_eq!(recv_seqs(&b, 1).await, [1]);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_WAKE_ACK, 1)
    );
    assert!(read_wakes(&metrics, WAKE_OUTCOME_OK) >= 1);
    assert!(a.take_terminal_exit().is_none() && b.take_terminal_exit().is_none());
    assert_eq!(read_permanent_exits(&metrics, CONSUMER_PHASE_XREAD), 0);
}

/// The operator-visible side of the wake-ack verdict above: one WARN, naming
/// the reader and the wake-ack timeout it outlived. Current-thread, so
/// [`ThreadCapture`] sees the reader task's line.
#[tokio::test]
async fn a_wake_ack_verdict_logs_one_warn_naming_the_reader_and_its_timeout() {
    const WAKE_ACK_TIMEOUT: Duration = Duration::from_millis(200);
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        wake_ack_timeout: WAKE_ACK_TIMEOUT,
        ..fast_backoff_knobs(tv.client())
    });
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Xread, [MuxFault::Hang]);
    let capture = ThreadCapture::start_debug();

    let _a = consume(&tv, 62).await;
    await_mux(tv.client(), "the hung read is in flight", |s| {
        s.xreads() >= 1
    })
    .await;
    // A new key: its join wakes the hung read, and the wake lands.
    let _b = consume(&tv, 63).await;
    await_series(
        &metrics,
        "live_edit_valkey_consumer_forced_reconnects_total",
        &[("phase", CONSUMER_PHASE_WAKE_ACK)],
        1,
    )
    .await;

    let captured = capture.snapshot();
    let line = captured.only(WAKE_ACK_WEDGE_MSG);
    assert_eq!(line["level"], "WARN", "{line}");
    assert_field(line, "reader", 0.into());
    assert_field(
        line,
        "timeout_ms",
        (WAKE_ACK_TIMEOUT.as_millis() as u64).into(),
    );
    drop(capture);
    purge_keys(&tv).await;
}

/// A wake whose every attempt fails counts once under
/// `wakes_total{outcome="error"}`, and — never having landed — arms no
/// wake-ack deadline: the read it was meant to wake stays in flight and no
/// reconnect is forced, however long the (short) wake-ack timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_wake_counts_one_error_and_arms_no_wake_ack_deadline() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        wake_ack_timeout: Duration::from_millis(50),
        ..fast_backoff_knobs(tv.client())
    });
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Xread, [MuxFault::Hang]);
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Wake,
        (0..3).map(|_| fault(ErrorKind::IO, "synthetic wake failure")),
    );

    let _a = consume(&tv, 70).await;
    await_mux(tv.client(), "the hung read is in flight", |s| {
        s.xreads() >= 1
    })
    .await;
    let _b = consume(&tv, 71).await;
    await_series(
        &metrics,
        "live_edit_stream_mux_wakes_total",
        &[("outcome", WAKE_OUTCOME_ERROR)],
        1,
    )
    .await;
    assert_eq!(
        tv.client()
            .stream_mux
            .pending_faults_for_test(MuxFaultSite::Wake),
        0,
        "one wake made all three attempts"
    );

    tokio::time::sleep(QUIET).await;
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_WAKE_ACK, 0),
        "a wake that never landed draws no wedge verdict"
    );
    assert_eq!(
        tv.client().stream_mux.snapshot().xreads(),
        1,
        "the hung read is still the read in flight"
    );
    assert_eq!(read_wakes(&metrics, WAKE_OUTCOME_ERROR), 1);
    assert_eq!(read_wakes(&metrics, WAKE_OUTCOME_OK), 0);
}

/// The operator-visible side of a failed coverage wake (a new key joining the
/// in-flight read): one WARN, with the coverage wake's text and
/// `reason="coverage"`, naming the wake key, the last attempt's error, and the
/// XREAD block the new subscription may wait out (`block_ms`, the reader's own
/// setting — a distinctive value here, so the field cannot match by
/// accident). Current-thread, so [`ThreadCapture`] sees the spawned wake
/// task's line.
#[tokio::test]
async fn a_failed_wake_logs_one_warn_with_its_key_error_and_block_ms() {
    const BLOCK_MS: u64 = 4_321;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: BLOCK_MS,
        ..fast_backoff_knobs(tv.client())
    });
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Xread, [MuxFault::Hang]);
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Wake,
        (0..3).map(|_| fault(ErrorKind::IO, "synthetic wake failure")),
    );
    let capture = ThreadCapture::start_debug();

    let _a = consume(&tv, 74).await;
    await_mux(tv.client(), "the hung read is in flight", |s| {
        s.xreads() >= 1
    })
    .await;
    let _b = consume(&tv, 75).await;
    await_series(
        &metrics,
        "live_edit_stream_mux_wakes_total",
        &[("outcome", WAKE_OUTCOME_ERROR)],
        1,
    )
    .await;
    // The WARN follows the count inside the wake task.
    await_lines(&capture, WAKE_FAILED_COVERAGE_MSG, 1).await;

    assert_one_wake_failed_warn(
        &capture.snapshot(),
        WAKE_FAILED_COVERAGE_MSG,
        "coverage",
        WAKE_FAILED_LIVENESS_MSG,
        BLOCK_MS,
        &tv,
    );
    drop(capture);
    purge_keys(&tv).await;
}

/// Asserts `captured` holds exactly one failed-wake WARN with `msg`, carrying
/// `reason`, the reader's `block_ms`, the injected wake error and a wake key
/// under `tv`'s prefix — and none with the other kind's text, `other_msg`.
fn assert_one_wake_failed_warn(
    captured: &Captured,
    msg: &str,
    reason: &str,
    other_msg: &str,
    block_ms: u64,
    tv: &TestValkey,
) {
    let line = captured.only(msg);
    assert_eq!(line["level"], "WARN", "{line}");
    assert_field(line, "reason", reason.into());
    assert_field(line, "block_ms", block_ms.into());
    let error = line["fields"]["error"].as_str().unwrap_or_default();
    assert!(
        error.contains("synthetic wake failure"),
        "the WARN carries the last attempt's error; {line}"
    );
    let wake_key = line["fields"]["wake_key"].as_str().unwrap_or_default();
    assert!(
        wake_key.starts_with(tv.prefix()) && wake_key.contains(":mux:wake:"),
        "the WARN names the reader's wake key; {line}"
    );
    assert!(
        captured.with_message(other_msg).is_empty(),
        "a {reason} wake's failure is not reported with the other kind's text; full \
         capture:\n{}",
        captured.text
    );
}

/// A failed liveness wake — sent for a join the in-flight read already
/// covers, because the reader's connection has not answered within the
/// liveness age — logs its own WARN: the liveness text with
/// `reason="liveness"`, the wake key, the last attempt's error and the
/// reader's `block_ms`, and counts once under `wakes_total{outcome="error"}`.
/// Its cost is what that text says: the covered subscription's delivery is
/// not delayed — an XADD on the layer reaches it at once, far inside the
/// minute-long BLOCK — and the wake, never having landed, draws no wake-ack
/// verdict.
///
/// The read is a real one held by the long BLOCK; the liveness age is waited
/// out (a timer) after the probe's answer, so the covered join is a liveness
/// one. Current-thread, so [`ThreadCapture`] sees the spawned wake task's line.
#[tokio::test]
async fn a_failed_liveness_wake_logs_its_own_warn_and_delays_no_delivery() {
    const BLOCK_MS: u64 = 60_000;
    const LIVENESS_AGE: Duration = Duration::from_millis(100);
    const OID: u64 = 76;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: BLOCK_MS,
        liveness_age: LIVENESS_AGE,
        // Short, so a deadline the failed wake wrongly armed would force a
        // reconnect inside the observation window below.
        wake_ack_timeout: Duration::from_millis(100),
        ..fast_backoff_knobs(tv.client())
    });
    let capture = ThreadCapture::start_debug();

    let a = consume(&tv, OID).await;
    await_mux(tv.client(), "the read is in flight", |s| s.xreads() >= 1).await;
    // The liveness age is a timer: let the probe's answer age past it.
    tokio::time::sleep(2 * LIVENESS_AGE).await;
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Wake,
        (0..3).map(|_| fault(ErrorKind::IO, "synthetic wake failure")),
    );
    // Covered: the same key, at the read's own cursor.
    let b = consume(&tv, OID).await;
    await_series(
        &metrics,
        "live_edit_stream_mux_wakes_total",
        &[("outcome", WAKE_OUTCOME_ERROR)],
        1,
    )
    .await;
    await_lines(&capture, WAKE_FAILED_LIVENESS_MSG, 1).await;

    tokio::time::sleep(QUIET).await;
    xadd_update(&tv, OID, 1).await;
    assert_eq!(recv_seqs(&a, 1).await, [1]);
    assert_eq!(
        recv_seqs(&b, 1).await,
        [1],
        "the covered subscription's delivery is not delayed by the failed wake"
    );
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_WAKE_ACK, 0),
        "a wake that never landed draws no wedge verdict"
    );
    assert_eq!(read_wakes(&metrics, WAKE_OUTCOME_ERROR), 1);
    assert_eq!(read_wakes(&metrics, WAKE_OUTCOME_OK), 0);

    assert_one_wake_failed_warn(
        &capture.snapshot(),
        WAKE_FAILED_LIVENESS_MSG,
        "liveness",
        WAKE_FAILED_COVERAGE_MSG,
        BLOCK_MS,
        &tv,
    );
    drop(capture);
    purge_keys(&tv).await;
}

/// After a failed wake, the next subscription that joins the same read sends
/// a fresh wake; a removal does not (nor does the failure report itself), so a
/// failed wake is retried by joins, never in a loop.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn after_a_failed_wake_the_next_join_sends_a_fresh_wake_and_a_removal_does_not() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        // The fresh wake lands on a read that never returns; keep its wake-ack
        // verdict out of this test.
        wake_ack_timeout: Duration::from_secs(60),
        ..fast_backoff_knobs(tv.client())
    });
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Xread, [MuxFault::Hang]);
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Wake,
        (0..3).map(|_| fault(ErrorKind::IO, "synthetic wake failure")),
    );
    let wakes = |tv: &TestValkey| tv.client().stream_mux.snapshot().readers[0].wakes;

    let a = consume(&tv, 80).await;
    await_mux(tv.client(), "the hung read is in flight", |s| {
        s.xreads() >= 1
    })
    .await;
    let _b = consume(&tv, 81).await;
    await_series(&metrics, STREAMS, &[], 2).await;
    await_series(
        &metrics,
        "live_edit_stream_mux_wakes_total",
        &[("outcome", WAKE_OUTCOME_ERROR)],
        1,
    )
    .await;
    assert_eq!(wakes(&tv), 1);

    // A removal: `b` is still uncovered by the read in flight, yet no wake.
    drop(a);
    await_series(&metrics, STREAMS, &[], 1).await;
    assert_eq!(wakes(&tv), 1, "a removal must not send a wake");

    // A join: a fresh wake, which lands (the fault queue is spent).
    let _c = consume(&tv, 82).await;
    await_series(
        &metrics,
        "live_edit_stream_mux_wakes_total",
        &[("outcome", WAKE_OUTCOME_OK)],
        1,
    )
    .await;
    assert_eq!(wakes(&tv), 2);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_WAKE_ACK, 0)
    );
}

/// A wake that hangs (`MuxFault::Hang` at `MuxFaultSite::Wake`) never ends on
/// its own: it reports neither outcome, arms no wake-ack deadline, and — still
/// outstanding — suppresses a second wake for a later join. The read it meant
/// to wake picks the new subscriptions up when it next returns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_hung_wake_never_resolves_and_the_read_picks_its_joiners_up_when_it_returns() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        wake_ack_timeout: Duration::from_millis(50),
        ..tv.client().stream_mux.knobs()
    });
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Wake, [MuxFault::Hang]);

    let a = consume(&tv, 90).await;
    await_mux(tv.client(), "a read is in flight", |s| s.xreads() >= 1).await;
    let b = consume(&tv, 91).await;
    let c = consume(&tv, 92).await;
    await_series(&metrics, STREAMS, &[], 3).await;
    assert_eq!(
        tv.client().stream_mux.snapshot().readers[0].wakes,
        1,
        "the hung wake is still outstanding, so the second join sends none"
    );

    tokio::time::sleep(QUIET).await;
    assert_eq!(read_wakes(&metrics, WAKE_OUTCOME_OK), 0);
    assert_eq!(read_wakes(&metrics, WAKE_OUTCOME_ERROR), 0);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_WAKE_ACK, 0)
    );

    // An entry on the read's own key makes it return; the next read covers
    // the joiners.
    xadd_update(&tv, 90, 1).await;
    assert_eq!(recv_seqs(&a, 1).await, [1]);
    xadd_update(&tv, 91, 1).await;
    xadd_update(&tv, 92, 1).await;
    assert_eq!(recv_seqs(&b, 1).await, [1]);
    assert_eq!(recv_seqs(&c, 1).await, [1]);
}

// ── Panics ───────────────────────────────────────────────────────────────────

/// A reader panic drops its subscriptions' sinks without a recorded reason:
/// each ends `UNAVAILABLE` with the one "no recorded reason" message, the
/// reader's stream and in-use gauges return to zero, and the restarted reader
/// serves new subscriptions.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reader_panic_ends_its_subscriptions_unavailable_and_the_reader_recovers() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Xread, [MuxFault::Panic]);
    let c = consume(&tv, 8).await;
    let message = assert_ends_with(&c, STREAM_EXIT_ERROR).await;
    assert_eq!(message, READER_ENDED_WITHOUT_REASON_MSG);
    await_series(&metrics, STREAMS, &[], 0).await;
    await_series(
        &metrics,
        "live_edit_valkey_pool_in_use",
        &[("pool", POOL_LABEL_SUBSCRIBER)],
        0,
    )
    .await;

    let fresh = consume(&tv, 8).await;
    xadd_update(&tv, 8, 1).await;
    assert_eq!(recv_seqs(&fresh, 1).await, [1]);
}

// ── Slow subscribers ─────────────────────────────────────────────────────────
//
// The soft watchdog (`channel_capacity` entries or `backlog_bytes` bytes held
// past `send_watchdog`) applies only once a consumer is draining — from its
// first `recv` or `try_recv`, whatever that returns. The hard caps
// (`backlog_hard_cap` entries, `backlog_hard_bytes` bytes) apply from
// registration. A test of the soft rule therefore polls its subject once
// (an empty `try_recv`) before the writes.

const OVERFLOW_KILLS: &str = "live_edit_stream_mux_overflow_kills_total";

/// `live_edit_stream_mux_overflow_kills_total{bound}`, both cells, as
/// `(entries, bytes)`.
fn overflow_kills(metrics: &live_edit::observability::metrics::Metrics) -> (i64, i64) {
    (
        read_series(metrics, OVERFLOW_KILLS, &[("bound", "entries")]),
        read_series(metrics, OVERFLOW_KILLS, &[("bound", "bytes")]),
    )
}

/// The `ompf` bytes of the entries the byte-cap tests write: large enough that
/// an entry's size is dominated by it.
const OMPF_BYTES: usize = 1_000;

/// XADDs an `update` entry `{seq}-0` carrying an `ompf` field of `ompf_bytes`.
async fn xadd_sized(tv: &TestValkey, object_id: u64, seq: u64, ompf_bytes: usize) {
    use fred::interfaces::StreamsInterface;
    let _: String = tv
        .client()
        .pool
        .xadd(
            tv.key_schema().layer_stream(object_id),
            false,
            None,
            format!("{seq}-0"),
            vec![
                (
                    "payload",
                    Value::from(format!("update|{object_id}|{seq}|probe|0|0")),
                ),
                ("ompf", Value::Bytes(vec![0xa5; ompf_bytes].into())),
            ],
        )
        .await
        .expect("XADD must succeed");
}

/// The size the reader measures for [`xadd_sized`]'s entry: its id plus its
/// `payload` and `ompf` values.
fn sized_entry_bytes(object_id: u64, seq: u64, ompf_bytes: usize) -> u64 {
    (format!("{seq}-0").len() + format!("update|{object_id}|{seq}|probe|0|0").len() + ompf_bytes)
        as u64
}

/// Marks `consumer` draining with a poll that finds nothing queued.
async fn start_draining(consumer: &LayerStreamConsumer) {
    assert!(
        consumer.try_recv().await.is_none(),
        "fixture self-check: nothing is queued yet"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_subscriber_stuck_past_the_watchdog_ends_while_its_sibling_keeps_receiving() {
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        channel_capacity: 4,
        backlog_hard_cap: 1_000,
        send_watchdog: Duration::from_millis(200),
        ..tv.client().stream_mux.knobs()
    });
    let stuck = consume(&tv, 9).await;
    // Draining, so the soft watchdog applies to it.
    start_draining(&stuck).await;
    let sibling = consume(&tv, 9).await;
    for seq in 1..=6 {
        xadd_update(&tv, 9, seq).await;
    }
    assert_eq!(recv_seqs(&sibling, 6).await, [1, 2, 3, 4, 5, 6]);
    // The watchdog is a timer: let it run out for the stuck backlog.
    tokio::time::sleep(Duration::from_millis(300)).await;
    xadd_update(&tv, 9, 7).await;
    assert_eq!(recv_seqs(&sibling, 1).await, [7]);

    assert_eq!(
        recv_seqs(&stuck, 6).await,
        [1, 2, 3, 4, 5, 6],
        "entries accepted inside the watchdog window drain before the ending"
    );
    assert_ends_with(&stuck, STREAM_EXIT_INNER_WATCHDOG).await;
}

/// The entry hard cap applies from registration: `stuck` is never polled
/// before the reader offers past it, and is ended at once, on `entries`, with
/// what it held delivered first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backlog_past_the_hard_cap_ends_the_subscription_immediately() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        channel_capacity: 2,
        backlog_hard_cap: 3,
        send_watchdog: Duration::from_secs(60),
        ..tv.client().stream_mux.knobs()
    });
    let stuck = consume(&tv, 10).await;
    for seq in 1..=5 {
        xadd_update(&tv, 10, seq).await;
    }
    // Nothing is drained until the reader has offered past the hard cap.
    await_series(&metrics, OVERFLOW_KILLS, &[("bound", "entries")], 1).await;
    assert_eq!(
        overflow_kills(&metrics),
        (1, 0),
        "an entry-count ending is counted on `entries` only"
    );
    assert_eq!(recv_seqs(&stuck, 3).await, [1, 2, 3]);
    assert_ends_with(&stuck, STREAM_EXIT_INNER_WATCHDOG).await;
}

/// The byte hard cap applies from registration too: a never-polled subscriber
/// holding exactly `backlog_hard_bytes` is ended at once by the next entry, on
/// `bytes` — far below its entry caps — with what it held delivered first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backlog_past_the_hard_byte_cap_ends_the_subscription_immediately_on_bytes() {
    const OID: u64 = 13;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    let entry = sized_entry_bytes(OID, 1, OMPF_BYTES);
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        channel_capacity: 1_000,
        backlog_hard_cap: 10_000,
        backlog_bytes: 1 << 30,
        backlog_hard_bytes: 3 * entry,
        send_watchdog: Duration::from_secs(60),
        ..tv.client().stream_mux.knobs()
    });
    let stuck = consume(&tv, OID).await;
    for seq in 1..=5 {
        xadd_sized(&tv, OID, seq, OMPF_BYTES).await;
    }
    await_series(&metrics, OVERFLOW_KILLS, &[("bound", "bytes")], 1).await;
    assert_eq!(
        overflow_kills(&metrics),
        (0, 1),
        "a byte ending with the entry limits far away is counted on `bytes` only"
    );
    assert_eq!(
        recv_seqs(&stuck, 3).await,
        [1, 2, 3],
        "the queue holds exactly the hard byte cap"
    );
    let message = assert_ends_with(&stuck, STREAM_EXIT_INNER_WATCHDOG).await;
    assert!(
        message.contains("fell behind its delivery watchdog"),
        "{message}"
    );
}

/// A draining subscriber whose undelivered bytes stay at or above
/// `backlog_bytes` past the watchdog is ended on `bytes`, while its entry
/// count is far below `channel_capacity`; its sibling keeps receiving.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_byte_backlog_held_past_the_watchdog_ends_the_subscription_on_bytes() {
    const OID: u64 = 14;
    const WATCHDOG: Duration = Duration::from_millis(300);
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    let entry = sized_entry_bytes(OID, 1, OMPF_BYTES);
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        channel_capacity: 1_000,
        backlog_hard_cap: 10_000,
        backlog_bytes: 3 * entry,
        backlog_hard_bytes: 1 << 30,
        send_watchdog: WATCHDOG,
        ..tv.client().stream_mux.knobs()
    });
    let stuck = consume(&tv, OID).await;
    start_draining(&stuck).await;
    let sibling = consume(&tv, OID).await;
    for seq in 1..=6 {
        xadd_sized(&tv, OID, seq, OMPF_BYTES).await;
    }
    assert_eq!(recv_seqs(&sibling, 6).await, [1, 2, 3, 4, 5, 6]);
    // The watchdog is a timer: let it run out for the stuck byte backlog.
    tokio::time::sleep(2 * WATCHDOG).await;
    xadd_sized(&tv, OID, 7, OMPF_BYTES).await;
    assert_eq!(recv_seqs(&sibling, 1).await, [7]);

    await_series(&metrics, OVERFLOW_KILLS, &[("bound", "bytes")], 1).await;
    assert_eq!(overflow_kills(&metrics), (0, 1));
    assert_eq!(
        recv_seqs(&stuck, 6).await,
        [1, 2, 3, 4, 5, 6],
        "entries accepted inside the watchdog window drain before the ending"
    );
    assert_ends_with(&stuck, STREAM_EXIT_INNER_WATCHDOG).await;
    assert!(sibling.take_terminal_exit().is_none());
}

/// An empty queue always accepts one entry, so a single entry larger than the
/// whole hard byte cap never ends a subscriber that keeps up; a subscriber
/// still holding one is ended on `bytes` by the next.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_oversized_entry_into_an_empty_queue_never_ends_its_subscriber() {
    const OID: u64 = 15;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        channel_capacity: 1_000,
        backlog_hard_cap: 10_000,
        backlog_bytes: 1 << 30,
        backlog_hard_bytes: 64,
        send_watchdog: Duration::from_secs(60),
        ..tv.client().stream_mux.knobs()
    });
    assert!(
        sized_entry_bytes(OID, 1, OMPF_BYTES) > 64,
        "fixture self-check"
    );
    let keeps_up = consume(&tv, OID).await;
    let holds_one = consume(&tv, OID).await;

    xadd_sized(&tv, OID, 1, OMPF_BYTES).await;
    assert_eq!(
        recv_seqs(&keeps_up, 1).await,
        [1],
        "accepted into an empty queue"
    );
    // `keeps_up` drained its entry, so the next one is offered into an empty
    // queue again; `holds_one` still holds its first.
    xadd_sized(&tv, OID, 2, OMPF_BYTES).await;
    assert_eq!(recv_seqs(&keeps_up, 1).await, [2]);
    await_series(&metrics, OVERFLOW_KILLS, &[("bound", "bytes")], 1).await;
    xadd_sized(&tv, OID, 3, OMPF_BYTES).await;
    assert_eq!(recv_seqs(&keeps_up, 1).await, [3]);
    assert!(keeps_up.take_terminal_exit().is_none());

    assert_eq!(recv_seqs(&holds_one, 1).await, [1]);
    assert_ends_with(&holds_one, STREAM_EXIT_INNER_WATCHDOG).await;
    assert_eq!(overflow_kills(&metrics), (0, 1));
}

/// Before its first poll a subscriber is bounded by the hard caps only: a
/// backlog past both soft thresholds (entries and bytes) outlives the
/// watchdog, and every entry is delivered once it drains. (Its hard caps still
/// apply before the first poll:
/// `a_backlog_past_the_hard_cap_ends_the_subscription_immediately` and its
/// byte twin never poll their subject either.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn before_its_first_poll_a_backlog_past_the_soft_thresholds_outlives_the_watchdog() {
    const OID: u64 = 16;
    const WATCHDOG: Duration = Duration::from_millis(200);
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        channel_capacity: 2,
        backlog_hard_cap: 1_000,
        backlog_bytes: 64,
        backlog_hard_bytes: 1 << 30,
        send_watchdog: WATCHDOG,
        ..tv.client().stream_mux.knobs()
    });
    let unpolled = consume(&tv, OID).await;
    let sibling = consume(&tv, OID).await;
    for seq in 1..=4 {
        xadd_sized(&tv, OID, seq, OMPF_BYTES).await;
    }
    assert_eq!(recv_seqs(&sibling, 4).await, [1, 2, 3, 4]);
    // The watchdog is a timer: let it run out past both soft thresholds.
    tokio::time::sleep(2 * WATCHDOG).await;
    for seq in 5..=6 {
        xadd_sized(&tv, OID, seq, OMPF_BYTES).await;
    }
    assert_eq!(recv_seqs(&sibling, 2).await, [5, 6]);

    assert_eq!(
        recv_seqs(&unpolled, 6).await,
        [1, 2, 3, 4, 5, 6],
        "no soft watchdog before the first poll"
    );
    assert_quiet(&unpolled, Duration::from_millis(100), "no duplicates").await;
    assert!(unpolled.take_terminal_exit().is_none());
    assert_eq!(overflow_kills(&metrics), (0, 0));
}

/// A draining subscriber whose backlog goes past the soft capacity starts the
/// watchdog clock, and one that drains inside the watchdog window loses
/// nothing and is not ended.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stalled_subscriber_that_drains_in_time_loses_nothing() {
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        channel_capacity: 4,
        backlog_hard_cap: 1_000,
        send_watchdog: Duration::from_secs(5),
        ..tv.client().stream_mux.knobs()
    });
    let slow = consume(&tv, 11).await;
    // Draining, so its backlog over the capacity runs the soft clock.
    start_draining(&slow).await;
    let sibling = consume(&tv, 11).await;
    for seq in 1..=20 {
        xadd_update(&tv, 11, seq).await;
    }
    let expected: Vec<u64> = (1..=20).collect();
    // The sibling has all 20, so the reader has offered them to `slow` too:
    // its backlog went well past the capacity before it drains any.
    assert_eq!(recv_seqs(&sibling, 20).await, expected);
    assert_eq!(recv_seqs(&slow, 20).await, expected);
    assert_quiet(&slow, Duration::from_millis(100), "no duplicates").await;
    assert!(slow.take_terminal_exit().is_none());
}

/// A backlog that drains below the capacity clears the watchdog clock, so a
/// later backlog starts a fresh watchdog. Without the reset, the second
/// backlog would inherit the first one's start — long past the watchdog — and
/// be ended the moment it reached the capacity again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backlog_that_drains_restarts_the_watchdog_for_the_next_one() {
    const WATCHDOG: Duration = Duration::from_millis(300);
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        channel_capacity: 2,
        backlog_hard_cap: 1_000,
        send_watchdog: WATCHDOG,
        ..tv.client().stream_mux.knobs()
    });
    let slow = consume(&tv, 12).await;
    // Draining, so the soft watchdog applies to it: without this poll neither
    // backlog would start the clock and the test would pass vacuously.
    start_draining(&slow).await;
    let sibling = consume(&tv, 12).await;

    // First backlog: three entries against a capacity of two start the clock.
    for seq in 1..=3 {
        xadd_update(&tv, 12, seq).await;
    }
    assert_eq!(recv_seqs(&sibling, 3).await, [1, 2, 3]);
    assert_eq!(recv_seqs(&slow, 3).await, [1, 2, 3]);

    // The watchdog is a timer: let the first backlog's clock run past it.
    tokio::time::sleep(2 * WATCHDOG).await;

    // Second backlog, the same size: accepted, because the drain reset the
    // clock when the next entry was offered below the capacity.
    for seq in 4..=6 {
        xadd_update(&tv, 12, seq).await;
    }
    assert_eq!(recv_seqs(&sibling, 3).await, [4, 5, 6]);
    assert_eq!(recv_seqs(&slow, 3).await, [4, 5, 6]);
    assert!(slow.take_terminal_exit().is_none());
}

// ── Read sizing ──────────────────────────────────────────────────────────────

const REPLY_BYTES: &str = "live_edit_stream_mux_xread_reply_bytes";

/// A reply that carries only the reader's wake entry — what a join that needs
/// a wake produces — records no `live_edit_stream_mux_xread_reply_bytes`
/// sample, not a 0-byte one; the first reply carrying a layer entry does.
///
/// A second layer joins a parked reader with nothing written, so the read it
/// wakes returns the wake entry alone; the next read, naming both keys, shows
/// that reply was handled.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wake_only_reply_records_no_reply_bytes_sample() {
    use live_edit::observability::metrics::test_util::{render, sample_value};

    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    let replies =
        || sample_value(&render(&metrics), &format!("{REPLY_BYTES}_count"), &[]).unwrap_or(0.0);

    let first = consume(&tv, 17).await;
    await_readers_parked(tv.client()).await;
    let parked = tv.client().stream_mux.snapshot();
    assert_eq!((parked.readers[0].keys, parked.readers[0].wakes), (1, 0));

    let second = consume(&tv, 18).await;
    await_mux(
        tv.client(),
        "the wake-only reply is handled and re-planned",
        |s| s.readers[0].wakes == 1 && s.readers[0].keys == 2 && s.xreads() > parked.xreads(),
    )
    .await;
    await_series(
        &metrics,
        "live_edit_stream_mux_wakes_total",
        &[("outcome", WAKE_OUTCOME_OK)],
        1,
    )
    .await;
    assert_eq!(
        replies(),
        0.0,
        "a reply holding only the wake entry records no reply-bytes sample"
    );

    // Non-vacuity: a reply carrying a layer entry is recorded.
    xadd_update(&tv, 18, 1).await;
    assert_eq!(recv_seqs(&second, 1).await, [1]);
    let deadline = tokio::time::Instant::now() + SETTLE;
    while replies() != 1.0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the reply carrying a layer entry must be recorded; count {}",
            replies()
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    drop(first);
}

/// Receives `n` deliveries, recording after each the `COUNT` the reader put
/// on its latest XREAD.
async fn recv_seqs_sampling_count(
    tv: &TestValkey,
    consumer: &LayerStreamConsumer,
    n: usize,
    counts: &mut std::collections::BTreeSet<Option<u64>>,
) -> Vec<u64> {
    let mut seqs = Vec::with_capacity(n);
    for _ in 0..n {
        seqs.extend(recv_seqs(consumer, 1).await);
        counts.insert(tv.client().stream_mux.last_planned_count_for_test(0));
    }
    seqs
}

/// Reads are sized by bytes. A reader serving a layer of 256 KiB entries and a
/// layer of small ones, once it holds both keys' entry estimates, puts
/// `COUNT = ⌊1 MiB ÷ (256 KiB + a small entry)⌋ = 3` on every read — through
/// the live backlog and through the re-read a joiner behind the read cursor
/// forces — so no reply exceeds the 1 MiB target plus one entry per key
/// (`live_edit_stream_mux_xread_reply_bytes`, whose nearest bucket above that
/// is 2 MiB). Delivery stays exactly-once: the joiner gets all thirteen large
/// entries once, and the caught-up subscribers nothing twice.
///
/// Read with the ceiling of 512 instead, the joiner's re-read would return all
/// thirteen large entries (≈ 3.3 MiB) in one reply.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reads_are_sized_by_bytes_and_a_behind_cursor_joiner_gets_every_entry_exactly_once() {
    use live_edit::observability::metrics::test_util::{histogram_bucket, render, sample_value};

    const LARGE: u64 = 40;
    const SMALL: u64 = 41;
    const LARGE_OMPF: usize = 256 * 1024;
    const LARGE_N: u64 = 13;
    const SMALL_N: u64 = 21;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    let target = tv.client().stream_mux.knobs().reply_target_bytes;
    assert_eq!(
        target,
        1024 * 1024,
        "fixture self-check: the production target"
    );
    let large = consume(&tv, LARGE).await;
    let small = consume(&tv, SMALL).await;

    // One entry per key first, so the reader holds both keys' estimates before
    // any backlog exists (an unseen key is read at the 16 KiB default).
    xadd_sized(&tv, LARGE, 1, LARGE_OMPF).await;
    xadd_sized(&tv, SMALL, 1, 0).await;
    assert_eq!(recv_seqs(&large, 1).await, [1]);
    assert_eq!(recv_seqs(&small, 1).await, [1]);
    let sized_count =
        target / (sized_entry_bytes(LARGE, 1, LARGE_OMPF) + sized_entry_bytes(SMALL, 1, 0));
    assert_eq!(
        sized_count, 3,
        "fixture self-check: ⌊1 MiB ÷ (256 KiB + ≈ 60 B)⌋"
    );
    let deadline = tokio::time::Instant::now() + SETTLE;
    while tv.client().stream_mux.last_planned_count_for_test(0) != Some(sized_count) {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the read after both keys' first replies must be sized at {sized_count}; latest \
             COUNT {:?}",
            tv.client().stream_mux.last_planned_count_for_test(0)
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // The live backlog.
    let mut counts = std::collections::BTreeSet::new();
    for seq in 2..=LARGE_N {
        xadd_sized(&tv, LARGE, seq, LARGE_OMPF).await;
    }
    for seq in 2..=SMALL_N {
        xadd_sized(&tv, SMALL, seq, 0).await;
    }
    assert_eq!(
        recv_seqs_sampling_count(&tv, &large, (LARGE_N - 1) as usize, &mut counts).await,
        (2..=LARGE_N).collect::<Vec<_>>()
    );
    assert_eq!(
        recv_seqs_sampling_count(&tv, &small, (SMALL_N - 1) as usize, &mut counts).await,
        (2..=SMALL_N).collect::<Vec<_>>()
    );

    // A joiner behind the reader's read id on the large key forces a re-read
    // of the whole large stream.
    let joiner = consume(&tv, LARGE).await;
    assert_eq!(
        recv_seqs_sampling_count(&tv, &joiner, LARGE_N as usize, &mut counts).await,
        (1..=LARGE_N).collect::<Vec<_>>(),
        "the joiner gets every entry after its cursor, in order"
    );
    assert_eq!(
        counts,
        std::collections::BTreeSet::from([Some(sized_count)]),
        "every read after the estimates were known was sized by the rule"
    );
    assert_quiet(&joiner, QUIET, "the joiner gets no entry twice").await;
    assert_quiet(
        &large,
        QUIET,
        "the re-read is not re-delivered to the caught-up subscriber",
    )
    .await;
    assert_quiet(&small, QUIET, "the small layer gets no entry twice").await;

    let text = render(&metrics);
    let replies = sample_value(&text, &format!("{REPLY_BYTES}_count"), &[])
        .expect("every reply that carried layer entries is observed");
    let bucket = |le: &str| {
        histogram_bucket(&text, REPLY_BYTES, le, &[])
            .unwrap_or_else(|| panic!("{REPLY_BYTES} has an le=\"{le}\" bucket"))
    };
    assert_eq!(
        bucket("2097152"),
        replies,
        "no reply exceeds the 1 MiB target plus one entry per key (under 2 MiB)"
    );
    assert!(
        bucket("524288") < replies,
        "fixture self-check: some replies carried three large entries (≈ 768 KiB)"
    );
}
