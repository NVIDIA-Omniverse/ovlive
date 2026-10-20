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

//! A stream reader's checks of its own connection, driven by injected faults
//! at the reader's `Probe`, `Keepalive`, `Xread` and `Wake` sites and by the
//! test knobs (`reader_probe_timeout`, `keepalive_interval`, `liveness_age`,
//! `wake_ack_timeout`):
//!
//! - **The idle→active probe** — one `PING` before the first XREAD after the
//!   reader had nothing to read — at start, after it went idle, and for a
//!   join during the backoff after a failure that ended every subscription —
//!   and, on a cluster, before the first XREAD bound to a different hash slot
//!   than the reader's previous read, even with no idle moment between (a
//!   rebind while a read was in flight). Never for a read on the slot already
//!   bound, nor on standalone Valkey for a change of stream with a read in
//!   flight. A probe that does not answer, or that fred times out, force-reconnects
//!   under `phase="probe"` and loses nothing; a timeout and a transient error
//!   are each charged to the transient budget once, and past the budget end
//!   every subscription on the reader; a permanent error ends them at once, counts
//!   `permanent_exits{phase="probe"}` and heals the reader; an answered probe
//!   neither charges nor resets the budget.
//! - **The idle keepalive** — `PING`s only while the reader holds a
//!   connection and has nothing to read (a reservation registers nothing, so a
//!   connected reader holding only reservations is idle), never with a read in
//!   flight or a backoff pending, and never before the reader's first
//!   subscription: readers connect lazily (b.azh), and an unconnected reader
//!   has nothing to keep alive. An answered tick keeps the connection; a failed
//!   one counts `failed` and force-reconnects under `phase="keepalive"`. The
//!   budget is neither charged nor reset. A cluster reader pings its one
//!   connection, to its slot's owner; against a store that closes idle
//!   connections, the keepalive is what keeps an idle reader's connection
//!   open. The keepalive tests first connect the reader by serving one
//!   subscription ([`connect_the_reader_then_idle`]).
//! - **The liveness wake** — a join the in-flight read already covers wakes it
//!   when the reader's connection has not answered within `liveness_age`, or
//!   has failed since (an empty or delivered reply, a reply whose head checks
//!   all answered, and `-UNBLOCKED` are answers; an XREAD error and a failed
//!   head check void the last one), including a join made between reads, which
//!   is decided as the
//!   next read is issued. The wake's no-answer verdict shares the reader's
//!   timeout streak: it reconnects under `phase="wake_ack"` on a streak's
//!   first (and every 8th) verdict and is otherwise only a WARN.
//!
//! Compiled only under `test-fault-injection` (see `storage/mod.rs`). The
//! cluster keepalive and slot-rebind tests also self-gate on
//! `LIVE_EDIT_TEST_CLUSTER_URL`; they are not `#[ignore]`d, so CI's
//! `test-cluster:` job, which runs the whole suite with that variable and the
//! feature set, runs them. One test is `#[ignore]`d and self-gates on
//! `LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL`. Each prints a SKIP line when its
//! variable is unset (the disruptive one fails instead under CI's
//! `GITLAB_CI`).
//!
//! Waits are deadlines (`SETTLE`), not measurements. Three kinds of test sleep
//! on purpose: the liveness tests let a short `liveness_age` run out, because
//! that age is a timer; the negative checks — "no keepalive tick happened" —
//! hold for a bounded observation window ([`QUIET`]) of many keepalive
//! intervals, which can only let a regression through under extreme load,
//! never fail a correct build; and the disruptive test waits out three of the
//! store's idle timeouts before it claims a connection survived them. The
//! tests that read a reader task's log lines run on a current-thread runtime,
//! so [`ThreadCapture`] sees them.

use std::time::Duration;

use fred::clients::Client;
use fred::error::ErrorKind;
use fred::interfaces::{ClientInterface, ClientLike, ClusterInterface, ConfigInterface};
use fred::types::config::Server;

use live_edit::observability::metrics::{
    Metrics, CONSUMER_PHASE_KEEPALIVE, CONSUMER_PHASE_PERMANENT, CONSUMER_PHASE_PROBE,
    CONSUMER_PHASE_WAKE_ACK, CONSUMER_PHASE_XREAD, HEAD_CHECK_RESULT_NO_GAP,
    KEEPALIVE_RESULT_FAILED, KEEPALIVE_RESULT_OK, READER_CONNECT_CAUSE_RETARGET,
    READER_CONNECT_RESULT_FAILED, READER_CONNECT_RESULT_OK, STREAM_EXIT_ERROR, WAKE_OUTCOME_ERROR,
    WAKE_OUTCOME_OK,
};
use live_edit::storage::stream_mux::{MuxFault, MuxFaultSite, StreamMuxKnobs};
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::{
    consume_layer_stream, reserve_layer_stream, LayerStreamConsumer, ReservedStream, StreamCursor,
};

use super::mux_test_helpers::{
    assert_ends_with, assert_quiet, await_faults_consumed, await_lines, await_mux,
    await_readers_parked, await_series, consume, fast_backoff_knobs, fault, forced_reconnect_row,
    forced_reconnects_only_on, install_metrics, read_permanent_exits, read_series,
    read_unblocked_retries, read_wakes, recv_seqs, xadd_update, QUIET, SETTLE, STREAMS,
    UNBLOCKED_REPLY, WAKE_ACK_WEDGE_MSG, WHOLE_READER_BACKOFF_MSG,
};
use crate::grpc::log_capture::{purge_keys, ThreadCapture};

const KEEPALIVES: &str = "live_edit_stream_mux_keepalives_total";
const FORCED_RECONNECTS: &str = "live_edit_valkey_consumer_forced_reconnects_total";

/// The DEBUG line a reader logs for every wake it sends a joining
/// subscription, with `reason` `coverage` or `liveness`.
const WAKE_SENT_MSG: &str = "waking the in-flight multiplexed XREAD for a joining subscription";

/// The DEBUG line a reader logs when it backs off after a failed probe.
const PROBE_BACKOFF_MSG: &str = "backing off before re-probing the stream reader connection";

/// The shortened idle→active probe bound the probe tests run on.
const PROBE_TIMEOUT: Duration = Duration::from_millis(100);

/// The keepalive interval the keepalive tests run on.
const KEEPALIVE: Duration = Duration::from_millis(20);

/// A liveness age short enough to wait out.
const SHORT_LIVENESS_AGE: Duration = Duration::from_millis(100);

/// A liveness age a join made right after a read's issue cannot outlive.
const LIVENESS_AGE: Duration = Duration::from_secs(2);

/// The shortened wake-ack deadline the liveness tests run on.
const WAKE_ACK: Duration = Duration::from_millis(200);

fn keepalives(metrics: &Metrics, result: &str) -> i64 {
    read_series(metrics, KEEPALIVES, &[("result", result)])
}

/// Polls until `keepalives_total{result}` reads at least `want`.
async fn await_keepalives_at_least(metrics: &Metrics, result: &str, want: i64) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    loop {
        let got = keepalives(metrics, result);
        if got >= want {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{KEEPALIVES}{{result=\"{result}\"}} stayed at {got}, want at least {want}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// The multiplexer's current knobs with the idle keepalive every `interval`.
fn keepalive_every(tv: &TestValkey, interval: Duration) -> StreamMuxKnobs {
    StreamMuxKnobs {
        keepalive_interval: Some(interval),
        ..tv.client().stream_mux.knobs()
    }
}

/// The `reason` of every wake the reader sent a joining subscription, in
/// order.
fn wake_reasons(capture: &ThreadCapture) -> Vec<String> {
    capture
        .snapshot()
        .with_message(WAKE_SENT_MSG)
        .iter()
        .map(|line| {
            line["fields"]["reason"]
                .as_str()
                .unwrap_or("<missing>")
                .to_owned()
        })
        .collect()
}

async fn reserve(tv: &TestValkey, object_id: u64) -> ReservedStream {
    reserve_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        tokio::time::Instant::now() + SETTLE,
    )
    .await
    .expect("standalone routing always has a reader")
}

// ── The idle→active probe ───────────────────────────────────────────────────

/// A probe that times out (`fault`) reads the connection as wedged: one
/// forced reconnect under `phase="probe"` and no other phase. The reader then
/// backs off, probes again, and its reads deliver every entry written before
/// the subscription and after the reconnect, once each — the failed probe
/// lost nothing.
async fn a_timed_out_probe_is_reconnected_under_probe_and_then_loses_nothing(
    fault: MuxFault,
    object_id: u64,
) {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        reader_probe_timeout: PROBE_TIMEOUT,
        ..fast_backoff_knobs(tv.client())
    });
    for seq in 1..=3 {
        xadd_update(&tv, object_id, seq).await;
    }
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Probe, [fault]);

    let c = consume(&tv, object_id).await;
    await_series(
        &metrics,
        FORCED_RECONNECTS,
        &[("phase", CONSUMER_PHASE_PROBE)],
        1,
    )
    .await;
    xadd_update(&tv, object_id, 4).await;
    assert_eq!(recv_seqs(&c, 4).await, [1, 2, 3, 4]);
    assert_quiet(&c, QUIET, "no entry is delivered twice").await;

    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_PROBE, 1)
    );
    assert_eq!(
        tv.client()
            .stream_mux
            .pending_faults_for_test(MuxFaultSite::Probe),
        0,
        "the reader probed before it read"
    );
    assert_eq!(read_permanent_exits(&metrics, CONSUMER_PHASE_PROBE), 0);
    assert!(c.take_terminal_exit().is_none());
}

/// A probe that does not answer within `reader_probe_timeout`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_probe_that_does_not_answer_is_reconnected_under_probe_and_then_loses_nothing() {
    a_timed_out_probe_is_reconnected_under_probe_and_then_loses_nothing(MuxFault::Hang, 120).await;
}

/// A probe fred fails with its own `Timeout`: the same wedge verdict, not the
/// transient path's backoff-only one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_probe_fred_times_out_is_reconnected_under_probe_and_then_loses_nothing() {
    a_timed_out_probe_is_reconnected_under_probe_and_then_loses_nothing(
        fault(ErrorKind::Timeout, "synthetic probe timeout"),
        127,
    )
    .await;
}

/// A reader that went idle — served a subscription, lost it, and let its
/// last read return — probes again before the read for the next
/// subscription: the idle→active probe re-arms, it is not only the reader's
/// first. The keepalive ticking is the proof the reader went idle (it ticks
/// only with nothing to read and no read in flight); the short BLOCK lets the
/// last read return.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reader_that_went_idle_probes_again_before_its_next_read() {
    const FIRST: u64 = 128;
    const NEXT: u64 = 129;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: 100,
        keepalive_interval: Some(KEEPALIVE),
        reader_probe_timeout: PROBE_TIMEOUT,
        ..fast_backoff_knobs(tv.client())
    });
    let first = consume(&tv, FIRST).await;
    xadd_update(&tv, FIRST, 1).await;
    assert_eq!(recv_seqs(&first, 1).await, [1]);
    drop(first);
    let ticks = keepalives(&metrics, KEEPALIVE_RESULT_OK);
    await_keepalives_at_least(&metrics, KEEPALIVE_RESULT_OK, ticks + 1).await;

    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Probe, [MuxFault::Hang]);
    let next = consume(&tv, NEXT).await;
    await_series(
        &metrics,
        FORCED_RECONNECTS,
        &[("phase", CONSUMER_PHASE_PROBE)],
        1,
    )
    .await;
    xadd_update(&tv, NEXT, 1).await;
    assert_eq!(recv_seqs(&next, 1).await, [1]);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_PROBE, 1)
    );
}

/// A subscription that joins during the backoff after a failure that ended
/// every subscription on the reader is probed before the next read: the
/// reader re-arms the probe as soon as it has nothing to read, backoff or
/// not. Here a permanent XREAD error ends the first subscription, heals
/// (`phase="permanent"`) and backs off (up to 1 s); the joiner's probe then
/// hangs (`phase="probe"`), and the joiner is served.
///
/// The join lands in the backoff unless the full-jitter draw is shorter than
/// the step from the first subscription's ending to the join; either way it
/// is probed (the idle reader re-arms the probe too).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_join_during_the_backoff_after_a_whole_reader_failure_is_probed_before_the_next_read() {
    const FIRST: u64 = 134;
    const NEXT: u64 = 135;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        backoff_initial: Duration::from_secs(1),
        backoff_cap: Duration::from_secs(1),
        reader_probe_timeout: PROBE_TIMEOUT,
        ..tv.client().stream_mux.knobs()
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [fault(ErrorKind::Auth, "synthetic permanent failure")],
    );
    let first = consume(&tv, FIRST).await;
    assert_ends_with(&first, STREAM_EXIT_ERROR).await;

    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Probe, [MuxFault::Hang]);
    let next = consume(&tv, NEXT).await;
    await_series(
        &metrics,
        FORCED_RECONNECTS,
        &[("phase", CONSUMER_PHASE_PROBE)],
        1,
    )
    .await;
    xadd_update(&tv, NEXT, 1).await;
    assert_eq!(recv_seqs(&next, 1).await, [1]);
    let row: Vec<(&str, u64)> = forced_reconnect_row(&metrics)
        .into_iter()
        .filter(|(_, n)| *n > 0)
        .collect();
    assert_eq!(
        row,
        [(CONSUMER_PHASE_PROBE, 1), (CONSUMER_PHASE_PERMANENT, 1)],
        "one heal (permanent) and the joiner's probe, and no other phase"
    );
}

/// A transient probe error is one failure on the transient budget: the reader
/// backs off one window of the first failure (no forced reconnect — fred is
/// already healing underneath, as for a transient XREAD error), probes again,
/// and serves the subscription once the probe answers.
#[tokio::test]
async fn a_transient_probe_error_is_charged_once_backs_off_and_is_probed_again() {
    const OID: u64 = 121;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    let knobs = fast_backoff_knobs(tv.client());
    tv.client().stream_mux.set_knobs_for_test(knobs);
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Probe,
        [fault(ErrorKind::IO, "synthetic connection reset")],
    );
    let capture = ThreadCapture::start_debug();

    let c = consume(&tv, OID).await;
    let line = await_lines(&capture, PROBE_BACKOFF_MSG, 1).await.remove(0);
    assert_eq!(line["fields"]["consecutive_failures"], 1, "{line}");
    assert_eq!(
        line["fields"]["backoff_window_ms"],
        knobs.backoff_initial.as_millis() as u64,
        "{line}"
    );
    xadd_update(&tv, OID, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);

    assert_eq!(
        capture.snapshot().with_message(PROBE_BACKOFF_MSG).len(),
        1,
        "one failed probe, then an answered one"
    );
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_PROBE, 0),
        "a transient probe error is backoff-only"
    );
    assert!(c.take_terminal_exit().is_none());
    drop(capture);
    purge_keys(&tv).await;
}

/// A probe timeout and a transient probe error each count one failure: with a
/// budget of 1, the second failed probe exhausts it and ends the reader's
/// subscription `Unavailable`, naming the count and the last probe failure.
/// Only the timeout forces a reconnect. The budget resets with the ending, so
/// the reader probes again for the next subscription and serves it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn failed_probes_count_one_failure_each_and_past_the_budget_end_the_readers_subscriptions() {
    const OID: u64 = 122;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        max_transient_failures: 1,
        reader_probe_timeout: PROBE_TIMEOUT,
        ..fast_backoff_knobs(tv.client())
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Probe,
        [
            MuxFault::Hang,
            fault(ErrorKind::IO, "synthetic connection reset"),
        ],
    );

    let c = consume(&tv, OID).await;
    let message = assert_ends_with(&c, STREAM_EXIT_ERROR).await;
    let key = tv.key_schema().layer_stream(OID);
    assert_eq!(
        message,
        format!(
            "stream reader for {key} gave up after 2 consecutive transient failures (last: its \
             connection probe failed transiently (kind: IO Error))"
        )
    );
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_PROBE, 1),
        "the probe that did not answer reconnects; the transient error does not"
    );
    assert_eq!(read_permanent_exits(&metrics, CONSUMER_PHASE_PROBE), 0);

    let fresh = consume(&tv, OID).await;
    xadd_update(&tv, OID, 1).await;
    assert_eq!(recv_seqs(&fresh, 1).await, [1]);
    assert!(fresh.take_terminal_exit().is_none());
}

/// An answered probe charges nothing: with a budget of 1, the one transient
/// XREAD failure after it is the budget's first, so the subscription is
/// served.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_answered_probe_charges_nothing_to_the_transient_budget() {
    const OID: u64 = 123;
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        max_transient_failures: 1,
        ..fast_backoff_knobs(tv.client())
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [fault(ErrorKind::IO, "synthetic connection reset")],
    );

    let c = consume(&tv, OID).await;
    await_faults_consumed(tv.client()).await;
    xadd_update(&tv, OID, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    assert!(c.take_terminal_exit().is_none());
}

/// An answered probe resets nothing either: after a failed probe, the probe
/// that answers leaves that failure on the budget, so with a budget of 1 the
/// next transient XREAD failure is the second and ends the subscription.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_answered_probe_does_not_reset_the_transient_budget() {
    const OID: u64 = 124;
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        max_transient_failures: 1,
        ..fast_backoff_knobs(tv.client())
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Probe,
        [fault(ErrorKind::IO, "synthetic connection reset")],
    );
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [fault(ErrorKind::IO, "synthetic connection reset")],
    );

    let c = consume(&tv, OID).await;
    let message = assert_ends_with(&c, STREAM_EXIT_ERROR).await;
    assert!(
        message.contains("gave up after 2 consecutive transient XREAD failures"),
        "the failed probe is still on the budget when the XREAD fails; got {message:?}"
    );
}

/// A permanent probe error is a whole-reader failure — no key can explain a
/// `PING` failing: every subscription on the reader ends `Unavailable` with the
/// probe's message, one permanent exit counts under `phase="probe"`, and the
/// reader heals (one forced reconnect under `phase="permanent"`, and no other
/// phase) and serves the next subscription.
///
/// Current-thread, so both subscriptions are registered before the reader
/// probes.
#[tokio::test]
async fn a_permanent_probe_error_ends_every_subscription_counts_under_probe_and_heals() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client()
        .stream_mux
        .set_knobs_for_test(fast_backoff_knobs(tv.client()));
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Probe,
        [fault(ErrorKind::Auth, "synthetic permanent failure")],
    );

    let subs = [consume(&tv, 125).await, consume(&tv, 126).await];
    for sub in &subs {
        let message = assert_ends_with(sub, STREAM_EXIT_ERROR).await;
        assert!(
            message.contains(
                "failed its connection probe with a permanent error (kind: Authentication Error)"
            ),
            "{message:?}"
        );
    }
    assert_eq!(read_permanent_exits(&metrics, CONSUMER_PHASE_PROBE), 1);
    assert_eq!(read_permanent_exits(&metrics, CONSUMER_PHASE_XREAD), 0);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_PERMANENT, 1)
    );

    let fresh = consume(&tv, 125).await;
    xadd_update(&tv, 125, 1).await;
    assert_eq!(recv_seqs(&fresh, 1).await, [1]);
    purge_keys(&tv).await;
}

// ── The cluster slot-rebind probe ───────────────────────────────────────────

/// A one-reader cluster fixture, or `None` (with a `SKIP:` line) when no
/// cluster is configured.
async fn one_reader_cluster() -> Option<TestValkey> {
    let Ok(url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
        return None;
    };
    let tv = TestValkey::setup_with_url_and_readers(url, 4, 1).await;
    assert!(tv.client().pool.next().is_clustered());
    Some(tv)
}

/// The hash slot of `object_id`'s layer stream under `tv`'s prefix.
fn stream_slot(tv: &TestValkey, object_id: u64) -> u16 {
    fred::util::redis_keyslot(tv.key_schema().layer_stream(object_id).as_bytes())
}

/// The first object id after `after` whose stream's hash slot is (`same`) or
/// is not (`!same`) that of `after`'s.
fn object_in_slot_of(tv: &TestValkey, after: u64, same: bool) -> u64 {
    let slot = stream_slot(tv, after);
    (after + 1..)
        .find(|&oid| (stream_slot(tv, oid) == slot) == same)
        .expect("some object id hashes to the slot wanted")
}

/// The primary owning `slot` in the readers' routing view (the pinned
/// data-pool client's cached cluster map).
fn slot_owner(tv: &TestValkey, slot: u16) -> Server {
    tv.client().pool.clients()[0]
        .cached_cluster_state()
        .expect("a cluster client caches the cluster map")
        .get_server(slot)
        .cloned()
        .expect("the routing view names every slot's owner")
}

/// The first object id after `after` whose stream is in another hash slot
/// than `after`'s, owned by the same primary. A rebind then keeps the
/// reader's connection — its node still owns the new slot — so only the
/// rebind trigger can make the probe due: on another primary the reader
/// would retarget, and the new connection's own probe would mask a missing
/// trigger.
fn object_in_another_slot_on_the_same_node(tv: &TestValkey, after: u64) -> u64 {
    let slot = stream_slot(tv, after);
    let owner = slot_owner(tv, slot);
    let next = (after + 1..after + 100_000)
        .find(|&oid| {
            let other = stream_slot(tv, oid);
            other != slot && slot_owner(tv, other) == owner
        })
        .expect("some object id hashes to another slot of the same primary");
    assert_eq!(
        slot_owner(tv, stream_slot(tv, next)),
        owner,
        "fixture self-check: both slots have one owner"
    );
    next
}

/// `live_edit_stream_mux_reader_connects_total{cause="retarget"}`, both
/// results: how often a reader moved to another primary.
fn retargets(metrics: &Metrics) -> i64 {
    [READER_CONNECT_RESULT_OK, READER_CONNECT_RESULT_FAILED]
        .iter()
        .map(|result| {
            read_series(
                metrics,
                "live_edit_stream_mux_reader_connects_total",
                &[("cause", READER_CONNECT_CAUSE_RETARGET), ("result", result)],
            )
        })
        .sum()
}

/// Idle keepalive ticks so far, answered or failed.
fn keepalive_ticks(metrics: &Metrics) -> i64 {
    keepalives(metrics, KEEPALIVE_RESULT_OK) + keepalives(metrics, KEEPALIVE_RESULT_FAILED)
}

/// Moves a one-reader multiplexer from `first`'s stream to `next`'s while a
/// read is in flight, with `fault` queued at the `Probe` site, and returns
/// once a subscription to `next` has been delivered an update written after
/// it joined.
///
/// The reader probes (at start) and parks its read on `first`. Then `fault`
/// is queued, `first`'s subscription dropped (a removal sends no wake, so the
/// read stays in flight while the reader's slot unbinds), and `next`
/// subscribed: the join's coverage wake returns the old read, and the reader
/// plans its next read over `next` alone. Every step after the park lands
/// inside the parked read's 30 s BLOCK and nothing is written to `first`, so
/// the reader always has a read in flight or something to read. It is never
/// idle, which would make the probe due for that reason instead. The keepalive
/// proves it: it ticks only while the reader is idle, every [`KEEPALIVE`]
/// here, and must not tick from the park to the delivery.
async fn move_the_reader_with_a_read_in_flight(
    tv: &TestValkey,
    metrics: &Metrics,
    first: u64,
    next: u64,
    fault: MuxFault,
) {
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        reader_probe_timeout: PROBE_TIMEOUT,
        keepalive_interval: Some(KEEPALIVE),
        ..fast_backoff_knobs(tv.client())
    });
    let old = consume(tv, first).await;
    await_readers_parked(tv.client()).await;
    let ticks = keepalive_ticks(metrics);

    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Probe, [fault]);
    drop(old);
    await_mux(tv.client(), "the first subscription is removed", |s| {
        s.subscriptions() == 0
    })
    .await;
    let moved = consume(tv, next).await;
    xadd_update(tv, next, 1).await;
    assert_eq!(recv_seqs(&moved, 1).await, [1]);
    assert!(moved.take_terminal_exit().is_none());
    assert_eq!(
        keepalive_ticks(metrics),
        ticks,
        "fixture self-check: the reader was never idle between the park and the delivery, so no \
         idle→active probe was due"
    );
}

/// On a cluster, a reader rebound to another hash slot while its read was in
/// flight probes the new slot's node before its first XREAD there, though it
/// never had nothing to read: the `Pass` queued at the `Probe` site is
/// consumed, and only a probe consumes it. Without the rebind trigger the
/// `Pass` stays queued: the reader's last answer came from the old slot's
/// node, and the join's wake decision went to the old read as its coverage
/// wake, so nothing would check the connection serving the new slot. Both
/// slots belong to one primary and the reader never retargets, so no new
/// connection's own probe can stand in for the rebind trigger.
///
/// Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL`; needs `test-fault-injection`
/// (the whole module does), without which it is compiled out, not skipped.
/// Not `#[ignore]`d: CI's `test-cluster:` job runs it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cluster_reader_rebound_to_another_slot_with_a_read_in_flight_probes_before_reading_there(
) {
    let Some(tv) = one_reader_cluster().await else {
        return;
    };
    let metrics = install_metrics(&tv).await;
    let first = 1;
    let next = object_in_another_slot_on_the_same_node(&tv, first);

    move_the_reader_with_a_read_in_flight(&tv, &metrics, first, next, MuxFault::Pass).await;
    assert_eq!(
        tv.client()
            .stream_mux
            .pending_faults_for_test(MuxFaultSite::Probe),
        0,
        "the rebind probed the new slot's node before the first read there"
    );
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_PROBE, 0),
        "the probe answered"
    );
    assert_eq!(
        retargets(&metrics),
        0,
        "both slots have one owner: the reader kept its connection"
    );
}

/// A rebind probe that does not answer reads the new slot's connection as
/// wedged, like any probe: one forced reconnect under `phase="probe"` and no
/// other phase. The reader backs off, probes again, and delivers the update.
/// Both slots belong to one primary, so the reader rebuilds its connection in
/// place, never retargeting.
///
/// Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL`; needs `test-fault-injection`.
/// Not `#[ignore]`d: CI's `test-cluster:` job runs it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cluster_rebind_probe_that_does_not_answer_is_reconnected_under_probe_and_loses_nothing()
{
    let Some(tv) = one_reader_cluster().await else {
        return;
    };
    let metrics = install_metrics(&tv).await;
    let first = 1;
    let next = object_in_another_slot_on_the_same_node(&tv, first);

    move_the_reader_with_a_read_in_flight(&tv, &metrics, first, next, MuxFault::Hang).await;
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_PROBE, 1),
        "the rebind probe hung, and its timeout reconnected the reader"
    );
    assert_eq!(
        tv.client()
            .stream_mux
            .pending_faults_for_test(MuxFaultSite::Probe),
        0
    );
    assert_eq!(read_permanent_exits(&metrics, CONSUMER_PHASE_PROBE), 0);
    assert_eq!(
        retargets(&metrics),
        0,
        "both slots have one owner: no retarget"
    );
}

/// Negative control: on a cluster, a reader moved to another stream in the
/// hash slot it was already bound to is not probed — the `Pass` stays queued —
/// since the same node serves the read and the reader never went idle.
///
/// Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL`; needs `test-fault-injection`.
/// Not `#[ignore]`d: CI's `test-cluster:` job runs it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cluster_reader_moved_to_another_stream_in_the_same_slot_is_not_probed() {
    let Some(tv) = one_reader_cluster().await else {
        return;
    };
    let metrics = install_metrics(&tv).await;
    let first = 1;
    let next = object_in_slot_of(&tv, first, true);

    move_the_reader_with_a_read_in_flight(&tv, &metrics, first, next, MuxFault::Pass).await;
    assert_eq!(
        tv.client()
            .stream_mux
            .pending_faults_for_test(MuxFaultSite::Probe),
        1,
        "a read on the slot already bound is not probed"
    );
}

/// Negative control: on standalone Valkey, where one node serves every
/// stream, a reader moved to another stream while its read was in flight is
/// not probed — the `Pass` stays queued.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_standalone_reader_moved_to_another_stream_with_a_read_in_flight_is_not_probed() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    assert_ne!(
        stream_slot(&tv, 136),
        stream_slot(&tv, 137),
        "fixture self-check: two slots"
    );

    move_the_reader_with_a_read_in_flight(&tv, &metrics, 136, 137, MuxFault::Pass).await;
    assert_eq!(
        tv.client()
            .stream_mux
            .pending_faults_for_test(MuxFaultSite::Probe),
        1,
        "standalone, a change of stream is no rebind"
    );
}

/// On a cluster, a reader that went idle and comes back on another hash slot
/// probes ONCE before its next read, not twice: the idle reader's due probe
/// and the slot change are one probe. Two `Pass`es are queued at the `Probe`
/// site and exactly one is consumed. This pins the order of the reader's
/// `note_slot_rebind(..) || probe_due` check: `note_slot_rebind` records the
/// new slot even when a probe is already due, so the pass after the probe
/// sees the same slot. Swapped, the slot would be recorded only after that
/// probe and the next pass would probe again (no `Pass` left).
///
/// The reader serves `FIRST`, loses it, and lets its last read return (the
/// short BLOCK); the keepalive ticking proves it went idle, since it ticks
/// only with nothing to read and no read in flight. Both slots belong to one
/// primary, so no retarget's connect stands between the two.
///
/// Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL`; needs `test-fault-injection`.
/// Not `#[ignore]`d: CI's `test-cluster:` job runs it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cluster_reader_that_went_idle_and_comes_back_on_another_slot_probes_once() {
    let Some(tv) = one_reader_cluster().await else {
        return;
    };
    let metrics = install_metrics(&tv).await;
    let first = 1;
    let next = object_in_another_slot_on_the_same_node(&tv, first);
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: 100,
        keepalive_interval: Some(KEEPALIVE),
        reader_probe_timeout: PROBE_TIMEOUT,
        ..fast_backoff_knobs(tv.client())
    });
    let old = consume(&tv, first).await;
    xadd_update(&tv, first, 1).await;
    assert_eq!(recv_seqs(&old, 1).await, [1]);
    drop(old);
    let ticks = keepalives(&metrics, KEEPALIVE_RESULT_OK);
    await_keepalives_at_least(&metrics, KEEPALIVE_RESULT_OK, ticks + 1).await;

    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Probe, [MuxFault::Pass, MuxFault::Pass]);
    let moved = consume(&tv, next).await;
    xadd_update(&tv, next, 1).await;
    assert_eq!(recv_seqs(&moved, 1).await, [1]);
    assert_eq!(
        tv.client()
            .stream_mux
            .pending_faults_for_test(MuxFaultSite::Probe),
        1,
        "one probe before the read on the new slot, not two"
    );
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_PROBE, 0)
    );
    assert_eq!(retargets(&metrics), 0, "both slots have one owner");
}

// ── The idle keepalive ──────────────────────────────────────────────────────

/// The BLOCK the keepalive fixtures read with, so a reader's last read returns
/// soon after its subscription leaves and the reader goes idle.
const SHORT_BLOCK_MS: u64 = 100;

/// Connects `tv`'s reader 0 — readers connect lazily, the first time they
/// serve a subscription — and leaves it connected and idle. A subscription to
/// `object_id` is served one update and dropped, under a [`SHORT_BLOCK_MS`]
/// BLOCK (kept in the knobs), so the read in flight returns and the reader has
/// nothing to read. Leaves the keepalive interval as it was.
async fn connect_the_reader_then_idle(tv: &TestValkey, object_id: u64) {
    let mux = &tv.client().stream_mux;
    mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: SHORT_BLOCK_MS,
        ..mux.knobs()
    });
    let c = consume(tv, object_id).await;
    xadd_update(tv, object_id, 1).await;
    assert_eq!(recv_seqs(&c, 1).await, [1]);
    drop(c);
    await_mux(tv.client(), "the subscription left", |s| {
        s.subscriptions() == 0 && s.readers[0].connected
    })
    .await;
}

/// A reader that has never served a subscription holds no connection (b.azh),
/// so it has nothing to keep alive: over many keepalive intervals it ticks
/// neither `ok` nor `failed`, and a reservation alone does not connect it
/// either (a reservation registers no stream). Before b.azh every reader was
/// connected at startup and an idle one ticked from the start.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unconnected_reader_runs_no_keepalive() {
    const OID: u64 = 129;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client()
        .stream_mux
        .set_knobs_for_test(keepalive_every(&tv, KEEPALIVE));
    let reserved = reserve(&tv, OID).await;

    tokio::time::sleep(QUIET).await;
    assert_eq!(
        keepalive_ticks(&metrics),
        0,
        "an unconnected reader does not tick"
    );
    assert_eq!(
        tv.client().stream_mux.connected_readers(),
        0,
        "nor does a reservation connect it"
    );
    drop(reserved);
}

/// An idle connected reader's keepalive ticks `ok` on the connection it
/// already has: the reader's connection ids are the same after several ticks,
/// nothing was force-reconnected, and no XREAD was issued while it ticked.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_idle_readers_keepalive_ticks_ok_on_the_same_connection() {
    const OID: u64 = 134;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    connect_the_reader_then_idle(&tv, OID).await;
    let reader = tv
        .client()
        .stream_mux
        .reader_client_for_test(0)
        .expect("the reader connected to serve its subscription");
    let ids = reader.connection_ids();
    assert!(!ids.is_empty(), "fred caches CLIENT ID on connect");

    tv.client()
        .stream_mux
        .set_knobs_for_test(keepalive_every(&tv, KEEPALIVE));
    await_keepalives_at_least(&metrics, KEEPALIVE_RESULT_OK, 1).await;
    let xreads = tv.client().stream_mux.snapshot().xreads();
    await_keepalives_at_least(&metrics, KEEPALIVE_RESULT_OK, 3).await;

    assert_eq!(keepalives(&metrics, KEEPALIVE_RESULT_FAILED), 0);
    assert_eq!(
        reader.connection_ids(),
        ids,
        "an answered keepalive keeps the connection"
    );
    assert_eq!(
        tv.client()
            .stream_mux
            .reader_client_for_test(0)
            .map(|c| ClientLike::id(&c).to_owned()),
        Some(ClientLike::id(&reader).to_owned()),
        "and the client"
    );
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_KEEPALIVE, 0)
    );
    assert_eq!(
        tv.client().stream_mux.snapshot().xreads(),
        xreads,
        "an idle reader reads nothing"
    );
}

/// A keepalive tick whose `PING` fails (`fault`) counts `failed` once and
/// force-reconnects once under `phase="keepalive"` (and no other phase); the
/// reader keeps ticking on the new connection.
async fn a_failed_keepalive_tick_counts_failed_and_reconnects(fault: MuxFault) {
    const OID: u64 = 135;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    connect_the_reader_then_idle(&tv, OID).await;
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Keepalive, [fault]);
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        // Bounds a hung `PING`.
        probe_timeout: PROBE_TIMEOUT,
        ..keepalive_every(&tv, KEEPALIVE)
    });

    await_series(
        &metrics,
        KEEPALIVES,
        &[("result", KEEPALIVE_RESULT_FAILED)],
        1,
    )
    .await;
    await_series(
        &metrics,
        FORCED_RECONNECTS,
        &[("phase", CONSUMER_PHASE_KEEPALIVE)],
        1,
    )
    .await;
    await_keepalives_at_least(&metrics, KEEPALIVE_RESULT_OK, 1).await;

    assert_eq!(keepalives(&metrics, KEEPALIVE_RESULT_FAILED), 1);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_KEEPALIVE, 1)
    );
    assert_eq!(
        tv.client()
            .stream_mux
            .pending_faults_for_test(MuxFaultSite::Keepalive),
        0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_keepalive_ping_error_counts_failed_and_reconnects_under_keepalive() {
    a_failed_keepalive_tick_counts_failed_and_reconnects(fault(
        ErrorKind::IO,
        "synthetic connection reset",
    ))
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_keepalive_ping_that_does_not_answer_counts_failed_and_reconnects_under_keepalive() {
    a_failed_keepalive_tick_counts_failed_and_reconnects(MuxFault::Hang).await;
}

/// The keepalive neither charges nor resets the transient budget. A permanent
/// probe error leaves one failure on the budget of 1 (its heal); the idle
/// reader then ticks once `failed` and once `ok`; a subscription's first
/// transient XREAD failure is then the budget's second — "2", neither "3" (a
/// failed tick charged) nor served (an answered tick reset).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_idle_keepalive_neither_charges_nor_resets_the_transient_budget() {
    const OID: u64 = 130;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        max_transient_failures: 1,
        keepalive_interval: Some(KEEPALIVE),
        ..fast_backoff_knobs(tv.client())
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Probe,
        [fault(ErrorKind::Auth, "synthetic permanent failure")],
    );
    let healed = consume(&tv, OID).await;
    assert_ends_with(&healed, STREAM_EXIT_ERROR).await;

    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Keepalive,
        [fault(ErrorKind::IO, "synthetic connection reset")],
    );
    await_series(
        &metrics,
        KEEPALIVES,
        &[("result", KEEPALIVE_RESULT_FAILED)],
        1,
    )
    .await;
    let ok = keepalives(&metrics, KEEPALIVE_RESULT_OK);
    await_keepalives_at_least(&metrics, KEEPALIVE_RESULT_OK, ok + 1).await;

    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [fault(ErrorKind::IO, "synthetic connection reset")],
    );
    let c = consume(&tv, OID).await;
    let message = assert_ends_with(&c, STREAM_EXIT_ERROR).await;
    assert!(
        message.contains("gave up after 2 consecutive transient XREAD failures"),
        "the heal's failure and the XREAD's, and nothing from the keepalive; got {message:?}"
    );
}

/// With a read in flight the keepalive never runs: over many intervals no
/// tick is counted and the read is never interrupted — a `PING` on the
/// reader's connection would `CLIENT UNBLOCK` it, re-planning a new read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_keepalive_never_runs_while_a_read_is_in_flight() {
    const OID: u64 = 131;
    const CONNECT_OID: u64 = 138;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    let production_block_ms = tv.client().stream_mux.knobs().block_ms;
    connect_the_reader_then_idle(&tv, CONNECT_OID).await;
    // The read below blocks for the production BLOCK again, so it stays in
    // flight across the observation window instead of returning every
    // `SHORT_BLOCK_MS`.
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: production_block_ms,
        ..keepalive_every(&tv, KEEPALIVE)
    });
    // Non-vacuity: the idle connected reader does tick at this interval.
    await_keepalives_at_least(&metrics, KEEPALIVE_RESULT_OK, 1).await;

    let idle_xreads = tv.client().stream_mux.snapshot().xreads();
    let _c = consume(&tv, OID).await;
    await_mux(tv.client(), "the read is in flight", |s| {
        s.xreads() > idle_xreads
    })
    .await;
    let ticks =
        keepalives(&metrics, KEEPALIVE_RESULT_OK) + keepalives(&metrics, KEEPALIVE_RESULT_FAILED);
    let xreads = tv.client().stream_mux.snapshot().xreads();
    tokio::time::sleep(QUIET).await;

    assert_eq!(
        keepalives(&metrics, KEEPALIVE_RESULT_OK) + keepalives(&metrics, KEEPALIVE_RESULT_FAILED),
        ticks,
        "no keepalive tick while the read is in flight"
    );
    assert_eq!(read_unblocked_retries(&metrics), 0);
    assert_eq!(
        tv.client().stream_mux.snapshot().xreads(),
        xreads,
        "the read in flight was not interrupted"
    );
}

/// With a backoff pending the keepalive never runs, even with nothing to read:
/// after a whole-reader failure ended the only subscription, the reader sleeps
/// out its backoff (an hour here) without ticking. It still holds its
/// connection through the window (checked first), so the silence is the
/// backoff's, not that of a reader with nothing to keep alive.
///
/// Current-thread, so the reader's DEBUG backoff line — which carries the
/// full-jitter draw — is captured; the check is skipped when the draw lands
/// inside the observation window.
#[tokio::test]
async fn the_keepalive_never_runs_while_a_backoff_is_pending() {
    const HOUR: Duration = Duration::from_secs(3_600);
    const OID: u64 = 132;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        backoff_initial: HOUR,
        backoff_cap: HOUR,
        ..keepalive_every(&tv, KEEPALIVE)
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [fault(ErrorKind::Auth, "synthetic permanent failure")],
    );
    let capture = ThreadCapture::start_debug();

    let c = consume(&tv, OID).await;
    assert_ends_with(&c, STREAM_EXIT_ERROR).await;
    let line = await_lines(&capture, WHOLE_READER_BACKOFF_MSG, 1)
        .await
        .remove(0);
    let sleep_ms = line["fields"]["sleep_ms"]
        .as_u64()
        .expect("sleep_ms is a number");
    if sleep_ms > 4 * QUIET.as_millis() as u64 {
        let ticks = keepalives(&metrics, KEEPALIVE_RESULT_OK)
            + keepalives(&metrics, KEEPALIVE_RESULT_FAILED);
        assert!(
            tv.client().stream_mux.snapshot().readers[0].connected,
            "precondition: the backing-off reader holds its connection — an unconnected reader \
             runs no keepalive either, which would make the check below vacuous"
        );
        tokio::time::sleep(QUIET).await;
        assert!(
            tv.client().stream_mux.snapshot().readers[0].connected,
            "precondition: and it held it through the observation window"
        );
        assert_eq!(
            keepalives(&metrics, KEEPALIVE_RESULT_OK)
                + keepalives(&metrics, KEEPALIVE_RESULT_FAILED),
            ticks,
            "no keepalive tick while the reader backs off"
        );
    } else {
        eprintln!(
            "the full-jitter backoff drew {sleep_ms} ms, inside the observation window; the \
             no-tick check is skipped this run"
        );
    }
    drop(capture);
    purge_keys(&tv).await;
}

/// A reservation registers no stream, so a connected reader holding only
/// reservations has nothing to read: it issues no XREAD and runs the
/// keepalive. (The reader is connected first: an unconnected one has nothing
/// to keep alive, see [`an_unconnected_reader_runs_no_keepalive`].)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_reader_holding_only_a_reservation_is_idle_and_runs_the_keepalive() {
    const OID: u64 = 133;
    const CONNECT_OID: u64 = 139;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    connect_the_reader_then_idle(&tv, CONNECT_OID).await;
    let reserved = reserve(&tv, OID).await;
    tv.client()
        .stream_mux
        .set_knobs_for_test(keepalive_every(&tv, KEEPALIVE));

    await_keepalives_at_least(&metrics, KEEPALIVE_RESULT_OK, 1).await;
    // `keys` is the key count of the reader's last planned read, so it still
    // shows the connecting subscription's; it must not move either.
    let idle = tv.client().stream_mux.snapshot();
    let ok = keepalives(&metrics, KEEPALIVE_RESULT_OK);
    await_keepalives_at_least(&metrics, KEEPALIVE_RESULT_OK, ok + 2).await;
    let snap = tv.client().stream_mux.snapshot();
    assert_eq!(snap.subscriptions(), 1, "the reservation holds the reader");
    assert_eq!(
        (snap.readers[0].keys, snap.xreads()),
        (idle.readers[0].keys, idle.xreads()),
        "and the reader has nothing to read: it plans and issues no read"
    );
    assert_eq!(keepalives(&metrics, KEEPALIVE_RESULT_FAILED), 0);
    drop(reserved);
}

/// The commands `node` has processed on connection `id` (`tot-cmds` of
/// `CLIENT LIST ID <id>`, sent to `node`: connection ids are per node).
async fn commands_on_connection(tv: &TestValkey, node: &Server, id: i64) -> u64 {
    use fred::types::config::Options;

    let pinned = Options {
        cluster_node: Some(node.clone()),
        max_attempts: Some(1),
        ..Default::default()
    };
    let listing: String = tv
        .client()
        .pool
        .next()
        .with_options(&pinned)
        .client_list::<String, String>(None, Some(vec![id.to_string()]))
        .await
        .unwrap_or_else(|e| panic!("CLIENT LIST ID {id} on {node} must succeed: {e}"));
    let line = listing.trim();
    assert!(
        !line.is_empty(),
        "{node} no longer lists the reader's connection {id}: it was replaced"
    );
    line.split_whitespace()
        .find_map(|kv| kv.strip_prefix("tot-cmds="))
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("CLIENT LIST carries tot-cmds; got {line:?}"))
}

/// On a cluster an idle reader's keepalive pings its one connection — to the
/// primary that owns the slot it last served (b.azh: a reader connects to its
/// slot's owner alone) — and that connection is what the keepalive keeps
/// alive: across `n` answered ticks it processes at least `n - 1` commands
/// (the tick in progress at the first reading may land on either side of it).
/// Counted per connection — the reader's, idle, so the keepalive is the only
/// thing sending on it — not per node, where any other client's `PING` would
/// count too. (`stream_mux_reader_connect_test.rs` pins, on a two-node facade
/// that runs without a cluster, that a tick consumes one keepalive fault.)
///
/// Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL`, and needs
/// `test-fault-injection` for the keepalive knob (the whole module does).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_idle_cluster_readers_keepalive_pings_its_one_connection_to_its_slot_owner() {
    const TICKS: i64 = 3;
    const OID: u64 = 1;
    let Ok(url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
        return;
    };
    let tv = TestValkey::setup_with_url_and_readers(url, 4, 1).await;
    assert!(tv.client().pool.next().is_clustered());
    let metrics = install_metrics(&tv).await;
    connect_the_reader_then_idle(&tv, OID).await;
    let reader = tv
        .client()
        .stream_mux
        .reader_client_for_test(0)
        .expect("the reader connected to serve its subscription");
    let owner = tv.client().pool.clients()[0]
        .cached_cluster_state()
        .expect("the routing view is a cluster map")
        .get_server(stream_slot(&tv, OID))
        .cloned()
        .expect("the routing view names the slot's owner");
    assert_eq!(
        reader.active_connections(),
        vec![owner.clone()],
        "the reader holds one connection, to the slot's owner"
    );
    let ids = reader.connection_ids();
    let id = *ids
        .get(&owner)
        .unwrap_or_else(|| panic!("the reader's connection is to {owner}; ids {ids:?}"));

    tv.client()
        .stream_mux
        .set_knobs_for_test(keepalive_every(&tv, Duration::from_millis(50)));
    await_keepalives_at_least(&metrics, KEEPALIVE_RESULT_OK, 1).await;
    let before = commands_on_connection(&tv, &owner, id).await;
    let first = keepalives(&metrics, KEEPALIVE_RESULT_OK);
    await_keepalives_at_least(&metrics, KEEPALIVE_RESULT_OK, first + TICKS).await;
    let ticks = keepalives(&metrics, KEEPALIVE_RESULT_OK) - first;
    let pinged = commands_on_connection(&tv, &owner, id).await - before;
    assert!(
        pinged as i64 >= ticks - 1,
        "the reader's connection {id} to {owner} processed {pinged} commands over {ticks} \
         keepalive ticks"
    );
    assert_eq!(
        reader.connection_ids(),
        ids,
        "the reader kept its connection"
    );
    assert_eq!(keepalives(&metrics, KEEPALIVE_RESULT_FAILED), 0);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_KEEPALIVE, 0)
    );
}

// ── The idle keepalive against a store's idle timeout (disruptive) ───────────

const DISRUPTIVE_VALKEY_URL: &str = "LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL";

/// The server `timeout` (seconds) the disruptive test sets: it closes a client
/// connection idle for longer.
const IDLE_TIMEOUT_SECS: u64 = 2;

/// Puts the server's `timeout` back when dropped — on a panic too, so a
/// failing assertion never leaves the server closing idle connections.
struct RestoreIdleTimeout {
    client: Option<Client>,
    original: String,
}

impl RestoreIdleTimeout {
    /// Reads the server's `timeout` through `client`, then sets it to
    /// [`IDLE_TIMEOUT_SECS`]. A value equal to ours can only be left over by an
    /// earlier run killed before it restored, so the server default (`0`,
    /// never close) is restored instead.
    async fn set(client: &Client) -> Self {
        let current: std::collections::HashMap<String, String> = client
            .config_get("timeout")
            .await
            .expect("CONFIG GET timeout must succeed");
        let current = current
            .get("timeout")
            .cloned()
            .expect("CONFIG GET timeout names the parameter");
        let original = if current == IDLE_TIMEOUT_SECS.to_string() {
            "0".to_owned()
        } else {
            current
        };
        client
            .config_set("timeout", IDLE_TIMEOUT_SECS)
            .await
            .expect("CONFIG SET timeout must succeed");
        Self {
            client: Some(client.clone()),
            original,
        }
    }

    async fn restore(mut self) {
        let client = self.client.take().expect("restored once");
        client
            .config_set("timeout", self.original.as_str())
            .await
            .expect("restoring the server's timeout must succeed");
    }
}

impl Drop for RestoreIdleTimeout {
    fn drop(&mut self) {
        let Some(client) = self.client.take() else {
            return;
        };
        // Only reached when the test failed first: restore without asserting,
        // since a second panic while unwinding would abort the run.
        let original = std::mem::take(&mut self.original);
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let _ = tokio::time::timeout(
                    SETTLE,
                    client.config_set::<_, _>("timeout", original.as_str()),
                )
                .await;
            });
        });
    }
}

/// Every stream reader connection id `tv`'s client holds, sorted.
fn reader_connection_ids(tv: &TestValkey) -> Vec<i64> {
    let mut ids: Vec<i64> = tv
        .client()
        .stream_mux
        .reader_clients_for_test()
        .iter()
        .flat_map(|c| c.connection_ids().into_values())
        .collect();
    ids.sort_unstable();
    ids
}

/// Which of `ids` the server still lists (`CLIENT LIST ID`), sorted. Retried
/// until [`SETTLE`]: the server's idle timeout closes the observer's own
/// connection too, and fred reconnects it.
async fn listed(observer: &Client, ids: &[i64]) -> Vec<i64> {
    let deadline = tokio::time::Instant::now() + SETTLE;
    let wanted: Vec<String> = ids.iter().map(ToString::to_string).collect();
    loop {
        match observer
            .client_list::<String, String>(None, Some(wanted.clone()))
            .await
        {
            Ok(listing) => {
                let mut listed: Vec<i64> = listing
                    .lines()
                    .filter_map(|line| {
                        line.split_whitespace()
                            .find_map(|kv| kv.strip_prefix("id="))
                            .and_then(|id| id.parse().ok())
                    })
                    .collect();
                listed.sort_unstable();
                return listed;
            }
            Err(e) => assert!(
                tokio::time::Instant::now() < deadline,
                "CLIENT LIST ID kept failing: {e}"
            ),
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// Against a store that closes idle connections (`timeout 2`), the idle
/// keepalive (every 500 ms here) keeps every idle reader's connection open:
/// after three idle timeouts the server still lists each reader connection
/// and the readers hold the same ones. Non-vacuity: with the keepalive turned
/// off, the server closes them. Both readers are connected first, by serving
/// one subscription each: a reader connects lazily.
///
/// Disruptive — it changes the server's `timeout` — so `#[ignore]`d, and it
/// runs only against its own Valkey, named by
/// `LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL` (CI's `test-cluster-disruptive:` job
/// sets it to that job's `valkey-disruptive` service), never the shared one.
/// It skips when that variable is unset, except in CI (`GITLAB_CI` set), where
/// an unset variable is broken job wiring and fails it. It restores `timeout`
/// when it ends,
/// on a failure too. Both arms are one test on purpose: they share the
/// server's setting, so as two tests nextest could run them in parallel and
/// one's restore would land inside the other's wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "ci: test-cluster-disruptive; disruptive, sets the server's `timeout` on its own Valkey \
            (LIVE_EDIT_TEST_DISRUPTIVE_VALKEY_URL)"]
async fn stream_mux_disruptive_idle_readers_outlive_the_idle_timeout_only_with_the_keepalive() {
    const KEEPALIVE_EVERY: Duration = Duration::from_millis(500);
    const IDLE_TIMEOUT: Duration = Duration::from_secs(IDLE_TIMEOUT_SECS);
    let Ok(url) = std::env::var(DISRUPTIVE_VALKEY_URL) else {
        // In CI the only job that runs this test sets the variable, so an
        // unset one there is broken wiring, not a local run: fail rather than
        // pass having checked nothing.
        assert!(
            std::env::var_os("GITLAB_CI").is_none(),
            "{DISRUPTIVE_VALKEY_URL} is unset in a CI job that runs the disruptive tests; the \
             job must point it at its own `valkey-disruptive` service"
        );
        eprintln!("SKIP: {DISRUPTIVE_VALKEY_URL} not set");
        return;
    };
    let tv = TestValkey::setup_with_url_and_readers(url, 2, 2).await;
    let metrics = install_metrics(&tv).await;
    // Readers connect lazily: serve one subscription on each (standalone
    // routing is `object_id % 2`), under a short BLOCK so both go idle.
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: SHORT_BLOCK_MS,
        ..tv.client().stream_mux.knobs()
    });
    for oid in [0, 1] {
        let c = consume(&tv, oid).await;
        xadd_update(&tv, oid, 1).await;
        assert_eq!(recv_seqs(&c, 1).await, [1]);
    }
    await_mux(tv.client(), "both readers connected and free", |s| {
        s.subscriptions() == 0 && s.connected_readers() == 2
    })
    .await;
    tv.client()
        .stream_mux
        .set_knobs_for_test(keepalive_every(&tv, KEEPALIVE_EVERY));
    let observer = tv.client().pool.next().clone();
    let restore = RestoreIdleTimeout::set(&observer).await;

    // With the keepalive: three idle timeouts later, every reader connection
    // is still open.
    let ids = reader_connection_ids(&tv);
    assert_eq!(
        ids.len(),
        2,
        "fixture self-check: one connection per reader"
    );
    let ticks = keepalives(&metrics, KEEPALIVE_RESULT_OK);
    tokio::time::sleep(3 * IDLE_TIMEOUT).await;
    assert_eq!(
        listed(&observer, &ids).await,
        ids,
        "the server closed an idle reader connection the keepalive should have kept open"
    );
    assert_eq!(
        reader_connection_ids(&tv),
        ids,
        "the readers hold the connections they had"
    );
    assert!(
        keepalives(&metrics, KEEPALIVE_RESULT_OK) > ticks,
        "the keepalive ticked while the readers were idle"
    );

    // Without it: the server closes them.
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        keepalive_interval: None,
        ..tv.client().stream_mux.knobs()
    });
    let ids = reader_connection_ids(&tv);
    let deadline = tokio::time::Instant::now() + 3 * IDLE_TIMEOUT + SETTLE;
    loop {
        let still_open = listed(&observer, &ids).await;
        if still_open.is_empty() {
            break;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "with the keepalive off the server never closed idle reader connections \
             {still_open:?}: the check above proves nothing on this server"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    restore.restore().await;
}

// ── The liveness wake ───────────────────────────────────────────────────────

/// L1. A join the in-flight read covers, once the connection's last answer
/// (the probe's) is older than `liveness_age`, sends one wake, for liveness.
/// The read is wedged (`MuxFault::Hang`), so it does not return within the
/// wake-ack deadline of the wake landing: the reader force-reconnects under
/// `phase="wake_ack"` (and no other phase), and the retried read delivers to
/// both subscriptions.
#[tokio::test]
async fn a_covered_join_onto_an_aged_read_sends_a_liveness_wake_and_a_wedge_is_reconnected() {
    const OID: u64 = 140;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        liveness_age: SHORT_LIVENESS_AGE,
        wake_ack_timeout: WAKE_ACK,
        ..fast_backoff_knobs(tv.client())
    });
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Xread, [MuxFault::Hang]);
    let capture = ThreadCapture::start_debug();

    let a = consume(&tv, OID).await;
    await_mux(tv.client(), "the hung read is in flight", |s| {
        s.xreads() >= 1
    })
    .await;
    // The liveness age is a timer: let the probe's answer age past it.
    tokio::time::sleep(2 * SHORT_LIVENESS_AGE).await;
    // Covered: the same key, at the read's own cursor.
    let b = consume(&tv, OID).await;
    await_series(
        &metrics,
        FORCED_RECONNECTS,
        &[("phase", CONSUMER_PHASE_WAKE_ACK)],
        1,
    )
    .await;

    xadd_update(&tv, OID, 1).await;
    assert_eq!(recv_seqs(&a, 1).await, [1]);
    assert_eq!(recv_seqs(&b, 1).await, [1]);
    assert_eq!(wake_reasons(&capture), ["liveness"]);
    assert_eq!(read_wakes(&metrics, WAKE_OUTCOME_OK), 1);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_WAKE_ACK, 1)
    );
    assert!(a.take_terminal_exit().is_none() && b.take_terminal_exit().is_none());
    drop(capture);
    purge_keys(&tv).await;
}

/// L2. A join the in-flight read covers, while the connection answered within
/// `liveness_age`, sends no wake — here after the read has aged past L1's
/// short age, so it is the knob that decides. The join is served by the read
/// already in flight.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_covered_join_while_the_connection_answered_within_the_liveness_age_sends_no_wake() {
    const OID: u64 = 141;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        liveness_age: Duration::from_secs(60),
        ..tv.client().stream_mux.knobs()
    });

    let a = consume(&tv, OID).await;
    await_mux(tv.client(), "the read is in flight", |s| s.xreads() >= 1).await;
    tokio::time::sleep(2 * SHORT_LIVENESS_AGE).await;
    let b = consume(&tv, OID).await;
    xadd_update(&tv, OID, 1).await;
    assert_eq!(recv_seqs(&a, 1).await, [1]);
    // `b` received, so its join was applied — and decided — before this.
    assert_eq!(recv_seqs(&b, 1).await, [1]);

    assert_eq!(tv.client().stream_mux.snapshot().readers[0].wakes, 0);
    assert_eq!(read_wakes(&metrics, WAKE_OUTCOME_OK), 0);
    assert_eq!(read_wakes(&metrics, WAKE_OUTCOME_ERROR), 0);
}

/// L3. An XREAD error voids the connection's last answer. The read issued
/// right after the error (and its backoff) is not evidence the connection
/// answers, however new it is: a covered join onto it at once — well inside
/// `liveness_age` of its issue — sends a liveness wake. The read is wedged, so
/// the verdict force-reconnects under `phase="wake_ack"` (the IO error left the
/// timeout streak at 0, so this is a streak's first verdict), and both
/// subscriptions are served.
#[tokio::test]
async fn a_covered_join_onto_a_read_issued_after_an_xread_error_sends_a_liveness_wake() {
    const OID: u64 = 142;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        liveness_age: LIVENESS_AGE,
        wake_ack_timeout: WAKE_ACK,
        ..fast_backoff_knobs(tv.client())
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [
            fault(ErrorKind::IO, "synthetic connection reset"),
            MuxFault::Hang,
        ],
    );
    let capture = ThreadCapture::start_debug();

    let a = consume(&tv, OID).await;
    await_mux(
        tv.client(),
        "the hung read is issued after the error",
        |s| s.xreads() >= 2,
    )
    .await;
    let b = consume(&tv, OID).await;
    await_series(
        &metrics,
        FORCED_RECONNECTS,
        &[("phase", CONSUMER_PHASE_WAKE_ACK)],
        1,
    )
    .await;

    xadd_update(&tv, OID, 1).await;
    assert_eq!(recv_seqs(&a, 1).await, [1]);
    assert_eq!(recv_seqs(&b, 1).await, [1]);
    assert_eq!(wake_reasons(&capture), ["liveness"]);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_WAKE_ACK, 1)
    );
    drop(capture);
    purge_keys(&tv).await;
}

/// L4. `-UNBLOCKED` is the server answering: after an IO error voided the last
/// answer, an `-UNBLOCKED` reply (re-planned inside the cheap allowance)
/// stamps it again, so a covered join right after the next read is issued
/// sends no wake.
///
/// The join sends nothing observable, so a second change is queued behind it:
/// a removal on another key, which the reader applies after it (in the same
/// step or the next), and which lowers the stream count. Once the count has
/// dropped, the join has been decided.
#[tokio::test]
async fn a_covered_join_after_an_unblocked_reply_sends_no_wake() {
    const OID: u64 = 143;
    const SIDE: u64 = 144;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        liveness_age: LIVENESS_AGE,
        wake_ack_timeout: WAKE_ACK,
        ..fast_backoff_knobs(tv.client())
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [
            fault(ErrorKind::IO, "synthetic connection reset"),
            fault(ErrorKind::Unknown, UNBLOCKED_REPLY),
            MuxFault::Hang,
        ],
    );
    let capture = ThreadCapture::start_debug();

    let _a = consume(&tv, OID).await;
    let side = consume(&tv, SIDE).await;
    await_mux(
        tv.client(),
        "the hung read is issued after the reply",
        |s| s.xreads() >= 3,
    )
    .await;
    let _b = consume(&tv, OID).await;
    drop(side);
    await_series(&metrics, STREAMS, &[], 1).await;

    assert_eq!(
        wake_reasons(&capture),
        Vec::<String>::new(),
        "the -UNBLOCKED reply is a recent answer"
    );
    assert_eq!(tv.client().stream_mux.snapshot().readers[0].wakes, 0);
    assert_eq!(
        read_unblocked_retries(&metrics),
        1,
        "fixture self-check: the reply was re-planned inside the allowance"
    );
    drop(capture);
    purge_keys(&tv).await;
}

/// L5. A join made between reads — here during the backoff after an XREAD
/// error — is kept and decided as the next read is issued: that read covers
/// it, but the connection has not answered since the error, so it gets a
/// liveness wake, a wake-ack verdict (the read is wedged) under
/// `phase="wake_ack"`, and both subscriptions are served.
///
/// The join lands in the backoff unless the full-jitter draw (up to 1 s) is
/// shorter than the reader's step to the next read; either way the outcome is
/// the same, and `take_wake_reason`'s unit test pins the carry itself.
#[tokio::test]
async fn a_join_made_between_reads_after_an_xread_error_gets_a_liveness_wake_at_the_next_read() {
    const OID: u64 = 145;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        backoff_initial: Duration::from_secs(1),
        backoff_cap: Duration::from_secs(1),
        liveness_age: LIVENESS_AGE,
        wake_ack_timeout: WAKE_ACK,
        ..tv.client().stream_mux.knobs()
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [
            fault(ErrorKind::IO, "synthetic connection reset"),
            MuxFault::Hang,
        ],
    );
    let capture = ThreadCapture::start_debug();

    let a = consume(&tv, OID).await;
    let deadline = tokio::time::Instant::now() + SETTLE;
    while tv
        .client()
        .stream_mux
        .pending_faults_for_test(MuxFaultSite::Xread)
        > 1
    {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the first read never took its fault"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let b = consume(&tv, OID).await;
    await_series(
        &metrics,
        FORCED_RECONNECTS,
        &[("phase", CONSUMER_PHASE_WAKE_ACK)],
        1,
    )
    .await;

    xadd_update(&tv, OID, 1).await;
    assert_eq!(recv_seqs(&a, 1).await, [1]);
    assert_eq!(recv_seqs(&b, 1).await, [1]);
    assert_eq!(wake_reasons(&capture), ["liveness"]);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_WAKE_ACK, 1)
    );
    drop(capture);
    purge_keys(&tv).await;
}

/// L6. Inside a timeout streak the liveness wake's verdict is a WARN, not a
/// reconnect: an XREAD `Timeout` starts the streak (forcing one reconnect under
/// `phase="xread"`), a covered join onto the wedged read issued after it
/// sends a liveness wake, and the read's no-answer verdict — the streak's
/// second — logs exactly one wake-ack WARN and forces nothing further.
#[tokio::test]
async fn inside_a_timeout_streak_a_liveness_wakes_verdict_is_a_warn_not_a_reconnect() {
    const OID: u64 = 146;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        liveness_age: LIVENESS_AGE,
        wake_ack_timeout: WAKE_ACK,
        ..fast_backoff_knobs(tv.client())
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [
            fault(ErrorKind::Timeout, "synthetic wedged-connection timeout"),
            MuxFault::Hang,
        ],
    );
    let capture = ThreadCapture::start_debug();

    let _a = consume(&tv, OID).await;
    await_mux(
        tv.client(),
        "the hung read is issued after the timeout",
        |s| s.xreads() >= 2,
    )
    .await;
    let _b = consume(&tv, OID).await;
    await_lines(&capture, WAKE_ACK_WEDGE_MSG, 1).await;
    tokio::time::sleep(QUIET).await;

    let captured = capture.snapshot();
    let line = captured.only(WAKE_ACK_WEDGE_MSG);
    assert_eq!(line["level"], "WARN", "{line}");
    assert_eq!(wake_reasons(&capture), ["liveness"]);
    assert_eq!(
        forced_reconnect_row(&metrics),
        forced_reconnects_only_on(CONSUMER_PHASE_XREAD, 1),
        "the streak's first verdict (the XREAD timeout) reconnects; its second (the wake's) \
         does not"
    );
    drop(capture);
    purge_keys(&tv).await;
}

// ── What stamps and voids the connection's last answer ──────────────────────

/// The first commit after a restore band: its id jumps past the entry after
/// the five commits [`five_commits`] writes, so a subscription at `5-0`
/// reading it gets a head check.
const RESTORED: u64 = 1_005;

/// Commits `1..=5` on `object_id`'s stream.
async fn five_commits(tv: &TestValkey, object_id: u64) {
    for seq in 1..=5 {
        xadd_update(tv, object_id, seq).await;
    }
}

/// Subscribes to `object_id`'s layer stream from `from`.
async fn consume_from(tv: &TestValkey, object_id: u64, from: &str) -> LayerStreamConsumer {
    consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From(from.to_owned()),
        tokio::time::Instant::now() + Duration::from_secs(2),
    )
    .await
    .expect("registration is in-memory and must succeed")
}

/// An empty XREAD reply is an answer: with reads cycling on a short BLOCK, a
/// covered join made after the probe's answer is older than `liveness_age`
/// still sends no wake — the empty replies since keep the answer fresh.
/// Delivering to the joiner proves its join was decided.
#[tokio::test]
async fn a_covered_join_after_empty_replies_within_the_liveness_age_sends_no_wake() {
    const OID: u64 = 150;
    const AGE: Duration = Duration::from_millis(300);
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        block_ms: 50,
        liveness_age: AGE,
        ..tv.client().stream_mux.knobs()
    });
    let capture = ThreadCapture::start_debug();

    let a = consume(&tv, OID).await;
    await_mux(tv.client(), "the reader reads", |s| s.xreads() >= 1).await;
    // The liveness age is a timer: let the probe's answer age well past it.
    tokio::time::sleep(2 * AGE).await;
    let b = consume(&tv, OID).await;
    xadd_update(&tv, OID, 1).await;
    assert_eq!(recv_seqs(&a, 1).await, [1]);
    assert_eq!(recv_seqs(&b, 1).await, [1]);

    assert_eq!(
        wake_reasons(&capture),
        Vec::<String>::new(),
        "an empty reply within the age is the connection's answer"
    );
    drop(capture);
    purge_keys(&tv).await;
}

/// A delivered reply is an answer, even right after an XREAD error voided the
/// last one: after `[Err(IO)]`, a reply carrying an entry (no head check: it
/// is the entry after the cursor) stamps the answer again, so a covered join
/// at the new cursor sends no wake.
#[tokio::test]
async fn a_covered_join_after_a_delivered_reply_sends_no_wake_even_after_an_xread_error() {
    const OID: u64 = 151;
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        liveness_age: LIVENESS_AGE,
        ..fast_backoff_knobs(tv.client())
    });
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [fault(ErrorKind::IO, "synthetic connection reset")],
    );
    let capture = ThreadCapture::start_debug();

    let a = consume(&tv, OID).await;
    await_mux(tv.client(), "the read after the error is issued", |s| {
        s.xreads() >= 2
    })
    .await;
    xadd_update(&tv, OID, 1).await;
    assert_eq!(recv_seqs(&a, 1).await, [1]);
    let b = consume_from(&tv, OID, "1-0").await;
    xadd_update(&tv, OID, 2).await;
    assert_eq!(recv_seqs(&a, 1).await, [2]);
    assert_eq!(recv_seqs(&b, 1).await, [2]);

    assert_eq!(
        wake_reasons(&capture),
        Vec::<String>::new(),
        "the delivered reply is the connection's answer"
    );
    drop(capture);
    purge_keys(&tv).await;
}

/// A reply whose head check answered is an answer too: after `[Err(IO)]`, a
/// restore-band jump is checked (`no_gap`) and delivered, and a covered join
/// at the new cursor sends no wake.
#[tokio::test]
async fn a_covered_join_after_a_head_checked_reply_sends_no_wake_even_after_an_xread_error() {
    const OID: u64 = 152;
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        liveness_age: LIVENESS_AGE,
        ..fast_backoff_knobs(tv.client())
    });
    five_commits(&tv, OID).await;
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::Xread,
        [fault(ErrorKind::IO, "synthetic connection reset")],
    );
    let capture = ThreadCapture::start_debug();

    let a = consume_from(&tv, OID, "5-0").await;
    await_mux(tv.client(), "the read after the error is issued", |s| {
        s.xreads() >= 2
    })
    .await;
    xadd_update(&tv, OID, RESTORED).await;
    assert_eq!(recv_seqs(&a, 1).await, [RESTORED]);
    assert_eq!(
        read_series(
            &metrics,
            "live_edit_stream_mux_head_checks_total",
            &[("result", HEAD_CHECK_RESULT_NO_GAP)]
        ),
        1,
        "fixture self-check: the jump was head-checked"
    );
    let b = consume_from(&tv, OID, &format!("{RESTORED}-0")).await;
    xadd_update(&tv, OID, RESTORED + 1).await;
    assert_eq!(recv_seqs(&a, 1).await, [RESTORED + 1]);
    assert_eq!(recv_seqs(&b, 1).await, [RESTORED + 1]);

    assert_eq!(
        wake_reasons(&capture),
        Vec::<String>::new(),
        "the head-checked reply is the connection's answer"
    );
    drop(capture);
    purge_keys(&tv).await;
}

/// A failed head check voids the answer its reply gave: the reply stamped it
/// on arrival, but a check that failed on the same connection means nothing
/// vouches for it. A covered join made during the backoff that follows is
/// carried to the re-read and gets a liveness wake (the age, 2 s, is far
/// longer than the reply is old), and both subscriptions get the re-read
/// entry.
///
/// The join lands in the backoff (up to 1 s) unless the full-jitter draw is
/// shorter than the step to the join; landing on the re-read in flight gives
/// the same wake.
#[tokio::test]
async fn a_covered_join_after_a_failed_head_check_gets_a_liveness_wake() {
    const OID: u64 = 153;
    let tv = TestValkey::setup_with_readers(1).await;
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        backoff_initial: Duration::from_secs(1),
        backoff_cap: Duration::from_secs(1),
        liveness_age: LIVENESS_AGE,
        ..tv.client().stream_mux.knobs()
    });
    five_commits(&tv, OID).await;
    let capture = ThreadCapture::start_debug();

    let a = consume_from(&tv, OID, "5-0").await;
    await_mux(tv.client(), "the read is in flight", |s| s.xreads() >= 1).await;
    tv.client().stream_mux.push_faults_for_test(
        MuxFaultSite::HeadCheck,
        [fault(ErrorKind::IO, "synthetic head check failure")],
    );
    xadd_update(&tv, OID, RESTORED).await;
    let deadline = tokio::time::Instant::now() + SETTLE;
    while tv
        .client()
        .stream_mux
        .pending_faults_for_test(MuxFaultSite::HeadCheck)
        > 0
    {
        assert!(
            tokio::time::Instant::now() < deadline,
            "no head check took the queued fault"
        );
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let b = consume_from(&tv, OID, "5-0").await;
    assert_eq!(recv_seqs(&a, 1).await, [RESTORED]);
    assert_eq!(recv_seqs(&b, 1).await, [RESTORED]);

    assert_eq!(wake_reasons(&capture), ["liveness"]);
    drop(capture);
    purge_keys(&tv).await;
}
