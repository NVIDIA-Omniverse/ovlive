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

//! b.xtb stage 4 — a stream multiplexer reader against layer streams whose ids
//! jump, driven through `consume_layer_stream` on a real Valkey (SDD §12.3,
//! §15.3).
//!
//! When a reply's first entry for a stream is past the entry after the id the
//! stream was read from, the reader checks the stream's head
//! (`XRANGE key - + COUNT 1`) before fanning it out, and counts
//! `live_edit_stream_mux_head_checks_total{result}`:
//!
//! * a restore band's id jump over retained entries is `no_gap`: nothing is
//!   marked and the entry is delivered;
//! * a stream removed and recreated above a subscription's position is `gap`:
//!   the subscription ends `stream_gap` — the intended resync onto the layer's
//!   restored content — while one whose next edit is the recreated stream's
//!   first entry continues;
//! * a `Deleted` sentinel first entry is never checked, even over a trimmed
//!   head: it is delivered, and the handler re-bootstraps;
//! * a failed or unanswered check holds the stream out of that reply: nothing
//!   is lost, the reader backs off, and the stream is re-read and checked
//!   again. Each such reply is one transient failure on the reader's budget,
//!   never reset by the XREAD that carried it, so a check that keeps failing
//!   ends every subscription on the reader `error` once the budget is spent;
//! * a shutdown during a hung check stops the reader inside the grace.
//!
//! The trim cases, which need a real `MAXLEN ~` turnover under a stalled
//! reader and a caught-up floor, are driven through `SubscribeReadLayer` in
//! `grpc/subscribe_stream_gap_test.rs`; the marker and floor rules themselves
//! are unit-tested in `src/storage/stream_mux/` and property-tested in
//! `tests/proptests.rs` (`stream_mux_gaps`).
//!
//! The fault cases drive `MuxFaultSite::HeadCheck` and exist only under
//! **`test-fault-injection`**; without it they are compiled out and the run
//! reports fewer tests.
//!
//! Waits are deadlines (`SETTLE`), never measurements. The two negative checks
//! — nothing delivered twice after a re-read, nothing read during a backoff —
//! hold for a bounded observation window ([`QUIET`]).

use std::time::Duration;

use fred::interfaces::{KeysInterface, StreamsInterface};

use live_edit::observability::metrics::{
    Metrics, HEAD_CHECK_RESULTS, HEAD_CHECK_RESULT_ERROR, HEAD_CHECK_RESULT_GAP,
    HEAD_CHECK_RESULT_NO_GAP, STREAM_EXIT_STREAM_GAP,
};
use live_edit::storage::notifications::STREAM_GAP_MSG;
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::{consume_layer_stream, LayerStreamConsumer, StreamCursor};

use super::mux_test_helpers::{
    await_readers_parked, install_metrics, read_series, recv_seqs, recv_within, xadd_update, SETTLE,
};

/// Bounded observation window for the two negative checks (module header).
#[cfg(feature = "test-fault-injection")]
const QUIET: Duration = Duration::from_millis(300);

/// The first commit after a restore: its seeded sequence (`RESTORED - 1`)
/// skips a band well clear of the five commits every test here writes first.
const RESTORED: u64 = 1_005;

async fn consume(tv: &TestValkey, object_id: u64, from: &str) -> LayerStreamConsumer {
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

/// Commits `1..=5` on `object_id`'s stream.
async fn five_commits(tv: &TestValkey, object_id: u64) {
    for seq in 1..=5 {
        xadd_update(tv, object_id, seq).await;
    }
}

/// The id of the stream's first retained entry (`XRANGE key - + COUNT 1`).
async fn head_of(tv: &TestValkey, object_id: u64) -> Option<String> {
    let first: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = tv
        .client()
        .pool
        .next()
        .xrange_values(tv.key_schema().layer_stream(object_id), "-", "+", Some(1))
        .await
        .expect("XRANGE must succeed");
    first.into_iter().next().map(|(id, _)| id)
}

/// `live_edit_stream_mux_head_checks_total` by result, in
/// `HEAD_CHECK_RESULTS` order.
fn head_checks(metrics: &Metrics) -> Vec<(&'static str, i64)> {
    HEAD_CHECK_RESULTS
        .iter()
        .map(|result| {
            (
                *result,
                read_series(
                    metrics,
                    "live_edit_stream_mux_head_checks_total",
                    &[("result", result)],
                ),
            )
        })
        .collect()
}

/// `(gap, no_gap, error)` as a [`head_checks`] row.
fn head_check_row(gap: i64, no_gap: i64, error: i64) -> Vec<(&'static str, i64)> {
    vec![
        (HEAD_CHECK_RESULT_GAP, gap),
        (HEAD_CHECK_RESULT_NO_GAP, no_gap),
        (HEAD_CHECK_RESULT_ERROR, error),
    ]
}

/// Asserts the subscription ended with `reason` and a retryable
/// `Unavailable` — carrying the shutdown retry hint for the `shutdown`
/// ending and no refusal detail for any other — and returns that status's
/// message.
async fn assert_ends_with(consumer: &LayerStreamConsumer, reason: &str) -> String {
    assert!(
        recv_within(consumer, SETTLE).await.is_none(),
        "the subscription must end"
    );
    let exit = consumer
        .take_terminal_exit()
        .expect("an ending must never read as a clean close");
    assert_eq!(exit.reason, reason);
    crate::common::refusal_detail::mux_ending_message(exit.error, reason)
}

/// A restore band jumps the next commit's id over ids that were never written,
/// while the stream keeps its entries below the jump. The head check finds
/// that retained head at or below the read position: `no_gap`, no marker, and
/// the jumped entry is delivered — the client gap-fills the band.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restore_band_jump_over_retained_entries_is_no_gap_and_is_delivered() {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    five_commits(&tv, 1).await;
    let c = consume(&tv, 1, "5-0").await;
    xadd_update(&tv, 1, RESTORED).await;
    assert_eq!(
        head_of(&tv, 1).await.as_deref(),
        Some("1-0"),
        "precondition: the entries below the jump are retained"
    );

    assert_eq!(recv_seqs(&c, 1).await, [RESTORED]);
    assert_eq!(head_checks(&metrics), head_check_row(0, 1, 0));
    assert!(c.take_terminal_exit().is_none());
}

/// A layer stream removed while subscribed and recreated by a later commit
/// (a restore after a keyspace loss, a stream-key removal) holds nothing below
/// that commit, so the head check reads `gap`. A subscription behind it ends
/// `stream_gap` with the retryable status — an intended resync: its reconnect
/// catches up onto the layer's restored content instead of gap-filling the
/// missing edits as no-ops. A subscription whose next edit IS the recreated
/// stream's first entry (it joined at the restore's seeded sequence,
/// `RESTORED - 1`) gets no marker and receives it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_recreated_stream_ends_the_subscription_behind_it_stream_gap_and_one_at_its_first_entry_continues(
) {
    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    five_commits(&tv, 2).await;
    let behind = consume(&tv, 2, "5-0").await;
    await_readers_parked(tv.client()).await;
    let removed: i64 = tv
        .client()
        .pool
        .unlink(tv.key_schema().layer_stream(2))
        .await
        .expect("UNLINK must succeed");
    assert_eq!(removed, 1, "precondition: the stream existed and is gone");
    let current = consume(&tv, 2, &format!("{}-0", RESTORED - 1)).await;
    xadd_update(&tv, 2, RESTORED).await;

    let message = assert_ends_with(&behind, STREAM_EXIT_STREAM_GAP).await;
    assert_eq!(message, STREAM_GAP_MSG);
    assert_eq!(recv_seqs(&current, 1).await, [RESTORED]);
    assert!(current.take_terminal_exit().is_none());
    assert_eq!(head_checks(&metrics), head_check_row(1, 0, 0));
}

#[cfg(feature = "test-fault-injection")]
mod faults {
    use super::*;
    use fred::error::ErrorKind;
    use live_edit::observability::metrics::STREAM_EXIT_ERROR;
    use live_edit::storage::stream_mux::{MuxFault, MuxFaultSite, StreamMuxKnobs};

    use super::super::mux_test_helpers::{await_mux, fast_backoff_knobs};

    fn head_check_error() -> MuxFault {
        MuxFault::Err(fred::error::Error::new(
            ErrorKind::IO,
            "synthetic head check failure",
        ))
    }

    fn pending_head_check_faults(tv: &TestValkey) -> usize {
        tv.client()
            .stream_mux
            .pending_faults_for_test(MuxFaultSite::HeadCheck)
    }

    /// One head check that fails (`fault`) or does not answer holds its stream
    /// out of the reply; the stream is re-read, checked again and delivered
    /// exactly once. The two checks — one `error`, then one `no_gap` — are
    /// the re-read.
    async fn a_failed_head_check_loses_nothing(fault: MuxFault, object_id: u64) {
        let tv = TestValkey::setup_with_readers(1).await;
        let metrics = install_metrics(&tv).await;
        tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
            probe_timeout: Duration::from_millis(100),
            ..fast_backoff_knobs(tv.client())
        });
        five_commits(&tv, object_id).await;
        let c = consume(&tv, object_id, "5-0").await;
        tv.client()
            .stream_mux
            .push_faults_for_test(MuxFaultSite::HeadCheck, [fault]);
        xadd_update(&tv, object_id, RESTORED).await;

        assert_eq!(recv_seqs(&c, 1).await, [RESTORED]);
        assert_eq!(pending_head_check_faults(&tv), 0);
        assert_eq!(head_checks(&metrics), head_check_row(0, 1, 1));
        super::super::mux_test_helpers::assert_quiet(&c, QUIET, "delivered exactly once").await;
        assert!(c.take_terminal_exit().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_head_check_error_loses_nothing_and_the_stream_is_re_read() {
        a_failed_head_check_loses_nothing(head_check_error(), 3).await;
    }

    /// A check that never answers ends at the knobs' `probe_timeout` as a
    /// failed one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_head_check_that_never_answers_times_out_and_the_stream_is_re_read() {
        a_failed_head_check_loses_nothing(MuxFault::Hang, 4).await;
    }

    /// Subscribes `A` behind a jump on object `a` and `B` on the empty stream
    /// of object `b`, one reader parked over both, then queues `failures`
    /// head-check errors and writes the jump. Every read then returns the
    /// jump, so each failure is one more consecutive transient failure: no
    /// read in between can succeed and reset the budget.
    async fn fail_head_checks(
        tv: &TestValkey,
        failures: usize,
        (a, b): (u64, u64),
    ) -> (LayerStreamConsumer, LayerStreamConsumer) {
        tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
            max_transient_failures: 3,
            ..fast_backoff_knobs(tv.client())
        });
        five_commits(tv, a).await;
        let behind_the_jump = consume(tv, a, "5-0").await;
        let bystander = consume(tv, b, "0-0").await;
        await_mux(tv.client(), "the reader reads both streams", |s| {
            s.readers[0].keys == 2
        })
        .await;
        await_readers_parked(tv.client()).await;
        tv.client().stream_mux.push_faults_for_test(
            MuxFaultSite::HeadCheck,
            (0..failures).map(|_| head_check_error()),
        );
        xadd_update(tv, a, RESTORED).await;
        (behind_the_jump, bystander)
    }

    /// A reply with a failed head check is one transient failure, and the
    /// successful XREAD that carried it does not reset the budget: with a
    /// budget of 3, the fourth failed check in a row ends every subscription
    /// on the reader `error` — the bystander on another stream included —
    /// naming the head check. (Production's budget is 60,
    /// `MAX_CONSECUTIVE_TRANSIENT_FAILURES`, pinned in `reader.rs`.)
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_head_checks_charge_the_transient_budget_without_a_reset_until_it_ends_the_reader(
    ) {
        let tv = TestValkey::setup_with_readers(1).await;
        let metrics = install_metrics(&tv).await;
        let (behind_the_jump, bystander) = fail_head_checks(&tv, 4, (5, 6)).await;

        let message = assert_ends_with(&behind_the_jump, STREAM_EXIT_ERROR).await;
        assert!(
            message.contains(
                "gave up after 4 consecutive transient failures (last: a stream head check \
                 failed)"
            ),
            "got {message:?}"
        );
        assert_ends_with(&bystander, STREAM_EXIT_ERROR).await;
        assert_eq!(pending_head_check_faults(&tv), 0);
        assert_eq!(head_checks(&metrics), head_check_row(0, 0, 4));
    }

    /// The boundary: as many failed checks in a row as the budget holds end
    /// nothing, and the next check delivers the entry.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_head_checks_up_to_the_budget_end_nothing_and_the_entry_is_delivered() {
        let tv = TestValkey::setup_with_readers(1).await;
        let metrics = install_metrics(&tv).await;
        let (behind_the_jump, bystander) = fail_head_checks(&tv, 3, (7, 8)).await;

        assert_eq!(recv_seqs(&behind_the_jump, 1).await, [RESTORED]);
        assert_eq!(head_checks(&metrics), head_check_row(0, 1, 3));
        xadd_update(&tv, 8, 1).await;
        assert_eq!(recv_seqs(&bystander, 1).await, [1]);
        assert!(behind_the_jump.take_terminal_exit().is_none());
        assert!(bystander.take_terminal_exit().is_none());
    }

    /// Waits until every queued head-check fault has been taken by a check.
    async fn await_head_check_faults_taken(tv: &TestValkey) {
        let deadline = tokio::time::Instant::now() + SETTLE;
        while pending_head_check_faults(tv) > 0 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "no head check took the queued fault"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// The head checks race the reader's cancellation: a shutdown while a
    /// check hangs — here under a 600 s probe timeout — stops the reader inside
    /// the 1 s grace, and the subscription ends `shutdown`, not after the
    /// probe timeout and not `error`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_during_a_hung_head_check_ends_the_subscription_shutdown_inside_the_grace() {
        use crate::grpc::log_capture::capture_debug;
        use live_edit::observability::metrics::STREAM_EXIT_SHUTDOWN;

        let tv = TestValkey::setup_with_readers(1).await;
        tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
            probe_timeout: Duration::from_secs(600),
            ..tv.client().stream_mux.knobs()
        });
        five_commits(&tv, 10).await;
        let c = consume(&tv, 10, "5-0").await;
        tv.client()
            .stream_mux
            .push_faults_for_test(MuxFaultSite::HeadCheck, [MuxFault::Hang]);
        xadd_update(&tv, 10, RESTORED).await;
        await_head_check_faults_taken(&tv).await;

        let ((), captured) =
            capture_debug(tv.client().stream_mux.shutdown(Duration::from_secs(1))).await;
        assert!(
            captured
                .with_message("stream multiplexer readers did not stop within the shutdown grace")
                .is_empty(),
            "the reader must stop inside the grace, not after its hung head check; full \
             capture:\n{}",
            captured.text
        );
        assert_ends_with(&c, STREAM_EXIT_SHUTDOWN).await;
    }

    /// The DEBUG line a reader logs when it backs off after a failed head
    /// check.
    const HEAD_CHECK_BACKOFF_MSG: &str =
        "backing off before re-reading streams whose head check failed";

    /// A reply with a failed head check backs off one transient window before
    /// the stream is re-read, rather than re-reading at once — a check that
    /// keeps failing would otherwise spin XREAD → XRANGE until the budget is
    /// spent. One DEBUG line names the first consecutive failure and its
    /// window; while the reader backs off (an hour here) no read is issued and
    /// nothing is delivered.
    ///
    /// Current-thread, so [`ThreadCapture`] sees the reader task's DEBUG line.
    /// The no-read check is skipped, with a note, on the rare run whose
    /// full-jitter draw lands inside the observation window.
    #[tokio::test]
    async fn a_failed_head_check_backs_off_before_the_stream_is_re_read() {
        use crate::grpc::log_capture::{purge_keys, ThreadCapture};
        const HOUR: Duration = Duration::from_secs(3_600);

        let tv = TestValkey::setup_with_readers(1).await;
        let metrics = install_metrics(&tv).await;
        tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
            backoff_initial: HOUR,
            backoff_cap: HOUR,
            ..tv.client().stream_mux.knobs()
        });
        let capture = ThreadCapture::start_debug();
        five_commits(&tv, 11).await;
        let c = consume(&tv, 11, "5-0").await;
        tv.client()
            .stream_mux
            .push_faults_for_test(MuxFaultSite::HeadCheck, [head_check_error()]);
        xadd_update(&tv, 11, RESTORED).await;

        let deadline = tokio::time::Instant::now() + SETTLE;
        let line = loop {
            let captured = capture.snapshot();
            if let Some(line) = captured.with_message(HEAD_CHECK_BACKOFF_MSG).first() {
                break (*line).clone();
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "no `{HEAD_CHECK_BACKOFF_MSG}` line; full capture:\n{}",
                captured.text
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        };
        assert_eq!(line["fields"]["consecutive_failures"], 1, "{line}");
        assert_eq!(
            line["fields"]["backoff_window_ms"],
            HOUR.as_millis() as u64,
            "{line}"
        );
        let sleep_ms = line["fields"]["sleep_ms"]
            .as_u64()
            .expect("sleep_ms is a number");
        if sleep_ms > 4 * QUIET.as_millis() as u64 {
            let xreads = tv.client().stream_mux.snapshot().xreads();
            super::super::mux_test_helpers::assert_quiet(
                &c,
                QUIET,
                "nothing is re-read while the reader backs off",
            )
            .await;
            assert_eq!(tv.client().stream_mux.snapshot().xreads(), xreads);
            assert_eq!(head_checks(&metrics), head_check_row(0, 0, 1));
        } else {
            eprintln!(
                "the full-jitter backoff drew {sleep_ms} ms, inside the observation window; the \
                 no-read check is skipped this run"
            );
        }
        assert_eq!(
            capture
                .snapshot()
                .with_message(HEAD_CHECK_BACKOFF_MSG)
                .len(),
            1,
            "one failed check, one backoff"
        );
        drop(capture);
        purge_keys(&tv).await;
    }
}

/// A `Deleted` sentinel (auto id, so always past the entry after the read
/// position) is never a gap suspect, even when the stream was trimmed down to
/// it: the reader runs no head check and delivers it, and the handler
/// re-bootstraps. Were the sentinel checked like an update, the trimmed head
/// would read `gap` and end the subscription `stream_gap` in place of the
/// re-bootstrap.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_deleted_sentinel_past_a_trimmed_head_is_delivered_without_a_head_check() {
    use fred::types::streams::XCap;
    use live_edit::storage::notifications::LayerNotification;

    let tv = TestValkey::setup_with_readers(1).await;
    let metrics = install_metrics(&tv).await;
    five_commits(&tv, 12).await;
    let key = tv.key_schema().layer_stream(12);
    let _: String = tv
        .client()
        .pool
        .xadd(&key, false, None, "*", vec![("payload", "deleted")])
        .await
        .expect("XADD of the sentinel must succeed");
    let _: u64 = tv
        .client()
        .pool
        .xtrim(
            &key,
            XCap::try_from(("MAXLEN", "=", 1)).expect("build the MAXLEN cap"),
        )
        .await
        .expect("XTRIM must succeed");
    let head = head_of(&tv, 12).await.expect("the sentinel is retained");
    assert_ne!(
        head, "5-0",
        "precondition: the stream holds only the sentinel"
    );

    let c = consume(&tv, 12, "5-0").await;
    match recv_within(&c, SETTLE).await {
        Some((id, LayerNotification::Deleted)) => assert_eq!(id, head),
        other => panic!("the sentinel must be delivered; got {other:?}"),
    }
    assert!(c.take_terminal_exit().is_none());
    assert_eq!(head_checks(&metrics), head_check_row(0, 0, 0));
}
