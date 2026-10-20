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

//! b.xtb stage 4 — `SubscribeReadLayer` over a layer stream that `MAXLEN ~`
//! trimmed past a subscription's position before its stream reader read it
//! (SDD §12.3, PRD FR-2.6), driven end-to-end through
//! `handle_subscribe_read_layer`.
//!
//! What is pinned here:
//!
//! * **A trim nobody covered ends the subscription.** A caught-up subscriber
//!   whose stream reader is stalled while the stream turns over — its next
//!   commit trimmed away — ends `UNAVAILABLE "layer stream was trimmed past
//!   this subscription's position; reconnect to catch up"` under
//!   `reason="stream_gap"`, with one WARN `stage=stream` completion line,
//!   instead of skipping the trimmed commits silently.
//! * **A trim the catch-up covered never does (FR-2.6).** A catch-up held past
//!   the same turnover acks the layer's state after it, so every trimmed
//!   commit is at or below its caught-up floor: it continues, and receives
//!   every commit after its ack exactly once. Held before its atomic read, it
//!   registers its stream only at that read's cut, after the turnover, so its
//!   reader never finds a trim past its position at all; in the first test
//!   `live_edit_stream_mux_head_checks_total{result="gap"}` shows the reader
//!   did see the trim, for the uncovered sibling.
//!
//! # How a turnover is staged
//!
//! The service's `layer_stream_maxlen` is [`MAXLEN`]. `MAXLEN ~` trims only
//! whole stream nodes (`stream-node-max-entries`, 100 by default), so how many
//! commits a turnover takes depends on the server: [`turn_the_stream_over`]
//! commits until `XRANGE key - + COUNT 1` shows the head past the entry after
//! the subscriptions' position, and fails if [`TURNOVER_CAP`] commits never
//! get it there.
//!
//! The reader is stalled by an injected hung XREAD (`MuxFault::Hang`), issued
//! when the first subscription registers. In the first test every turnover
//! commit lands while it hangs; in the second the held catch-up has registered
//! nothing yet, so the reader is idle through the turnover and the hang begins
//! at the catch-up's registration. The writer discipline both turnover tests
//! follow: after the covered
//! catch-up's ack, [`K`] (< [`MAXLEN`]) more commits, so no trim can reach an
//! acked commit; then the writer stops and the reader is released. A writer
//! still running when the reader resumed could trim past an ack for real,
//! which rightly ends `stream_gap`, and "exactly once after the ack" needs a
//! closed set of commits. The release joins a stream on another key: its wake
//! lands on the hung read, which outlives the shortened wake-ack timeout and is
//! force-reconnected, and the retried read sees the turned-over stream.
//!
//! # A second, silent activation condition
//!
//! Every test here drives `MuxFaultSite::Xread` and the
//! `CatchUpStateReadSeam`, so this module exists only under
//! **`test-fault-injection`**; without it the run reports fewer tests.
//!
//! ```text
//! cargo nextest run -p live-edit --test integration \
//!     --features test-support,test-fault-injection,test-checkpoint-fault-injection \
//!     grpc::subscribe_stream_gap_test
//! ```
//!
//! # Wait budget
//!
//! Every wait is a deadline, not a measurement: `SETTLE` (10 s) per awaited
//! stream item, park, polled condition or catch-up ack. Summed at full value
//! the uncovered test's waits — the hung read, the park and its release, two
//! acks, the forced reconnect, the ending, four deliveries (header and body
//! each) and the exit count — pass nextest's 120 s hang-killer, so each test
//! body runs under one [`SCENARIO_BUDGET`] (100 s) and a stuck run fails with
//! its own message. The healthy path takes the turnover's commits — a few
//! hundred milliseconds to a couple of seconds — and pays no budget.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::{Stream, StreamExt};
use tonic::{Code, Request, Status};

use live_edit::grpc::layer::handle_update_layer;
use live_edit::grpc::proto::{
    subscribe_read_layer_response, SubscribeReadLayerHeader, SubscribeReadLayerRequest,
    SubscribeReadLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::{handle_subscribe_read_layer, CatchUpStateReadSeam};
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    Metrics, CONSUMER_PHASE_WAKE_ACK, HEAD_CHECK_RESULT_GAP, STREAM_EXIT_REASONS,
    STREAM_EXIT_STREAM_GAP,
};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::notifications::STREAM_GAP_MSG;
use live_edit::storage::stream_mux::{MuxFault, MuxFaultSite, StreamMuxKnobs};
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::{consume_layer_stream, LayerStreamConsumer, StreamCursor};
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};

use super::helpers::{
    build_disabled_recovery_hook, build_service, create_session_and_layer, make_request,
    make_update_data, make_update_header, streaming_update, AllowAllPermissions,
};
use super::log_capture::{assert_common_fields, assert_field, purge_keys, ThreadCapture};
use crate::storage::mux_test_helpers::{await_series, fast_backoff_knobs};

// ── Budgets and constants ───────────────────────────────────────────────────

/// Upper bound on one awaited message, park, condition or catch-up.
const SETTLE: Duration = Duration::from_secs(10);

/// Whole-test deadline, below nextest's 120 s hang-killer (module header).
const SCENARIO_BUDGET: Duration = Duration::from_secs(100);

/// The service's `layer_stream_maxlen`: small, so a turnover is a few hundred
/// commits.
const MAXLEN: u64 = 10;

/// Commits written after the covered catch-up's ack, before the reader is
/// released. Fewer than [`MAXLEN`], so `MAXLEN ~` — which always keeps at
/// least that many — cannot trim any of them, or the ack.
const K: u64 = 3;

/// The most commits a turnover may take before the test gives up: ten times
/// what two default stream nodes need.
const TURNOVER_CAP: u64 = 2_000;

/// A stream on another key, whose registration releases the stalled reader.
const RELEASE_OBJECT: u64 = 999_991;

const LAYER: &str = "stream-gap-layer";

type SubStream = Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

// ── Fixture ─────────────────────────────────────────────────────────────────

/// A service over `tv` with `layer_stream_maxlen = MAXLEN`, its metrics also
/// installed on the stream multiplexer (head checks, forced reconnects), and a
/// catch-up parked before its state read.
async fn service(tv: &TestValkey) -> (LiveEditService, Arc<CatchUpStateReadSeam>) {
    let base = build_service(tv, Arc::new(AllowAllPermissions));
    let layer_store = Arc::new(LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        MAXLEN,
    ));
    let recovery =
        build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&base.metrics));
    let seam = Arc::new(CatchUpStateReadSeam::new_before_full_state());
    let svc = LiveEditService {
        layer_store,
        recovery,
        catch_up_state_read_seam_for_test: Some(Arc::clone(&seam)),
        ..base
    };
    tv.client()
        .install_observability_hooks(Arc::clone(&svc.metrics))
        .await;
    (svc, seam)
}

/// Stall the stream reader: its next XREAD hangs until the reader is
/// force-reconnected. The wake-ack timeout is out of reach until
/// [`release_the_reader`] shortens it.
fn stall_the_reader(tv: &TestValkey) {
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        wake_ack_timeout: Duration::from_secs(600),
        ..fast_backoff_knobs(tv.client())
    });
    tv.client()
        .stream_mux
        .push_faults_for_test(MuxFaultSite::Xread, [MuxFault::Hang]);
}

/// Waits until the reader has issued its hung XREAD.
async fn await_the_hung_read(tv: &TestValkey) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while tv
        .client()
        .stream_mux
        .pending_faults_for_test(MuxFaultSite::Xread)
        > 0
    {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the reader never issued the hung XREAD"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Ends the stall: a registration on another key wakes the hung read, which
/// outlives the now-short wake-ack timeout and is force-reconnected; the
/// reader's next read covers every stream. Returns the releasing stream,
/// which the caller keeps registered.
async fn release_the_reader(tv: &TestValkey, metrics: &Metrics) -> LayerStreamConsumer {
    tv.client().stream_mux.set_knobs_for_test(StreamMuxKnobs {
        wake_ack_timeout: Duration::from_millis(50),
        ..tv.client().stream_mux.knobs()
    });
    let release = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        RELEASE_OBJECT,
        StreamCursor::From("0-0".to_owned()),
        tokio::time::Instant::now() + SETTLE,
    )
    .await
    .expect("registration is in-memory and must succeed");
    await_series(
        metrics,
        "live_edit_valkey_consumer_forced_reconnects_total",
        &[("phase", CONSUMER_PHASE_WAKE_ACK)],
        1,
    )
    .await;
    release
}

// ── Commits and the turnover ────────────────────────────────────────────────

/// One commit adding section `n`, from `test-client` (no subscriber here), so
/// self-echo never hides it. Returns its sequence number.
async fn commit(svc: &LiveEditService, session_id: &str, object_id: u64, n: u64) -> u64 {
    let delta = OwnedDelta {
        is_diff: false,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: n,
            section_name: format!("sec-{n}"),
            section_type: 0,
            section_order: 0,
        })],
        ..Default::default()
    };
    let ompf = crate::common::owned_delta::delta_to_ompf_bytes(&delta, &[]);
    Box::pin(handle_update_layer(
        svc,
        make_request(streaming_update(vec![
            make_update_header(session_id, LAYER, object_id),
            make_update_data(ompf),
        ])),
    ))
    .await
    .unwrap_or_else(|e| panic!("UpdateLayer (section {n}) must succeed: {e}"))
    .into_inner()
    .sequence_num
}

/// The id of the stream's first retained entry (`XRANGE key - + COUNT 1`).
async fn stream_head(tv: &TestValkey, object_id: u64) -> u64 {
    use fred::interfaces::StreamsInterface;
    let first: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = tv
        .client()
        .pool
        .next()
        .xrange_values(tv.key_schema().layer_stream(object_id), "-", "+", Some(1))
        .await
        .expect("XRANGE must succeed");
    let (id, _) = first
        .into_iter()
        .next()
        .expect("a layer stream always keeps its newest entry");
    id.split_once('-')
        .and_then(|(ms, _)| ms.parse().ok())
        .expect("layer-stream update ids are `{seq}-0`")
}

/// Commits until `MAXLEN ~` has trimmed the stream past the entry after
/// `position` — its head above `position + 1` — and returns the last
/// commit's sequence number.
async fn turn_the_stream_over(
    tv: &TestValkey,
    svc: &LiveEditService,
    session_id: &str,
    object_id: u64,
    position: u64,
) -> u64 {
    for n in 1..=TURNOVER_CAP {
        let last = commit(svc, session_id, object_id, 10_000 + n).await;
        if n % 25 == 0 && stream_head(tv, object_id).await > position + 1 {
            return last;
        }
    }
    panic!("{TURNOVER_CAP} commits never trimmed the stream past {position}");
}

// ── Stream helpers ──────────────────────────────────────────────────────────

async fn subscribe(
    svc: &LiveEditService,
    session_id: &str,
    object_id: u64,
    sequence_num: u64,
    client_id: &str,
) -> SubStream {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: LAYER.to_owned(),
        object_id,
        sequence_num,
        client_id: client_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("stream-gap-bearer"));
    handle_subscribe_read_layer(svc, request)
        .await
        .unwrap_or_else(|s| panic!("the subscribe must be admitted: {s:?}"))
        .into_inner()
}

async fn next_item(
    stream: &mut SubStream,
    what: &str,
) -> Result<SubscribeReadLayerResponse, Status> {
    tokio::time::timeout(SETTLE, stream.next())
        .await
        .unwrap_or_else(|_| panic!("{what} did not arrive within {SETTLE:?}"))
        .unwrap_or_else(|| panic!("the stream ended cleanly before {what}"))
}

/// One header, its body skipped.
async fn recv_header(stream: &mut SubStream, what: &str) -> SubscribeReadLayerHeader {
    let header = match next_item(stream, what).await {
        Ok(SubscribeReadLayerResponse {
            response_type: Some(subscribe_read_layer_response::ResponseType::Header(h)),
        }) => h,
        other => panic!("{what}: expected a header, got {other:?}"),
    };
    let mut read = 0u64;
    while read < header.total_size {
        match next_item(stream, what).await {
            Ok(SubscribeReadLayerResponse {
                response_type: Some(subscribe_read_layer_response::ResponseType::Chunk(c)),
            }) => read += c.data.len() as u64,
            other => panic!("{what}: expected a body chunk, got {other:?}"),
        }
    }
    header
}

/// The catch-up's caught-up ack (the one `total_size == 0` header), past
/// any content before it.
async fn recv_ack(stream: &mut SubStream, what: &str) -> SubscribeReadLayerHeader {
    tokio::time::timeout(SETTLE, async {
        loop {
            let header = recv_header(stream, what).await;
            if header.total_size == 0 {
                return header;
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what}: no caught-up ack within {SETTLE:?}"))
}

/// Asserts the next [`K`] deliveries are the commits after `ack`, in order,
/// and that the next is a fresh probe commit: nothing was delivered twice.
async fn assert_every_later_commit_once(
    stream: &mut SubStream,
    svc: &LiveEditService,
    session_id: &str,
    object_id: u64,
    ack: u64,
) {
    for n in 1..=K {
        let header = recv_header(stream, "a commit after the ack").await;
        assert_eq!(
            header.sequence_num,
            ack + n,
            "every commit after the ack arrives once, in order, and nothing at or below it"
        );
    }
    let probe = commit(svc, session_id, object_id, 20_000).await;
    assert_eq!(probe, ack + K + 1);
    let header = recv_header(stream, "the probe commit").await;
    assert_eq!(
        header.sequence_num, probe,
        "the next delivery is the probe: no commit came twice"
    );
}

// ── Metric readers ──────────────────────────────────────────────────────────

fn stream_exits(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_stream_exits_total",
        &[("reason", reason)],
    )
    .unwrap_or_else(|| panic!("stream_exits_total{{reason=\"{reason}\"}} is pre-initialised"))
        as u64
}

fn total_stream_exits(metrics: &Metrics) -> u64 {
    STREAM_EXIT_REASONS
        .iter()
        .map(|reason| stream_exits(metrics, reason))
        .sum()
}

fn gap_head_checks(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_stream_mux_head_checks_total",
        &[("result", HEAD_CHECK_RESULT_GAP)],
    )
    .expect("head_checks_total{result=\"gap\"} is pre-initialised") as u64
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// A caught-up subscriber (up to date at `s0`, so its floor is `s0`) whose
/// stalled reader finds the stream trimmed past `s0 + 1` ends `stream_gap`:
/// its first item after its ack is the retryable status, counted once under
/// `stream_gap`, with one WARN `stage=stream` completion line. A brand-new
/// sibling on the same reader, whose catch-up was held past the turnover,
/// registers at its atomic read's cut (`last`, after the turnover) rather
/// than at `s0`, so it covers every trimmed commit: it continues, and receives
/// every commit after its ack exactly once.
///
/// Current-thread, so the thread-local capture sees the line the spawned
/// subscription task logs.
#[tokio::test]
async fn a_trim_past_a_stalled_reader_ends_the_uncovered_subscriber_stream_gap_and_a_covered_sibling_continues(
) {
    let tv = TestValkey::setup_with_readers(1).await;
    let (svc, seam) = service(&tv).await;
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "stream-gap-session", LAYER).await;
    let capture = ThreadCapture::start();
    tokio::time::timeout(
        SCENARIO_BUDGET,
        Box::pin(async {
            let s0 = commit(&svc, &session_id, object_id, 1).await;
            stall_the_reader(&tv);

            let mut uncovered = subscribe(&svc, &session_id, object_id, s0, "gap-uncovered").await;
            let ack = recv_ack(&mut uncovered, "the up-to-date ack").await;
            assert_eq!(ack.sequence_num, s0);
            await_the_hung_read(&tv).await;
            let mut covered = subscribe(&svc, &session_id, 0, 0, "gap-covered").await;
            tokio::time::timeout(SETTLE, seam.wait_until_parked_pre_full_state())
                .await
                .expect("the brand-new catch-up parks before its atomic read, reserved");

            let last = turn_the_stream_over(&tv, &svc, &session_id, object_id, s0).await;
            tokio::time::timeout(SETTLE, seam.arrive_at_pre_full_state())
                .await
                .expect("the parked catch-up accepts its release");
            let covered_ack = recv_ack(&mut covered, "the covered catch-up's ack").await;
            assert_eq!(
                covered_ack.sequence_num, last,
                "the catch-up acks the state after the turnover: its floor covers the trim"
            );
            for n in 1..=K {
                commit(&svc, &session_id, object_id, 30_000 + n).await;
            }
            let _release = release_the_reader(&tv, &svc.metrics).await;

            let status = match next_item(&mut uncovered, "the uncovered subscriber's ending").await
            {
                Err(status) => status,
                Ok(msg) => panic!(
                "a trimmed-away commit must end the subscription before anything is delivered; \
                 got {msg:?}"
            ),
            };
            assert_eq!(status.code(), Code::Unavailable, "got {status:?}");
            assert_eq!(status.message(), STREAM_GAP_MSG);
            assert_every_later_commit_once(&mut covered, &svc, &session_id, object_id, last).await;

            await_series(
                &svc.metrics,
                "live_edit_subscriber_stream_exits_total",
                &[("reason", STREAM_EXIT_STREAM_GAP)],
                1,
            )
            .await;
            assert_eq!(
                total_stream_exits(&svc.metrics),
                1,
                "one exit, the uncovered subscriber's, observed once"
            );
            assert!(gap_head_checks(&svc.metrics) >= 1);
        }),
    )
    .await
    .unwrap_or_else(|_| panic!("the scenario did not finish within {SCENARIO_BUDGET:?}"));

    let captured = capture.snapshot();
    let line = captured.only(super::log_capture::RPC_FAILURE_MSG);
    assert_eq!(
        captured.completion_lines().len(),
        1,
        "one ending, one completion line; full capture:\n{}",
        captured.text
    );
    assert_common_fields(line, "WARN", "SubscribeReadLayer", "stream", "UNAVAILABLE");
    assert_field(line, "reason", STREAM_EXIT_STREAM_GAP.into());
    assert_field(line, "status_message", STREAM_GAP_MSG.into());
    assert_field(line, "caught_up", true.into());
    drop(capture);
    purge_keys(&tv).await;
}

/// FR-2.6: a catch-up held past a turnover — parked before its atomic read,
/// holding only a reader reservation, while the stream trims past the header
/// sequence it read first — acks the layer's state after the turnover and
/// registers its stream at that same cut. Its reader, stalled once the stream
/// registers so the commits after the ack form a closed set, finds no trim past
/// its position (`head_checks_total{result="gap"}` stays 0), so the
/// subscription never ends `stream_gap`: it receives every commit after its
/// ack exactly once.
///
/// Registered at the header sequence it read first instead, the stream would
/// start below the trim, and the reader would find the gap.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_catch_up_held_past_a_turnover_ends_caught_up_with_no_stream_gap_and_every_later_commit_once(
) {
    let tv = TestValkey::setup_with_readers(1).await;
    let (svc, seam) = service(&tv).await;
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "stream-gap-covered-session", LAYER).await;
    tokio::time::timeout(SCENARIO_BUDGET, async {
        let first_read = commit(&svc, &session_id, object_id, 1).await;
        stall_the_reader(&tv);

        let mut stream = subscribe(&svc, &session_id, 0, 0, "gap-held-catch-up").await;
        tokio::time::timeout(SETTLE, seam.wait_until_parked_pre_full_state())
            .await
            .expect("the brand-new catch-up parks before its atomic read, reserved");
        assert_eq!(
            svc.valkey.stream_mux.snapshot().subscriptions(),
            1,
            "the parked catch-up holds its reader reservation"
        );

        let last = turn_the_stream_over(&tv, &svc, &session_id, object_id, first_read).await;
        tokio::time::timeout(SETTLE, seam.arrive_at_pre_full_state())
            .await
            .expect("the parked catch-up accepts its release");
        let ack = recv_ack(&mut stream, "the held catch-up's ack").await;
        assert_eq!(
            ack.sequence_num, last,
            "the catch-up acks the state after the turnover"
        );
        // Its registration at the cut issued the reader's first, hung, XREAD.
        await_the_hung_read(&tv).await;
        for n in 1..=K {
            commit(&svc, &session_id, object_id, 30_000 + n).await;
        }
        let _release = release_the_reader(&tv, &svc.metrics).await;

        assert_every_later_commit_once(&mut stream, &svc, &session_id, object_id, last).await;
        assert_eq!(
            gap_head_checks(&svc.metrics),
            0,
            "the stream starts at the catch-up's cut, after the turnover, so its reader finds no \
             trim past its position"
        );
        assert_eq!(stream_exits(&svc.metrics, STREAM_EXIT_STREAM_GAP), 0);
        assert_eq!(
            total_stream_exits(&svc.metrics),
            0,
            "the subscription is still open"
        );
    })
    .await
    .unwrap_or_else(|_| panic!("the scenario did not finish within {SCENARIO_BUDGET:?}"));
}
