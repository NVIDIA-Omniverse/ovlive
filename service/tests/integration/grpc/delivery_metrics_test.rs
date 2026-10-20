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

// End-to-end coverage for the SubscribeReadLayer-side delivery instruments,
// driven through real `handle_subscribe_read_layer` / `handle_update_layer` /
// `handle_create_layer` calls against Valkey and read off the service's own
// Prometheus exporter:
//
//   * `live_subscriber_updates_total{endpoint="SubscribeReadLayer", service}` —
//     once per steady-state commit delivered to a subscriber; never for
//     catch-up content (including a re-bootstrap's full state), self-echo, or a
//     commit whose send never happened.
//   * `live_edit_join_latency_seconds{endpoint="SubscribeReadLayer", service}` —
//     once per subscription that reaches its first caught-up ack; never on
//     failure, and never again for a steady-state re-bootstrap's ack.
//   * `live_edit_subscriber_delivery_lag_seconds{stage}` — the commit's age when
//     a live delivery's last message is enqueued (`enqueued`) and when it is
//     polled out of the response stream (`transport`), one window maximum per
//     subscription per window: exactly the deliveries
//     `live_subscriber_updates_total` counts, never catch-up content or
//     self-echo. And the rate-limited "subscriber delivery is lagging" INFO line
//     that names a lagging stream.
//
// The lag tests that need a commit of a known age XADD an `update` entry
// straight onto the layer's stream at the next sequence, with a synthetic
// `server_ts` (30 s ago, an hour ahead, or `0`), so the subscriber delivers it
// as it would a real commit: relayed when the entry carries an `ompf` field, as
// the empty delta when it does not (an empty commit). For the histogram tests
// the lower bound on the lag therefore comes from the stamp, never from
// wall-clock speed. The lagging line counts only time since the subscription
// entered steady state, so its two tests get their `lag_ms` floor from holding
// steady state for `STEADY_STATE_HELD` before the commit arrives.
//
// The slow-reader cases — "send never completed", and the send-stall counter
// and wait histogram — and the admission-refusal case live in
// `subscribe_admission_test.rs`, beside the fixture that can change the outer
// send deadline and saturate the pool.
//
// ── Reading a batched counter ────────────────────────────────────────────────
//
// A subscription publishes its deliveries to `live_subscriber_updates_total`,
// and its `enqueued` lag window, in batches
// (`grpc::subscriber_delivery::DeliveryTally`): on the idle timer, at the batch
// threshold, before a steady-state re-bootstrap, and when the subscription
// ends. Its `transport` window is published by the response stream itself, at
// the latest when the stream is dropped. So "+N after N commits" is read AFTER
// ending the stream, once
// `live_edit_subscriber_stream_exits_total{reason="client_disconnect"}` has
// ticked: that exit is recorded after the subscription task's frame — and the
// tally with it — has dropped, so the count read then is final. A zero read at
// that point is therefore sound too, not merely "not yet". The one exception is
// `subscriber_updates_are_published_by_the_idle_flush_while_the_stream_stays_open`,
// which polls the count while the stream is still open because the idle timer
// is its subject.
//
// ── Wait budget ──────────────────────────────────────────────────────────────
//
// Every wait for a message or a condition is a deadline: `SETTLE` (10 s) per
// message and per polled condition (`await_condition` sleeps 10 ms between
// probes). The heaviest test here reads at most ~16 messages and polls three
// conditions, so its worst case at full budget is well inside nextest's 120 s
// hang-killer, and the healthy path finishes in well under a second — except
// the two lagging-line tests, which take about 1.5 s and 3 s. The only sleeps
// outside a deadline's probe are theirs: the 1.5 s `STEADY_STATE_HELD`, a lower
// bound on time spent in steady state, paid in full — once in
// `a_lagging_subscription_logs_one_info_line_naming_the_stream`, twice in
// `a_rebootstrap_restarts_the_lagging_lines_steady_state_clock`.

use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use fred::interfaces::StreamsInterface;
use futures::{Stream, StreamExt};
use tonic::{Request, Status};

use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, update_layer_request, SubscribeReadLayerHeader,
    SubscribeReadLayerRequest, SubscribeReadLayerResponse, UpdateLayerHeader, UpdateLayerRequest,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::subscriber_delivery::{LAGGING_LOG_MESSAGE, LAGGING_LOG_THRESHOLD};
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{histogram_bucket, render, sample_value};
use live_edit::observability::metrics::{
    Metrics, DELIVERY_LAG_STAGES, DELIVERY_LAG_STAGE_ENQUEUED, DELIVERY_LAG_STAGE_TRANSPORT,
    SEND_PHASES, STREAM_EXIT_CLIENT_DISCONNECT, STREAM_EXIT_ERROR,
};
use live_edit::storage::notifications::{NOTIFICATION_PAYLOAD_FIELD, RELAY_OMPF_FIELD};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};

use crate::common::owned_delta::delta_to_ompf_bytes;

use super::helpers::*;
use super::log_capture::{purge_keys, ThreadCapture};

/// Upper bound on any single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// `build_service`'s `service` label.
const SERVICE: &str = "test";

/// The subscriber's own client id; commits carrying it are self-echo.
const SUBSCRIBER: &str = "delivery-metrics-subscriber";

/// Another client's id; commits carrying it are delivered to the subscriber.
const WRITER: &str = "delivery-metrics-writer";

/// How long a lagging-line test keeps its subscription in steady state before
/// a commit arrives: past the 1 s trigger with margin. A sleep used as a lower
/// bound on elapsed time, never to coordinate anything.
const STEADY_STATE_HELD: Duration = Duration::from_millis(1_500);

// ── Request and commit helpers ──────────────────────────────────────────────

fn subscribe_request(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    sequence_num: u64,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
        object_id,
        sequence_num,
        client_id: SUBSCRIBER.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// A delta adding one fresh section.
fn section_delta(section_id: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id,
            section_name: format!("sec-{section_id}"),
            section_type: 0,
            section_order: 0,
        })],
        ..Default::default()
    }
}

/// One UpdateLayer from `client_id` adding a fresh section, so every commit
/// produces a non-empty delivery.
async fn commit(
    svc: &LiveEditService,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    client_id: &str,
    section_id: u64,
) {
    let delta = section_delta(section_id);
    let header = UpdateLayerRequest {
        message_type: Some(update_layer_request::MessageType::Header(
            UpdateLayerHeader {
                session_id: session_id.to_owned(),
                layer_name: layer_name.to_owned(),
                object_id,
                timestamp_microseconds_since_epoch: section_id * 10,
                client_id: client_id.to_owned(),
                operation_id: String::new(),
            },
        )),
    };
    let stream = streaming_update(vec![
        header,
        make_update_data(delta_to_ompf_bytes(&delta, &[])),
    ]);
    handle_update_layer(svc, make_request(stream))
        .await
        .unwrap_or_else(|e| panic!("UpdateLayer (section {section_id}) failed: {e}"));
}

/// A CreateLayer on an existing (session, layer) — an overwrite onto a new
/// object — carrying one section so the new generation's full state is
/// non-empty.
async fn overwrite_layer(svc: &LiveEditService, session_id: &str, layer_name: &str) -> u64 {
    let delta = OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: 9_001,
            section_name: "overwrite".to_owned(),
            section_type: 0,
            section_order: 0,
        })],
        ..Default::default()
    };
    let stream = streaming_create(vec![
        make_create_header(session_id, layer_name),
        make_create_data(delta_to_ompf_bytes(&delta, &[])),
    ]);
    handle_create_layer(svc, make_request(stream))
        .await
        .expect("CreateLayer overwrite must succeed")
        .into_inner()
        .object_id
}

/// The reading replica's clock in microseconds since the epoch: the scale of a
/// stream entry's `server_ts`.
fn now_us() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("the test host's clock is after 1970")
        .as_micros() as i64
}

/// A commit stamp 30 s before now: older than the `le="20"` lag boundary,
/// whatever the test's own speed. Its age alone does not fire the lagging
/// line, which counts only time since the subscription entered steady state.
fn thirty_seconds_ago() -> i64 {
    now_us() - 30_000_000
}

/// A commit stamp an hour ahead of now — a writer replica whose clock runs
/// ahead of this one — so its lag clamps to zero.
fn an_hour_ahead() -> i64 {
    now_us() + 3_600_000_000
}

/// Which of the steady-state loop's two delivery arms a planted entry takes.
#[derive(Debug, Clone, Copy)]
enum Entry {
    /// An entry with an `ompf` field: the writer's OMPF, relayed verbatim.
    Relayed,
    /// An entry with no `ompf` field: an empty commit, sent as the empty delta
    /// at its sequence.
    EmptyCommit,
}

/// XADD an `update` entry for `sequence_num` straight onto the layer's stream
/// with the given commit stamp, in the shape the commit path writes
/// (`{seq}-0`, `payload`, and for [`Entry::Relayed`] an `ompf` field carrying a
/// fresh section). The steady-state loop delivers it like any other commit;
/// the object's header is not advanced, so each test plants consecutive
/// sequences from the layer's current head and commits nothing after them.
async fn plant_update(
    tv: &TestValkey,
    object_id: u64,
    sequence_num: u64,
    client_id: &str,
    server_ts: i64,
    entry: Entry,
) {
    let mut fields: Vec<(&str, fred::types::Value)> = vec![(
        NOTIFICATION_PAYLOAD_FIELD,
        format!("update|{object_id}|{sequence_num}|{client_id}|0|{server_ts}").into(),
    )];
    if let Entry::Relayed = entry {
        let ompf = Bytes::from(delta_to_ompf_bytes(&section_delta(800 + sequence_num), &[]));
        fields.push((RELAY_OMPF_FIELD, fred::types::Value::Bytes(ompf)));
    }
    let entry_id = format!("{sequence_num}-0");
    let _: String = tv
        .client()
        .pool
        .xadd(
            tv.key_schema().layer_stream(object_id),
            false,
            None,
            entry_id.as_str(),
            fields,
        )
        .await
        .unwrap_or_else(|e| panic!("planting the entry at {entry_id} must succeed: {e}"));
}

/// Read one delivery: a header and, for a non-empty one, its body chunks.
async fn recv_delivery<S>(stream: &mut S, what: &str) -> SubscribeReadLayerHeader
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    let first = tokio::time::timeout(SETTLE, stream.next())
        .await
        .unwrap_or_else(|_| panic!("{what}: no header within {SETTLE:?}"))
        .unwrap_or_else(|| panic!("{what}: the stream ended"))
        .unwrap_or_else(|e| panic!("{what}: error status {e:?}"));
    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("{what}: expected a Header, got {other:?}"),
    };
    let mut seen = 0u64;
    while seen < header.total_size {
        let next = tokio::time::timeout(SETTLE, stream.next())
            .await
            .unwrap_or_else(|_| panic!("{what}: no body chunk within {SETTLE:?}"))
            .unwrap_or_else(|| panic!("{what}: the stream ended mid-body"))
            .unwrap_or_else(|e| panic!("{what}: error status {e:?}"));
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                seen += c.data.len() as u64
            }
            other => panic!("{what}: expected a Chunk, got {other:?}"),
        }
    }
    header
}

/// Drain a catch-up up to and including its caught-up ack (the one header with
/// `total_size == 0`), returning the ack.
async fn drain_catch_up<S>(stream: &mut S, what: &str) -> SubscribeReadLayerHeader
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    loop {
        let header = recv_delivery(stream, what).await;
        if header.total_size == 0 {
            return header;
        }
    }
}

/// The response stream `handle_subscribe_read_layer` returns.
type ResponseStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

/// A subscription that has drained its brand-new catch-up and sits in steady
/// state: where every planted-entry lag test starts.
struct CaughtUp {
    stream: ResponseStream,
    session_id: String,
    object_id: u64,
    /// The caught-up ack; the layer's next sequence is `ack.sequence_num + 1`.
    ack: SubscribeReadLayerHeader,
}

/// Create a session and layer on `svc`, subscribe [`SUBSCRIBER`] to it from
/// version 0, and drain the catch-up through its ack.
async fn subscribe_caught_up(
    tv: &TestValkey,
    svc: &LiveEditService,
    session_name: &str,
    layer_name: &str,
) -> CaughtUp {
    let (session_id, object_id) = create_session_and_layer(tv, svc, session_name, layer_name).await;
    let mut stream =
        handle_subscribe_read_layer(svc, subscribe_request(&session_id, layer_name, 0, 0))
            .await
            .expect("subscribe must succeed")
            .into_inner();
    let ack = drain_catch_up(&mut stream, "brand-new catch-up").await;
    CaughtUp {
        stream,
        session_id,
        object_id,
        ack,
    }
}

// ── Metric readers ──────────────────────────────────────────────────────────
//
// Strict on absence: every counter read here is seeded in `Metrics::new`, and
// its reader fails on an absent cell. The histograms are the exception — none
// has a cell until its first observation — so `join_latency_count`,
// `lag_count`, `lag_bucket` and `backlog_windows` fold absence to 0; each test
// that reads a zero through one of them also reads a non-zero through the same
// family, or a positive control. One counter reader is not strict either:
// `snapshot_reads`, whose cells are seeded but whose sum folds an absent cell to
// 0, is compared only against itself or guarded by a positive control.

fn subscriber_updates(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_subscriber_updates_total",
        &[("endpoint", "SubscribeReadLayer"), ("service", SERVICE)],
    )
    .expect("live_subscriber_updates_total{endpoint=\"SubscribeReadLayer\"} is seeded") as u64
}

/// `_count` of the SubscribeReadLayer join-latency series. Absent until the
/// first observation (histograms are not seeded), so absence reads as 0.
fn join_latency_count(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_join_latency_seconds_count",
        &[("endpoint", "SubscribeReadLayer"), ("service", SERVICE)],
    )
    .unwrap_or(0.0) as u64
}

/// `live_edit_snapshot_read_total` summed over its cells: with a `consumer`
/// (e.g. `catchup`), that consumer's reads; with `None`, every snapshot read
/// of any consumer, path and outcome. A sum over the rendered cells, so a cell
/// not rendered reads as 0. Recorded only on a `Metrics` installed on the
/// Valkey client.
fn snapshot_reads(metrics: &Metrics, consumer: Option<&str>) -> u64 {
    let label = consumer.map(|consumer| format!("consumer=\"{consumer}\""));
    render(metrics)
        .lines()
        .filter(|line| {
            line.starts_with("live_edit_snapshot_read_total{")
                && label.as_ref().is_none_or(|label| line.contains(label))
        })
        .filter_map(|line| line.rsplit_once(' '))
        .map(|(_, value)| {
            value
                .trim()
                .parse::<f64>()
                .expect("a counter sample is a number") as u64
        })
        .sum()
}

fn stream_exits(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_stream_exits_total",
        &[("reason", reason)],
    )
    .expect("every stream-exit reason is seeded") as u64
}

/// Observations on the `stage` cell of the delivery-lag histogram. Not seeded
/// (a histogram), so absence reads as 0; every test reading a zero here also
/// reads a non-zero through the same reader or its sibling below.
fn lag_count(metrics: &Metrics, stage: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_delivery_lag_seconds_count",
        &[("stage", stage)],
    )
    .unwrap_or(0.0) as u64
}

/// Cumulative count of the `stage` cell's bucket `le`; absence reads as 0.
fn lag_bucket(metrics: &Metrics, stage: &str, le: &str) -> u64 {
    histogram_bucket(
        &render(metrics),
        "live_edit_subscriber_delivery_lag_seconds",
        le,
        &[("stage", stage)],
    )
    .unwrap_or(0.0) as u64
}

/// `live_edit_subscriber_send_stalls_total{phase}`; both cells are seeded.
fn send_stalls(metrics: &Metrics, phase: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_send_stalls_total",
        &[("phase", phase)],
    )
    .expect("both send_stalls_total phase cells are seeded at zero") as u64
}

/// Windows published on `live_edit_subscriber_backlog_entries`; absence reads
/// as 0.
fn backlog_windows(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_backlog_entries_count",
        &[],
    )
    .unwrap_or(0.0) as u64
}

/// Poll `condition` until it holds or `SETTLE` elapses. The sleep is the probe
/// interval, not a coordination primitive.
async fn await_condition<F: FnMut() -> bool>(mut condition: F, what: &str) {
    let deadline = Instant::now() + SETTLE;
    while !condition() {
        assert!(
            Instant::now() < deadline,
            "{what} did not hold within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// End the subscription from the client side and wait until its task has
/// finished — see the module header on why this makes the batched counter
/// final.
async fn end_subscription_and_settle<S>(stream: S, metrics: &Metrics) {
    end_nth_subscription_and_settle(stream, metrics, 1).await;
}

/// [`end_subscription_and_settle`] for a test that ends several subscriptions
/// one after another: `nth` is how many `client_disconnect` exits there are
/// once this one has ended.
async fn end_nth_subscription_and_settle<S>(stream: S, metrics: &Metrics, nth: u64) {
    drop(stream);
    await_condition(
        || stream_exits(metrics, STREAM_EXIT_CLIENT_DISCONNECT) == nth,
        "the dropped subscription's task records its client_disconnect exit",
    )
    .await;
}

// ═══════════════════════════════════════════════════════════════════════════
// live_subscriber_updates_total
// ═══════════════════════════════════════════════════════════════════════════

/// N commits from another client leave the subscriber's cell at exactly N —
/// and the brand-new catch-up that preceded them, onto a layer already holding
/// several commits, adds nothing: catch-up content is bootstrap cost, carried
/// by the catch-up histograms, not delivered-update volume.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscriber_updates_count_each_delivered_commit_once_and_not_the_catch_up() {
    const PRE_EXISTING: u64 = 3;
    const DELIVERED: u64 = 5;

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "dm-count-session", "dm-count-layer").await;
    assert_eq!(
        subscriber_updates(&svc.metrics),
        0,
        "the cell is seeded at zero before any subscription"
    );

    for i in 0..PRE_EXISTING {
        commit(
            &svc,
            &session_id,
            "dm-count-layer",
            object_id,
            WRITER,
            100 + i,
        )
        .await;
    }

    let mut stream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, "dm-count-layer", 0, 0))
            .await
            .expect("subscribe must succeed")
            .into_inner();
    let ack = drain_catch_up(&mut stream, "brand-new catch-up").await;
    assert_eq!(
        ack.sequence_num,
        1 + PRE_EXISTING,
        "precondition: the catch-up served the layer's pre-existing commits"
    );

    for i in 0..DELIVERED {
        commit(
            &svc,
            &session_id,
            "dm-count-layer",
            object_id,
            WRITER,
            200 + i,
        )
        .await;
    }
    for i in 0..DELIVERED {
        let header = recv_delivery(&mut stream, "a steady-state delivery").await;
        assert_eq!(header.sequence_num, 2 + PRE_EXISTING + i);
    }

    end_subscription_and_settle(stream, &svc.metrics).await;
    assert_eq!(
        subscriber_updates(&svc.metrics),
        DELIVERED,
        "exactly one tick per steady-state commit delivered: {DELIVERED} commits after catch-up. \
         {PRE_EXISTING} more commits reached the subscriber inside the brand-new catch-up and must \
         not count; a value above {DELIVERED} means catch-up content was counted, below it means \
         a delivery was lost between the tally and the counter"
    );
}

/// The subscriber's own commits never reach it (self-echo suppression), so
/// they must not count either. A commit from another client after them proves
/// the steady-state loop has already processed — and skipped — the echoes by
/// the time the count is read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscriber_updates_do_not_count_the_subscribers_own_commits() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "dm-echo-session", "dm-echo-layer").await;

    let mut stream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, "dm-echo-layer", 0, 0))
            .await
            .expect("subscribe must succeed")
            .into_inner();
    drain_catch_up(&mut stream, "brand-new catch-up").await;

    for i in 0..3 {
        commit(
            &svc,
            &session_id,
            "dm-echo-layer",
            object_id,
            SUBSCRIBER,
            300 + i,
        )
        .await;
    }
    commit(&svc, &session_id, "dm-echo-layer", object_id, WRITER, 400).await;
    let header = recv_delivery(&mut stream, "the other client's commit").await;
    assert_eq!(
        header.sequence_num, 5,
        "the first delivery is the other client's commit (seq 5): the three self-echo commits \
         before it were suppressed"
    );

    end_subscription_and_settle(stream, &svc.metrics).await;
    assert_eq!(
        subscriber_updates(&svc.metrics),
        1,
        "only the other client's commit was delivered; the subscriber's own three commits are \
         self-echo and must not count"
    );
}

/// An empty commit (an UpdateLayer with no chunks, SDD §8.4) has no relay
/// bytes: its stream entry carries no `ompf` field, so the steady-state loop
/// sends it as the empty delta at its sequence — a non-empty OMPF, built with
/// no read of the layer — rather than the writer's OMPF, and that delivery
/// counts exactly like a relayed one. This test pins the empty-commit arm's
/// count and its no-read property;
/// `delivery_lag_of_an_empty_commit_thirty_seconds_old_lands_above_twenty_seconds_in_both_stages`
/// pins its lag stamping. No snapshot
/// read of any kind runs for it: the total of `live_edit_snapshot_read_total`
/// over every consumer is unchanged from before the empty commit to its
/// delivery.
///
/// `send_per_commit_payload`'s zero-byte arm — sent nothing, so not counted —
/// is not driven: no caller builds a zero-byte per-commit payload (the empty
/// delta never is one), and reaching it would take a change to the source
/// rather than any request a client can send.
///
/// Snapshot reads are recorded on the `Metrics` installed on the Valkey
/// client, which `build_service` leaves unwired, so the test installs the
/// service's own `Metrics` there first. The brand-new catch-up's read, counted
/// under `consumer="catchup"`, is the positive control that the unchanged
/// total is read from a `Metrics` that records reads. The total is compared
/// before the relayed commit, whose structural write reads the layer itself.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscriber_updates_count_an_empty_commits_empty_delta_like_a_relayed_one() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    tv.client()
        .install_observability_hooks(Arc::clone(&svc.metrics))
        .await;
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "dm-empty-session", "dm-empty-layer").await;

    let mut stream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, "dm-empty-layer", 0, 0))
            .await
            .expect("subscribe must succeed")
            .into_inner();
    drain_catch_up(&mut stream, "brand-new catch-up").await;
    assert!(
        snapshot_reads(&svc.metrics, Some("catchup")) >= 1,
        "fixture self-check: the brand-new catch-up's read is recorded on the service's Metrics, \
         so the client records snapshot reads there"
    );
    let reads_before_the_empty_commit = snapshot_reads(&svc.metrics, None);

    // `make_update_header`'s client id is `test-client`, not the subscriber's.
    handle_update_layer(
        &svc,
        make_request(streaming_update(vec![make_update_header(
            &session_id,
            "dm-empty-layer",
            object_id,
        )])),
    )
    .await
    .expect("an empty commit must succeed");
    let empty = recv_delivery(&mut stream, "the empty commit's empty delta").await;
    assert_eq!(
        (empty.sequence_num, empty.total_size > 0),
        (2, true),
        "precondition: the empty commit (seq 2) was delivered as a non-empty OMPF"
    );
    // Read before the relayed commit below: a structural commit reads the
    // layer itself (the structural path's unfenced read), which this total
    // would count.
    assert_eq!(
        snapshot_reads(&svc.metrics, None),
        reads_before_the_empty_commit,
        "no snapshot read of any kind ran for the empty commit, its write or its delivery"
    );
    commit(&svc, &session_id, "dm-empty-layer", object_id, WRITER, 700).await;
    let relayed = recv_delivery(&mut stream, "the relayed commit").await;
    assert_eq!(relayed.sequence_num, 3);

    end_subscription_and_settle(stream, &svc.metrics).await;
    assert_eq!(
        subscriber_updates(&svc.metrics),
        2,
        "the empty delta (seq 2) and the relayed delivery (seq 3) each count once"
    );
}

/// A client that has gone away when a commit lands gets nothing sent, so that
/// commit counts nothing: the steady-state loop checks for the dropped
/// receiver ahead of the stream read. The commit is issued while the
/// subscription is still winding down, not after it has ended.
///
/// The discriminating "a send that started but never completed" case — a
/// commit read off the stream whose send then fails — is
/// `subscribe_admission_test::a_commit_whose_send_never_completes_is_not_counted_as_a_subscriber_update`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscriber_updates_do_not_count_a_commit_after_the_client_left() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "dm-gone-session", "dm-gone-layer").await;

    let mut stream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, "dm-gone-layer", 0, 0))
            .await
            .expect("subscribe must succeed")
            .into_inner();
    drain_catch_up(&mut stream, "brand-new catch-up").await;

    drop(stream);
    commit(&svc, &session_id, "dm-gone-layer", object_id, WRITER, 500).await;
    await_condition(
        || stream_exits(&svc.metrics, STREAM_EXIT_CLIENT_DISCONNECT) == 1,
        "the dropped subscription's task records its client_disconnect exit",
    )
    .await;

    assert_eq!(
        subscriber_updates(&svc.metrics),
        0,
        "the subscription ended before the commit landed, so nothing was delivered and nothing \
         may be counted"
    );
}

/// Deliveries reach the counter while the subscription is still OPEN and idle:
/// the steady-state loop's idle flush timer publishes them. Every other test
/// here reads the count after the stream ended or at a re-bootstrap, both of
/// which flush on their own, so this is the only one that fails if the timer
/// arm stops publishing.
///
/// `DELIVERED` stays far below the tally's 64-delivery batch threshold, so the
/// threshold cannot be what publishes them. The final read after teardown
/// proves the timer's flush cleared what it published: `Drop` must not count
/// the same deliveries again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscriber_updates_are_published_by_the_idle_flush_while_the_stream_stays_open() {
    const DELIVERED: u64 = 3;

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "dm-idle-session", "dm-idle-layer").await;

    let mut stream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, "dm-idle-layer", 0, 0))
            .await
            .expect("subscribe must succeed")
            .into_inner();
    drain_catch_up(&mut stream, "brand-new catch-up").await;

    for i in 0..DELIVERED {
        commit(
            &svc,
            &session_id,
            "dm-idle-layer",
            object_id,
            WRITER,
            600 + i,
        )
        .await;
    }
    for i in 0..DELIVERED {
        let header = recv_delivery(&mut stream, "a steady-state delivery").await;
        assert_eq!(header.sequence_num, 2 + i);
    }

    await_condition(
        || subscriber_updates(&svc.metrics) == DELIVERED,
        "the idle flush timer publishes the open subscription's pending deliveries",
    )
    .await;
    assert_eq!(
        (
            stream_exits(&svc.metrics, STREAM_EXIT_CLIENT_DISCONNECT),
            stream_exits(&svc.metrics, STREAM_EXIT_ERROR),
        ),
        (0, 0),
        "precondition: the subscription was still open when its deliveries were counted, so no \
         end-of-stream flush can have published them"
    );

    end_subscription_and_settle(stream, &svc.metrics).await;
    assert_eq!(
        subscriber_updates(&svc.metrics),
        DELIVERED,
        "ending the stream must not count again what the idle flush already published"
    );
}

/// Deliveries pending in the tally are published BEFORE a steady-state
/// re-bootstrap's catch-up runs — not left behind it until the loop next goes
/// idle, which a multi-second full-state catch-up can postpone indefinitely —
/// and the stream's end does not count them a second time.
///
/// The re-bootstrap is parked at its pre-full-state seam, i.e. after the
/// `Deleted` notification was handled and before any of the new generation's
/// catch-up content is sent, so the count read while it is parked is the count
/// the re-bootstrap started from.
///
/// Discrimination relies on the steady-state loop's idle flush timer (1 s) not
/// having fired between the first delivery and the overwrite; the healthy path
/// covers that gap in milliseconds. On a runner slow enough to cross it the
/// test still passes, it just stops discriminating on that run.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn steady_state_rebootstrap_publishes_pending_deliveries_before_its_catch_up() {
    const DELIVERED: u64 = 4;

    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let seam = Arc::new(live_edit::grpc::subscribe::CatchUpStateReadSeam::new_before_full_state());
    svc.catch_up_state_read_seam_for_test = Some(Arc::clone(&seam));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "dm-reboot-session", "dm-reboot-layer").await;

    // have_version == current (1) → `up_to_date`: the initial catch-up does not
    // pass through `catch_up_brand_new`, so the seam's only rendezvous is the
    // re-bootstrap's.
    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "dm-reboot-layer", object_id, 1),
    )
    .await
    .expect("subscribe must succeed")
    .into_inner();
    drain_catch_up(&mut stream, "up_to_date catch-up").await;

    for i in 0..DELIVERED {
        commit(
            &svc,
            &session_id,
            "dm-reboot-layer",
            object_id,
            WRITER,
            600 + i,
        )
        .await;
    }
    for _ in 0..DELIVERED {
        recv_delivery(&mut stream, "a steady-state delivery").await;
    }

    let new_object_id = overwrite_layer(&svc, &session_id, "dm-reboot-layer").await;
    assert_ne!(new_object_id, object_id);

    tokio::time::timeout(SETTLE, seam.arrive_at_pre_full_state())
        .await
        .expect("the steady-state re-bootstrap must reach its pre-full-state seam");
    assert_eq!(
        subscriber_updates(&svc.metrics),
        DELIVERED,
        "the {DELIVERED} deliveries made before the overwrite must be on the counter once the \
         re-bootstrap has started, ahead of its full-state catch-up"
    );

    let rebootstrap = recv_delivery(&mut stream, "the re-bootstrap's full state").await;
    assert_eq!(rebootstrap.object_id, new_object_id);
    drain_catch_up(&mut stream, "the re-bootstrap's caught-up ack").await;

    end_subscription_and_settle(stream, &svc.metrics).await;
    assert_eq!(
        subscriber_updates(&svc.metrics),
        DELIVERED,
        "the stream's end must not count the already-published deliveries again, and the \
         re-bootstrap's full state is catch-up content, which never counts"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// live_edit_join_latency_seconds{endpoint="SubscribeReadLayer"}
// ═══════════════════════════════════════════════════════════════════════════

/// A subscription that reaches its first caught-up ack observes one join
/// latency; a steady-state overwrite re-bootstrap, which sends a second
/// caught-up ack on the new object, does not observe again. The histogram is
/// "time to first state", once per RPC.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_latency_is_observed_once_per_subscription_even_across_a_rebootstrap() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "dm-join-session", "dm-join-layer").await;
    assert_eq!(join_latency_count(&svc.metrics), 0);

    let mut stream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "dm-join-layer", object_id, 1),
    )
    .await
    .expect("subscribe must succeed")
    .into_inner();
    drain_catch_up(&mut stream, "up_to_date catch-up").await;
    await_condition(
        || join_latency_count(&svc.metrics) >= 1,
        "the first caught-up ack observes a join latency",
    )
    .await;

    let new_object_id = overwrite_layer(&svc, &session_id, "dm-join-layer").await;
    let rebootstrap = recv_delivery(&mut stream, "the re-bootstrap's full state").await;
    assert_eq!(rebootstrap.object_id, new_object_id);
    let ack = drain_catch_up(&mut stream, "the re-bootstrap's caught-up ack").await;
    assert_eq!(ack.object_id, new_object_id);

    end_subscription_and_settle(stream, &svc.metrics).await;
    assert_eq!(
        join_latency_count(&svc.metrics),
        1,
        "the re-bootstrap's caught-up ack must not observe a second join latency"
    );
}

/// A subscription to a layer that does not exist ends NOT_FOUND and observes
/// nothing: join latency is success-only.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_latency_is_not_observed_for_an_unknown_layer() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("dm-unknown-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let mut stream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, "no-such-layer", 0, 0))
            .await
            .expect("NOT_FOUND is delivered on the stream, not from the handler")
            .into_inner();
    let status = tokio::time::timeout(SETTLE, stream.next())
        .await
        .expect("the stream must end promptly")
        .expect("the stream must yield its status")
        .expect_err("the first item must be the NOT_FOUND status");
    assert_eq!(status.code(), tonic::Code::NotFound);

    await_condition(
        || stream_exits(&svc.metrics, STREAM_EXIT_ERROR) == 1,
        "the failed subscription's task records its exit",
    )
    .await;
    assert_eq!(
        join_latency_count(&svc.metrics),
        0,
        "a subscription that never reached state must not observe a join latency"
    );
}

/// A client that leaves before its first caught-up ack observes nothing: the
/// catch-up's send fails, so there is no "first delivered state" to time.
///
/// The chunk seam parks the brand-new catch-up in its delivery, after its
/// first content chunk is sent and before its caught-up ack. The client drops
/// its stream only once the catch-up is parked there, and then the catch-up is
/// released, so the ending lands on the catch-up's next send. Dropped any
/// earlier, the client would be seen leaving by one of the catch-up's waits —
/// the slot or the read permit — instead, which also ends `client_disconnect`
/// but never reaches a send.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_latency_is_not_observed_when_the_client_leaves_before_caught_up() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let seam = Arc::new(live_edit::grpc::subscribe::CatchUpChunkSeam::new());
    svc.catch_up_chunk_seam_for_test = Some(Arc::clone(&seam));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "dm-leave-session", "dm-leave-layer").await;
    // A non-empty layer, so the catch-up sends content and reaches the seam.
    commit(&svc, &session_id, "dm-leave-layer", object_id, WRITER, 1).await;

    let stream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, "dm-leave-layer", 0, 0))
            .await
            .expect("subscribe must succeed")
            .into_inner();
    tokio::time::timeout(SETTLE, seam.wait_until_parked())
        .await
        .expect("the brand-new catch-up must park after its first content chunk");
    drop(stream);
    tokio::time::timeout(SETTLE, seam.release())
        .await
        .expect("the parked catch-up must accept its release");

    await_condition(
        || stream_exits(&svc.metrics, STREAM_EXIT_CLIENT_DISCONNECT) == 1,
        "the abandoned subscription records its client_disconnect exit",
    )
    .await;
    assert_eq!(
        join_latency_count(&svc.metrics),
        0,
        "the client left before its first caught-up ack, so no join latency may be observed"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// live_edit_subscriber_delivery_lag_seconds and the lagging-stream line
// ═══════════════════════════════════════════════════════════════════════════

/// A subscription that only catches up observes no delivery lag in either
/// stage and stalls no send: catch-up content has no single commit time. A
/// second subscription on the same pod then receives `DELIVERED` live commits
/// from another client, and each stage observes between one window and one per
/// delivery — the positive control that the readers above resolve.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delivery_lag_is_observed_for_live_commits_and_never_for_catch_up_content() {
    const PRE_EXISTING: u64 = 3;
    const DELIVERED: u64 = 5;

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "dm-lag-session", "dm-lag-layer").await;
    for i in 0..PRE_EXISTING {
        commit(
            &svc,
            &session_id,
            "dm-lag-layer",
            object_id,
            WRITER,
            100 + i,
        )
        .await;
    }

    let mut catch_up_only =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, "dm-lag-layer", 0, 0))
            .await
            .expect("subscribe must succeed")
            .into_inner();
    let ack = drain_catch_up(&mut catch_up_only, "brand-new catch-up").await;
    assert_eq!(
        ack.sequence_num,
        1 + PRE_EXISTING,
        "precondition: the catch-up carried the layer's pre-existing commits"
    );
    end_nth_subscription_and_settle(catch_up_only, &svc.metrics, 1).await;
    for stage in DELIVERY_LAG_STAGES {
        assert_eq!(
            lag_count(&svc.metrics, stage),
            0,
            "a catch-up-only subscription must observe no {stage} lag: catch-up content has no \
             single commit time"
        );
    }
    for phase in SEND_PHASES {
        assert_eq!(
            send_stalls(&svc.metrics, phase),
            0,
            "a short catch-up to a reading client stalls no {phase} send"
        );
    }

    let mut live =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, "dm-lag-layer", 0, 0))
            .await
            .expect("subscribe must succeed")
            .into_inner();
    drain_catch_up(&mut live, "second brand-new catch-up").await;
    for i in 0..DELIVERED {
        commit(
            &svc,
            &session_id,
            "dm-lag-layer",
            object_id,
            WRITER,
            200 + i,
        )
        .await;
    }
    for _ in 0..DELIVERED {
        recv_delivery(&mut live, "a steady-state delivery").await;
    }
    end_nth_subscription_and_settle(live, &svc.metrics, 2).await;

    for stage in DELIVERY_LAG_STAGES {
        let observed = lag_count(&svc.metrics, stage);
        assert!(
            (1..=DELIVERED).contains(&observed),
            "{DELIVERED} live deliveries must give between 1 (one window) and {DELIVERED} (one \
             per delivery) {stage} observations; got {observed}"
        );
    }
}

/// The subscriber's own commits are never delivered, so they are never
/// measured. The self-echo entry is stamped 30 s ago and the delivered one an
/// hour ahead: had the echo entered the window, its maximum would sit above
/// `le="0.001"`; instead every observation in both stages is the delivered
/// commit's clamped zero.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delivery_lag_is_not_observed_for_the_subscribers_own_commits() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let CaughtUp {
        mut stream,
        object_id,
        ack,
        ..
    } = subscribe_caught_up(&tv, &svc, "dm-lag-echo-session", "dm-lag-echo-layer").await;

    plant_update(
        &tv,
        object_id,
        ack.sequence_num + 1,
        SUBSCRIBER,
        thirty_seconds_ago(),
        Entry::Relayed,
    )
    .await;
    plant_update(
        &tv,
        object_id,
        ack.sequence_num + 2,
        WRITER,
        an_hour_ahead(),
        Entry::Relayed,
    )
    .await;
    let header = recv_delivery(&mut stream, "the other client's commit").await;
    assert_eq!(
        header.sequence_num,
        ack.sequence_num + 2,
        "precondition: the self-echo entry was suppressed and the other client's delivered"
    );

    end_subscription_and_settle(stream, &svc.metrics).await;
    assert_eq!(subscriber_updates(&svc.metrics), 1);
    for stage in DELIVERY_LAG_STAGES {
        let observed = lag_count(&svc.metrics, stage);
        assert!(observed >= 1, "the delivered commit is measured at {stage}");
        assert_eq!(
            lag_bucket(&svc.metrics, stage, "0.001"),
            observed,
            "every {stage} observation is the delivered commit's clamped zero; one above \
             le=\"0.001\" means the 30 s-old self-echo commit was measured"
        );
    }
}

/// A relayed commit stamped 30 s before it reached the stream is at least 30 s
/// old in both stages: one window each, above `le="20"`, in a finite bucket.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delivery_lag_of_a_relayed_commit_thirty_seconds_old_lands_above_twenty_seconds_in_both_stages(
) {
    assert_thirty_seconds_old_commit_lands_above_twenty_seconds(Entry::Relayed, "dm-lag-old").await;
}

/// The same for an empty commit, which the steady-state loop sends from its
/// other arm, as the empty delta: that arm must stamp the delta's last message
/// for the `transport` stage and record the delivery's `enqueued` lag too.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delivery_lag_of_an_empty_commit_thirty_seconds_old_lands_above_twenty_seconds_in_both_stages(
) {
    assert_thirty_seconds_old_commit_lands_above_twenty_seconds(
        Entry::EmptyCommit,
        "dm-lag-old-empty",
    )
    .await;
}

/// Shared body of the two 30 s-old tests: one entry of shape `entry`, stamped
/// 30 s ago, delivered to a caught-up subscriber.
async fn assert_thirty_seconds_old_commit_lands_above_twenty_seconds(entry: Entry, name: &str) {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let CaughtUp {
        mut stream,
        object_id,
        ack,
        ..
    } = subscribe_caught_up(
        &tv,
        &svc,
        &format!("{name}-session"),
        &format!("{name}-layer"),
    )
    .await;
    plant_update(
        &tv,
        object_id,
        ack.sequence_num + 1,
        WRITER,
        thirty_seconds_ago(),
        entry,
    )
    .await;
    let header = recv_delivery(&mut stream, "the 30 s-old commit").await;
    assert_eq!(
        (header.sequence_num, header.total_size > 0),
        (ack.sequence_num + 1, true),
        "precondition: the {entry:?} entry was delivered with content (an empty commit as the \
         non-empty empty delta)"
    );
    end_subscription_and_settle(stream, &svc.metrics).await;

    for stage in DELIVERY_LAG_STAGES {
        assert_eq!(
            lag_count(&svc.metrics, stage),
            1,
            "one delivery is one {stage} window"
        );
        assert_eq!(
            lag_bucket(&svc.metrics, stage, "20"),
            0,
            "a commit stamped 30 s ago is more than 20 s old at {stage}"
        );
        assert_eq!(
            lag_bucket(&svc.metrics, stage, "120"),
            1,
            "and still lands in a finite bucket"
        );
    }
}

/// A commit stamped by a writer whose clock runs ahead of this replica's
/// records a zero lag in both stages — clamped, never negative or wrapped.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delivery_lag_of_a_commit_stamped_ahead_of_this_replica_lands_in_the_first_bucket() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let CaughtUp {
        mut stream,
        object_id,
        ack,
        ..
    } = subscribe_caught_up(&tv, &svc, "dm-lag-ahead-session", "dm-lag-ahead-layer").await;
    plant_update(
        &tv,
        object_id,
        ack.sequence_num + 1,
        WRITER,
        an_hour_ahead(),
        Entry::Relayed,
    )
    .await;
    recv_delivery(&mut stream, "the commit stamped ahead").await;
    end_subscription_and_settle(stream, &svc.metrics).await;

    for stage in DELIVERY_LAG_STAGES {
        assert_eq!(lag_count(&svc.metrics, stage), 1, "{stage}");
        assert_eq!(
            lag_bucket(&svc.metrics, stage, "0.001"),
            1,
            "a commit stamped an hour ahead records a clamped zero at {stage}"
        );
    }
}

/// A commit with no usable commit time (`server_ts = 0`, what a writer whose
/// clock reads before 1970 stamps) is still a delivery — counted on
/// `live_subscriber_updates_total`, its window's backlog published — but it has
/// no lag to observe, in either stage.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delivery_lag_is_not_observed_for_a_commit_with_no_commit_time() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let CaughtUp {
        mut stream,
        object_id,
        ack,
        ..
    } = subscribe_caught_up(&tv, &svc, "dm-lag-zero-session", "dm-lag-zero-layer").await;
    plant_update(
        &tv,
        object_id,
        ack.sequence_num + 1,
        WRITER,
        0,
        Entry::Relayed,
    )
    .await;
    recv_delivery(&mut stream, "the commit with no commit time").await;
    end_subscription_and_settle(stream, &svc.metrics).await;

    assert_eq!(
        subscriber_updates(&svc.metrics),
        1,
        "the commit was delivered and counts"
    );
    assert_eq!(
        backlog_windows(&svc.metrics),
        1,
        "positive control: the window holding the delivery was published"
    );
    for stage in [DELIVERY_LAG_STAGE_ENQUEUED, DELIVERY_LAG_STAGE_TRANSPORT] {
        assert_eq!(
            lag_count(&svc.metrics, stage),
            0,
            "a commit with no commit time has no {stage} lag; an observation means `now − 0` was \
             recorded"
        );
    }
}

/// A lagging subscription logs the rate-limited INFO line naming it, once.
/// The line is the deliverable — per-stream identity lives here, never in a
/// metric label — so its message, level and every one of its ten fields are
/// pinned.
///
/// The trigger counts lag only from when the subscription entered steady
/// state, so a commit's age alone cannot fire it. The subscription therefore
/// stays caught up for `STEADY_STATE_HELD` — a sleep, used as a lower bound on
/// elapsed time, not to coordinate anything — before two commits each stamped
/// 30 s ago arrive back to back: the first logs, and the second falls inside
/// the 60 s interval and is suppressed. Its `lag_ms` is the steady-state lag:
/// at least `STEADY_STATE_HELD`, and no more than the time since the test
/// subscribed — far below the commits' 30 s age, which is what a line
/// reporting the commit lag would carry.
///
/// The line is emitted by the spawned subscription task, so the capture is the
/// thread-local one on a current-thread runtime (test-writing guide, § What NOT
/// to Test). `TestValkey`'s `Drop` cannot purge there, so the test purges its
/// keys itself.
#[tokio::test]
async fn a_lagging_subscription_logs_one_info_line_naming_the_stream() {
    /// What operators and their load-test log watchers grep for; spelled out so
    /// a rename breaks this test.
    const LAGGING_LINE: &str = "subscriber delivery is lagging";
    assert_eq!(LAGGING_LOG_MESSAGE, LAGGING_LINE);
    assert!(STEADY_STATE_HELD > LAGGING_LOG_THRESHOLD);

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let capture = ThreadCapture::start();
    // On the replica's own clock (`server_ts` scale): steady state is entered
    // after this instant.
    let subscribed_at_us = now_us();
    let CaughtUp {
        mut stream,
        session_id,
        object_id,
        ack,
    } = subscribe_caught_up(&tv, &svc, "dm-lag-log-session", "dm-lag-log-layer").await;
    // The caught-up ack has been received, so steady state began before now.
    tokio::time::sleep(STEADY_STATE_HELD).await;
    let first = ack.sequence_num + 1;
    for sequence_num in [first, first + 1] {
        plant_update(
            &tv,
            object_id,
            sequence_num,
            WRITER,
            thirty_seconds_ago(),
            Entry::Relayed,
        )
        .await;
    }
    recv_delivery(&mut stream, "the first lagging commit").await;
    recv_delivery(&mut stream, "the second lagging commit").await;
    let since_subscribe_ms = (now_us() - subscribed_at_us) as u64 / 1_000;
    end_subscription_and_settle(stream, &svc.metrics).await;

    let captured = capture.snapshot();
    drop(capture);
    purge_keys(&tv).await;

    let line = captured.only(LAGGING_LINE);
    assert_eq!(line["level"], "INFO", "full event: {line}");
    let fields = line["fields"]
        .as_object()
        .expect("a JSON event carries a fields object");
    let mut names: Vec<&str> = fields.keys().map(String::as_str).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        [
            "backlog_bytes",
            "backlog_entries",
            "client_id",
            "lag_ms",
            "lagging_deliveries",
            "layer_name",
            "max_lag_ms",
            "message",
            "object_id",
            "sequence_num",
            "session_id",
        ],
        "the line carries exactly its ten fields; full event: {line}"
    );
    assert_eq!(line["fields"]["session_id"], session_id.as_str());
    assert_eq!(line["fields"]["layer_name"], "dm-lag-log-layer");
    assert_eq!(line["fields"]["client_id"], SUBSCRIBER);
    assert_eq!(line["fields"]["object_id"], object_id);
    assert_eq!(
        line["fields"]["sequence_num"], first,
        "the first lagging delivery is the one that logs"
    );
    let lag_ms = line["fields"]["lag_ms"]
        .as_u64()
        .expect("lag_ms is an integer");
    assert!(
        lag_ms >= STEADY_STATE_HELD.as_millis() as u64,
        "the subscription had been in steady state for at least {STEADY_STATE_HELD:?} when the \
         commit was enqueued; got lag_ms={lag_ms}"
    );
    assert!(
        lag_ms <= since_subscribe_ms,
        "lag_ms is the steady-state lag, so it cannot exceed the {since_subscribe_ms} ms since \
         the test subscribed; a value near the commit's 30 s age means the line reports the \
         commit lag; got lag_ms={lag_ms}"
    );
    assert_eq!(
        line["fields"]["max_lag_ms"], lag_ms,
        "the first line reports only itself"
    );
    assert_eq!(line["fields"]["lagging_deliveries"], 1);
    for depth in ["backlog_entries", "backlog_bytes"] {
        assert!(
            line["fields"][depth].is_u64(),
            "`{depth}` is the sampled backlog depth; full event: {line}"
        );
    }
}

/// A steady-state re-bootstrap restarts the lagging line's steady-state
/// clock: its catch-up, and the steady state before it, do not count toward
/// the trigger.
///
/// The subscription sits caught up for `STEADY_STATE_HELD`, then the layer is
/// overwritten and the subscription re-bootstraps onto the new object. After a
/// second `STEADY_STATE_HELD`, one commit stamped 30 s ago lands on the new
/// object. It logs, and its `lag_ms` is at least the second hold and at most
/// the time since the overwrite. Measured from the subscription's first
/// steady-state entry instead, it would also carry the first hold and exceed
/// that ceiling.
///
/// Current-thread runtime and thread-local capture, as in
/// [`a_lagging_subscription_logs_one_info_line_naming_the_stream`]; the test
/// purges its own keys.
#[tokio::test]
async fn a_rebootstrap_restarts_the_lagging_lines_steady_state_clock() {
    assert!(STEADY_STATE_HELD > LAGGING_LOG_THRESHOLD);

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let capture = ThreadCapture::start();
    let CaughtUp {
        mut stream,
        session_id,
        ..
    } = subscribe_caught_up(&tv, &svc, "dm-lag-reboot-session", "dm-lag-reboot-layer").await;
    tokio::time::sleep(STEADY_STATE_HELD).await;

    // On the replica's own clock (`server_ts` scale): the re-bootstrap's
    // steady state is entered after this instant.
    let overwrite_at_us = now_us();
    let new_object_id = overwrite_layer(&svc, &session_id, "dm-lag-reboot-layer").await;
    let full_state = recv_delivery(&mut stream, "the re-bootstrap's full state").await;
    assert_eq!(
        full_state.object_id, new_object_id,
        "precondition: the subscription re-bootstrapped onto the new object"
    );
    let ack = drain_catch_up(&mut stream, "the re-bootstrap's caught-up ack").await;
    assert_eq!(ack.object_id, new_object_id);

    tokio::time::sleep(STEADY_STATE_HELD).await;
    plant_update(
        &tv,
        new_object_id,
        ack.sequence_num + 1,
        WRITER,
        thirty_seconds_ago(),
        Entry::Relayed,
    )
    .await;
    let delivered = recv_delivery(&mut stream, "the lagging commit on the new object").await;
    let since_overwrite_ms = (now_us() - overwrite_at_us) as u64 / 1_000;
    assert_eq!(delivered.object_id, new_object_id);
    end_subscription_and_settle(stream, &svc.metrics).await;

    let captured = capture.snapshot();
    drop(capture);
    purge_keys(&tv).await;

    let line = captured.only(LAGGING_LOG_MESSAGE);
    assert_eq!(line["fields"]["object_id"], new_object_id);
    let lag_ms = line["fields"]["lag_ms"]
        .as_u64()
        .expect("lag_ms is an integer");
    assert!(
        lag_ms >= STEADY_STATE_HELD.as_millis() as u64,
        "the re-bootstrapped subscription had been in steady state for at least \
         {STEADY_STATE_HELD:?}; got lag_ms={lag_ms}"
    );
    assert!(
        lag_ms <= since_overwrite_ms,
        "lag_ms counts only from the re-bootstrap's caught-up ack, so it cannot exceed the \
         {since_overwrite_ms} ms since the overwrite; a larger value means the steady state \
         before the overwrite was counted too; got lag_ms={lag_ms}"
    );
}
