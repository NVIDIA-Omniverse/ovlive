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

//! Graceful shutdown ends live streams **retryably**, and **paced** (b.e1h,
//! b.kgb; SDD §5.6, §12.6).
//!
//! The Kit client never retries gRPC `OK`, so a stream that a draining pod ends
//! `OK` is a live feed silently lost on every rolling restart. Shutdown
//! reaches the streaming RPCs only through the graceful-shutdown handle — each
//! stream's enrolled child of the service shutdown token, never a per-session
//! token, which means "the session is gone" — and ends them `UNAVAILABLE
//! "server shutting down"` with the `SERVER_SHUTTING_DOWN` retry hint, which
//! the client retries with the position it holds. The endings are paced: from
//! just behind the GOAWAY (t0) the streams end in shuffled batches at
//! `LIVE_EDIT_SHUTDOWN_STREAM_END_RATE`, so their reconnects reach the other
//! replicas as a ramp rather than a step.
//!
//! This file owns the cross-cutting cases:
//!
//! * an **idle, caught-up** subscriber parked on a quiet layer, which used to
//!   end `OK` (or not at all) — alone, and under the old fan-out order in which
//!   shutdown also cancelled every session token;
//! * a `JoinSession` whose client has **stopped reading**, which a shutdown, a
//!   deletion or both must still end, and whose final status must still fit,
//!   after teardown, without the client draining anything;
//! * a subscription and a join the paced ending has **not reached yet** whose
//!   session is deleted: they end as a deletion, not as a shutdown;
//! * the **real server** wiring: streams keep serving through the drain and end
//!   `UNAVAILABLE` starting just behind the GOAWAY, and the server returns; the
//!   **ramp** many streams end in; the listening socket **closed** when the
//!   drain ends; and the **bound after GOAWAY** on a server one of whose
//!   clients stopped reading (b.uqx);
//! * the **reconnect**: a reference client that retries only `UNAVAILABLE`
//!   moves to a second replica and loses nothing written in between — in
//!   window, out of window, and for a join;
//! * the **pool quit** that ends the sequence, on a multi-shard cluster: not
//!   held up by the drained subscriptions' server-side reads on any shard
//!   (cluster-gated; the single-node case is `subscribe_admission_test.rs`'s
//!   tracker-drain test).
//!
//! The paced ending's exact schedule and switches, the enrollment registry and
//! the bound's arithmetic are pinned on paused time in the unit tests of
//! `grpc::transport` and `grpc::graceful_shutdown`.
//!
//! The per-phase endings (admission, mid-catch-up, mid-delivery, mid in-window
//! replay) and their deletion twins live in `subscribe_admission_test.rs`; the
//! completion lines in `stream_completion_log_test.rs`; the join's own shutdown
//! arms in `join_session_cleanup_test.rs` and `join_lifecycle_test.rs`.
//!
//! # Wait budget
//!
//! Every wait is a deadline: `SETTLE` (10 s) per awaited message, whole
//! catch-up read, condition or drain, `SERVER_EXIT` (15 s) for the real server
//! to return. The healthy path pays none of it. The heaviest tests —
//! `a_subscriber_reconnecting_out_of_window_after_shutdown_converges_to_a_fresh_subscribers_state`,
//! about a dozen waits plus three drains, the cluster quit test, one
//! catch-up per subscription plus a drain, and the ramp test, forty streams'
//! setup plus their endings — could exceed nextest's 120 s hang-killer at
//! every budget's full value, so their bodies run under one `SCENARIO_BUDGET`
//! (100 s) deadline, and a stuck run fails with its own message rather than a
//! silent SIGKILL.
//!
//! Three tests take real time on purpose, each a few seconds: the ramp test's
//! paced ending runs its whole 2 s window (forty streams at 20/s); the bound
//! test waits out the bound after GOAWAY, which is at least the 5 s
//! `STREAM_CLOSE_GRACE` (a constant) past a 200 ms test RPC deadline; and the
//! listener test's open stream ends in its batch at t0 + 0.9 s.
//!
//! The measurements are lower bounds or generous on purpose. The cluster quit
//! test times nothing: the failure it rules out — a `QUIT` queued behind a
//! blocked `XREAD` — overruns the 2 s `QUIT_TIMEOUT` and logs the reader
//! pool's quit-timeout WARN, so it asserts that WARN's absence. The ramp test's
//! bounds only separate a ramp from a step; exact batch sizes and instants
//! are pinned on paused time in `grpc::transport`'s tests. Load does not only
//! spread the endings: the paced ending sleeps until each batch's instant, so
//! after a stall longer than Δ the overdue batches fire back to back, and a
//! late client observation can move an ending into the next 100 ms bucket.
//! So the ramp test requires only that the endings span at least half the
//! paced window, that no 100 ms bucket holds more than a quarter of them, and
//! that they fall into at least half as many buckets as there are batches;
//! today's single cancel puts all forty in one bucket within milliseconds.
//! The bound test requires the server to be given up on no earlier than its
//! bound, and no later than 10 s after it.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use futures::{Stream, StreamExt};
use tokio_util::sync::CancellationToken;
use tonic::{Code, Request, Status};

use live_edit::domain::delta::apply_delta;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::ObjectState;
use live_edit::grpc::graceful_shutdown::{StreamPacing, STREAM_END_BATCH_INTERVAL};
use live_edit::grpc::layer::handle_update_layer;
use live_edit::grpc::proto::{
    join_session_response, subscribe_read_layer_response, SubscribeReadLayerHeader,
    SubscribeReadLayerRequest, SubscribeReadLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::transport::{
    drain_stream_tasks, end_streams_after_drain, STREAM_END_DELAY_AFTER_GOAWAY,
};
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    Metrics, STREAM_EXIT_REASONS, STREAM_EXIT_SESSION_CANCEL, STREAM_EXIT_SHUTDOWN,
};
use live_edit::presence::PresenceEvent;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, StructCommand};
use rand::SeedableRng;

use super::helpers::*;
use super::log_capture::{
    assert_common_fields, assert_field, purge_keys, ThreadCapture, RPC_FAILURE_MSG,
};
use crate::common::refusal_detail::{assert_no_refusal_detail, assert_shutdown_status};

// =============================================================================
// Budgets and builders
// =============================================================================

/// Upper bound on any single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// Upper bound on the real server returning once its streams have ended.
const SERVER_EXIT: Duration = Duration::from_secs(15);

/// Whole-test deadline for the multi-phase reconnect tests, below nextest's
/// 120 s hang-killer.
const SCENARIO_BUDGET: Duration = Duration::from_secs(100);

/// The one layer every test here uses.
const LAYER: &str = "e1h-layer";

/// Bearer on every in-process request; the mock permission client ignores it.
const BEARER: &str = "e1h-bearer";

/// Delivery capacity of a `JoinSession` response channel: 33 slots, one
/// reserved for the final status when the channel is created.
const JOIN_DELIVERY_SLOTS: usize = 32;

type SubscribeStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

/// Run a test body under [`SCENARIO_BUDGET`]. The body is boxed: a multi-phase
/// body over two replicas is a large future, and held inline it overflows the
/// test thread's stack in an unoptimised build.
async fn scenario<F: std::future::Future<Output = ()>>(body: F) {
    tokio::time::timeout(SCENARIO_BUDGET, Box::pin(body))
        .await
        .unwrap_or_else(|_| panic!("the scenario did not finish within {SCENARIO_BUDGET:?}"));
}

fn subscribe_request(
    session_id: &str,
    object_id: u64,
    sequence_num: u64,
    client_id: &str,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: LAYER.to_owned(),
        object_id,
        sequence_num,
        client_id: client_id.to_owned(),
    });
    request.extensions_mut().insert(BearerToken::new(BEARER));
    request
}

/// A structural delta adding `section_id` under the root, applied against the
/// current state (not a reset), so every commit accumulates.
fn add_section(section_id: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
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

/// One commit through `svc`, from the writer `test-client` (never a subscriber
/// here, so self-echo suppression never swallows it). Returns the committed
/// sequence number.
async fn commit(svc: &LiveEditService, session_id: &str, object_id: u64, section_id: u64) -> u64 {
    let ompf = crate::common::owned_delta::delta_to_ompf_bytes(&add_section(section_id), &[]);
    // Boxed: the UpdateLayer handler's future is large, and several commits
    // inline in one test body would all live on the test thread's stack.
    Box::pin(handle_update_layer(
        svc,
        make_request(streaming_update(vec![
            make_update_header(session_id, LAYER, object_id),
            make_update_data(ompf),
        ])),
    ))
    .await
    .unwrap_or_else(|e| panic!("UpdateLayer (section {section_id}) must succeed: {e}"))
    .into_inner()
    .sequence_num
}

/// The next header and its whole body, with no deadline of its own: callers
/// bound it — [`next_message`] per message, [`read_catch_up`] per catch-up.
async fn read_message<S>(stream: &mut S, what: &str) -> (SubscribeReadLayerHeader, Vec<u8>)
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    let first = stream
        .next()
        .await
        .unwrap_or_else(|| panic!("{what}: the stream ended"))
        .unwrap_or_else(|s| panic!("{what}: the stream failed: {s:?}"));
    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("{what}: expected a header, got {other:?}"),
    };
    let mut body = Vec::new();
    while (body.len() as u64) < header.total_size {
        let next = stream
            .next()
            .await
            .unwrap_or_else(|| panic!("{what}: the stream ended mid-body"))
            .unwrap_or_else(|s| panic!("{what}: the stream failed mid-body: {s:?}"));
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                body.extend_from_slice(&c.data)
            }
            other => panic!("{what}: expected a body chunk, got {other:?}"),
        }
    }
    (header, body)
}

/// The next header and its whole body, under one `SETTLE` deadline.
async fn next_message<S>(stream: &mut S, what: &str) -> (SubscribeReadLayerHeader, Vec<u8>)
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    tokio::time::timeout(SETTLE, read_message(stream, what))
        .await
        .unwrap_or_else(|_| panic!("{what}: nothing arrived within {SETTLE:?}"))
}

/// Read the catch-up through its caught-up ack (the only `total_size == 0`
/// header), returning every content body in order and the ack — all under
/// one `SETTLE` deadline, however many messages the catch-up turns out to be.
async fn read_catch_up<S>(stream: &mut S, what: &str) -> (Vec<Vec<u8>>, SubscribeReadLayerHeader)
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    tokio::time::timeout(SETTLE, async {
        let mut bodies = Vec::new();
        loop {
            let (header, body) = read_message(stream, what).await;
            if header.total_size == 0 {
                return (bodies, header);
            }
            bodies.push(body);
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what}: the catch-up did not finish within {SETTLE:?}"))
}

/// Read to the end under one `SETTLE` deadline: the terminal status, `None`
/// for a clean end of stream. Data messages on the way are discarded.
async fn read_to_terminal<S>(stream: &mut S, what: &str) -> Option<Status>
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    tokio::time::timeout(SETTLE, async {
        while let Some(item) = stream.next().await {
            if let Err(status) = item {
                return Some(status);
            }
        }
        None
    })
    .await
    .unwrap_or_else(|_| {
        panic!("{what}: the subscription did not end within {SETTLE:?} — shutdown never reached it")
    })
}

fn stream_exits(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_stream_exits_total",
        &[("reason", reason)],
    )
    .unwrap_or_else(|| panic!("stream-exit series for reason={reason} is absent")) as u64
}

fn total_stream_exits(metrics: &Metrics) -> u64 {
    STREAM_EXIT_REASONS
        .iter()
        .map(|reason| stream_exits(metrics, reason))
        .sum()
}

/// Wait until the subscription task's tail has counted its exit — which it does
/// after logging its line, so the line (if any) is in the capture by then.
async fn await_stream_exit(metrics: &Metrics, reason: &str) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while stream_exits(metrics, reason) == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "no `{reason}` stream exit within {SETTLE:?}; exits so far: {}",
            total_stream_exits(metrics)
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The ending a reference client acts on: the Kit client's retry policy reduced
/// to the one code these tests produce. `UNAVAILABLE` is retried; every other ending
/// — `OK` included — is final.
fn reference_client_retries(ending: &Option<Status>) -> bool {
    matches!(ending, Some(status) if status.code() == Code::Unavailable)
}

/// Apply one OMPF payload to a client-side mirror, through the same pipeline
/// the server applies deltas with.
fn apply_ompf(mirror: &mut ObjectState, ompf: &[u8]) {
    let container = OmpfContainer::parse(ompf).expect("a delivered payload must parse as OMPF");
    let delta = if container.delta_bytes.is_empty() {
        None
    } else {
        Some(oodf_ffi::parse_delta(container.delta_bytes).expect("parse OODF delta"))
    };
    apply_delta(mirror, delta).expect("a delivered delta must apply to the mirror");
}

/// A state's structure — every section's identity and placement — for
/// comparing two reconstructions. Version annotations are left out: a
/// coalesced catch-up legitimately discards per-commit history.
fn structure(state: &ObjectState) -> BTreeMap<u64, (u64, String, u8, u32)> {
    state
        .sections
        .iter()
        .map(|(id, s)| {
            (
                *id,
                (s.parent_id, s.name.clone(), s.section_type, s.section_order),
            )
        })
        .collect()
}

/// `live_subscribe_regime_total{regime, service}` on `svc` — 0 when the series
/// is absent, since the regime counter is not pre-initialised per regime.
fn regime_count(svc: &LiveEditService, regime: &str) -> u64 {
    sample_value(
        &render(&svc.metrics),
        "live_subscribe_regime_total",
        &[("regime", regime), ("service", svc.service_label.as_str())],
    )
    .unwrap_or(0.0) as u64
}

/// A second replica over the same Valkey keyspace as `tv`: its own
/// `ValkeyClient` (pools, subscriber), session-token map, presence router,
/// trackers and shutdown token, with `layer_stream_maxlen = maxlen`.
async fn replica(tv: &TestValkey, maxlen: u64) -> (LiveEditService, ValkeyClient) {
    let config = live_edit::config::Config {
        valkey_connect_timeout_secs: 30,
        ..crate::common::test_config()
    };
    let valkey = live_edit::storage::test_support::connect_test_client(&config)
        .await
        .expect("the second replica's ValkeyClient must connect");
    let svc = replica_over(tv, valkey.clone(), maxlen);
    (svc, valkey)
}

/// A replica over `valkey` and `tv`'s keyspace. `tv`'s own client is fine for
/// the first replica; [`replica`] makes the second.
fn replica_over(tv: &TestValkey, valkey: ValkeyClient, maxlen: u64) -> LiveEditService {
    let base = build_service(tv, Arc::new(AllowAllPermissions));
    let session_store = Arc::new(SessionStore::new(valkey.clone(), tv.key_schema().clone()));
    let layer_store = Arc::new(LayerStore::new(
        valkey.clone(),
        tv.key_schema().clone(),
        maxlen,
    ));
    let recovery =
        build_disabled_recovery_hook(Arc::clone(&layer_store), Arc::clone(&base.metrics));
    LiveEditService {
        valkey,
        session_store,
        layer_store,
        recovery,
        ..base
    }
}

/// Drain `svc`'s streaming tasks as `main.rs` does — `drain_stream_tasks`
/// over both trackers, each bounded by `SETTLE` — and require both to have
/// drained inside that bound.
async fn drain_streams(svc: &LiveEditService) {
    drain_stream_tasks(
        &svc.presence_cleanup_tracker,
        &svc.subscription_tracker,
        SETTLE,
    )
    .await;
    assert!(
        svc.subscription_tracker.is_empty(),
        "the subscription tasks did not drain within {SETTLE:?}; {} left",
        svc.subscription_tracker.len()
    );
    assert!(
        svc.presence_cleanup_tracker.is_empty(),
        "the join tasks did not drain within {SETTLE:?}; {} left",
        svc.presence_cleanup_tracker.len()
    );
}

// =============================================================================
// The idle subscriber (owed test 1 and 2)
// =============================================================================

/// **An idle, caught-up subscriber ends `UNAVAILABLE` when only the service
/// shutdown token is cancelled** — the ending the Kit client retries.
///
/// The subscriber has drained its catch-up through the caught-up ack, the layer
/// is quiet, and nothing is in flight: it is parked in its idle wait. Its
/// session token is never touched, as in production, where shutdown no longer
/// cancels session tokens. The stream must end `UNAVAILABLE "server shutting
/// down"`, the exit is counted once under `shutdown` (not `session_cancel`),
/// and the subscription's tail logs one INFO line carrying `reason=shutdown`.
///
/// Before b.e1h the subscription did not read the service shutdown token at
/// all, so this stream never ended.
///
/// Current-thread runtime, so the thread-local capture sees the line the
/// spawned subscription task logs (`log_capture` module doc).
#[tokio::test]
async fn an_idle_caught_up_subscriber_ends_unavailable_on_service_shutdown_alone() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "e1h-idle-shutdown", LAYER).await;
    let capture = ThreadCapture::start();

    let mut stream: SubscribeStream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, 0, 0, "idle-reader"))
            .await
            .unwrap_or_else(|s| panic!("the subscription must be admitted: {s:?}"))
            .into_inner();
    let _ = read_catch_up(&mut stream, "the idle subscriber's catch-up").await;
    let session_token = svc
        .session_tokens
        .get(&session_id)
        .expect("the subscribe handler registered the session's token")
        .clone();

    svc.shutdown.cancel();
    let ending = read_to_terminal(&mut stream, "an idle subscriber on shutdown").await;

    assert!(
        !session_token.is_cancelled(),
        "precondition: only the service shutdown token was cancelled"
    );
    let status = ending.expect(
        "an idle subscriber on a shutting-down replica ended with a clean end of stream (gRPC \
         OK). The Kit client never retries OK, so its live feed would stop here for good",
    );
    assert_eq!(
        status.code(),
        Code::Unavailable,
        "a shutdown must end an idle subscription retryably; got {status:?}"
    );
    assert_shutdown_status(&status);

    await_stream_exit(&svc.metrics, STREAM_EXIT_SHUTDOWN).await;
    assert_eq!(stream_exits(&svc.metrics, STREAM_EXIT_SHUTDOWN), 1);
    assert_eq!(
        stream_exits(&svc.metrics, STREAM_EXIT_SESSION_CANCEL),
        0,
        "a shutdown is not a session deletion and must not be counted as one"
    );
    assert_eq!(
        total_stream_exits(&svc.metrics),
        1,
        "one subscription, one exit: sum by (reason) must equal the exits"
    );

    let captured = capture.snapshot();
    let event = captured.only(RPC_FAILURE_MSG);
    assert_eq!(
        captured.completion_lines().len(),
        1,
        "one ending, one completion line; full capture:\n{}",
        captured.text
    );
    assert_common_fields(event, "INFO", "SubscribeReadLayer", "stream", "UNAVAILABLE");
    assert_field(event, "reason", STREAM_EXIT_SHUTDOWN.into());
    assert_field(event, "caught_up", true.into());
    assert_field(event, "status_message", "server shutting down".into());

    drop(capture);
    purge_keys(&tv).await;
}

/// **Shutdown wins when an idle subscriber sees it together with a session
/// cancel** — the order the pre-b.e1h shutdown fan-out produced (the service
/// shutdown token, then every session token).
///
/// Both tokens are cancelled before the subscription task runs again (the
/// runtime is current-thread and the test does not yield between them), so
/// when it wakes both are observable. It must end as a shutdown —
/// `UNAVAILABLE`, `reason=shutdown` — because its session still exists and
/// the client must reconnect, wherever the ingress routes it; a deletion
/// ending (`NOT_FOUND`, not retried) would strand it.
///
/// Before b.e1h this ended `OK`.
#[tokio::test]
async fn an_idle_subscriber_seeing_shutdown_and_its_session_token_together_ends_as_a_shutdown() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "e1h-old-fanout", LAYER).await;

    let mut stream: SubscribeStream =
        handle_subscribe_read_layer(&svc, subscribe_request(&session_id, 0, 0, "fanout-reader"))
            .await
            .unwrap_or_else(|s| panic!("the subscription must be admitted: {s:?}"))
            .into_inner();
    let _ = read_catch_up(&mut stream, "the idle subscriber's catch-up").await;

    // The old fan-out, emulated: the shutdown token, then the session token.
    svc.shutdown.cancel();
    svc.session_tokens
        .get(&session_id)
        .expect("the subscribe handler registered the session's token")
        .cancel();

    let status = read_to_terminal(&mut stream, "an idle subscriber under the old fan-out")
        .await
        .expect(
            "the subscriber ended with a clean end of stream (gRPC OK) — the ending the Kit \
             client never retries",
        );
    assert_eq!(
        status.code(),
        Code::Unavailable,
        "with shutdown and a session cancel both observable, shutdown must win; got {status:?}"
    );
    assert_shutdown_status(&status);
    await_stream_exit(&svc.metrics, STREAM_EXIT_SHUTDOWN).await;
    assert_eq!(
        stream_exits(&svc.metrics, STREAM_EXIT_SESSION_CANCEL),
        0,
        "the exit must be counted as a shutdown, not a deletion"
    );
    assert_eq!(total_stream_exits(&svc.metrics), 1);

    purge_keys(&tv).await;
}

// =============================================================================
// A JoinSession whose client stopped reading (owed test 10)
// =============================================================================

/// `live_presence_events_total{event_type="message"}` on `svc`.
fn delivered_messages(svc: &LiveEditService) -> u64 {
    sample_value(
        &render(&svc.metrics),
        "live_presence_events_total",
        &[
            ("event_type", "message"),
            ("service", svc.service_label.as_str()),
        ],
    )
    .expect("live_presence_events_total{event_type=\"message\"} is seeded") as u64
}

fn peer_message(i: usize) -> PresenceEvent {
    PresenceEvent::PeerMessage {
        connection_id: "e1h-peer".to_owned(),
        data: (i as u32).to_be_bytes().to_vec(),
    }
}

/// What ends a stalled join in the tests below.
#[derive(Clone, Copy, Debug)]
enum StalledJoinCause {
    /// The service shutdown token alone.
    Shutdown,
    /// The session token alone, as a deletion fires it.
    Deletion,
    /// Both, shutdown first, with no yield between them — so on this
    /// current-thread runtime the parked send, when it next runs, finds both
    /// cancelled.
    ShutdownAndDeletion,
    /// The session token alone, cancelled through the map entry with the cause
    /// recorded for a session found missing from the live store with no
    /// deletion marker (`cancel_absent_from_store`).
    AbsentFromStore,
}

/// **A join whose client has stopped reading still ends when shutdown, a
/// deletion or a session found missing from the live store reaches it**: its
/// teardown runs, its ending — `UNAVAILABLE` into the slot reserved for it for
/// a shutdown or a missing session (each with its own message), a clean end
/// for a deletion — is delivered behind what the client left unread, and the
/// join task — and so the presence tracker the shutdown sequence drains —
/// finishes without the client reading anything.
///
/// The client reads `SessionConnected` and then nothing. The test fills all
/// `JOIN_DELIVERY_SLOTS` delivery slots with peer messages broadcast straight
/// into the service's presence router, waits until the join has counted every
/// one of them as enqueued, and broadcasts one more, which the join takes and
/// then cannot send. The runtime is current-thread and nothing on that path
/// awaits I/O, so yielding lets the join reach that parked send before the
/// cause fires. Only then does it fire.
///
/// The parked send races shutdown, then the session token, then the client
/// going away. Without its session arm a deletion could not reach a join whose
/// client stopped reading at all; with the two token arms the other way round,
/// both together would end the join `OK` — as a deletion — though its session
/// still exists.
///
/// Before b.e1h the parked send watched nothing, so the join task never ended
/// and the drain below timed out; and a join status was sent with an awaited
/// send into a full channel, which could not complete either.
async fn a_stalled_join_ends_on(cause: StalledJoinCause) {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, session_token) = create_session(&tv, &svc, "e1h-stalled-join", true).await;

    let (mut stream, tx) = open_join(&svc, &session_id, "stalled-app").await;
    let cid = expect_session_connected(&mut stream, SETTLE).await;

    for i in 0..JOIN_DELIVERY_SLOTS {
        svc.presence_router.broadcast(&session_id, peer_message(i));
    }
    // The tally counts an event only once it is enqueued, and flushes on its
    // one-second timer only while the loop is idle — so this reaching the full
    // count proves every delivery slot is taken and the loop is waiting again.
    let deadline = tokio::time::Instant::now() + SETTLE;
    while delivered_messages(&svc) < JOIN_DELIVERY_SLOTS as u64 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "the join enqueued {} of {JOIN_DELIVERY_SLOTS} peer messages within {SETTLE:?}",
            delivered_messages(&svc)
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    svc.presence_router
        .broadcast(&session_id, peer_message(JOIN_DELIVERY_SLOTS));
    for _ in 0..8 {
        tokio::task::yield_now().await;
    }

    match cause {
        StalledJoinCause::Shutdown => svc.shutdown.cancel(),
        StalledJoinCause::Deletion => session_token.cancel(),
        StalledJoinCause::ShutdownAndDeletion => {
            svc.shutdown.cancel();
            session_token.cancel();
        }
        StalledJoinCause::AbsentFromStore => svc
            .session_tokens
            .get(&session_id)
            .expect("the session's token is registered on this replica")
            .cancel_absent_from_store(),
    }
    // The client has read nothing since SessionConnected, and does not start
    // until the join task has finished.
    drain_tracker(&svc).await;

    assert_eq!(
        presence_keys(&tv, &session_id).await,
        vec![tv.key_schema().presence_gone(&session_id, &cid)],
        "{cause:?}: teardown must have removed the member entry, info hash and lease, leaving \
         only its teardown marker"
    );
    assert!(
        !is_indexed(&tv, &session_id).await,
        "{cause:?}: the session's only member left, so teardown must unindex it"
    );

    let (delivered, status) = read_to_end(&mut stream, "a stalled join after its cause").await;
    assert_eq!(
        delivered.len(),
        JOIN_DELIVERY_SLOTS,
        "the client finds exactly the {JOIN_DELIVERY_SLOTS} messages it left unread; the one \
         the join could not send was dropped with it"
    );
    assert!(
        delivered.iter().all(|msg| matches!(
            &msg.response_type,
            Some(join_session_response::ResponseType::PeerMessage(m)) if m.connection_id != cid
        )),
        "every buffered message is a peer's message"
    );
    match cause {
        StalledJoinCause::Shutdown | StalledJoinCause::ShutdownAndDeletion => {
            let status = status.unwrap_or_else(|| {
                panic!(
                    "{cause:?}: the stalled join ended with a clean end of stream (gRPC OK), which \
                     the Kit client never retries — and its session still exists"
                )
            });
            assert_eq!(
                status.code(),
                Code::Unavailable,
                "{cause:?}: got {status:?}"
            );
            assert_shutdown_status(&status);
        }
        StalledJoinCause::Deletion => assert!(
            status.is_none(),
            "a deleted session's stalled join must end OK, so its client does not rejoin a \
             session that is gone; got {status:?}"
        ),
        StalledJoinCause::AbsentFromStore => {
            let status = status.unwrap_or_else(|| {
                panic!(
                    "a stalled join whose session went missing from the live store ended OK, which \
                     the Kit client never retries — so a session a checkpoint could restore is lost \
                     to it"
                )
            });
            assert_eq!(status.code(), Code::Unavailable, "got {status:?}");
            assert_eq!(
                status.message(),
                live_edit::grpc::session::session_absent_status().message()
            );
        }
    }

    drop(tx);
    purge_keys(&tv).await;
}

#[tokio::test]
async fn a_join_whose_client_stopped_reading_tears_down_and_ends_unavailable_on_shutdown() {
    a_stalled_join_ends_on(StalledJoinCause::Shutdown).await;
}

#[tokio::test]
async fn a_join_whose_client_stopped_reading_tears_down_and_ends_ok_on_deletion() {
    a_stalled_join_ends_on(StalledJoinCause::Deletion).await;
}

#[tokio::test]
async fn a_join_whose_client_stopped_reading_ends_unavailable_on_shutdown_and_deletion_together() {
    a_stalled_join_ends_on(StalledJoinCause::ShutdownAndDeletion).await;
}

/// The parked event send's session arm ends the join by the cancel's cause: a
/// session missing from the live store ends it `UNAVAILABLE`, retried, not the
/// deletion's `OK`.
#[tokio::test]
async fn a_join_whose_client_stopped_reading_tears_down_and_ends_unavailable_on_session_absence() {
    a_stalled_join_ends_on(StalledJoinCause::AbsentFromStore).await;
}

// =============================================================================
// A stream the paced ending has not reached yet (b.kgb Policy 1)
// =============================================================================

/// **A stream the paced ending has not reached yet ends by its own cause.**
/// An idle, caught-up subscription and an established join are live when the
/// paced ending starts; while it is under way — `begun` cancelled, the
/// service shutdown token not — their session is deleted through the real
/// `DeleteSession`. Neither stream's own shutdown token has fired, so each
/// ends as the deletion it observed: the subscription `NOT_FOUND` (the session
/// token or the layer's `deleted` sentinel, whichever it sees first), the join
/// `OK` — terminal for their clients, which do not retry into a session that
/// is gone. A design that made every stream observe the paced ending's start
/// would end both `UNAVAILABLE`, and their clients would retry into a
/// `NOT_FOUND` at setup. (Shutdown and deletion seen *together* still end as a
/// shutdown: the `*_together` tests above.)
///
/// **A timing margin, not a guarantee.** The two streams sit in a 22-entry
/// snapshot (with the 20 placeholders) that is shuffled, so with probability
/// 2/22 one of them is in the first batch, whose token fires at t0 + 0.9 s —
/// the most margin a rate of 1/s allows. The deletion and both observations
/// must therefore finish inside that margin. The test checks this as an
/// explicitly labelled precondition before the outcome, so a run too slow to
/// meet it fails as a missed precondition rather than as a false Policy 1
/// regression.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stream_not_yet_reached_by_the_paced_ending_whose_session_is_deleted_ends_as_a_deletion()
{
    use live_edit::grpc::proto::DeleteSessionRequest;
    use live_edit::grpc::session::handle_delete_session;

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) = create_session_and_layer(&tv, &svc, "kgb-policy-1", LAYER).await;

    let mut subscription: SubscribeStream = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, 0, 0, "policy-1-reader"),
    )
    .await
    .unwrap_or_else(|s| panic!("the subscription must be admitted: {s:?}"))
    .into_inner();
    let _ = read_catch_up(&mut subscription, "the subscription's catch-up").await;
    let (mut join, join_tx) = open_join(&svc, &session_id, "policy-1-app").await;
    let _ = expect_session_connected(&mut join, SETTLE).await;
    assert_eq!(
        svc.shutdown.enrolled(),
        2,
        "precondition: both streams are enrolled, so both are in the paced ending's snapshot"
    );

    /// The earliest the paced ending can reach either stream is t0 + 0.9 s.
    /// `started_at` is taken before `begin_paced_ending_held_open` is called,
    /// and `end_streams_after_drain` sleeps `STREAM_END_DELAY_AFTER_GOAWAY`
    /// (250 ms) before t0, so `started_at` is at most t0 − 250 ms: anything
    /// observed within 250 ms + 900 ms of it is before the first batch, however
    /// late the test wakes after `begun` fires.
    const POLICY_1_MARGIN: Duration =
        STREAM_END_DELAY_AFTER_GOAWAY.saturating_add(Duration::from_millis(900));

    let started_at = tokio::time::Instant::now();
    let paced = begin_paced_ending_held_open(&svc.shutdown).await;
    let mut delete = Request::new(DeleteSessionRequest {
        session_id: session_id.clone(),
    });
    delete.extensions_mut().insert(BearerToken::new(BEARER));
    handle_delete_session(&svc, delete)
        .await
        .expect("DeleteSession must succeed while the paced ending is under way");

    let subscription_end = read_to_terminal(&mut subscription, "the subscription on deletion")
        .await
        .expect("a deleted session's subscription must end with a status, not OK");
    let subscription_ended_after = started_at.elapsed();
    let (_, join_end) = read_to_end(&mut join, "the join on deletion").await;
    let join_ended_after = started_at.elapsed();
    assert!(
        subscription_ended_after < POLICY_1_MARGIN && join_ended_after < POLICY_1_MARGIN,
        "TIMING PRECONDITION, not a Policy 1 result: the deletion and both endings must land \
         within {POLICY_1_MARGIN:?} of the call that starts the paced ending, before its first \
         batch can reach either stream; the subscription ended after {subscription_ended_after:?} and \
         the join after {join_ended_after:?}"
    );
    assert_eq!(
        subscription_end.code(),
        Code::NotFound,
        "a subscription the paced ending had not reached ends as the deletion it observed, \
         not as a shutdown; got {subscription_end:?}"
    );
    assert_no_refusal_detail(&subscription_end);
    assert!(
        join_end.is_none(),
        "a join the paced ending had not reached ends OK on its session's deletion, so its \
         client does not rejoin a session that is gone; got {join_end:?}"
    );
    assert!(
        !svc.shutdown.is_cancelled(),
        "precondition: both endings came while the paced ending was still under way"
    );
    assert_eq!(
        stream_exits(&svc.metrics, STREAM_EXIT_SHUTDOWN),
        0,
        "the subscription's exit is not counted as a shutdown"
    );

    drop(paced);
    drop(join_tx);
    drain_streams(&svc).await;
    purge_keys(&tv).await;
}

// =============================================================================
// The real server: serve through the drain, end paced behind GOAWAY
// =============================================================================

mod real_server {
    use super::*;

    use tokio::sync::{mpsc, oneshot};
    use tokio_stream::wrappers::ReceiverStream;
    use tonic::metadata::MetadataValue;
    use tonic::transport::Endpoint;

    use live_edit::grpc::proto::live_service_client::LiveServiceClient;
    use live_edit::grpc::proto::JoinSessionRequest;
    use live_edit::grpc::transport::{live_service_server, production_server_builder};

    fn with_bearer<T>(inner: T) -> Request<T> {
        let mut request = Request::new(inner);
        request.metadata_mut().insert(
            "authorization",
            MetadataValue::try_from(format!("Bearer {BEARER}")).expect("valid metadata"),
        );
        request
    }

    /// **The production shutdown wiring, over TCP.** The server is built from
    /// `production_server_builder` and shut down through
    /// `end_streams_after_drain`, as `main.rs` does it, with an idle
    /// `SubscribeReadLayer` and an established `JoinSession` open on it. The
    /// pacing is off (`StreamPacing::ALL_AT_ONCE`), so both streams end at t0;
    /// the paced ramp is the next test's.
    ///
    /// * **During the drain** — the shutdown signal has begun but the drain
    ///   future has not resolved — nothing is cancelled: a commit is still
    ///   delivered to the idle subscriber.
    /// * **After the drain**, both streams end `UNAVAILABLE "server shutting
    ///   down"` with the shutdown retry hint, the server returns within
    ///   `SERVER_EXIT`, and both trackers the shutdown sequence drains —
    ///   subscriptions and presence cleanup — are empty.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_real_server_serves_streams_through_the_drain_and_ends_them_unavailable_after_it() {
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, object_id) =
            create_session_and_layer(&tv, &svc, "e1h-real-server", LAYER).await;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind a random port");
        let addr = listener.local_addr().expect("local addr");
        let (drained_tx, drained_rx) = oneshot::channel::<()>();
        let server = {
            let svc = svc.clone();
            let service_shutdown = svc.shutdown.clone();
            tokio::spawn(async move {
                production_server_builder()
                    .add_service(live_service_server(svc))
                    .serve_with_incoming_shutdown(
                        tokio_stream::wrappers::TcpListenerStream::new(listener),
                        end_streams_after_drain(
                            async move {
                                let _ = drained_rx.await;
                            },
                            service_shutdown,
                            StreamPacing::ALL_AT_ONCE,
                            rand::rngs::StdRng::seed_from_u64(0),
                        ),
                    )
                    .await
            })
        };

        let channel = tokio::time::timeout(
            SETTLE,
            Endpoint::from_shared(format!("http://{addr}"))
                .expect("valid endpoint")
                .connect(),
        )
        .await
        .expect("connect within the deadline")
        .expect("connect to the test server");
        let mut client = LiveServiceClient::new(channel);

        let mut subscription = client
            .subscribe_read_layer(with_bearer(SubscribeReadLayerRequest {
                session_id: session_id.clone(),
                layer_name: LAYER.to_owned(),
                object_id: 0,
                sequence_num: 0,
                client_id: "server-reader".to_owned(),
            }))
            .await
            .unwrap_or_else(|s| panic!("the subscription must be admitted: {s:?}"))
            .into_inner();
        let _ = read_catch_up(&mut subscription, "the subscription's catch-up").await;

        let (join_tx, join_rx) = mpsc::channel::<JoinSessionRequest>(4);
        join_tx
            .send(join_header(&session_id, "server-join-app"))
            .await
            .expect("the request channel accepts the header");
        let mut join = client
            .join_session(with_bearer(ReceiverStream::new(join_rx)))
            .await
            .unwrap_or_else(|s| panic!("the join must be admitted: {s:?}"))
            .into_inner();
        let first = tokio::time::timeout(SETTLE, join.next())
            .await
            .expect("SessionConnected within the deadline")
            .expect("the join stream must not be empty")
            .unwrap_or_else(|s| panic!("the join failed before SessionConnected: {s:?}"));
        assert!(
            matches!(
                first.response_type,
                Some(join_session_response::ResponseType::Connected(_))
            ),
            "precondition: the join is established; got {first:?}"
        );

        // The drain: readiness is NOT_SERVING in production and the signal is
        // sleeping; here, the drain future simply has not resolved yet.
        let committed = commit(&svc, &session_id, object_id, 7).await;
        let (relayed, body) =
            next_message(&mut subscription, "a commit made during the drain").await;
        assert_eq!(
            relayed.sequence_num, committed,
            "a commit made during the drain must still be delivered"
        );
        assert!(!body.is_empty());
        assert!(
            !svc.shutdown.is_cancelled(),
            "nothing may end the streams before the drain does"
        );

        drained_tx
            .send(())
            .expect("the server holds the drain receiver");

        let subscription_end =
            read_to_terminal(&mut subscription, "the subscription after the drain")
                .await
                .expect("the subscription ended OK, which the Kit client never retries");
        assert_eq!(
            subscription_end.code(),
            Code::Unavailable,
            "got {subscription_end:?}"
        );
        assert_shutdown_status(&subscription_end);

        let join_end = tokio::time::timeout(SETTLE, async {
            while let Some(item) = join.next().await {
                if let Err(status) = item {
                    return Some(status);
                }
            }
            None
        })
        .await
        .expect("the join must end within the deadline once the drain is over")
        .expect("the join ended OK, which the Kit client never retries");
        assert_eq!(join_end.code(), Code::Unavailable, "got {join_end:?}");
        assert_shutdown_status(&join_end);

        // The client's request half of the join is closed too, so no stream
        // stays open on the connection.
        drop(join_tx);
        drop(join);
        drop(subscription);
        drop(client);

        let served = tokio::time::timeout(SERVER_EXIT, server)
            .await
            .expect("the server must return once its streams have ended")
            .expect("the server task must not panic");
        assert!(
            served.is_ok(),
            "the server must shut down cleanly: {served:?}"
        );

        // The post-server step `main.rs` runs next, through the same helper.
        drain_streams(&svc).await;
    }

    // ── The server as `main.rs` runs it, around a closable listener ─────────

    use std::net::SocketAddr;

    use live_edit::grpc::graceful_shutdown::GracefulShutdown;
    use live_edit::grpc::transport::serve_with_post_goaway_bound;
    use live_edit::observability::startup::{closable_incoming, ListenerControl};
    use tonic::transport::server::TcpIncoming;

    use super::super::log_capture::{capture, Captured};

    /// What [`RealServer`]'s task returns: `serve_with_post_goaway_bound`'s
    /// answer (`None` when the bound fired) and the lines it logged.
    type Served = (Option<Result<(), tonic::transport::Error>>, Captured);

    /// The full-API server over TCP, shut down as `main.rs` shuts it down:
    /// the production HTTP/2 settings and LiveService stack, served from a
    /// [`closable_incoming`] listener that the drain closes as it ends,
    /// `end_streams_after_drain` with the given pacing, and the bound after
    /// GOAWAY (`serve_with_post_goaway_bound`) with the given RPC deadline,
    /// reporting the listener's open connections.
    struct RealServer {
        addr: SocketAddr,
        listener: ListenerControl,
        drained_tx: Option<oneshot::Sender<()>>,
        task: tokio::task::JoinHandle<Served>,
    }

    impl RealServer {
        async fn start(
            svc: &LiveEditService,
            pacing: StreamPacing,
            rpc_deadline: Duration,
        ) -> Self {
            let tcp = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("bind a random port");
            let addr = tcp.local_addr().expect("local addr");
            let (incoming, listener) = closable_incoming(TcpIncoming::from(tcp));
            let (drained_tx, drained_rx) = oneshot::channel::<()>();
            let task = {
                let svc = svc.clone();
                let shutdown: GracefulShutdown = svc.shutdown.clone();
                let drain_listener = listener.clone();
                let count_listener = listener.clone();
                tokio::spawn(async move {
                    let serve = production_server_builder()
                        .add_service(live_service_server(svc))
                        .serve_with_incoming_shutdown(
                            incoming,
                            end_streams_after_drain(
                                async move {
                                    let _ = drained_rx.await;
                                    // As `main.rs`'s drain does: close the
                                    // listening socket as the drain ends.
                                    drain_listener.close();
                                },
                                shutdown.clone(),
                                pacing,
                                rand::rngs::StdRng::seed_from_u64(0),
                            ),
                        );
                    capture(serve_with_post_goaway_bound(
                        serve,
                        &shutdown,
                        rpc_deadline,
                        move || count_listener.open_connections(),
                    ))
                    .await
                })
            };
            Self {
                addr,
                listener,
                drained_tx: Some(drained_tx),
                task,
            }
        }

        /// A client channel to the server, with tonic's default settings.
        async fn client(&self) -> LiveServiceClient<tonic::transport::Channel> {
            self.client_from(Endpoint::from_shared(format!("http://{}", self.addr)).expect("valid"))
                .await
        }

        async fn client_from(
            &self,
            endpoint: Endpoint,
        ) -> LiveServiceClient<tonic::transport::Channel> {
            let channel = tokio::time::timeout(SETTLE, endpoint.connect())
                .await
                .expect("connect within the deadline")
                .expect("connect to the test server");
            LiveServiceClient::new(channel)
        }

        /// End the readiness drain: the listener closes and the shutdown
        /// signal resolves, so tonic sends GOAWAY and the paced ending starts
        /// `STREAM_END_DELAY_AFTER_GOAWAY` later.
        fn end_drain(&mut self) {
            self.drained_tx
                .take()
                .expect("the drain ends once")
                .send(())
                .expect("the server holds the drain receiver");
        }
    }

    async fn subscribe_over(
        client: &mut LiveServiceClient<tonic::transport::Channel>,
        session_id: &str,
        client_id: &str,
    ) -> tonic::Streaming<SubscribeReadLayerResponse> {
        client
            .subscribe_read_layer(with_bearer(SubscribeReadLayerRequest {
                session_id: session_id.to_owned(),
                layer_name: LAYER.to_owned(),
                object_id: 0,
                sequence_num: 0,
                client_id: client_id.to_owned(),
            }))
            .await
            .unwrap_or_else(|s| panic!("{client_id}: the subscription must be admitted: {s:?}"))
            .into_inner()
    }

    /// Open a join over `client` and read its `SessionConnected`. Returns the
    /// response stream and the request sender, which must be kept open.
    async fn join_over(
        client: &mut LiveServiceClient<tonic::transport::Channel>,
        session_id: &str,
        app: &str,
    ) -> (
        tonic::Streaming<live_edit::grpc::proto::JoinSessionResponse>,
        mpsc::Sender<JoinSessionRequest>,
    ) {
        let (tx, rx) = mpsc::channel::<JoinSessionRequest>(4);
        tx.send(join_header(session_id, app))
            .await
            .expect("the request channel accepts the header");
        let mut join = client
            .join_session(with_bearer(ReceiverStream::new(rx)))
            .await
            .unwrap_or_else(|s| panic!("{app}: the join must be admitted: {s:?}"))
            .into_inner();
        let first = tokio::time::timeout(SETTLE, join.next())
            .await
            .expect("SessionConnected within the deadline")
            .expect("the join stream must not be empty")
            .unwrap_or_else(|s| panic!("{app}: the join failed before SessionConnected: {s:?}"));
        assert!(
            matches!(
                first.response_type,
                Some(join_session_response::ResponseType::Connected(_))
            ),
            "{app}: precondition: the join is established; got {first:?}"
        );
        (join, tx)
    }

    // ── The ramp (b.kgb test 6) ──────────────────────────────────────────────

    /// Subscriptions and joins the ramp test holds open on the server.
    const RAMP_SUBSCRIPTIONS: usize = 30;
    const RAMP_JOINS: usize = 10;
    const RAMP_STREAMS: usize = RAMP_SUBSCRIPTIONS + RAMP_JOINS;
    /// The test's ending rate: forty streams over a 2 s window, twenty
    /// batches of two.
    const RAMP_RATE: u32 = 20;
    /// The shortest share of the paced window the observed span may cover:
    /// loose enough to absorb a stall that bunches batches, far above the
    /// milliseconds a step takes.
    const RAMP_MIN_SPAN_SHARE: f64 = 0.5;
    /// How long after the mid-window commit was acknowledged a subscription
    /// still open must have received it.
    const RAMP_DELIVERY_GRACE: Duration = Duration::from_secs(1);

    /// What one stream's client saw: its ending, when it arrived, and (for a
    /// subscription) every sequence number delivered after its catch-up.
    struct Observed {
        what: String,
        ending: Option<Status>,
        ended_at: tokio::time::Instant,
        delivered: Vec<u64>,
    }

    /// **The real server ends its streams as a ramp, not a step.** Thirty
    /// subscriptions and ten joins are open on the server over TCP; the paced
    /// ending runs at a test rate of 20/s, so a 2 s window of twenty batches
    /// of two, Δ = 100 ms apart. On the client side:
    ///
    /// * every ending is `UNAVAILABLE "server shutting down"` with the
    ///   `SERVER_SHUTTING_DOWN` retry hint and a delay under 100 ms;
    /// * the endings span at least half the window, no 100 ms bucket holds
    ///   more than a quarter of them, and they fall into at least ten
    ///   buckets (half the batches) — bounds loose enough to survive a stall
    ///   that fires overdue batches back to back, while a step puts all forty
    ///   in one bucket (exact batch sizes are the paused-time unit tests');
    /// * a commit made through another replica during the window still
    ///   reaches every subscription that is still open a second after it is
    ///   acknowledged.
    ///
    /// Today's single cancel puts all forty endings within a few
    /// milliseconds: one bucket, a span near zero.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_real_server_ends_its_streams_as_a_paced_ramp_not_a_step() {
        scenario(async {
            let tv = TestValkey::setup().await;
            let svc = replica_over(&tv, tv.client().clone(), 4096);
            let (svc_b, _client_b) = replica(&tv, 4096).await;
            let (session_id, object_id) =
                create_session_and_layer(&tv, &svc, "kgb-ramp", LAYER).await;
            let pacing = StreamPacing {
                rate_per_sec: RAMP_RATE,
                max_window: Duration::from_secs(20),
            };
            let batches = pacing.batches(RAMP_STREAMS);
            assert_eq!(batches, 20, "fixture: twenty batches");
            let mut server = RealServer::start(&svc, pacing, Duration::from_secs(30)).await;
            let mut client = server.client().await;

            let (ended_tx, mut ended_rx) = mpsc::unbounded_channel::<Observed>();
            let mut join_senders = Vec::new();
            for i in 0..RAMP_SUBSCRIPTIONS {
                let what = format!("subscription {i}");
                let mut stream = subscribe_over(&mut client, &session_id, &what).await;
                let _ = read_catch_up(&mut stream, &what).await;
                let ended_tx = ended_tx.clone();
                tokio::spawn(async move {
                    let mut delivered = Vec::new();
                    let ending = loop {
                        match stream.next().await {
                            Some(Ok(msg)) => {
                                if let Some(subscribe_read_layer_response::ResponseType::Header(
                                    h,
                                )) = msg.response_type
                                {
                                    delivered.push(h.sequence_num);
                                }
                            }
                            Some(Err(status)) => break Some(status),
                            None => break None,
                        }
                    };
                    let _ = ended_tx.send(Observed {
                        what,
                        ending,
                        ended_at: tokio::time::Instant::now(),
                        delivered,
                    });
                });
            }
            for i in 0..RAMP_JOINS {
                let what = format!("join {i}");
                let (mut join, tx) = join_over(&mut client, &session_id, &what).await;
                join_senders.push(tx);
                let ended_tx = ended_tx.clone();
                tokio::spawn(async move {
                    let ending = loop {
                        match join.next().await {
                            Some(Ok(_)) => continue,
                            Some(Err(status)) => break Some(status),
                            None => break None,
                        }
                    };
                    let _ = ended_tx.send(Observed {
                        what,
                        ending,
                        ended_at: tokio::time::Instant::now(),
                        delivered: Vec::new(),
                    });
                });
            }
            drop(ended_tx);
            assert_eq!(
                svc.shutdown.enrolled(),
                RAMP_STREAMS,
                "precondition: every stream is enrolled for the paced ending"
            );

            server.end_drain();
            let first = tokio::time::timeout(SETTLE, ended_rx.recv())
                .await
                .expect("the first ending within the deadline")
                .expect("a reader reports its ending");
            let committed = commit(&svc_b, &session_id, object_id, 9).await;
            let committed_at = tokio::time::Instant::now();
            let mut observed = vec![first];
            while observed.len() < RAMP_STREAMS {
                observed.push(
                    tokio::time::timeout(SETTLE, ended_rx.recv())
                        .await
                        .unwrap_or_else(|_| {
                            panic!(
                                "only {} of {RAMP_STREAMS} streams ended within {SETTLE:?} of \
                                 the previous ending",
                                observed.len()
                            )
                        })
                        .expect("every reader reports its ending"),
                );
            }

            for o in &observed {
                let status = o.ending.as_ref().unwrap_or_else(|| {
                    panic!("{}: ended OK, which the Kit client never retries", o.what)
                });
                assert_shutdown_status(status);
            }

            observed.sort_by_key(|o| o.ended_at);
            let t_first = observed[0].ended_at;
            let span = observed[RAMP_STREAMS - 1].ended_at - t_first;
            let window = pacing.window(RAMP_STREAMS);
            assert!(
                span >= window.mul_f64(RAMP_MIN_SPAN_SHARE),
                "the {RAMP_STREAMS} endings spanned {span:?}; a {window:?} paced window must \
                 spread them over at least {:?}. A span near zero is every stream ended at once",
                window.mul_f64(RAMP_MIN_SPAN_SHARE)
            );
            let mut buckets = BTreeMap::<u128, usize>::new();
            for o in &observed {
                let offset = o.ended_at - t_first;
                *buckets
                    .entry(offset.as_nanos() / STREAM_END_BATCH_INTERVAL.as_nanos())
                    .or_insert(0) += 1;
            }
            let fullest = buckets.values().copied().max().unwrap_or(0);
            assert!(
                fullest <= RAMP_STREAMS / 4,
                "a {STREAM_END_BATCH_INTERVAL:?} bucket held {fullest} of the {RAMP_STREAMS} \
                 endings; a ramp puts at most a quarter in any one. Buckets (index → endings): \
                 {buckets:?}"
            );
            assert!(
                buckets.len() >= batches / 2,
                "the endings fell into {} {STREAM_END_BATCH_INTERVAL:?} buckets; {batches} paced \
                 batches must reach at least {}. Buckets (index → endings): {buckets:?}",
                buckets.len(),
                batches / 2
            );

            let still_open: Vec<&Observed> = observed
                .iter()
                .filter(|o| o.what.starts_with("subscription"))
                .filter(|o| o.ended_at > committed_at + RAMP_DELIVERY_GRACE)
                .collect();
            assert!(
                !still_open.is_empty(),
                "fixture: some subscription must still be open a second after the mid-window \
                 commit (the window is {window:?})"
            );
            for o in still_open {
                assert!(
                    o.delivered.contains(&committed),
                    "{}: still open {:?} after the commit was acknowledged, it must have \
                     received it (seq {committed}); it received {:?}",
                    o.what,
                    o.ended_at - committed_at,
                    o.delivered
                );
            }

            drop(join_senders);
            drop(client);
            let (served, _) = tokio::time::timeout(SERVER_EXIT, server.task)
                .await
                .expect("the server must return once its streams have ended")
                .expect("the server task must not panic");
            assert!(
                matches!(served, Some(Ok(()))),
                "the server must shut down cleanly, within its bound: {served:?}"
            );
            drain_streams(&svc).await;
        })
        .await;
    }

    // ── The listener closes with the drain (b.kgb test 10) ───────────────────

    /// **Once the drain ends, a new connect is refused** while a stream
    /// opened earlier is still being served: the drain closes the listening
    /// socket, so a stale client's connect fails at once and is retried
    /// elsewhere instead of sitting in a backlog nobody accepts through the
    /// whole paced ending. One stream at 1/s ends in its batch at t0 + 0.9 s,
    /// so the connect, made the moment the listener is closed, comes well
    /// before it; the stream then still ends with the shutdown status.
    ///
    /// A listener left open until the server returns — the behaviour before
    /// this — accepts the connect into its backlog.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn once_the_drain_ends_a_new_connect_is_refused_while_an_open_stream_is_still_served() {
        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, _object_id) =
            create_session_and_layer(&tv, &svc, "kgb-listener", LAYER).await;
        let pacing = StreamPacing {
            rate_per_sec: 1,
            max_window: Duration::from_secs(20),
        };
        let mut server = RealServer::start(&svc, pacing, Duration::from_secs(30)).await;
        let mut client = server.client().await;
        let mut subscription = subscribe_over(&mut client, &session_id, "listener-reader").await;
        let _ = read_catch_up(&mut subscription, "the subscription's catch-up").await;
        let probe = tokio::time::timeout(SETTLE, tokio::net::TcpStream::connect(server.addr))
            .await
            .expect("connect within the deadline");
        assert!(
            probe.is_ok(),
            "precondition: the listener accepts a connect during the drain; got {probe:?}"
        );
        drop(probe);

        server.end_drain();
        let deadline = tokio::time::Instant::now() + SETTLE;
        while !server.listener.is_closed() {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the drain did not close the listener within {SETTLE:?}"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let refused = tokio::time::timeout(SETTLE, tokio::net::TcpStream::connect(server.addr))
            .await
            .expect("a connect to a closed listener fails at once");
        assert_eq!(
            refused.as_ref().err().map(std::io::Error::kind),
            Some(std::io::ErrorKind::ConnectionRefused),
            "once the drain has ended a new connect must be refused, not queued; got {refused:?}"
        );
        assert!(
            server.listener.open_connections() >= 1,
            "the subscription's connection is still open when the connect is refused"
        );

        let ending = read_to_terminal(&mut subscription, "the subscription after the drain")
            .await
            .expect("the subscription ended OK, which the Kit client never retries");
        assert_shutdown_status(&ending);
        drop(subscription);
        drop(client);
        let (served, _) = tokio::time::timeout(SERVER_EXIT, server.task)
            .await
            .expect("the server must return once its streams have ended")
            .expect("the server task must not panic");
        assert!(matches!(served, Some(Ok(()))), "{served:?}");
        drain_streams(&svc).await;
    }

    // ── The bound after GOAWAY (b.kgb test 9, b.uqx) ─────────────────────────

    /// The bound's WARN line, word for word.
    const BOUND_WARN_MSG: &str = "the gRPC server did not close every connection within its \
                                  bound after GOAWAY (the later of LIVE_EDIT_RPC_DEADLINE_MS \
                                  and the end of the paced stream ending, plus the close \
                                  grace); continuing the shutdown sequence — the connections \
                                  still open finish flushing on their own or end with the \
                                  process";

    /// A small RPC deadline, so the bound is the 5 s close grace plus 200 ms.
    const TEST_RPC_DEADLINE: Duration = Duration::from_millis(200);

    /// How long past its bound the server may be given up on: generous, so
    /// only a missing bound — a wait with no end — fails it.
    const BOUND_SLACK: Duration = Duration::from_secs(10);

    /// **A client that stopped reading cannot hold the shutdown past the bound
    /// after GOAWAY.** One subscription's client advertises a zero HTTP/2
    /// stream window and never reads, so nothing the server sends on it —
    /// its catch-up, then its shutdown ending behind it — can be flushed, and
    /// its connection can never finish closing. With the pacing off (the
    /// window ends at t0) and a 200 ms RPC deadline, the bound is t0 +
    /// 200 ms + `STREAM_CLOSE_GRACE`:
    ///
    /// * the bounded serve returns `None` (the bound fired) no earlier than
    ///   that, and no more than [`BOUND_SLACK`] after it;
    /// * it logs one WARN, counting the one connection still open, with the
    ///   numbers the bound came from;
    /// * the post-server drain then runs: every subscription and join task
    ///   has ended, and the shutdown handle is cancelled;
    /// * the stalled connection is left to its own task rather than torn
    ///   down: it stays counted open until its client goes away.
    ///
    /// Before the bound, the server waited for that connection with no limit,
    /// and the shutdown sequence never got past it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_client_that_stopped_reading_cannot_hold_the_server_past_its_bound_after_goaway() {
        use live_edit::grpc::transport::STREAM_CLOSE_GRACE;

        let tv = TestValkey::setup().await;
        let svc = build_service(&tv, Arc::new(AllowAllPermissions));
        let (session_id, object_id) =
            create_session_and_layer(&tv, &svc, "kgb-stalled-reader", LAYER).await;
        let mut server =
            RealServer::start(&svc, StreamPacing::ALL_AT_ONCE, TEST_RPC_DEADLINE).await;
        // A zero stream window: the server may send this client headers but no
        // DATA until the client grants window, which a client that never
        // reads never does.
        let mut stalled = server
            .client_from(
                Endpoint::from_shared(format!("http://{}", server.addr))
                    .expect("valid endpoint")
                    .initial_stream_window_size(0),
            )
            .await;
        let unread = subscribe_over(&mut stalled, &session_id, "stalled-reader").await;
        // A live update queued behind the unread catch-up as well.
        commit(&svc, &session_id, object_id, 11).await;
        assert_eq!(
            svc.subscription_tracker.len(),
            1,
            "precondition: one subscription"
        );

        server.end_drain();
        let drained_at = tokio::time::Instant::now();
        let bound = STREAM_END_DELAY_AFTER_GOAWAY + TEST_RPC_DEADLINE + STREAM_CLOSE_GRACE;
        let (served, captured) = tokio::time::timeout(bound + BOUND_SLACK, server.task)
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "the server was still waiting {:?} after the drain ended: the wait after \
                     GOAWAY has no bound",
                    bound + BOUND_SLACK
                )
            })
            .expect("the server task must not panic");
        let waited = drained_at.elapsed();

        assert!(
            served.is_none(),
            "the server cannot have closed a connection whose client never reads; the bound \
             must have fired. Got {served:?}"
        );
        assert!(
            waited >= bound,
            "the server was given up on {waited:?} after the drain, inside its {bound:?} bound"
        );
        let warn = captured.only(BOUND_WARN_MSG);
        assert_eq!(warn["level"], "WARN", "{warn}");
        assert_field(warn, "open_connections", 1.into());
        assert_field(warn, "rpc_deadline_ms", 200.into());
        assert_field(warn, "close_grace_ms", 5_000.into());
        assert!(
            svc.shutdown.is_cancelled(),
            "the shutdown handle is cancelled once the bound has fired"
        );

        // The post-server drain, as `main.rs` runs it next.
        drain_streams(&svc).await;

        // The stalled connection was left to its own task, not torn down:
        // it stays counted until its client goes away, then the count drops.
        assert_eq!(server.listener.open_connections(), 1);
        drop(unread);
        drop(stalled);
        let deadline = tokio::time::Instant::now() + SETTLE;
        while server.listener.open_connections() != 0 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "the stalled connection is still counted open {SETTLE:?} after its client went \
                 away"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

// =============================================================================
// Reconnect onto a second replica (owed test 12)
// =============================================================================

/// **A subscriber shut down on replica A reconnects on replica B and loses
/// nothing written in between** (in window).
///
/// A reference client that retries only `UNAVAILABLE` holds a subscription on
/// A, caught up and past one live update. A is shut down through
/// `end_streams_after_drain`, as `main.rs` does. The client must see a
/// retryable ending — or it would stop there — and, before it reconnects, N
/// commits land through B. Re-subscribing on B with the last `(object_id,
/// sequence_num)` it holds, it must receive all N, in order, then its
/// caught-up ack, then the next live update; and the state it has mirrored
/// from everything it received must equal the server's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_subscriber_reconnecting_in_window_after_shutdown_receives_every_missed_update() {
    /// Commits made while the client is between replicas.
    const MISSED: u64 = 5;

    scenario(async {
        let tv = TestValkey::setup().await;
        let svc_a = replica_over(&tv, tv.client().clone(), 4096);
        let (svc_b, _client_b) = replica(&tv, 4096).await;
        let (session_id, object_id) =
            create_session_and_layer(&tv, &svc_a, "e1h-reconnect-in-window", LAYER).await;

        let mut mirror = ObjectState::default();
        let mut on_a: SubscribeStream =
            handle_subscribe_read_layer(&svc_a, subscribe_request(&session_id, 0, 0, "ref-client"))
                .await
                .unwrap_or_else(|s| panic!("subscribe on A: {s:?}"))
                .into_inner();
        let (bodies, ack) = read_catch_up(&mut on_a, "the catch-up on A").await;
        for body in &bodies {
            apply_ompf(&mut mirror, body);
        }
        let committed = commit(&svc_a, &session_id, object_id, 100).await;
        let (live, body) = next_message(&mut on_a, "a live update on A").await;
        assert_eq!(live.sequence_num, committed);
        apply_ompf(&mut mirror, &body);
        let held = (ack.object_id, live.sequence_num);

        end_streams_after_drain(
            async {},
            svc_a.shutdown.clone(),
            StreamPacing::ALL_AT_ONCE,
            rand::rngs::StdRng::seed_from_u64(0),
        )
        .await;
        let ending = read_to_terminal(&mut on_a, "A's subscription on shutdown").await;
        assert!(
            reference_client_retries(&ending),
            "A's subscription ended {ending:?}. A client that retries only UNAVAILABLE stops \
             here and never sees the updates made after this point"
        );

        let mut missed = Vec::new();
        for i in 0..MISSED {
            missed.push(commit(&svc_b, &session_id, object_id, 200 + i).await);
        }

        let mut on_b: SubscribeStream = handle_subscribe_read_layer(
            &svc_b,
            subscribe_request(&session_id, held.0, held.1, "ref-client"),
        )
        .await
        .unwrap_or_else(|s| panic!("the retry on B must be admitted: {s:?}"))
        .into_inner();
        let mut received = Vec::new();
        for _ in 0..MISSED {
            let (header, body) = next_message(&mut on_b, "a missed update on B").await;
            assert!(header.total_size > 0, "a missed update must carry content");
            received.push(header.sequence_num);
            apply_ompf(&mut mirror, &body);
        }
        assert_eq!(
            received, missed,
            "the reconnect must deliver every update made while the client was away, in order"
        );
        let (caught_up, _) = next_message(&mut on_b, "the caught-up ack on B").await;
        assert_eq!(
            caught_up.total_size, 0,
            "the replay ends with the caught-up ack"
        );
        assert_eq!(caught_up.sequence_num, *missed.last().expect("MISSED > 0"));

        let next = commit(&svc_b, &session_id, object_id, 300).await;
        let (live, body) = next_message(&mut on_b, "a live update on B after the reconnect").await;
        assert_eq!(live.sequence_num, next);
        apply_ompf(&mut mirror, &body);

        let server = live_edit::storage::diff_read::snapshot_to_object_state(
            &live_edit::storage::test_support::read_object_state_for_test(
                &svc_b.valkey,
                &svc_b.key_schema,
                &object_id.to_string(),
            )
            .await
            .expect("load the server's snapshot"),
        );
        assert_eq!(
            structure(&mirror),
            structure(&server),
            "the reconnected client's reconstruction must equal the server's state"
        );

        drop(on_b);
        drain_streams(&svc_a).await;
    })
    .await;
}

/// **The same reconnect, out of window**: more commits land while the client
/// is away than the stream retains (`layer_stream_maxlen` 4, and the stream
/// trimmed past the client's position so the regime is deterministic). The
/// retry on B falls to the out-of-window catch-up, and the state the client
/// reconstructs from what it received must equal what a brand-new subscriber
/// on B reconstructs — and the next live update must still arrive.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_subscriber_reconnecting_out_of_window_after_shutdown_converges_to_a_fresh_subscribers_state(
) {
    use fred::interfaces::StreamsInterface;
    use fred::types::streams::XCap;

    /// The configured `layer_stream_maxlen` on both replicas.
    const MAXLEN: u64 = 4;
    /// Commits made while the client is between replicas: more than `MAXLEN`.
    const MISSED: u64 = 3 * MAXLEN;

    scenario(async {
        let tv = TestValkey::setup().await;
        let svc_a = replica_over(&tv, tv.client().clone(), MAXLEN);
        let (svc_b, _client_b) = replica(&tv, MAXLEN).await;
        let (session_id, object_id) =
            create_session_and_layer(&tv, &svc_a, "e1h-reconnect-oow", LAYER).await;

        let mut mirror = ObjectState::default();
        let mut on_a: SubscribeStream =
            handle_subscribe_read_layer(&svc_a, subscribe_request(&session_id, 0, 0, "ref-client"))
                .await
                .unwrap_or_else(|s| panic!("subscribe on A: {s:?}"))
                .into_inner();
        let (bodies, ack) = read_catch_up(&mut on_a, "the catch-up on A").await;
        for body in &bodies {
            apply_ompf(&mut mirror, body);
        }
        let held = (ack.object_id, ack.sequence_num);

        end_streams_after_drain(
            async {},
            svc_a.shutdown.clone(),
            StreamPacing::ALL_AT_ONCE,
            rand::rngs::StdRng::seed_from_u64(0),
        )
        .await;
        let ending = read_to_terminal(&mut on_a, "A's subscription on shutdown").await;
        assert!(
            reference_client_retries(&ending),
            "A's subscription ended {ending:?}; a client that retries only UNAVAILABLE stops here"
        );

        let mut last = 0;
        for i in 0..MISSED {
            last = commit(&svc_b, &session_id, object_id, 200 + i).await;
        }
        // Approximate `MAXLEN ~` trimming does not clip a stream this short, so
        // make the regime deterministic: keep only the newest entry.
        let _: () = tv
            .client()
            .pool
            .xtrim(
                tv.key_schema().layer_stream(object_id),
                XCap::try_from(("MINID", "=", format!("{last}-0").as_str())).expect("MINID cap"),
            )
            .await
            .expect("XTRIM must succeed");

        let mut on_b: SubscribeStream = handle_subscribe_read_layer(
            &svc_b,
            subscribe_request(&session_id, held.0, held.1, "ref-client"),
        )
        .await
        .unwrap_or_else(|s| panic!("the retry on B must be admitted: {s:?}"))
        .into_inner();
        let (bodies, caught_up) = read_catch_up(&mut on_b, "the out-of-window catch-up on B").await;
        assert!(
            !bodies.is_empty(),
            "the out-of-window catch-up must carry the missed content"
        );
        for body in &bodies {
            apply_ompf(&mut mirror, body);
        }
        assert_eq!(caught_up.sequence_num, last);
        // The regime this test is named for: converging to a fresh
        // subscriber's state would hold just as well had the retry been routed
        // as a brand-new subscriber, so the retry's catch-up regime is read
        // off B before the fresh subscriber below adds a `brand_new` of its own.
        assert_eq!(
            regime_count(&svc_b, "out_of_window"),
            1,
            "the retry must be served by the out-of-window catch-up"
        );
        assert_eq!(
            regime_count(&svc_b, "brand_new"),
            0,
            "the retry holds a position, so it must not be served as a brand-new subscriber"
        );

        let mut fresh_state = ObjectState::default();
        let mut fresh: SubscribeStream =
            handle_subscribe_read_layer(&svc_b, subscribe_request(&session_id, 0, 0, "fresh"))
                .await
                .unwrap_or_else(|s| panic!("a fresh subscribe on B: {s:?}"))
                .into_inner();
        let (fresh_bodies, fresh_ack) =
            read_catch_up(&mut fresh, "a fresh subscriber's catch-up").await;
        for body in &fresh_bodies {
            apply_ompf(&mut fresh_state, body);
        }
        assert_eq!(fresh_ack.sequence_num, last);
        assert_eq!(
            structure(&mirror),
            structure(&fresh_state),
            "after an out-of-window reconnect the client must hold what a fresh subscriber holds"
        );
        assert_eq!(
            structure(&mirror).len(),
            MISSED as usize,
            "fixture self-check: every missed section is in the reconstruction"
        );

        let next = commit(&svc_b, &session_id, object_id, 300).await;
        let (live, _) = next_message(&mut on_b, "a live update on B after the reconnect").await;
        assert_eq!(live.sequence_num, next);

        drop(on_b);
        drop(fresh);
        drain_streams(&svc_a).await;
    })
    .await;
}

/// **A join shut down on replica A rejoins on replica B**: its stream ends
/// `UNAVAILABLE` (so a client that retries only that code rejoins), and on B it
/// gets `SessionConnected`, then the membership replay — the peer that was
/// there all along — and then a live event from that peer.
///
/// The peer is joined on B, and B runs the Valkey-to-router presence bridge
/// `main.rs` runs, so a peer's message reaches B's local joins.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_join_shut_down_on_one_replica_rejoins_on_another_with_the_replay_and_live_events() {
    scenario(async {
        let tv = TestValkey::setup().await;
        let svc_a = replica_over(&tv, tv.client().clone(), 4096);
        let (svc_b, client_b) = replica(&tv, 4096).await;
        let bridge_cancel = CancellationToken::new();
        let bridge = live_edit::presence::spawn_presence_subscriber(
            client_b.subscriber.clone(),
            Arc::clone(&svc_b.presence_router),
            tv.key_schema().clone(),
            Arc::clone(&svc_b.metrics),
            bridge_cancel.clone(),
        );
        let (session_id, _token) = create_session(&tv, &svc_a, "e1h-rejoin", true).await;

        let (mut peer, peer_tx) = open_join(&svc_b, &session_id, "peer-app").await;
        let peer_cid = expect_session_connected(&mut peer, SETTLE).await;

        let (mut on_a, client_tx_a) = open_join(&svc_a, &session_id, "ref-app").await;
        let _ = expect_session_connected(&mut on_a, SETTLE).await;

        end_streams_after_drain(
            async {},
            svc_a.shutdown.clone(),
            StreamPacing::ALL_AT_ONCE,
            rand::rngs::StdRng::seed_from_u64(0),
        )
        .await;
        let (_, ending) = read_to_end(&mut on_a, "the join on A at shutdown").await;
        assert!(
            reference_client_retries(&ending),
            "the join on A ended {ending:?}; a client that retries only UNAVAILABLE would not \
             rejoin"
        );
        drain_tracker(&svc_a).await;

        drop(client_tx_a);
        let (mut on_b, client_tx_b) = open_join(&svc_b, &session_id, "ref-app").await;
        let cid_b = expect_session_connected(&mut on_b, SETTLE).await;
        assert_ne!(cid_b, peer_cid);
        let replayed = tokio::time::timeout(SETTLE, async {
            loop {
                let msg = on_b
                    .next()
                    .await
                    .expect("the rejoined stream ended during its replay")
                    .unwrap_or_else(|s| panic!("the rejoined stream failed: {s:?}"));
                if let Some(join_session_response::ResponseType::PeerJoined(p)) = msg.response_type
                {
                    return p.connection_id;
                }
            }
        })
        .await
        .expect("the replay on B must deliver the peer that is already joined");
        assert_eq!(replayed, peer_cid, "the replay must name the existing peer");

        peer_tx
            .send(encode_join_frame(&join_message(b"after-rejoin".to_vec())))
            .expect("the peer's client channel accepts a message");
        let live = tokio::time::timeout(SETTLE, async {
            loop {
                let msg = on_b
                    .next()
                    .await
                    .expect("the rejoined stream ended before the live event")
                    .unwrap_or_else(|s| panic!("the rejoined stream failed: {s:?}"));
                if let Some(join_session_response::ResponseType::PeerMessage(m)) = msg.response_type
                {
                    return (m.connection_id, m.data);
                }
            }
        })
        .await
        .expect("a live event from the peer must reach the rejoined stream");
        assert_eq!(live, (peer_cid.clone(), b"after-rejoin".to_vec()));

        drop(peer_tx);
        drop(client_tx_b);
        let _ = read_to_end(&mut on_b, "the rejoined stream at the end").await;
        let _ = read_to_end(&mut peer, "the peer at the end").await;
        drain_tracker(&svc_b).await;
        bridge_cancel.cancel();
        let _ = tokio::time::timeout(SETTLE, bridge).await;
    })
    .await;
}

// =============================================================================
// The pool quit after the drain, on a multi-shard cluster
// =============================================================================

/// **On a multi-shard cluster, the pool quit that ends the shutdown sequence is
/// not held up by the subscriptions' server-side reads, on any shard.**
///
/// `SUBSCRIPTIONS` idle, caught-up subscribers on layers in distinct hash
/// slots, so in cluster mode each is read by its own slot-bound stream reader
/// parked in `XREAD BLOCK`, with those streams on at least two different
/// shards. The test runs the shutdown sequence's last three steps as `main.rs`
/// does — cancel the service shutdown token, `drain_stream_tasks`,
/// `ValkeyClient::quit` — with nothing in between. The drain ends every
/// subscription, but removals do not wake a reader, so each reader's
/// `XREAD BLOCK` stays blocked on its own shard, and a `QUIT` queued behind one
/// would overrun the 2 s `QUIT_TIMEOUT` and log the reader pool's quit-timeout
/// WARN. The stream multiplexer's shutdown XADDs each reader's wake key, which
/// hashes to that reader's slot and so reaches the shard its read was routed
/// to; the quit must log none of the reader pool's WARNs and report every one
/// of the `SUBSCRIPTIONS` reads woken.
///
/// The service's `ValkeyClient` is its own, sized to one reader per
/// subscription and quit by the test; `TestValkey`'s client is left running for its key
/// cleanup, which would otherwise `SCAN` through a quit client and never
/// complete.
///
/// Self-gates on `LIVE_EDIT_TEST_CLUSTER_URL` and prints a `SKIP:` line when it
/// is unset. CI's `test-cluster` job runs it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn on_a_cluster_the_pool_quit_after_the_drain_is_not_held_by_blocked_reads_on_any_shard() {
    use fred::interfaces::{ClientLike, ClusterInterface};
    use std::collections::BTreeSet;

    /// Idle subscriptions, each on its own slot.
    const SUBSCRIPTIONS: usize = 4;
    /// Sessions to try before concluding the cluster cannot put two layer
    /// streams on different shards.
    const LAYER_CANDIDATES: usize = 64;

    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    scenario(async {
        let tv = TestValkey::setup_with_url(cluster_url.clone(), 4).await;
        assert!(
            tv.client().pool.next().is_clustered(),
            "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment (redis-cluster://...)"
        );
        let config = live_edit::config::Config {
            valkey_url: cluster_url,
            subscriber_pool_size: SUBSCRIPTIONS as u32,
            valkey_connect_timeout_secs: 30,
            ..crate::common::test_config()
        };
        let client = live_edit::storage::test_support::connect_test_client(&config)
            .await
            .expect("the service's own cluster ValkeyClient must connect");
        let svc = replica_over(&tv, client.clone(), 4096);

        // Sessions whose layer streams cover at least two shards.
        let routing = tv
            .client()
            .pool
            .next()
            .cached_cluster_state()
            .expect("a connected clustered client must have a cached CLUSTER SLOTS view");
        let owner = |object_id: u64| {
            let key = tv.key_schema().layer_stream(object_id);
            routing
                .get_server(fred::util::redis_keyslot(key.as_bytes()))
                .expect("every slot has a primary")
                .clone()
        };
        let mut chosen: Vec<(String, fred::types::config::Server)> = Vec::new();
        for i in 0..LAYER_CANDIDATES {
            let (session_id, object_id) =
                create_session_and_layer(&tv, &svc, &format!("e1h-cluster-quit-{i}"), LAYER).await;
            let shard = owner(object_id);
            let shards: BTreeSet<String> = chosen.iter().map(|(_, s)| s.to_string()).collect();
            // Fill all but the last place with whatever comes; keep the last
            // for a second shard if the others share one.
            let needs_second_shard = chosen.len() == SUBSCRIPTIONS - 1 && shards.len() == 1;
            if !needs_second_shard || !shards.contains(&shard.to_string()) {
                chosen.push((session_id, shard));
            }
            if chosen.len() == SUBSCRIPTIONS {
                break;
            }
        }
        let shards: BTreeSet<String> = chosen.iter().map(|(_, s)| s.to_string()).collect();
        assert!(
            chosen.len() == SUBSCRIPTIONS && shards.len() >= 2,
            "fixture: {LAYER_CANDIDATES} layers did not give {SUBSCRIPTIONS} streams over two \
             shards; chose {} over {shards:?}",
            chosen.len()
        );

        // Every subscription idle and caught up: parked in `XREAD BLOCK`.
        let mut streams: Vec<SubscribeStream> = Vec::with_capacity(SUBSCRIPTIONS);
        for (i, (session_id, _)) in chosen.iter().enumerate() {
            let mut stream: SubscribeStream = handle_subscribe_read_layer(
                &svc,
                subscribe_request(session_id, 0, 0, &format!("cluster-reader-{i}")),
            )
            .await
            .unwrap_or_else(|s| panic!("subscription {i} must be admitted: {s:?}"))
            .into_inner();
            let _ = read_catch_up(&mut stream, "a cluster subscriber's catch-up").await;
            streams.push(stream);
        }

        // One slot-bound reader per distinct slot, each parked in its read. A
        // caught-up ack does not mean the reader has issued that read yet: a
        // reader probes its connection, and computes its slot's wake-key tag,
        // before its first XREAD, and a reader still there when shutdown
        // begins issues no read at all, so there is none to wake. So wait for
        // every bound reader to have issued a read and then for the reads to
        // stop: every reader is then parked in its BLOCK.
        crate::storage::mux_test_helpers::await_mux(
            &client,
            "every bound reader has issued its read",
            |s| {
                s.readers
                    .iter()
                    .filter(|r| r.subscriptions > 0)
                    .all(|r| r.xreads >= 1)
            },
        )
        .await;
        crate::storage::mux_test_helpers::await_readers_parked(&client).await;
        let busy_readers = client.stream_mux.snapshot().active_readers() as u64;
        assert!(busy_readers >= 2, "fixture: reads span two shards");

        // The sequence's last three steps, back to back.

        svc.shutdown.cancel();
        drain_stream_tasks(
            &svc.presence_cleanup_tracker,
            &svc.subscription_tracker,
            SETTLE,
        )
        .await;
        assert!(
            svc.subscription_tracker.is_empty(),
            "the subscription drain did not finish within {SETTLE:?}; {} left",
            svc.subscription_tracker.len()
        );
        let ((), captured) = super::log_capture::capture_debug(client.quit()).await;
        super::log_capture::assert_quit_was_not_held_by_a_blocked_read(&captured, busy_readers);

        for (i, stream) in streams.iter_mut().enumerate() {
            let ending = read_to_terminal(stream, "a cluster subscriber on shutdown")
                .await
                .unwrap_or_else(|| panic!("subscription {i} ended OK on shutdown"));
            assert_eq!(
                ending.code(),
                Code::Unavailable,
                "subscription {i}: {ending:?}"
            );
        }
    })
    .await;
}
