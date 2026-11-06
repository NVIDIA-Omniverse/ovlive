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

//! The disaster-recovery hold ending through the real streaming handlers (SDD
//! SR-6.3, SR-6.5): `GracefulShutdown::end_streams_for_hold` ends every live
//! `SubscribeReadLayer` and `JoinSession` at once, each by its own stream's
//! cause, without shutting anything down.
//!
//! * **The endings, by cause.** An idle, caught-up subscription and an
//!   established join end `UNAVAILABLE` `SERVICE_HELD` naming the incident for
//!   a fleet hold — counted under `reason="held"`, never `shutdown`, and logged
//!   at INFO — and `UNAVAILABLE` `UPSTREAM_UNAVAILABLE` for a stale hold
//!   verdict, counted under `stale_verdict` and logged at WARN. The join's
//!   teardown writes still run (its presence entry is removed and the session
//!   unindexed), no per-session token is cancelled, and `begun` and the
//!   service shutdown token stay uncancelled.
//! * **Late enrolment.** A join that arrives after the hold is born cancelled
//!   with the hold's cause (the never-cleared `held` flag) and ends before it
//!   writes anything to the store. (A subscription that arrives after the hold
//!   is refused at admission: `subscribe_admission_test.rs`.)
//! * **SIGTERM after a hold.** Streams the hold ended keep `SERVICE_HELD` even
//!   when SIGTERM's `cancel()` lands before they wake; a subscription that
//!   arrives afterwards is refused `server shutting down` (`closed` wins over
//!   `held`), and so is a join, at the handler's `begun` check — a
//!   shutdown-only site.
//!
//! The hold ending's mechanics — cause recorded before cancel, the `held`
//! flag, `closed` winning, the registry drained, `begun` and the parent left
//! alone — are pinned in `grpc::graceful_shutdown`'s unit tests. The other
//! subscription phases (mid-delivery, mid in-window catch-up, mid-catch-up, a
//! catch-up queued for its slot, a subscribe queued for admission) end by the
//! hold's cause in `subscribe_admission_test.rs`, and an upload waiting at the
//! external-value release gate ends `server shutting down` after a hold and a
//! SIGTERM in `ext_release_gate_test.rs`.
//!
//! # Log capture
//!
//! The completion lines come from the spawned stream tasks, so every test runs
//! on a current-thread runtime under a [`ThreadCapture`] and ends with
//! [`purge_keys`] (`TestValkey` cannot clean up there).
//!
//! # Wait budget
//!
//! Every wait is a deadline, not a measurement: `SETTLE` (10 s) per catch-up
//! drain (one deadline for the whole drain), awaited message, stream end or
//! polled condition, and the helpers' `PRESENCE_WAIT` (10 s) per join read or
//! tracker drain. The heaviest tests — the fleet-hold and stale-verdict
//! endings and the SIGTERM one — wait at most six times: the catch-up drain,
//! `SessionConnected`, the subscription's end, its exit poll, the join's end
//! and the tracker drain, about 60 s at every budget's full value, inside
//! nextest's 120 s hang-killer. The late-join test waits three times (about
//! 30 s). The healthy path pays none of it.

use std::sync::Arc;
use std::time::Duration;

use futures::{Stream, StreamExt};
use serde_json::Value;
use tonic::{Request, Status};

use live_edit::grpc::graceful_shutdown::HoldCause;
use live_edit::grpc::proto::{
    subscribe_read_layer_response, SubscribeReadLayerRequest, SubscribeReadLayerResponse,
};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{
    Metrics, STREAM_EXIT_HELD, STREAM_EXIT_REASONS, STREAM_EXIT_SHUTDOWN, STREAM_EXIT_STALE_VERDICT,
};
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;
use super::log_capture::{
    assert_common_fields, assert_field, purge_keys, Captured, ThreadCapture, RPC_FAILURE_MSG,
};
use crate::common::refusal_detail::{
    assert_service_held, assert_shutdown_status, assert_upstream_unavailable,
};

/// Upper bound on any single wait. Unreachable on the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// The one layer every test here uses.
const LAYER: &str = "hold-layer";

/// Bearer on every in-process request; the mock permission client ignores it.
const BEARER: &str = "hold-bearer";

/// The incident the fleet hold names, in the shape of SDD SR-4.1's ids.
const INCIDENT: &str = "20261013t1708-a3f9";

type SubscribeStream =
    std::pin::Pin<Box<dyn Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Send>>;

fn subscribe_request(session_id: &str, client_id: &str) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: LAYER.to_owned(),
        object_id: 0,
        sequence_num: 0,
        client_id: client_id.to_owned(),
    });
    request.extensions_mut().insert(BearerToken::new(BEARER));
    request
}

/// Open a subscription on `svc` and drain its catch-up through the caught-up
/// ack, under one `SETTLE` deadline: it is then parked in its idle wait.
async fn open_idle_subscription(
    svc: &LiveEditService,
    session_id: &str,
    client_id: &str,
) -> SubscribeStream {
    let mut stream: SubscribeStream =
        handle_subscribe_read_layer(svc, subscribe_request(session_id, client_id))
            .await
            .unwrap_or_else(|s| panic!("the subscription must be admitted: {s:?}"))
            .into_inner();
    tokio::time::timeout(SETTLE, async {
        loop {
            let msg = stream
                .next()
                .await
                .expect("the stream ended during catch-up")
                .unwrap_or_else(|s| panic!("catch-up failed: {s:?}"));
            if let Some(subscribe_read_layer_response::ResponseType::Header(h)) = msg.response_type
            {
                if h.total_size == 0 {
                    return;
                }
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("the catch-up did not reach its caught-up ack within {SETTLE:?}"));
    stream
}

/// Drain to the terminal status under one deadline; `None` for a clean end.
async fn drain_to_terminal(stream: &mut SubscribeStream) -> Option<Status> {
    tokio::time::timeout(SETTLE, async {
        while let Some(item) = stream.next().await {
            if let Err(status) = item {
                return Some(status);
            }
        }
        None
    })
    .await
    .unwrap_or_else(|_| panic!("the subscription did not end within {SETTLE:?}"))
}

fn stream_exits(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_subscriber_stream_exits_total",
        &[("reason", reason)],
    )
    .unwrap_or_else(|| panic!("the stream-exit series for reason={reason} is absent")) as u64
}

fn total_stream_exits(metrics: &Metrics) -> u64 {
    STREAM_EXIT_REASONS
        .iter()
        .map(|reason| stream_exits(metrics, reason))
        .sum()
}

/// Wait until the subscription's tail has counted its exit (it logs its line
/// first), then require exactly one exit, under `reason`.
async fn await_the_one_exit(metrics: &Metrics, reason: &str) {
    let deadline = tokio::time::Instant::now() + SETTLE;
    while total_stream_exits(metrics) == 0 {
        assert!(
            tokio::time::Instant::now() < deadline,
            "no stream exit counted within {SETTLE:?}"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(
        stream_exits(metrics, reason),
        1,
        "the subscription's one exit must be counted under `{reason}`; exits: {:?}",
        STREAM_EXIT_REASONS
            .iter()
            .map(|r| (*r, stream_exits(metrics, r)))
            .collect::<Vec<_>>()
    );
    assert_eq!(total_stream_exits(metrics), 1, "exactly one exit");
}

/// Every completion line `endpoint` logged.
fn failures_of<'c>(captured: &'c Captured, endpoint: &str) -> Vec<&'c Value> {
    captured
        .with_message(RPC_FAILURE_MSG)
        .into_iter()
        .filter(|e| e["fields"]["endpoint"] == endpoint)
        .collect()
}

/// The one completion line `endpoint` logged.
fn only_failure_of<'c>(captured: &'c Captured, endpoint: &str) -> &'c Value {
    let lines = failures_of(captured, endpoint);
    assert_eq!(
        lines.len(),
        1,
        "expected exactly one {endpoint} completion line; full capture:\n{}",
        captured.text
    );
    lines[0]
}

/// Which hold ending a test drives.
#[derive(Clone, Copy, Debug)]
enum HoldEnding {
    /// A fleet hold naming [`INCIDENT`].
    Held,
    /// This replica's hold verdict went stale.
    StaleVerdict,
}

impl HoldEnding {
    fn cause(self) -> HoldCause {
        match self {
            HoldEnding::Held => HoldCause::Held {
                incident: Some(INCIDENT.to_owned()),
            },
            HoldEnding::StaleVerdict => HoldCause::StaleVerdict,
        }
    }

    /// Assert the status a stream this ending ended carries, as a client
    /// decodes it.
    #[track_caller]
    fn assert_status(self, status: &Status) {
        match self {
            HoldEnding::Held => assert_service_held(status, Some(INCIDENT)),
            HoldEnding::StaleVerdict => assert_upstream_unavailable(status),
        }
    }

    fn exit_reason(self) -> &'static str {
        match self {
            HoldEnding::Held => STREAM_EXIT_HELD,
            HoldEnding::StaleVerdict => STREAM_EXIT_STALE_VERDICT,
        }
    }

    /// The level of each ended stream's completion line: a fleet hold is
    /// reported once on its own, so its per-stream lines are INFO; a stale
    /// verdict is a dependency outage, WARN.
    fn level(self) -> &'static str {
        match self {
            HoldEnding::Held => "INFO",
            HoldEnding::StaleVerdict => "WARN",
        }
    }
}

/// An idle subscription and an established join on one session; the hold
/// ending runs with `ending`'s cause. Both streams end by that cause, the
/// subscription counted and logged under its own reason, the join after its
/// teardown — and nothing is shut down and no session token is cancelled.
async fn a_hold_ends_an_idle_subscription_and_a_join_by_its_cause(ending: HoldEnding) {
    let what = format!("{ending:?}");
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, &format!("hold-{what}"), LAYER).await;
    let capture = ThreadCapture::start();

    let mut subscription = open_idle_subscription(&svc, &session_id, "hold-reader").await;
    let (mut join, _client) = open_join(&svc, &session_id, "hold-app").await;
    let cid = expect_session_connected(&mut join, SETTLE).await;
    let session_token = svc
        .session_tokens
        .get(&session_id)
        .expect("both streams registered the session's token")
        .clone();
    assert_eq!(
        svc.shutdown.enrolled(),
        2,
        "precondition: both streams are enrolled"
    );

    svc.shutdown.end_streams_for_hold(ending.cause());

    // SubscribeReadLayer.
    let status = drain_to_terminal(&mut subscription)
        .await
        .unwrap_or_else(|| panic!("{what}: the subscription ended cleanly (gRPC OK)"));
    ending.assert_status(&status);
    await_the_one_exit(&svc.metrics, ending.exit_reason()).await;
    assert_eq!(
        stream_exits(&svc.metrics, STREAM_EXIT_SHUTDOWN),
        0,
        "{what}: a hold is not a shutdown and must not be counted as one"
    );

    // JoinSession, after its teardown.
    let (_, join_status) = read_to_end(&mut join, &what).await;
    let join_status =
        join_status.unwrap_or_else(|| panic!("{what}: the join ended OK, which Kit never retries"));
    ending.assert_status(&join_status);
    drain_tracker(&svc).await;
    assert_eq!(
        presence_keys(&tv, &session_id).await,
        vec![tv.key_schema().presence_gone(&session_id, &cid)],
        "{what}: the join's teardown (`unregister_connection`) must have removed its member \
         entry, info hash and lease, leaving only its teardown marker"
    );
    assert!(
        !is_indexed(&tv, &session_id).await,
        "{what}: the session's only member left, so teardown (`remove_session_from_index`) \
         must unindex it"
    );

    // Not a shutdown, and not a session ending.
    assert!(
        !session_token.is_cancelled(),
        "{what}: a hold cancels no per-session token (SDD §7.8)"
    );
    assert!(
        !svc.shutdown.begun().is_cancelled() && !svc.shutdown.is_cancelled(),
        "{what}: a hold starts no shutdown"
    );
    assert_eq!(svc.shutdown.enrolled(), 0, "{what}: both streams are gone");

    // One completion line per stream, at the cause's level, never a shutdown.
    let captured = capture.snapshot();
    let sub_line = only_failure_of(&captured, "SubscribeReadLayer");
    assert_common_fields(
        sub_line,
        ending.level(),
        "SubscribeReadLayer",
        "stream",
        "UNAVAILABLE",
    );
    assert_field(sub_line, "reason", ending.exit_reason().into());
    assert_field(sub_line, "status_message", status.message().into());
    let join_line = only_failure_of(&captured, "JoinSession");
    assert_common_fields(
        join_line,
        ending.level(),
        "JoinSession",
        "stream",
        "UNAVAILABLE",
    );
    assert_field(join_line, "status_message", join_status.message().into());
    drop(capture);
    purge_keys(&tv).await;
}

/// A fleet hold ends both streams `UNAVAILABLE` `SERVICE_HELD` naming the
/// incident, with a retry delay in `[30 s, 35 s)`, each logged once at INFO,
/// the subscription counted under `held`.
#[tokio::test]
async fn a_fleet_hold_ends_an_idle_subscription_and_a_join_service_held_without_a_shutdown() {
    a_hold_ends_an_idle_subscription_and_a_join_by_its_cause(HoldEnding::Held).await;
}

/// A stale hold verdict ends both streams `UNAVAILABLE`
/// `UPSTREAM_UNAVAILABLE`, each logged once at WARN, the subscription counted
/// under `stale_verdict`.
#[tokio::test]
async fn a_stale_hold_verdict_ends_an_idle_subscription_and_a_join_upstream_unavailable() {
    a_hold_ends_an_idle_subscription_and_a_join_by_its_cause(HoldEnding::StaleVerdict).await;
}

/// **Late enrolment.** Once the hold has run, a join that arrives — on a
/// replica whose `begun` is untouched, so the handler's own shutdown check
/// admits it — is born cancelled carrying the hold's cause: its task ends
/// `SERVICE_HELD` at its task-start check, before it writes anything — no
/// presence entry, no session-index entry — and logs one INFO line. Without
/// the never-cleared `held` flag it would be served during the hold. (The
/// subscription twin is
/// `subscribe_admission_test::a_subscribe_after_a_hold_is_refused_service_held_before_any_task_is_spawned`.)
#[tokio::test]
async fn a_join_arriving_after_a_hold_ends_service_held_and_writes_nothing() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "hold-late-join", LAYER).await;
    let capture = ThreadCapture::start();

    svc.shutdown.end_streams_for_hold(HoldEnding::Held.cause());
    assert!(
        !svc.shutdown.begun().is_cancelled(),
        "precondition: the handler's shutdown check cannot refuse; only the held flag can"
    );

    // Admitted by the handler, ended at its task-start check.
    let (mut join, _client) = open_join(&svc, &session_id, "late-app").await;
    let first = tokio::time::timeout(SETTLE, join.next())
        .await
        .unwrap_or_else(|_| panic!("the late join neither ended nor answered within {SETTLE:?}"));
    let join_status = match first {
        Some(Err(status)) => status,
        other => panic!(
            "a join arriving after the hold must end SERVICE_HELD before anything else; it got \
             {other:?} — it was served during the hold"
        ),
    };
    assert_service_held(&join_status, Some(INCIDENT));
    let (rest, end) = read_to_end(&mut join, "the late join after its status").await;
    assert!(
        rest.is_empty() && end.is_none(),
        "the status is the stream's only item"
    );
    drain_tracker(&svc).await;
    assert_eq!(
        presence_keys(&tv, &session_id).await,
        Vec::<String>::new(),
        "a join born cancelled writes nothing: no member entry, info hash, lease or teardown \
         marker"
    );
    assert!(
        !is_indexed(&tv, &session_id).await,
        "a join born cancelled never adds its session to the session index"
    );

    let captured = capture.snapshot();
    let join_line = only_failure_of(&captured, "JoinSession");
    assert_common_fields(join_line, "INFO", "JoinSession", "stream", "UNAVAILABLE");
    assert_field(join_line, "status_message", join_status.message().into());
    drop(capture);
    purge_keys(&tv).await;
}

/// **SIGTERM right after a hold.** The hold and SIGTERM's `cancel()` both land
/// before either stream runs again (current-thread runtime, no yield in
/// between), so each stream wakes to a token both cancelled. Each still ends
/// `SERVICE_HELD`: the hold recorded its cause before it cancelled, and the
/// stream is out of the registry, so `cancel()` neither cancels it again nor
/// changes its ending. The subscription is counted under `held` and logged at
/// INFO with `reason=held`, never as a shutdown. After `cancel()`, a new
/// subscription is born cancelled with no cause — `closed` wins over `held` —
/// and refused `server shutting down`; a new join is refused at the handler's
/// `begun` check, a shutdown-only site, `server shutting down` too.
#[tokio::test]
async fn sigterm_right_after_a_hold_keeps_the_hold_endings_and_refuses_later_streams_as_a_shutdown()
{
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _object_id) =
        create_session_and_layer(&tv, &svc, "hold-then-sigterm", LAYER).await;
    let capture = ThreadCapture::start();

    let mut subscription = open_idle_subscription(&svc, &session_id, "held-reader").await;
    let (mut join, _client) = open_join(&svc, &session_id, "held-app").await;
    let cid = expect_session_connected(&mut join, SETTLE).await;

    svc.shutdown.end_streams_for_hold(HoldEnding::Held.cause());
    svc.shutdown.cancel();

    let status = drain_to_terminal(&mut subscription)
        .await
        .expect("the subscription ended cleanly (gRPC OK)");
    assert_service_held(&status, Some(INCIDENT));
    await_the_one_exit(&svc.metrics, STREAM_EXIT_HELD).await;
    let (_, join_status) = read_to_end(&mut join, "the held join after SIGTERM").await;
    assert_service_held(
        &join_status.expect("the join ended OK, which Kit never retries"),
        Some(INCIDENT),
    );

    // A subscription after SIGTERM: born cancelled with no cause.
    let result = handle_subscribe_read_layer(&svc, subscribe_request(&session_id, "late-reader"))
        .await
        .map(|_| ());
    assert_shutdown_status(&expect_err_status(
        result,
        "a subscription arriving after SIGTERM was admitted",
    ));
    // A join after SIGTERM: refused at the handler's `begun` check.
    let (stream, tx) = open_join_streaming();
    tx.send(encode_join_frame(&join_header(&session_id, "late-app")))
        .expect("the client channel must accept the header frame");
    let result =
        live_edit::grpc::presence::handle_join_session(&svc, make_join_request(stream)).await;
    assert_shutdown_status(&expect_err_status(
        result.map(|_| ()),
        "a join arriving after SIGTERM was admitted",
    ));

    drain_tracker(&svc).await;
    assert_eq!(
        presence_keys(&tv, &session_id).await,
        vec![tv.key_schema().presence_gone(&session_id, &cid)],
        "the held join's teardown ran; the refused join wrote nothing"
    );

    let captured = capture.snapshot();
    let sub_lines = failures_of(&captured, "SubscribeReadLayer");
    assert_eq!(
        sub_lines.len(),
        2,
        "the held subscription's ending and the later refusal; full capture:\n{}",
        captured.text
    );
    let held_line = sub_lines
        .iter()
        .find(|e| e["fields"]["stage"] == "stream")
        .unwrap_or_else(|| panic!("no stream-stage line; full capture:\n{}", captured.text));
    assert_common_fields(
        held_line,
        "INFO",
        "SubscribeReadLayer",
        "stream",
        "UNAVAILABLE",
    );
    assert_field(held_line, "reason", STREAM_EXIT_HELD.into());
    assert!(
        captured
            .with_message(RPC_FAILURE_MSG)
            .iter()
            .all(|e| e["fields"]["reason"] != STREAM_EXIT_SHUTDOWN),
        "no stream ended as a shutdown; full capture:\n{}",
        captured.text
    );
    drop(tx);
    drop(capture);
    purge_keys(&tv).await;
}
