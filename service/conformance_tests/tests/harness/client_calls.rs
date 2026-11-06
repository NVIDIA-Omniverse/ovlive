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

//! [`Client`] on the wire: the deadline each call sends and enforces, and the
//! sessions and calls it owns and cleans up.
//!
//! An unanswered call is tested against a peer that never answers, so nothing
//! but the client can end it: tonic's own server would also enforce the
//! `grpc-timeout` header and hide a client that does not. Two such peers:
//!
//! - `support::silent_listener` never accepts the connection, so a call to it
//!   never reaches tonic's channel and only the suite's own timer can end it.
//!   Those tests run on paused time.
//! - `support::SilentServer` accepts and reads but never writes, so a call to
//!   it reaches tonic's channel, whose own `grpc-timeout` timer expires at the
//!   same instant as the suite's and ends the call as `CANCELLED` "Timeout
//!   expired" whenever it is polled first. Those tests pin that the client
//!   still reports `DEADLINE_EXCEEDED`: in real time with several calls at
//!   once, and with the clock frozen once the connection is up, so that both
//!   timers are set at the same instant and tonic's always ends the call.
//!
//! The suite's `with_deadline` itself, the status mapping on its own, and
//! `DEADLINE_EXCEEDED` ending a stream that is still open are pinned in
//! `client.rs`'s unit tests and by
//! `answered_subscribe_ends_deadline_exceeded_at_110_seconds_from_its_start`
//! here.

use std::time::Duration;

use futures::future::join_all;
use futures::FutureExt as _;
use live_edit_conformance::client::{Auth, CallError, Client, LIVE_SERVICE_NAME};
use live_edit_conformance::proto;
use tokio::time::Instant;
use tonic::{Code, Status};

use crate::fake_live::{Answer, FakeLive};
use crate::support::{grpc_timeout, silent_listener, SilentServer, Vars, TOKEN};

/// The deadline of `SubscribeReadLayer` and `JoinSession` the suite promises
/// (110 s), written out rather than read from `client::STREAM_DEADLINE`, so
/// that changing the constant fails these tests.
const STREAM_DEADLINE: Duration = Duration::from_secs(110);

/// How long a test against a peer that never answers may take in real time
/// before it fails rather than hangs.
const REAL_TIME_BUDGET: Duration = Duration::from_secs(30);

fn client(endpoint: &str, vars: Vars) -> Client {
    Client::new(
        vars.set("CONFORMANCE_GRPC_ENDPOINT", endpoint)
            .arc_settings(),
        None,
    )
    .unwrap()
}

/// The `grpc-timeout` of the only request received.
fn sent_timeout(server: &FakeLive) -> Duration {
    let received = server.received();
    assert_eq!(received.len(), 1, "{received:?}");
    grpc_timeout(
        received[0]
            .grpc_timeout
            .as_deref()
            .expect("a grpc-timeout header"),
    )
}

#[tokio::test]
async fn ping_within_sends_its_deadline_as_grpc_timeout_with_the_bearer_token() {
    let server = FakeLive::ok().await;
    let client = client(&server.endpoint, Vars::base());

    client
        .ping_within(proto::PingRequest::default(), Duration::from_millis(4250))
        .await
        .unwrap();

    assert_eq!(sent_timeout(&server), Duration::from_millis(4250));
    assert_eq!(
        server.received()[0].authorization.as_deref(),
        Some(format!("Bearer {TOKEN}").as_str())
    );
}

#[tokio::test]
async fn unary_and_upload_calls_send_the_configured_timeout() {
    let server = FakeLive::ok().await;
    let client = client(
        &server.endpoint,
        Vars::base().set("CONFORMANCE_TIMEOUT", "7"),
    );

    let session = client.session().await.unwrap();
    client.create(&session, b"payload").await.unwrap();

    let sent: Vec<(String, Duration)> = server
        .received()
        .into_iter()
        .map(|r| (r.method, grpc_timeout(&r.grpc_timeout.unwrap())))
        .collect();
    assert_eq!(
        sent,
        [
            ("CreateSession".to_string(), Duration::from_secs(7)),
            ("CreateLayer".to_string(), Duration::from_secs(7)),
        ]
    );
}

#[tokio::test]
async fn subscribe_and_join_send_the_110_second_stream_deadline() {
    // The fake server answers a subscription with one header and refuses a
    // join: either way the request (and its headers) arrived.
    let server = FakeLive::ok().await;
    let client = client(&server.endpoint, Vars::base());

    let mut stream = client.subscribe("session-1", 1, 1).await.unwrap();
    assert!(stream.next().await.is_ok());
    let (mut join, _sender) = client.join("session-1", false).await.unwrap();
    assert_eq!(
        join.next().await.unwrap_err().code(),
        Some(Code::Unimplemented)
    );

    let sent: Vec<(String, Duration)> = server
        .received()
        .into_iter()
        .map(|r| (r.method, grpc_timeout(&r.grpc_timeout.unwrap())))
        .collect();
    assert_eq!(
        sent,
        [
            ("SubscribeReadLayer".to_string(), STREAM_DEADLINE),
            ("JoinSession".to_string(), STREAM_DEADLINE),
        ]
    );
}

#[tokio::test]
async fn health_check_is_sent_without_credentials() {
    let server = FakeLive::ok().await;
    let client = client(&server.endpoint, Vars::base());

    let response = client
        .health(LIVE_SERVICE_NAME, Duration::from_secs(3))
        .await
        .unwrap();

    assert_eq!(
        response.status,
        tonic_health::pb::health_check_response::ServingStatus::Serving as i32
    );
    let received = server.received();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].method, "Check");
    assert_eq!(received[0].authorization, None);
}

/// The code a call no server answers ends with on the client, whichever
/// timer ended it: the suite's own deadline (`with_deadline`, `call_stream`)
/// or tonic's channel's `grpc-timeout` timer, whose `CANCELLED` "Timeout
/// expired" the client reports as the `DEADLINE_EXCEEDED` it is.
const UNANSWERED_CODE: Code = Code::DeadlineExceeded;

/// Asserts `elapsed` is `deadline`, give or take the timer wheel's
/// millisecond (paused time).
fn assert_at(elapsed: Duration, deadline: Duration) {
    assert!(
        elapsed >= deadline && elapsed <= deadline + Duration::from_millis(1),
        "ended after {elapsed:?}, not at {deadline:?}"
    );
}

#[tokio::test(start_paused = true)]
async fn unanswered_unary_call_ends_on_the_client_at_its_deadline() {
    let (_listener, endpoint) = silent_listener();
    let client = client(&endpoint, Vars::base());
    let deadline = Duration::from_secs(3);
    let started = Instant::now();

    let result = client
        .ping_within(proto::PingRequest::default(), deadline)
        .await;

    assert_eq!(result.unwrap_err().code(), Some(UNANSWERED_CODE));
    assert_at(started.elapsed(), deadline);
}

#[tokio::test(start_paused = true)]
async fn unanswered_subscribe_ends_on_the_client_at_110_seconds() {
    let (_listener, endpoint) = silent_listener();
    let client = client(&endpoint, Vars::base());
    let started = Instant::now();

    let mut stream = client.subscribe("session-1", 1, 1).await.unwrap();
    // Wait past the deadline, so only the call's own end can come.
    let ended = stream
        .next_until(started + STREAM_DEADLINE * 2)
        .await
        .unwrap_err();

    assert_eq!(ended.code(), Some(UNANSWERED_CODE), "{ended:?}");
    assert_at(started.elapsed(), STREAM_DEADLINE);
}

#[tokio::test(start_paused = true)]
async fn unanswered_join_ends_on_the_client_at_110_seconds() {
    let (_listener, endpoint) = silent_listener();
    let client = client(&endpoint, Vars::base());
    let started = Instant::now();

    let (mut join, _sender) = client.join("session-1", true).await.unwrap();
    let ended = join
        .next_until(started + STREAM_DEADLINE * 2)
        .await
        .unwrap_err();

    assert_eq!(ended.code(), Some(UNANSWERED_CODE), "{ended:?}");
    assert_at(started.elapsed(), STREAM_DEADLINE);
}

/// A client of `server` whose connection is up: one call to it has been
/// made, in real time, and ended unanswered. (On a channel still
/// connecting, tonic's timer starts only once the connection is up, after
/// the suite's own, so a first call shows little of tonic's timer: the
/// tests below make theirs on the connection this one opened.)
async fn connected_client(server: &SilentServer) -> Client {
    let client = client(&server.endpoint, Vars::base());
    let first = tokio::time::timeout(
        REAL_TIME_BUDGET,
        client.ping_within(proto::PingRequest::default(), Duration::from_millis(300)),
    )
    .await
    .expect("the first call ends at its deadline");
    assert!(matches!(first, Err(CallError::Status(_))), "{first:?}");
    server.wait_for_a_call(REAL_TIME_BUDGET).await;
    client
}

#[tokio::test]
async fn unanswered_unary_and_health_calls_on_a_connection_end_deadline_exceeded() {
    // Real time, several calls at once on a connection that is up: each
    // call's two timers expire within a millisecond of each other, and
    // tonic's ends most calls (`CANCELLED` "Timeout expired") before the
    // suite's own; every one must still be reported DEADLINE_EXCEEDED.
    const CALLS: usize = 8;
    let server = SilentServer::start().await;
    let client = connected_client(&server).await;
    let deadline = Duration::from_millis(300);
    let started = Instant::now();

    let pings = (0..CALLS).map(|_| {
        client
            .ping_within(proto::PingRequest::default(), deadline)
            .map(|result| result.map(drop))
            .boxed()
    });
    let checks = (0..CALLS).map(|_| {
        client
            .health(LIVE_SERVICE_NAME, deadline)
            .map(|result| result.map(drop))
            .boxed()
    });
    let results = tokio::time::timeout(REAL_TIME_BUDGET, join_all(pings.chain(checks)))
        .await
        .expect("every call ends at its deadline");

    assert!(started.elapsed() >= deadline);
    let codes: Vec<Option<Code>> = results
        .iter()
        .map(|result| result.as_ref().unwrap_err().code())
        .collect();
    assert_eq!(codes, vec![Some(UNANSWERED_CODE); 2 * CALLS], "{results:?}");
}

// With the connection up and the clock frozen, a call's own deadline and the
// `grpc-timeout` timer in tonic's channel are set at the same instant and
// fire together, and the call (polled before the suite's timer) always ends
// first with tonic's `CANCELLED` "Timeout expired": these tests fail
// whenever that status is not reported as DEADLINE_EXCEEDED.

#[tokio::test]
async fn unanswered_unary_call_ended_by_tonics_timer_is_deadline_exceeded() {
    let server = SilentServer::start().await;
    let client = connected_client(&server).await;
    tokio::time::pause();
    let deadline = Duration::from_secs(3);
    let started = Instant::now();

    let result = client
        .ping_within(proto::PingRequest::default(), deadline)
        .await;

    assert_eq!(result.unwrap_err().code(), Some(UNANSWERED_CODE));
    assert_at(started.elapsed(), deadline);
}

#[tokio::test]
async fn unanswered_health_check_ended_by_tonics_timer_is_deadline_exceeded() {
    let server = SilentServer::start().await;
    let client = connected_client(&server).await;
    tokio::time::pause();
    let deadline = Duration::from_secs(3);
    let started = Instant::now();

    let result = client.health(LIVE_SERVICE_NAME, deadline).await;

    assert_eq!(result.unwrap_err().code(), Some(UNANSWERED_CODE));
    assert_at(started.elapsed(), deadline);
}

#[tokio::test]
async fn unanswered_subscribe_ended_by_tonics_timer_is_deadline_exceeded_at_110_seconds() {
    // No response headers ever arrive: the stream ends before its first
    // message.
    let server = SilentServer::start().await;
    let client = connected_client(&server).await;
    tokio::time::pause();
    let started = Instant::now();

    let mut stream = client.subscribe("session-1", 1, 1).await.unwrap();
    let ended = stream
        .next_until(started + STREAM_DEADLINE * 2)
        .await
        .unwrap_err();

    assert_eq!(ended.code(), Some(UNANSWERED_CODE), "{ended:?}");
    assert_at(started.elapsed(), STREAM_DEADLINE);
}

#[tokio::test]
async fn unanswered_join_ended_by_tonics_timer_is_deadline_exceeded_at_110_seconds() {
    let server = SilentServer::start().await;
    let client = connected_client(&server).await;
    tokio::time::pause();
    let started = Instant::now();

    let (mut join, _sender) = client.join("session-1", false).await.unwrap();
    let ended = join
        .next_until(started + STREAM_DEADLINE * 2)
        .await
        .unwrap_err();

    assert_eq!(ended.code(), Some(UNANSWERED_CODE), "{ended:?}");
    assert_at(started.elapsed(), STREAM_DEADLINE);
}

#[tokio::test]
async fn server_cancelled_is_reported_cancelled_whenever_it_arrives() {
    // A CANCELLED the server sends, even one worded as tonic words its own
    // timer's, is the server's answer: never the client's deadline.
    let server = FakeLive::start(|method, n| match (method, n) {
        ("Ping", 1) => Answer::Fail(Status::cancelled("Timeout expired")),
        ("Ping", _) => Answer::After(
            Duration::from_millis(500),
            Box::new(Answer::Fail(Status::cancelled("Timeout expired"))),
        ),
        ("SubscribeReadLayer", _) => Answer::Fail(Status::cancelled("Timeout expired")),
        _ => Answer::Ok,
    })
    .await;
    let client = client(&server.endpoint, Vars::base());
    let deadline = Duration::from_secs(10);

    let at_once = client
        .ping_within(proto::PingRequest::default(), deadline)
        .await;
    let later = client
        .ping_within(proto::PingRequest::default(), deadline)
        .await;
    let mut stream = client.subscribe("session-1", 1, 1).await.unwrap();
    let refused = stream.next().await;

    assert_eq!(at_once.unwrap_err().code(), Some(Code::Cancelled));
    assert_eq!(later.unwrap_err().code(), Some(Code::Cancelled));
    assert_eq!(refused.unwrap_err().code(), Some(Code::Cancelled));
}

#[tokio::test]
async fn answered_subscribe_ends_deadline_exceeded_at_110_seconds_from_its_start() {
    // The server answers with one message and then sends nothing more, never
    // ending the stream. tonic bounds only the wait for the response's
    // headers, so after them only the suite's own stream deadline ends it.
    let server = FakeLive::ok().await;
    let client = client(&server.endpoint, Vars::base());
    let started = Instant::now();

    let mut stream = client.subscribe("session-1", 1, 1).await.unwrap();
    // In real time: paused time could jump past the wait for this message
    // while it is still on the wire.
    assert!(stream.next().await.is_ok(), "the first message arrives");
    // Nothing more crosses the wire: from here the clock may jump to the
    // stream's deadline, taken when the call started.
    tokio::time::pause();
    let ended = stream
        .next_until(started + STREAM_DEADLINE * 2)
        .await
        .unwrap_err();

    assert_eq!(ended.code(), Some(Code::DeadlineExceeded), "{ended:?}");
    // The real-time part (the call and its first message) adds a little.
    let elapsed = started.elapsed();
    assert!(
        elapsed >= STREAM_DEADLINE && elapsed <= STREAM_DEADLINE + Duration::from_secs(1),
        "ended after {elapsed:?}"
    );
}

#[tokio::test]
async fn close_deletes_owned_sessions_in_reverse_creation_order() {
    let server = FakeLive::ok().await;
    let client = client(&server.endpoint, Vars::base());
    for _ in 0..3 {
        client.session().await.unwrap();
    }
    assert_eq!(
        client.owned_sessions(),
        ["session-1", "session-2", "session-3"]
    );

    client.close().await.unwrap();

    assert_eq!(server.deleted(), ["session-3", "session-2", "session-1"]);
}

#[tokio::test]
async fn close_tolerates_not_found_and_reports_any_other_delete_failure_after_trying_all() {
    // The deletes run session-3, session-2, session-1.
    let server = FakeLive::start(|method, n| match (method, n) {
        ("DeleteSession", 1) => Answer::Fail(Status::not_found("already gone")),
        ("DeleteSession", 2) => Answer::Fail(Status::internal("store down")),
        _ => Answer::Ok,
    })
    .await;
    let client = client(&server.endpoint, Vars::base());
    for _ in 0..3 {
        client.session().await.unwrap();
    }

    let error = client.close().await.unwrap_err();

    assert_eq!(
        error.message(),
        "Resource cleanup failed: DeleteSession(session-2): Internal"
    );
    assert_eq!(server.deleted(), ["session-3", "session-2", "session-1"]);
}

#[tokio::test]
async fn create_session_unexpectedly_accepted_without_credentials_is_still_owned() {
    let server = FakeLive::ok().await;
    let client = client(&server.endpoint, Vars::base());

    // A negative case expects this refused; an implementation that accepts
    // it must not leak the session.
    client
        .create_session_as(
            proto::CreateSessionRequest {
                path: "/container/conformance".into(),
                name: "anonymous".into(),
            },
            Auth::Anonymous,
        )
        .await
        .unwrap();

    assert_eq!(server.received()[0].authorization, None);
    assert_eq!(client.owned_sessions(), ["session-1"]);
    client.close().await.unwrap();
    assert_eq!(server.deleted(), ["session-1"]);
}

#[tokio::test]
async fn create_session_refused_owns_nothing() {
    let server = FakeLive::start(|_, _| Answer::Fail(Status::unauthenticated("no token"))).await;
    let client = client(&server.endpoint, Vars::base());

    let refused = client.session().await.unwrap_err();

    assert_eq!(refused.code(), Some(Code::Unauthenticated));
    assert!(client.owned_sessions().is_empty());
}

#[tokio::test]
async fn close_cancels_an_upload_that_never_ends() {
    let server = FakeLive::ok().await;
    let client = client(&server.endpoint, Vars::base());
    let header = proto::UpdateLayerRequest {
        message_type: Some(proto::update_layer_request::MessageType::Header(
            proto::UpdateLayerHeader {
                session_id: "session-1".into(),
                layer_name: "layer".into(),
                object_id: 1,
                ..Default::default()
            },
        )),
    };
    let pending = client.unended_update_async(vec![header]).await;
    assert!(!pending.is_finished());

    client.close().await.unwrap();

    assert!(pending.is_finished());
    assert!(pending.result().await.is_err());
}
