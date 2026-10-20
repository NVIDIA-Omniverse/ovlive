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

// Integration tests for JoinSession per-session cleanup (bug b.i2e).
//
// The bug: a join's presence cleanup did not release the `LocalPresenceRouter`
// DashMap entry or the Valkey pub/sub subscription when the last local
// JoinSession stream disconnected. The fix is a per-session connection count
// the join's teardown releases through `router.leave_channel(session_id)`;
// when that was the session's last local connection, `leave_channel` hands
// teardown the duty to `unsubscribe_presence_channel`, and the router drops
// the entry once that `UNSUBSCRIBE` has completed.
//
// These tests drive `handle_join_session` through real Valkey and observe:
//   * `presence_router.has_session(sid)` transitions.
//   * Whether Valkey pub/sub events still reach the local router after the
//     last disconnect (inverted delivery test).
//
// The canonical regression test is
// `both_clients_disconnect_removes_router_entry_and_stops_pubsub_delivery`:
// before the fix the router entry persisted forever so `has_session` returned
// true, and the Valkey pub/sub subscription continued to deliver events into
// the now-empty broadcast channel.
//
// The service shutdown token (`LiveEditService.shutdown`, SDD §5.6, SR-4.7) is
// covered at the end of this file, at each point JoinSession reads it, and
// each ends the join `UNAVAILABLE "server shutting down"` — the ending the Kit
// client retries (where the retry lands is the ingress's choice), where `OK`
// would end its presence feed
// for good: the handler refuses a join that arrives after shutdown began, the
// join task's start-of-task check ends a join the handler already admitted
// before it writes anything, and the bidirectional loop's shutdown arm ends an
// established join after teardown. The handler and task-start tests run on a
// current-thread runtime: a task the handler spawns cannot run until the test
// yields, which makes "no task was spawned" and "the task had not started when
// shutdown began" observable. `TestValkey` cannot clean up on that runtime, so
// those tests end with `purge_keys`.
//
// In the shutdown-token tests every wait is a deadline (`SETTLE`, 10 s), never
// a sleep. The heaviest,
// `join_session_service_shutdown_ends_established_join_unavailable_after_teardown`,
// is at most 5 s (SessionConnected) + 5 × `SETTLE` ≈ 55 s, inside nextest's
// 120 s hang-killer.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tonic::Code;

use fred::interfaces::{ClientLike, KeysInterface, PubsubInterface, SetsInterface};
use live_edit::grpc::presence::handle_join_session;
use live_edit::grpc::proto::join_session_response;
use live_edit::grpc::service::LiveEditService;
use live_edit::presence::{publish_presence_event, spawn_presence_subscriber, PresenceEvent};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;
use super::log_capture::purge_keys;
use crate::common::refusal_detail::assert_shutdown_status;

/// Upper bound on any single wait in the shutdown-token tests. Unreachable on
/// the healthy path.
const SETTLE: Duration = Duration::from_secs(10);

/// Boxed response stream as returned by `handle_join_session`.
type ResponseStream = JoinResponseStream;

// ── Streaming helpers ────────────────────────────────────────────────────────
//
// Frame encoding and the channel-backed client stream (`encode_join_frame`,
// `join_header`, `join_message`, `open_join_streaming`, `make_join_request`,
// `open_join`, `expect_session_connected`) live in `super::helpers`.

async fn poll_until<F>(deadline: Duration, mut check: F) -> bool
where
    F: FnMut() -> bool,
{
    let start = Instant::now();
    while start.elapsed() < deadline {
        if check() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    check()
}

/// Kick off a JoinSession, wait for the SessionConnected message, and return
/// the response stream + frame sender. Dropping the returned sender closes the
/// client stream, which propagates as `Ok(None)` in the bidi loop and triggers
/// the cleanup path under test.
async fn start_join_session(
    svc: &LiveEditService,
    session_id: &str,
    user_app: &str,
) -> (ResponseStream, mpsc::UnboundedSender<bytes::Bytes>) {
    let (mut resp_stream, tx) = open_join(svc, session_id, user_app).await;
    // SessionConnected follows the router subscribe, so the subscribe has
    // landed in the router once it arrives.
    expect_session_connected(&mut resp_stream, Duration::from_secs(5)).await;
    (resp_stream, tx)
}

/// Drain the response stream until it ends or yields an error. Any remaining
/// items are discarded. This consumes the stream so that the drop of `tx` on
/// the client side has a chance to be observed in the server task.
async fn drain_response_stream(mut stream: ResponseStream) {
    let _ = tokio::time::timeout(Duration::from_secs(5), async {
        while let Some(next) = stream.next().await {
            if next.is_err() {
                break;
            }
        }
    })
    .await;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── 1. Two clients, one disconnects: router entry persists ──────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn one_of_two_clients_disconnects_router_keeps_session_entry() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Create the session and install a CancellationToken the way CreateSession does.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("two-client-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    svc.session_tokens
        .insert(session_id.clone(), CancellationToken::new().into());

    // First client joins.
    let (stream_a, tx_a) = start_join_session(&svc, &session_id, "app-a").await;
    // Second client joins the same session.
    let (stream_b, tx_b) = start_join_session(&svc, &session_id, "app-b").await;

    // Sanity: router now tracks the session.
    assert!(
        svc.presence_router.has_session(&session_id),
        "router must track the session while clients are connected"
    );
    assert_eq!(svc.presence_router.session_count(), 1);

    // Client A disconnects. Drop tx first so the server sees Ok(None), then
    // drain the response stream so the cleanup task runs to completion.
    drop(tx_a);
    drain_response_stream(stream_a).await;

    // Give the cleanup task a moment to run `leave_channel`. The router entry
    // must still exist because client B is still connected (counter went
    // 2 -> 1, not 1 -> 0).
    let still_present = poll_until(Duration::from_secs(2), || {
        svc.presence_router.has_session(&session_id)
    })
    .await;
    assert!(
        still_present,
        "router entry must persist while any local client remains connected"
    );
    assert_eq!(
        svc.presence_router.session_count(),
        1,
        "exactly one session should be tracked after partial disconnect"
    );

    // Teardown: disconnect B as well so server tasks exit cleanly.
    drop(tx_b);
    drain_response_stream(stream_b).await;
}

// ── 2. Both clients disconnect: router drops entry; pub/sub no longer delivers
//     (REGRESSION TEST for b.i2e) ──────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn both_clients_disconnect_removes_router_entry_and_stops_pubsub_delivery() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Install the shared presence subscriber task so pub/sub events from
    // Valkey get delivered into `svc.presence_router`. This matches the wiring
    // in `src/main.rs`.
    let presence_cancel = CancellationToken::new();
    let subscriber_handle = spawn_presence_subscriber(
        tv.client().subscriber.clone(),
        Arc::clone(&svc.presence_router),
        tv.key_schema().clone(),
        Arc::clone(&svc.metrics),
        presence_cancel.clone(),
    );

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("both-disconnect", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    svc.session_tokens
        .insert(session_id.clone(), CancellationToken::new().into());

    // Two clients join, then both disconnect.
    let (stream_a, tx_a) = start_join_session(&svc, &session_id, "app-a").await;
    let (stream_b, tx_b) = start_join_session(&svc, &session_id, "app-b").await;
    assert!(svc.presence_router.has_session(&session_id));
    assert_eq!(
        svc.shutdown.enrolled(),
        2,
        "precondition: both live joins are enrolled for graceful shutdown's paced ending"
    );

    drop(tx_a);
    drain_response_stream(stream_a).await;
    drop(tx_b);
    drain_response_stream(stream_b).await;

    // Router entry must be removed once the last subscriber disconnects.
    // Before the fix this assertion failed: the DashMap entry persisted
    // forever so `has_session` stayed true.
    let removed = poll_until(Duration::from_secs(2), || {
        !svc.presence_router.has_session(&session_id)
    })
    .await;
    assert!(
        removed,
        "router entry must be removed after the last local JoinSession disconnects \
         (regression: before the b.i2e fix this stayed true forever)"
    );
    assert_eq!(svc.presence_router.session_count(), 0);

    // Each join task's exit releases its shutdown enrollment: a leaked one
    // would leave the registry growing by one per join ever opened, and every
    // shutdown would pace endings for joins long gone.
    drain_tracker(&svc).await;
    assert_eq!(
        svc.shutdown.enrolled(),
        0,
        "ended joins must leave graceful shutdown's enrollment registry"
    );

    // Valkey-side UNSUBSCRIBE check: the presence-events channel for this
    // session must have zero subscribers now that the last local client has
    // disconnected. Before the fix, the join's cleanup never called
    // `unsubscribe_presence_channel`, so NUMSUB remained 1 forever.
    //
    // Counting subscribers is a direct proof that the UNSUBSCRIBE ran; the
    // previous "publish and watch for nothing" style was racy because the
    // router `has_session` short-circuit also drops events.
    let channel = tv.key_schema().presence_events_channel(&session_id);
    let numsub_zero = {
        let start = Instant::now();
        let deadline = Duration::from_secs(2);
        loop {
            // PUBSUB NUMSUB <channel> returns [channel_name, count].
            let numsub: Vec<fred::types::Value> = tv
                .client()
                .pool
                .next()
                .pubsub_numsub(vec![channel.as_str()])
                .await
                .expect("PUBSUB NUMSUB must succeed");
            let count = numsub
                .get(1)
                .and_then(|v| match v {
                    fred::types::Value::Integer(i) => Some(*i),
                    _ => None,
                })
                .unwrap_or(i64::MAX);
            if count == 0 {
                break true;
            }
            if start.elapsed() >= deadline {
                break false;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    };
    assert!(
        numsub_zero,
        "Valkey must have zero subscribers on {} after all local clients disconnect \
         (regression: before the b.i2e fix this stayed at 1 forever)",
        channel,
    );

    // Additional sanity: publishing on the channel post-cleanup must not
    // deliver anything to a fresh local subscriber, because the router entry
    // is absent and `spawn_presence_subscriber` short-circuits on
    // `!has_session`. We assert this *without* creating a fresh router entry
    // first — we publish, wait, then subscribe, and expect no queued events.
    publish_presence_event(
        &tv.client().pool,
        tv.key_schema(),
        &session_id,
        &PresenceEvent::PeerLeft {
            connection_id: "ghost-after-cleanup".to_owned(),
        },
    )
    .await
    .expect("publish must succeed");
    // Allow time for the subscriber task to observe-and-drop the event.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let (mut fresh_rx, _first) = svc.presence_router.subscribe(&session_id);
    let delivery = tokio::time::timeout(Duration::from_millis(200), fresh_rx.recv()).await;
    assert!(
        delivery.is_err(),
        "fresh subscriber must not receive events published before it subscribed; \
         got {delivery:?}"
    );
    let _ = svc.presence_router.unsubscribe(&session_id);

    // Shut the presence subscriber down.
    presence_cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), subscriber_handle).await;
}

// ── 3. Reconnect after full disconnect re-fires the first-subscriber path ───

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnect_after_full_disconnect_refires_first_subscriber_and_pubsub() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Install the shared presence subscriber task.
    let presence_cancel = CancellationToken::new();
    let subscriber_handle = spawn_presence_subscriber(
        tv.client().subscriber.clone(),
        Arc::clone(&svc.presence_router),
        tv.key_schema().clone(),
        Arc::clone(&svc.metrics),
        presence_cancel.clone(),
    );

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("reconnect-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    svc.session_tokens
        .insert(session_id.clone(), CancellationToken::new().into());

    // First connect cycle: join, assert entry, disconnect, assert removed.
    let (stream_a, tx_a) = start_join_session(&svc, &session_id, "app-a").await;
    assert!(svc.presence_router.has_session(&session_id));
    drop(tx_a);
    drain_response_stream(stream_a).await;
    let removed = poll_until(Duration::from_secs(2), || {
        !svc.presence_router.has_session(&session_id)
    })
    .await;
    assert!(
        removed,
        "router entry must be removed before we attempt reconnect"
    );

    // Second connect cycle: a fresh JoinSession must re-populate the router.
    // If the fix is in place, `join_channel` hands this join
    // `ChannelJoin::Subscribe` and the handler re-SUBSCRIBEs on Valkey, so
    // pub/sub events land again.
    let (mut stream_b, tx_b) = start_join_session(&svc, &session_id, "app-b").await;
    assert!(
        svc.presence_router.has_session(&session_id),
        "router must be re-populated after reconnect"
    );
    assert_eq!(svc.presence_router.session_count(), 1);

    // Prove pub/sub delivery has been re-established: publish an event on the
    // presence-events channel from an independent connection and assert that
    // the live JoinSession stream (client B) forwards it. This exercises the
    // full "fresh SUBSCRIBE on Valkey" -> "subscriber task" -> "router
    // broadcast" -> "JoinSession writer" path.
    //
    // We use a PeerJoined event with a different connection_id so the
    // self-filter in the bidi loop doesn't drop it.
    let injected = PresenceEvent::PeerJoined(live_edit::presence::ConnectedUser {
        connection_id: "ghost-peer".to_owned(),
        user_name: "Ghost".to_owned(),
        user_application: "ghost-app".to_owned(),
    });
    // Publish from a peer ValkeyClient-like path: just use the same pool since
    // self-echo is handled by connection_id, not by pool identity.
    publish_presence_event(&tv.client().pool, tv.key_schema(), &session_id, &injected)
        .await
        .expect("publish must succeed");

    // Client B should receive the PeerJoined event within a short deadline.
    let received = tokio::time::timeout(Duration::from_secs(3), async {
        while let Some(item) = stream_b.next().await {
            let msg = item.expect("stream item must not be error");
            if let Some(join_session_response::ResponseType::PeerJoined(pj)) = msg.response_type {
                return pj.connection_id;
            }
        }
        String::new()
    })
    .await
    .expect("must receive the injected PeerJoined within timeout");
    assert_eq!(
        received, "ghost-peer",
        "reconnect must re-establish Valkey pub/sub delivery end-to-end"
    );

    // Teardown.
    drop(tx_b);
    drain_response_stream(stream_b).await;
    presence_cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), subscriber_handle).await;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Service shutdown token (SDD SR-4.7)
// ═══════════════════════════════════════════════════════════════════════════════

// Session setup, presence store reads and stream drains (`create_session`,
// `presence_keys`, `is_indexed`, `drain_tracker`, `read_to_ok_end`) live in
// `super::helpers`, shared with `join_lifecycle_test.rs`.

// ── A join arriving during shutdown is refused before any task or write ─────

/// With the shutdown token already cancelled, the handler answers
/// `UNAVAILABLE("server shutting down")` (which Kit retries, wherever the
/// ingress routes it), spawns
/// no task, and writes nothing: no member set, info hash, lease or
/// `active_sessions` entry, and no presence-router subscription.
///
/// Current-thread runtime: a task the handler spawned could not have run by the
/// time the handler returns, so an empty tracker proves none was spawned.
#[tokio::test]
async fn join_session_during_shutdown_returns_unavailable_before_any_task_or_write() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _session_token) = create_session(&tv, &svc, "shutdown-refused", true).await;
    svc.shutdown.cancel();

    let (stream, tx) = open_join_streaming();
    tx.send(encode_join_frame(&join_header(&session_id, "app-a")))
        .expect("channel open");
    let result = handle_join_session(&svc, make_join_request(stream)).await;
    let tasks_after_handler = svc.presence_cleanup_tracker.len();

    let status = expect_err_status(result, "a join arriving during shutdown must be refused");
    assert_eq!(
        status.code(),
        Code::Unavailable,
        "a join during shutdown must be UNAVAILABLE, which Kit retries; got {status:?}"
    );
    assert_shutdown_status(&status);
    assert_eq!(
        tasks_after_handler, 0,
        "the refusal must come before the join task is spawned"
    );
    drain_tracker(&svc).await;

    assert_eq!(
        presence_keys(&tv, &session_id).await,
        Vec::<String>::new(),
        "a refused join must write no member set, info hash or lease"
    );
    assert!(
        !is_indexed(&tv, &session_id).await,
        "a refused join must not index the session in active_sessions"
    );
    assert!(
        !svc.presence_router.has_session(&session_id),
        "a refused join must not subscribe to the presence router"
    );

    drop(tx);
    purge_keys(&tv).await;
}

/// A join arriving **once graceful shutdown's paced stream ending has
/// started** — `begun` cancelled, the streams enrolled before it still being
/// ended in their batches, the service shutdown token not yet cancelled — is
/// refused at the handler's check with `UNAVAILABLE "server shutting down"`
/// and its retry hint, spawning no task and writing nothing. A check that
/// waited for the service shutdown token would admit it and run a whole
/// presence join on a replica that is going away.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn join_session_after_the_paced_ending_began_is_refused_with_the_retry_hint_before_any_task()
{
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _session_token) =
        create_session(&tv, &svc, "paced-ending-refused", true).await;
    let paced = begin_paced_ending_held_open(&svc.shutdown).await;

    let (stream, tx) = open_join_streaming();
    tx.send(encode_join_frame(&join_header(&session_id, "app-a")))
        .expect("channel open");
    let result = handle_join_session(&svc, make_join_request(stream)).await;
    let tasks_after_handler = svc.presence_cleanup_tracker.len();

    let status = expect_err_status(
        result,
        "a join arriving after the paced ending began must be refused",
    );
    assert_shutdown_status(&status);
    assert!(
        !svc.shutdown.is_cancelled(),
        "precondition: the refusal came while the service shutdown token was still uncancelled"
    );
    assert_eq!(
        tasks_after_handler, 0,
        "the refusal must come before the join task is spawned"
    );
    drain_tracker(&svc).await;
    assert_eq!(
        presence_keys(&tv, &session_id).await,
        Vec::<String>::new(),
        "a refused join must write no member set, info hash or lease"
    );
    assert!(!is_indexed(&tv, &session_id).await);
    assert!(!svc.presence_router.has_session(&session_id));

    drop(paced);
    drop(tx);
    purge_keys(&tv).await;
}

// ── Shutdown between the handler and the join task's start ──────────────────

/// A join the handler admitted, whose task has not started when shutdown
/// begins, ends at the task's first check: the stream ends `UNAVAILABLE
/// "server shutting down"` with no `SessionConnected`, and nothing was
/// written, so there is nothing to tear down.
///
/// Current-thread runtime: the handler returns without yielding after the
/// spawn, so the task has not run when the test cancels the token.
#[tokio::test]
async fn join_session_shutdown_before_task_start_ends_unavailable_without_writes() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, _session_token) =
        create_session(&tv, &svc, "shutdown-before-task", true).await;

    let (stream, tx) = open_join_streaming();
    tx.send(encode_join_frame(&join_header(&session_id, "app-a")))
        .expect("channel open");
    let response = handle_join_session(&svc, make_join_request(stream))
        .await
        .unwrap_or_else(|s| panic!("the handler must admit the join before shutdown: {s:?}"));
    assert_eq!(
        svc.presence_cleanup_tracker.len(),
        1,
        "precondition: the join task is spawned and has not yet finished"
    );

    svc.shutdown.cancel();
    let mut resp_stream: ResponseStream = response.into_inner();
    let (delivered, status) = read_to_end(&mut resp_stream, "shutdown before task start").await;

    assert!(
        delivered.is_empty(),
        "the task must stop before registering, so no SessionConnected is sent; got \
         {delivered:?}"
    );
    let status = status.expect(
        "a join stopped by shutdown at task start ended OK, which the Kit client never retries",
    );
    assert_eq!(status.code(), Code::Unavailable, "got {status:?}");
    assert_shutdown_status(&status);
    drain_tracker(&svc).await;
    assert_eq!(
        presence_keys(&tv, &session_id).await,
        Vec::<String>::new(),
        "the task must stop before its first presence write"
    );
    assert!(!is_indexed(&tv, &session_id).await);
    assert!(!svc.presence_router.has_session(&session_id));

    drop(tx);
    purge_keys(&tv).await;
}

// ── Shutdown ends an established join through the bidirectional loop ───────

/// An established join — past `SessionConnected` and the replay, relaying its
/// client's messages from the bidirectional loop — ends when only the service
/// shutdown token is cancelled, as graceful shutdown cancels it: the session
/// token stays uncancelled. Teardown removes the member entry, info hash and
/// lease, a peer on the presence channel sees `PeerLeft`, the cleanup tracker
/// drains, and the stream ends `UNAVAILABLE "server shutting down"` — the
/// ending the Kit client retries (where the retry lands is the ingress's
/// choice).
///
/// The client's request stream and the response stream are both held open
/// throughout, so neither a client disconnect nor a closed response channel
/// can end the loop in the shutdown arm's place.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_service_shutdown_ends_established_join_unavailable_after_teardown() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, session_token) = create_session(&tv, &svc, "shutdown-in-loop", true).await;
    let (mut events_rx, observer_handle, observer_subscriber) =
        spawn_external_channel_subscriber(&tv, &session_id).await;

    let (mut resp_stream, tx) = start_join_session(&svc, &session_id, "app-a").await;
    let members: Vec<String> = tv
        .client()
        .pool
        .next()
        .smembers(tv.key_schema().presence_members(&session_id))
        .await
        .expect("SMEMBERS must succeed");
    let [connection_id] = members.as_slice() else {
        panic!("precondition: exactly one registered member; got {members:?}");
    };
    let info_key = tv.key_schema().presence_user(&session_id, connection_id);
    let lease_key = tv.key_schema().presence_lease(&session_id, connection_id);
    let existing: i64 = tv
        .client()
        .pool
        .next()
        .exists(vec![info_key.clone(), lease_key.clone()])
        .await
        .expect("EXISTS must succeed");
    assert_eq!(existing, 2, "precondition: the info hash and lease exist");

    // Only the bidirectional loop publishes a client's message, so the peer
    // seeing it proves the join is in its loop.
    let payload = b"in-loop-probe".to_vec();
    tx.send(encode_join_frame(&join_message(payload.clone())))
        .expect("channel open");
    tokio::time::timeout(SETTLE, async {
        while let Some(event) = events_rx.recv().await {
            if let PresenceEvent::PeerMessage {
                connection_id: from,
                data,
            } = event
            {
                if &from == connection_id && data == payload {
                    return;
                }
            }
        }
        panic!("the external observer's channel closed");
    })
    .await
    .unwrap_or_else(|_| panic!("precondition: the join's loop did not relay a message"));

    svc.shutdown.cancel();
    let (_, status) = read_to_end(&mut resp_stream, "shutdown in the bidirectional loop").await;

    assert!(
        !session_token.is_cancelled(),
        "precondition: only the shutdown token ended the join"
    );
    let status = status.expect(
        "an established join ended by shutdown ended OK, which the Kit client never retries — \
         its presence feed would stop for good",
    );
    assert_eq!(status.code(), Code::Unavailable, "got {status:?}");
    assert_eq!(status.message(), "server shutting down");
    drain_tracker(&svc).await;
    let still_member: bool = tv
        .client()
        .pool
        .next()
        .sismember(tv.key_schema().presence_members(&session_id), connection_id)
        .await
        .expect("SISMEMBER must succeed");
    assert!(!still_member, "teardown must remove the member entry");
    let remaining: i64 = tv
        .client()
        .pool
        .next()
        .exists(vec![info_key, lease_key])
        .await
        .expect("EXISTS must succeed");
    assert_eq!(remaining, 0, "teardown must delete the info hash and lease");
    assert!(
        wait_for_peer_left(&mut events_rx, connection_id, SETTLE).await,
        "a peer on the presence channel must see PeerLeft for {connection_id}"
    );

    drop(tx);
    let _ = observer_subscriber
        .unsubscribe(tv.key_schema().presence_events_channel(&session_id))
        .await;
    let _ = observer_subscriber.quit().await;
    let _ = tokio::time::timeout(SETTLE, observer_handle).await;
}
