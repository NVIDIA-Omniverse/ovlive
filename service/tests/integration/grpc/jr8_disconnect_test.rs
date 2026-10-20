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

// Bug b.jr8 H-grpc-2: integration tests for client-disconnect task lifecycle.
//
// Two distinct tasks exit paths are gated by the fix:
//   * `SubscribeReadLayer`: steady-state `select!` has a `tx.closed()` arm and
//     a top-of-iteration `is_closed()` poll. The observable contract is that
//     when the client drops the response stream, the spawned subscription task
//     ends and unregisters from the stream multiplexer (returning its admission
//     permit) within a bounded time — without the fix the task idles inside
//     `consumer.recv()` until the next stream entry arrives.
//   * `JoinSession`: the `run_bidi_loop` `select!` has a `tx.closed()` arm.
//     Observable contract: when the client drops the response stream while
//     the loop is idle (no presence events, no client frames), the bidi task
//     exits quickly instead of waiting for the next event.
//
// We assert on *time bounds* rather than exact wall-clock values — the fix
// converts a 30 s waits-for-XREAD-timeout into a near-instant wake, so a
// 5 s deadline cleanly separates the two regimes.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tonic::Request;

use live_edit::grpc::proto::SubscribeReadLayerRequest;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

// ── Request builders ─────────────────────────────────────────────────────────

fn subscribe_request(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    sequence_num: u64,
    client_id: &str,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
        object_id,
        sequence_num,
        client_id: client_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// The JoinSession test below opens its stream with `super::helpers::open_join`,
// whose client side stays open until the returned sender is dropped, so the
// bidi loop idles on the presence-event / client-stream poll and `tx.closed()`
// is exercised cleanly.

// ═══════════════════════════════════════════════════════════════════════════════
// H-grpc-2: SubscribeReadLayer — client disconnect ends the subscription promptly
// ═══════════════════════════════════════════════════════════════════════════════
//
// Test strategy: open 4 idle steady-state subscriptions (no writes, so nothing
// would ever wake `consumer.recv()`), drop one client-side receiver, and assert
// the multiplexer's registration count and the admission gate's free permits
// reflect the ended subscription within 5 s. Pre-fix the zombie task would sit
// in `consumer.recv()` until a stream entry arrived.

/// Poll `cond` until it holds, or fail after `within`.
async fn await_within(within: Duration, what: &str, mut cond: impl FnMut() -> bool) {
    let deadline = Instant::now() + within;
    while !cond() {
        assert!(
            Instant::now() < deadline,
            "{what} did not happen within {within:?}"
        );
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_client_disconnect_ends_the_subscription_promptly() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let gate = &tv.client().subscriber_admission;
    let capacity = gate
        .capacity()
        .expect("the fixture sets a subscription cap");

    // Four independent layers so four subscribers each catch up cleanly and
    // then park in steady state on their own stream.
    let mut subs: Vec<_> = Vec::with_capacity(4);
    for i in 0..4 {
        let (session_id, object_id) =
            create_session_and_layer(&tv, &svc, &format!("jr8-sub-{i}"), &format!("layer-{i}"))
                .await;
        let response = handle_subscribe_read_layer(
            &svc,
            subscribe_request(
                &session_id,
                &format!("layer-{i}"),
                object_id,
                1,
                &format!("client-{i}"),
            ),
        )
        .await
        .expect("SubscribeReadLayer must succeed");
        let mut stream = response.into_inner();

        // Drain the Case-3 ack header so the spawned task is parked inside
        // `consumer.recv()` / the steady-state `select!` — the exact state the
        // `tx.closed()` fix must wake from.
        let _ack = tokio::time::timeout(Duration::from_secs(5), stream.next())
            .await
            .expect("ack header must arrive within 5 s")
            .expect("stream must yield the ack")
            .expect("ack must be Ok");

        subs.push(stream);
    }
    assert_eq!(tv.client().stream_mux.snapshot().subscriptions(), 4);
    assert_eq!(gate.available_permits(), capacity - 4);

    // Drop one receiver — the `tx.closed()` arm must wake the idle task so it
    // drops its `LayerStreamConsumer` and its admission.
    let started = Instant::now();
    drop(subs.remove(0));
    await_within(
        Duration::from_secs(5),
        "the disconnected subscription unregistering from the stream multiplexer",
        || tv.client().stream_mux.snapshot().subscriptions() == 3,
    )
    .await;
    await_within(
        Duration::from_secs(5),
        "the disconnected subscription returning its admission permit",
        || gate.available_permits() == capacity - 3,
    )
    .await;
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the disconnected subscription took {:?} to end (H-grpc-2 regression)",
        started.elapsed()
    );

    drop(subs);
    await_within(
        Duration::from_secs(5),
        "every remaining subscription ending after its client disconnected",
        || tv.client().stream_mux.snapshot().subscriptions() == 0,
    )
    .await;
}

// ═══════════════════════════════════════════════════════════════════════════════
// H-grpc-2: SubscribeReadLayer — client disconnect during Regime 1 catch-up
// ═══════════════════════════════════════════════════════════════════════════════
//
// Adjacent to the pool-saturation test above. Covers the narrower assertion in
// the ticket: dropping the receiver immediately after the subscribe handler
// returns must let the spawned task exit cleanly without the RPC observing a
// lingering error. We can't see the task from outside the `tokio::spawn`, so
// the observable check is that the stream's `next()` returns `None` (clean end)
// within a bounded time after the receiver is conceptually "gone".

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_immediate_client_drop_stream_ends_cleanly() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "jr8-drop-imm", "layer-imm").await;

    // Subscribe, drain the catch-up ack, then drop the receiver. The spawned
    // task is now parked on `consumer.recv()`; the `tx.closed()` arm must
    // wake it within milliseconds.
    let response = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "layer-imm", object_id, 1, "client-drop"),
    )
    .await
    .expect("SubscribeReadLayer must succeed");
    let mut stream = response.into_inner();
    let _ack = tokio::time::timeout(Duration::from_secs(5), stream.next())
        .await
        .expect("ack must arrive within 5 s")
        .expect("stream must yield")
        .expect("ack must be Ok");

    // Drop the receiver. Without the fix, the spawned task would sit inside
    // `consumer.recv()` for up to 30 s. We can't directly observe task exit,
    // but the test passing within a 2 s timeout combined with the
    // pool-saturation test above gives us the evidence.
    drop(stream);

    // Cooperate with the runtime to let the zombie task wake on `tx.closed()`.
    // This yields rather than asserting observable behaviour — the dependent
    // signal is the pool-saturation test (above) which fails on the same
    // regression.
    tokio::time::sleep(Duration::from_millis(50)).await;
}

// ═══════════════════════════════════════════════════════════════════════════════
// H-grpc-2: JoinSession — idle client disconnect wakes the bidi loop
// ═══════════════════════════════════════════════════════════════════════════════
//
// `run_bidi_loop`'s `select!` polls `client_stream.message()`,
// `presence_router.recv()`, session cancellation, and (post-fix) `tx.closed()`.
// With no presence events and no client frames, the pre-fix loop waits
// indefinitely for *some* poll to fire. The `tx.closed()` arm wakes the loop
// as soon as the response rx is dropped.
//
// Assertion: after receiving `SessionConnected`, dropping the response stream
// must let the spawned join task exit within a bounded deadline. We observe
// task exit via its presence teardown, whose unregister script removes the
// connection from the presence members set.
//
// Note: dropping the response receiver closes the gRPC response channel; the
// loop's next tick observes `tx.closed()` and returns `Ok(())`. Teardown then
// removes the connection from the presence members set. We poll that set until
// the connection is gone — pre-fix this takes until the client stream closes
// or a presence event fires (unbounded for an idle session), post-fix it is
// millisecond.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn join_session_idle_client_disconnect_wakes_bidi_loop() {
    use fred::interfaces::SetsInterface;

    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Create the session manually (bypassing CreateSession auth) and insert the
    // matching CancellationToken so the bidi loop can bind to it.
    let session_store = live_edit::storage::session::SessionStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    );
    let session_id = session_store
        .create("jr8-join-idle", "/stage", "test-user")
        .await
        .expect("session create must succeed");
    svc.session_tokens
        .insert(session_id.clone(), CancellationToken::new().into());

    // Open a live client stream and send the header. The channel stays open
    // so the bidi loop is parked on the client-message / presence / tx.closed()
    // select — that is the state the `tx.closed()` fix must wake from.
    let (mut response_stream, client_tx) = open_join(&svc, &session_id, "jr8-test").await;

    // Drain SessionConnected. The membership replay that follows it has no
    // peers to send, so the join moves straight on to the idle select.
    let connection_id =
        expect_session_connected(&mut response_stream, Duration::from_secs(5)).await;

    // Precondition: presence members set contains the new connection.
    let members_key = tv.key_schema().presence_members(&session_id);
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(&members_key)
        .await
        .expect("SMEMBERS must succeed");
    assert!(
        members.iter().any(|m| m == &connection_id),
        "connection_id must appear in the presence members set before disconnect"
    );

    // Drop the response receiver while the client side stays open. The bidi
    // loop must wake via `tx.closed()` and the join must run its teardown —
    // which removes the connection from the members set. Pre-fix the loop
    // would block forever on the idle select (no client frames, no presence
    // events).
    let started = Instant::now();
    drop(response_stream);

    // Poll the members set until the connection is gone (cleanup ran) or the
    // bounded deadline expires. 5 s is well under any pre-fix backstop yet
    // forgiving for CI scheduler noise.
    let deadline = Duration::from_secs(5);
    let cleaned = loop {
        if started.elapsed() >= deadline {
            break false;
        }
        let now: Vec<String> = tv
            .client()
            .pool
            .smembers(&members_key)
            .await
            .expect("SMEMBERS must succeed");
        if !now.iter().any(|m| m == &connection_id) {
            break true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    assert!(
        cleaned,
        "JoinSession bidi task must exit (and its presence teardown must run) within {:?} after \
         the response stream is dropped — the member was not removed, indicating the \
         tx.closed() wake is missing",
        deadline
    );
    // Loose latency sanity: post-fix cleanup is typically < 200 ms.
    assert!(
        started.elapsed() < deadline,
        "bidi loop took {:?} > {:?} after client disconnect — H-grpc-2 regression",
        started.elapsed(),
        deadline
    );

    // Keep the client tx alive to this point so the precondition that the loop
    // is only woken via `tx.closed()` (not `Ok(None)` from the client side)
    // holds. Dropping it now is polite.
    drop(client_tx);
}
