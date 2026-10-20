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

// Regression tests for b.3z7 C-1: Valkey pub/sub subscriptions must survive
// a connection drop.
//
// Pre-fix, `SubscriberClient::manage_subscriptions()` was never called on
// the client constructed by `ValkeyClient::connect`. fred tracks pending
// subscriptions locally but only re-issues `SUBSCRIBE` commands on
// reconnect if `manage_subscriptions` is spawned. Without it, a Valkey
// failover or a brief network blip would silently drop every pub/sub
// subscription on the affected replica — presence events
// (`le:presence:events:{sid}`) and cross-replica session-deleted
// broadcasts (`le:session:deleted`) would stop firing until the pod
// restarted, with no alert.
//
// These tests use `CLIENT KILL TYPE pubsub` from a second `ValkeyClient`
// to deterministically force-disconnect the subscriber connection. fred's
// reconnect policy picks up the drop and re-establishes the TCP
// connection within milliseconds; with the fix, `manage_subscriptions`
// re-issues every tracked subscription on reconnect, so a subsequent
// `PUBLISH` on the data pool still reaches the subscriber. Without the
// fix, the subscriber receives nothing.

use std::time::Duration;

use fred::{
    interfaces::{ClientLike, EventInterface, PubsubInterface},
    types::{CustomCommand, Message},
};
use tokio::sync::oneshot;
use uuid::Uuid;

use live_edit::config::Config;
use live_edit::storage::client::ValkeyClient;

/// Build a minimal `Config` pointing at the integration Valkey. Used to
/// stand up two independent `ValkeyClient`s (publisher + subscriber) so
/// the kill-from-the-other-side pattern works.
///
/// b.ohv item 8: delegate to `common::test_config()` and override only
/// the fields this test cares about. `valkey_reconnect_max_delay_ms` is
/// tightened so the reconnect window stays predictable — without a short
/// cap fred backs off to the default 30 s and the test stalls waiting
/// for a reconnect that does come, just too slowly to observe cleanly.
fn test_config(pool_size: u32) -> Config {
    let mut cfg = crate::common::test_config();
    cfg.valkey_pool_size = pool_size;
    cfg.valkey_reconnect_max_delay_ms = 500;
    cfg
}

/// Connect a `ValkeyClient`, returning `None` if Valkey is not reachable
/// (CI machines without a live Valkey skip these tests gracefully — same
/// pattern as existing `tests/integration/storage/client_test.rs`).
async fn maybe_connect() -> Option<ValkeyClient> {
    let cfg = test_config(2);
    match tokio::time::timeout(
        Duration::from_secs(5),
        live_edit::storage::test_support::connect_test_client(&cfg),
    )
    .await
    {
        Ok(Ok(client)) => Some(client),
        _ => None,
    }
}

/// Kill the specific subscriber connection identified by `victim_id`.
///
/// Earlier drafts of this helper used `CLIENT KILL TYPE pubsub` from the
/// killer's data pool, but that kills *every* pubsub-type connection
/// visible to the killer — including SubscriberClients held open by
/// other tests running in parallel under `cargo test`. Switching to
/// `CLIENT KILL ID <victim_id>` scopes the kill to exactly one
/// connection so the test is cross-process-parallel-safe against
/// `cargo-nextest` and against other tests in this binary that also use
/// `SubscriberClient`.
///
/// Uses fred's `custom` API directly because `CLIENT KILL` is under
/// fred's `i-client` feature which this crate does not depend on.
async fn kill_connection_by_id(killer: &ValkeyClient, victim_id: i64) -> i64 {
    killer
        .pool
        .next()
        .custom::<i64, String>(
            CustomCommand::new_static("CLIENT KILL", None::<u16>, false),
            vec!["ID".to_string(), victim_id.to_string()],
        )
        .await
        .expect("CLIENT KILL ID must succeed")
}

/// Read the Valkey-server-assigned connection ID of the subscriber's
/// pub/sub TCP session.
///
/// Must be called BEFORE the subscriber issues its first `SUBSCRIBE` —
/// Redis/Valkey's subscribe mode only accepts a narrow command set
/// (`SUBSCRIBE` / `UNSUBSCRIBE` / `PING` / `QUIT` / `RESET`), which does
/// not include `CLIENT ID`. Calling `CLIENT ID` on a connection already
/// in subscribe mode returns an `ERR Can't execute 'client|id'...`
/// response.
///
/// fred's `SubscriberClient::id()` returns fred's *internal* client ID,
/// not Valkey's server-side connection ID, so we issue `CLIENT ID` over
/// the subscription connection itself to get the value that
/// `CLIENT KILL ID ...` will match on.
async fn subscriber_connection_id(subscriber: &fred::clients::SubscriberClient) -> i64 {
    subscriber
        .custom::<i64, String>(
            CustomCommand::new_static("CLIENT ID", None::<u16>, false),
            vec![],
        )
        .await
        .expect("CLIENT ID must succeed")
}

/// Sanity check that the fix is actually installed: the subscriber's
/// `reconnect_rx` sees a reconnect event after a targeted CLIENT KILL
/// of the subscriber connection. If this fails, fred's reconnect policy
/// itself is broken and the downstream `manage_subscriptions` assertion
/// would be untestable — distinguishing "reconnect never happened" from
/// "manage_subscriptions not called" up front.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reconnect_event_fires_after_client_kill_pubsub() {
    let Some(client) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };

    let mut reconnect_rx = client.subscriber.reconnect_rx();

    // Capture the subscriber's server-assigned connection ID BEFORE
    // subscribing — Valkey's subscribe mode rejects `CLIENT ID` with
    // `ERR Can't execute 'client|id'...`. The server-assigned ID stays
    // stable for the life of the TCP session, so reading it pre-
    // subscribe is sound.
    let victim_id = subscriber_connection_id(&client.subscriber).await;

    // Now subscribe so the pub/sub connection is actually used.
    let channel = format!("le:test:{}:pubsub-reconnect", Uuid::new_v4());
    client
        .subscriber
        .subscribe(&channel)
        .await
        .expect("SUBSCRIBE must succeed");

    let killed = kill_connection_by_id(&client, victim_id).await;
    assert_eq!(
        killed, 1,
        "CLIENT KILL ID {victim_id} must have killed exactly one connection (got {killed})"
    );

    let got_reconnect = tokio::time::timeout(Duration::from_secs(5), reconnect_rx.recv()).await;
    assert!(
        matches!(got_reconnect, Ok(Ok(_))),
        "fred must emit a reconnect event within 5s of CLIENT KILL; got {got_reconnect:?}"
    );

    let _ = client.subscriber.unsubscribe(&channel).await;
}

/// The core C-1 regression test. Subscribe, kill the pub/sub connection,
/// wait for reconnect, publish, expect the message to arrive.
///
/// Pre-fix: `manage_subscriptions` is never spawned, so fred reconnects
/// but does not re-issue `SUBSCRIBE`. The published message is dropped
/// server-side (no subscriber on the channel) and this test's
/// `message_rx().recv()` times out.
///
/// Post-fix: `ValkeyClient::connect` spawns `manage_subscriptions`, which
/// re-subscribes on every reconnect. The published message hits the
/// re-subscribed channel and our `message_rx()` receives it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pubsub_subscription_survives_connection_drop() {
    // Use two independent ValkeyClients so the killer's data pool
    // connection is definitely different from the victim's SubscriberClient.
    // Single-client with SKIPME=yes would work too but two clients read
    // cleaner and let the test also exercise cross-client publish.
    let Some(subscriber_side) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };
    let publisher_side = maybe_connect()
        .await
        .expect("publisher-side Valkey must be reachable if subscriber-side was");

    let channel = format!("le:test:{}:c1-regression", Uuid::new_v4());

    // Capture the victim's subscriber connection ID BEFORE subscribing —
    // CLIENT ID is not allowed on a connection already in subscribe mode.
    let victim_id = subscriber_connection_id(&subscriber_side.subscriber).await;

    // Subscribe to the channel so fred records it in the tracked-subs set.
    // manage_subscriptions reads from that set on reconnect.
    subscriber_side
        .subscriber
        .subscribe(&channel)
        .await
        .expect("SUBSCRIBE must succeed");

    // Set up a dedicated message receiver so we don't race the test harness
    // against fred's internal broadcast channel.
    let mut msg_rx = subscriber_side.subscriber.message_rx();
    let mut reconnect_rx = subscriber_side.subscriber.reconnect_rx();

    let killed = kill_connection_by_id(&publisher_side, victim_id).await;
    assert_eq!(
        killed, 1,
        "CLIENT KILL ID {victim_id} must have killed exactly one connection (got {killed})"
    );

    // Wait for fred to raise the reconnect event. Without this wait the
    // PUBLISH below could race the reconnect — if PUBLISH lands before
    // re-subscribe, the message is lost server-side regardless of the fix.
    // Waiting on `reconnect_rx` pushes the PUBLISH after the reconnect
    // round-trip, which is when `manage_subscriptions` re-subscribes.
    let got_reconnect = tokio::time::timeout(Duration::from_secs(5), reconnect_rx.recv())
        .await
        .expect("fred must reconnect within 5s of CLIENT KILL")
        .expect("reconnect broadcast channel must not be closed");
    let _ = got_reconnect;

    // Give manage_subscriptions a small window to finish re-issuing the
    // SUBSCRIBE that fires on the reconnect edge. We can't use reconnect_rx
    // as the exact "subscriptions restored" signal because fred's API does
    // not emit one — but re-SUBSCRIBE runs in fred's own task and is
    // typically a single RTT to Valkey.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Pipe the message receiver into a oneshot so the recv assertion below
    // doesn't hold open a long-lived await that cross-pollutes with the
    // publish round-trip.
    let channel_bytes = channel.as_bytes().to_vec();
    let (tx, rx) = oneshot::channel::<Message>();
    let tx = std::sync::Arc::new(std::sync::Mutex::new(Some(tx)));
    let tx_clone = tx.clone();
    tokio::spawn(async move {
        while let Ok(msg) = msg_rx.recv().await {
            if msg.channel.as_bytes() == channel_bytes.as_slice() {
                if let Some(tx) = tx_clone.lock().unwrap().take() {
                    let _ = tx.send(msg);
                }
                break;
            }
        }
    });

    // Publish from the *other* client's pool (a different set of
    // connections from the victim's SubscriberClient).
    publisher_side
        .pool
        .next()
        .publish::<(), _, _>(&channel, "post-reconnect-payload")
        .await
        .expect("PUBLISH on publisher-side pool must succeed");

    let received = tokio::time::timeout(Duration::from_secs(5), rx)
        .await
        .expect(
            "post-reconnect publish must reach the subscriber within 5s — \
             this is the C-1 regression: manage_subscriptions is not wired, \
             so fred reconnected but never re-issued SUBSCRIBE",
        )
        .expect("oneshot channel must not be dropped");

    let got = received.value.as_str().unwrap_or_default();
    assert_eq!(
        got.as_ref(),
        "post-reconnect-payload",
        "received payload must be the one we published after the reconnect"
    );

    // Cleanup.
    let _ = subscriber_side.subscriber.unsubscribe(&channel).await;
}

/// The DeleteSession fan-out path uses the exact same `SubscriberClient`
/// as the presence path — they share one TCP connection and one
/// tracked-subscriptions set (see `main.rs` → `spawn_session_delete_subscriber`).
/// So a successful presence-reconnect test also proves DeleteSession
/// fan-out survives. But the shapes of the two subscriptions are
/// different (presence uses per-session `le:presence:events:{sid}`,
/// DeleteSession uses the static `le:session:deleted`), so this test
/// covers the specific channel name to make sure a future refactor that
/// routes session-deleted through a different SubscriberClient would
/// trip a dedicated regression.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_deleted_fanout_survives_connection_drop() {
    let Some(subscriber_side) = maybe_connect().await else {
        eprintln!("SKIP: Valkey not reachable");
        return;
    };
    let publisher_side = maybe_connect()
        .await
        .expect("publisher-side Valkey must be reachable");

    // `le:session:deleted` is the static channel used for cross-replica
    // DeleteSession fan-out. Use a per-test prefix channel instead so
    // we don't race a sibling session-deleted publish from production
    // handlers — the subscription-resume behavior is identical because
    // it's the same SubscriberClient, same reconnect path.
    let channel = format!("le:test:{}:session-deleted-regression", Uuid::new_v4());

    // Capture connection ID BEFORE subscribing.
    let victim_id = subscriber_connection_id(&subscriber_side.subscriber).await;

    subscriber_side
        .subscriber
        .subscribe(&channel)
        .await
        .expect("SUBSCRIBE must succeed");

    let mut msg_rx = subscriber_side.subscriber.message_rx();
    let mut reconnect_rx = subscriber_side.subscriber.reconnect_rx();

    let killed = kill_connection_by_id(&publisher_side, victim_id).await;
    assert_eq!(killed, 1, "CLIENT KILL ID must kill exactly the victim");

    let _ = tokio::time::timeout(Duration::from_secs(5), reconnect_rx.recv())
        .await
        .expect("reconnect within 5s")
        .expect("reconnect broadcast not closed");

    tokio::time::sleep(Duration::from_millis(100)).await;

    let channel_bytes = channel.as_bytes().to_vec();
    let (tx, rx) = oneshot::channel::<Message>();
    let tx = std::sync::Arc::new(std::sync::Mutex::new(Some(tx)));
    let tx_clone = tx.clone();
    tokio::spawn(async move {
        while let Ok(msg) = msg_rx.recv().await {
            if msg.channel.as_bytes() == channel_bytes.as_slice() {
                if let Some(tx) = tx_clone.lock().unwrap().take() {
                    let _ = tx.send(msg);
                }
                break;
            }
        }
    });

    // Simulate the DeleteSession fan-out: a second replica PUBLISHes
    // the deleted session id on `le:session:deleted` (emulated here by
    // our per-test channel).
    publisher_side
        .pool
        .next()
        .publish::<(), _, _>(&channel, "deleted-session-id-fake")
        .await
        .expect("PUBLISH must succeed");

    let received = tokio::time::timeout(Duration::from_secs(5), rx)
        .await
        .expect(
            "session-deleted fan-out after reconnect must arrive within 5s — \
             otherwise cross-replica DeleteSession is silently broken after a Valkey blip",
        )
        .expect("oneshot not dropped");

    let got = received.value.as_str().unwrap_or_default();
    assert_eq!(got.as_ref(), "deleted-session-id-fake");

    let _ = subscriber_side.subscriber.unsubscribe(&channel).await;
}
