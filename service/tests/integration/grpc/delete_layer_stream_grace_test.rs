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

// Regression tests for b.k2t item 3: `delete_layer` must PEXPIRE the
// per-object stream (with `STREAM_DELETE_GRACE_MS`) instead of UNLINKing it
// immediately.
//
// Before the fix, a subscriber between XREAD round-trips that woke up for its
// next poll after UNLINK saw an empty/missing stream and never observed the
// `deleted` sentinel appended by delete_layer — a silent hang.
// PEXPIRE keeps the stream readable inside the grace window so lagging
// subscribers can still catch the sentinel and re-bootstrap.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{KeysInterface, StreamsInterface};
use tonic::Request;

use live_edit::grpc::layer::handle_delete_layer;
use live_edit::grpc::proto::DeleteLayerRequest;
use live_edit::grpc::BearerToken;
use live_edit::storage::layer::STREAM_DELETE_GRACE_MS;
use live_edit::storage::notifications::{consume_layer_stream, LayerNotification, StreamCursor};
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

fn delete_request(session_id: &str, layer_name: &str) -> Request<DeleteLayerRequest> {
    let mut request = Request::new(DeleteLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// After `delete_layer`, the per-object stream must still exist with a bounded
/// positive PTTL — proving PEXPIRE (not UNLINK) was applied and subscribers
/// lagging between XREAD round-trips still have a window to observe the
/// deleted sentinel.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_pexpires_stream_instead_of_unlinking() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "k2t-pexpire", "target-layer").await;

    let stream_key = tv.key_schema().layer_stream(object_id);
    let pool = tv.client().pool.clone();

    // Precondition: stream exists and has no TTL (created by CreateLayer).
    let ttl_before: i64 = pool.next().pttl(&stream_key).await.unwrap();
    assert_eq!(
        ttl_before, -1,
        "precondition: stream must be persistent (no TTL) before delete"
    );

    handle_delete_layer(&svc, delete_request(&session_id, "target-layer"))
        .await
        .expect("delete_layer must succeed");

    // Stream must STILL exist after delete — immediate UNLINK would have
    // dropped it and caused this check to fail with PTTL=-2 (key missing).
    let ttl_after: i64 = pool.next().pttl(&stream_key).await.unwrap();
    assert!(
        ttl_after > 0,
        "stream must have a positive PTTL after delete (immediate UNLINK would return -2), got {ttl_after}ms"
    );
    assert!(
        ttl_after <= STREAM_DELETE_GRACE_MS,
        "PTTL must be bounded by STREAM_DELETE_GRACE_MS ({STREAM_DELETE_GRACE_MS}ms), got {ttl_after}ms"
    );
    // Sanity: grace should be close to the configured value (within 1 second).
    assert!(
        ttl_after >= STREAM_DELETE_GRACE_MS - 1_000,
        "PTTL should be close to the configured grace ({STREAM_DELETE_GRACE_MS}ms), got {ttl_after}ms"
    );
}

/// Inside the grace window, a lagging subscriber can still read the `deleted`
/// sentinel from the stream and recognise the layer deletion.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_deleted_sentinel_is_readable_within_grace_window() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "k2t-sentinel", "target-layer").await;

    handle_delete_layer(&svc, delete_request(&session_id, "target-layer"))
        .await
        .expect("delete_layer must succeed");

    let stream_key = tv.key_schema().layer_stream(object_id);
    let pool = tv.client().pool.clone();

    // XRANGE the full stream and look for the `deleted` sentinel.
    let entries: Vec<(String, Vec<(String, String)>)> = pool
        .next()
        .xrange(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE must succeed within grace window");

    let saw_deleted = entries
        .iter()
        .any(|(_, fields)| fields.iter().any(|(k, v)| k == "payload" && v == "deleted"));
    assert!(
        saw_deleted,
        "stream must contain a `deleted` sentinel within the grace window — \
         subscribers lagging between XREAD round-trips rely on this to exit cleanly, \
         got entries: {entries:?}"
    );
}

/// End-to-end subscriber scenario — a subscriber that fires XREAD *after*
/// DeleteLayer completes must still observe the `Deleted` notification via
/// the real consumer API (not just XRANGE). This is the ticket's explicit
/// "suggested test coverage" for item 3: a subscriber pauses between XREAD
/// round-trips while DeleteLayer fires, then resumes reading and catches
/// the sentinel inside the grace window.
///
/// Under the old UNLINK behaviour, the stream key would be gone by the time
/// the subscriber issued its next XREAD and the `deleted` sentinel would be
/// unobservable — the subscriber would hang until its XREAD BLOCK timer
/// expired with no explanation. Under the PEXPIRE fix, the sentinel is
/// readable until the grace window expires.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lagging_subscriber_observes_deleted_sentinel_via_xread_after_delete() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "k2t-lag-sub", "target-layer").await;

    // Fire DeleteLayer BEFORE the subscriber starts its XREAD. This simulates
    // the worst-case race: the `deleted` XADD has happened, the other keys
    // have been UNLINKed, and the stream is in its PEXPIRE grace window. The
    // subscriber's next XREAD must still see the sentinel.
    handle_delete_layer(&svc, delete_request(&session_id, "target-layer"))
        .await
        .expect("delete_layer must succeed");

    // Verify the PEXPIRE is in place — establishes the grace-window state
    // this test exercises. Without it, the assertion below would pass for
    // the wrong reason (a persistent key that never got cleaned up at all).
    let stream_key = tv.key_schema().layer_stream(object_id);
    let ttl: i64 = tv.client().pool.pttl(&stream_key).await.unwrap();
    assert!(
        ttl > 0 && ttl <= STREAM_DELETE_GRACE_MS,
        "precondition: stream must be in PEXPIRE grace window, got PTTL={ttl}ms"
    );

    // Start the consumer from "0-0" (strictly after "0-0" means: every entry
    // ever XADDed on this stream, including both the initial "1-0" and the
    // auto-ID `deleted` sentinel). This simulates a subscriber that was
    // between XREAD round-trips when the delete fired and is now resuming.
    let consumer = consume_layer_stream(
        tv.client(),
        tv.key_schema(),
        object_id,
        StreamCursor::From("0-0".to_string()),
        tokio::time::Instant::now() + std::time::Duration::from_secs(2),
    )
    .await
    .expect("consume_layer_stream must succeed");

    // Drain until we see Deleted. The first recv() should yield the initial
    // "1-0" Update entry (from CreateLayer); the second should yield the
    // Deleted sentinel appended by delete_layer. Under the old UNLINK
    // behaviour, recv() would time out here because the stream is gone.
    let mut saw_deleted = false;
    for _ in 0..4 {
        let next = tokio::time::timeout(Duration::from_secs(5), consumer.recv())
            .await
            .expect("recv must not time out — deleted sentinel must be reachable in grace window");
        let (_id, notification) = next.expect("consumer must yield a notification, not None");
        if matches!(notification, LayerNotification::Deleted) {
            saw_deleted = true;
            break;
        }
    }

    consumer.unsubscribe().await;
    assert!(
        saw_deleted,
        "subscriber that resumed XREAD after DeleteLayer must observe Deleted within grace window"
    );
}
