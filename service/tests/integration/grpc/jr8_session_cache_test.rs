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

// Bug b.jr8 mediums: transient `session_id -> base_path` cache.
//
// Covers the four correctness points from the ticket:
//   * Miss: first call populates the cache from storage.
//   * Hit:  second call within TTL returns the cached value even after the
//           underlying Valkey hash is mutated (we don't have a counter
//           surface, so "hit" is demonstrated by stale-read behavior vs a
//           post-mutation direct HGET).
//   * Invalidate on local DeleteSession: after the RPC runs, the cache no
//     longer holds an entry for that session (next call misses, reads
//     storage, observes NotFound).
//   * Invalidate on `session:deleted` pub/sub: peer replica's cache drops
//     its entry when the fanout arrives.

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use fred::interfaces::HashesInterface;
use tokio_util::sync::CancellationToken;
use tonic::Request;

use live_edit::error::ServiceError;
use live_edit::grpc::proto::DeleteSessionRequest;
use live_edit::grpc::session::{
    handle_delete_session, spawn_session_delete_subscriber, subscribe_session_deleted_channel,
    SessionCancel,
};
use live_edit::grpc::session_cache::SessionBasePathCache;
use live_edit::grpc::BearerToken;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::*;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn delete_session_request(session_id: &str) -> Request<DeleteSessionRequest> {
    let mut request = Request::new(DeleteSessionRequest {
        session_id: session_id.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// ═══════════════════════════════════════════════════════════════════════════════
// Miss → storage read populates the cache
// ═══════════════════════════════════════════════════════════════════════════════
//
// First call reads through to Valkey; subsequent call within the TTL must
// return the same value without touching storage. We can't observe the
// round-trip count directly, so we demonstrate the cache path via the
// stale-read invariant exercised in the "hit" test below.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_base_path_cache_miss_populates_from_storage() {
    let tv = TestValkey::setup().await;
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let cache = SessionBasePathCache::new();

    let session_id = session_store
        .create("jr8-cache-miss", "/stage/alpha", "test-user")
        .await
        .expect("session create must succeed");

    // Miss: first call. Must return the stored base_path.
    let got = cache
        .get_base_path(&session_store, &session_id)
        .await
        .expect("first get must succeed");
    assert_eq!(got, "/stage/alpha");

    // Immediate re-call: hit. Must still return the same value.
    let again = cache
        .get_base_path(&session_store, &session_id)
        .await
        .expect("second get must succeed");
    assert_eq!(again, "/stage/alpha");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hit → serves stale value while TTL is live
// ═══════════════════════════════════════════════════════════════════════════════
//
// After the cache is warm, we mutate the underlying hash directly (bypassing
// the cache). A cached lookup must return the *old* value — proving it came
// from memory, not from storage. A direct HGET is used as the control to
// show storage itself has the new value.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_base_path_cache_hit_serves_value_without_storage_reread() {
    let tv = TestValkey::setup().await;
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let cache = SessionBasePathCache::new();

    let session_id = session_store
        .create("jr8-cache-hit", "/stage/original", "test-user")
        .await
        .expect("session create must succeed");

    // Warm the cache.
    let warm = cache
        .get_base_path(&session_store, &session_id)
        .await
        .expect("warm call must succeed");
    assert_eq!(warm, "/stage/original");

    // Stomp on the underlying `base_path` field directly in Valkey. A direct
    // HGET now returns "/stage/mutated", but the cache should still hand out
    // "/stage/original" until TTL or invalidation.
    let key = tv.key_schema().session_meta(&session_id);
    tv.client()
        .pool
        .hset::<(), _, _>(&key, [("base_path", "/stage/mutated")])
        .await
        .expect("HSET mutation must succeed");

    // Control: direct storage read sees the mutation.
    let direct = session_store
        .get_base_path(&session_id)
        .await
        .expect("direct get must succeed");
    assert_eq!(
        direct, "/stage/mutated",
        "control: direct get_base_path must see the underlying mutation"
    );

    // Cache must still return the original (hit path).
    let cached = cache
        .get_base_path(&session_store, &session_id)
        .await
        .expect("cached call must succeed");
    assert_eq!(
        cached, "/stage/original",
        "cache hit must serve the pre-mutation value (proof of cache path)"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Invalidate on local DeleteSession
// ═══════════════════════════════════════════════════════════════════════════════
//
// `handle_delete_session` calls `session_base_path_cache.invalidate(session_id)`
// after the cascade + storage delete. Subsequent calls must miss and
// short-circuit to the now-NotFound storage state.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_base_path_cache_invalidates_on_local_delete_session() {
    let tv = TestValkey::setup().await;
    let svc = build_service(&tv, Arc::new(AllowAllPermissions));

    // Create a session so the cache has something to warm from.
    let (session_id, _oid) = create_session_and_layer(&tv, &svc, "jr8-cache-del", "layer-x").await;

    // Warm the cache via the service's own field (same instance that the
    // handler will invalidate).
    let warmed = svc
        .session_base_path_cache
        .get_base_path(&svc.session_store, &session_id)
        .await
        .expect("warm call must succeed");
    assert_eq!(warmed, "/stage");

    // DeleteSession via the handler — must invalidate the local entry.
    handle_delete_session(&svc, delete_session_request(&session_id))
        .await
        .expect("DeleteSession must succeed");

    // Post-delete: cache must miss. Since storage is now empty for this
    // session, the call propagates `NotFound`. If the cache still held the
    // stale entry, this would incorrectly return `Ok("/stage")`.
    let after = svc
        .session_base_path_cache
        .get_base_path(&svc.session_store, &session_id)
        .await;
    match after {
        Err(ServiceError::NotFound(_)) => {} // expected
        other => panic!("cache must not retain an entry after local DeleteSession; got {other:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Invalidate on `session:deleted` pub/sub
// ═══════════════════════════════════════════════════════════════════════════════
//
// Peer replicas receive the session_id on `le:session:deleted` and must drop
// their cached entry. We wire the subscriber directly on the peer's cache
// and verify.

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_base_path_cache_invalidates_on_pubsub_fanout() {
    let tv = TestValkey::setup().await;
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

    // Create a session and pre-warm a "peer" cache keyed to that session.
    let session_id = session_store
        .create("jr8-cache-pubsub", "/stage/beta", "test-user")
        .await
        .expect("session create must succeed");

    let peer_cache = SessionBasePathCache::new();
    let peer_tokens: Arc<DashMap<String, SessionCancel>> = Arc::new(DashMap::new());
    let warmed = peer_cache
        .get_base_path(&session_store, &session_id)
        .await
        .expect("warm call must succeed");
    assert_eq!(warmed, "/stage/beta");

    // Wire the peer subscriber on our TestValkey's shared subscriber client.
    // Spawn the relay BEFORE firing `SUBSCRIBE`: `spawn_session_delete_subscriber`
    // takes its broadcast receiver before it returns, so the relay sees every
    // message published from then on, the first included.
    let cancel = CancellationToken::new();
    let subscriber = spawn_session_delete_subscriber(
        tv.client().subscriber.clone(),
        Arc::clone(&peer_tokens),
        peer_cache.clone(),
        tv.key_schema().clone(),
        Arc::new(crate::common::metrics_scrape_only("test")),
        live_edit::grpc::session::SessionRevalidation::new(),
        cancel.clone(),
    );
    subscribe_session_deleted_channel(&tv.client().subscriber, tv.key_schema())
        .await
        .expect("subscribe_session_deleted_channel must succeed");
    // Give the SUBSCRIBE round-trip a moment to settle before PUBLISH.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Publish `session:deleted` for this session_id. The peer subscriber must
    // observe it and invalidate the cache.
    use fred::interfaces::PubsubInterface;
    let channel = tv.key_schema().session_deleted_channel();
    tv.client()
        .pool
        .next()
        .publish::<(), _, _>(&channel, session_id.as_str())
        .await
        .expect("PUBLISH must succeed");

    // Poll the cache until it no longer serves a cached hit. Because
    // `get_base_path` re-reads from storage on a miss and storage still has
    // the session entry (we did not call DeleteSession here — purely the
    // pub/sub invalidation signal), the re-read *would* succeed. We need a
    // finer-grained signal. Mutate storage so a stale cache hit returns
    // the old value while a re-read returns the new one — then the signal
    // of "cache was invalidated" is observable as "new value is served".
    let key = tv.key_schema().session_meta(&session_id);
    tv.client()
        .pool
        .hset::<(), _, _>(&key, [("base_path", "/stage/beta-renamed")])
        .await
        .expect("HSET mutation must succeed");

    // Give the pub/sub fanout time to deliver on the shared subscriber
    // client. The observable assertion below is the actual signal — this
    // sleep is pure delivery jitter padding. 500 ms is well above the
    // tens-of-ms typical pub/sub round-trip on a local Valkey.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Observable assertion: after the pub/sub fanout, a fresh `get_base_path`
    // call must re-read storage (cache was invalidated) and return the new
    // mutated value. A stale cache would still return "/stage/beta".
    let after = peer_cache
        .get_base_path(&session_store, &session_id)
        .await
        .expect("post-fanout get must succeed");
    assert_eq!(
        after, "/stage/beta-renamed",
        "pub/sub fanout must invalidate the peer cache — a stale hit would return /stage/beta"
    );

    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), subscriber).await;
}
