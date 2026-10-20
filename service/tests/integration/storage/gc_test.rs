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

// Integration tests for GC background task and startup recovery (SDD §11.3, §11.4).
//
// Tests run against a real Valkey instance via the TestValkey fixture. Each test
// gets an isolated key namespace so tests can run in parallel safely.
//
// Requires a running Valkey:
//   docker run --rm -p 6379:6379 valkey/valkey:8
// Address is read from TEST_VALKEY_URL, defaulting to redis://127.0.0.1:6379.

use fred::interfaces::{ClientLike, HashesInterface, KeysInterface, SetsInterface};

use live_edit::storage::external_values::{
    process_pending_gc, ExtRelease, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::test_support::TestValkey;

/// Helper: build a ValkeyExternalValueStore from a TestValkey fixture.
fn ext_store(tv: &TestValkey) -> ValkeyExternalValueStore {
    ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
}

/// Helper: create a distinct 32-byte hash from a single byte value.
fn test_hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

/// Two distinct retiring objects (b.qpf).
///
/// `ref_count` counts **live objects referencing the hash**, and a pending-set
/// member is now `{hash_hex}:{object_id}:{event}` — so "two owed decrements on
/// one blob" is expressed by two DIFFERENT objects retiring, not by calling
/// `schedule_release` twice with the same arguments. The Set collapses equal
/// members by design (that idempotence is what makes `delete_layer`'s retry and
/// the `ext_gc_scheduled` guard safe to replay), so a repeated identical release
/// is ONE owed decrement, exactly as intended.
const OBJ_A: u64 = 7001;
/// The second referencing object. See [`OBJ_A`].
const OBJ_B: u64 = 7002;

/// One owed decrement, expressed as `object_id` retiring while holding `hash`.
fn retirement_of(hash: [u8; 32], object_id: u64) -> ExtRelease {
    ExtRelease::retirement(hash, object_id)
}

// =============================================================================
// Crash simulation: pending entries from a previous run are cleaned up
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_crash_simulation_cleans_up_leftover_pending_entries() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xA1);
    let hash_hex = sha256_to_hex(&hash);

    // Simulate a crash before GC ran: manually insert meta+data (content-
    // addressed: data key derived from hash) and add the hash to pending.
    let meta_key = keys.ext_meta(&hash_hex);
    let data_key = keys.ext_data(&hash_hex);

    tv.client()
        .pool
        .hset::<(), _, _>(&meta_key, [("ref_count", "1".to_string())])
        .await
        .unwrap();

    tv.client()
        .pool
        .set::<(), _, _>(&data_key, b"orphaned-data".to_vec(), None, None, false)
        .await
        .unwrap();

    let pending_key = keys.ext_pending();
    tv.client()
        .pool
        .sadd::<(), _, _>(&pending_key, hash_hex.as_str())
        .await
        .unwrap();

    // Run GC sweep -- simulates startup recovery.
    let stats = process_pending_gc(
        &store,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;

    assert_eq!(stats.processed, 1);
    assert_eq!(stats.deleted, 1);

    // Data and metadata should be deleted.
    let exists_meta: bool = tv.client().pool.exists(&meta_key).await.unwrap();
    assert!(!exists_meta, "metadata key should be deleted after GC");

    let exists_data: bool = tv.client().pool.exists(&data_key).await.unwrap();
    assert!(!exists_data, "data key should be deleted after GC");

    // Pending set should be empty.
    let pending_count: u64 = tv.client().pool.scard(&pending_key).await.unwrap();
    assert_eq!(pending_count, 0, "pending set should be empty after GC");
}

// =============================================================================
// Partial crash: one entry's metadata already deleted
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_partial_crash_handles_already_deleted_metadata() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let h1 = test_hash(0xB1);
    let h2 = test_hash(0xB2);
    let h1_hex = sha256_to_hex(&h1);
    let h2_hex = sha256_to_hex(&h2);

    // h1: has metadata and data (normal pending entry).
    let meta_key_1 = keys.ext_meta(&h1_hex);
    let data_key_1 = keys.ext_data(&h1_hex);
    tv.client()
        .pool
        .hset::<(), _, _>(&meta_key_1, [("ref_count", "1".to_string())])
        .await
        .unwrap();
    tv.client()
        .pool
        .set::<(), _, _>(&data_key_1, b"data-1".to_vec(), None, None, false)
        .await
        .unwrap();

    // h2: metadata already deleted (simulates partial crash -- previous GC
    // deleted meta but crashed before removing from pending).
    // Only add to pending set, no metadata/data keys.

    let pending_key = keys.ext_pending();
    tv.client()
        .pool
        .sadd::<(), _, _>(&pending_key, vec![h1_hex.as_str(), h2_hex.as_str()])
        .await
        .unwrap();

    // GC should complete without error, processing both entries.
    let stats = process_pending_gc(
        &store,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;

    assert_eq!(stats.processed, 2);
    // Both counted as deleted: h1 because ref_count reached 0, h2 because
    // HINCRBY on missing key creates it at -1 which triggers deletion cleanup.
    assert_eq!(stats.deleted, 2);

    // h1 data/meta deleted.
    let exists_meta_1: bool = tv.client().pool.exists(&meta_key_1).await.unwrap();
    assert!(!exists_meta_1, "h1 metadata should be deleted");
    let exists_data_1: bool = tv.client().pool.exists(&data_key_1).await.unwrap();
    assert!(!exists_data_1, "h1 data should be deleted");

    // Pending set should be empty.
    let pending_count: u64 = tv.client().pool.scard(&pending_key).await.unwrap();
    assert_eq!(pending_count, 0, "pending set should be empty after GC");
}

// =============================================================================
// End-to-end: store external values, schedule release, run GC
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_end_to_end_store_release_cleanup() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xC1);
    let hash_hex = sha256_to_hex(&hash);
    let data = b"e2e-external-data";

    // Store the external value via the normal write path.
    store
        .store_values(&[(&hash, data.as_slice())])
        .await
        .unwrap();

    // Verify it exists.
    let meta_key = keys.ext_meta(&hash_hex);
    let ref_count: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(ref_count, "1");

    // Schedule release (simulating layer delete/overwrite).
    store
        .schedule_release(&[retirement_of(hash, OBJ_A)])
        .await
        .unwrap();

    // Run GC.
    let stats = process_pending_gc(
        &store,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 1);
    assert_eq!(stats.deleted, 1);

    // Value should be fully cleaned up.
    let exists_meta: bool = tv.client().pool.exists(&meta_key).await.unwrap();
    assert!(!exists_meta, "metadata should be deleted after GC");

    let pending_key = keys.ext_pending();
    let pending_count: u64 = tv.client().pool.scard(&pending_key).await.unwrap();
    assert_eq!(pending_count, 0);
}

// =============================================================================
// Refcount sharing: two references, one release -> value preserved
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_refcount_sharing_preserves_value_when_still_referenced() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xD1);
    let hash_hex = sha256_to_hex(&hash);
    let data = b"shared-external";

    // Store twice -> ref_count = 2.
    store
        .store_values(&[(&hash, data.as_slice())])
        .await
        .unwrap();
    store
        .store_values(&[(&hash, data.as_slice())])
        .await
        .unwrap();

    let meta_key = keys.ext_meta(&hash_hex);
    let ref_count: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(ref_count, "2");

    // Schedule one release — one of the two referencing objects retires.
    store
        .schedule_release(&[retirement_of(hash, OBJ_A)])
        .await
        .unwrap();

    // Run GC.
    let stats = process_pending_gc(
        &store,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 1);
    assert_eq!(stats.deleted, 0, "value still has one reference");

    // Value should still exist with ref_count = 1.
    let exists_meta: bool = tv.client().pool.exists(&meta_key).await.unwrap();
    assert!(exists_meta, "metadata should still exist");

    let new_ref: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(new_ref, "1", "ref_count should be decremented to 1");

    // Data should still be readable via load_values.
    let loaded = store.load_values(&[&hash]).await.unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].1, data);
}

// =============================================================================
// Second release after refcount reaches 1 -> value deleted
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_second_release_deletes_after_refcount_exhausted() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xD2);
    let hash_hex = sha256_to_hex(&hash);

    // Store twice -> ref_count = 2.
    store.store_values(&[(&hash, b"shared")]).await.unwrap();
    store.store_values(&[(&hash, b"shared")]).await.unwrap();

    // First release + GC -> ref_count = 1, value preserved. b.qpf: the two
    // releases must name two DIFFERENT retiring objects — that is what "two
    // references" means now, and what makes them two distinct pending members
    // rather than one collapsed one.
    store
        .schedule_release(&[retirement_of(hash, OBJ_A)])
        .await
        .unwrap();
    let stats1 = process_pending_gc(
        &store,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats1.deleted, 0);

    // Second release + GC -> ref_count = 0, value deleted.
    store
        .schedule_release(&[retirement_of(hash, OBJ_B)])
        .await
        .unwrap();
    let stats2 = process_pending_gc(
        &store,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats2.deleted, 1);

    let meta_key = keys.ext_meta(&hash_hex);
    let exists: bool = tv.client().pool.exists(&meta_key).await.unwrap();
    assert!(!exists, "metadata should be deleted after second GC");
}

// =============================================================================
// Cluster-mode GC sweep (b.ohv item 1).
//
// Gated on `LIVE_EDIT_TEST_CLUSTER_URL`. Validates that `process_pending_gc`
// works correctly when the seeded hashes hash to different slots — the GC
// claims happen via `le:ext:{gc}:pending` (one shard owns the pending set
// via the `{gc}` hash tag), but the per-hash meta+data UNLINK happens on
// whichever shard owns `{hash_hex}`. Both moves must succeed end-to-end on
// cluster. A regression that dropped the `{hash_hex}` tag from `ext_meta`
// or `ext_data` would land the UNLINK on the wrong shard and silently leak
// the entries.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_gc_processes_hashes_spread_across_shards() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };

    let tv = live_edit::storage::test_support::TestValkey::setup_with_url(cluster_url, 4).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "cluster URL required for cluster GC regression"
    );

    let store = ext_store(&tv);
    let keys = tv.key_schema();

    // Seed N distinct hashes — different first bytes spread them across
    // shards (birthday-paradox at 16 hashes vs 3 shards = effectively certain
    // every shard sees at least one hash).
    const HASHES: u8 = 16;
    let mut hashes: Vec<[u8; 32]> = Vec::with_capacity(HASHES as usize);
    for i in 0..HASHES {
        let mut h = [0u8; 32];
        h[0] = i;
        h[1] = i.wrapping_mul(31);
        hashes.push(h);
    }

    // Store each twice so ref_count = 2 (one release won't delete; two will).
    for h in &hashes {
        store
            .store_values(&[(h, b"cluster-gc-payload")])
            .await
            .expect("store_values must succeed on cluster");
        store
            .store_values(&[(h, b"cluster-gc-payload")])
            .await
            .expect("second store must succeed (ref_count -> 2)");
    }

    // Two releases per hash -> all should hit ref_count=0 and be deleted.
    // b.qpf: the two passes retire two DIFFERENT objects, so each hash gets two
    // distinct pending members. Re-releasing the same `(hash, object, event)`
    // would collapse in the Set and the second pass would find nothing.
    let releases_a: Vec<ExtRelease> = hashes.iter().map(|h| retirement_of(*h, OBJ_A)).collect();
    let releases_b: Vec<ExtRelease> = hashes.iter().map(|h| retirement_of(*h, OBJ_B)).collect();
    store
        .schedule_release(&releases_a)
        .await
        .expect("first schedule_release must succeed on cluster");
    let stats1 = process_pending_gc(
        &store,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(
        stats1.deleted, 0,
        "first GC pass with ref_count=2 must not delete anything"
    );

    store
        .schedule_release(&releases_b)
        .await
        .expect("second schedule_release must succeed on cluster");
    let stats2 = process_pending_gc(
        &store,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(
        stats2.deleted as usize,
        hashes.len(),
        "second GC pass must delete every hash now that ref_count=0; \
         got deleted={}, expected {}",
        stats2.deleted,
        hashes.len()
    );

    // Every shard's per-hash keys must be gone — pin slot-co-located UNLINK.
    for hash in &hashes {
        let hash_hex = sha256_to_hex(hash);
        let meta_key = keys.ext_meta(&hash_hex);
        let data_key = keys.ext_data(&hash_hex);
        let meta_exists: bool = tv.client().pool.exists(&meta_key).await.unwrap();
        let data_exists: bool = tv.client().pool.exists(&data_key).await.unwrap();
        assert!(
            !meta_exists,
            "cluster GC must UNLINK {meta_key} on its owner shard"
        );
        assert!(
            !data_exists,
            "cluster GC must UNLINK {data_key} on its owner shard"
        );
    }
}
