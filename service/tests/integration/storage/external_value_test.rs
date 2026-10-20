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

// Integration tests for ValkeyExternalValueStore.
//
// Tests run against a real Valkey instance via the TestValkey fixture. Each test
// gets an isolated key namespace so tests can run in parallel safely.
//
// Requires a running Valkey. Address is read from TEST_VALKEY_URL, defaulting to
// redis://127.0.0.1:6379.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface,
    TransactionInterface,
};

use live_edit::storage::external_values::{
    ExtRelease, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::test_support::TestValkey;

use crate::storage::b_1jt_rollout_tests::{plant_own_claim, STALE_OWN_CLAIM_AGE};

/// Helper: build a ValkeyExternalValueStore from a TestValkey fixture.
fn ext_store(tv: &TestValkey) -> ValkeyExternalValueStore {
    ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
}

/// Helper: create a distinct 32-byte hash from a single byte value.
fn test_hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

/// The object every single-release test in this file retires (b.qpf).
///
/// A pending-set member is `{hash_hex}:{object_id}:{event}`, so every release
/// needs an owning object. Which id it is does not matter where only ONE
/// decrement is owed; it matters a great deal where two are (see `gc_test.rs`'s
/// `OBJ_A` / `OBJ_B`).
const RELEASING_OBJECT: u64 = 8100;

/// One owed decrement: `RELEASING_OBJECT` retiring while it held `hash`.
fn retirement_of(hash: [u8; 32]) -> ExtRelease {
    ExtRelease::retirement(hash, RELEASING_OBJECT)
}

// =============================================================================
// store_values: new hash
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_new_hash_creates_meta_and_data_keys() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xAA);
    let data = b"hello external world";

    store
        .store_values(&[(&hash, data.as_slice())])
        .await
        .unwrap();

    // Verify metadata hash exists with ref_count=1.
    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);

    let ref_count: String = tv
        .client()
        .pool
        .hget(&meta_key, "ref_count")
        .await
        .expect("HGET ref_count must succeed");
    assert_eq!(ref_count, "1");

    // Verify data key exists at the content-addressed location.
    let data_key = keys.ext_data(&hash_hex);
    let stored: bytes::Bytes = tv
        .client()
        .pool
        .get(&data_key)
        .await
        .expect("GET ext data must succeed");
    assert_eq!(stored.as_ref(), data);
}

// =============================================================================
// store_values: dedup (same hash twice)
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_same_hash_twice_increments_refcount_without_duplicate_data() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xBB);
    let data = b"dedup me";

    store
        .store_values(&[(&hash, data.as_slice())])
        .await
        .unwrap();
    store
        .store_values(&[(&hash, data.as_slice())])
        .await
        .unwrap();

    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);

    let ref_count: String = tv
        .client()
        .pool
        .hget(&meta_key, "ref_count")
        .await
        .expect("HGET ref_count must succeed");
    assert_eq!(ref_count, "2", "ref_count must be 2 after two stores");

    // Data bytes must round-trip exactly (content-addressed: same hash → same bytes).
    let results = store.load_values(&[&hash]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, data);
}

// =============================================================================
// store_values: two different hashes stored at distinct content-addressed keys
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_two_different_hashes_creates_independent_keys() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let h1 = test_hash(0x01);
    let h2 = test_hash(0x02);

    store
        .store_values(&[(&h1, b"data-a".as_slice())])
        .await
        .unwrap();
    store
        .store_values(&[(&h2, b"data-b".as_slice())])
        .await
        .unwrap();

    // Each hash has its own meta+data at the content-addressed key.
    for (hash, expected) in [(&h1, b"data-a" as &[u8]), (&h2, b"data-b")] {
        let hash_hex = sha256_to_hex(hash);
        let meta_key = keys.ext_meta(&hash_hex);
        let data_key = keys.ext_data(&hash_hex);

        let rc: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
        assert_eq!(rc, "1");

        let stored: bytes::Bytes = tv.client().pool.get(&data_key).await.unwrap();
        assert_eq!(stored.as_ref(), expected);
    }
}

// =============================================================================
// load_values: read stored data
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_returns_correct_bytes() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let hash = test_hash(0xCC);
    let data = b"payload bytes";

    store
        .store_values(&[(&hash, data.as_slice())])
        .await
        .unwrap();

    let results = store.load_values(&[&hash]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, hash);
    assert_eq!(results[0].1, data);
}

// =============================================================================
// load_values: missing hash returns empty
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_missing_hash_returns_empty() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let hash = test_hash(0xFF);
    let results = store.load_values(&[&hash]).await.unwrap();
    assert!(results.is_empty());
}

// =============================================================================
// load_values: batch with multiple hashes
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_batch_returns_all_stored_data() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let h1 = test_hash(0x01);
    let h2 = test_hash(0x02);
    let h3 = test_hash(0x03);

    store
        .store_values(&[
            (&h1, b"alpha".as_slice()),
            (&h2, b"beta".as_slice()),
            (&h3, b"gamma".as_slice()),
        ])
        .await
        .unwrap();

    let results = store.load_values(&[&h1, &h2, &h3]).await.unwrap();
    assert_eq!(results.len(), 3);

    let result_map: HashMap<[u8; 32], Vec<u8>> = results.into_iter().collect();
    assert_eq!(result_map[&h1], b"alpha");
    assert_eq!(result_map[&h2], b"beta");
    assert_eq!(result_map[&h3], b"gamma");
}

// =============================================================================
// load_values: empty input
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn load_batch_empty_input_returns_empty() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let results = store.load_values(&[]).await.unwrap();
    assert!(results.is_empty());
}

// =============================================================================
// process_pending: refcount=1 → deletes data and meta atomically
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_pending_deletes_when_refcount_reaches_zero() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xDD);
    store
        .store_values(&[(&hash, b"ephemeral".as_slice())])
        .await
        .unwrap();

    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);
    let data_key = keys.ext_data(&hash_hex);

    store
        .schedule_release(&[retirement_of(hash)])
        .await
        .unwrap();
    let batch = store
        .process_pending(
            &live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .unwrap();
    assert!(batch.processed >= 1);

    // Meta key should be gone.
    let exists: bool = tv.client().pool.exists(&meta_key).await.unwrap();
    assert!(!exists, "meta key must be deleted after GC");

    // Data key should be gone.
    let data_exists: bool = tv.client().pool.exists(&data_key).await.unwrap();
    assert!(!data_exists, "data key must be deleted after GC");

    // Pending set should be empty.
    let pending_count: u64 = tv.client().pool.scard(keys.ext_pending()).await.unwrap();
    assert_eq!(pending_count, 0, "pending set must be empty after GC");

    // Inflight set should be empty (no crash).
    // b.j9j item 2: inflight is a sorted set; use ZCARD.
    let inflight_count: u64 = tv.client().pool.zcard(keys.ext_inflight()).await.unwrap();
    assert_eq!(
        inflight_count, 0,
        "inflight set must be empty after clean GC"
    );
}

// =============================================================================
// process_pending: refcount=2 → decrements to 1, data still present
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_pending_decrements_refcount_and_keeps_data() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xEE);
    // Store twice → ref_count = 2.
    store
        .store_values(&[(&hash, b"shared".as_slice())])
        .await
        .unwrap();
    store
        .store_values(&[(&hash, b"shared".as_slice())])
        .await
        .unwrap();

    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);

    // Verify ref_count = 2 before GC.
    let rc_before: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(rc_before, "2");

    store
        .schedule_release(&[retirement_of(hash)])
        .await
        .unwrap();
    store
        .process_pending(
            &live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .unwrap();

    // ref_count should be 1 now.
    let rc_after: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(rc_after, "1");

    // Data should still be readable.
    let results = store.load_values(&[&hash]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, b"shared");
}

// =============================================================================
// process_pending: already-deleted metadata → skips gracefully
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_pending_with_missing_metadata_skips_gracefully() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xFF);
    // Add to pending without storing — simulates already-deleted metadata.
    store
        .schedule_release(&[retirement_of(hash)])
        .await
        .unwrap();

    // Should not error.
    let batch = store
        .process_pending(
            &live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .unwrap();
    assert!(batch.processed >= 1);

    // The meta key that HINCRBY created with -1 should be cleaned up.
    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);
    let exists: bool = tv.client().pool.exists(&meta_key).await.unwrap();
    assert!(
        !exists,
        "stale meta key created by HINCRBY on missing key should be deleted"
    );
}

// =============================================================================
// Concurrent dedup atomicity (PM gap G3)
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_store_same_hash_increments_refcount_exactly_once_per_call() {
    const WRITERS: usize = 10;

    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();

    let hash = test_hash(0xAB);
    let data = b"concurrent dedup test data";

    // Barrier-driven lockstep entry (b.ohv item 7). Without it the spawn
    // ramp-up plus the fred connect-pool warm-up can let one writer
    // complete its `store_values` round-trip before another even begins —
    // collapsing 10 concurrent writes to a sequential 10-write benchmark
    // that exercises a different code path. With the barrier every writer
    // hits the MULTI/EXEC inside Valkey simultaneously, so the meta-key
    // HSETNX race + ref_count HINCRBY atomicity invariant is genuinely
    // contended.
    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));

    // Spawn WRITERS concurrent store_values calls with the same hash.
    let mut handles = Vec::new();
    for _ in 0..WRITERS {
        let store = ext_store(&tv);
        let hash_copy = hash;
        let data_copy = data.to_vec();
        let barrier_i = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier_i.wait().await;
            store
                .store_values(&[(&hash_copy, data_copy.as_slice())])
                .await
                .unwrap();
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    // Verify: ref_count = 10 (one HINCRBY per call, atomic with SET data).
    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);

    let ref_count: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(
        ref_count, "10",
        "10 concurrent stores must produce ref_count=10"
    );

    // Meta must NOT have the legacy `value_id` sentinel field. Under the
    // old HSETNX+INCR protocol a crash window could leave `value_id=0` in
    // meta (b.i8x Bug 1); under the new content-addressed MULTI/EXEC there
    // is no such field.
    let has_value_id: bool = tv
        .client()
        .pool
        .hexists(&meta_key, "value_id")
        .await
        .unwrap();
    assert!(
        !has_value_id,
        "meta must not carry a legacy `value_id` field (b.i8x regression)"
    );

    // Verify data is readable and correct.
    let store = ext_store(&tv);
    let results = store.load_values(&[&hash]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, data);
}

// =============================================================================
// schedule_release: verify Valkey set contents
// =============================================================================

/// b.qpf: a pending-set member is `{hash_hex}:{object_id}:{event}`, not the bare
/// `{hash_hex}` it was before.
///
/// The member format IS the contract — `process_pending` reads the hash back off
/// it as the prefix up to the first `:`, and the event token is what keeps two
/// genuinely-distinct owed decrements on one blob from collapsing into one Set
/// member. Asserted as the exact string rather than "starts with the hex" so a
/// regression that dropped the identity suffix (restoring the collapse defect)
/// fails here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn schedule_release_adds_identity_carrying_members_to_valkey_set() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let h1 = test_hash(0x01);
    let h2 = test_hash(0x02);

    store
        .schedule_release(&[retirement_of(h1), retirement_of(h2)])
        .await
        .unwrap();

    let members: Vec<String> = tv.client().pool.smembers(keys.ext_pending()).await.unwrap();
    assert_eq!(members.len(), 2);
    for hash in [h1, h2] {
        let expected = format!("{}:{RELEASING_OBJECT}:ret", sha256_to_hex(&hash));
        assert!(
            members.contains(&expected),
            "the member must carry the releasing object and the event that owes the decrement; \
             expected {expected:?} in {members:?}"
        );
    }
}

// =============================================================================
// schedule_release: empty input is a no-op
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn schedule_release_empty_is_noop() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    store.schedule_release(&[]).await.unwrap();

    let count: u64 = tv.client().pool.scard(keys.ext_pending()).await.unwrap();
    assert_eq!(count, 0);
}

// =============================================================================
// Data integrity: large payload round-trip
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn large_payload_round_trips_correctly() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let hash = test_hash(0xCD);
    // 64 KB of pattern data.
    let data: Vec<u8> = (0..65536).map(|i| (i % 256) as u8).collect();

    store
        .store_values(&[(&hash, data.as_slice())])
        .await
        .unwrap();

    let results = store.load_values(&[&hash]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, data, "large payload must round-trip exactly");
}

// =============================================================================
// store_values: batch of multiple distinct values in a single call
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_batch_multiple_distinct_values_creates_all_entries() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let h1 = test_hash(0x10);
    let h2 = test_hash(0x20);
    let h3 = test_hash(0x30);

    store
        .store_values(&[
            (&h1, b"alpha".as_slice()),
            (&h2, b"beta".as_slice()),
            (&h3, b"gamma".as_slice()),
        ])
        .await
        .unwrap();

    // Each hash should have ref_count=1 and a content-addressed data key.
    for (hash, expected_data) in [(&h1, b"alpha" as &[u8]), (&h2, b"beta"), (&h3, b"gamma")] {
        let hash_hex = sha256_to_hex(hash);
        let meta_key = keys.ext_meta(&hash_hex);
        let data_key = keys.ext_data(&hash_hex);

        let ref_count: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
        assert_eq!(ref_count, "1", "ref_count must be 1 for hash {hash_hex}");

        let stored: bytes::Bytes = tv.client().pool.get(&data_key).await.unwrap();
        assert_eq!(
            stored.as_ref(),
            expected_data,
            "data must match for hash {hash_hex}"
        );
    }
}

// =============================================================================
// store_values: batch with mixed new and existing hashes
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_batch_mixed_new_and_existing_increments_correctly() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let h_existing = test_hash(0x40);
    let h_new = test_hash(0x50);

    // Pre-store h_existing so it already has ref_count=1.
    store
        .store_values(&[(&h_existing, b"existing-data".as_slice())])
        .await
        .unwrap();

    // Now store both in one batch: h_existing (dedup) + h_new (new).
    store
        .store_values(&[
            (&h_existing, b"existing-data".as_slice()),
            (&h_new, b"new-data".as_slice()),
        ])
        .await
        .unwrap();

    // h_existing: ref_count should be 2.
    let hash_hex_existing = sha256_to_hex(&h_existing);
    let meta_existing = keys.ext_meta(&hash_hex_existing);
    let rc: String = tv
        .client()
        .pool
        .hget(&meta_existing, "ref_count")
        .await
        .unwrap();
    assert_eq!(rc, "2", "existing hash ref_count must be 2");

    // h_new: ref_count should be 1, data stored.
    let hash_hex_new = sha256_to_hex(&h_new);
    let meta_new = keys.ext_meta(&hash_hex_new);
    let rc_new: String = tv.client().pool.hget(&meta_new, "ref_count").await.unwrap();
    assert_eq!(rc_new, "1", "new hash ref_count must be 1");

    let results = store.load_values(&[&h_new]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, b"new-data");
}

// =============================================================================
// store_values: same hash passed twice in one batch
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_batch_duplicate_hash_in_same_call_increments_refcount_twice() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0x60);

    // Pass the same hash twice in a single batch.
    store
        .store_values(&[
            (&hash, b"data-copy-1".as_slice()),
            (&hash, b"data-copy-2".as_slice()),
        ])
        .await
        .unwrap();

    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);

    // ref_count must be 2: one HINCRBY per entry in the batch.
    let ref_count: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(
        ref_count, "2",
        "same hash twice in one batch must produce ref_count=2"
    );

    // Data should be retrievable.
    let results = store.load_values(&[&hash]).await.unwrap();
    assert_eq!(results.len(), 1);
}

// =============================================================================
// store_values: empty batch is a no-op (integration)
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_empty_batch_is_noop() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    // Should succeed without any Valkey writes.
    store.store_values(&[]).await.unwrap();
}

// =============================================================================
// b.i8x Bug 1 regression: store_values is crash-safe
// =============================================================================

/// Regression test for b.i8x Bug 1: under the old HSETNX+INCR protocol a
/// crash between phase-1 (HSETNX meta) and phase-3 (SET data + HSET
/// meta.value_id) left the meta hash permanently pointing at `value_id=0`
/// with no data key. Subsequent writers' HSETNX returned `false` and the
/// hash stayed broken forever.
///
/// Under the new content-addressed MULTI/EXEC protocol, meta and data are
/// created together inside a single atomic transaction, so this crash
/// window does not exist. We verify the invariant directly: after a
/// successful `store_values`, there is no sentinel state (no `value_id=0`
/// field) and both meta+data are in sync.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_values_produces_no_sentinel_state_on_success() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0x77);
    let data = b"no sentinel allowed";

    store
        .store_values(&[(&hash, data.as_slice())])
        .await
        .unwrap();

    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);
    let data_key = keys.ext_data(&hash_hex);

    // The legacy sentinel field must not exist.
    let has_value_id: bool = tv
        .client()
        .pool
        .hexists(&meta_key, "value_id")
        .await
        .unwrap();
    assert!(
        !has_value_id,
        "meta must not carry a legacy `value_id` field (b.i8x Bug 1 regression)"
    );

    // ref_count must be a concrete positive integer.
    let rc: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(rc, "1");

    // Data must exist at the content-addressed location.
    let data_exists: bool = tv.client().pool.exists(&data_key).await.unwrap();
    assert!(
        data_exists,
        "data key must exist whenever meta exists (atomic MULTI/EXEC)"
    );
}

// =============================================================================
// b.i8x Bug 2 regression: GC inflight set + the in-place re-claim
// =============================================================================

/// Regression test for b.i8x Bug 2 combined with Bug 1: verify that the GC
/// path recovers a crash end-to-end. If a previous sweep of this build
/// crashed after its claim but before `HINCRBY -1`, a later pass re-claims
/// the stale claim in place (b.1jt) and handles it (the meta has NOT been
/// decremented yet, so ref_count goes 1→0 and the hash is deleted).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_recovers_from_crash_between_smove_and_decrement() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xB1);
    let hash_hex = sha256_to_hex(&hash);
    store
        .store_values(&[(&hash, b"gc-recovery-test".as_slice())])
        .await
        .unwrap();

    // Simulate a crash between the claim and HINCRBY: hash sits in
    // `inflight`, ref_count is still 1 in meta. The claim is this build's
    // (an encoded score) and older than the re-claim age.
    plant_own_claim(&tv, &hash_hex, STALE_OWN_CLAIM_AGE).await;
    // No entry in `pending` — the crashed GC had already moved it out.

    // A regular GC pass now processes it: HINCRBY -1 → 0 → DEL meta+data.
    let batch = store
        .process_pending(
            &live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .unwrap();
    assert_eq!(batch.processed, 1);
    assert_eq!(batch.deleted, 1);

    let meta_key = keys.ext_meta(&hash_hex);
    let data_key = keys.ext_data(&hash_hex);
    assert!(!(tv.client().pool.exists::<bool, _>(&meta_key).await.unwrap()));
    assert!(!(tv.client().pool.exists::<bool, _>(&data_key).await.unwrap()));
}

/// A crash between a member's decrement and its completion is survivable
/// without a second decrement (b.yuo). The crashed sweeper's decrement
/// applied (and recorded the member on the hash's own slot), and the member
/// sits in `inflight`; a later pass re-claims it in place, and its re-run of
/// the decrement answers `replayed` instead of decrementing again — a blob
/// shared with another reference keeps it.
///
/// Before b.yuo the protocol was only at-least-once: the re-run decremented
/// again, deleting a blob a live layer still referenced. (Before b.kf1 the
/// decrement and the delete were separate round trips; one script has made
/// "decremented but not deleted" unreachable, so the crash window is now
/// between the decrement and the completion.)
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_recovers_from_crash_between_decrement_and_completion_without_a_second_decrement() {
    use live_edit::storage::external_values::{
        GC_CLAIM_PENDING_SCRIPT, GC_DECREMENT_AND_COLLECT_SCRIPT, GC_DECREMENT_FENCE_MS,
    };

    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xB2);
    let hash_hex = sha256_to_hex(&hash);
    // Two references: the release owes one decrement, the other holder keeps
    // the blob alive.
    for _ in 0..2 {
        store
            .store_values(&[(&hash, b"partial-gc-crash".as_slice())])
            .await
            .unwrap();
    }
    let release = retirement_of(hash);
    let member = release.member();
    store.schedule_release(&[release]).await.unwrap();

    // The crashed sweeper: claim, decrement, then nothing — the member stays
    // in `inflight` with its decrement applied.
    let claim_ms: i64 = GC_CLAIM_PENDING_SCRIPT
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            vec![
                keys.ext_pending(),
                keys.ext_inflight(),
                keys.ext_pending_age(),
                keys.ext_pending_bytes(),
                keys.ext_pending_total(),
                keys.ext_settled(),
            ],
            vec![member.clone()],
        )
        .await
        .unwrap();
    assert!(claim_ms > 0, "the claim must succeed");
    let (action, ref_count, _): (String, i64, i64) = GC_DECREMENT_AND_COLLECT_SCRIPT
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            vec![
                keys.ext_meta(&hash_hex),
                keys.ext_data(&hash_hex),
                keys.ext_applied(&hash_hex),
            ],
            vec![
                member.clone(),
                claim_ms.to_string(),
                GC_DECREMENT_FENCE_MS.to_string(),
            ],
        )
        .await
        .unwrap();
    assert_eq!((action.as_str(), ref_count), ("decremented", 1));
    // Age the claim past the re-claim age (keeping this build's encoding).
    plant_own_claim(&tv, &member, STALE_OWN_CLAIM_AGE).await;

    // The next pass must re-claim and settle the member without decrementing.
    let batch = store
        .process_pending(
            &live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        (batch.processed, batch.replayed, batch.deleted),
        (1, 1, 0),
        "the re-claimed member is a replay"
    );

    let meta_key = keys.ext_meta(&hash_hex);
    let data_key = keys.ext_data(&hash_hex);
    let rc: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(rc, "1", "the other holder's reference survives");
    assert!(tv.client().pool.exists::<bool, _>(&data_key).await.unwrap());
    let inflight: u64 = tv.client().pool.zcard(keys.ext_inflight()).await.unwrap();
    assert_eq!(inflight, 0);
}

/// After a clean `process_pending` with no crashes, the inflight set is
/// empty. This is a positive invariant test: we do NOT want the inflight
/// set to linger as a leak on the happy path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_pending_leaves_inflight_set_empty_on_clean_path() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let h1 = test_hash(0xC1);
    let h2 = test_hash(0xC2);
    store
        .store_values(&[(&h1, b"x".as_slice()), (&h2, b"y".as_slice())])
        .await
        .unwrap();

    store
        .schedule_release(&[retirement_of(h1), retirement_of(h2)])
        .await
        .unwrap();
    store
        .process_pending(
            &live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .unwrap();

    // b.j9j item 2: inflight is a sorted set; use ZCARD.
    let inflight_count: u64 = tv.client().pool.zcard(keys.ext_inflight()).await.unwrap();
    assert_eq!(inflight_count, 0);
}

/// b.i8x, with no startup drain (b.yuo, b.1jt): a GC run cleans up both an
/// inflight survivor of a previous crash — a stale claim of this build's,
/// meta at ref_count=0 with data still present, which a pass re-claims in
/// place — and a pending-only survivor enqueued by an older build's bare
/// `SADD` (meta ref_count=0, data present, no age entry), which a pass stamps
/// with an age first and collects on the next. Both end with meta and data
/// gone and both sets empty.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gc_run_cleans_both_a_stale_own_claim_and_an_unaged_pending_survivor() {
    use live_edit::storage::external_values::process_pending_gc;

    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    // Inflight survivor: simulates a GC sweep that crashed between
    // HINCRBY -1 and the MULTI{DEL meta; DEL data}. Meta has ref_count=0,
    // data still exists, hash still sits in the inflight set.
    let h_inflight = test_hash(0xE1);
    let h_inflight_hex = sha256_to_hex(&h_inflight);
    let meta_inflight = keys.ext_meta(&h_inflight_hex);
    let data_inflight = keys.ext_data(&h_inflight_hex);
    let _: () = tv
        .client()
        .pool
        .hset(&meta_inflight, [("ref_count", "0")])
        .await
        .unwrap();
    let _: () = tv
        .client()
        .pool
        .set(
            &data_inflight,
            b"inflight-data".as_slice(),
            None,
            None,
            false,
        )
        .await
        .unwrap();
    // This build's claim (an encoded score), older than the re-claim age.
    plant_own_claim(&tv, &h_inflight_hex, STALE_OWN_CLAIM_AGE).await;

    // Pending survivor: an older build's bare `SADD` (no age entry) that no
    // GC has claimed yet. Meta already decremented to 0 by an earlier
    // partial sweep; still queued in pending for the delete step.
    let h_pending = test_hash(0xE2);
    let h_pending_hex = sha256_to_hex(&h_pending);
    let meta_pending = keys.ext_meta(&h_pending_hex);
    let data_pending = keys.ext_data(&h_pending_hex);
    let _: () = tv
        .client()
        .pool
        .hset(&meta_pending, [("ref_count", "0")])
        .await
        .unwrap();
    let _: () = tv
        .client()
        .pool
        .set(&data_pending, b"pending-data".as_slice(), None, None, false)
        .await
        .unwrap();
    let _: () = tv
        .client()
        .pool
        .sadd(keys.ext_pending(), &h_pending_hex)
        .await
        .unwrap();

    // One GC run: the first pass re-claims the stale claim and stamps the
    // un-aged member; the next collects the stamped one.
    let stats = process_pending_gc(
        &store,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(
        stats.processed, 2,
        "both survivors must be processed in the sweep"
    );
    assert_eq!(
        stats.deleted, 2,
        "both survivors reached ref_count <= 0 and must be deleted"
    );

    // Both hashes fully cleaned up: meta + data gone, sets empty.
    for meta in [&meta_inflight, &meta_pending] {
        let exists: bool = tv.client().pool.exists(meta).await.unwrap();
        assert!(!exists, "meta {meta} must be deleted");
    }
    for data in [&data_inflight, &data_pending] {
        let exists: bool = tv.client().pool.exists(data).await.unwrap();
        assert!(!exists, "data {data} must be deleted");
    }
    let pending_count: u64 = tv.client().pool.scard(keys.ext_pending()).await.unwrap();
    assert_eq!(pending_count, 0, "ext_pending must be empty after sweep");
    // b.j9j item 2: inflight is a sorted set; use ZCARD.
    let inflight_count: u64 = tv.client().pool.zcard(keys.ext_inflight()).await.unwrap();
    assert_eq!(inflight_count, 0, "ext_inflight must be empty after sweep");
}

/// Concurrent-store-during-GC race (b.i8x): `store_values` for a hash that
/// GC already claimed (moved into `ext:inflight` and decremented to
/// ref_count=0) is accepted at the Valkey level per SDD §11.1 — the write
/// path is a simple `SET data` + `HINCRBY ref_count +1` (one atomic
/// `ext_store_value.lua` call since b.rej) with no inflight-ownership check.
/// The GC worker, which already decided to
/// delete, then deletes meta+data anyway. This is documented at-least-once
/// behavior: the concurrent writer's bytes may be lost, and the caller is
/// expected to retry (re-store) if it observes the missing data.
///
/// The critical invariants asserted here are the *absence* of permanent
/// corruption:
///   - No dangling ref_count=-1 meta state.
///   - No orphan data key surviving without meta, or vice versa.
///   - Sets (pending + inflight) end up empty once the GC worker completes.
///   - A follow-up retry (re-run of store_values) recovers a healthy state
///     (ref_count=1, data present).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn store_values_during_gc_inflight_preserves_invariants_per_sdd_11_1() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xF3);
    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);
    let data_key = keys.ext_data(&hash_hex);

    // Initial store: ref_count=1, data present.
    store
        .store_values(&[(&hash, b"original-data".as_slice())])
        .await
        .unwrap();
    // Schedule for release: enters pending.
    let release = retirement_of(hash);
    store.schedule_release(&[release]).await.unwrap();

    // Simulate GC taking ownership: SREM pending + ZADD inflight (with
    // current timestamp, mirroring `gc_claim_pending.lua`), decrement
    // ref_count to 0. This is the exact state `process_pending` leaves
    // the keys in just before it runs the MULTI{DEL meta; DEL data}
    // step. b.j9j item 2: inflight is now a sorted set.
    //
    // b.qpf: the claim/inflight sets are MEMBER-keyed, and the member is
    // `{hash_hex}:{object_id}:{event}` — so the hand-rolled claim below must
    // move the same string `schedule_release` wrote. A bare-hex SREM would
    // remove nothing and leave the real member behind, and the invariants at
    // the bottom would then be asserted against a keyspace this test never
    // actually staged.
    let member = release.member();
    let removed: i64 = tv
        .client()
        .pool
        .srem(keys.ext_pending(), member.as_str())
        .await
        .unwrap();
    assert_eq!(removed, 1, "SREM must claim the pending entry");
    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as f64;
    let _: () = tv
        .client()
        .pool
        .zadd(
            keys.ext_inflight(),
            None,
            None,
            false,
            false,
            (now_ms, member.as_str()),
        )
        .await
        .unwrap();
    let new_ref: i64 = tv
        .client()
        .pool
        .hincrby(&meta_key, "ref_count", -1)
        .await
        .unwrap();
    assert_eq!(
        new_ref, 0,
        "post-decrement ref_count must be 0 (pre-delete state)"
    );

    // Racing writer comes in with the same hash while GC holds ownership.
    // Per SDD §11.1 this is a set-if-absent `SET … NX` of the data + `HINCRBY
    // +1` (one atomic `ext_store_value.lua` call) — it does NOT inspect
    // `ext_inflight`, so it bumps ref_count from 0 back to 1.
    store
        .store_values(&[(&hash, b"concurrent-data".as_slice())])
        .await
        .unwrap();

    let ref_mid: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(
        ref_mid, "1",
        "concurrent store bumps ref_count from 0 to 1 per SDD §11.1"
    );

    // GC finishes its sweep: it already decided `new_ref <= 0` above, so
    // it runs MULTI{DEL meta; DEL data}. The concurrent writer's bytes
    // are lost. This is the documented at-least-once outcome (Option A).
    let trx = tv.client().pool.next().multi();
    let _: () = trx.del(&meta_key).await.unwrap();
    let _: () = trx.del(&data_key).await.unwrap();
    trx.exec::<()>(true).await.unwrap();
    // b.j9j item 2: inflight is a sorted set; use ZREM.
    let _: () = tv
        .client()
        .pool
        .zrem(keys.ext_inflight(), member.as_str())
        .await
        .unwrap();

    // Invariant 1: no dangling ref_count=-1 (or any stray meta) state.
    let meta_exists: bool = tv.client().pool.exists(&meta_key).await.unwrap();
    assert!(
        !meta_exists,
        "meta must not survive GC as a dangling hash with negative refcount"
    );

    // Invariant 2: no orphan data without meta.
    let data_exists: bool = tv.client().pool.exists(&data_key).await.unwrap();
    assert!(
        !data_exists,
        "data must not survive GC without its meta (meta+data are atomic)"
    );

    // Invariant 3: sets are clean.
    let pending_count: u64 = tv.client().pool.scard(keys.ext_pending()).await.unwrap();
    assert_eq!(pending_count, 0);
    // b.j9j item 2: inflight is a sorted set; use ZCARD.
    let inflight_count: u64 = tv.client().pool.zcard(keys.ext_inflight()).await.unwrap();
    assert_eq!(inflight_count, 0);

    // Invariant 4: the caller's documented recovery — a retried store
    // brings the entry back to a healthy ref_count=1 with readable data.
    // This proves the system is *recoverable* from the race; the data
    // loss is transient, not permanent.
    store
        .store_values(&[(&hash, b"retry-after-loss".as_slice())])
        .await
        .unwrap();
    let rc_retry: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(rc_retry, "1", "retry must re-establish ref_count=1");
    let loaded = store.load_values(&[&hash]).await.unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].1, b"retry-after-loss");
}

// =============================================================================
// read_batch: Valkey-backed success + error-disambiguation paths
// =============================================================================

/// Happy path: read_batch returns exactly the stored bytes for every hash.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_batch_returns_stored_bytes_for_all_hashes() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let h1 = test_hash(0x11);
    let h2 = test_hash(0x22);
    let h3 = test_hash(0x33);

    store
        .store_values(&[
            (&h1, b"one".as_slice()),
            (&h2, b"two".as_slice()),
            (&h3, b"three".as_slice()),
        ])
        .await
        .unwrap();

    let result = store.read_batch(&[h1, h2, h3]).await.unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[&h1], b"one");
    assert_eq!(result[&h2], b"two");
    assert_eq!(result[&h3], b"three");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_batch_empty_input_performs_no_valkey_calls() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let result = store.read_batch(&[]).await.unwrap();
    assert!(result.is_empty());
}

/// A hash with no meta and no data must surface as `MetaNotFound` so the
/// caller can distinguish "unknown hash" from a data-integrity violation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_batch_unknown_hash_returns_meta_not_found() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let hash = test_hash(0x9F);
    let err = store.read_batch(&[hash]).await.unwrap_err();

    match err {
        live_edit::error::ServiceError::Internal(msg) => {
            assert!(
                msg.contains("meta not found"),
                "expected MetaNotFound message, got: {msg}"
            );
            let hash_hex = sha256_to_hex(&hash);
            assert!(
                msg.contains(&hash_hex),
                "error should include the hash_hex, got: {msg}"
            );
        }
        other => panic!("expected ServiceError::Internal, got: {other:?}"),
    }
}

/// Data-integrity violation: meta exists but the data key does not. This
/// state must be reported as `DataNotFound` (not MetaNotFound), because the
/// meta/data pair diverging indicates corruption rather than an unknown
/// hash. This directly exercises the HGET-data + HEXISTS-meta
/// disambiguation branch in `read_batch`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_batch_dangling_meta_returns_data_not_found() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xD0);
    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);

    // Seed a dangling meta: ref_count present but no data key. Under the
    // atomic MULTI/EXEC protocol this cannot happen on the happy path, but
    // read_batch still has to disambiguate the two failure modes if it ever
    // encounters one (b.i8x read-side guard).
    let _: () = tv
        .client()
        .pool
        .hset(&meta_key, [("ref_count", "1")])
        .await
        .unwrap();

    let err = store.read_batch(&[hash]).await.unwrap_err();
    match err {
        live_edit::error::ServiceError::Internal(msg) => {
            assert!(
                msg.contains("data not found"),
                "expected DataNotFound message, got: {msg}"
            );
            assert!(
                !msg.contains("meta not found"),
                "must not report MetaNotFound when meta exists, got: {msg}"
            );
        }
        other => panic!("expected ServiceError::Internal, got: {other:?}"),
    }
}

/// Mixed batch: some hashes present, one missing. `read_batch` fails fast
/// with the appropriate error variant — the hash that is actually missing
/// determines the variant.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_batch_mixed_hits_and_miss_returns_error() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let h_present = test_hash(0x11);
    let h_missing = test_hash(0xEE);

    store
        .store_values(&[(&h_present, b"data".as_slice())])
        .await
        .unwrap();

    // The missing hash has no meta and no data, so the read must surface
    // `MetaNotFound` (not `DataNotFound`). Variant-stable assertion: a
    // future change that silently downgrades the error to a different
    // variant (or plain `Internal`) must fail this test.
    let err = store.read_batch(&[h_present, h_missing]).await.unwrap_err();
    match err {
        live_edit::error::ServiceError::Internal(msg) => {
            assert!(
                msg.contains("meta not found"),
                "expected MetaNotFound for missing hash in mixed batch, got: {msg}"
            );
            let missing_hex = sha256_to_hex(&h_missing);
            assert!(
                msg.contains(&missing_hex),
                "error must name the missing hash, got: {msg}"
            );
            assert!(
                !msg.contains("data not found"),
                "must not report DataNotFound when meta is absent, got: {msg}"
            );
        }
        other => panic!("expected ServiceError::Internal(MetaNotFound), got: {other:?}"),
    }
}

// =============================================================================
// store_values idempotency: retry semantics
// =============================================================================

/// Simulates an application-level retry after a client sees a transient
/// error: the writer issues the SAME `store_values` call twice. Under the
/// content-addressed protocol each call is one atomic `EVALSHA` of
/// `ext_store_value.lua`, so the retry simply produces +1 to ref_count (benign
/// leak, drained by the reference-count reconciler — see SDD §11.1 note). The
/// key invariants: data bytes are preserved and the meta row contains only the
/// expected fields.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn store_values_retry_after_success_is_additive_with_stable_data() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xA9);
    let data = b"retry-safe payload";

    // First call succeeds; the client treats the response as lost and
    // retries the exact same request.
    store
        .store_values(&[(&hash, data.as_slice())])
        .await
        .unwrap();
    store
        .store_values(&[(&hash, data.as_slice())])
        .await
        .unwrap();

    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);

    // ref_count lands at 2 — retry is "at-least-once", not "exactly-once".
    let rc: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(rc, "2");

    // Meta hash must contain EXACTLY `ref_count` + `last_store_ms`: no
    // `value_id`, no legacy sentinels. This is a regression guard against
    // re-introducing the old HSETNX+INCR protocol, which is why the assertion
    // is an exact set rather than a containment check — a stray sentinel is
    // the thing it exists to catch, and `contains_key` alone would not see one.
    //
    // b.rej added the second field: `ext_store_value.lua` stamps
    // `last_store_ms` from Valkey's own clock in the same atomic step as the
    // pre-flight `HINCRBY`. It is asserted here as a member of the exact set
    // (its VALUE is pinned in `b_rej_tests`), so a writer that stopped stamping
    // reddens this test too.
    let fields: std::collections::HashMap<String, String> =
        tv.client().pool.hgetall(&meta_key).await.unwrap();
    let mut names: Vec<&str> = fields.keys().map(String::as_str).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        vec!["last_store_ms", "ref_count"],
        "meta hash must carry exactly ref_count + last_store_ms; got fields: {fields:?}"
    );

    // Data bytes must survive the retry unchanged (content-addressed).
    let loaded = store.load_values(&[&hash]).await.unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].1, data);
}

/// Concurrent writers that race on the same hash each land exactly one
/// HINCRBY — no writer is lost, no writer double-counts. This is the
/// Valkey-level guarantee that underpins the dedup + retry story in SDD
/// §11.1. Two writers × two calls each → ref_count = 4.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn store_values_concurrent_writers_each_increment_refcount_once() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();

    let hash = test_hash(0xBE);
    let data = b"concurrent-refcount";

    let mut handles = Vec::new();
    for _ in 0..4 {
        let store = ext_store(&tv);
        let hash_copy = hash;
        let data_copy = data.to_vec();
        handles.push(tokio::spawn(async move {
            store
                .store_values(&[(&hash_copy, data_copy.as_slice())])
                .await
                .unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);
    let rc: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(
        rc, "4",
        "four concurrent writers must land four HINCRBYs exactly"
    );
}

// =============================================================================
// Cluster-mode regressions (b.ohv item 1).
//
// Gated on `LIVE_EDIT_TEST_CLUSTER_URL`. CI's `test-cluster:` job sets this
// to `redis-cluster://127.0.0.1:7000` and the single-node `test:` job leaves
// it unset, so these tests skip cleanly on the default path.
//
// What's at stake on cluster: `{hash_hex}` hash-tagged keys must co-locate on
// the same shard so the meta-HSET + data-SET MULTI/EXEC inside `store_values`
// stays inside one slot (no CROSSSLOT rejection). Multiple hashes spread
// across multiple shards by design — that's the dedup+slot-spread property
// SDD §11 relies on. We exercise both: a single hash's intra-slot MULTI/EXEC
// shape, and a fan-out write across many hashes that lands on >1 shards.
// =============================================================================

/// Cluster: storing N distinct hashes lands them across multiple shards.
/// Pre-fix code that accidentally dropped the `{hash_hex}` hash tag from
/// any of the key builders would re-route the keys to a different slot
/// from their meta sibling, surfacing as a CROSSSLOT error inside the
/// MULTI/EXEC. The fix keeps every per-hash key on the same slot — this
/// test confirms a fan-out of distinct hashes commits cleanly on cluster.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_store_many_distinct_hashes_spans_shards_without_crossslot() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };

    let tv = live_edit::storage::test_support::TestValkey::setup_with_url(cluster_url, 4).await;

    // Assert the fixture is actually clustered — a single-node URL would
    // skip the slot-spread coverage silently otherwise.
    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment \
         (use `redis-cluster://...`); the meta/data slot-co-location \
         contract is only exercised when `is_clustered()` is true"
    );

    let store = ext_store(&tv);
    let keys = tv.key_schema();

    // 32 distinct hashes — birthday-paradox math against 3 shards makes it
    // effectively certain that at least 2 shards see at least one write.
    let mut payloads: Vec<([u8; 32], Vec<u8>)> = Vec::with_capacity(32);
    for i in 0..32u8 {
        let mut h = [0u8; 32];
        h[0] = i;
        h[1] = i.wrapping_mul(7);
        payloads.push((h, format!("payload-{i}").into_bytes()));
    }

    let payload_refs: Vec<(&[u8; 32], &[u8])> =
        payloads.iter().map(|(h, v)| (h, v.as_slice())).collect();
    store.store_values(&payload_refs).await.expect(
        "cluster mode: per-hash MULTI/EXEC must commit cleanly for every hash \
             — a CROSSSLOT here would mean the `{hash_hex}` hash tag was dropped \
             on one of the key builders",
    );

    // Each hash must now have its `ref_count = 1` meta + corresponding data
    // key. The keys.ext_meta / keys.ext_data builders apply the hash tag,
    // so this is a direct pin of the slot-co-location contract.
    for (hash, payload) in &payloads {
        let hash_hex = sha256_to_hex(hash);
        let meta_key = keys.ext_meta(&hash_hex);
        let data_key = keys.ext_data(&hash_hex);

        let rc: String = tv
            .client()
            .pool
            .hget(&meta_key, "ref_count")
            .await
            .expect("HGET ref_count must succeed on cluster");
        assert_eq!(rc, "1", "cluster: each distinct hash lands ref_count=1");

        let loaded: Vec<u8> = tv
            .client()
            .pool
            .get(&data_key)
            .await
            .expect("GET data must succeed on cluster");
        assert_eq!(
            loaded, *payload,
            "cluster: per-hash data must round-trip via the {{hash_hex}}-tagged data key"
        );
    }
}

/// Cluster: concurrent writers on the SAME hash hit the same shard's
/// single-threaded slot, so the HSET+HINCRBY MULTI/EXEC is serialised
/// server-side. This is the cluster analogue of the
/// `store_values_concurrent_writers_each_increment_refcount_once` single-node
/// test — same invariant, but exercised under cluster's slot-routing path.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cluster_concurrent_writers_same_hash_increment_refcount_atomically() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set");
        return;
    };

    const WRITERS: usize = 8;

    let tv = live_edit::storage::test_support::TestValkey::setup_with_url(cluster_url, 8).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "cluster URL required"
    );

    let keys = tv.key_schema();
    let hash = test_hash(0xCC);
    let data = b"cluster-concurrent-payload";

    // Barrier so all writers MULTI/EXEC into the same slot simultaneously
    // (b.ohv item 7) — without it, the cluster's slot serialization is
    // exercised on consecutive non-overlapping writes, defeating the
    // concurrency intent.
    let barrier = Arc::new(tokio::sync::Barrier::new(WRITERS));

    let mut handles = Vec::with_capacity(WRITERS);
    for _ in 0..WRITERS {
        let store = ext_store(&tv);
        let hash_copy = hash;
        let data_copy = data.to_vec();
        let barrier_i = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier_i.wait().await;
            store
                .store_values(&[(&hash_copy, data_copy.as_slice())])
                .await
                .unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);
    let rc: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(
        rc,
        WRITERS.to_string(),
        "cluster: {WRITERS} concurrent writers must produce ref_count={WRITERS} \
         (HINCRBY atomic inside the slot owner)"
    );
}
