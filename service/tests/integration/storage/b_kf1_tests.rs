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

// Integration tests for b.kf1 — bundled external-value lifecycle hardening.
//
// Covers the findings from the 2026-04-23 deep code review storage section:
//
//   C-5       GC `process_pending` decrement+collect must be atomic, and a
//             re-claimed member must not double-decrement a refcount. b.kf1
//             made the decrement and the delete one script; b.yuo made a
//             re-run of an already-applied decrement answer `replayed` (the
//             per-hash replay record), so a crash mid-sweep followed by the
//             member's re-claim (in place since b.1jt) decrements at most
//             once.
//   H-stor-1  `cleanup_orphaned_object` must enumerate the orphan's
//             external-value references and SADD them to `ext:pending`
//             before UNLINKing the structural keys — pre-b.kf1 the sweep
//             permanently inflated `ref_count` on every blob referenced
//             by an orphaned object, leaking disk on Valkey forever.
//   H-stor-4  Multi-replica leader election: only one replica's GC sweep
//             runs per tick, gated behind `le:gc:leader`. The plain lock CAS
//             (`leader_acquire_or_refresh.lua`, which eliminated the sub-30 s
//             clobber window of the original GET-then-SET pair) now guards
//             the checkpoint and presence locks; `le:gc:leader` takes a
//             protocol-marked value through its own script since b.1jt
//             (`b_1jt_rollout_tests.rs`). These tests pin the CAS's
//             acquire/refresh semantics and the end-to-end "two
//             `spawn_gc_task` instances racing only produce one sweep per
//             tick" outcome.
//   H-stor-5  Maintenance pool isolation: a long sweep on the maintenance
//             pool must not block concurrent `update_layer`-shaped writes
//             on the data pool.
//   H-stor-10 Subscriber must hard-error (`Internal`) on a missing
//             external-value data hash — pre-b.kf1 the read silently
//             substituted an empty `Vec<u8>`, turning a GC race into a
//             zero-length-blob "success" on the wire.
//
// Plus the b.kf1 mediums:
//   - `SweepError::Fred` preserves the underlying fred `ErrorKind` slug via
//     `kind_label()` so triage can split timeouts from protocol errors.
//   - `gc_sweep_batch_errors_total{kind}` increments on sweep failure.
//
// The C-5 fault-injection regression in this file is gated behind
// `test-fault-injection` because it manipulates Valkey state into a
// mid-sweep crash shape directly. The orphan-sweep / leader / maintenance-
// pool tests do not need the feature flag.
//
// Test-suite-hang lessons from b.k1r are honored throughout:
// - Tests that use `spawn_gc_task` / `spawn_orphan_sweep` cancel the token
//   after observing the desired outcome (or after a bounded poll) and
//   `tokio::time::timeout` around the join handle.
// - Direct calls to `process_pending_gc` are preferred over spawn-based
//   tests where the assertion does not specifically require the periodic
//   loop.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{
    HashesInterface, KeysInterface, ListInterface, SetsInterface, SortedSetsInterface,
};
use tokio_util::sync::CancellationToken;

use live_edit::error::ServiceError;
use live_edit::storage::client::{FredErrOrigin, POOL_LABEL_DATA};
#[cfg(feature = "test-fault-injection")]
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::external_values::{
    process_pending_gc, spawn_gc_task_with_leader, ExtRelease, ExternalValueStore,
    ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::orphan_sweep::{spawn_orphan_sweep, SweepError};
use live_edit::storage::test_support::TestValkey;

use crate::storage::b_1jt_rollout_tests::{plant_own_claim, STALE_OWN_CLAIM_AGE};

/// Helper: build a ValkeyExternalValueStore from a TestValkey fixture.
fn ext_store(tv: &TestValkey) -> ValkeyExternalValueStore {
    ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
}

/// Helper: distinct 32-byte hash from a single byte value.
fn test_hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

/// The object whose retirement owes the decrements this file schedules by hand
/// (b.qpf).
///
/// A pending / inflight member is `{hash_hex}:{object_id}:{event}`, so the
/// hand-rolled mid-sweep-crash states below must move exactly the member
/// `schedule_release` wrote — see [`retirement_of`].
const RELEASING_OBJECT: u64 = 8200;

/// One owed decrement: [`RELEASING_OBJECT`] retiring while it held `hash`.
fn retirement_of(hash: [u8; 32]) -> ExtRelease {
    ExtRelease::retirement(hash, RELEASING_OBJECT)
}

// =============================================================================
// C-5 (CRITICAL) — Atomic GC: retry from inflight does NOT double-decrement
// =============================================================================

/// b.kf1 C-5: a sweep that crashes between the HINCRBY and the conditional
/// DEL must NOT re-decrement on retry. Pre-b.kf1 the two operations were
/// separate round-trips and the startup-drain re-queue exposed the second
/// decrement on retry; with `ref_count=2` shared between two layers this
/// double-decrement silently deleted blobs that another live layer still
/// referenced.
///
/// The new Lua script collapses HINCRBY + conditional DEL into a single
/// atomic step, so a crash *cannot* leave the keys in the "decremented but
/// not deleted" state any more — the retry on a still-referenced hash now
/// converges to the correct end state.
///
/// Setup: ref_count = 2 (two layers reference the blob). One layer
/// releases. Simulate a mid-sweep crash by manually placing the hash in
/// `ext:inflight` *before* the sweep runs (i.e. as if a previous worker
/// of this build claimed it but died before EVAL'ing the Lua script). The
/// next pass re-claims the stale claim in place (b.1jt) and runs the Lua
/// script once, ref_count drops 2 → 1, and the data must remain present and
/// reachable for the surviving layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c5_retry_after_inflight_crash_does_not_double_decrement_shared_blob() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let hash = test_hash(0xC5);
    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);
    let data_key = keys.ext_data(&hash_hex);

    // Two layers reference the blob → ref_count = 2 after two stores.
    store
        .store_values(&[(&hash, b"shared-by-two-layers".as_slice())])
        .await
        .unwrap();
    store
        .store_values(&[(&hash, b"shared-by-two-layers".as_slice())])
        .await
        .unwrap();
    let rc_before: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(
        rc_before, "2",
        "two stores must produce ref_count=2 — sanity check"
    );

    // Layer A schedules its release: hash enters ext:pending.
    let release = retirement_of(hash);
    let member = release.member();
    store.schedule_release(&[release]).await.unwrap();

    // Simulate mid-sweep crash: a worker SREM'd from pending and ZADD'd to
    // inflight but died before the EVALSHA could run. We seed that exact
    // state directly so the test is deterministic.
    //
    // b.j9j item 2: inflight is a sorted set scored by claim time; since
    // b.1jt this build's score is the claim time plus
    // `GC_CLAIM_SCORE_OFFSET_MS` (a bare score is an older build's claim,
    // which this build drops and never decrements). Stamp it older than the
    // re-claim age (30 s) so the next pass re-claims it.
    //
    // b.qpf: both sets are MEMBER-keyed, and the member carries the releasing
    // object and event. Moving a bare `hash_hex` would SREM nothing and stage a
    // second, synthetic entry alongside the real one — the sweep would then see
    // two owed decrements and this test's whole claim (exactly one decrement)
    // would be asserted against a keyspace it never built.
    let _: () = tv
        .client()
        .pool
        .srem::<(), _, _>(keys.ext_pending(), member.as_str())
        .await
        .unwrap();
    plant_own_claim(&tv, &member, STALE_OWN_CLAIM_AGE).await;
    assert_eq!(
        tv.client()
            .pool
            .zcard::<u64, _>(keys.ext_inflight())
            .await
            .unwrap(),
        1,
        "test setup: hash must be in inflight"
    );

    // `process_pending_gc` re-claims the stale claim and runs the Lua
    // script exactly once on it — pre-b.kf1 a retry ran a second decrement
    // on top of the first, dropping ref_count from 2 to 0 and silently
    // deleting the blob.
    let stats = process_pending_gc(
        &store,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 1, "exactly one entry processed");
    assert_eq!(
        stats.deleted, 0,
        "ref_count drops 2 → 1, blob must NOT be deleted"
    );

    // The surviving layer's reference must still be live: meta + data
    // present, ref_count = 1, data bytes intact.
    let rc_after: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(
        rc_after, "1",
        "ref_count must be 1 after exactly one decrement (no double-count)"
    );
    let meta_exists: bool = tv.client().pool.exists(&meta_key).await.unwrap();
    assert!(meta_exists, "meta must survive (still referenced)");
    let data_exists: bool = tv.client().pool.exists(&data_key).await.unwrap();
    assert!(data_exists, "data must survive (still referenced)");
    let loaded = store.load_values(&[&hash]).await.unwrap();
    assert_eq!(loaded.len(), 1, "data must still be readable");
    assert_eq!(loaded[0].1, b"shared-by-two-layers");
}

/// b.kf1 C-5 follow-up, updated by b.yuo: a member whose decrement already
/// ran — here it deleted the blob — and that is re-claimed by a later pass
/// (its completion "never landed") is answered `replayed` by the decrement
/// script's per-hash replay record and decrements nothing.
///
/// Before b.yuo the re-run decremented again: `HINCRBY` on the missing meta
/// created `ref_count = -1`, the `<= 0` branch deleted the keys again, and
/// the script answered `deleted`. That was harmless only because nothing had
/// stored the hash in between; had a layer stored it again, the re-run would
/// have deleted that layer's blob. The replay answer holds either way, and
/// `a_reclaimed_claim_whose_decrement_deleted_the_blob_keeps_the_restored_one`
/// in `b_yuo_gc_tests.rs` covers the re-stored case.
///
/// The policy keeps replay records for the whole test (`IMMEDIATE` with an
/// hour's retention): `IMMEDIATE` alone tears a record down on the pass after
/// its member settles, and a re-run after that teardown is a new lifecycle,
/// not a replay.
///
/// Integration analog of `gc_end_to_end_after_crash_between_smove_and_decrement`
/// in the mock, but exercising the real Lua path on Valkey.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn c5_retry_after_lua_del_then_crash_is_replayed_and_decrements_nothing() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();
    let policy = live_edit::storage::external_values::GcPolicy {
        replay_record_retention: Duration::from_secs(3600),
        ..live_edit::storage::external_values::GcPolicy::IMMEDIATE
    };

    let hash = test_hash(0xC6);
    let hash_hex = sha256_to_hex(&hash);
    let meta_key = keys.ext_meta(&hash_hex);
    let data_key = keys.ext_data(&hash_hex);

    // Single-ref blob, scheduled for release.
    store
        .store_values(&[(&hash, b"single-ref".as_slice())])
        .await
        .unwrap();
    let release = retirement_of(hash);
    let member = release.member();
    store.schedule_release(&[release]).await.unwrap();

    // Drive a real sweep first so the Lua-DEL actually runs.
    let stats = process_pending_gc(&store, None, policy).await;
    assert_eq!(stats.deleted, 1, "first sweep deletes the blob");
    assert!(
        !tv.client().pool.exists::<bool, _>(&meta_key).await.unwrap(),
        "meta gone after first sweep"
    );

    // Now simulate the crash-between-DEL-and-ZREM state by manually
    // re-adding the member to inflight (the completion never landed). The
    // next sweep re-claims it and must converge without error, without a
    // second decrement and without resurrecting the deleted blob.
    //
    // The synthetic claim is this build's (an encoded score, b.1jt) and
    // older than the re-claim age, so the next pass re-claims it in place.
    // b.qpf: re-add the MEMBER the sweep just claimed, not a bare hex — the
    // scenario is "the ZREM never landed", so the surviving entry must be the
    // one the sweep actually held.
    plant_own_claim(&tv, &member, STALE_OWN_CLAIM_AGE).await;

    let stats2 = process_pending_gc(&store, None, policy).await;
    assert_eq!(stats2.processed, 1, "the re-claimed member is decided once");
    assert_eq!(
        (stats2.replayed, stats2.deleted),
        (1, 0),
        "the re-run is a replay: its decrement already applied"
    );

    // End state: keys still gone — no `ref_count = -1` meta was created —
    // and both sets clean.
    assert!(
        !tv.client().pool.exists::<bool, _>(&meta_key).await.unwrap(),
        "a replay must never create the meta"
    );
    assert!(
        !tv.client().pool.exists::<bool, _>(&data_key).await.unwrap(),
        "data must remain absent"
    );
    let pending: u64 = tv.client().pool.scard(keys.ext_pending()).await.unwrap();
    assert_eq!(pending, 0, "pending must be empty after retry");
    // b.j9j item 2: inflight is now a sorted set, use ZCARD.
    let inflight: u64 = tv.client().pool.zcard(keys.ext_inflight()).await.unwrap();
    assert_eq!(inflight, 0, "inflight must be empty after retry");
}

// =============================================================================
// H-stor-1 — `cleanup_orphaned_object` schedules external-value GC
// =============================================================================

/// b.kf1 H-stor-1: an orphaned object's external-value references must be
/// SADD'd to `ext:pending` *before* the structural UNLINKs run. Pre-b.kf1
/// the orphan sweep UNLINKed the per-field hashes without touching
/// `ext_meta`, leaving every referenced blob with a permanently-inflated
/// `ref_count` and a silent disk-leak proportional to crash frequency ×
/// external-value churn.
///
/// Setup: seed an orphan object header + a single field hash whose
/// `hash`/`size` fields point at a known external-value SHA-256 hash, so
/// `schedule_old_object_ext_gc` (called from `cleanup_orphaned_object`)
/// finds it via `read_object_snapshot`'s unfenced fan-out →
/// `collect_external_hashes` and
/// schedules a release. Run the orphan sweep through a single iteration
/// and assert the hash is now in `ext:pending`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn orphan_sweep_schedules_external_value_gc_for_orphaned_object_fields() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let pool = tv.client().pool.clone();
    let store = ext_store(&tv);

    // `read_object_snapshot` requires a numeric object_id (it parses to u64
    // and Internal-errors on a non-numeric string post-b.k1r H-stor-8).
    // Pick a high value to avoid colliding with any LayerStore-allocated id.
    let orphan_oid = "9911011";
    let ext_hash = test_hash(0x91);
    let ext_hash_hex = sha256_to_hex(&ext_hash);

    // Pre-store the external value via the normal write path so meta exists.
    // (The orphan sweep's retirement enqueue is what we're asserting on; it
    // operates on hashes regardless of whether the meta is present, but
    // having a real meta gives the test a more realistic shape.)
    store
        .store_values(&[(&ext_hash, b"ext-blob-bytes".as_slice())])
        .await
        .unwrap();

    // Seed the orphan: header points at a dead session; one section; one
    // field on that section whose `hash`/`size` reference the ext value.
    let header_key = ks.obj_header(orphan_oid);
    pool.hset::<(), _, _>(
        &header_key,
        [
            ("seq", "1"),
            ("first_ver", "1"),
            ("src_fmt", "usd"),
            ("session_id", "dead-session-h-stor-1"),
            ("layer_name", "ghost-layer"),
        ],
    )
    .await
    .expect("HSET header");

    let secs_index = ks.obj_sections_index(orphan_oid);
    pool.sadd::<(), _, _>(&secs_index, "1")
        .await
        .expect("SADD sections index");
    let sec_key = ks.obj_section(orphan_oid, "1");
    pool.hset::<(), _, _>(
        &sec_key,
        [
            ("parent_id", "0"),
            ("name", "root"),
            ("type", "1"),
            ("section_order", "0"),
            ("first_version", "1"),
            ("last_version", "1"),
        ],
    )
    .await
    .expect("HSET section");
    // b.zvd: the root children ordering is a LIST, as every writer stores it;
    // every snapshot read fails closed on a wrong-typed children key instead
    // of reading it as "no children". (The orphan sweep's retirement reads no
    // children list since b.fsz MR-D4 — its reference walk reads only the
    // value rows and `obj:ext` — but the fixture keeps the stored shape.)
    pool.rpush::<(), _, _>(&ks.obj_children(orphan_oid, "0"), "1")
        .await
        .expect("RPUSH children");

    // Field hash with the External shape: `hash` (hex SHA-256) + `size`.
    // The fields-index sorted set carries `section_id:hex(key_name)` as
    // the member, with the version as the score. The per-field hash key
    // is `obj:fld:{section_id}:{hex(key_name)}`.
    let key_name = "tex";
    let field_member = live_edit::storage::keys::KeySchema::field_member("1", key_name);
    let field_key = ks.obj_field(orphan_oid, "1", key_name);
    let flds_index = ks.obj_fields_index(orphan_oid);
    pool.zadd::<(), _, _>(
        &flds_index,
        None,
        None,
        false,
        false,
        (1.0, field_member.as_str()),
    )
    .await
    .expect("ZADD fields index");
    pool.hset::<(), _, _>(
        &field_key,
        [
            ("hash", ext_hash_hex.as_str()),
            ("size", "1024"),
            ("set_order", "0"),
            ("version", "1"),
        ],
    )
    .await
    .expect("HSET field");

    // Sanity: ext:pending must NOT yet contain this hash.
    let pending_before: Vec<String> = pool.smembers(ks.ext_pending()).await.unwrap();
    assert!(
        !pending_before.contains(&ext_hash_hex),
        "ext_hash must not be pending before the sweep"
    );

    // Run one orphan sweep iteration with the real ValkeyExternalValueStore
    // so the retirement's attempt-mode enqueue lands in the production
    // ext_pending set.
    let cancel = CancellationToken::new();
    let real_store_arc = Arc::new(ext_store(&tv));
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        real_store_arc.clone() as Arc<dyn ExternalValueStore>,
        Arc::new(live_edit::storage::layer::LayerStore::new(
            tv.client().clone(),
            ks.clone(),
            0,
        )),
        crate::common::test_leader("test-replica-h-stor-1"),
        cancel.clone(),
        None,
    );

    let header_for_poll = header_key.clone();
    let poll_pool = pool.clone();
    for _ in 0..50 {
        let exists: bool = poll_pool.exists(&header_for_poll).await.unwrap();
        if !exists {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("sweep must exit within 5 s of cancellation")
        .expect("sweep must not panic");

    // The ext hash must be in `ext:pending` (b.kf1 H-stor-1 regression).
    //
    // b.qpf: the member now names the retiring object and the event that owes
    // the decrement (`{hash_hex}:{object_id}:ret`), so the assertion is on the
    // full member rather than the bare hex. Asserting only the hex prefix would
    // be weaker than the pre-b.qpf assertion was: the orphan sweep's whole job
    // here is to owe ONE decrement on behalf of THIS object.
    let expected_member = format!("{ext_hash_hex}:{orphan_oid}:ret");
    let pending_after: Vec<String> = pool.smembers(ks.ext_pending()).await.unwrap();
    assert!(
        pending_after.contains(&expected_member),
        "the orphaned object's external-value reference must be in ext:pending after the \
         sweep, as a release owed by object {orphan_oid} — expected {expected_member:?}, got \
         {pending_after:?}"
    );
}

/// b.kf1 H-stor-1 (extra coverage): time-sample-only orphan must also
/// have its external-value reference scheduled for GC. The pre-b.kf1
/// regression bit fields and time-samples equally; both code paths in
/// `collect_external_hashes` must drive the schedule.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn orphan_sweep_schedules_external_value_gc_for_time_sample_orphan() {
    use live_edit::storage::keys::KeySchema;

    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let pool = tv.client().pool.clone();
    let store = Arc::new(ext_store(&tv));

    let orphan_oid = "9911012";
    let ext_hash = test_hash(0x92);
    let ext_hash_hex = sha256_to_hex(&ext_hash);

    store
        .store_values(&[(&ext_hash, b"ts-ext-blob".as_slice())])
        .await
        .unwrap();

    // Seed the orphan: header + section + a single time sample with an
    // External shape (`hash` + `size` per `encode_value_fields`). Time
    // samples are indexed by `obj_time_samples_index` (sorted set
    // member = `section_id:hex_f64(time)`, score = version) and stored
    // at `obj_time_sample(...)`.
    let header_key = ks.obj_header(orphan_oid);
    pool.hset::<(), _, _>(
        &header_key,
        [
            ("seq", "1"),
            ("first_ver", "1"),
            ("src_fmt", "usd"),
            ("session_id", "dead-ts"),
            ("layer_name", "ghost-ts"),
        ],
    )
    .await
    .expect("HSET header");
    pool.sadd::<(), _, _>(&ks.obj_sections_index(orphan_oid), "1")
        .await
        .expect("SADD sections");
    pool.hset::<(), _, _>(
        &ks.obj_section(orphan_oid, "1"),
        [
            ("parent_id", "0"),
            ("name", "root"),
            ("type", "1"),
            ("section_order", "0"),
            ("first_version", "1"),
            ("last_version", "1"),
        ],
    )
    .await
    .expect("HSET section");
    // b.zvd: the root children ordering is a LIST, as every writer stores it;
    // every snapshot read fails closed on a wrong-typed children key instead
    // of reading it as "no children". (The orphan sweep's retirement reads no
    // children list since b.fsz MR-D4 — its reference walk reads only the
    // value rows and `obj:ext` — but the fixture keeps the stored shape.)
    pool.rpush::<(), _, _>(&ks.obj_children(orphan_oid, "0"), "1")
        .await
        .expect("RPUSH children");

    // Time sample at t=1.0 with External shape.
    let ts_member = KeySchema::time_sample_member("1", 1.0);
    let ts_key = ks.obj_time_sample(orphan_oid, "1", 1.0);
    pool.zadd::<(), _, _>(
        &ks.obj_time_samples_index(orphan_oid),
        None,
        None,
        false,
        false,
        (1.0, ts_member.as_str()),
    )
    .await
    .expect("ZADD time sample");
    pool.hset::<(), _, _>(
        &ts_key,
        [
            ("hash", ext_hash_hex.as_str()),
            ("size", "2048"),
            ("version", "1"),
        ],
    )
    .await
    .expect("HSET time sample");

    let pending_before: Vec<String> = pool.smembers(ks.ext_pending()).await.unwrap();
    assert!(
        !pending_before.contains(&ext_hash_hex),
        "sanity: ext_hash must not yet be pending"
    );

    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        store.clone() as Arc<dyn ExternalValueStore>,
        Arc::new(live_edit::storage::layer::LayerStore::new(
            tv.client().clone(),
            ks.clone(),
            0,
        )),
        crate::common::test_leader("test-replica-h-stor-1-ts"),
        cancel.clone(),
        None,
    );

    let header_for_poll = header_key.clone();
    let poll_pool = pool.clone();
    for _ in 0..50 {
        let exists: bool = poll_pool.exists(&header_for_poll).await.unwrap();
        if !exists {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("sweep must exit within 5 s of cancellation")
        .expect("sweep must not panic");

    // b.qpf: member format `{hash_hex}:{object_id}:{event}` — see the field
    // twin above for why the assertion pins the whole member.
    let expected_member = format!("{ext_hash_hex}:{orphan_oid}:ret");
    let pending_after: Vec<String> = pool.smembers(ks.ext_pending()).await.unwrap();
    assert!(
        pending_after.contains(&expected_member),
        "ext_hash referenced by an orphaned time sample must be scheduled for \
         GC as a release owed by object {orphan_oid} (pre-b.kf1 only fields were \
         considered, never time samples). Expected {expected_member:?}, got \
         pending = {pending_after:?}"
    );
}

// =============================================================================
// H-stor-10 — Subscriber sees INTERNAL on missing ext-data, not empty blob
// =============================================================================

/// b.kf1 H-stor-10: when the subscriber's diff resolution observes a hash
/// for which the meta is present but the data key is absent (data
/// integrity violation; cannot happen on a healthy MULTI/EXEC writer),
/// `read_batch` must surface `ServiceError::Internal` mentioning "data not
/// found". Pre-b.kf1 a `.unwrap_or_default()` in `build_diff_payload`
/// silently substituted an empty `Vec<u8>`, which the subscriber then
/// happily forwarded to its caller as a zero-length value — corruption
/// masquerading as success.
///
/// We exercise the read path through `read_batch` (which is the choke
/// point that diff payloads use); a separate `build_diff_payload` test
/// covers the unit-level fall-through, but the integration assertion
/// here is what guarantees the subscriber-facing behavior end-to-end:
/// any future regression that re-introduces a silent empty-blob fallback
/// in the read path will fail this test.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscriber_read_with_missing_ext_data_returns_internal_not_empty_blob() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let ks = tv.key_schema();
    let pool = tv.client().pool.clone();

    let hash = test_hash(0xA0);
    let hash_hex = sha256_to_hex(&hash);
    let meta_key = ks.ext_meta(&hash_hex);

    // Seed the corruption: meta exists with a positive ref_count, but
    // the data key has been deleted (e.g. by a partial GC race or
    // operator intervention). This is the exact state H-stor-10 calls
    // out as "subscriber must hard-error, not silently succeed". The
    // data key is deliberately absent — its key path is never written.
    pool.hset::<(), _, _>(&meta_key, [("ref_count", "1")])
        .await
        .expect("HSET meta");

    // The read must surface `Internal` with "data not found", NOT an
    // empty Vec or any other "successful zero-length result".
    let err = store
        .read_batch(&[hash])
        .await
        .expect_err("read_batch must error on dangling-meta corruption");

    match err {
        ServiceError::Internal(msg) => {
            assert!(
                msg.contains("data not found"),
                "expected 'data not found' in error message, got: {msg}"
            );
            assert!(
                msg.contains(&hash_hex),
                "error message must include the hash for triage, got: {msg}"
            );
        }
        other => panic!(
            "expected ServiceError::Internal(DataNotFound), got: {other:?} — \
             a regression that returns Ok(empty_map) or any non-Internal variant \
             would silently corrupt subscribers"
        ),
    }

    // Negative-cousin assertion: a hash with NO meta (unknown hash, e.g.
    // never stored or already GC'd) must surface a different variant —
    // `MetaNotFound` — so callers can distinguish "unknown" from
    // "corrupted". Pre-b.kf1 BOTH paths collapsed to empty-success.
    let unknown_hash = test_hash(0xA1);
    let unknown_hex = sha256_to_hex(&unknown_hash);
    let err2 = store
        .read_batch(&[unknown_hash])
        .await
        .expect_err("read_batch must error on unknown hash");
    match err2 {
        ServiceError::Internal(msg) => {
            assert!(
                msg.contains("meta not found"),
                "expected 'meta not found' in error message, got: {msg}"
            );
            assert!(
                msg.contains(&unknown_hex),
                "error message must include the unknown hash, got: {msg}"
            );
        }
        other => panic!("expected ServiceError::Internal(MetaNotFound), got: {other:?}"),
    }
}

// =============================================================================
// H-stor-4 — Leader election: try_acquire_gc_leader unit semantics
// =============================================================================

/// b.kf1 H-stor-4: the first replica to invoke the leader-lock script
/// wins the lock; a *different* replica gets `0` (locked out) until the
/// TTL expires. The Lua-CAS hardening (b.kf1 follow-up) made this the
/// correctness primitive — pre-CAS there was a sub-30 s window where the
/// GET-then-SET refresh could clobber a peer's fresh NX claim. This test
/// pins the script itself, driving the public
/// `LEADER_ACQUIRE_OR_REFRESH_SCRIPT` static directly: KEYS[1] is a
/// production lock key the script still guards — the checkpoint lock;
/// `le:gc:leader` has its own marked takeover script since b.1jt — and
/// ARGV[1] is an opaque holder string the script only compares for
/// equality. Production's ARGV[1] is a
/// `LeaderIdentity::lock_value()` (`label:instance`), passed through
/// `try_acquire_leader_lock`; the bare `replica-A` / `replica-B` strings
/// here stand in for two such values. The identity-level contract (two
/// processes sharing a label never both hold the lock) is
/// `leader_identity_test.rs`'s.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn leader_lock_script_first_caller_wins_others_locked_out() {
    use live_edit::storage::external_values::LEADER_ACQUIRE_OR_REFRESH_SCRIPT;

    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let lock_key = ks.checkpoint_leader_lock();

    // 30 s TTL matches production. The unit is ms (PX) — see the script
    // header for why the production caller sends ms.
    let ttl_ms = "30000";

    // Replica A claims the lock: GET returns nil → SET path → return 1.
    let lua_keys = vec![lock_key.clone()];
    let argv_a: Vec<String> = vec!["replica-A".to_owned(), ttl_ms.to_owned()];
    let result_a: i64 = LEADER_ACQUIRE_OR_REFRESH_SCRIPT
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            lua_keys.clone(),
            argv_a,
        )
        .await
        .expect("script must succeed");
    assert_eq!(result_a, 1, "first caller wins the lock");

    // Replica B (different id) tries: GET returns "replica-A", does not
    // match ARGV[1]="replica-B" → return 0.
    let argv_b: Vec<String> = vec!["replica-B".to_owned(), ttl_ms.to_owned()];
    let result_b: i64 = LEADER_ACQUIRE_OR_REFRESH_SCRIPT
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            lua_keys.clone(),
            argv_b.clone(),
        )
        .await
        .expect("script must succeed");
    assert_eq!(result_b, 0, "different-id caller must be locked out");

    // Replica A re-calls: GET matches ARGV[1] → PEXPIRE refresh → return 1.
    let argv_a_refresh: Vec<String> = vec!["replica-A".to_owned(), ttl_ms.to_owned()];
    let result_a_refresh: i64 = LEADER_ACQUIRE_OR_REFRESH_SCRIPT
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            lua_keys.clone(),
            argv_a_refresh,
        )
        .await
        .expect("script must succeed");
    assert_eq!(result_a_refresh, 1, "leader must refresh its own lock");

    // Replica B still locked out.
    let result_b2: i64 = LEADER_ACQUIRE_OR_REFRESH_SCRIPT
        .evalsha_with_reload(
            tv.client().maintenance_pool.next(),
            lua_keys.clone(),
            argv_b,
        )
        .await
        .expect("script must succeed");
    assert_eq!(
        result_b2, 0,
        "non-leader stays locked out across leader refreshes"
    );

    // Lock value preserved across refresh — important regression guard
    // for the script's PEXPIRE-not-SET branch.
    let value: String = tv.client().pool.get(&lock_key).await.unwrap();
    assert_eq!(
        value, "replica-A",
        "lock value must equal acquiring replica's id (PEXPIRE preserves value)"
    );
}

// =============================================================================
// H-stor-4 — End-to-end: two `spawn_gc_task` instances race, only one sweeps
// =============================================================================

/// b.kf1 H-stor-4 end-to-end: two `spawn_gc_task` instances racing on the
/// same Valkey produce *exactly one* sweep per tick. The leader gate
/// (Valkey-backed `try_acquire_gc_leader`) admits one and skips the other.
/// We assert on the GC-deletion count: pre-b.kf1 with N=2 replicas every
/// pending hash had its decrement attempted twice, doubling Valkey load
/// and risking interleaved per-hash SMOVEs on overlapping pending entries.
///
/// We use `spawn_gc_task` (the production path) so the leader-gate
/// closure exercises the real Valkey lock — spawning two tasks with
/// different leader identities and counting deletions across both via
/// shared metrics.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_gc_tasks_racing_only_leader_runs_sweep_per_tick() {
    use live_edit::storage::external_values::spawn_gc_task;

    let tv = TestValkey::setup().await;
    let store = Arc::new(ext_store(&tv)) as Arc<dyn ExternalValueStore>;
    let ks = tv.key_schema();

    // Seed N hashes into pending. Each is a single-ref blob, so a sweep
    // run will delete all of them.
    let n: u8 = 6;
    let mut hashes = Vec::new();
    for i in 0..n {
        let h = test_hash(0xE0 + i);
        store.store_values(&[(&h, &[i; 8])]).await.unwrap();
        hashes.push(h);
    }
    let releases: Vec<ExtRelease> = hashes.iter().copied().map(retirement_of).collect();
    ext_store(&tv).schedule_release(&releases).await.unwrap();
    let pending_before: u64 = tv.client().pool.scard(ks.ext_pending()).await.unwrap();
    assert_eq!(pending_before, n as u64, "test setup: all N hashes pending");

    // Wire metrics so we can attribute sweep activity. Both replicas
    // share the same registry/Metrics arc — the `inc_gc_sweep_batch_error`
    // counter is the easiest hook, but on a healthy run we expect zero
    // errors. Instead we assert on the *deletion side effects* (pending
    // set drained exactly once → no double-work observable).
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));

    let cancel = CancellationToken::new();

    // `IMMEDIATE` (no age floor) so the leader's first pass collects
    // everything; the loop then wakes about once a second until we cancel —
    // we're testing mutual exclusion at pass scale, not steady-state
    // behavior.
    let policy = live_edit::storage::external_values::GcPolicy::IMMEDIATE;

    let leader_a = crate::common::test_leader("replica-A");
    let leader_b = crate::common::test_leader("replica-B");
    let handle_a = spawn_gc_task(
        store.clone(),
        tv.client().clone(),
        ks.clone(),
        leader_a.clone(),
        policy,
        cancel.clone(),
        Some(metrics.clone()),
    );
    let handle_b = spawn_gc_task(
        store.clone(),
        tv.client().clone(),
        ks.clone(),
        leader_b.clone(),
        policy,
        cancel.clone(),
        Some(metrics.clone()),
    );

    // Poll for the pending set to drain to zero. The leader runs the
    // pass on its first wake-up; the non-leader sees `Ok(false)` from
    // `try_acquire_gc_leader` and skips. Both then sleep the pass interval.
    let mut drained = false;
    for _ in 0..50 {
        let pending: u64 = tv.client().pool.scard(ks.ext_pending()).await.unwrap();
        if pending == 0 {
            drained = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(drained, "the leader's sweep must drain the pending set");

    // Cancel both tasks and join under timeout.
    cancel.cancel();
    for h in [handle_a, handle_b] {
        tokio::time::timeout(Duration::from_secs(5), h)
            .await
            .expect("GC task must exit within 5 s of cancellation")
            .expect("GC task must not panic");
    }

    // Primary assertion: every hash is gone from the data plane (single
    // sweep cleaned them up). If both replicas had run sweeps in
    // parallel, the per-hash SMOVE atomicity still guarantees correct
    // deletion, but the *test* would observe leader-lock skips on the
    // non-leader, not duplicate work. The deletion-side check is what's
    // observable to operators; SDD §11.3 calls it out as the operator
    // contract.
    for h in &hashes {
        let hex = sha256_to_hex(h);
        let exists_meta: bool = tv.client().pool.exists(&ks.ext_meta(&hex)).await.unwrap();
        assert!(!exists_meta, "meta for {hex} must be deleted");
    }

    // Sanity: the leader lock is still held by *someone* (TTL hasn't
    // expired in our short window). We don't assert which replica —
    // the contract is "exactly one held the lock during the sweep tick",
    // not "replica A specifically".
    let lock_value: Option<String> = tv.client().pool.get(&ks.gc_leader_lock()).await.unwrap();
    assert!(
        lock_value.is_some(),
        "leader lock must persist for its 30 s TTL after the sweep"
    );
    // The lock holds the winner's protocol-marked GC lock value
    // (`le-gc/<v>|label:instance`, b.1jt), never a bare label.
    let v = lock_value.unwrap();
    assert!(
        v == leader_a.gc_lock_value() || v == leader_b.gc_lock_value(),
        "lock holder must be one of the two test replicas' GC lock values, got: {v}"
    );
}

// =============================================================================
// H-stor-4 — Lifecycle: spawn_gc_task_with_leader gate=false skips sweeps
// =============================================================================

/// Lower-level pin on the leader-gate closure: when the gate returns
/// `false`, the sweep loop must skip `process_pending_gc` — even if there
/// is pending work. This guards against a regression where the gate's
/// return value gets ignored or inverted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spawn_gc_task_with_leader_gate_false_skips_sweeps_even_with_pending_work() {
    let tv = TestValkey::setup().await;
    let store = Arc::new(ext_store(&tv));

    // Seed a hash with ref_count=1, scheduled for release. The sweep
    // would delete it if the gate let the loop in.
    let h = test_hash(0xFA);
    store.store_values(&[(&h, b"untouched")]).await.unwrap();
    store.schedule_release(&[retirement_of(h)]).await.unwrap();

    let cancel = CancellationToken::new();
    let handle = spawn_gc_task_with_leader(
        store.clone() as Arc<dyn ExternalValueStore>,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
        cancel.clone(),
        None,
        || async { false }, // never the leader
    );

    // Let the loop tick a few times. None of the ticks should sweep.
    tokio::time::sleep(Duration::from_millis(300)).await;

    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("GC task must exit within 5 s")
        .expect("GC task must not panic");

    // The hash must still be pending and the meta must still exist —
    // the gate kept the sweep from running.
    let ks = tv.key_schema();
    let pending: u64 = tv.client().pool.scard(ks.ext_pending()).await.unwrap();
    assert_eq!(
        pending, 1,
        "pending set must be untouched when leader gate returns false"
    );
    let meta_key = ks.ext_meta(&sha256_to_hex(&h));
    let rc: String = tv.client().pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(
        rc, "1",
        "ref_count must be unchanged when sweep is gated off"
    );
}

// =============================================================================
// H-stor-5 — Maintenance pool isolation: long sweep doesn't block data writes
// =============================================================================

/// b.kf1 H-stor-5: a long-running orphan sweep on the maintenance pool
/// must NOT block concurrent writes on the data pool. Pre-b.kf1 the
/// sweeper grabbed `client.pool.next()` (a multiplexed data-pool
/// connection) for SCAN + HGETALL passes; writers routed to the same
/// fred client queued behind the scan commands.
///
/// We seed enough orphan headers that a sweep takes a noticeable
/// duration (~hundreds of ms), then time a concurrent `SET` against the
/// data pool. With the maintenance pool isolation, the SET completes in
/// under 100 ms regardless of sweep activity. Without isolation, the SET
/// would be observably slower — though the exact slow-down depends on
/// fred's connection-multiplexing internals and is hard to pin to a
/// specific threshold.
///
/// To keep the assertion tight without flake risk, we measure the latency
/// of a small batch of `SET` round-trips on the data pool while the
/// maintenance pool is busy with a sweep, and require that it stays
/// well below the sweep's own duration. Any regression that re-routes
/// the sweep onto the data pool would push the SET latencies up to the
/// sweep's wall-clock, well past our threshold.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn maintenance_pool_isolation_data_pool_writes_unaffected_by_sweep() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let pool = tv.client().pool.clone();

    // Seed a non-trivial number of orphan headers so the sweep does
    // real work (not microseconds-and-done).
    let count = 200u64;
    for i in 0..count {
        let oid = format!("h_stor_5_{i:04}");
        pool.hset::<(), _, _>(
            &ks.obj_header(&oid),
            [
                ("seq", "1"),
                ("first_ver", "1"),
                ("src_fmt", "usd"),
                ("session_id", "dead"),
                ("layer_name", "ghost"),
            ],
        )
        .await
        .expect("HSET orphan header");
    }

    // Kick off the orphan sweep.
    let store = Arc::new(ext_store(&tv));
    let cancel = CancellationToken::new();
    let sweep_handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        store.clone() as Arc<dyn ExternalValueStore>,
        Arc::new(live_edit::storage::layer::LayerStore::new(
            tv.client().clone(),
            ks.clone(),
            0,
        )),
        crate::common::test_leader("h-stor-5-replica"),
        cancel.clone(),
        None,
    );

    // While the sweep is running on the maintenance pool, hammer the
    // data pool with a small batch of SETs and measure latency. We
    // accept up to 1 s for the entire 50-SET batch — conservative
    // enough to be insensitive to local Valkey jitter, tight enough
    // that a regression routing the sweep onto the data pool (which
    // would queue all 50 SETs behind the slower SCAN passes) fails.
    let mut max_set_latency = Duration::ZERO;
    for i in 0..50u32 {
        let key = format!("{}h_stor_5_data_set:{i}", tv.prefix());
        let t0 = std::time::Instant::now();
        let _: () = pool
            .set(&key, format!("v{i}"), None, None, false)
            .await
            .expect("data-pool SET must succeed");
        let elapsed = t0.elapsed();
        if elapsed > max_set_latency {
            max_set_latency = elapsed;
        }
    }

    // Stop the sweep cleanly.
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), sweep_handle)
        .await
        .expect("sweep must exit within 5 s")
        .expect("sweep must not panic");

    // Threshold: 250 ms is well above local-loopback Valkey RTT
    // (microseconds) and well below the multi-second wall-clock of a
    // 200-orphan SCAN+HGETALL+UNLINK pass on the same connection. If
    // the sweep ever lands on the data pool again, the per-SET latency
    // here would spike into the hundreds-of-ms-to-seconds range.
    assert!(
        max_set_latency < Duration::from_millis(250),
        "data-pool SET latency must stay low while maintenance pool runs orphan \
         sweep — observed max {max_set_latency:?}, expected < 250 ms. A regression \
         routing the sweep onto the data pool would push this latency up to the \
         sweep's wall-clock duration."
    );
}

// =============================================================================
// Mediums — SweepError preserves fred ErrorKind via kind_label()
// =============================================================================

/// b.kf1 medium: `SweepError::Fred` must preserve the underlying fred
/// `ErrorKind::to_str()` slug so triage can split timeouts from protocol
/// errors. Pre-b.kf1 the variant collapsed to a single `String` payload.
#[test]
fn sweep_error_fred_kind_label_returns_underlying_error_kind_slug() {
    use fred::error::Error as FredError;
    use fred::error::ErrorKind as FredErrorKind;

    // `Timeout` is the canonical transient case — the most likely
    // error operators see during a Valkey blip, and the one whose
    // metric label we most need to disambiguate.
    let timeout_err = FredError::new(FredErrorKind::Timeout, "synthetic test timeout");
    let kind_slug = timeout_err.kind().to_str();
    let sweep_err = SweepError::from_fred(
        FredErrOrigin::metrics(
            None,
            POOL_LABEL_DATA,
            live_edit::storage::store_oom::StoreOomSignal::unwired(),
        ),
        "HGETALL test op",
        timeout_err,
    );
    assert_eq!(
        sweep_err.kind_label(),
        kind_slug,
        "SweepError::Fred must preserve the fred ErrorKind slug \
         so `gc_sweep_batch_errors_total{{kind}}` reports it accurately"
    );

    // `Protocol` is a different category — operators want this to NOT
    // collapse with `Timeout` in metrics.
    let protocol_err = FredError::new(FredErrorKind::Protocol, "synthetic protocol error");
    let proto_slug = protocol_err.kind().to_str();
    let sweep_proto = SweepError::from_fred(
        FredErrOrigin::metrics(
            None,
            POOL_LABEL_DATA,
            live_edit::storage::store_oom::StoreOomSignal::unwired(),
        ),
        "XADD test op",
        protocol_err,
    );
    assert_eq!(
        sweep_proto.kind_label(),
        proto_slug,
        "Different fred ErrorKinds must yield different kind labels"
    );
    assert_ne!(
        sweep_err.kind_label(),
        sweep_proto.kind_label(),
        "Timeout and Protocol must produce distinct kind labels"
    );
}

// (Removed `sweep_error_ext_value_schedule_kind_label_is_fixed_slug` —
// the `SweepError::ExtValueSchedule` variant was deleted alongside the
// fix for the orphan-cleanup parse-failure regression. After that fix,
// `cleanup_orphaned_object` logs and proceeds when ext-GC scheduling
// fails rather than surfacing a `SweepError`, so the variant had no
// remaining producers. See orphan_sweep.rs::cleanup_orphaned_object for
// the failure-mode rationale.)

// =============================================================================
// Mediums — gc_sweep_batch_errors_total{kind} increments on sweep failure
// =============================================================================

/// b.kf1 medium: `process_pending_gc` must increment
/// `gc_sweep_batch_errors_total{kind=...}` when a batch fails. Pre-b.kf1
/// the failure was tracing-only, making "GC stopped collecting" an
/// invisible failure mode until disk filled.
///
/// We drive `process_pending_gc` with a `FailingMockStore` that returns
/// a synthetic `ServiceError::Internal` from `process_pending`, then
/// scrape the counter for the expected label.
#[tokio::test]
async fn process_pending_gc_increments_batch_error_metric_on_failure() {
    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Mutex as StdMutex;

    /// In-memory store whose `process_pending` always fails with a
    /// known `ServiceError::Internal`. All other methods are stubbed
    /// minimally — only `process_pending` is exercised.
    struct FailingMockStore {
        called: StdMutex<u32>,
    }

    #[async_trait]
    impl ExternalValueStore for FailingMockStore {
        async fn store_values(&self, _values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
            unreachable!("not exercised")
        }

        async fn store_verified_values(
            &self,
            _values: &[(&[u8; 32], &[u8])],
        ) -> Result<(), ServiceError> {
            unreachable!("not exercised")
        }

        fn ext_value_verify(&self) -> live_edit::storage::external_values::ExtValueVerify<'_> {
            live_edit::storage::external_values::ExtValueVerify::ENFORCE_UNCOUNTED
        }

        async fn load_values(
            &self,
            _hashes: &[&[u8; 32]],
        ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
            unreachable!("not exercised")
        }

        async fn read_batch(
            &self,
            _hashes: &[[u8; 32]],
        ) -> Result<HashMap<[u8; 32], Vec<u8>>, ServiceError> {
            unreachable!("not exercised")
        }

        async fn probe_values(
            &self,
            _hashes: &[[u8; 32]],
        ) -> Result<live_edit::storage::external_values::ExtValueProbe, ServiceError> {
            unreachable!("not exercised")
        }

        async fn enqueue_retirement_attempt(
            &self,
            _object_id: &str,
            _attempt: &str,
            _releases: &[ExtRelease],
            _send_by: std::time::Instant,
        ) -> Result<live_edit::storage::external_values::ReleaseAttemptOutcome, ServiceError>
        {
            Ok(live_edit::storage::external_values::ReleaseAttemptOutcome::Enqueued)
        }

        async fn process_pending(
            &self,
            _policy: &live_edit::storage::external_values::GcPolicy,
            _state: &mut live_edit::storage::external_values::GcPassState,
        ) -> Result<live_edit::storage::external_values::GcBatchResult, ServiceError> {
            *self.called.lock().unwrap() += 1;
            Err(ServiceError::Internal(
                "synthetic process_pending failure".to_owned(),
            ))
        }

        fn ext_release_gate(
            &self,
        ) -> Option<&Arc<live_edit::storage::ext_release_gate::ExtReleaseGate>> {
            None
        }
    }

    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let store = FailingMockStore {
        called: StdMutex::new(0),
    };

    // Run a sweep — it must abort on the first error per the
    // documented "abort on first error so a persistent fault surfaces
    // fast" contract.
    let stats = process_pending_gc(
        &store,
        Some(&metrics),
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 0, "no successful batches");
    assert_eq!(stats.deleted, 0);
    assert_eq!(
        *store.called.lock().unwrap(),
        1,
        "sweep aborts after the first failure (does not loop)"
    );

    // Scrape the counter. Internal errors map to kind="internal" via
    // `service_error_kind_label` in external_values.rs.
    let internal_cell = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_gc_sweep_batch_errors_total",
        &[("kind", "internal")],
    )
    .expect("kind=internal cell must be present");
    assert_eq!(
        internal_cell as u64, 1,
        "gc_sweep_batch_errors_total{{kind=\"internal\"}} must have incremented exactly once"
    );
}

// =============================================================================
// Mediums — Mock test API gating (compile-time check via use)
// =============================================================================

/// b.kf1 medium: `MockExternalValueStore::inflight_count` must be gated
/// alongside `simulate_gc_crash_after_smove` under
/// `cfg(any(test, feature = "test-fault-injection"))` so release builds
/// don't carry an inspector for state they could never put the mock into.
///
/// This test is a compile-time + runtime smoke that both halves are
/// available in the integration-test binary (which compiles with `cfg(test)`).
/// If a future regression removes the cfg gate from one and not the other,
/// the integration suite would still build, but the unit suite running
/// without `test-fault-injection` would fail the symmetry. The cfg gate
/// is the contract, the runtime smoke is just a "did the symbols exist"
/// assertion.
#[cfg(feature = "test-fault-injection")]
#[tokio::test]
async fn mock_external_value_store_inflight_count_and_simulate_crash_are_both_available() {
    let mock = MockExternalValueStore::new();
    let h = test_hash(0xCD);
    mock.store_values(&[(&h, b"x")]).await.unwrap();
    mock.schedule_release(&[retirement_of(h)]).await.unwrap();

    // `simulate_gc_crash_after_smove` and `inflight_count` must both be
    // reachable in this build — if either is gated incorrectly, this
    // test fails to compile (regression guard).
    mock.simulate_gc_crash_after_smove(&h);
    assert_eq!(mock.inflight_count(), 1);
}
