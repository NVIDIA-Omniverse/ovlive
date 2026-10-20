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

// Integration tests for the orphan sweep (SDD section 5.5 step 5).
//
// Tests run against a real Valkey instance via the TestValkey fixture. Each test
// gets an isolated key namespace so tests can run in parallel safely.
//
// Requires a running Valkey. Address is read from TEST_VALKEY_URL, defaulting to
// redis://127.0.0.1:6379.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, ListInterface, SetsInterface, SortedSetsInterface,
};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use live_edit::domain::types::ObjectState;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::{self, KeySchema};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::orphan_sweep::spawn_orphan_sweep;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

/// Drive the now-periodic `spawn_orphan_sweep` task through exactly one
/// iteration (b.kf1 H-stor-4 made the orphan sweep periodic with a 300 s
/// interval — pre-b.kf1 it ran once at startup and exited). Tests still want
/// "did the first sweep clean what I seeded?" semantics, so this helper waits
/// up to ~5 seconds for the predicate to flip true (the actual sweep work is
/// milliseconds), then cancels the task and joins it under a timeout. If the
/// predicate never flips, the helper still cancels + joins the task — callers
/// then assert the failure mode they care about.
async fn run_one_sweep_iteration<F, Fut>(
    handle: JoinHandle<()>,
    cancel: CancellationToken,
    mut done: F,
) where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    // Iteration 1 needs to acquire the leader lock + run the actual sweep.
    // Both round-trips together complete in tens of ms against a local
    // Valkey; 5 s is generous and bounds runaway tests.
    for _ in 0..50 {
        if done().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("orphan sweep must exit within 5 s of cancellation")
        .expect("orphan sweep must not panic");
}

/// Specialized helper for tests that seed many orphans across many keys:
/// poll until *all* of `keys` are gone (or timeout), then cancel + join.
/// More robust than polling on a single "last" key since SCAN ordering is
/// not predictable — if you poll on a key that the sweep happens to
/// process first, the cancel fires before later orphans are reached, and
/// the test fails with a misleading "header should be deleted" message.
async fn run_sweep_until_all_keys_deleted(
    handle: JoinHandle<()>,
    cancel: CancellationToken,
    pool: fred::clients::Pool,
    keys: Vec<String>,
) {
    // 100 × 100 ms = 10 s budget. A 200-orphan SCAN+UNLINK pass against
    // a local Valkey completes in well under a second; this is generous
    // enough that the budget itself never fires in practice but tight
    // enough that a regression (sweep stuck or the "skip ext-GC" guard
    // re-broken) surfaces as a hard failure rather than a hang.
    for _ in 0..100 {
        let mut all_gone = true;
        for k in &keys {
            let exists: bool = pool.exists::<bool, _>(k).await.unwrap_or(true);
            if exists {
                all_gone = false;
                break;
            }
        }
        if all_gone {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("orphan sweep must exit within 5 s of cancellation")
        .expect("orphan sweep must not panic");
}

/// Variant of [`run_one_sweep_iteration`] that does not poll for an outcome —
/// callers that want to assert "no orphan, no work, sweep exits cleanly" use
/// this so the test still validates clean shutdown without a flaky poll.
async fn cancel_and_join_sweep(handle: JoinHandle<()>, cancel: CancellationToken) {
    // Give the first iteration ~500 ms to run + reach the select. On an
    // empty keyspace there is nothing to scan, so this is essentially the
    // round-trip of the leader-lock SET plus the two SCAN cursors.
    tokio::time::sleep(Duration::from_millis(500)).await;
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("orphan sweep must exit within 5 s of cancellation")
        .expect("orphan sweep must not panic");
}

fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
}

fn session_store(tv: &TestValkey) -> SessionStore {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
}

async fn create_test_session(tv: &TestValkey) -> String {
    session_store(tv)
        .create("test-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed")
}

// =============================================================================
// 1. Orphaned objects -- obj:hdr with no layer mapping
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_cleans_orphaned_object_with_no_layer_mapping() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let pool = &tv.client().pool;

    // Manually create an orphaned object (header exists, no layer mapping).
    let orphan_oid = "999";
    let header_key = ks.obj_header(orphan_oid);
    pool.hset::<(), _, _>(
        &header_key,
        [
            ("seq", "1"),
            ("first_ver", "1"),
            ("src_fmt", "usd"),
            ("session_id", "dead-session"),
            ("layer_name", "ghost-layer"),
        ],
    )
    .await
    .expect("HSET header");

    // Also create a sections index and a section for this object.
    let secs_index = ks.obj_sections_index(orphan_oid);
    pool.sadd::<(), _, _>(&secs_index, "7")
        .await
        .expect("SADD sections index");
    let sec_key = ks.obj_section(orphan_oid, "7");
    pool.hset::<(), _, _>(
        &sec_key,
        [("parent_id", "0"), ("name", "root"), ("type", "1")],
    )
    .await
    .expect("HSET section");

    // Verify the keys exist.
    assert!(
        pool.exists::<bool, _>(&header_key).await.unwrap(),
        "header should exist before sweep"
    );
    assert!(
        pool.exists::<bool, _>(&sec_key).await.unwrap(),
        "section should exist before sweep"
    );

    // Run the sweep.
    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    let header_for_poll = header_key.clone();
    let poll_pool = tv.client().pool.clone();
    run_one_sweep_iteration(handle, cancel, move || {
        let key = header_for_poll.clone();
        let pool = poll_pool.clone();
        async move { !pool.exists::<bool, _>(&key).await.unwrap() }
    })
    .await;

    // Verify the orphaned keys are gone.
    assert!(
        !pool.exists::<bool, _>(&header_key).await.unwrap(),
        "orphaned header should be deleted"
    );
    assert!(
        !pool.exists::<bool, _>(&sec_key).await.unwrap(),
        "orphaned section should be deleted"
    );
    assert!(
        !pool.exists::<bool, _>(&secs_index).await.unwrap(),
        "orphaned sections index should be deleted"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_does_not_delete_valid_object_with_layer_mapping() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    // Create a real layer+object via LayerStore.
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };

    let oid = ls
        .create_layer(
            &session_id,
            "valid-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let header_key = ks.obj_header(&oid.to_string());
    assert!(
        tv.client()
            .pool
            .exists::<bool, _>(&header_key)
            .await
            .unwrap(),
        "header should exist before sweep"
    );

    // A zero grace, so it is the mapping, not the object's youth, that keeps
    // the valid object.
    tv.client().set_orphan_sweep_grace_for_test(Duration::ZERO);

    // Run the sweep.
    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    // Negative-case test: nothing should be deleted, so just give the sweep
    // time to run and exit cleanly.
    cancel_and_join_sweep(handle, cancel).await;

    // The valid object should still exist.
    assert!(
        tv.client()
            .pool
            .exists::<bool, _>(&header_key)
            .await
            .unwrap(),
        "valid object header must survive sweep"
    );
}

// =============================================================================
// 2. Orphaned layer mappings -- layer key for non-existent session
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_cleans_orphaned_layer_mapping_for_dead_session() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let pool = &tv.client().pool;

    // Manually create a layer mapping pointing to a non-existent session.
    let dead_session = "dead-sess-id";
    let layer_name = "orphaned-layer";
    let mapping_key = ks.layer_mapping(dead_session, layer_name);
    let layers_index_key = ks.layers_index(dead_session);
    let hex_layer = keys::hex_encode(layer_name);

    pool.set::<(), _, _>(&mapping_key, "42", None, None, false)
        .await
        .expect("SET mapping");
    pool.sadd::<(), _, _>(&layers_index_key, &hex_layer)
        .await
        .expect("SADD layers index");

    // Verify the keys exist.
    assert!(
        pool.exists::<bool, _>(&mapping_key).await.unwrap(),
        "mapping should exist before sweep"
    );

    // Run the sweep.
    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    let mapping_for_poll = mapping_key.clone();
    let poll_pool = tv.client().pool.clone();
    run_one_sweep_iteration(handle, cancel, move || {
        let key = mapping_for_poll.clone();
        let pool = poll_pool.clone();
        async move { !pool.exists::<bool, _>(&key).await.unwrap() }
    })
    .await;

    // Verify the orphaned mapping is gone.
    assert!(
        !pool.exists::<bool, _>(&mapping_key).await.unwrap(),
        "orphaned layer mapping should be deleted"
    );
    assert!(
        !pool.exists::<bool, _>(&layers_index_key).await.unwrap(),
        "orphaned layers index should be deleted when empty"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_does_not_delete_layer_mapping_for_live_session() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    // Create a real layer.
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };

    ls.create_layer(&session_id, "live-layer", "bob", &state, mock_ext_store())
        .await
        .expect("create_layer must succeed");

    let mapping_key = ks.layer_mapping(&session_id, "live-layer");
    assert!(
        tv.client()
            .pool
            .exists::<bool, _>(&mapping_key)
            .await
            .unwrap(),
        "mapping should exist before sweep"
    );

    // Run the sweep.
    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    cancel_and_join_sweep(handle, cancel).await;

    // The valid mapping should still exist.
    assert!(
        tv.client()
            .pool
            .exists::<bool, _>(&mapping_key)
            .await
            .unwrap(),
        "live layer mapping must survive sweep"
    );
}

// =============================================================================
// 3. Overwrite scenario -- old object ID is orphaned after overwrite
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_cleans_orphaned_object_after_simulated_overwrite_crash() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let ls = layer_store(&tv);
    let pool = &tv.client().pool;
    let session_id = create_test_session(&tv).await;

    // Create initial layer.
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };

    let old_oid = ls
        .create_layer(
            &session_id,
            "overwrite-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // Simulate an overwrite: create a new object and update the mapping,
    // but "crash" before cleaning up the old object.
    let new_oid_str = "99999";
    let new_header_key = ks.obj_header(new_oid_str);
    pool.hset::<(), _, _>(
        &new_header_key,
        [
            ("seq", "2"),
            ("first_ver", "1"),
            ("src_fmt", "usd"),
            ("session_id", session_id.as_str()),
            ("layer_name", "overwrite-layer"),
        ],
    )
    .await
    .expect("HSET new header");

    // Update the mapping to point to the new object.
    let mapping_key = ks.layer_mapping(&session_id, "overwrite-layer");
    pool.set::<(), _, _>(&mapping_key, new_oid_str, None, None, false)
        .await
        .expect("SET mapping to new oid");

    // The old object's header still exists (simulating crash before cleanup).
    let old_header = ks.obj_header(&old_oid.to_string());
    assert!(
        pool.exists::<bool, _>(&old_header).await.unwrap(),
        "old header should still exist (simulated crash)"
    );

    // Wait a moment for any fire-and-forget cleanup to complete.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // The old object was created a moment ago: without a zero grace the sweep
    // would leave it to the CreateLayer that might still be committing it.
    tv.client().set_orphan_sweep_grace_for_test(Duration::ZERO);

    // Run the sweep.
    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    let old_for_poll = old_header.clone();
    let poll_pool = tv.client().pool.clone();
    run_one_sweep_iteration(handle, cancel, move || {
        let key = old_for_poll.clone();
        let pool = poll_pool.clone();
        async move { !pool.exists::<bool, _>(&key).await.unwrap() }
    })
    .await;

    // The old object should be cleaned up (it's orphaned — mapping now points elsewhere).
    assert!(
        !pool.exists::<bool, _>(&old_header).await.unwrap(),
        "old orphaned object header should be deleted by sweep"
    );

    // The new object should still exist.
    assert!(
        pool.exists::<bool, _>(&new_header_key).await.unwrap(),
        "new object header must survive sweep"
    );
}

// =============================================================================
// 4. Empty sweep -- no orphans found
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_on_empty_valkey_completes_without_error() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();

    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    cancel_and_join_sweep(handle, cancel).await;
}

// =============================================================================
// 5. Cancellation -- sweep respects cancellation token
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_stops_when_cancelled() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();

    // Cancel immediately.
    let cancel = CancellationToken::new();
    cancel.cancel();

    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel,
        None,
    );
    // Should complete promptly without error.
    tokio::time::timeout(std::time::Duration::from_secs(5), handle)
        .await
        .expect("sweep should stop promptly when cancelled")
        .expect("sweep must not panic");
}

// =============================================================================
// 6. Full sub-key cleanup -- fields, time samples, deleted sections, children
//    (root b.9fn + dangling-parent b.jdr), the `obj:chs` parent-id index, the
//    b.75i commit-token record `obj:wtok`, and the snapshot fence and
//    writer-intent keys `obj:snapfence` / `obj:snapintent`
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_cleans_all_sub_keys_of_orphaned_object() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let pool = &tv.client().pool;
    let prefix = ks.prefix();

    let orphan_oid = "777";

    // Header (orphaned: points to nonexistent session/layer).
    let header_key = ks.obj_header(orphan_oid);
    pool.hset::<(), _, _>(
        &header_key,
        [
            ("seq", "1"),
            ("first_ver", "1"),
            ("src_fmt", "usd"),
            ("session_id", "gone-session"),
            ("layer_name", "gone-layer"),
        ],
    )
    .await
    .expect("HSET header");

    // Sections index + section + children.
    let secs_index = ks.obj_sections_index(orphan_oid);
    pool.sadd::<(), _, _>(&secs_index, "10")
        .await
        .expect("SADD sections index");
    let sec_key = ks.obj_section(orphan_oid, "10");
    pool.hset::<(), _, _>(&sec_key, [("name", "root"), ("type", "1")])
        .await
        .expect("HSET section");
    // Children-ordering lists are Lists in production (`write_object` and the
    // `SetChildren` mutation arm both RPUSH child ids in order), so seed them
    // with RPUSH. The sweep unlinks by key name and never reads the value, so
    // the type does not change behaviour — but seeding a Hash here would
    // misrepresent the key family to the next reader of this fixture.
    let children_key = ks.obj_children(orphan_oid, "10");
    pool.rpush::<(), _, _>(&children_key, vec!["11"])
        .await
        .expect("RPUSH children");

    // Root children-ordering list (parent id 0 — the synthetic root, i.e. the
    // top-level section ordering). b.9fn: `0` is not a section id, so it is a
    // member of NEITHER `obj:secs` NOR `obj:dels`. Both cleanup paths used to
    // derive every `obj:ch:{parent}` key by walking those two indices, so this
    // key was never on the UNLINK list and leaked one key per retired object
    // forever. `KeySchema::object_cleanup_keys` now emits it unconditionally.
    let root_children_key = ks.obj_children(orphan_oid, "0");
    pool.rpush::<(), _, _>(&root_children_key, vec!["10"])
        .await
        .expect("RPUSH root children");

    // Fields index (sorted set) + field hash.
    let flds_index = ks.obj_fields_index(orphan_oid);
    let field_member = KeySchema::field_member("10", "color");
    pool.zadd::<(), _, _>(&flds_index, None, None, false, false, (1.0, &field_member))
        .await
        .expect("ZADD fields index");
    let field_key = format!("{prefix}{{{orphan_oid}}}:obj:fld:{field_member}");
    pool.hset::<(), _, _>(&field_key, [("val", "red")])
        .await
        .expect("HSET field");

    // Time samples index (sorted set) + time sample hash.
    let tss_index = ks.obj_time_samples_index(orphan_oid);
    let ts_member = KeySchema::time_sample_member("10", 1.0);
    pool.zadd::<(), _, _>(&tss_index, None, None, false, false, (1.0, &ts_member))
        .await
        .expect("ZADD time samples index");
    let ts_key = format!("{prefix}{{{orphan_oid}}}:obj:ts:{ts_member}");
    pool.hset::<(), _, _>(&ts_key, [("val", "sample")])
        .await
        .expect("HSET time sample");

    // Deleted sections index + deleted section tombstone + children for deleted section.
    let dels_index = ks.obj_deleted_sections_index(orphan_oid);
    pool.sadd::<(), _, _>(&dels_index, "20")
        .await
        .expect("SADD deleted sections index");
    let del_key = ks.obj_deleted_section(orphan_oid, "20");
    pool.hset::<(), _, _>(&del_key, [("ver", "3")])
        .await
        .expect("HSET deleted section");
    let del_children_key = ks.obj_children(orphan_oid, "20");
    pool.rpush::<(), _, _>(&del_children_key, vec!["21"])
        .await
        .expect("RPUSH deleted section children");

    // Children list under a DANGLING parent + the index that names it (b.jdr).
    // Parent 99 is not a section: it is in neither `obj:secs` nor `obj:dels`,
    // and unlike the synthetic root it is not a fixed id the inventory can
    // hard-code. `apply_diff_change_section` creates exactly this shape —
    // parent existence is not validated, and a dangling parent is a tolerated
    // state (SDD §10.4) — so before b.jdr this key survived every sweep, one
    // leak per dangling parent per swept object, forever. `obj:chs` is the
    // recorded parent-id index that makes it enumerable;
    // `cleanup_orphaned_object` SMEMBERSes it and unions the result into
    // `KeySchema::object_cleanup_keys`. Both keys are in `all_keys` below, so
    // the sweep must reap the index along with what it indexes.
    let dangling_children_key = ks.obj_children(orphan_oid, "99");
    pool.rpush::<(), _, _>(&dangling_children_key, vec!["98"])
        .await
        .expect("RPUSH dangling parent children");
    let children_index_key = ks.obj_children_index(orphan_oid);
    pool.sadd::<(), _, _>(&children_index_key, vec!["0", "10", "20", "99"])
        .await
        .expect("SADD children index");

    // The UpdateLayer commit-token record (b.75i). Written by every commit of
    // an object that took an `UpdateLayer`, named by no index, and keyed only
    // by the object id — `object_cleanup_keys` emits it unconditionally, and
    // without that it would survive every sweep of an orphan that was ever
    // written to.
    let commit_token_key = ks.obj_write_token(orphan_oid);
    pool.hset::<(), _, _>(
        &commit_token_key,
        [(
            "gone-writer",
            "0123456789abcdef0123456789abcdef:3:1700000000000000",
        )],
    )
    .await
    .expect("HSET commit-token record");

    // The snapshot fence and the writer-intent key (b.y3m Epic 3, Spec SDD
    // SR-5.1), shaped as their scripts write them: a fence left by a closed
    // epoch (its gap and recorded size) and a WATCH-path writer's intent.
    // Named by no index and keyed only by the object id, like `obj:wtok`:
    // `object_cleanup_keys` emits both unconditionally, and without that they
    // would survive every sweep of an orphan that was ever fenced.
    let snapshot_fence_key = ks.obj_snapshot_fence(orphan_oid);
    pool.hset::<(), _, _>(
        &snapshot_fence_key,
        [
            ("gap_until", "1700000000000"),
            ("size_e", "150216"),
            ("size_b", "1799752"),
        ],
    )
    .await
    .expect("HSET snapshot fence");
    let snapshot_intent_key = ks.obj_snapshot_intent(orphan_oid);
    pool.hset::<(), _, _>(
        &snapshot_intent_key,
        [("w:0123456789abcdef0123456789abcdef", "1700000000000")],
    )
    .await
    .expect("HSET writer intent");

    // Collect all keys we expect to be cleaned up.
    let all_keys = vec![
        header_key.clone(),
        secs_index.clone(),
        sec_key.clone(),
        children_key.clone(),
        root_children_key.clone(),
        flds_index.clone(),
        field_key.clone(),
        tss_index.clone(),
        ts_key.clone(),
        dels_index.clone(),
        del_key.clone(),
        del_children_key.clone(),
        dangling_children_key.clone(),
        children_index_key.clone(),
        commit_token_key.clone(),
        snapshot_fence_key.clone(),
        snapshot_intent_key.clone(),
    ];

    // Verify they all exist before the sweep.
    for key in &all_keys {
        assert!(
            pool.exists::<bool, _>(key).await.unwrap(),
            "{key} should exist before sweep"
        );
    }

    // Run the sweep.
    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    let header_for_poll = header_key.clone();
    let poll_pool = tv.client().pool.clone();
    run_one_sweep_iteration(handle, cancel, move || {
        let key = header_for_poll.clone();
        let pool = poll_pool.clone();
        async move { !pool.exists::<bool, _>(&key).await.unwrap() }
    })
    .await;

    // Verify every sub-key was cleaned up.
    for key in &all_keys {
        assert!(
            !pool.exists::<bool, _>(key).await.unwrap(),
            "{key} should be deleted after sweep"
        );
    }
}

// =============================================================================
// 7. Corrupt header -- missing session_id or layer_name fields
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_cleans_object_with_missing_header_fields() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let pool = &tv.client().pool;

    let orphan_oid = "888";
    let header_key = ks.obj_header(orphan_oid);

    // Header with only seq — missing session_id and layer_name.
    pool.hset::<(), _, _>(&header_key, [("seq", "1"), ("src_fmt", "usd")])
        .await
        .expect("HSET corrupt header");

    assert!(
        pool.exists::<bool, _>(&header_key).await.unwrap(),
        "header should exist before sweep"
    );

    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    let header_for_poll = header_key.clone();
    let poll_pool = tv.client().pool.clone();
    run_one_sweep_iteration(handle, cancel, move || {
        let key = header_for_poll.clone();
        let pool = poll_pool.clone();
        async move { !pool.exists::<bool, _>(&key).await.unwrap() }
    })
    .await;

    assert!(
        !pool.exists::<bool, _>(&header_key).await.unwrap(),
        "object with corrupt header (missing session_id/layer_name) should be cleaned up"
    );
}

// =============================================================================
// 8. Many orphans -- SCAN pagination across multiple batches
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_handles_many_orphaned_objects_across_scan_batches() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let pool = &tv.client().pool;

    // Create 150 orphaned objects (SCAN page size is 100, so this spans 2+ pages).
    let count = 150u64;
    let mut header_keys = Vec::with_capacity(count as usize);

    for i in 0..count {
        let oid = format!("batch-{i}");
        let header_key = ks.obj_header(&oid);
        pool.hset::<(), _, _>(
            &header_key,
            [
                ("seq", "1"),
                ("first_ver", "1"),
                ("src_fmt", "usd"),
                ("session_id", "dead-session"),
                ("layer_name", "dead-layer"),
            ],
        )
        .await
        .expect("HSET header");
        header_keys.push(header_key);
    }

    // Verify all exist.
    for key in &header_keys {
        assert!(
            pool.exists::<bool, _>(key).await.unwrap(),
            "{key} should exist before sweep"
        );
    }

    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    // Wait until ALL orphan headers are gone — polling on just the "last"
    // key races SCAN order (the sweep is serial, so polling on a
    // first-processed key would cancel mid-iteration and leave later
    // orphans unprocessed).
    run_sweep_until_all_keys_deleted(
        handle,
        cancel,
        tv.client().pool.clone(),
        header_keys.clone(),
    )
    .await;

    // Verify all orphaned headers are deleted.
    for key in &header_keys {
        assert!(
            !pool.exists::<bool, _>(key).await.unwrap(),
            "{key} should be deleted after sweep"
        );
    }
}

// =============================================================================
// 9. Mixed valid and orphan -- sweep deletes orphans, preserves valid
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_mixed_valid_and_orphan_objects_only_removes_orphans() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let ls = layer_store(&tv);
    let pool = &tv.client().pool;
    let session_id = create_test_session(&tv).await;

    // Create a valid layer.
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let valid_oid = ls
        .create_layer(
            &session_id,
            "valid-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let valid_header = ks.obj_header(&valid_oid.to_string());

    // Create an orphaned object in the same namespace.
    let orphan_oid = "orphan-mixed";
    let orphan_header = ks.obj_header(orphan_oid);
    pool.hset::<(), _, _>(
        &orphan_header,
        [
            ("seq", "1"),
            ("first_ver", "1"),
            ("src_fmt", "usd"),
            ("session_id", "no-such-session"),
            ("layer_name", "no-such-layer"),
        ],
    )
    .await
    .expect("HSET orphan header");

    assert!(pool.exists::<bool, _>(&valid_header).await.unwrap());
    assert!(pool.exists::<bool, _>(&orphan_header).await.unwrap());

    // A zero grace, so it is the mapping, not the valid object's youth, that
    // keeps it.
    tv.client().set_orphan_sweep_grace_for_test(Duration::ZERO);

    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    let orphan_for_poll = orphan_header.clone();
    let poll_pool = tv.client().pool.clone();
    run_one_sweep_iteration(handle, cancel, move || {
        let key = orphan_for_poll.clone();
        let pool = poll_pool.clone();
        async move { !pool.exists::<bool, _>(&key).await.unwrap() }
    })
    .await;

    assert!(
        pool.exists::<bool, _>(&valid_header).await.unwrap(),
        "valid object must survive sweep"
    );
    assert!(
        !pool.exists::<bool, _>(&orphan_header).await.unwrap(),
        "orphaned object must be deleted"
    );
}

// =============================================================================
// 10a. Multi-shard regression (b.deq) -- orphans across many distinct hash
//      tags must ALL be swept.
//
//      Regression guard for the `pool.next().scan_buffered(...)` bug: on a
//      multi-shard Valkey Cluster, a single-connection SCAN only walks the
//      slot range owned by the node fred's pool picked. Sibling-shard orphans
//      would accumulate silently. The fix routes through `scan_keyspace`,
//      which calls `scan_cluster` (fred's per-primary page form) when
//      `client.is_clustered()`.
//
//      This test runs unconditionally against the single-node test Valkey.
//      On a single-node instance every key lives on the same "shard", so this
//      test cannot directly detect the cluster bug — but it establishes the
//      contract the sweeper MUST meet (every orphan, regardless of hash tag,
//      is swept) and guards against a regression that collapses coverage back
//      to a single hash tag or similar narrowing. The true multi-shard case
//      is covered by `sweep_cluster_*` below (gated on
//      `LIVE_EDIT_TEST_CLUSTER_URL`).
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_cleans_orphans_spanning_many_distinct_hash_tags() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let pool = &tv.client().pool;

    // Seed orphans under 64 distinct hash-tag values. In a real Valkey Cluster
    // 64 uniformly-distributed hash tags virtually guarantee spread across all
    // shards (16384 slots / N shards). `fred::util::redis_keyslot` hashes only
    // the characters inside `{...}`, so each `object_id` below maps to a
    // different slot.
    let count = 64u64;
    let mut headers = Vec::with_capacity(count as usize);
    for i in 0..count {
        // Use a hash-tag value that varies significantly across i.
        let oid = format!(
            "shardspread-{i:03}-{:x}",
            i.wrapping_mul(0x9E3779B97F4A7C15)
        );
        let header_key = ks.obj_header(&oid);
        pool.hset::<(), _, _>(
            &header_key,
            [
                ("seq", "1"),
                ("first_ver", "1"),
                ("src_fmt", "usd"),
                ("session_id", "gone-session"),
                ("layer_name", "gone-layer"),
            ],
        )
        .await
        .expect("HSET orphan header");
        headers.push(header_key);
    }

    // Verify every seeded header exists before the sweep.
    for key in &headers {
        assert!(
            pool.exists::<bool, _>(key).await.unwrap(),
            "{key} should exist before sweep"
        );
    }

    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    run_sweep_until_all_keys_deleted(handle, cancel, tv.client().pool.clone(), headers.clone())
        .await;

    // Every orphan header — regardless of hash tag / slot — must be cleaned up.
    // If a future regression replaced `scan_keyspace(...)` with a scan that
    // silently scopes to a subset of keys (e.g. a single hash tag), this assert
    // would fire on the first un-swept header.
    for key in &headers {
        assert!(
            !pool.exists::<bool, _>(key).await.unwrap(),
            "{key} should be deleted after sweep (regression guard for b.deq single-node SCAN bug)"
        );
    }
}

// =============================================================================
// 10b. Multi-shard regression (b.deq) -- same invariant against a real
//      multi-shard Valkey Cluster. Gated behind `LIVE_EDIT_TEST_CLUSTER_URL`:
//      the repo does not ship a cluster harness, so this test is opt-in. The
//      URL must be a `redis-cluster://` URL pointing at any node of a
//      running ≥2-shard cluster (e.g. `redis-cluster://127.0.0.1:7000`).
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_cluster_cleans_orphans_on_non_owner_shards() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };

    let tv = live_edit::storage::test_support::TestValkey::setup_with_url(cluster_url, 4).await;
    let ks = tv.key_schema();
    let pool = &tv.client().pool;

    assert!(
        tv.client().pool.next().is_clustered(),
        "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment \
         (use `redis-cluster://...`); the scan_keyspace cluster branch is \
         only exercised when `is_clustered()` is true"
    );

    // Seed orphans across 32 distinct hash tags. On a ≥2-shard cluster the
    // birthday-paradox math makes it effectively certain that some of these
    // land on shards other than the one fred's `pool.next()` picks first;
    // that is exactly the condition under which the single-node
    // `scan_buffered(...)` bug would silently skip orphans.
    let count = 32u64;
    let mut headers = Vec::with_capacity(count as usize);
    for i in 0..count {
        let oid = format!("cluster-{i:03}-{:x}", i.wrapping_mul(0x9E3779B97F4A7C15));
        let header_key = ks.obj_header(&oid);
        pool.hset::<(), _, _>(
            &header_key,
            [
                ("seq", "1"),
                ("first_ver", "1"),
                ("src_fmt", "usd"),
                ("session_id", "gone-session"),
                ("layer_name", "gone-layer"),
            ],
        )
        .await
        .expect("HSET orphan header");
        headers.push(header_key);
    }

    for key in &headers {
        assert!(
            pool.exists::<bool, _>(key).await.unwrap(),
            "{key} should exist before cluster sweep"
        );
    }

    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    run_sweep_until_all_keys_deleted(handle, cancel, tv.client().pool.clone(), headers.clone())
        .await;

    // Every orphan — including those on non-owner shards — must be cleaned up.
    // Before the fix, `pool.next().scan_buffered(...)` on a multi-shard cluster
    // would only walk the slot range of the single picked node, so headers on
    // sibling shards would survive and this assertion would fail.
    for key in &headers {
        assert!(
            !pool.exists::<bool, _>(key).await.unwrap(),
            "{key} must be swept regardless of which shard owns it (b.deq regression)"
        );
    }
}

// =============================================================================
// 11. Partial orphaned layers -- session exists but some mappings are for a
//     deleted session; layers_index is preserved when other entries remain
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sweep_orphaned_layer_preserves_layers_index_when_other_entries_remain() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let pool = &tv.client().pool;

    // Create a dead session with two layer mappings and a layers index.
    // One mapping belongs to a dead session, the other to a live one.
    let dead_session = "dead-partial";
    let live_session = create_test_session(&tv).await;

    // Dead session: layer mapping + layers index.
    let dead_mapping = ks.layer_mapping(dead_session, "layer-a");
    let dead_layers_index = ks.layers_index(dead_session);
    let hex_a = keys::hex_encode("layer-a");
    pool.set::<(), _, _>(&dead_mapping, "100", None, None, false)
        .await
        .expect("SET dead mapping");
    pool.sadd::<(), _, _>(&dead_layers_index, &hex_a)
        .await
        .expect("SADD dead layers index");

    // Live session: manually add two layer mappings to test that the layers
    // index is preserved when only one orphan mapping is removed.
    let live_mapping_a = ks.layer_mapping(&live_session, "alive-a");
    let live_mapping_b = ks.layer_mapping(&live_session, "alive-b");
    let live_layers_index = ks.layers_index(&live_session);
    let hex_alive_a = keys::hex_encode("alive-a");
    let hex_alive_b = keys::hex_encode("alive-b");
    pool.set::<(), _, _>(&live_mapping_a, "200", None, None, false)
        .await
        .expect("SET live mapping a");
    pool.set::<(), _, _>(&live_mapping_b, "201", None, None, false)
        .await
        .expect("SET live mapping b");
    pool.sadd::<(), _, _>(&live_layers_index, vec![&hex_alive_a, &hex_alive_b])
        .await
        .expect("SADD live layers index");

    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        mock_ext_store(),
        Arc::new(layer_store(&tv)),
        crate::common::test_leader("test-replica"),
        cancel.clone(),
        None,
    );
    let dead_for_poll = dead_mapping.clone();
    let poll_pool = tv.client().pool.clone();
    run_one_sweep_iteration(handle, cancel, move || {
        let key = dead_for_poll.clone();
        let pool = poll_pool.clone();
        async move { !pool.exists::<bool, _>(&key).await.unwrap() }
    })
    .await;

    // Dead session's mapping and index should be cleaned up.
    assert!(
        !pool.exists::<bool, _>(&dead_mapping).await.unwrap(),
        "dead session's layer mapping should be deleted"
    );
    assert!(
        !pool.exists::<bool, _>(&dead_layers_index).await.unwrap(),
        "dead session's layers index should be deleted (empty after SREM)"
    );

    // Live session's mappings and index should survive.
    assert!(
        pool.exists::<bool, _>(&live_mapping_a).await.unwrap(),
        "live mapping a must survive"
    );
    assert!(
        pool.exists::<bool, _>(&live_mapping_b).await.unwrap(),
        "live mapping b must survive"
    );
    assert!(
        pool.exists::<bool, _>(&live_layers_index).await.unwrap(),
        "live layers index must survive (still has members)"
    );
}
