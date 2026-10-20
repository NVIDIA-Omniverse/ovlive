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

// Integration tests for LayerStore: create_layer, overwrite, empty content,
// per-object stream entries.
//
// Tests run against a real Valkey instance via the TestValkey fixture. Each test
// gets an isolated key namespace so tests can run in parallel safely.
//
// Requires a running Valkey. Address is read from TEST_VALKEY_URL, defaulting to
// redis://127.0.0.1:6379.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, StreamsInterface};

use live_edit::domain::types::ObjectState;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys;
use live_edit::storage::layer::{LayerStore, STREAM_DELETE_GRACE_MS};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::OwnedDelta;

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

/// Helper: build a LayerStore from a TestValkey fixture.
fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
}

/// Helper: build a SessionStore from a TestValkey fixture.
fn session_store(tv: &TestValkey) -> SessionStore {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
}

/// Helper: create a session and return its ID. Panics on failure.
async fn create_test_session(tv: &TestValkey) -> String {
    session_store(tv)
        .create("test-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed")
}

// =============================================================================
// 1. New layer -- mapping and header correct
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_mapping_key_points_to_object_id() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };

    let oid = ls
        .create_layer(&session_id, "my-layer", "alice", &state, mock_ext_store())
        .await
        .expect("create_layer must succeed");

    // Verify mapping key exists and points to the object_id.
    let mapping_key = keys.layer_mapping(&session_id, "my-layer");
    let stored_oid: String = tv
        .client()
        .pool
        .get(&mapping_key)
        .await
        .expect("GET mapping must succeed");
    assert_eq!(
        stored_oid,
        oid.to_string(),
        "mapping must point to the allocated object_id"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_header_has_correct_fields() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    let state = ObjectState {
        sequence_num: 5,
        first_version: 3,
        source_format: "usda".to_string(),
        next_set_order: 10,
        ..Default::default()
    };

    let oid = ls
        .create_layer(&session_id, "hdr-layer", "bob", &state, mock_ext_store())
        .await
        .expect("create_layer must succeed");

    let header_key = keys.obj_header(&oid.to_string());
    let header: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL header must succeed");

    assert_eq!(header.get("seq").map(String::as_str), Some("5"));
    assert_eq!(header.get("first_ver").map(String::as_str), Some("3"));
    assert_eq!(header.get("src_fmt").map(String::as_str), Some("usda"));
    assert_eq!(header.get("next_set_order").map(String::as_str), Some("10"));
    assert_eq!(
        header.get("session_id").map(String::as_str),
        Some(session_id.as_str())
    );
    assert_eq!(
        header.get("layer_name").map(String::as_str),
        Some("hdr-layer")
    );
    assert_eq!(header.get("created_by").map(String::as_str), Some("bob"));
    assert!(
        header.contains_key("created_at"),
        "created_at must be present"
    );
    assert!(
        header.contains_key("modified_at"),
        "modified_at must be present"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_layers_index_contains_layer_name() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    let state = ObjectState::default();

    ls.create_layer(&session_id, "idx-layer", "carol", &state, mock_ext_store())
        .await
        .expect("create_layer must succeed");

    let layers_index_key = keys.layers_index(&session_id);
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(&layers_index_key)
        .await
        .expect("SMEMBERS must succeed");

    let hex_name = keys::hex_encode("idx-layer");
    assert!(
        members.contains(&hex_name),
        "layers index must contain the hex-encoded layer name; got: {members:?}"
    );
}

// =============================================================================
// 2. New layer -- XADD 1-0 on per-object stream (atomic with write_object)
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_xadd_initial_stream_entry_on_per_object_stream() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    let state = ObjectState::default();
    let oid = ls
        .create_layer(
            &session_id,
            "notify-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // The initial XADD rides inside the write_object MULTI/EXEC, so it is
    // visible via XRANGE the moment create_layer returns.
    let stream_key = keys.layer_stream(oid);
    let entries: Vec<(String, HashMap<String, String>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE must succeed");

    assert_eq!(
        entries.len(),
        1,
        "new-layer create must append exactly one stream entry, got {entries:?}"
    );
    let (entry_id, fields) = &entries[0];
    assert_eq!(
        entry_id, "1-0",
        "CreateLayer new-layer XADD must use deterministic ID 1-0"
    );
    assert_eq!(
        fields.len(),
        1,
        "stream entry must carry exactly one field, got {fields:?}"
    );
    let payload = fields
        .get("payload")
        .expect("stream entry must have a `payload` field");
    assert!(
        payload.starts_with(&format!("update|{oid}|1|")),
        "payload must start with 'update|<oid>|1|', got: {payload}"
    );
}

// =============================================================================
// 3. Overwrite -- old keys cleaned, mapping swapped
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_mapping_points_to_new_object_id() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    let state = ObjectState::default();

    let oid1 = ls
        .create_layer(
            &session_id,
            "overwrite-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("first create_layer must succeed");

    let oid2 = ls
        .create_layer(
            &session_id,
            "overwrite-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("second create_layer (overwrite) must succeed");

    assert_ne!(oid1, oid2, "overwrite must allocate a new object_id");

    // Mapping must point to the NEW object_id.
    let mapping_key = keys.layer_mapping(&session_id, "overwrite-layer");
    let stored_oid: String = tv
        .client()
        .pool
        .get(&mapping_key)
        .await
        .expect("GET mapping must succeed");
    assert_eq!(
        stored_oid,
        oid2.to_string(),
        "mapping must point to the new object_id after overwrite"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_old_header_cleaned_up() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    let state = ObjectState::default();

    let oid1 = ls
        .create_layer(
            &session_id,
            "cleanup-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("first create_layer must succeed");

    let _oid2 = ls
        .create_layer(
            &session_id,
            "cleanup-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("second create_layer (overwrite) must succeed");

    // The cleanup is async (tokio::spawn). Poll until the old header key is
    // deleted rather than sleeping a fixed duration, which is timing-dependent
    // and flaky under CI load.
    let old_header_key = keys.obj_header(&oid1.to_string());
    let poll_interval = std::time::Duration::from_millis(50);
    let timeout = std::time::Duration::from_secs(5);
    let start = tokio::time::Instant::now();
    loop {
        let exists: bool = tv
            .client()
            .pool
            .exists(&old_header_key)
            .await
            .expect("EXISTS must succeed");
        if !exists {
            break;
        }
        assert!(
            start.elapsed() < timeout,
            "old object header key must be deleted after overwrite cleanup, \
             but still exists after {timeout:?}"
        );
        tokio::time::sleep(poll_interval).await;
    }
}

// =============================================================================
// 4. Empty content -- header written, no section keys
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_empty_state_header_exists_no_section_keys() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    // ObjectState::default() has empty sections, fields, time_samples, children.
    let state = ObjectState::default();

    let oid = ls
        .create_layer(
            &session_id,
            "empty-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer with empty state must succeed");

    let oid_str = oid.to_string();

    // Header must exist.
    let header: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(keys.obj_header(&oid_str))
        .await
        .expect("HGETALL header must succeed");
    assert!(
        !header.is_empty(),
        "header must exist even for empty ObjectState"
    );

    // Sections index must be empty (key should not exist).
    let sections_exists: bool = tv
        .client()
        .pool
        .exists(keys.obj_sections_index(&oid_str))
        .await
        .expect("EXISTS must succeed");
    assert!(
        !sections_exists,
        "sections index must not exist for empty ObjectState"
    );

    // Fields index must be empty (key should not exist).
    let fields_exists: bool = tv
        .client()
        .pool
        .exists(keys.obj_fields_index(&oid_str))
        .await
        .expect("EXISTS must succeed");
    assert!(
        !fields_exists,
        "fields index must not exist for empty ObjectState"
    );

    // Time samples index must be empty (key should not exist).
    let ts_exists: bool = tv
        .client()
        .pool
        .exists(keys.obj_time_samples_index(&oid_str))
        .await
        .expect("EXISTS must succeed");
    assert!(
        !ts_exists,
        "time samples index must not exist for empty ObjectState"
    );
}

// =============================================================================
// 5. Object ID monotonically increasing
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_object_ids_monotonically_increasing() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let state = ObjectState::default();

    let mut prev_oid = 0u64;
    for i in 0..5 {
        let oid = ls
            .create_layer(
                &session_id,
                &format!("mono-layer-{i}"),
                "alice",
                &state,
                mock_ext_store(),
            )
            .await
            .expect("create_layer must succeed");
        assert!(
            oid > prev_oid,
            "object_id {oid} must be greater than previous {prev_oid}"
        );
        prev_oid = oid;
    }
}

// =============================================================================
// 6. Layer name with special characters
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_special_chars_hex_encoding_correct() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    // Non-ASCII layer name with colons and spaces -- these must be hex-encoded in keys.
    let layer_name = "layer:with/special chars\u{00e9}";
    let state = ObjectState::default();

    let oid = ls
        .create_layer(&session_id, layer_name, "alice", &state, mock_ext_store())
        .await
        .expect("create_layer with special chars must succeed");

    // Verify the mapping key uses hex encoding.
    let mapping_key = keys.layer_mapping(&session_id, layer_name);
    let stored_oid: String = tv
        .client()
        .pool
        .get(&mapping_key)
        .await
        .expect("GET mapping must succeed");
    assert_eq!(
        stored_oid,
        oid.to_string(),
        "mapping for special-char layer name must resolve correctly"
    );

    // Verify the layers index contains the hex-encoded name.
    let layers_index_key = keys.layers_index(&session_id);
    let members: Vec<String> = tv
        .client()
        .pool
        .smembers(&layers_index_key)
        .await
        .expect("SMEMBERS must succeed");

    let hex_name = keys::hex_encode(layer_name);
    assert!(
        members.contains(&hex_name),
        "layers index must contain hex-encoded special-char name; expected {hex_name}, got: {members:?}"
    );

    // Verify the hex encoding doesn't contain literal colons (which would conflict
    // with the key delimiter).
    assert!(
        !hex_name.contains(':'),
        "hex-encoded layer name must not contain literal colons: {hex_name}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_unicode_name_round_trips() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    // Japanese layer name.
    let layer_name = "\u{30ec}\u{30a4}\u{30e4}\u{30fc}";
    let state = ObjectState::default();

    let oid = ls
        .create_layer(&session_id, layer_name, "alice", &state, mock_ext_store())
        .await
        .expect("create_layer with unicode name must succeed");

    // Verify we can read the header using the allocated object_id.
    let header: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(keys.obj_header(&oid.to_string()))
        .await
        .expect("HGETALL header must succeed");

    assert_eq!(
        header.get("layer_name").map(String::as_str),
        Some(layer_name),
        "layer_name stored in header must match the original unicode name"
    );
}

// =============================================================================
// 7. wait_for_cleanup awaits background overwrite cleanup (regression: b.rbu)
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wait_for_cleanup_ensures_old_object_keys_deleted() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    let state = ObjectState::default();

    let oid1 = ls
        .create_layer(
            &session_id,
            "tracked-cleanup",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("first create_layer must succeed");

    let _oid2 = ls
        .create_layer(
            &session_id,
            "tracked-cleanup",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("second create_layer (overwrite) must succeed");

    // Use wait_for_cleanup instead of polling — this is the exact fix for b.rbu.
    // If cleanup tasks were fire-and-forget (no tracking), this would return
    // immediately and the assertion below would be timing-dependent.
    ls.wait_for_cleanup(Duration::from_secs(5)).await;

    // After wait_for_cleanup returns, the old object's keys must be gone.
    let old_header_key = keys.obj_header(&oid1.to_string());
    let exists: bool = tv
        .client()
        .pool
        .exists(&old_header_key)
        .await
        .expect("EXISTS must succeed");
    assert!(
        !exists,
        "old object header must be deleted after wait_for_cleanup completes"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wait_for_cleanup_no_pending_tasks_returns_immediately() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    // Create a single layer (no overwrite → no background cleanup tasks).
    let state = ObjectState::default();
    ls.create_layer(
        &session_id,
        "no-overwrite",
        "alice",
        &state,
        mock_ext_store(),
    )
    .await
    .expect("create_layer must succeed");

    // wait_for_cleanup must return promptly when there are no pending tasks.
    // Use a short timeout to prove it doesn't block.
    let start = tokio::time::Instant::now();
    ls.wait_for_cleanup(Duration::from_secs(5)).await;
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(1),
        "wait_for_cleanup with no pending tasks should return nearly instantly, took {elapsed:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wait_for_cleanup_multiple_overwrites_all_cleaned() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    let state = ObjectState::default();

    // Create the layer, then overwrite it twice → two background cleanup tasks.
    let oid1 = ls
        .create_layer(
            &session_id,
            "multi-overwrite",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("first create_layer must succeed");

    let oid2 = ls
        .create_layer(
            &session_id,
            "multi-overwrite",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("second create_layer must succeed");

    let oid3 = ls
        .create_layer(
            &session_id,
            "multi-overwrite",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("third create_layer must succeed");

    // wait_for_cleanup must drain ALL pending cleanup tasks.
    ls.wait_for_cleanup(Duration::from_secs(5)).await;

    // Both old objects (oid1 and oid2) must be cleaned up.
    for old_oid in [oid1, oid2] {
        let header_key = keys.obj_header(&old_oid.to_string());
        let exists: bool = tv
            .client()
            .pool
            .exists(&header_key)
            .await
            .expect("EXISTS must succeed");
        assert!(
            !exists,
            "old object {old_oid} header must be deleted after wait_for_cleanup"
        );
    }

    // The current object (oid3) must still exist.
    let current_header_key = keys.obj_header(&oid3.to_string());
    let exists: bool = tv
        .client()
        .pool
        .exists(&current_header_key)
        .await
        .expect("EXISTS must succeed");
    assert!(
        exists,
        "current object {oid3} header must still exist after cleanup"
    );
}

// =============================================================================
// 8. Stream-key cleanup regression (bug J1 / b.k2t): per-object stream must be
//    scheduled for cleanup by DeleteLayer and the overwrite path so it does not
//    leak indefinitely after its object is gone.
//
//    b.k2t upgraded this from immediate UNLINK to PEXPIRE with a grace window
//    (`STREAM_DELETE_GRACE_MS`) so that subscribers lagging between XREAD
//    round-trips can still observe the `deleted` sentinel before the key
//    disappears. These tests verify PEXPIRE is applied — not UNLINK —
//    and that TTL is bounded by the configured grace.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_pexpires_layer_stream_key_with_grace() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };

    let oid = ls
        .create_layer(
            &session_id,
            "stream-gc-layer",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // A few UpdateLayer calls to push extra entries onto the stream, proving the
    // PEXPIRE schedules deletion of a non-empty stream (not just the initial
    // XADD 1-0 entry).
    for i in 0..3 {
        ls.update_layer_relayed(
            &session_id,
            "stream-gc-layer",
            oid,
            Some(OwnedDelta::default()),
            "client-a",
            1000 + i as u64,
        )
        .await
        .expect("update_layer must succeed");
    }

    let stream_key = keys.layer_stream(oid);
    let ttl_before: i64 = tv
        .client()
        .pool
        .pttl(&stream_key)
        .await
        .expect("PTTL must succeed");
    assert_eq!(
        ttl_before, -1,
        "sanity: stream key must be persistent (no TTL) before DeleteLayer"
    );

    ls.delete_layer(&session_id, "stream-gc-layer", mock_ext_store())
        .await
        .expect("delete_layer must succeed");

    let ttl_after: i64 = tv
        .client()
        .pool
        .pttl(&stream_key)
        .await
        .expect("PTTL must succeed");
    // b.k2t contract: DeleteLayer must PEXPIRE the stream (not UNLINK).
    //   PTTL = -2 would mean the key was UNLINKed (bug, regresses b.k2t).
    //   PTTL = -1 would mean PEXPIRE never ran (bug, regresses J1).
    //   PTTL > 0 and <= STREAM_DELETE_GRACE_MS is the expected state.
    assert!(
        ttl_after > 0,
        "stream key {stream_key} must have a positive PTTL after DeleteLayer \
         (b.k2t: PEXPIRE, not immediate UNLINK — lagging subscribers need a grace window); \
         got PTTL={ttl_after}ms"
    );
    assert!(
        ttl_after <= STREAM_DELETE_GRACE_MS,
        "stream key PTTL must be bounded by STREAM_DELETE_GRACE_MS \
         ({STREAM_DELETE_GRACE_MS}ms), got {ttl_after}ms"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn overwrite_layer_pexpires_old_layer_stream_key_with_grace() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };

    let oid_a = ls
        .create_layer(
            &session_id,
            "overwrite-stream-gc",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("first create_layer must succeed");

    // Push a few stream entries onto the old object's stream.
    for i in 0..3 {
        ls.update_layer_relayed(
            &session_id,
            "overwrite-stream-gc",
            oid_a,
            Some(OwnedDelta::default()),
            "client-a",
            1000 + i as u64,
        )
        .await
        .expect("update_layer must succeed");
    }

    let old_stream_key = keys.layer_stream(oid_a);
    let ttl_before: i64 = tv
        .client()
        .pool
        .pttl(&old_stream_key)
        .await
        .expect("PTTL must succeed");
    assert_eq!(
        ttl_before, -1,
        "sanity: old stream key must be persistent (no TTL) before overwrite"
    );

    // Overwrite triggers async cleanup of the old object's keys.
    let oid_b = ls
        .create_layer(
            &session_id,
            "overwrite-stream-gc",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("overwrite create_layer must succeed");
    assert_ne!(oid_a, oid_b, "overwrite must allocate a new object_id");

    // Wait for the spawned cleanup task(s) to drain.
    ls.wait_for_cleanup(Duration::from_secs(5)).await;

    let old_ttl: i64 = tv
        .client()
        .pool
        .pttl(&old_stream_key)
        .await
        .expect("PTTL must succeed");
    // b.k2t: overwrite cleanup must PEXPIRE the old stream rather than
    // UNLINKing it, for the same subscriber-race reason as DeleteLayer.
    assert!(
        old_ttl > 0,
        "old stream key {old_stream_key} must carry a positive PTTL after \
         overwrite cleanup (b.k2t: PEXPIRE, not UNLINK); got PTTL={old_ttl}ms"
    );
    assert!(
        old_ttl <= STREAM_DELETE_GRACE_MS,
        "old stream PTTL must be bounded by STREAM_DELETE_GRACE_MS \
         ({STREAM_DELETE_GRACE_MS}ms), got {old_ttl}ms"
    );

    // The new object's stream must still exist and must NOT carry a TTL —
    // cleanup must be scoped to the old object_id only.
    let new_stream_key = keys.layer_stream(oid_b);
    let new_ttl: i64 = tv
        .client()
        .pool
        .pttl(&new_stream_key)
        .await
        .expect("PTTL must succeed");
    assert_eq!(
        new_ttl, -1,
        "new stream key {new_stream_key} must survive overwrite cleanup \
         with no TTL (PTTL=-1 = persistent); got {new_ttl}"
    );
}

// =============================================================================
// b.avd item 1: concurrent create_layer MULTI/EXEC pairs must commit cleanly
//
// Pre-fix `create_layer`'s new-layer mapping transaction used
// `pool.next().multi()` on the multiplexed data pool. Two concurrent
// callers could land on the same TCP connection, interleaving each other's
// MULTI commands between MULTI and EXEC and corrupting both transactions
// (silent SET/SADD loss + EXEC committing the wrong queue). Post-fix the
// MULTI ran on an exclusive `txn_pool` connection, so two concurrent
// callers serialised at `acquire_txn` and each got a clean session. Since
// b.v6d the mapping `SET` and layers-index `SADD` are one atomic `EVALSHA` of
// `scripts/create_layer_commit.lua`, still on an exclusive `txn_pool`
// connection; the end-state invariant below is unchanged.
//
// The regression shape here drives N concurrent `create_layer` calls
// against the SAME session, then verifies the final mapping + layers
// index state is consistent — every layer must have a unique object_id,
// every layer name must appear in the layers index, and every mapping
// must resolve to one of those object_ids. With the pre-fix bug, some
// SET / SADD commands silently disappeared into the wrong transaction
// queue and either the mapping or the layers index ended up missing
// entries.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_create_layer_does_not_interleave_multi_exec() {
    let tv = TestValkey::setup().await;
    let ls = Arc::new(layer_store(&tv));
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    const N: usize = 16;
    let mut handles = Vec::with_capacity(N);
    for i in 0..N {
        let ls = Arc::clone(&ls);
        let session_id = session_id.clone();
        handles.push(tokio::spawn(async move {
            let state = ObjectState {
                sequence_num: 1,
                first_version: 1,
                source_format: "usd".to_string(),
                next_set_order: 1,
                ..ObjectState::default()
            };
            let name = format!("layer-{i:03}");
            ls.create_layer(&session_id, &name, "alice", &state, mock_ext_store())
                .await
                .map(|oid| (name, oid))
        }));
    }

    // Collect all results: every call must succeed.
    let mut results: Vec<(String, u64)> = Vec::with_capacity(N);
    for h in handles {
        let res = h.await.expect("task must not panic");
        results.push(res.expect("concurrent create_layer must succeed"));
    }
    assert_eq!(results.len(), N);

    // Each object_id must be unique — counter-allocator is atomic, but
    // if MULTI/EXEC pairs ever interleaved, two callers could land on
    // the same `existing` branch and clobber each other's SET, and we
    // wouldn't detect that here directly. The next two assertions cover
    // the real interleaving signal.
    let unique_oids: std::collections::HashSet<u64> = results.iter().map(|(_, o)| *o).collect();
    assert_eq!(
        unique_oids.len(),
        N,
        "every concurrent create_layer must allocate a unique object_id"
    );

    // 1) Every layer mapping must resolve to the object_id we got back.
    //    A lost SET in the multiplex-interleave scenario surfaces here
    //    as a missing key or an unexpected value.
    for (name, oid) in &results {
        let mapping_key = keys.layer_mapping(&session_id, name);
        let got: Option<String> = tv.client().pool.get(&mapping_key).await.expect("GET");
        assert_eq!(
            got.as_deref(),
            Some(oid.to_string().as_str()),
            "mapping {mapping_key} must point at the object_id from create_layer; \
             pre-fix interleaving could lose the SET",
        );
    }

    // 2) The layers index must contain every layer name. A lost SADD in
    //    the multiplex-interleave scenario surfaces as a missing index
    //    member — list_layers / SubscribeReadLayer would 404 the layer
    //    even though create returned OK.
    let layers_index_key = keys.layers_index(&session_id);
    let index_members: Vec<String> = tv
        .client()
        .pool
        .smembers(&layers_index_key)
        .await
        .expect("SMEMBERS");
    let expected: std::collections::HashSet<String> = results
        .iter()
        .map(|(name, _)| keys::hex_encode(name))
        .collect();
    let got: std::collections::HashSet<String> = index_members.into_iter().collect();
    assert_eq!(
        got, expected,
        "layers index must contain every concurrently-created layer name; \
         pre-fix interleaving could lose SADD members",
    );
}
