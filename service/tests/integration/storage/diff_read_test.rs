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

// Integration tests for diff_read: the object-snapshot round trip, the empty
// commit's payload, the two faithful catch-up builders' envelopes and metrics,
// and build_diff_payload.
//
// Section A's round-trip cases run through three readers with identical
// expectations: the fenced read's whole-row fan-out, and
// `read_object_snapshot`'s atomic read on the object as written (marked) and
// with its `chs_complete` marker stripped (unmarked — the full children
// read). The atomic read's own cases live in `snapshot_read_atomic_test.rs`,
// the fan-outs' in `snapshot_read_fenced_test.rs` and
// `snapshot_read_unfenced_test.rs`.
//
// Tests run against a real Valkey instance via the TestValkey fixture. Each test
// gets an isolated key namespace so tests can run in parallel safely.
//
// Requires a running Valkey. Address is read from TEST_VALKEY_URL, defaulting to
// redis://127.0.0.1:6379.

use std::collections::HashMap;
use std::sync::Arc;

use fred::interfaces::HashesInterface;
use ordered_float::OrderedFloat;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::diff::{DiffPlan, ExternalValueScan};
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};
use live_edit::error::ServiceError;
use live_edit::storage::diff_read::{
    build_diff_payload, empty_commit_payload, read_object_snapshot, snapshot_to_object_state,
    DiffCase, DiffCut, ObjectSnapshot, SnapshotRead, SnapshotReadConsumer,
};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{LayerStore, HEADER_FIELD_CHS_COMPLETE};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::{read_object_state_for_test, TestValkey};
use oodf_ffi::{parse_delta, StructCommand};

use crate::common::owned_delta::delta_with_set_field_inline;
use crate::storage::snapshot_read_support::read_atomic;

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

fn empty_scan() -> ExternalValueScan {
    ExternalValueScan {
        ordered_hashes: Vec::new(),
        hash_to_index: HashMap::new(),
    }
}

fn empty_ext_data() -> HashMap<[u8; 32], Vec<u8>> {
    HashMap::new()
}

/// Helper: build a LayerStore from a TestValkey fixture.
fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
}

/// Helper: build a SessionStore from a TestValkey fixture.
fn session_store(tv: &TestValkey) -> SessionStore {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
}

/// Helper: create a session and return its ID.
async fn create_test_session(tv: &TestValkey) -> String {
    session_store(tv)
        .create("test-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed")
}

/// Helper: build an ObjectState with sections, children, fields, and time samples.
fn rich_object_state() -> ObjectState {
    let mut state = ObjectState {
        object_id: 0, // will be assigned by LayerStore
        sequence_num: 7,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 10,
        ..Default::default()
    };

    // Two sections: sec 1 (root child) and sec 2 (child of 1).
    let sec1 = Section {
        section_id: 1,
        parent_id: 0,
        name: "World".to_string(),
        section_type: 1,
        section_order: 0,
        first_version: 1,
        last_version: 5,
    };
    let sec2 = Section {
        section_id: 2,
        parent_id: 1,
        name: "Cube".to_string(),
        section_type: 2,
        section_order: 1,
        first_version: 2,
        last_version: 7,
    };

    state.sections.insert(1, sec1);
    state.sections.insert(2, sec2);

    // Children: root has [1], section 1 has [2].
    state.children.insert(0, vec![1]);
    state.children.insert(1, vec![2]);

    // Two fields. `authoring_set_order` is seeded DISTINCT from the cumulative
    // `set_order` so a read-path bug that folds one into the other (or drops
    // one) is caught by the round-trip assertions below.
    let field1 = Field {
        section_id: 1,
        key_name: "color".to_string(),
        value: FieldValue::Inline(vec![0xFF, 0x00, 0x80]),
        set_order: 1,
        authoring_set_order: 101,
        version: 3,
    };
    let field2 = Field {
        section_id: 2,
        key_name: "size".to_string(),
        value: FieldValue::Inline(vec![0x42]),
        set_order: 2,
        authoring_set_order: 202,
        version: 5,
    };
    state
        .fields
        .insert((field1.section_id, field1.key_name.clone()), field1);
    state
        .fields
        .insert((field2.section_id, field2.key_name.clone()), field2);

    // Two time samples.
    let ts1 = TimeSample {
        section_id: 1,
        time: OrderedFloat(0.0),
        value: FieldValue::Inline(vec![0xAA, 0xBB]),
        version: 2,
    };
    let ts2 = TimeSample {
        section_id: 2,
        time: OrderedFloat(1.5),
        value: FieldValue::Inline(vec![0xCC]),
        version: 6,
    };
    state.time_samples.insert((ts1.section_id, ts1.time), ts1);
    state.time_samples.insert((ts2.section_id, ts2.time), ts2);

    state
}

// =============================================================================
// A. Integration tests: the object-snapshot round trip, on every reader
// =============================================================================

/// The reader a section-A case reads through. Every case asserts the same
/// expectations on all three: the fan-out and the atomic read read the same
/// key set under the same children rule, so any state one round-trips the
/// other must round-trip identically (Spec SDD SR-1.2), and the atomic read
/// must do so on a marked object (reads the lists `obj:chs` names) and on an
/// unmarked one (reads the root plus every section's list).
#[derive(Debug, Clone, Copy)]
enum Reader {
    /// The catch-up's fenced read (`read_object_snapshot_fenced`): the
    /// bounded fan-out's pass, decoding every row whole, on the object as
    /// written (marked). The one full-state fan-out reader since b.fsz
    /// MR-D4 removed the ext-GC snapshot read, which read the unfenced
    /// fan-out whole; the structural consumer reads the unfenced fan-out but
    /// keeps only each row's key and hash, so it cannot show the values these
    /// tests check.
    Fenced,
    /// `read_object_snapshot` on the object as `create_layer` wrote it, which
    /// carries the `chs_complete` marker.
    AtomicMarked,
    /// `read_object_snapshot` after the marker is stripped — an object
    /// materialized before the marker existed.
    AtomicUnmarked,
}

impl Reader {
    async fn read(self, tv: &TestValkey, oid: u64) -> ObjectSnapshot {
        match self {
            Reader::Fenced => {
                crate::storage::b_atj_tests::fenced_read(tv.client(), tv.key_schema(), oid)
                    .await
                    .unwrap_or_else(|e| panic!("the fenced fan-out read must succeed; got {e:?}"))
                    .state
            }
            Reader::AtomicMarked => read_atomic(tv.client(), tv.key_schema(), oid).await.state,
            Reader::AtomicUnmarked => {
                let removed: i64 = tv
                    .client()
                    .pool
                    .hdel(
                        tv.key_schema().obj_header(&oid.to_string()),
                        HEADER_FIELD_CHS_COMPLETE,
                    )
                    .await
                    .expect("HDEL chs_complete must succeed");
                assert_eq!(
                    removed, 1,
                    "fixture precondition: create_layer marks the object, so the marker must \
                     have been there to strip"
                );
                read_atomic(tv.client(), tv.key_schema(), oid).await.state
            }
        }
    }
}

// 1. Round-trip: write via LayerStore, read back.

async fn round_trip_all_data_matches(reader: Reader) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let state = rich_object_state();

    let oid = ls
        .create_layer(&session_id, "rt-layer", "alice", &state, mock_ext_store())
        .await
        .expect("create_layer must succeed");

    let snapshot = reader.read(&tv, oid).await;

    // Header fields.
    assert_eq!(snapshot.sequence_num, state.sequence_num);
    assert_eq!(snapshot.first_version, state.first_version);
    assert_eq!(snapshot.source_format, state.source_format);
    assert_eq!(snapshot.next_set_order, state.next_set_order);

    // Sections.
    assert_eq!(snapshot.sections.len(), 2, "should have 2 sections");
    let sec1 = &snapshot.sections[&1];
    assert_eq!(sec1.name, "World");
    assert_eq!(sec1.parent_id, 0);
    assert_eq!(sec1.section_type, 1);
    assert_eq!(sec1.section_order, 0);
    assert_eq!(sec1.first_version, 1);
    assert_eq!(sec1.last_version, 5);

    let sec2 = &snapshot.sections[&2];
    assert_eq!(sec2.name, "Cube");
    assert_eq!(sec2.parent_id, 1);
    assert_eq!(sec2.section_type, 2);
    assert_eq!(sec2.section_order, 1);

    // Children: root has [1], section 1 has [2].
    assert_eq!(
        snapshot.children.get(&0).map(|v| v.as_slice()),
        Some([1u64].as_slice()),
        "root children"
    );
    assert_eq!(
        snapshot.children.get(&1).map(|v| v.as_slice()),
        Some([2u64].as_slice()),
        "section 1 children"
    );

    // Fields.
    assert_eq!(snapshot.fields.len(), 2, "should have 2 fields");
    let f_color = snapshot
        .fields
        .iter()
        .find(|f| f.key_name == "color")
        .expect("color field must exist");
    assert_eq!(f_color.section_id, 1);
    assert_eq!(f_color.value, FieldValue::Inline(vec![0xFF, 0x00, 0x80]));
    assert_eq!(f_color.set_order, 1);
    assert_eq!(
        f_color.authoring_set_order, 101,
        "authoring_set_order must round-trip independently of set_order"
    );
    assert_ne!(
        f_color.authoring_set_order, f_color.set_order,
        "seeded values are distinct; a bug folding one into the other must be caught"
    );
    assert_eq!(f_color.version, 3);

    let f_size = snapshot
        .fields
        .iter()
        .find(|f| f.key_name == "size")
        .expect("size field must exist");
    assert_eq!(f_size.section_id, 2);
    assert_eq!(f_size.value, FieldValue::Inline(vec![0x42]));
    assert_eq!(f_size.set_order, 2);
    assert_eq!(
        f_size.authoring_set_order, 202,
        "authoring_set_order must round-trip independently of set_order"
    );
    assert_ne!(
        f_size.authoring_set_order, f_size.set_order,
        "seeded values are distinct; a bug folding one into the other must be caught"
    );
    assert_eq!(f_size.version, 5);

    // Time samples.
    assert_eq!(snapshot.time_samples.len(), 2, "should have 2 time samples");
    let ts_0 = snapshot
        .time_samples
        .iter()
        .find(|ts| ts.time == OrderedFloat(0.0))
        .expect("time sample at 0.0 must exist");
    assert_eq!(ts_0.section_id, 1);
    assert_eq!(ts_0.value, FieldValue::Inline(vec![0xAA, 0xBB]));
    assert_eq!(ts_0.version, 2);

    let ts_1 = snapshot
        .time_samples
        .iter()
        .find(|ts| ts.time == OrderedFloat(1.5))
        .expect("time sample at 1.5 must exist");
    assert_eq!(ts_1.section_id, 2);
    assert_eq!(ts_1.value, FieldValue::Inline(vec![0xCC]));
    assert_eq!(ts_1.version, 6);

    // Tombstones should be empty (no deleted sections).
    assert!(
        snapshot.tombstones.is_empty(),
        "no tombstones expected for fresh layer"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_round_trip_all_data_matches() {
    round_trip_all_data_matches(Reader::Fenced).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_round_trip_all_data_matches_marked() {
    round_trip_all_data_matches(Reader::AtomicMarked).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_round_trip_all_data_matches_unmarked() {
    round_trip_all_data_matches(Reader::AtomicUnmarked).await;
}

// 2. NotFound: a non-existent object_id. (No marked / unmarked split: there is
// no header to carry the marker.)

/// The test-support state reader's `NotFound` for an absent header is a
/// contract the recovery tests lean on (they assert a wipe left the object
/// unreadable through it), so it is pinned here beside the service readers'.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_state_for_test_nonexistent_object_returns_not_found() {
    let tv = TestValkey::setup().await;

    let result = read_object_state_for_test(tv.client(), tv.key_schema(), "999999999").await;

    match result {
        Err(ServiceError::NotFound(_)) => {} // expected
        Err(other) => panic!("expected NotFound, got: {other:?}"),
        Ok(_) => panic!("expected NotFound error, got Ok"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_nonexistent_object_returns_not_found() {
    let tv = TestValkey::setup().await;

    let result = read_object_snapshot(
        tv.client(),
        tv.key_schema(),
        "999999999",
        SnapshotReadConsumer::Catchup,
    )
    .await;

    match result {
        Err(ServiceError::NotFound(msg)) => assert!(
            msg.contains("999999999"),
            "the NotFound must name the object id, got: {msg}"
        ),
        Err(other) => panic!("expected NotFound, got: {other:?}"),
        Ok(SnapshotRead::Snapshot(_)) => panic!("expected NotFound error, got a snapshot"),
        Ok(SnapshotRead::TooLarge(t)) => panic!("expected NotFound error, got TOO_LARGE {t:?}"),
    }
}

// 3. Header fields correct.

async fn header_fields_match_stored_values(reader: Reader) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let state = ObjectState {
        sequence_num: 42,
        first_version: 10,
        source_format: "usd".to_string(),
        next_set_order: 99,
        ..Default::default()
    };

    let oid = ls
        .create_layer(&session_id, "header-test", "bob", &state, mock_ext_store())
        .await
        .expect("create_layer must succeed");

    let snapshot = reader.read(&tv, oid).await;

    assert_eq!(snapshot.sequence_num, 42);
    assert_eq!(snapshot.first_version, 10);
    assert_eq!(snapshot.source_format, "usd");
    assert_eq!(snapshot.next_set_order, 99);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_header_fields_match_stored_values() {
    header_fields_match_stored_values(Reader::Fenced).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_header_fields_match_stored_values_marked() {
    header_fields_match_stored_values(Reader::AtomicMarked).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_header_fields_match_stored_values_unmarked() {
    header_fields_match_stored_values(Reader::AtomicUnmarked).await;
}

// 4. Sections and children.

async fn sections_and_children_correct(reader: Reader) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let mut state = ObjectState {
        sequence_num: 3,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };

    // Three sections: root -> 10 -> 20, and root -> 30.
    state.sections.insert(
        10,
        Section {
            section_id: 10,
            parent_id: 0,
            name: "A".to_string(),
            section_type: 0,
            section_order: 0,
            first_version: 1,
            last_version: 2,
        },
    );
    state.sections.insert(
        20,
        Section {
            section_id: 20,
            parent_id: 10,
            name: "B".to_string(),
            section_type: 1,
            section_order: 1,
            first_version: 1,
            last_version: 3,
        },
    );
    state.sections.insert(
        30,
        Section {
            section_id: 30,
            parent_id: 0,
            name: "C".to_string(),
            section_type: 0,
            section_order: 2,
            first_version: 2,
            last_version: 3,
        },
    );
    state.children.insert(0, vec![10, 30]);
    state.children.insert(10, vec![20]);

    let oid = ls
        .create_layer(
            &session_id,
            "sec-children",
            "carol",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let snapshot = reader.read(&tv, oid).await;

    assert_eq!(snapshot.sections.len(), 3);
    assert_eq!(snapshot.sections[&10].name, "A");
    assert_eq!(snapshot.sections[&20].name, "B");
    assert_eq!(snapshot.sections[&30].name, "C");

    // Children: root -> [10, 30], 10 -> [20].
    let root_children = snapshot.children.get(&0).expect("root children");
    assert_eq!(root_children, &vec![10, 30]);

    let sec10_children = snapshot.children.get(&10).expect("sec 10 children");
    assert_eq!(sec10_children, &vec![20]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_sections_and_children_correct() {
    sections_and_children_correct(Reader::Fenced).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_sections_and_children_correct_marked() {
    sections_and_children_correct(Reader::AtomicMarked).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_sections_and_children_correct_unmarked() {
    sections_and_children_correct(Reader::AtomicUnmarked).await;
}

// 5. Fields and time samples.

async fn fields_and_time_samples_correct(reader: Reader) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let mut state = ObjectState {
        sequence_num: 5,
        first_version: 1,
        source_format: "usd".to_string(),
        next_set_order: 3,
        ..Default::default()
    };

    state.sections.insert(
        1,
        Section {
            section_id: 1,
            parent_id: 0,
            name: "root".to_string(),
            section_type: 0,
            section_order: 0,
            first_version: 1,
            last_version: 5,
        },
    );
    state.children.insert(0, vec![1]);

    let f = Field {
        section_id: 1,
        key_name: "opacity".to_string(),
        value: FieldValue::Inline(vec![0x3F, 0x80, 0x00, 0x00]),
        set_order: 1,
        authoring_set_order: 77,
        version: 4,
    };
    state.fields.insert((1, "opacity".to_string()), f);

    let ts = TimeSample {
        section_id: 1,
        time: OrderedFloat(2.5),
        value: FieldValue::Inline(vec![0xDE, 0xAD]),
        version: 3,
    };
    state.time_samples.insert((1, OrderedFloat(2.5)), ts);

    let oid = ls
        .create_layer(
            &session_id,
            "fld-ts-layer",
            "dave",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let snapshot = reader.read(&tv, oid).await;

    assert_eq!(snapshot.fields.len(), 1);
    let loaded_field = &snapshot.fields[0];
    assert_eq!(loaded_field.section_id, 1);
    assert_eq!(loaded_field.key_name, "opacity");
    assert_eq!(
        loaded_field.value,
        FieldValue::Inline(vec![0x3F, 0x80, 0x00, 0x00])
    );
    assert_eq!(loaded_field.set_order, 1);
    assert_eq!(
        loaded_field.authoring_set_order, 77,
        "authoring_set_order must round-trip independently of set_order"
    );
    assert_ne!(
        loaded_field.authoring_set_order, loaded_field.set_order,
        "seeded distinct values guard against a fold bug"
    );
    assert_eq!(loaded_field.version, 4);

    assert_eq!(snapshot.time_samples.len(), 1);
    let loaded_ts = &snapshot.time_samples[0];
    assert_eq!(loaded_ts.section_id, 1);
    assert!((loaded_ts.time.into_inner() - 2.5).abs() < f64::EPSILON);
    assert_eq!(loaded_ts.value, FieldValue::Inline(vec![0xDE, 0xAD]));
    assert_eq!(loaded_ts.version, 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_fields_and_time_samples_correct() {
    fields_and_time_samples_correct(Reader::Fenced).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_fields_and_time_samples_correct_marked() {
    fields_and_time_samples_correct(Reader::AtomicMarked).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_fields_and_time_samples_correct_unmarked() {
    fields_and_time_samples_correct(Reader::AtomicUnmarked).await;
}

// 5b. Direct-write round-trip: a Field whose authoring rank != cumulative
// set_order survives the read AND snapshot_to_object_state.

async fn reconstructs_authoring_set_order_distinct_from_set_order(reader: Reader) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let mut state = ObjectState {
        sequence_num: 4,
        first_version: 1,
        source_format: "usd".to_string(),
        next_set_order: 50,
        ..Default::default()
    };
    state.sections.insert(
        1,
        Section {
            section_id: 1,
            parent_id: 0,
            name: "root".to_string(),
            section_type: 0,
            section_order: 0,
            first_version: 1,
            last_version: 4,
        },
    );
    state.children.insert(0, vec![1]);

    // Cumulative set_order (42) deliberately != raw authoring rank (7).
    let f = Field {
        section_id: 1,
        key_name: "translate".to_string(),
        value: FieldValue::Inline(vec![0x01, 0x02, 0x03, 0x04]),
        set_order: 42,
        authoring_set_order: 7,
        version: 4,
    };
    state.fields.insert((1, "translate".to_string()), f);

    let oid = ls
        .create_layer(
            &session_id,
            "authoring-rt",
            "erin",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let snapshot = reader.read(&tv, oid).await;

    assert_eq!(snapshot.fields.len(), 1);
    let loaded = &snapshot.fields[0];
    assert_eq!(loaded.set_order, 42, "cumulative set_order round-trips");
    assert_eq!(
        loaded.authoring_set_order, 7,
        "raw authoring rank round-trips independently of the cumulative set_order"
    );

    // snapshot_to_object_state must preserve the authoring field into ObjectState.
    let object_state = snapshot_to_object_state(&snapshot);
    let os_field = object_state
        .fields
        .get(&(1, "translate".to_string()))
        .expect("field must survive snapshot_to_object_state");
    assert_eq!(os_field.set_order, 42);
    assert_eq!(
        os_field.authoring_set_order, 7,
        "snapshot_to_object_state must carry authoring_set_order into ObjectState"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_reconstructs_authoring_set_order_distinct_from_set_order() {
    reconstructs_authoring_set_order_distinct_from_set_order(Reader::Fenced).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_reconstructs_authoring_set_order_distinct_from_set_order_marked() {
    reconstructs_authoring_set_order_distinct_from_set_order(Reader::AtomicMarked).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_reconstructs_authoring_set_order_distinct_from_set_order_unmarked() {
    reconstructs_authoring_set_order_distinct_from_set_order(Reader::AtomicUnmarked).await;
}

// 5c. Commit path: update_layer captures the writer's raw per-delta set_order
// as authoring_set_order and reconstructs it on read — proven distinct from the
// cumulative set_order AFTER next_set_order has advanced across prior writes.
// Exercised on both backends (Lua fast path + Txn slow path) which must agree,
// and read back through every reader.

async fn commit_path_authoring_rank_reconstructed(backend: UpdateLayerBackend, reader: Reader) {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend);
    let session_id = create_test_session(&tv).await;

    // Fresh layer: next_set_order starts at 0.
    let create_state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let oid = ls
        .create_layer(
            &session_id,
            "commit-authoring",
            "frank",
            &create_state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // First write: field "a" with per-delta rank 4.
    //   stored set_order = next_set_order(0) + 4 = 4; next_set_order advances to 5.
    let delta1 = delta_with_set_field_inline(1, "a", vec![0xAA], 4);
    ls.update_layer_relayed(
        &session_id,
        "commit-authoring",
        oid,
        Some(delta1),
        "writer-1",
        1000,
    )
    .await
    .expect("first update_layer must succeed");

    // Second write: field "b" with per-delta rank 2, AFTER next_set_order advanced.
    //   stored set_order = next_set_order(>=5) + 2 = 7; authoring rank stays 2.
    let delta2 = delta_with_set_field_inline(1, "b", vec![0xBB], 2);
    ls.update_layer_relayed(
        &session_id,
        "commit-authoring",
        oid,
        Some(delta2),
        "writer-2",
        2000,
    )
    .await
    .expect("second update_layer must succeed");

    let snapshot = reader.read(&tv, oid).await;

    let f_a = snapshot
        .fields
        .iter()
        .find(|f| f.key_name == "a")
        .expect("field a must exist");
    assert_eq!(f_a.set_order, 4, "field a cumulative set_order = 0 + 4");
    assert_eq!(
        f_a.authoring_set_order, 4,
        "field a authoring rank equals the writer's per-delta set_order"
    );

    let f_b = snapshot
        .fields
        .iter()
        .find(|f| f.key_name == "b")
        .expect("field b must exist");
    assert_eq!(
        f_b.set_order, 7,
        "field b cumulative set_order folds in the advanced next_set_order (5 + 2)"
    );
    assert_eq!(
        f_b.authoring_set_order, 2,
        "field b authoring rank is the raw per-delta set_order (2), NOT the cumulative cursor (7)"
    );
    assert_ne!(
        f_b.authoring_set_order, f_b.set_order,
        "the whole point: authoring rank diverges from the cumulative set_order once next_set_order advances"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_reconstructs_authoring_rank_not_cumulative_set_order() {
    commit_path_authoring_rank_reconstructed(UpdateLayerBackend::Lua, Reader::Fenced).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_reconstructs_authoring_rank_not_cumulative_set_order() {
    commit_path_authoring_rank_reconstructed(UpdateLayerBackend::Txn, Reader::Fenced).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_authoring_rank_read_by_read_object_snapshot_marked() {
    commit_path_authoring_rank_reconstructed(UpdateLayerBackend::Lua, Reader::AtomicMarked).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_authoring_rank_read_by_read_object_snapshot_unmarked() {
    commit_path_authoring_rank_reconstructed(UpdateLayerBackend::Lua, Reader::AtomicUnmarked).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_authoring_rank_read_by_read_object_snapshot_marked() {
    commit_path_authoring_rank_reconstructed(UpdateLayerBackend::Txn, Reader::AtomicMarked).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_authoring_rank_read_by_read_object_snapshot_unmarked() {
    commit_path_authoring_rank_reconstructed(UpdateLayerBackend::Txn, Reader::AtomicUnmarked).await;
}

// 5d. Backward-compatible parse: an object hash written WITHOUT the
// authoring_set_order hash field (older-shaped row, pre-t3.msb.35.ey.9u)
// parses without panic and falls back to the cumulative set_order, matching
// the `parse_u64(..).unwrap_or(set_order)` convention in diff_read.rs — a
// decoder both readers share.

async fn missing_authoring_field_falls_back_to_set_order(reader: Reader) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;

    let mut state = ObjectState {
        sequence_num: 2,
        first_version: 1,
        source_format: "usd".to_string(),
        next_set_order: 9,
        ..Default::default()
    };
    state.sections.insert(
        1,
        Section {
            section_id: 1,
            parent_id: 0,
            name: "root".to_string(),
            section_type: 0,
            section_order: 0,
            first_version: 1,
            last_version: 2,
        },
    );
    state.children.insert(0, vec![1]);
    let f = Field {
        section_id: 1,
        key_name: "legacy".to_string(),
        value: FieldValue::Inline(vec![0x11, 0x22]),
        set_order: 8,
        authoring_set_order: 3, // distinct — will be stripped from the row below
        version: 2,
    };
    state.fields.insert((1, "legacy".to_string()), f);

    let oid = ls
        .create_layer(&session_id, "legacy-row", "grace", &state, mock_ext_store())
        .await
        .expect("create_layer must succeed");
    let oid_str = oid.to_string();

    // Simulate an older-shaped row: delete the authoring_set_order hash field.
    let field_key = tv.key_schema().obj_field(&oid_str, "1", "legacy");
    let pool = &tv.client().pool;
    let removed: i64 = pool
        .hdel(&field_key, "authoring_set_order")
        .await
        .expect("HDEL authoring_set_order must succeed");
    assert_eq!(
        removed, 1,
        "authoring_set_order field must have been present"
    );
    let after: HashMap<String, String> = pool
        .hgetall(&field_key)
        .await
        .expect("HGETALL after HDEL must succeed");
    assert!(
        !after.contains_key("authoring_set_order"),
        "authoring_set_order must be absent to simulate an older row"
    );
    assert!(after.contains_key("set_order"), "set_order must remain");

    // Parse must not panic and must fall back to the cumulative set_order.
    let snapshot = reader.read(&tv, oid).await;

    assert_eq!(snapshot.fields.len(), 1);
    let loaded = &snapshot.fields[0];
    assert_eq!(loaded.set_order, 8);
    assert_eq!(
        loaded.authoring_set_order, loaded.set_order,
        "missing authoring_set_order must fall back to the cumulative set_order (deterministic default)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_missing_authoring_field_falls_back_to_set_order() {
    missing_authoring_field_falls_back_to_set_order(Reader::Fenced).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_missing_authoring_field_falls_back_to_set_order_marked() {
    missing_authoring_field_falls_back_to_set_order(Reader::AtomicMarked).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_object_snapshot_missing_authoring_field_falls_back_to_set_order_unmarked() {
    missing_authoring_field_falls_back_to_set_order(Reader::AtomicUnmarked).await;
}

// =============================================================================
// A3. The steady state's empty delta, built with no read
// =============================================================================

/// `empty_commit_payload(base)` — what a subscriber is sent for an empty
/// commit, with no read — is the empty delta at the commit: non-empty bytes,
/// one OMPF part (the OODF delta) and no external value, a diff
/// (`is_diff=true`) based at the previous sequence, `base`, carrying no
/// struct command, field or time sample; labelled `Incremental`, zero
/// sections, at no cut (`DiffCut::Unpinned`: it reads nothing to be cut at).
/// Checked at the smallest base an empty commit can have (1, the commit
/// before it being at least the `CreateLayer` seed) and at larger ones.
#[test]
fn empty_commit_payload_is_the_empty_delta_based_at_the_previous_sequence() {
    use oodf_ffi::OwnedDelta;

    for base in [1u64, 7, 1 << 40] {
        let payload = empty_commit_payload(base).expect("the empty-commit payload must build");
        assert!(
            !payload.payload.is_empty(),
            "base {base}: the payload is never zero bytes — the subscriber still receives a \
             message at the commit's sequence"
        );
        let bytes = payload.payload.to_vec();
        let container = OmpfContainer::parse(&bytes).expect("the payload is an OMPF container");
        assert!(
            container.external_values.is_empty(),
            "base {base}: the empty delta carries no external value"
        );
        let delta = parse_delta(container.delta_bytes).expect("the payload carries an OODF delta");
        assert_eq!(
            (
                delta.struct_commands.len(),
                delta.set_fields.len(),
                delta.time_samples.len()
            ),
            (0, 0, 0),
            "base {base}: no struct command, field or time sample"
        );
        assert_eq!(
            delta,
            OwnedDelta {
                is_diff: true,
                base_version: base,
                ..Default::default()
            },
            "base {base}: a diff based at the previous sequence, changing nothing"
        );
        assert_eq!(
            (payload.case, payload.sections_count, payload.cut),
            (DiffCase::Incremental, 0, DiffCut::Unpinned),
            "base {base}: labelled Incremental, zero sections, at no cut"
        );
    }
}

// =============================================================================
// A4. The two catch-up builders, read-then-build, and the metrics they record
// =============================================================================

/// The `live_diff_case_total{case, service="test"}` sample in `text`, or 0.
fn diff_case_count(text: &str, case: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        text,
        "live_diff_case_total",
        &[("case", case), ("service", "test")],
    )
    .unwrap_or(0.0) as u64
}

/// Every catch-up build goes through one of the two faithful builders, each
/// from one atomic read of the object: the brand-new subscriber's from-empty
/// full state (`compute_faithful_full_content_at_cut`) and the reconnector's
/// coalesced diff (`compute_faithful_coalesced_at_cut`). Each records ONE
/// `live_diff_case_total` sample under its own case — `full_content` and
/// `incremental` — and one `live_diff_build_duration_seconds` observation;
/// no build emits the `up_to_date` case any more (the up-to-date reconnect
/// builds nothing).
///
/// The payload envelopes are pinned beside it: the full state is a diff based
/// at 0 that names its source format and carries every field and time sample;
/// the coalesced diff is based at the reconnector's `have_version`, names no
/// format, and carries only the field and time sample written after it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn faithful_builders_record_their_own_diff_case_and_one_build_duration_each() {
    use live_edit::observability::metrics::SharedMetrics;
    use live_edit::storage::diff_read::{
        compute_faithful_coalesced_at_cut, compute_faithful_full_content_at_cut, ExtValueReads,
    };

    let tv = TestValkey::setup().await;
    let session_id = create_test_session(&tv).await;
    let oid = layer_store(&tv)
        .create_layer(
            &session_id,
            "faithful-builders",
            "alice",
            &rich_object_state(),
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only("test"));
    let ext_store = mock_ext_store();
    let snapshot = read_atomic(tv.client(), tv.key_schema(), oid).await;

    let full = compute_faithful_full_content_at_cut(
        &snapshot,
        ext_store.as_ref(),
        ExtValueReads::recover_for_test(),
        Some(&metrics),
        "test",
    )
    .await
    .expect("the full-state build must succeed")
    .into_ready()
    .expect("a build with fetch-on-miss never answers Missing")
    .restored_for_test(ext_store.as_ref())
    .await
    .expect("the full-state layout's restores must succeed")
    .fill(ext_store.as_ref(), None)
    .await
    .expect("the full-state payload must fill")
    .into_ready()
    .expect("a build with fetch-on-miss never answers Missing");
    let full_bytes = full.payload.to_vec();
    let full_delta = parse_delta(
        OmpfContainer::parse(&full_bytes)
            .expect("the full state is an OMPF container")
            .delta_bytes,
    )
    .expect("the full state carries an OODF delta");
    assert_eq!(full.case, DiffCase::FullContent);
    assert_eq!(
        (
            full_delta.is_diff,
            full_delta.base_version,
            full_delta.source_format.as_str(),
            full_delta.set_fields.len(),
            full_delta.time_samples.len(),
        ),
        (true, 0, "usda", 2, 2),
        "the full state is a diff based at 0, naming its format, carrying every field and \
         time sample"
    );

    // `rich_object_state` is at sequence 7; a reconnector at 4 is missing the
    // `size` field (version 5) and the t=1.5 sample (version 6), and nothing
    // else that carries a value.
    let have_version = 4;
    let coalesced = compute_faithful_coalesced_at_cut(
        &snapshot,
        have_version,
        ext_store.as_ref(),
        ExtValueReads::recover_for_test(),
        Some(&metrics),
        "test",
    )
    .await
    .expect("the coalesced build must succeed")
    .into_ready()
    .expect("a build with fetch-on-miss never answers Missing")
    .restored_for_test(ext_store.as_ref())
    .await
    .expect("the coalesced layout's restores must succeed")
    .fill(ext_store.as_ref(), None)
    .await
    .expect("the coalesced payload must fill")
    .into_ready()
    .expect("a build with fetch-on-miss never answers Missing");
    let coalesced_bytes = coalesced.payload.to_vec();
    let coalesced_delta = parse_delta(
        OmpfContainer::parse(&coalesced_bytes)
            .expect("the coalesced diff is an OMPF container")
            .delta_bytes,
    )
    .expect("the coalesced diff carries an OODF delta");
    assert_eq!(coalesced.case, DiffCase::Incremental);
    assert_eq!(
        (
            coalesced_delta.is_diff,
            coalesced_delta.base_version,
            coalesced_delta.source_format.as_str(),
        ),
        (true, have_version, ""),
        "the coalesced diff is based at the reconnector's version and names no format"
    );
    assert_eq!(
        coalesced_delta
            .set_fields
            .iter()
            .map(|f| (f.section_id, f.key_name.as_str()))
            .collect::<Vec<_>>(),
        vec![(2, "size")],
        "only the field written after have_version"
    );
    assert_eq!(
        coalesced_delta
            .time_samples
            .iter()
            .map(|ts| (ts.section_id, ts.time))
            .collect::<Vec<_>>(),
        vec![(2, 1.5)],
        "only the time sample written after have_version"
    );

    let text = live_edit::observability::metrics::test_util::render(&metrics);
    assert_eq!(
        (
            diff_case_count(&text, "full_content"),
            diff_case_count(&text, "incremental"),
        ),
        (1, 1),
        "each builder records one sample under its own case"
    );
    assert!(
        !text.contains("case=\"up_to_date\""),
        "no build emits the up_to_date case"
    );
    let durations: u64 = text
        .lines()
        .filter_map(|line| {
            line.strip_prefix("live_diff_build_duration_seconds_count{")?
                .rsplit_once(char::is_whitespace)
                .and_then(|(_, v)| v.trim().parse::<f64>().ok())
                .map(|v| v as u64)
        })
        .sum();
    assert_eq!(durations, 2, "one build-duration observation per build");
}

// =============================================================================
// B. Unit tests: build_diff_payload
// =============================================================================

// 7. FullContent returns valid OMPF with parseable FlatBuffer.

#[test]
fn build_diff_payload_full_content_returns_valid_ompf_and_flatbuffer() {
    let sections = vec![Section {
        section_id: 1,
        parent_id: 0,
        name: "Root".to_string(),
        section_type: 0,
        section_order: 0,
        first_version: 1,
        last_version: 3,
    }];
    let fields = vec![Field {
        section_id: 1,
        key_name: "color".to_string(),
        value: FieldValue::Inline(vec![0xFF, 0x00, 0x80]),
        set_order: 1,
        authoring_set_order: 1,
        version: 2,
    }];
    let time_samples = vec![TimeSample {
        section_id: 1,
        time: OrderedFloat(0.0),
        value: FieldValue::Inline(vec![0xAB]),
        version: 1,
    }];

    let plan = DiffPlan::FullContent {
        sections,
        fields,
        time_samples,
    };

    let payload = build_diff_payload(&plan, "usda", 0, &empty_scan(), &empty_ext_data())
        .expect("build_diff_payload must succeed for FullContent");
    assert!(
        !payload.is_empty(),
        "FullContent must produce non-empty payload"
    );

    // Parse OMPF container.
    let container = OmpfContainer::parse(&payload).expect("OMPF parse must succeed");
    assert!(
        !container.delta_bytes.is_empty(),
        "delta bytes must be non-empty"
    );

    // Parse OODF delta.
    let delta = parse_delta(container.delta_bytes).expect("OODF parse must succeed");

    assert!(delta.is_diff, "is_diff must be true for FullContent");
    assert_eq!(
        delta.base_version, 0,
        "base_version must be 0 for FullContent"
    );
    assert_eq!(
        delta.source_format, "usda",
        "source_format must be present for FullContent"
    );

    // Verify struct commands (sections).
    assert_eq!(delta.struct_commands.len(), 1, "one section command");

    // Verify fields.
    assert_eq!(delta.set_fields.len(), 1, "one field");
    assert_eq!(delta.set_fields[0].key_name, "color");

    // Verify time samples.
    assert_eq!(delta.time_samples.len(), 1, "one time sample");
    assert!((delta.time_samples[0].time - 0.0_f64).abs() < f64::EPSILON);
}

// 8. Incremental returns valid OMPF with baseVersion = have_version.

#[test]
fn build_diff_payload_incremental_returns_valid_ompf_with_correct_base_version() {
    let new_sections = vec![Section {
        section_id: 5,
        parent_id: 0,
        name: "NewPrim".to_string(),
        section_type: 1,
        section_order: 2,
        first_version: 8,
        last_version: 10,
    }];
    let modified_sections = vec![Section {
        section_id: 1,
        parent_id: 0,
        name: "ModPrim".to_string(),
        section_type: 0,
        section_order: 0,
        first_version: 1,
        last_version: 9,
    }];
    let deleted_sections = vec![(3u64, 1u64, 7u64)];
    let changed_fields = vec![Field {
        section_id: 1,
        key_name: "opacity".to_string(),
        value: FieldValue::Inline(vec![0x3F, 0x80]),
        set_order: 3,
        authoring_set_order: 3,
        version: 8,
    }];
    let changed_time_samples = vec![TimeSample {
        section_id: 5,
        time: OrderedFloat(7.25),
        value: FieldValue::Inline(vec![0xBE, 0xEF]),
        version: 9,
    }];

    let plan = DiffPlan::Incremental {
        new_sections,
        modified_sections,
        deleted_sections,
        changed_fields,
        changed_time_samples,
    };

    let have_version = 6u64;
    let payload = build_diff_payload(
        &plan,
        "usda",
        have_version,
        &empty_scan(),
        &empty_ext_data(),
    )
    .expect("build_diff_payload must succeed for Incremental");
    assert!(
        !payload.is_empty(),
        "Incremental must produce non-empty payload"
    );

    // Parse OMPF.
    let container = OmpfContainer::parse(&payload).expect("OMPF parse must succeed");

    // Parse OODF delta.
    let delta = parse_delta(container.delta_bytes).expect("OODF parse must succeed");

    assert!(delta.is_diff, "is_diff must be true for Incremental");
    assert_eq!(
        delta.base_version, have_version,
        "base_version must equal have_version for Incremental"
    );
    // source_format is empty for Incremental.
    assert!(
        delta.source_format.is_empty(),
        "source_format must be empty for Incremental"
    );

    // Struct commands: 2 change sections (new + modified) + 1 delete section = 3 entries.
    assert_eq!(
        delta.struct_commands.len(),
        3,
        "2 change + 1 delete = 3 struct commands"
    );

    // Fields.
    assert_eq!(delta.set_fields.len(), 1);
    assert_eq!(delta.set_fields[0].key_name, "opacity");

    // Time samples.
    assert_eq!(delta.time_samples.len(), 1);
    assert!((delta.time_samples[0].time - 7.25_f64).abs() < f64::EPSILON);
}

// 9. FullContent with empty collections produces valid OMPF with empty delta.

#[test]
fn build_diff_payload_full_content_empty_collections_returns_valid_ompf() {
    let plan = DiffPlan::FullContent {
        sections: vec![],
        fields: vec![],
        time_samples: vec![],
    };

    let payload = build_diff_payload(&plan, "usd", 0, &empty_scan(), &empty_ext_data())
        .expect("build_diff_payload must succeed for empty FullContent");
    assert!(
        !payload.is_empty(),
        "empty FullContent still produces OMPF wrapper"
    );

    let container = OmpfContainer::parse(&payload).expect("OMPF parse must succeed");
    let delta = parse_delta(container.delta_bytes).expect("parse must succeed");

    assert!(delta.is_diff);
    assert_eq!(delta.base_version, 0);
    assert_eq!(delta.source_format, "usd");
    assert!(
        delta.struct_commands.is_empty(),
        "no struct commands for empty"
    );
    assert!(delta.set_fields.is_empty(), "no fields for empty");
    assert!(delta.time_samples.is_empty(), "no time samples for empty");
}

// 10. Incremental with only deleted sections.

#[test]
fn build_diff_payload_incremental_deleted_sections_only() {
    let plan = DiffPlan::Incremental {
        new_sections: vec![],
        modified_sections: vec![],
        deleted_sections: vec![(10, 1, 5), (20, 2, 6)],
        changed_fields: vec![],
        changed_time_samples: vec![],
    };

    let payload = build_diff_payload(&plan, "usda", 3, &empty_scan(), &empty_ext_data())
        .expect("build_diff_payload must succeed for deleted-only Incremental");
    let container = OmpfContainer::parse(&payload).expect("OMPF parse must succeed");
    let delta = parse_delta(container.delta_bytes).expect("parse must succeed");

    assert!(delta.is_diff);
    assert_eq!(delta.base_version, 3);

    assert_eq!(delta.struct_commands.len(), 2, "two delete commands");
    for cmd in &delta.struct_commands {
        assert!(
            matches!(cmd, StructCommand::DiffDeleteSection(_)),
            "expected DiffDeleteSection, got {cmd:?}"
        );
    }

    assert!(delta.set_fields.is_empty());
    assert!(delta.time_samples.is_empty());
}
