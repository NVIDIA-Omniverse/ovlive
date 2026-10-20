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

// End-to-end integration tests for the legacy version diff
// (`compute_diff_at_cut`): create a layer via LayerStore, read it the way the
// service does (one atomic `read_object_snapshot`), then build the diff over
// that proven cut at various subscriber versions to exercise all three diff
// cases plus the NOT_FOUND error path and metrics recording.
//
// Tests run against a real Valkey instance via the TestValkey fixture.

use std::sync::Arc;

use ordered_float::OrderedFloat;

use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};
use live_edit::error::ServiceError;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::diff_read::{
    compute_diff_at_cut, read_object_snapshot, DiffCase, DiffResponse, ExtValueReads, SnapshotRead,
    SnapshotReadConsumer, SubscriberState,
};
use live_edit::storage::external_values::{ExternalValueStore, MockExternalValueStore};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{parse_delta, StructCommand};

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

/// Helper: create a session and return its ID.
async fn create_test_session(tv: &TestValkey) -> String {
    session_store(tv)
        .create("diff-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed")
}

/// Helper: create an isolated metrics instance for testing.
fn test_metrics() -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only("test"))
}

/// The legacy version diff of `object_id` for `subscriber`, built the way the
/// service builds it: one atomic read (`read_object_snapshot`, catch-up
/// consumer), then `compute_diff_at_cut` over its proven cut. A read error —
/// `NotFound` for an absent header — is returned as the build's.
async fn diff_at_cut(
    tv: &TestValkey,
    object_id: &str,
    subscriber: &SubscriberState,
    ext_store: &dyn ExternalValueStore,
    metrics: Option<&SharedMetrics>,
) -> Result<DiffResponse, ServiceError> {
    let snapshot = match read_object_snapshot(
        tv.client(),
        tv.key_schema(),
        object_id,
        SnapshotReadConsumer::Catchup,
    )
    .await?
    {
        SnapshotRead::Snapshot(snapshot) => snapshot,
        SnapshotRead::TooLarge(t) => panic!(
            "every fixture here sits far below the atomic-read limits; object {object_id} \
             answered TOO_LARGE {t:?}"
        ),
    };
    compute_diff_at_cut(
        &snapshot,
        subscriber,
        ext_store,
        ExtValueReads::recover_for_test(),
        metrics,
        "test",
    )
    .await
    .map(|built| {
        built
            .into_ready()
            .expect("a build with fetch-on-miss never answers Missing")
    })
}

/// Build an ObjectState with sections at various version levels, useful for
/// testing all three diff cases with a single stored layer.
///
/// Layout:
///   - sequence_num = 5
///   - Section 1 ("World"): first_version=1, last_version=3 (old, updated before v3)
///   - Section 2 ("Cube"):  first_version=4, last_version=5 (new after v3)
///   - Field "color" on section 1: version=2 (old)
///   - Field "opacity" on section 2: version=4 (new after v3)
///   - TimeSample at t=0.0 on section 1: version=1 (old)
///   - TimeSample at t=1.5 on section 2: version=5 (new after v3)
fn multi_version_state() -> ObjectState {
    let mut state = ObjectState {
        object_id: 0,
        sequence_num: 5,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 6,
        ..Default::default()
    };

    let sec1 = Section {
        section_id: 1,
        parent_id: 0,
        name: "World".to_string(),
        section_type: 1,
        section_order: 0,
        first_version: 1,
        last_version: 3,
    };
    let sec2 = Section {
        section_id: 2,
        parent_id: 1,
        name: "Cube".to_string(),
        section_type: 2,
        section_order: 1,
        first_version: 4,
        last_version: 5,
    };
    state.sections.insert(1, sec1);
    state.sections.insert(2, sec2);
    state.children.insert(0, vec![1]);
    state.children.insert(1, vec![2]);

    // Old field (version 2, will NOT appear in incremental diff with have_version=3).
    let f1 = Field {
        section_id: 1,
        key_name: "color".to_string(),
        value: FieldValue::Inline(vec![0xFF, 0x00, 0x80]),
        set_order: 1,
        authoring_set_order: 1,
        version: 2,
    };
    // New field (version 4, will appear in incremental diff with have_version=3).
    let f2 = Field {
        section_id: 2,
        key_name: "opacity".to_string(),
        value: FieldValue::Inline(vec![0x3F, 0x80]),
        set_order: 2,
        authoring_set_order: 2,
        version: 4,
    };
    state
        .fields
        .insert((f1.section_id, f1.key_name.clone()), f1);
    state
        .fields
        .insert((f2.section_id, f2.key_name.clone()), f2);

    // Old time sample (version 1).
    let ts1 = TimeSample {
        section_id: 1,
        time: OrderedFloat(0.0),
        value: FieldValue::Inline(vec![0xAA]),
        version: 1,
    };
    // New time sample (version 5).
    let ts2 = TimeSample {
        section_id: 2,
        time: OrderedFloat(1.5),
        value: FieldValue::Inline(vec![0xBB, 0xCC]),
        version: 5,
    };
    state.time_samples.insert((ts1.section_id, ts1.time), ts1);
    state.time_samples.insert((ts2.section_id, ts2.time), ts2);

    state
}

// =============================================================================
// Case 1: Full content (have_version=0)
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compute_diff_at_cut_have_version_zero_returns_full_content() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let state = multi_version_state();

    let oid = ls
        .create_layer(&session_id, "full-diff", "alice", &state, mock_ext_store())
        .await
        .expect("create_layer must succeed");

    let subscriber = SubscriberState {
        have_object_id: oid,
        have_version: 0,
    };

    let ext_store = mock_ext_store();
    let resp = diff_at_cut(&tv, &oid.to_string(), &subscriber, ext_store.as_ref(), None)
        .await
        .expect("the diff build must succeed");

    assert_eq!(
        resp.case,
        DiffCase::FullContent,
        "have_version=0 must be Case 1"
    );
    assert!(
        !resp.ompf_bytes.is_empty(),
        "full content must produce non-empty OMPF"
    );
    assert!(resp.sections_count >= 2, "must include at least 2 sections");

    // Parse OMPF and verify FlatBuffer delta.
    let container = OmpfContainer::parse(&resp.ompf_bytes).expect("OMPF parse must succeed");
    assert!(
        !container.delta_bytes.is_empty(),
        "delta bytes must be non-empty"
    );

    let delta = parse_delta(container.delta_bytes).expect("OODF parse must succeed");

    assert!(delta.is_diff, "is_diff must be true");
    assert_eq!(
        delta.base_version, 0,
        "base_version must be 0 for full content"
    );
    assert_eq!(
        delta.source_format, "usda",
        "source_format must be present for full content"
    );

    // All sections, fields, and time samples must be present.
    assert_eq!(
        delta.struct_commands.len(),
        2,
        "both sections must appear as struct commands"
    );
    assert_eq!(delta.set_fields.len(), 2, "both fields must be included");
    assert_eq!(
        delta.time_samples.len(),
        2,
        "both time samples must be included"
    );
}

// =============================================================================
// Case 2: Incremental (have_version between 0 and current)
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compute_diff_at_cut_incremental_includes_only_changes_since_have_version() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let state = multi_version_state();

    let oid = ls
        .create_layer(&session_id, "incr-diff", "alice", &state, mock_ext_store())
        .await
        .expect("create_layer must succeed");

    // Subscriber at version 3: should get section 2 (first_version=4), field "opacity"
    // (version=4), and time sample at t=1.5 (version=5). Section 1's last_version=3
    // which is not > have_version, so it should NOT appear as modified.
    let subscriber = SubscriberState {
        have_object_id: oid,
        have_version: 3,
    };

    let ext_store = mock_ext_store();
    let resp = diff_at_cut(&tv, &oid.to_string(), &subscriber, ext_store.as_ref(), None)
        .await
        .expect("the diff build must succeed");

    assert_eq!(
        resp.case,
        DiffCase::Incremental,
        "must be Case 2 incremental"
    );
    assert!(
        !resp.ompf_bytes.is_empty(),
        "incremental must produce non-empty OMPF"
    );

    // Parse the delta and verify only newer items are included.
    let container = OmpfContainer::parse(&resp.ompf_bytes).expect("OMPF parse must succeed");
    let delta = parse_delta(container.delta_bytes).expect("OODF parse must succeed");

    assert!(delta.is_diff, "is_diff must be true");
    assert_eq!(
        delta.base_version, 3,
        "base_version must equal have_version for incremental"
    );
    assert!(
        delta.source_format.is_empty(),
        "source_format must be empty for incremental"
    );

    // Only the new section (section 2, first_version=4 > have_version=3) should appear.
    assert_eq!(
        delta.struct_commands.len(),
        1,
        "only section 2 (new since v3) should be in struct commands"
    );

    // Only the new field ("opacity", version=4) should appear.
    assert_eq!(delta.set_fields.len(), 1, "only one field changed since v3");
    assert_eq!(delta.set_fields[0].key_name, "opacity");

    // Only the new time sample (version=5) should appear.
    assert_eq!(
        delta.time_samples.len(),
        1,
        "only one time sample changed since v3"
    );
    assert!((delta.time_samples[0].time - 1.5_f64).abs() < f64::EPSILON);
}

// =============================================================================
// Case 3: Up to date (have_version == current sequence_num)
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compute_diff_at_cut_up_to_date_returns_empty_ompf() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let state = multi_version_state();

    let oid = ls
        .create_layer(
            &session_id,
            "uptodate-diff",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // Subscriber at version 5 (== sequence_num) → up to date.
    let subscriber = SubscriberState {
        have_object_id: oid,
        have_version: 5,
    };

    let ext_store = mock_ext_store();
    let resp = diff_at_cut(&tv, &oid.to_string(), &subscriber, ext_store.as_ref(), None)
        .await
        .expect("the diff build must succeed");

    assert_eq!(resp.case, DiffCase::UpToDate, "must be Case 3 up_to_date");
    assert!(
        resp.ompf_bytes.is_empty(),
        "up_to_date must produce empty OMPF bytes"
    );
    assert_eq!(
        resp.sections_count, 0,
        "up_to_date must report zero sections"
    );
}

// =============================================================================
// Object not found
// =============================================================================

/// The catch-up read of an absent object — the `read_object_snapshot` inside
/// [`diff_at_cut`], before any build — ends `NotFound` naming the object id.
/// `compute_diff_at_cut` itself takes a snapshot already read, so it never
/// sees an absent object; this pins the read-then-build pipeline's error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn catch_up_read_of_a_nonexistent_object_returns_not_found_naming_the_id() {
    let tv = TestValkey::setup().await;

    let subscriber = SubscriberState {
        have_object_id: 0,
        have_version: 0,
    };

    let ext_store = mock_ext_store();
    let result = diff_at_cut(&tv, "999999999", &subscriber, ext_store.as_ref(), None).await;

    match result {
        Err(ServiceError::NotFound(msg)) => {
            assert!(
                msg.contains("999999999"),
                "error message should mention the object_id, got: {msg}"
            );
        }
        Err(other) => panic!("expected NotFound, got: {other:?}"),
        Ok(_) => panic!("expected NotFound error, got Ok"),
    }
}

// =============================================================================
// Metrics recorded
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compute_diff_at_cut_records_diff_case_total_metric() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let state = multi_version_state();
    let metrics = test_metrics();

    let oid = ls
        .create_layer(
            &session_id,
            "metrics-diff",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let oid_str = oid.to_string();
    let ext_store = mock_ext_store();

    // Case 1: full content (have_version=0).
    let sub_full = SubscriberState {
        have_object_id: oid,
        have_version: 0,
    };
    diff_at_cut(&tv, &oid_str, &sub_full, ext_store.as_ref(), Some(&metrics))
        .await
        .expect("the full diff build must succeed");

    // Case 2: incremental (have_version=3).
    let sub_incr = SubscriberState {
        have_object_id: oid,
        have_version: 3,
    };
    diff_at_cut(&tv, &oid_str, &sub_incr, ext_store.as_ref(), Some(&metrics))
        .await
        .expect("the incremental diff build must succeed");

    // Case 3: up to date (have_version=5).
    let sub_utd = SubscriberState {
        have_object_id: oid,
        have_version: 5,
    };
    diff_at_cut(&tv, &oid_str, &sub_utd, ext_store.as_ref(), Some(&metrics))
        .await
        .expect("the up-to-date diff build must succeed");

    // Verify diff_case_total counters.
    let full_count = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_diff_case_total",
        &[("case", "full_content"), ("service", "test")],
    )
    .unwrap_or(0.0) as u64;
    assert_eq!(full_count as u64, 1, "full_content counter must be 1");

    let incr_count = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_diff_case_total",
        &[("case", "incremental"), ("service", "test")],
    )
    .unwrap_or(0.0) as u64;
    assert_eq!(incr_count as u64, 1, "incremental counter must be 1");

    let utd_count = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_diff_case_total",
        &[("case", "up_to_date"), ("service", "test")],
    )
    .unwrap_or(0.0) as u64;
    assert_eq!(utd_count as u64, 1, "up_to_date counter must be 1");

    // Verify diff_build_duration_seconds histogram was observed. Sum the
    // per-case `_count` series across the rendered scrape exposition.
    let text = live_edit::observability::metrics::test_util::render(&metrics);
    let total_samples: u64 = text
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix("live_diff_build_duration_seconds_count")?;
            if !rest.starts_with('{') {
                return None;
            }
            line.rsplit_once(char::is_whitespace)
                .and_then(|(_, v)| v.trim().parse::<f64>().ok())
                .map(|v| v as u64)
        })
        .sum();
    assert_eq!(
        total_samples, 3,
        "histogram must have 3 observations (one per diff call)"
    );
}

// =============================================================================
// Full content via object_id mismatch (subscriber has stale object_id)
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compute_diff_at_cut_object_id_mismatch_returns_full_content() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let state = multi_version_state();

    let oid = ls
        .create_layer(
            &session_id,
            "mismatch-diff",
            "alice",
            &state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // Subscriber has a different object_id (simulating a layer overwrite).
    let subscriber = SubscriberState {
        have_object_id: oid + 999,
        have_version: 3,
    };

    let ext_store = mock_ext_store();
    let resp = diff_at_cut(&tv, &oid.to_string(), &subscriber, ext_store.as_ref(), None)
        .await
        .expect("the diff build must succeed");

    assert_eq!(
        resp.case,
        DiffCase::FullContent,
        "object_id mismatch must produce Case 1 full content"
    );
    assert!(
        !resp.ompf_bytes.is_empty(),
        "full content must produce non-empty OMPF"
    );
}

// =============================================================================
// Cold-start (Case 1) wire determinism — b.gcb / b.vz7
// =============================================================================
//
// `DiffEngine::plan_full_content` used to collect `fields` and `time_samples`
// straight out of `ObjectState`'s `HashMap`s, and to append orphan sections in
// `state.sections.values()` order. All three were `HashMap` iteration order,
// i.e. process-dependent: two replicas (or two calls in one process) serving the
// same stored state emitted the same SET semantically but in a different
// sequence, so the cold-start OMPF was byte-nondeterministic. That is
// harmless for an LWW consumer but it defeats byte-level diffing/caching and it
// makes the differential oracle (an internal acceptance harness) unusable on
// this path — the oracle is how this repo adjudicates delivery/parity claims.
//
// The damage reached the OMPF part table too: `resolve_external_values` builds
// `ExternalValueScan::ordered_hashes` by first-appearance over
// `plan.fields` THEN `plan.time_samples` (`collect_external_value_hashes`), and
// `build_diff_payload` emits one OMPF part per entry in that order. So the part
// sequence inherited the field/time-sample nondeterminism verbatim.
//
// The tests below are the wire-level proof, one behaviour each:
//   * byte identity across two independent read-and-build calls (delta bytes AND
//     the external-value part sequence),
//   * the emitted `SetField.set_order` shape,
//   * the section emission order (reachable DFS prefix, then orphan tail),
//   * the time-sample emission order.

/// Number of sections in [`deterministic_cold_start_state`] (2 reachable + 2 orphans).
const COLD_START_SECTION_COUNT: usize = 4;
/// Number of fields in [`deterministic_cold_start_state`].
const COLD_START_FIELD_COUNT: usize = 8;
/// Number of time samples in [`deterministic_cold_start_state`].
const COLD_START_TIME_SAMPLE_COUNT: usize = 4;
/// Number of DISTINCT external values referenced by [`deterministic_cold_start_state`],
/// hence the number of OMPF parts after Part 0 (the delta).
const COLD_START_EXT_PART_COUNT: usize = 4;

// External-value tags. Each tag seeds both the 32-byte content hash and the
// blob bytes, so a mis-ordered OMPF part table names itself in the failure
// message. H1/H2 first appear on FIELDS; H3/H4 first appear on TIME SAMPLES —
// see `deterministic_cold_start_ext_blobs` for why that split is load-bearing.
const EXT_TAG_H1: u8 = 0xA1;
const EXT_TAG_H2: u8 = 0xB2;
const EXT_TAG_H3: u8 = 0xC3;
const EXT_TAG_H4: u8 = 0xD4;

/// A distinct, deterministic 32-byte content hash per `tag`.
fn ext_hash(tag: u8) -> [u8; 32] {
    let mut hash = [0u8; 32];
    hash[0] = tag;
    hash[31] = !tag;
    hash
}

/// Blob bytes for `tag`. Distinct lengths as well as distinct contents so a
/// swapped part is unambiguous in a failure message.
fn ext_blob(tag: u8, len: usize) -> Vec<u8> {
    vec![tag; len]
}

/// The four external blobs [`deterministic_cold_start_state`] references, in the
/// first-appearance order the fix makes canonical: H1, H2 (from the sorted
/// fields) then H3, H4 (from the sorted time samples).
///
/// **H3 and H4 first appear on TIME SAMPLES, not on fields.** Without that the
/// part-order assertion would be vacuous: `collect_external_value_hashes` walks
/// fields and time samples into ONE shared pool, so a fixture whose every ext
/// ref hangs off a field would pass even if the time-sample emission order were
/// still `HashMap`-dependent (engineer seam E-1(i)).
fn deterministic_cold_start_ext_blobs() -> Vec<([u8; 32], Vec<u8>)> {
    vec![
        (ext_hash(EXT_TAG_H1), ext_blob(EXT_TAG_H1, 11)),
        (ext_hash(EXT_TAG_H2), ext_blob(EXT_TAG_H2, 13)),
        (ext_hash(EXT_TAG_H3), ext_blob(EXT_TAG_H3, 17)),
        (ext_hash(EXT_TAG_H4), ext_blob(EXT_TAG_H4, 19)),
    ]
}

/// Build a `FieldValue::External` for `tag` with the size the blob actually has.
/// `ext_value_index` is recomputed at emit time from the scan, so `0` here is
/// the same placeholder `decode_field_value` produces on the read path.
fn ext_field_value(tag: u8, size: u64) -> FieldValue {
    FieldValue::External {
        hash: ext_hash(tag),
        size,
        ext_value_index: 0,
    }
}

/// Non-vacuous cold-start fixture for the b.gcb/b.vz7 determinism proof.
///
/// Every condition below exists to make one of the assertions below bite; none
/// is decoration.
///
/// **Fields tie on the leading sort key, across AND within sections.** The field
/// comparator is `(version, authoring_set_order, section_id, key_name)`
/// (`DiffEngine::sort_fields_by_authoring`, seam E-2). A fixture whose fields
/// all carry distinct `version`s would never reach the tiebreaks, so the sort
/// would look total while the `HashMap` order still leaked through them. Here:
///   * `alpha`/`beta` tie on `(version=1, authoring_set_order=0)` and are broken
///     by `section_id` (1 < 2) — a tie ACROSS sections;
///   * `delta`/`epsilon` tie on `(version=2, authoring_set_order=0)`, broken by
///     `section_id` (2 < 5) — a second cross-section tie, on an orphan section;
///   * `aardvark`/`eta` tie on `(version=3, authoring_set_order=0, section_id=1)`
///     and are broken by `key_name` alone — a tie WITHIN one section, the only
///     shape that exercises the final tiebreak.
///
/// **Two orphan sections.** 5 and 7 sit in the section index but in no children
/// list (their `parent_id`s 98/99 do not exist), so `sections_in_dfs_order`
/// appends them after the reachable DFS prefix. Pre-fix that append walked
/// `state.sections.values()`; it is now `section_id`-sorted (seam E-1(ii)). One
/// orphan could not distinguish "sorted" from "happened to come out that way".
///
/// **Four distinct external values, two of them first seen on time samples.**
/// See [`deterministic_cold_start_ext_blobs`]. `epsilon` re-references H1 and
/// the `t=0.5` sample re-references H2, so the first-appearance dedupe is
/// exercised rather than assumed.
///
/// **`set_order` values are deliberately in the 900s and NOT ascending in emit
/// order.** `eta` (`set_order = 906`) sorts AFTER `aardvark` (`set_order = 907`),
/// so the emitted sequence is `[900, 901, 902, 903, 904, 905, 907, 906]` — a
/// shape neither a dense `0..N-1` reindex nor a plain `set_order` sort could
/// produce.
fn deterministic_cold_start_state() -> ObjectState {
    let mut state = ObjectState {
        object_id: 0,
        sequence_num: 3,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 908,
        ..Default::default()
    };

    // -- Sections: 1 -> 2 reachable from root; 5 and 7 orphaned. --
    let sections = [
        // (section_id, parent_id, name, type, order, first_ver, last_ver)
        (1_u64, 0_u64, "World", 1_u8, 0_u32, 1_u64, 3_u64),
        (2, 1, "Cube", 2, 1, 1, 2),
        // Orphans: parent ids 98/99 are not sections and hold no children list.
        (7, 99, "OrphanLate", 4, 1, 2, 2),
        (5, 98, "OrphanEarly", 3, 0, 2, 2),
    ];
    for (section_id, parent_id, name, section_type, section_order, first_version, last_version) in
        sections
    {
        state.sections.insert(
            section_id,
            Section {
                section_id,
                parent_id,
                name: name.to_string(),
                section_type,
                section_order,
                first_version,
                last_version,
            },
        );
    }
    state.children.insert(0, vec![1]);
    state.children.insert(1, vec![2]);

    // -- Fields. (section_id, key_name, version, authoring_set_order, set_order, value) --
    let fields = [
        (
            1_u64,
            "alpha",
            1_u64,
            0_u64,
            900_u64,
            ext_field_value(EXT_TAG_H1, 11),
        ),
        (2, "beta", 1, 0, 901, FieldValue::Inline(vec![0x01])),
        (1, "gamma", 1, 1, 902, ext_field_value(EXT_TAG_H2, 13)),
        (2, "delta", 2, 0, 903, FieldValue::Inline(vec![0x02])),
        // Re-references H1: exercises first-appearance dedupe.
        (5, "epsilon", 2, 0, 904, ext_field_value(EXT_TAG_H1, 11)),
        (7, "zeta", 2, 3, 905, FieldValue::Inline(vec![0x03])),
        (1, "eta", 3, 0, 906, FieldValue::Inline(vec![0x04])),
        (1, "aardvark", 3, 0, 907, FieldValue::Inline(vec![0x05])),
    ];
    for (section_id, key_name, version, authoring_set_order, set_order, value) in fields {
        state.fields.insert(
            (section_id, key_name.to_string()),
            Field {
                section_id,
                key_name: key_name.to_string(),
                value,
                set_order,
                authoring_set_order,
                version,
            },
        );
    }

    // -- Time samples. (section_id, time, version, value) --
    let time_samples = [
        // H3's ONLY appearance anywhere is here.
        (2_u64, 1.5_f64, 1_u64, ext_field_value(EXT_TAG_H3, 17)),
        // Re-references H2 (already seen on `gamma`): dedupe again.
        (1, 0.5, 2, ext_field_value(EXT_TAG_H2, 13)),
        (1, 0.25, 2, FieldValue::Inline(vec![0x06])),
        // H4's ONLY appearance anywhere is here.
        (5, 0.75, 2, ext_field_value(EXT_TAG_H4, 19)),
    ];
    for (section_id, time, version, value) in time_samples {
        state.time_samples.insert(
            (section_id, OrderedFloat(time)),
            TimeSample {
                section_id,
                time: OrderedFloat(time),
                value,
                version,
            },
        );
    }

    state
}

/// Create a session, materialize [`deterministic_cold_start_state`] as a layer,
/// and return the new `object_id` plus an external-value store pre-seeded with
/// the four blobs the state references.
///
/// `create_layer`'s `ext_value_store` argument is only used to release the
/// PREVIOUS generation's blobs on an overwrite — it never stores the new state's
/// values — so the store handed to the diff build has to be seeded explicitly or
/// `build_diff_payload` fails closed with `Internal("external value missing")`.
async fn seed_cold_start_layer(
    tv: &TestValkey,
    layer_name: &str,
) -> (u64, Arc<MockExternalValueStore>) {
    let ls = layer_store(tv);
    let session_id = create_test_session(tv).await;
    let state = deterministic_cold_start_state();

    let oid = ls
        .create_layer(&session_id, layer_name, "alice", &state, mock_ext_store())
        .await
        .expect("create_layer must succeed");

    let ext_store = mock_ext_store();
    let blobs = deterministic_cold_start_ext_blobs();
    let pairs: Vec<(&[u8; 32], &[u8])> = blobs.iter().map(|(h, d)| (h, d.as_slice())).collect();
    ext_store
        .store_values(&pairs)
        .await
        .expect("seeding external values must succeed");

    (oid, ext_store)
}

/// Run one cold-start read-and-build against `oid`.
///
/// Cold-start arm: the object's REAL `object_id` with `have_version = 0` — the
/// same arm the neighbouring `compute_diff_at_cut_have_version_zero_returns_full_content`
/// uses. `DiffEngine::plan` routes `have_object_id != state.object_id ||
/// have_version == 0` to `plan_full_content`, so the `have_object_id = 0` arm
/// reaches the identical code; matching the neighbour keeps the two cold-start
/// tests describing one scenario.
async fn cold_start_diff(
    tv: &TestValkey,
    oid: u64,
    ext_store: &MockExternalValueStore,
) -> live_edit::storage::diff_read::DiffResponse {
    let subscriber = SubscriberState {
        have_object_id: oid,
        have_version: 0,
    };
    diff_at_cut(tv, &oid.to_string(), &subscriber, ext_store, None)
        .await
        .expect("the diff build must succeed")
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compute_diff_at_cut_cold_start_is_byte_identical_across_repeated_calls() {
    let tv = TestValkey::setup().await;
    let (oid, ext_store) = seed_cold_start_layer(&tv, "determinism-diff").await;

    // Two INDEPENDENT calls, not one call cloned. Each call re-runs
    // `read_object_snapshot` -> `snapshot_to_object_state`, which builds fresh
    // `HashMap`s; `RandomState::new()` bumps its thread-local seed on every
    // instantiation, so the two calls' maps hash and iterate differently in
    // exactly the way two replicas' maps do (measured directly against this
    // fixture: two `snapshot_to_object_state` results over the same stored state
    // iterate their time-sample maps in different orders). Pre-fix, that is what
    // leaked into the wire. The divergence is not ASSERTED because two draws can
    // legitimately coincide — asserting it would be the flaky half of the claim.
    let first = cold_start_diff(&tv, oid, ext_store.as_ref()).await;
    let second = cold_start_diff(&tv, oid, ext_store.as_ref()).await;

    assert_eq!(
        first.case,
        DiffCase::FullContent,
        "have_version=0 must select the cold-start FullContent arm"
    );
    assert_eq!(
        second.case,
        DiffCase::FullContent,
        "the second call must select the same arm"
    );

    // Non-vacuity guard: the byte comparison below is only meaningful if the
    // payload actually carries the ties / orphans / ext refs the fixture builds.
    let container_a = OmpfContainer::parse(&first.ompf_bytes).expect("OMPF parse must succeed");
    let delta_a = parse_delta(container_a.delta_bytes).expect("OODF parse must succeed");
    assert_eq!(
        delta_a.struct_commands.len(),
        COLD_START_SECTION_COUNT,
        "fixture must emit 2 reachable + 2 orphan sections"
    );
    assert_eq!(
        delta_a.set_fields.len(),
        COLD_START_FIELD_COUNT,
        "fixture must emit all 8 fields"
    );
    assert_eq!(
        delta_a.time_samples.len(),
        COLD_START_TIME_SAMPLE_COUNT,
        "fixture must emit all 4 time samples"
    );
    assert_eq!(
        container_a.external_values.len(),
        COLD_START_EXT_PART_COUNT,
        "fixture must emit 4 distinct external-value parts"
    );

    // The delta bytes.
    assert_eq!(
        first.ompf_bytes.len(),
        second.ompf_bytes.len(),
        "two cold-start diffs over the same stored state must be the same length"
    );
    assert_eq!(
        first.ompf_bytes, second.ompf_bytes,
        "two cold-start diffs over the same stored state must be byte-identical"
    );

    // The OMPF external-value PART sequence is a distinct observable from the
    // delta bytes: it comes from `ExternalValueScan::ordered_hashes`, not from
    // the FlatBuffer. Compare it part-for-part.
    let container_b = OmpfContainer::parse(&second.ompf_bytes).expect("OMPF parse must succeed");
    assert_eq!(
        container_a.external_values.len(),
        container_b.external_values.len(),
        "both payloads must carry the same number of external-value parts"
    );
    for (idx, (part_a, part_b)) in container_a
        .external_values
        .iter()
        .zip(container_b.external_values.iter())
        .enumerate()
    {
        assert_eq!(
            part_a,
            part_b,
            "OMPF external-value part {} differs between calls: {:?} vs {:?}",
            idx + 1,
            part_a.first(),
            part_b.first()
        );
    }

    // Nothing on `DiffResponse` is timing- or timestamp-derived, so no field had
    // to be excluded from the comparison above: it carries `ompf_bytes`, `case`,
    // `sections_count` and `cut` only (the cut is the read's header `seq`, the
    // same for both reads of an idle layer). The one wall-clock value the build
    // computes (`start.elapsed()`) is fed to the
    // `diff_build_duration_seconds` histogram and never reaches the response.
    assert_eq!(
        first.sections_count, second.sections_count,
        "sections_count is derived from the plan and must match"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compute_diff_at_cut_cold_start_emits_cumulative_set_order_not_dense_reindex() {
    let tv = TestValkey::setup().await;
    let (oid, ext_store) = seed_cold_start_layer(&tv, "set-order-diff").await;

    let resp = cold_start_diff(&tv, oid, ext_store.as_ref()).await;
    let container = OmpfContainer::parse(&resp.ompf_bytes).expect("OMPF parse must succeed");
    let delta = parse_delta(container.delta_bytes).expect("OODF parse must succeed");

    // The DECIDED shape (b.vz7's DECISION block; Task t2.dfh.be.q7, engineer
    // seam E-3): `plan_full_content` is SORT-ONLY. It orders fields by
    // `(version, authoring_set_order, section_id, key_name)` and carries the
    // CUMULATIVE `set_order` off `ObjectState` verbatim — it deliberately does
    // NOT reuse `DiffEngine::dense_reindex_by_authoring`, which overwrites
    // `set_order` with the dense emit index `0..N-1`.
    //
    // This cold-start wire is the ONE place where a dense reindex would be
    // observable, which is why the pin lives here. `src/domain/diff.rs`
    // documents the same contract on `plan_full_content` ("SORT ONLY — the
    // cumulative `set_order` is preserved verbatim") and the domain-level pin
    // `plan_full_content_unchanged_for_representative_state` asserts it on the
    // Rust side; this is the wire-level half.
    let observed: Vec<(&str, u64)> = delta
        .set_fields
        .iter()
        .map(|f| (f.key_name.as_str(), f.set_order))
        .collect();
    let expected: Vec<(&str, u64)> = vec![
        // (version=1, aso=0, section=1)
        ("alpha", 900),
        // (version=1, aso=0, section=2) — cross-section tie broken by section_id
        ("beta", 901),
        // (version=1, aso=1, section=1)
        ("gamma", 902),
        // (version=2, aso=0, section=2)
        ("delta", 903),
        // (version=2, aso=0, section=5) — cross-section tie broken by section_id
        ("epsilon", 904),
        // (version=2, aso=3, section=7)
        ("zeta", 905),
        // (version=3, aso=0, section=1) — within-section tie broken by key_name,
        // which is why 907 lands ahead of 906.
        ("aardvark", 907),
        ("eta", 906),
    ];
    assert_eq!(
        observed, expected,
        "cold-start wire must emit fields in canonical authoring order carrying \
         the cumulative set_order verbatim"
    );

    // Explicit negative on the rejected alternative, so a future dense-reindex
    // reuse fails here with a message that names it.
    let emitted_set_orders: Vec<u64> = delta.set_fields.iter().map(|f| f.set_order).collect();
    let dense: Vec<u64> = (0..delta.set_fields.len() as u64).collect();
    assert_ne!(
        emitted_set_orders, dense,
        "cold-start wire must NOT dense-reindex set_order to 0..N-1 — that is a \
         wire-behaviour change beyond the sort and was adjudicated against"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compute_diff_at_cut_cold_start_emits_orphan_sections_after_reachable_dfs_in_id_order() {
    let tv = TestValkey::setup().await;
    let (oid, ext_store) = seed_cold_start_layer(&tv, "orphan-order-diff").await;

    let resp = cold_start_diff(&tv, oid, ext_store.as_ref()).await;
    let container = OmpfContainer::parse(&resp.ompf_bytes).expect("OMPF parse must succeed");
    let delta = parse_delta(container.delta_bytes).expect("OODF parse must succeed");

    let observed: Vec<u64> = delta
        .struct_commands
        .iter()
        .map(|cmd| match cmd {
            StructCommand::DiffChangeSection(c) => c.section_id,
            other => {
                panic!("cold-start full content must emit only DiffChangeSection, got {other:?}")
            }
        })
        .collect();

    // Reachable DFS prefix (1 -> 2) is unchanged by b.gcb — it walks
    // `state.children`'s authored `Vec`s and was always deterministic (seam
    // E-5). The orphan TAIL is what gained an order: sections 5 and 7 are in the
    // section index but in no children list, and are now emitted
    // `section_id`-ascending rather than in `state.sections.values()` order.
    assert_eq!(
        observed,
        vec![1, 2, 5, 7],
        "cold-start wire must emit the reachable DFS prefix first, then orphans \
         in section_id order"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn compute_diff_at_cut_cold_start_emits_time_samples_in_version_section_time_order() {
    let tv = TestValkey::setup().await;
    let (oid, ext_store) = seed_cold_start_layer(&tv, "ts-order-diff").await;

    let resp = cold_start_diff(&tv, oid, ext_store.as_ref()).await;
    let container = OmpfContainer::parse(&resp.ompf_bytes).expect("OMPF parse must succeed");
    let delta = parse_delta(container.delta_bytes).expect("OODF parse must succeed");

    // `(version, section_id, time)` — the key both faithful builders already use
    // and a total order over `ObjectState::time_samples`'s `(section_id, time)`
    // map key (seam E-4). All three of the version-2 samples tie on the leading
    // key, so section_id then time are what actually order them.
    let observed: Vec<(u64, f64)> = delta
        .time_samples
        .iter()
        .map(|ts| (ts.section_id, ts.time))
        .collect();
    let expected: [(u64, f64); COLD_START_TIME_SAMPLE_COUNT] = [
        (2, 1.5),  // version 1
        (1, 0.25), // version 2, lowest section then lowest time
        (1, 0.5),  // version 2, same section, later time
        (5, 0.75), // version 2, higher section
    ];
    assert_eq!(
        observed.len(),
        expected.len(),
        "all 4 time samples must be emitted"
    );
    for (idx, ((sec, time), (exp_sec, exp_time))) in
        observed.iter().zip(expected.iter()).enumerate()
    {
        assert_eq!(
            sec, exp_sec,
            "time sample {idx} must belong to section {exp_sec}"
        );
        assert!(
            (time - exp_time).abs() < f64::EPSILON,
            "time sample {idx} must carry time {exp_time}, got {time}"
        );
    }
}
