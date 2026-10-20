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

// b.qpf — an external value's `ext_meta` `ref_count` must be released when a
// commit SUPERSEDES it, not only when the whole object retires.
//
// Pre-fix, `schedule_release` had exactly two production callers, both
// object-level retirement paths, and each computed its release set from the
// object's CURRENT values. So every superseded external value leaked its
// reference permanently: a field rewritten from hash A to hash B, or from an
// external value to an inline one, left A's `ref_count` contribution with
// nothing that would ever decrement it, and re-writing the SAME hash paid a
// second pre-flight `+1` against one eventual `-1`. The blob and its `ext:meta`
// were unreclaimable for the life of the store.
//
// ── The counting unit this file asserts ──────────────────────────────────────
//
// `ref_count` counts **live objects referencing the hash** — one per object per
// distinct hash, however many of that object's rows name it. Three mechanisms
// carry it, and every test here reaches at least one of them:
//
//   * `le:{object_id}:obj:ext` — a per-object HASH mapping `hash_hex` → number
//     of value rows currently holding it, maintained INSIDE the commit's atomic
//     unit by both backends and rebuilt absolutely by `write_object`. It is what
//     lets a commit decide locally whether the object has actually given the
//     hash up.
//   * Two commit-derived release reasons. `rewrite` — the object's last row
//     holding the hash gave it up. `surplus_acquire` — the commit referenced a
//     hash the object ALREADY held, so the caller's pre-flight `store_values`
//     `+1` bought nothing and is handed straight back.
//   * Identity-carrying pending members, `{hash_hex}:{object_id}:{event}`.
//     A counted decrement cannot ride a Set that collapses equal members: with
//     the pre-fix bare `{hash_hex}` member, two objects retiring before a GC
//     tick enqueued ONE member for TWO owed decrements and the second reference
//     leaked. Naming the event restores exactly-once PER EVENT while keeping
//     SADD idempotent WITHIN an event, which is what `delete_layer`'s retry and
//     the `ext_gc_scheduled` guard depend on.
//
// ── Why the pre-flight store is staged by hand ───────────────────────────────
//
// In production `src/grpc/layer.rs` calls `store_external_values` (one `+1` per
// DISTINCT hash the delta references) BEFORE handing the delta to
// `LayerStore::update_layer` / `create_layer`. These tests drive the storage
// layer directly, so each one stages that pre-flight `store_values` itself —
// see `pre_store`. Omitting it would leave `ref_count` at 0 and make every
// assertion below vacuous.
//
// ── Backend parity ───────────────────────────────────────────────────────────
//
// The rule has three transcriptions — `plan_ext_index` (the definition), the
// txn slow path's caller, and `scripts/update_layer_fast.lua` step 6b — so a
// fix in one is not a fix. Every behavioural arm that a fast-path delta can
// reach runs on BOTH backends as two `#[tokio::test]` wrappers over one body,
// following the b.zvd shape: kept as two tests rather than a loop so nextest
// isolates them and a failure names the backend.
//
// Two arms are structurally single-backend and say so in their own doc
// comments: the section-delete cascade (the Lua encoder emits only `SF` / `ST`
// records, so `DeleteField` / `DeleteTimeSample` are reachable on the slow path
// only) and the `WRONGTYPE obj:ext` refusal (the script's own validation
// pre-pass, which has no txn counterpart).
//
// ── Row-kind parity ──────────────────────────────────────────────────────────
//
// Backend parity is not the only axis the rule is transcribed along. FIELDS and
// TIME SAMPLES are separate arms at BOTH commit sites — `update_layer_fast.lua`
// has an `"SF"` arm and an `"ST"` arm, each with its own `HGET <row> hash`
// pre-image read, and `ext_rows_written` has a `SetField` arm and a
// `SetTimeSample` arm, each computing its own row key — so a time-sample
// transcription can be deleted while every field test stays green. The two
// rewrite transitions therefore run over [`RowKind`] as well as over the
// backend: arms 1 and 2 are four wrappers each.
//
// The other arms already reach the time-sample sites for their own reasons and
// are not multiplied: arm 7 drives the `ST` ACQUIRE arm (a time-sample row in
// the three-row write) and arm 9 drives `DeleteTimeSample` through the
// section-delete cascade.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface,
    StreamsInterface,
};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::{ObjectState, Section};
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_sum, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::{
    process_pending_gc, ExtRelease, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::is_fast_path;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandDeleteSection, OwnedDelta, SetField, StructCommand, TimeSample};

use crate::common::owned_delta::{
    delta_with_set_field_external, delta_with_set_field_inline, delta_with_time_sample_external,
    delta_with_time_sample_inline,
};

// ═════════════════════════════════════════════════════════════════════════════
// Contract constants
// ═════════════════════════════════════════════════════════════════════════════

/// The section every value row in this file hangs off.
pub(crate) const SECTION_ID: u64 = 1;
/// The primary field row.
pub(crate) const FIELD_A: &str = "color";
/// The second field row, used by the shared-hash and hash-move arms.
const FIELD_B: &str = "roughness";
/// The time-sample row's time.
const SAMPLE_TIME: f64 = 2.5;
/// Every external value stored here is this many bytes wide, per the row's
/// `size` column. The value is arbitrary; the rows only need to be decodable.
pub(crate) const EXT_SIZE: u64 = 1_024;

/// The four `reason` cells of `live_edit_ext_ref_released_total`.
///
/// Restated rather than imported: the label values are the operator-facing
/// contract and a test that read them from the same `const` the emitter reads
/// could not fail when one was renamed.
pub(crate) const REASON_REWRITE: &str = "rewrite";
/// See [`REASON_REWRITE`].
pub(crate) const REASON_SURPLUS: &str = "surplus_acquire";
/// See [`REASON_REWRITE`].
pub(crate) const REASON_RETIREMENT: &str = "retirement";
/// See [`REASON_REWRITE`]. Added by b.yda: a reference handed back out of band
/// by `storage::ext_reconcile`, which cannot attribute it to any object. No
/// commit or retirement in this file can produce one, which is exactly why the
/// still-cell assertions below sweep it too — a build that ticked the wrong cell
/// would satisfy a three-cell check.
pub(crate) const REASON_RECONCILE: &str = "reconcile";

/// Every cell of `live_edit_ext_ref_released_total`, for the arms that assert
/// NOTHING was released.
pub(crate) const ALL_RELEASE_REASONS: [&str; 4] = [
    REASON_REWRITE,
    REASON_SURPLUS,
    REASON_RETIREMENT,
    REASON_RECONCILE,
];

/// Budget for joining the spawned overwrite-cleanup tracker. Generous; the
/// healthy path finishes in milliseconds.
const CLEANUP_JOIN_BUDGET: Duration = Duration::from_secs(10);

// ═════════════════════════════════════════════════════════════════════════════
// Fixture helpers
// ═════════════════════════════════════════════════════════════════════════════

/// Which value-row shape a rewrite arm drives.
///
/// The row kind is a real coverage axis, not a cosmetic one: the release rule is
/// transcribed once per row kind at each of the two commit sites (see the
/// "Row-kind parity" note at the top of this file), so a body that only ever
/// wrote a FIELD row would leave both time-sample transcriptions deletable with
/// the suite green.
///
/// The two rows this enum selects are also the two the object can hold
/// simultaneously without interfering: they live in different Valkey keys
/// (`obj:fld:{sid}:{key}` vs `obj:ts:{sid}:{time}`) and are counted separately
/// by `obj:ext`, so an arm parameterised over the kind asserts exactly the same
/// index shape either way.
#[derive(Clone, Copy, Debug)]
enum RowKind {
    Field,
    TimeSample,
}

impl RowKind {
    /// Discriminator for session / layer names and assertion context, so a
    /// failure names the row shape as well as the backend.
    fn label(self) -> &'static str {
        match self {
            RowKind::Field => "field",
            RowKind::TimeSample => "ts",
        }
    }

    /// A one-row delta writing this kind with an EXTERNAL value.
    ///
    /// `set_order` is a field-only concept — time samples are ordered by their
    /// `time` and carry no authoring rank — so it is accepted and ignored on the
    /// time-sample arm rather than being pushed onto every caller.
    fn external_delta(self, hash_byte: u8, set_order: u64) -> OwnedDelta {
        match self {
            RowKind::Field => delta_with_set_field_external(
                SECTION_ID,
                FIELD_A,
                ext_hash_vec(hash_byte),
                EXT_SIZE,
                1,
                set_order,
            ),
            RowKind::TimeSample => delta_with_time_sample_external(
                SECTION_ID,
                SAMPLE_TIME,
                ext_hash_vec(hash_byte),
                EXT_SIZE,
                1,
            ),
        }
    }

    /// A one-row delta writing this kind with an INLINE value — the same row the
    /// [`RowKind::external_delta`] above writes, so the pair is a rewrite of one
    /// row rather than two rows in different shapes.
    fn inline_delta(self, value: Vec<u8>, set_order: u64) -> OwnedDelta {
        match self {
            RowKind::Field => delta_with_set_field_inline(SECTION_ID, FIELD_A, value, set_order),
            RowKind::TimeSample => delta_with_time_sample_inline(SECTION_ID, SAMPLE_TIME, value),
        }
    }
}

/// A distinct 32-byte content hash. The byte makes two hashes tellable apart,
/// which is what turns an external→external rewrite into an assertable claim.
pub(crate) fn ext_hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

/// The same hash in the `Vec<u8>` shape `OwnedDelta` carries.
pub(crate) fn ext_hash_vec(byte: u8) -> Vec<u8> {
    vec![byte; 32]
}

/// The hex spelling both the value row's `hash` column and the `obj:ext` field
/// name use.
pub(crate) fn ext_hash_hex(byte: u8) -> String {
    sha256_to_hex(&ext_hash(byte))
}

/// The real Valkey-backed store, so `ref_count` and the pending set are the
/// production keys rather than a mock's in-memory maps.
///
/// The mock is deliberately NOT used anywhere in this file. Commit-derived
/// releases are SADDed to Valkey directly by the commit paths (a `LayerStore`
/// holds no external-value store), while retirement releases go through the
/// injected `ExternalValueStore` — so a mock would make half the arms
/// unobservable and the other half observable somewhere else. One real store
/// keeps every release readable from one place.
pub(crate) fn ext_store(tv: &TestValkey) -> Arc<ValkeyExternalValueStore> {
    Arc::new(ValkeyExternalValueStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ))
}

pub(crate) fn layer_store(
    tv: &TestValkey,
    backend: UpdateLayerBackend,
    metrics: &Arc<Metrics>,
) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend)
        .with_metrics(metrics.clone())
}

pub(crate) async fn create_session(tv: &TestValkey, name: &str) -> String {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, &format!("/{name}"), "b-qpf-user")
        .await
        .expect("session creation must succeed")
}

/// The shape `create_layer` materializes for a fresh layer whose content
/// arrives by `update_layer`.
pub(crate) fn empty_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        next_set_order: 0,
        ..Default::default()
    }
}

/// [`empty_state`] plus one root-child section.
///
/// Needed only by the section-delete arm: `cascade_delete_section` walks
/// `state.sections`, so a `DeleteSection` against a section the object never
/// had emits no `DeleteField` / `DeleteTimeSample` at all and the test would
/// pass having driven nothing.
fn state_with_section() -> ObjectState {
    let mut state = empty_state();
    state.sections.insert(
        SECTION_ID,
        Section {
            section_id: SECTION_ID,
            parent_id: 0,
            name: "World".to_owned(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        },
    );
    state.children.insert(0, vec![SECTION_ID]);
    state
}

/// A delta setting BOTH field rows to the same external hash.
///
/// Not composable from the shared single-row helpers, and the arm it serves —
/// the over-release canary — depends on both rows landing in ONE commit: two
/// separate commits would make the second one a `surplus_acquire` and change
/// what is being tested.
fn delta_with_two_fields_sharing_one_external(hash: Vec<u8>) -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![
            SetField {
                section_id: SECTION_ID,
                key_name: FIELD_A.to_owned(),
                value_or_ext_hash: hash.clone(),
                ext_value_size: EXT_SIZE,
                ext_value_index: 1,
                set_order: 0,
            },
            SetField {
                section_id: SECTION_ID,
                key_name: FIELD_B.to_owned(),
                value_or_ext_hash: hash,
                ext_value_size: EXT_SIZE,
                ext_value_index: 1,
                set_order: 1,
            },
        ],
        ..Default::default()
    }
}

/// A delta that moves a hash from row A to row B within ONE commit: A becomes
/// inline, B takes the hash.
fn delta_moving_the_hash_from_a_to_b(hash: Vec<u8>) -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![
            SetField {
                section_id: SECTION_ID,
                key_name: FIELD_A.to_owned(),
                value_or_ext_hash: vec![0x11, 0x22],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            },
            SetField {
                section_id: SECTION_ID,
                key_name: FIELD_B.to_owned(),
                value_or_ext_hash: hash,
                ext_value_size: EXT_SIZE,
                ext_value_index: 1,
                set_order: 1,
            },
        ],
        ..Default::default()
    }
}

/// One external field and one external time sample, in the same section, in one
/// commit — the fixture the section-delete cascade needs so BOTH the
/// `DeleteField` and the `DeleteTimeSample` arms are driven.
fn delta_with_external_field_and_time_sample(field_hash: Vec<u8>, ts_hash: Vec<u8>) -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: FIELD_A.to_owned(),
            value_or_ext_hash: field_hash,
            ext_value_size: EXT_SIZE,
            ext_value_index: 1,
            set_order: 0,
        }],
        time_samples: vec![TimeSample {
            section_id: SECTION_ID,
            time: SAMPLE_TIME,
            value_or_ext_hash: ts_hash,
            ext_value_size: EXT_SIZE,
            ext_value_index: 2,
        }],
        ..Default::default()
    }
}

/// A delta setting the two field rows to two DIFFERENT external hashes, in one
/// commit.
///
/// The fixture the enqueue-failure arms need: two distinct hashes held by one
/// object, so the commit that supersedes both owes exactly TWO references and a
/// counter ticking per CALL rather than per REFERENCE reads 1 where it must
/// read 2.
fn delta_with_two_fields_holding_distinct_externals(first: Vec<u8>, second: Vec<u8>) -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![
            SetField {
                section_id: SECTION_ID,
                key_name: FIELD_A.to_owned(),
                value_or_ext_hash: first,
                ext_value_size: EXT_SIZE,
                ext_value_index: 1,
                set_order: 0,
            },
            SetField {
                section_id: SECTION_ID,
                key_name: FIELD_B.to_owned(),
                value_or_ext_hash: second,
                ext_value_size: EXT_SIZE,
                ext_value_index: 2,
                set_order: 1,
            },
        ],
        ..Default::default()
    }
}

/// A delta rewriting BOTH field rows to inline values in ONE commit — the twin
/// of [`delta_with_two_fields_holding_distinct_externals`], superseding both
/// hashes at once so the whole owed batch rides a single enqueue.
fn delta_with_two_inline_fields(set_order_base: u64) -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![
            SetField {
                section_id: SECTION_ID,
                key_name: FIELD_A.to_owned(),
                value_or_ext_hash: vec![0x51],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: set_order_base,
            },
            SetField {
                section_id: SECTION_ID,
                key_name: FIELD_B.to_owned(),
                value_or_ext_hash: vec![0x52],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: set_order_base + 1,
            },
        ],
        ..Default::default()
    }
}

/// Stage the pre-flight `store_values` `+1` the gRPC layer pays before it
/// dispatches a delta referencing `hash` — one increment per DISTINCT hash per
/// RPC, which is what `collect_external_values`' dedup produces.
///
/// This is the increment the `surplus_acquire` rule hands back, so getting its
/// multiplicity right is the difference between testing the rule and testing a
/// fixture bug.
pub(crate) async fn pre_store(ext: &ValkeyExternalValueStore, hashes: &[[u8; 32]]) {
    for hash in hashes {
        ext.store_values(&[(hash, b"b-qpf-external-payload".as_slice())])
            .await
            .expect("the pre-flight store_values must succeed");
    }
}

/// Prove which backend the delta about to be dispatched will really run on.
///
/// `update_layer_once` dispatches on `backend == Lua && is_fast_path(&delta)`,
/// and the delta reaching it is the one handed to `update_layer`. Evaluating the
/// second conjunct here is what keeps a `_lua` test from silently re-running its
/// `_txn` sibling: if a plain field / time-sample delta ever became
/// fast-path-ineligible, `scripts/update_layer_fast.lua`'s step 6b — one of the
/// three transcriptions of the rule — would be covered by nothing while the
/// suite stayed green.
fn assert_fast_path_eligible(delta: &OwnedDelta, ctx: &str) {
    assert!(
        is_fast_path(&Some(delta.clone())),
        "{ctx}: this delta must be fast-path ELIGIBLE. Under `Lua` an ineligible delta falls \
         through to the WATCH/MULTI/EXEC path, so the Lua transcription of the b.qpf rule would \
         be untested; under `Txn` it would have taken the slow path anyway, making the backend \
         knob — and the whole two-variant split — a no-op"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Keyspace + metric readers
// ═════════════════════════════════════════════════════════════════════════════

/// `HGETALL le:{object_id}:obj:ext` — the per-object reference index.
async fn obj_ext(tv: &TestValkey, object_id: u64) -> HashMap<String, String> {
    tv.client()
        .pool
        .hgetall(tv.key_schema().obj_ext_index(&object_id.to_string()))
        .await
        .expect("HGETALL obj:ext must succeed")
}

/// The index as `(hash_hex, count)` pairs, sorted, so an assertion can compare
/// whole maps without depending on hash iteration order.
pub(crate) async fn obj_ext_sorted(tv: &TestValkey, object_id: u64) -> Vec<(String, i64)> {
    let mut rows: Vec<(String, i64)> = obj_ext(tv, object_id)
        .await
        .into_iter()
        .map(|(k, v)| {
            let count = v
                .parse()
                .unwrap_or_else(|e| panic!("obj:ext value for {k} must be an integer: {e}"));
            (k, count)
        })
        .collect();
    rows.sort();
    rows
}

/// `HGET le:ext:meta:{hash_hex} ref_count`, `None` when the meta key is gone.
pub(crate) async fn ref_count(tv: &TestValkey, hash: [u8; 32]) -> Option<i64> {
    let raw: Option<String> = tv
        .client()
        .pool
        .hget(tv.key_schema().ext_meta(&sha256_to_hex(&hash)), "ref_count")
        .await
        .expect("HGET ref_count must succeed");
    raw.map(|v| {
        v.parse()
            .unwrap_or_else(|e| panic!("ref_count must be an integer, got {v:?}: {e}"))
    })
}

/// `(meta_present, data_present)` for one blob.
///
/// Read as a pair because the two are deleted atomically: half a blob surviving
/// is a different (worse) failure from the whole blob surviving, and an
/// assertion on only one of them cannot tell them apart.
async fn blob_present(tv: &TestValkey, hash: [u8; 32]) -> (bool, bool) {
    let hex = sha256_to_hex(&hash);
    let meta: bool = tv
        .client()
        .pool
        .exists(tv.key_schema().ext_meta(&hex))
        .await
        .expect("EXISTS ext_meta must succeed");
    let data: bool = tv
        .client()
        .pool
        .exists(tv.key_schema().ext_data(&hex))
        .await
        .expect("EXISTS ext_data must succeed");
    (meta, data)
}

async fn assert_blob_gone(tv: &TestValkey, hash: [u8; 32], ctx: &str) {
    assert_eq!(
        blob_present(tv, hash).await,
        (false, false),
        "{ctx}: `le:ext:meta:{hex}` and `le:ext:data:{hex}` must BOTH be gone after the GC tick. \
         A surviving pair is the b.qpf leak: the reference the commit superseded was never \
         released, so `ref_count` never reached zero and the blob is unreclaimable for the life \
         of the store",
        hex = sha256_to_hex(&hash)
    );
}

pub(crate) async fn assert_blob_alive(tv: &TestValkey, hash: [u8; 32], ctx: &str) {
    assert_eq!(
        blob_present(tv, hash).await,
        (true, true),
        "{ctx}: the blob must SURVIVE — an object still references it. Deleting here is \
         over-release, which is data loss rather than a leak: a live layer's field would read \
         back `MetaNotFound` forever",
    );
}

/// Every member of `le:ext:{gc}:pending`, sorted.
pub(crate) async fn pending_members(tv: &TestValkey) -> Vec<String> {
    let mut members: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().ext_pending())
        .await
        .expect("SMEMBERS ext_pending must succeed");
    members.sort();
    members
}

/// `live_edit_ext_ref_released_total{reason=…}`.
///
/// Panics on an absent series rather than folding to zero. Every cell is
/// zero-seeded in `Metrics::new`, so absence means the store never received this
/// metrics handle — and a count read off a handle nothing writes to is
/// vacuously whatever the test wants it to be.
///
/// `pub(crate)` because `b_yda_tests` asserts the same cells: the release REASON
/// reaches nothing but this counter — `ExtRelease::member()` is
/// `{hash}:{oid}:{event}` and carries no reason at all — so an arm that
/// distinguishes `rewrite` from `surplus_acquire` has to read it here or not at
/// all. One reader rather than two so the absent-series panic above cannot be
/// re-derived without it.
pub(crate) fn ext_ref_released(metrics: &Metrics, reason: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_ext_ref_released_total",
        &[("reason", reason)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_ext_ref_released_total{{reason=\"{reason}\"}} is absent from the \
             exposition. All three cells are zero-seeded in `Metrics::new`, so absence means \
             the LayerStore under test was built without `with_metrics` and every release \
             assertion in this test is vacuous"
        )
    }) as u64
}

/// `live_edit_ext_ref_release_enqueue_failed_total` — unlabeled.
///
/// The leak signal: one tick per reference an enqueue could not confirm. Same
/// absent-series reasoning as [`ext_ref_released`] — the counter is
/// zero-seeded at registration, so an absent series means the code under test
/// carries no metrics handle and every count read off it is vacuous.
fn ext_ref_release_enqueue_failed(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_ext_ref_release_enqueue_failed_total",
        &[],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_ext_ref_release_enqueue_failed_total is absent from the exposition. It is \
             zero-seeded in `Metrics::new`, so absence means the store under test was built \
             without `with_metrics` and every failure-count assertion here is vacuous"
        )
    }) as u64
}

/// `live_edit_ext_ref_release_deferred_total` — unlabeled (b.yda).
///
/// One tick per owed reference a retirement left recorded on
/// `le:{oid}:obj:extret` when neither of its `{gc}`-slot enqueue attempts was
/// confirmed. Same absent-series reasoning as
/// [`ext_ref_released`].
fn ext_ref_release_deferred(metrics: &Metrics) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_ext_ref_release_deferred_total",
        &[],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_ext_ref_release_deferred_total is absent from the exposition. It is \
             zero-seeded in `Metrics::new`, so absence means the store under test was built \
             without `with_metrics` and every deferral assertion here is vacuous"
        )
    }) as u64
}

/// `HKEYS le:{object_id}:obj:extret` — the retirement retry carrier (a HASH,
/// member → 1, since b.1jt), sorted.
async fn retire_carrier_members(tv: &TestValkey, object_id: u64) -> Vec<String> {
    let mut members: Vec<String> = tv
        .client()
        .pool
        .hkeys(
            tv.key_schema()
                .obj_ext_retire_carrier(&object_id.to_string()),
        )
        .await
        .expect("HKEYS obj:extret must succeed");
    members.sort();
    members
}

/// Every member of the `le:ext:{gc}:retire-retry` work list, sorted.
async fn retire_retry_entries(tv: &TestValkey) -> Vec<String> {
    let mut entries: Vec<String> = tv
        .client()
        .pool
        .zrange(
            tv.key_schema().ext_retire_retry(),
            0,
            -1,
            None,
            false,
            None,
            false,
        )
        .await
        .expect("ZRANGE retire-retry must succeed");
    entries.sort();
    entries
}

/// `live_edit_keyspace_type_error_total{backend, site}`. Same absent-series
/// reasoning as [`ext_ref_released`]. Summed over the `mode` series the
/// `{backend="lua", site="update"}` cell is split into.
fn keyspace_type_errors(metrics: &Metrics, backend: &str, site: &str) -> u64 {
    sample_sum(
        &render(metrics),
        "live_edit_keyspace_type_error_total",
        &[("backend", backend), ("site", site)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_keyspace_type_error_total{{backend=\"{backend}\", site=\"{site}\"}} is \
             absent — the store under test carries no metrics handle"
        )
    }) as u64
}

/// `live_edit_storage_watch_voided_total{path, disposition}`. Same
/// absent-series reasoning as [`ext_ref_released`] — every cell is pre-registered
/// at zero in `Metrics::new`.
///
/// Restated locally rather than imported from `b_bhb_tests`: this file needs only
/// the one cell, and b.bhb owns the counter's own contract. What arm 12 uses it
/// for is narrower — proof that the scenario reached the proven-unwatched
/// settlement at all, without which its `no pending member` assertion is
/// satisfied just as well by a commit that took an ordinary guarded path.
///
/// Feature-gated with its only callers — arm 12 here and the WATCH-window arms
/// in `postcomp_orphan_sweep_fence_tests` — because every scenario that can move
/// this counter needs the commit-window hooks.
#[cfg(feature = "test-fault-injection")]
pub(crate) fn watch_voided(metrics: &Metrics, path: &str, disposition: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_storage_watch_voided_total",
        &[("path", path), ("disposition", disposition)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_storage_watch_voided_total{{path=\"{path}\", \
             disposition=\"{disposition}\"}} is absent — the store under test carries no metrics \
             handle, so every count read off it is vacuous"
        )
    }) as u64
}

/// `HGET le:{object_id}:obj:hdr seq` — the sequence the header advertises.
///
/// Feature-gated for the same reason as [`watch_voided`]: both of its callers
/// use it to ask whether a commit LANDED across a disturbance that only the
/// commit-window hooks can stage.
#[cfg(feature = "test-fault-injection")]
pub(crate) async fn header_seq(tv: &TestValkey, object_id: u64) -> Option<u64> {
    let raw: Option<String> = tv
        .client()
        .pool
        .hget(tv.key_schema().obj_header(&object_id.to_string()), "seq")
        .await
        .expect("HGET obj:hdr seq must succeed");
    raw.map(|v| {
        v.parse()
            .unwrap_or_else(|e| panic!("obj:hdr seq must be an integer, got {v:?}: {e}"))
    })
}

/// The `le:ext:{gc}:pending` member a commit-derived release writes.
fn commit_member(hash: [u8; 32], object_id: u64, seq: u64) -> String {
    ExtRelease::commit(hash, object_id, seq).member()
}

/// The `le:ext:{gc}:pending` member a retirement release writes.
pub(crate) fn retirement_member(hash: [u8; 32], object_id: u64) -> String {
    ExtRelease::retirement(hash, object_id).member()
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. external → different external
// ═════════════════════════════════════════════════════════════════════════════

/// A value row rewritten from hash A to hash B releases A, and one GC tick
/// reclaims A's blob while B's survives.
///
/// The ticket's headline case. Pre-fix nothing at all was enqueued here: the
/// commit paths never called `schedule_release`, so `le:ext:{gc}:pending` stayed
/// empty, the GC tick found no work, and A's `ref_count` sat at 1 with no
/// remaining referrer — `le:ext:meta:{A}` and `le:ext:data:{A}` survived
/// forever. `assert_blob_gone` is the assertion that could not pass pre-fix.
///
/// The pending member is asserted as a whole string, not as "contains A's hex":
/// the `{object_id}:{seq}` suffix is what makes two owed decrements distinct,
/// and a release that dropped it would restore the collapse defect while still
/// satisfying a prefix check.
///
/// Run over both [`RowKind`]s. On the time-sample kind the discriminating
/// production line is `update_layer_fast.lua`'s `HGET ts_key "hash"` in the
/// `"ST"` arm / `ext_rows_written`'s `SetTimeSample` arm — a separate
/// transcription from the field one, deletable on its own.
async fn external_rewritten_to_a_different_external_releases_the_superseded_hash(
    backend: UpdateLayerBackend,
    kind: RowKind,
) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let layer = format!("qpf-ext-to-ext-{}", kind.label());
    let session_id = create_session(&tv, &layer).await;
    let oid = ls
        .create_layer(
            &session_id,
            &layer,
            "b-qpf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let a = ext_hash(0xA1);
    let b = ext_hash(0xB2);

    // ── Commit 1 (seq 2): the row takes hash A ──────────────────────────────
    let delta = kind.external_delta(0xA1, 0);
    assert_fast_path_eligible(&delta, &format!("the external {} write", kind.label()));
    pre_store(&ext, &[a]).await;
    ls.update_layer_relayed(&session_id, &layer, oid, Some(delta), "qpf-writer", 1)
        .await
        .expect("the first external commit must land");

    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xA1), 1)],
        "after the first commit the object holds exactly one row naming A"
    );
    assert_eq!(ref_count(&tv, a).await, Some(1));

    // ── Commit 2 (seq 3): the SAME row takes hash B ─────────────────────────
    let released_before = ext_ref_released(&metrics, REASON_REWRITE);
    let delta = kind.external_delta(0xB2, 1);
    assert_fast_path_eligible(
        &delta,
        &format!("the external→external {} rewrite", kind.label()),
    );
    pre_store(&ext, &[b]).await;
    let (seq, _) = ls
        .update_layer_relayed(&session_id, &layer, oid, Some(delta), "qpf-writer", 2)
        .await
        .expect("the rewrite commit must land");

    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xB2), 1)],
        "the index must name only the hash the object now holds: A's field HDELed, B's at 1. A \
         surviving A entry would keep the retirement union releasing a reference the commit \
         already released"
    );
    assert_eq!(
        pending_members(&tv).await,
        vec![commit_member(a, oid, seq)],
        "exactly one owed decrement, naming A, the committing object and the committing \
         sequence. Pre-fix this set was EMPTY — the commit paths never enqueued anything"
    );
    assert_eq!(
        ext_ref_released(&metrics, REASON_REWRITE),
        released_before + 1,
        "the superseded reference must be counted under reason=\"rewrite\" — the ratio of this \
         counter against ext blob growth is how an operator sees the leak reappear"
    );

    // ── One GC tick reclaims A and leaves B alone ───────────────────────────
    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 1, "one owed decrement, one claim");
    assert_eq!(stats.deleted, 1, "A's ref_count reached zero");

    assert_blob_gone(&tv, a, "the superseded hash A").await;
    assert_eq!(
        ref_count(&tv, b).await,
        Some(1),
        "B is the live value; its reference must be untouched"
    );
    assert_blob_alive(&tv, b, "the newly written hash B").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_field_external_rewritten_to_a_different_external_releases_the_superseded_hash(
) {
    external_rewritten_to_a_different_external_releases_the_superseded_hash(
        UpdateLayerBackend::Lua,
        RowKind::Field,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_field_external_rewritten_to_a_different_external_releases_the_superseded_hash(
) {
    external_rewritten_to_a_different_external_releases_the_superseded_hash(
        UpdateLayerBackend::Txn,
        RowKind::Field,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_ts_external_rewritten_to_a_different_external_releases_the_superseded_hash(
) {
    external_rewritten_to_a_different_external_releases_the_superseded_hash(
        UpdateLayerBackend::Lua,
        RowKind::TimeSample,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_ts_external_rewritten_to_a_different_external_releases_the_superseded_hash(
) {
    external_rewritten_to_a_different_external_releases_the_superseded_hash(
        UpdateLayerBackend::Txn,
        RowKind::TimeSample,
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. external → inline
// ═════════════════════════════════════════════════════════════════════════════

/// A value row rewritten from an external value to an INLINE one releases the
/// superseded hash and empties the index.
///
/// Worth its own arm rather than a parameter of the one above: the two differ in
/// what the commit `referenced`. Here the commit references nothing at all, so
/// the release comes from the row-count arm alone (`before > 0 && after <= 0`)
/// with no `referenced` membership to fall back on — a rule that only handled
/// the external→external shape would leave this one leaking.
///
/// Pre-fix the blob survived the GC tick for the same reason as arm 1.
///
/// Run over both [`RowKind`]s, for the reason arm 1 gives: this is the shape
/// where the superseded hash is recoverable ONLY from the row's pre-image, so a
/// missing `HGET ts_key "hash"` on the time-sample arm loses it outright — the
/// commit's own record carries no hash at all.
async fn external_rewritten_to_inline_releases_the_superseded_hash(
    backend: UpdateLayerBackend,
    kind: RowKind,
) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let layer = format!("qpf-ext-to-inline-{}", kind.label());
    let session_id = create_session(&tv, &layer).await;
    let oid = ls
        .create_layer(
            &session_id,
            &layer,
            "b-qpf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let a = ext_hash(0xC3);

    let delta = kind.external_delta(0xC3, 0);
    assert_fast_path_eligible(&delta, &format!("the external {} write", kind.label()));
    pre_store(&ext, &[a]).await;
    ls.update_layer_relayed(&session_id, &layer, oid, Some(delta), "qpf-writer", 1)
        .await
        .expect("the external commit must land");
    assert_eq!(ref_count(&tv, a).await, Some(1));

    let released_before = ext_ref_released(&metrics, REASON_REWRITE);
    let delta = kind.inline_delta(vec![0x22, 0x33], 1);
    assert_fast_path_eligible(
        &delta,
        &format!("the external→inline {} rewrite", kind.label()),
    );
    let (seq, _) = ls
        .update_layer_relayed(&session_id, &layer, oid, Some(delta), "qpf-writer", 2)
        .await
        .expect("the inline rewrite must land");

    assert!(
        obj_ext(&tv, oid).await.is_empty(),
        "the object holds no external value any more, so the index must be EMPTY — the \
         invariant is `field present ⟺ count > 0`, and a stale entry at 0 would make the \
         retirement union release a reference this commit already released"
    );
    assert_eq!(
        pending_members(&tv).await,
        vec![commit_member(a, oid, seq)],
        "the superseded hash is owed exactly one decrement even though the commit referenced \
         no external value at all"
    );
    assert_eq!(
        ext_ref_released(&metrics, REASON_REWRITE),
        released_before + 1
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.deleted, 1);
    assert_blob_gone(&tv, a, "the hash superseded by an inline value").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_field_external_rewritten_to_inline_releases_the_superseded_hash() {
    external_rewritten_to_inline_releases_the_superseded_hash(
        UpdateLayerBackend::Lua,
        RowKind::Field,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_field_external_rewritten_to_inline_releases_the_superseded_hash() {
    external_rewritten_to_inline_releases_the_superseded_hash(
        UpdateLayerBackend::Txn,
        RowKind::Field,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_ts_external_rewritten_to_inline_releases_the_superseded_hash() {
    external_rewritten_to_inline_releases_the_superseded_hash(
        UpdateLayerBackend::Lua,
        RowKind::TimeSample,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_ts_external_rewritten_to_inline_releases_the_superseded_hash() {
    external_rewritten_to_inline_releases_the_superseded_hash(
        UpdateLayerBackend::Txn,
        RowKind::TimeSample,
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. same-hash re-write across two commits
// ═════════════════════════════════════════════════════════════════════════════

/// Re-writing the SAME hash across two commits hands the second pre-flight
/// increment straight back, so `ref_count` returns to 1 rather than climbing.
///
/// The third transition the ticket names, and the one that leaks on exactly the
/// workload it describes: a client repeatedly saving the same large material or
/// mesh pays a `+1` per save against one eventual `-1` at retirement. Pre-fix
/// `ref_count` here ends at 2 and the retirement release only takes it to 1, so
/// the blob survives the object that owned it — asserted below both as the
/// count and as the blob's absence after retirement, because the count alone
/// would not say the storage was actually reclaimed.
///
/// The release is `surplus_acquire`, not `rewrite`: the object never gave the
/// hash up — its row count went 1 → 1 — so the reference being handed back is
/// the increment the caller paid for a reference it already held.
async fn same_hash_rewritten_twice_hands_back_the_surplus_increment(backend: UpdateLayerBackend) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, "qpf-same-hash").await;
    let layer = "qpf-same-hash";
    let oid = ls
        .create_layer(
            &session_id,
            layer,
            "b-qpf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let a = ext_hash(0xD4);

    // ── Commit 1: the field takes A. Nothing is owed — the object did not
    //    hold A before, so the pre-flight `+1` bought a genuine new reference.
    let delta =
        delta_with_set_field_external(SECTION_ID, FIELD_A, ext_hash_vec(0xD4), EXT_SIZE, 1, 0);
    assert_fast_path_eligible(&delta, "the first external write");
    pre_store(&ext, &[a]).await;
    ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 1)
        .await
        .expect("the first commit must land");
    assert_eq!(ref_count(&tv, a).await, Some(1));
    assert!(
        pending_members(&tv).await.is_empty(),
        "a first-time acquisition owes nothing"
    );

    // ── Commit 2: the SAME field takes the SAME hash again. ─────────────────
    let surplus_before = ext_ref_released(&metrics, REASON_SURPLUS);
    let delta =
        delta_with_set_field_external(SECTION_ID, FIELD_A, ext_hash_vec(0xD4), EXT_SIZE, 1, 1);
    assert_fast_path_eligible(&delta, "the same-hash re-write");
    pre_store(&ext, &[a]).await;
    assert_eq!(
        ref_count(&tv, a).await,
        Some(2),
        "fixture precondition: the second pre-flight store really did increment. Without it \
         there would be no surplus to hand back and the assertion below would be vacuous"
    );
    let (seq, _) = ls
        .update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 2)
        .await
        .expect("the same-hash re-write must land");

    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xD4), 1)],
        "the row count did not change, so the index must not either"
    );
    assert_eq!(
        pending_members(&tv).await,
        vec![commit_member(a, oid, seq)],
        "the surplus increment is owed straight back"
    );
    assert_eq!(
        ext_ref_released(&metrics, REASON_SURPLUS),
        surplus_before + 1,
        "the release must be counted under reason=\"surplus_acquire\" — the object gave nothing \
         up, so counting it as a rewrite would misreport what the store is doing on the busiest \
         cell of an edit-heavy workload"
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.deleted, 0, "the object still references A");
    assert_eq!(
        ref_count(&tv, a).await,
        Some(1),
        "ref_count must return to 1 — ONE live object references A, however many times it \
         re-wrote it. Pre-fix this read 2 and the retirement release below could only take it \
         to 1, stranding the blob forever"
    );
    assert_blob_alive(&tv, a, "the still-referenced hash").await;

    // ── Retirement reclaims it ──────────────────────────────────────────────
    ls.delete_layer(&session_id, layer, ext.clone())
        .await
        .expect("delete_layer must succeed");
    ls.wait_for_cleanup(CLEANUP_JOIN_BUDGET).await;

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.deleted, 1);
    assert_blob_gone(&tv, a, "the retired object's last reference").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_same_hash_rewritten_twice_hands_back_the_surplus_increment() {
    same_hash_rewritten_twice_hands_back_the_surplus_increment(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_same_hash_rewritten_twice_hands_back_the_surplus_increment() {
    same_hash_rewritten_twice_hands_back_the_surplus_increment(UpdateLayerBackend::Txn).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. two rows sharing one hash — the over-release canary
// ═════════════════════════════════════════════════════════════════════════════

/// Rewriting ONE of two rows that share a hash releases NOTHING; rewriting the
/// second releases exactly once.
///
/// The canary for the failure mode the naive fix produces. `store_values`
/// dedups, so a delta setting two rows to the same hash pays ONE `+1`; a
/// symmetric per-row release would then hand back two decrements for one
/// increment, `ref_count` would fall below the number of live referrers, and the
/// GC would delete `ext:meta` + `ext:data` under a layer still pointing at
/// them — data loss, not a leak.
///
/// The first half of this test therefore asserts an ABSENCE and passes on
/// pre-fix code trivially (pre-fix released nothing, ever). Its value is
/// forward-looking. The second half is what fails pre-fix: after the last row
/// gives the hash up, the blob must actually be reclaimed.
async fn rewriting_one_of_two_rows_sharing_a_hash_releases_nothing(backend: UpdateLayerBackend) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, "qpf-shared-hash").await;
    let layer = "qpf-shared-hash";
    let oid = ls
        .create_layer(
            &session_id,
            layer,
            "b-qpf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let h = ext_hash(0xE5);

    // ── Commit 1: BOTH rows take H, in ONE delta, on ONE pre-flight `+1`. ───
    let delta = delta_with_two_fields_sharing_one_external(ext_hash_vec(0xE5));
    assert_fast_path_eligible(&delta, "the two-row shared-hash write");
    pre_store(&ext, &[h]).await;
    ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 1)
        .await
        .expect("the shared-hash commit must land");

    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xE5), 2)],
        "the index counts ROWS, not references: two rows hold H, so the count is 2 while the \
         object's single `ref_count` contribution stays 1"
    );
    assert_eq!(
        ref_count(&tv, h).await,
        Some(1),
        "fixture precondition: one delta referencing H twice pays ONE pre-flight increment \
         (`collect_external_values` dedups). A 2 here would mean the over-release this test \
         guards against is not even expressible"
    );

    // ── Commit 2: FIELD_A gives H up. The object still holds it via FIELD_B. ─
    let delta = delta_with_set_field_inline(SECTION_ID, FIELD_A, vec![0x01], 2);
    assert_fast_path_eligible(&delta, "the first row's rewrite");
    ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 2)
        .await
        .expect("the first row's rewrite must land");

    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xE5), 1)],
        "one row gave H up, so the count drops to 1 — but the field must NOT be HDELed"
    );
    assert!(
        pending_members(&tv).await.is_empty(),
        "NOTHING may be released while the object still has a row holding H. A release here \
         would decrement a reference the object has not given up; with the GC behind it, \
         `ext:meta` + `ext:data` are deleted under a live row and the field reads back \
         `MetaNotFound` forever"
    );
    assert_eq!(
        ext_ref_released(&metrics, REASON_REWRITE),
        0,
        "the rewrite cell must not tick for a hash the object still holds"
    );
    process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(ref_count(&tv, h).await, Some(1));
    assert_blob_alive(&tv, h, "a hash still held by the object's second row").await;

    // ── Commit 3: FIELD_B gives H up too. Now the object has released it. ───
    let delta = delta_with_set_field_inline(SECTION_ID, FIELD_B, vec![0x02], 3);
    assert_fast_path_eligible(&delta, "the second row's rewrite");
    let (seq, _) = ls
        .update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 3)
        .await
        .expect("the second row's rewrite must land");

    assert!(
        obj_ext(&tv, oid).await.is_empty(),
        "the last row holding H is gone, so the index field must be HDELed"
    );
    assert_eq!(
        pending_members(&tv).await,
        vec![commit_member(h, oid, seq)],
        "exactly ONE decrement is owed for the object's one reference, no matter that two rows \
         named it"
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.deleted, 1);
    assert_blob_gone(&tv, h, "the hash both rows have now given up").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_rewriting_one_of_two_rows_sharing_a_hash_releases_nothing() {
    rewriting_one_of_two_rows_sharing_a_hash_releases_nothing(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_rewriting_one_of_two_rows_sharing_a_hash_releases_nothing() {
    rewriting_one_of_two_rows_sharing_a_hash_releases_nothing(UpdateLayerBackend::Txn).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 5. retirement after a rewrite history
// ═════════════════════════════════════════════════════════════════════════════

/// An object retired after rewriting A→B reclaims BOTH blobs, and its
/// `obj:ext` index is reaped with the rest of its keyspace.
///
/// The rewrite history is what makes this different from the pre-existing
/// retirement tests. Pre-fix, retirement released only
/// `collect_external_hashes(&state)` — the object's CURRENT values — so B was
/// reclaimed and A was not. That is the ticket's "for current values only"
/// qualifier, and A's surviving blob is the assertion that fails pre-fix.
///
/// The `obj:ext` reap is asserted through the same whole-`obj:*` SCAN
/// `b_9fn_tests` uses as its key-leak tripwire: a new per-object key that is not
/// in `KeySchema::object_cleanup_keys` leaks one key per retired object id
/// forever, which is precisely the b.9fn defect in a new family.
async fn retirement_after_a_rewrite_history_reclaims_every_hash(backend: UpdateLayerBackend) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, "qpf-retire-history").await;
    let layer = "qpf-retire-history";
    let oid = ls
        .create_layer(
            &session_id,
            layer,
            "b-qpf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let a = ext_hash(0xF6);
    let b = ext_hash(0x17);

    for (byte, hash, client_ts) in [(0xF6u8, a, 1u64), (0x17, b, 2)] {
        let delta = delta_with_set_field_external(
            SECTION_ID,
            FIELD_A,
            ext_hash_vec(byte),
            EXT_SIZE,
            1,
            client_ts,
        );
        assert_fast_path_eligible(&delta, "an external field write");
        pre_store(&ext, &[hash]).await;
        ls.update_layer_relayed(
            &session_id,
            layer,
            oid,
            Some(delta),
            "qpf-writer",
            client_ts,
        )
        .await
        .expect("the external commit must land");
    }

    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0x17), 1)],
        "fixture precondition: after the rewrite only B is held. A's reference is owed a \
         commit-derived decrement, which is the half pre-fix code never enqueued"
    );

    // ── Retire ──────────────────────────────────────────────────────────────
    let retirement_before = ext_ref_released(&metrics, REASON_RETIREMENT);
    ls.delete_layer(&session_id, layer, ext.clone())
        .await
        .expect("delete_layer must succeed");
    ls.wait_for_cleanup(CLEANUP_JOIN_BUDGET).await;

    assert_eq!(
        ext_ref_released(&metrics, REASON_RETIREMENT),
        retirement_before + 1,
        "retirement owes exactly one decrement — B, the object's one surviving reference. A was \
         already released at commit time and must not be released twice"
    );
    assert!(
        pending_members(&tv)
            .await
            .contains(&retirement_member(b, oid)),
        "the retirement release must name the retiring object and the `ret` event"
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(
        stats.deleted, 2,
        "BOTH blobs reach ref_count 0 — the commit-derived release for A and the retirement \
         release for B. Pre-fix only B was ever released, so this read 1 and A leaked"
    );
    assert_blob_gone(&tv, a, "the hash superseded before retirement").await;
    assert_blob_gone(&tv, b, "the retired object's current hash").await;

    // ── The index key itself must be reaped ─────────────────────────────────
    let full_prefix = format!("{}{{{oid}}}:", tv.prefix());
    let mut survivors: Vec<String> = crate::common::scan_object_keys(tv.client(), tv.prefix(), oid)
        .await
        .into_iter()
        .map(|key| {
            let key = key.as_str_lossy().into_owned();
            key.strip_prefix(&full_prefix).unwrap_or(&key).to_string()
        })
        .collect();
    survivors.sort();
    assert!(
        !survivors.iter().any(|k| k == "obj:ext"),
        "`le:{{oid}}:obj:ext` must be UNLINKed at retirement. It is a per-object key in no \
         index, so a `KeySchema::object_cleanup_keys` that forgot it would leak exactly one key \
         per retired object id forever — the b.9fn defect in a new family. Survivors: \
         {survivors:?}"
    );
    let leaked: Vec<&String> = survivors.iter().filter(|k| *k != "obj:stream").collect();
    assert!(
        leaked.is_empty(),
        "`obj:stream` is the only key family that may outlive retirement (PEXPIRE'd with grace \
         so lagging subscribers still see the `deleted` sentinel). Leaked: {leaked:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_retirement_after_a_rewrite_history_reclaims_every_hash() {
    retirement_after_a_rewrite_history_reclaims_every_hash(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_retirement_after_a_rewrite_history_reclaims_every_hash() {
    retirement_after_a_rewrite_history_reclaims_every_hash(UpdateLayerBackend::Txn).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 6. concurrent-release collapse — two objects, one hash
// ═════════════════════════════════════════════════════════════════════════════

/// Two distinct objects referencing one hash, both retired before any GC tick,
/// owe TWO decrements and the blob reaches zero.
///
/// The concurrent-release collapse, folded into this ticket because it is
/// inseparable from it: the release channel is `SADD le:ext:{gc}:pending
/// <member>` and a Set collapses equal members. Pre-fix the member was the bare
/// `{hash_hex}`, so both retirements wrote the SAME member, the GC saw ONE owed
/// decrement, `ref_count` fell 2 → 1 and stopped, and the blob outlived every
/// object that referenced it.
///
/// Both objects are retired BEFORE the first GC tick on purpose — that ordering
/// is the whole defect. Interleaving a tick between them would drain the first
/// member and let the second SADD land in an empty set, which is exactly the
/// case that never failed.
///
/// Not parameterised by backend: the release channel and its member format are
/// backend-independent (both retirements run through
/// `schedule_old_object_ext_gc`), so a second variant would re-run identical
/// code under a different name.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_objects_retiring_before_a_gc_tick_owe_two_distinct_decrements() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, UpdateLayerBackend::Lua, &metrics);
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, "qpf-collapse").await;

    let h = ext_hash(0x28);
    let mut oids = Vec::new();

    for layer in ["qpf-collapse-one", "qpf-collapse-two"] {
        let oid = ls
            .create_layer(
                &session_id,
                layer,
                "b-qpf-user",
                &empty_state(),
                ext.clone(),
            )
            .await
            .expect("create_layer must succeed");
        let delta =
            delta_with_set_field_external(SECTION_ID, FIELD_A, ext_hash_vec(0x28), EXT_SIZE, 1, 0);
        pre_store(&ext, &[h]).await;
        ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 1)
            .await
            .expect("the external commit must land");
        oids.push(oid);
    }

    assert_eq!(
        ref_count(&tv, h).await,
        Some(2),
        "fixture precondition: TWO live objects reference the blob, so ref_count is 2 and two \
         decrements are owed"
    );

    // Retire both BEFORE any GC tick — that ordering is the defect.
    for layer in ["qpf-collapse-one", "qpf-collapse-two"] {
        ls.delete_layer(&session_id, layer, ext.clone())
            .await
            .expect("delete_layer must succeed");
    }
    ls.wait_for_cleanup(CLEANUP_JOIN_BUDGET).await;

    let mut expected = vec![retirement_member(h, oids[0]), retirement_member(h, oids[1])];
    expected.sort();
    assert_eq!(
        pending_members(&tv).await,
        expected,
        "two independently-owed decrements must be TWO members. Pre-fix both retirements wrote \
         the bare `{{hash_hex}}`, the Set collapsed them into one, and the second reference \
         leaked permanently — this is the whole reason the event rides the member"
    );
    assert_eq!(
        ext_ref_released(&metrics, REASON_RETIREMENT),
        2,
        "both retirements are counted, one per retiring object"
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(
        stats.processed, 2,
        "the sweep must claim both owed decrements"
    );
    assert_eq!(
        stats.deleted, 1,
        "the blob is deleted once, at the second decrement"
    );
    assert_blob_gone(&tv, h, "a hash both referencing objects have retired").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 6b. the other half of the member-identity rule — one event replayed
// ═════════════════════════════════════════════════════════════════════════════

/// The SAME release event enqueued repeatedly stays ONE owed decrement, and one
/// GC tick decrements `ref_count` exactly once.
///
/// Arm 6 pins one half of the member-identity rule — distinct events must not
/// collapse. This is the other half, and it constrains the fix in the opposite
/// direction: SADD idempotence *within* an event is what makes the release
/// channel replay-safe, and two production mechanisms depend on it.
/// `delete_layer`'s cascade runs inside a retry loop that can re-run its
/// releases after a transient fred error, and `schedule_old_object_ext_gc`'s
/// `ext_gc_scheduled` header flag is a best-effort guard, not a lock — it is set
/// AFTER the walk, so a replica that dies between the enqueue and the `HSET`
/// leaves the next retirement attempt free to enqueue the same event again.
///
/// A regression that made the member unique per CALL rather than per EVENT —
/// appending a nonce, a wall-clock timestamp, or a retry counter — would satisfy
/// arm 6 completely and fail only here. Its consequence is the loss direction,
/// not the leak direction: two members for one owed decrement take `ref_count`
/// 2 → 0 and the GC UNLINKs `ext:meta` + `ext:data` underneath the SECOND,
/// still-live object, whose field then reads back `MetaNotFound` forever. Hence
/// the surviving object: without it `ref_count` would reach 0 legitimately at
/// the second decrement and the over-release would be invisible.
///
/// The first enqueue is a REAL `delete_layer` retirement rather than a third
/// hand-made call, so the member the replays must match is the one production
/// actually writes; a test that only ever called `schedule_release` could agree
/// with itself while disagreeing with the retirement path.
///
/// Not parameterised by backend, for arm 6's reason: the release channel and its
/// member format are backend-independent.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replaying_one_release_event_still_owes_exactly_one_decrement() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, UpdateLayerBackend::Lua, &metrics);
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, "qpf-replay").await;

    let h = ext_hash(0x6E);
    let mut oids = HashMap::new();

    // Two objects reference H. Only the first retires; the second is the live
    // victim an over-release would delete the blob underneath.
    for layer in ["qpf-replay-retiring", "qpf-replay-survivor"] {
        let oid = ls
            .create_layer(
                &session_id,
                layer,
                "b-qpf-user",
                &empty_state(),
                ext.clone(),
            )
            .await
            .expect("create_layer must succeed");
        let delta =
            delta_with_set_field_external(SECTION_ID, FIELD_A, ext_hash_vec(0x6E), EXT_SIZE, 1, 0);
        pre_store(&ext, &[h]).await;
        ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 1)
            .await
            .expect("the external commit must land");
        oids.insert(layer, oid);
    }
    let retiring = oids["qpf-replay-retiring"];

    assert_eq!(
        ref_count(&tv, h).await,
        Some(2),
        "fixture precondition: TWO live objects reference H, so exactly ONE decrement is owed by \
         the retirement below and the blob must survive it"
    );
    assert!(
        pending_members(&tv).await.is_empty(),
        "fixture precondition: first-time acquisitions owe nothing, so every member counted \
         below comes from the retirement and its replays"
    );

    // ── The real retirement enqueues the event once ─────────────────────────
    ls.delete_layer(&session_id, "qpf-replay-retiring", ext.clone())
        .await
        .expect("delete_layer must succeed");
    ls.wait_for_cleanup(CLEANUP_JOIN_BUDGET).await;

    let expected = vec![retirement_member(h, retiring)];
    assert_eq!(
        pending_members(&tv).await,
        expected,
        "the retirement owes exactly one decrement, naming the retiring object and the `ret` event"
    );

    // ── Replay the IDENTICAL event twice, as a retried cascade would ────────
    let replay = ExtRelease::retirement(h, retiring);
    for attempt in 1..=2 {
        ext.schedule_release(std::slice::from_ref(&replay))
            .await
            .unwrap_or_else(|e| panic!("replay {attempt} of the release must succeed: {e:?}"));
    }

    assert_eq!(
        pending_members(&tv).await,
        expected,
        "three enqueues of ONE event must leave ONE member. A member carrying a nonce, a \
         timestamp or a retry counter would leave three here — and the GC would then take \
         `ref_count` 2 → -1, deleting H underneath the surviving object"
    );

    // ── One tick, one decrement, blob intact ────────────────────────────────
    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 1, "one member, one claim");
    assert_eq!(
        stats.deleted, 0,
        "the surviving object still references H, so nothing may be reclaimed"
    );
    assert_eq!(
        ref_count(&tv, h).await,
        Some(1),
        "exactly ONE decrement for the one retirement event, however many times it was enqueued. \
         A 0 here is the over-release this test exists to catch"
    );
    assert_blob_alive(&tv, h, "a hash the surviving object still holds").await;

    assert_eq!(
        ext_ref_released(&metrics, REASON_RETIREMENT),
        1,
        "one retirement, one counted release — the replays are re-enqueues of an event already \
         counted, and counting them again would make the metric read as reference churn that \
         never happened"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 7. rolling deploy — an object with rows but no `obj:ext`
// ═════════════════════════════════════════════════════════════════════════════

/// An object materialized by a PRE-b.qpf replica carries no `obj:ext` key, and
/// must still release exactly one decrement per distinct hash at retirement.
///
/// `obj:ext` ships with the fix and nothing backfills it, so during a rolling
/// deploy a post-fix replica will retire objects whose value rows were written
/// by a replica that never maintained the index. `HKEYS` of a missing key is an
/// empty array rather than an error, so a retirement that read the index ALONE
/// would release nothing and leak every one of that object's references — which
/// is why `schedule_old_object_ext_gc` releases the UNION of the decoded state
/// walk and the index.
///
/// The fixture deletes a real `obj:ext` rather than hand-building rows, so the
/// value rows are exactly what a real commit writes. "Exactly one per distinct
/// hash" is the load-bearing half: the object has TWO rows holding A, and the
/// union must not turn that into two decrements against one increment.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retirement_of_an_object_with_no_ext_index_releases_once_per_distinct_hash() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, UpdateLayerBackend::Lua, &metrics);
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, "qpf-rolling").await;
    let layer = "qpf-rolling";
    let oid = ls
        .create_layer(
            &session_id,
            layer,
            "b-qpf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let a = ext_hash(0x39);
    let b = ext_hash(0x4A);

    // Two field rows holding A plus a time sample holding B, in one commit on
    // one pre-flight increment per distinct hash.
    let mut delta = delta_with_two_fields_sharing_one_external(ext_hash_vec(0x39));
    delta.time_samples =
        delta_with_time_sample_external(SECTION_ID, SAMPLE_TIME, ext_hash_vec(0x4A), EXT_SIZE, 2)
            .time_samples;
    assert_fast_path_eligible(&delta, "the three-row external write");
    pre_store(&ext, &[a, b]).await;
    ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 1)
        .await
        .expect("the external commit must land");

    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0x39), 2), (ext_hash_hex(0x4A), 1)],
        "fixture precondition: the index really was maintained before the DEL below removes it"
    );

    // Simulate the pre-fix object: drop the index the fix introduced.
    let removed: u64 = tv
        .client()
        .pool
        .del(tv.key_schema().obj_ext_index(&oid.to_string()))
        .await
        .expect("DEL obj:ext must succeed");
    assert_eq!(removed, 1, "the DEL must actually have removed the index");

    ls.delete_layer(&session_id, layer, ext.clone())
        .await
        .expect("delete_layer must succeed");
    ls.wait_for_cleanup(CLEANUP_JOIN_BUDGET).await;

    let mut expected = vec![retirement_member(a, oid), retirement_member(b, oid)];
    expected.sort();
    assert_eq!(
        pending_members(&tv).await,
        expected,
        "exactly one decrement per DISTINCT hash, recovered from the decoded state walk with no \
         index to read. Three members would mean the union counted A once per row and is about \
         to over-release; zero or one would mean the state walk was dropped in favour of the \
         index alone and every pre-fix object leaks its whole reference set"
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.deleted, 2);
    assert_blob_gone(&tv, a, "the hash two rows of the index-less object held").await;
    assert_blob_gone(&tv, b, "the hash its time sample held").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 8. legacy bare-hex pending member
// ═════════════════════════════════════════════════════════════════════════════

/// A bare `{hash_hex}` member — written by a pre-b.qpf replica, or already
/// sitting in `pending` at upgrade time — still claims, decrements and deletes.
///
/// The no-migration guarantee. A `hash_hex` is 64 hex characters and can never
/// contain a `:`, so `pending_member_hash_hex` takes the prefix up to the first
/// `:` and a bare member parses to itself. If that ever stopped holding, every
/// member queued before the upgrade would be claimed and then decrement the
/// WRONG key — or nothing — and the drain would spin on entries it could not
/// resolve.
///
/// Seeded with a raw `SADD` rather than through `schedule_release`, because the
/// production API can no longer produce this member shape: the point is that
/// the READER still accepts it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_legacy_bare_hex_pending_member_still_claims_and_deletes() {
    let tv = TestValkey::setup().await;
    let ext = ext_store(&tv);

    let h = ext_hash(0x5B);
    let hex = ext_hash_hex(0x5B);
    pre_store(&ext, &[h]).await;
    assert_eq!(ref_count(&tv, h).await, Some(1));

    let added: u64 = tv
        .client()
        .pool
        .sadd(tv.key_schema().ext_pending(), hex.as_str())
        .await
        .expect("SADD must succeed");
    assert_eq!(added, 1, "the legacy member must actually have been seeded");

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(
        stats.processed, 1,
        "the sweep must claim the legacy member — a parse that rejected it would leave it in \
         `pending` forever and the sweep would spin on it every tick"
    );
    assert_eq!(stats.deleted, 1);
    assert_blob_gone(&tv, h, "a blob released by a pre-upgrade bare-hex member").await;
    assert!(
        pending_members(&tv).await.is_empty(),
        "the legacy member must be removed from `pending` by its own claim"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 9. DeleteField / DeleteTimeSample via the section-delete cascade
// ═════════════════════════════════════════════════════════════════════════════

/// Deleting a section releases the external values its rows held.
///
/// A row that is DELETED gives its hash up exactly as a row that is REWRITTEN
/// does — the `hash` column goes with the key — so the release rule has to cover
/// `DeleteField` / `DeleteTimeSample` and not just the two `Set` arms. This
/// codebase has no "tombstoned row that RETAINS its hash column": the section
/// cascade emits a `DeleteField` / `DeleteTimeSample` per row and writes its
/// tombstone to a separate `obj:del:{sid}` hash that carries no value column.
///
/// **Slow path only, structurally.** `storage::lua::encode_delta` emits only
/// `SF` and `ST` records — the fast path has no delete record — and any
/// non-`DiffDeleteSection` struct command makes a delta fast-path-INELIGIBLE, so
/// a `DeleteSection` takes the WATCH/MULTI/EXEC path whichever backend is
/// configured. Asserted below rather than assumed, so a future encoder that
/// grew delete records fails here and asks for a Lua variant instead of silently
/// leaving one uncovered.
///
/// Pre-fix nothing was released on a delete either, so both blobs survived the
/// GC tick.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deleting_a_section_releases_the_external_values_its_rows_held() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    // The default backend, deliberately: the claim is that a structural delta
    // reaches the slow path even with the Lua fast path enabled.
    let ls = layer_store(&tv, UpdateLayerBackend::Lua, &metrics);
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, "qpf-section-delete").await;
    let layer = "qpf-section-delete";
    let oid = ls
        .create_layer(
            &session_id,
            layer,
            "b-qpf-user",
            &state_with_section(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let a = ext_hash(0x6C);
    let b = ext_hash(0x7D);

    let delta = delta_with_external_field_and_time_sample(ext_hash_vec(0x6C), ext_hash_vec(0x7D));
    assert_fast_path_eligible(&delta, "the external field + time-sample write");
    pre_store(&ext, &[a, b]).await;
    ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 1)
        .await
        .expect("the external commit must land");
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0x6C), 1), (ext_hash_hex(0x7D), 1)],
        "fixture precondition: both a field row and a time-sample row hold an external value, \
         so the cascade below drives BOTH the DeleteField and the DeleteTimeSample arm"
    );

    // ── Delete the section the two rows hang off ────────────────────────────
    let delete = OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id: SECTION_ID,
        })],
        ..Default::default()
    };
    assert!(
        !is_fast_path(&Some(delete.clone())),
        "a `DeleteSection` must be fast-path INELIGIBLE. `encode_delta` emits only `SF` / `ST` \
         records, so if this delta ever reached the Lua script its cascade's row deletions — \
         and the releases they owe — would have no transcription there at all"
    );
    let released_before = ext_ref_released(&metrics, REASON_REWRITE);
    let (seq, _) = ls
        .update_layer_relayed(&session_id, layer, oid, Some(delete), "qpf-writer", 2)
        .await
        .expect("the section delete must land");

    assert!(
        obj_ext(&tv, oid).await.is_empty(),
        "every row that held an external value is gone, so the index must be empty"
    );
    let mut expected = vec![commit_member(a, oid, seq), commit_member(b, oid, seq)];
    expected.sort();
    assert_eq!(
        pending_members(&tv).await,
        expected,
        "both the deleted FIELD's hash and the deleted TIME SAMPLE's hash are owed a decrement. \
         One member here means only one of the two cascade arms drives the index — they are \
         separate `StorageMutation` variants and one can be fixed without the other"
    );
    assert_eq!(
        ext_ref_released(&metrics, REASON_REWRITE),
        released_before + 2,
        "a deleted row is a rewrite for release purposes: the object gave the hash up"
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.deleted, 2);
    assert_blob_gone(&tv, a, "the deleted field's hash").await;
    assert_blob_gone(&tv, b, "the deleted time sample's hash").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 10. in-commit move of a hash between rows
// ═════════════════════════════════════════════════════════════════════════════

/// One commit that moves a hash from row A to row B leaves the index unchanged,
/// hands back the surplus increment once, and the blob survives.
///
/// The case where the `-1` and the `+1` land inside a single commit and cancel.
/// The row count never leaves 1, so the row-count arm cannot fire, and the only
/// correct release is the `surplus_acquire` for the pre-flight increment the
/// caller paid on a hash the object already held. A rule that folded the two
/// arms — releasing on any negative per-row transition — would emit a `rewrite`
/// here as well and over-release.
///
/// Pre-fix `ref_count` ends at 2 with nothing owed; the discriminating assertion
/// is that after the GC tick it reads 1.
async fn moving_a_hash_between_rows_in_one_commit_releases_only_the_surplus(
    backend: UpdateLayerBackend,
) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, "qpf-hash-move").await;
    let layer = "qpf-hash-move";
    let oid = ls
        .create_layer(
            &session_id,
            layer,
            "b-qpf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let h = ext_hash(0x8E);

    let delta =
        delta_with_set_field_external(SECTION_ID, FIELD_A, ext_hash_vec(0x8E), EXT_SIZE, 1, 0);
    assert_fast_path_eligible(&delta, "the initial external write");
    pre_store(&ext, &[h]).await;
    ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 1)
        .await
        .expect("the initial commit must land");
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0x8E), 1)]
    );

    // ── One commit: row A → inline, row B → H. Net row-count change zero. ───
    let surplus_before = ext_ref_released(&metrics, REASON_SURPLUS);
    let delta = delta_moving_the_hash_from_a_to_b(ext_hash_vec(0x8E));
    assert_fast_path_eligible(&delta, "the in-commit hash move");
    pre_store(&ext, &[h]).await;
    let (seq, _) = ls
        .update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 2)
        .await
        .expect("the hash-move commit must land");

    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0x8E), 1)],
        "one row gave H up and another took it in the SAME commit, so the count is unchanged. \
         An HDEL here would be the transitions folded in the wrong order"
    );
    assert_eq!(
        pending_members(&tv).await,
        vec![commit_member(h, oid, seq)],
        "exactly ONE owed decrement — the surplus. A second member would mean the row-count arm \
         fired too, and `ref_count` would fall below the number of live referrers"
    );
    assert_eq!(
        ext_ref_released(&metrics, REASON_SURPLUS),
        surplus_before + 1
    );
    assert_eq!(
        ext_ref_released(&metrics, REASON_REWRITE),
        0,
        "the object never gave H up, so the rewrite cell must stay clean"
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.deleted, 0);
    assert_eq!(
        ref_count(&tv, h).await,
        Some(1),
        "one live object references H, so ref_count must be 1. Pre-fix it read 2 — the second \
         pre-flight increment was never handed back"
    );
    assert_blob_alive(&tv, h, "a hash that merely moved rows").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_moving_a_hash_between_rows_in_one_commit_releases_only_the_surplus() {
    moving_a_hash_between_rows_in_one_commit_releases_only_the_surplus(UpdateLayerBackend::Lua)
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_moving_a_hash_between_rows_in_one_commit_releases_only_the_surplus() {
    moving_a_hash_between_rows_in_one_commit_releases_only_the_surplus(UpdateLayerBackend::Txn)
        .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 11. WRONGTYPE `obj:ext` — the Lua validation pre-pass
// ═════════════════════════════════════════════════════════════════════════════

/// A string planted at `le:{oid}:obj:ext` refuses the whole fast-path commit,
/// before the `XADD`.
///
/// `obj:ext` is a key the fast path WRITES, so it joins the script's validation
/// pre-pass alongside the two index ZSets. Getting this wrong is not a cosmetic
/// failure: the script's b.ph9 contract is that nothing is rejected AFTER the
/// `XADD`, so a `WRONGTYPE` discovered during the apply pass would abort a
/// commit whose stream entry subscribers had already been told about — the
/// half-commit b.459 exists to close, reopened in a new family.
///
/// Four claims, and each is separately load-bearing: the refusal is a
/// non-retriable `Internal` naming the key (no retry re-types a key), the header
/// sequence did NOT advance (a client reading it back would see a version it can
/// never fetch), no stream entry was appended (the observable half of the same
/// fact), and the counter ticked exactly once on the `lua` cell.
///
/// Lua-only by construction: the txn path has no script pre-pass — it refuses
/// the same keyspace through `probe_keyspace_types` before `MULTI`, which
/// `b_459_tests` covers for every key family it names.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_wrongtype_ext_index_refuses_the_fast_path_commit_before_the_stream_write() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, UpdateLayerBackend::Lua, &metrics);
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, "qpf-wrongtype-ext").await;
    let layer = "qpf-wrongtype-ext";
    let oid = ls
        .create_layer(
            &session_id,
            layer,
            "b-qpf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let a = ext_hash(0x9F);
    let delta =
        delta_with_set_field_external(SECTION_ID, FIELD_A, ext_hash_vec(0x9F), EXT_SIZE, 1, 0);
    pre_store(&ext, &[a]).await;
    ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 1)
        .await
        .expect("the healthy external commit must land");

    // Plant a string over the index an out-of-band writer could have created.
    let ext_index_key = tv.key_schema().obj_ext_index(&oid.to_string());
    let _: () = tv
        .client()
        .pool
        .set(
            &ext_index_key,
            "planted-by-an-out-of-band-writer",
            None,
            None,
            false,
        )
        .await
        .expect("planting a string must succeed");

    let stream_key = tv.key_schema().layer_stream(oid);
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let seq_before: String = tv
        .client()
        .pool
        .hget(&header_key, "seq")
        .await
        .expect("HGET seq must succeed");
    let xlen_before: u64 = tv
        .client()
        .pool
        .xlen(&stream_key)
        .await
        .expect("XLEN must succeed");
    let counter_before = keyspace_type_errors(&metrics, "lua", "update");

    // A commit carrying ANY value record probes the index — including one whose
    // value is INLINE, because an inline write can still overwrite a row that
    // held an external value.
    let delta = delta_with_set_field_inline(SECTION_ID, FIELD_A, vec![0x44], 1);
    assert_fast_path_eligible(&delta, "the refused commit");
    let err = ls
        .update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 2)
        .await
        .expect_err("a commit over a wrong-typed obj:ext must be refused");

    let ServiceError::Internal(msg) = &err else {
        panic!(
            "a wrong-typed key the commit would write must surface as a NON-RETRIABLE \
             `Internal`. `Aborted` would spin the 128-attempt retry loop against a condition no \
             retry can clear. Got: {err:?}"
        );
    };
    assert!(
        msg.contains(&ext_index_key),
        "the refusal must NAME the offending key — that name is the whole remediation; got \
         {msg:?}"
    );
    assert!(
        msg.contains("holds a string") && msg.contains("expected hash"),
        "the refusal must say what the key holds and what the commit needed, so an operator can \
         tell an out-of-band writer's leftover from a bug in this service; got {msg:?}"
    );

    let seq_after: String = tv
        .client()
        .pool
        .hget(&header_key, "seq")
        .await
        .expect("HGET seq must succeed");
    assert_eq!(
        seq_after, seq_before,
        "the header sequence must not advance for a commit that did not happen"
    );
    assert_eq!(
        tv.client()
            .pool
            .xlen::<u64, _>(&stream_key)
            .await
            .expect("XLEN must succeed"),
        xlen_before,
        "no stream entry may be appended. The pre-pass runs BEFORE the XADD, which is the b.ph9 \
         no-rejection proof: once the entry is published the script may not reject, so an index \
         probe that ran during the apply pass instead would tell subscribers about a commit the \
         caller was told had failed"
    );
    assert_eq!(
        keyspace_type_errors(&metrics, "lua", "update"),
        counter_before + 1,
        "exactly one refusal must tick live_edit_keyspace_type_error_total{{backend=\"lua\", \
         site=\"update\"}} — zero leaves operators blind to a layer that has silently stopped \
         accepting writes, more than one means the probe counts per key rather than per commit"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 12. retired underneath a proven-unwatched commit
// ═════════════════════════════════════════════════════════════════════════════

/// When a commit lands on a connection that provably did not hold its `WATCH`
/// and the retirement re-check then finds the object superseded, the commit's
/// owed releases are deliberately NOT enqueued.
///
/// The arm must choose one of {leak, loss} unconditionally, and chooses leak.
/// The `FailedPrecondition` verdict means a `schedule_old_object_ext_gc` for
/// this same `object_id` is concurrently in flight, releasing the object's whole
/// reference set at `event = Retirement` from a snapshot that may predate this
/// commit. Publishing a `Commit(seq)` release too would put TWO pending members
/// on the channel for ONE owed decrement — `ref_count` falls twice, and with
/// another live object referencing the same hash, `ext:meta` + `ext:data` are
/// deleted underneath it. That is data loss; the stranded decrement is a leak,
/// reclaimed by the retirement release already running for this doomed object.
///
/// The live-object sibling is asserted in the same test rather than in a second
/// one, because the claim is a DIFFERENCE: "no member on the retired verdict" is
/// satisfiable by a build that enqueues nothing at all, and only the live arm
/// distinguishes the gate from a broken enqueue.
///
/// Both arms additionally pin that the scenario was actually STAGED, which the
/// b.qpf post-completion review found neither of them did (finding 7). The
/// disturbance is a real `CLIENT KILL ID` race, so a kill that landed
/// harmlessly leaves both arms passing for the wrong reason: the live arm's
/// member is enqueued by any ordinary commit, and the retired arm's empty
/// pending set is what an ordinary pre-`EXEC` `superseded` refusal produces too.
/// So each arm asserts `storage_watch_voided_total{path="update_layer",
/// disposition="committed_unwatched"}` advanced by exactly one — the counter is
/// ticked once at the top of `settle_unwatched_commit` and nowhere else — and
/// the retired arm additionally asserts the doomed object's `obj:hdr seq` and
/// `obj:ext` reflect the write the unwatched `EXEC` landed, which is what makes
/// the suppressed release a real owed release rather than a hypothetical one.
///
/// Txn-only: the Lua fast path never opens a `WATCH`, so it cannot reach this
/// verdict. The delta is a plain external field rewrite, which IS fast-path
/// eligible, so the backend knob is what forces it down the slow path — see the
/// `with_update_layer_backend` call.
///
/// Feature-gated because `with_pre_dispatch_hook_for_test` is. The connection
/// helpers are shared with `b_bhb_tests` rather than copied: they derive the
/// watched connection exactly the way `WatchFingerprint::take` does, so the
/// socket this test kills is by construction the one production believes it is
/// watching on, and a second copy could drift from that.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_commit_on_an_object_retired_underneath_it_enqueues_no_release() {
    use fred::clients::Client;
    use futures::FutureExt;
    use live_edit::storage::watched_txn::WatchWindowHook;

    use crate::storage::b_bhb_tests::{
        await_reconnect, guarded_connection, kill_connection, once_only, POOL_SIZE, SCENARIO_BUDGET,
    };

    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, "qpf-retired-underneath").await;

    let h = ext_hash(0xAF);

    // Two layers reference H, so the doomed commit's would-be second decrement
    // has a live victim: over-release would delete the blob under the survivor.
    let plain = layer_store(&tv, UpdateLayerBackend::Txn, &metrics);
    let mut oids = HashMap::new();
    for layer in ["qpf-retired-doomed", "qpf-retired-survivor"] {
        let oid = plain
            .create_layer(
                &session_id,
                layer,
                "b-qpf-user",
                &empty_state(),
                ext.clone(),
            )
            .await
            .expect("create_layer must succeed");
        let delta =
            delta_with_set_field_external(SECTION_ID, FIELD_A, ext_hash_vec(0xAF), EXT_SIZE, 1, 0);
        pre_store(&ext, &[h]).await;
        plain
            .update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 1)
            .await
            .expect("the external commit must land");
        oids.insert(layer, oid);
    }
    let doomed = oids["qpf-retired-doomed"];
    let survivor = oids["qpf-retired-survivor"];
    assert_eq!(
        ref_count(&tv, h).await,
        Some(2),
        "fixture precondition: two live objects reference H"
    );

    // ── The live-object control arm: the same disturbance, no retirement. ───
    let killer = tv.client().pool.next().clone();
    let survivor_header = tv.key_schema().obj_header(&survivor.to_string());
    let hook: WatchWindowHook = once_only(Arc::new(move |client: &Client| {
        let killer = killer.clone();
        let header_key = survivor_header.clone();
        async move {
            let (node, conn_id) = guarded_connection(client, &header_key);
            kill_connection(&killer, &node, conn_id).await;
            await_reconnect(client, &node, conn_id).await;
        }
        .boxed()
    }));
    let disturbed =
        layer_store(&tv, UpdateLayerBackend::Txn, &metrics).with_pre_dispatch_hook_for_test(hook);

    // Re-write the SAME hash so the commit owes a `surplus_acquire` release.
    let delta =
        delta_with_set_field_external(SECTION_ID, FIELD_A, ext_hash_vec(0xAF), EXT_SIZE, 1, 1);
    pre_store(&ext, &[h]).await;
    let unwatched_before = watch_voided(&metrics, "update_layer", "committed_unwatched");
    let (live_seq, _) = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            "qpf-retired-survivor",
            survivor,
            Some(delta),
            "qpf-writer",
            2,
        ),
    )
    .await
    .expect("the disturbed live-object update did not settle within the scenario budget")
    .expect(
        "the commit landed and nothing contended for its sequence, so it must be reported \
             as committed",
    );

    assert!(
        pending_members(&tv)
            .await
            .contains(&commit_member(h, survivor, live_seq)),
        "control arm: a proven-unwatched commit on a LIVE object still enqueues the release it \
         owes. Without this, the retired arm's `no member` assertion would also pass on a build \
         whose enqueue is simply broken"
    );
    // The scenario's own precondition, and the one thing neither arm asserted
    // before the b.qpf post-completion review: that the commit reached
    // `settle_unwatched_commit` at all. The disturbance is a real `CLIENT KILL`
    // race, so nothing in the arms above rules out a build (or a scheduling
    // accident) in which the kill landed harmlessly and the commit took an
    // ordinary GUARDED path — which satisfies "the release was enqueued" here
    // and "no release was enqueued" in the retired arm below for entirely the
    // wrong reason. `committed_unwatched` is ticked exactly once, at the top of
    // `settle_unwatched_commit`, so it is the proof the proven-unwatched path
    // was taken; `+1` rather than `== 1` because the arms share one `Metrics`.
    assert_eq!(
        watch_voided(&metrics, "update_layer", "committed_unwatched"),
        unwatched_before + 1,
        "control arm: the commit must have been settled as PROVEN UNWATCHED exactly once. Zero \
         means the kill did not void the watch and this arm proved nothing about the gate; more \
         than one means the disturbance leaked into a retry, so `live_seq` names a different \
         commit than the member asserted above"
    );

    // ── The retired arm ────────────────────────────────────────────────────
    let killer = tv.client().pool.next().clone();
    let data_pool = tv.client().pool.next().clone();
    let doomed_header = tv.key_schema().obj_header(&doomed.to_string());
    let doomed_header_for_hook = doomed_header.clone();
    let hook: WatchWindowHook = once_only(Arc::new(move |client: &Client| {
        let killer = killer.clone();
        let data_pool = data_pool.clone();
        let header_key = doomed_header_for_hook.clone();
        async move {
            let (node, conn_id) = guarded_connection(client, &header_key);
            kill_connection(&killer, &node, conn_id).await;
            await_reconnect(client, &node, conn_id).await;
            // Retire the object from a different connection, exactly as a
            // concurrent CreateLayer overwrite would. Under a live WATCH this
            // would cancel the EXEC; the point is that the WATCH is gone.
            let _: () = data_pool
                .hset(&header_key, [("superseded", "1")])
                .await
                .expect("marking the header superseded must succeed");
        }
        .boxed()
    }));
    let disturbed =
        layer_store(&tv, UpdateLayerBackend::Txn, &metrics).with_pre_dispatch_hook_for_test(hook);

    let delta =
        delta_with_set_field_external(SECTION_ID, FIELD_A, ext_hash_vec(0xAF), EXT_SIZE, 1, 1);
    // Baselines are taken AFTER the pre-flight store, not before: the `+1` this
    // doomed RPC pays is exactly the increment the suppressed release would have
    // handed back, so folding it into the baseline would make the assertion
    // below measure the fixture instead of the gate. That stranded increment IS
    // the leak this arm deliberately chooses over the loss — it is reclaimed by
    // the retirement union release already running for this object.
    pre_store(&ext, &[h]).await;
    let pending_before = pending_members(&tv).await;
    let refcount_before = ref_count(&tv, h).await;
    let unwatched_before = watch_voided(&metrics, "update_layer", "committed_unwatched");
    let doomed_seq_before = header_seq(&tv, doomed)
        .await
        .expect("fixture precondition: the doomed object's header must carry a seq");
    let result = tokio::time::timeout(
        SCENARIO_BUDGET,
        disturbed.update_layer_relayed(
            &session_id,
            "qpf-retired-doomed",
            doomed,
            Some(delta),
            "qpf-writer",
            2,
        ),
    )
    .await
    .expect("the disturbed retired-object update did not settle within the scenario budget");

    match result {
        Err(ServiceError::FailedPrecondition(msg)) => assert!(
            msg.contains("superseded"),
            "the retirement re-check must surface the same superseded sentinel an ordinary \
             overwrite race produces; got {msg:?}"
        ),
        other => panic!(
            "a proven-unwatched commit onto a retired object must surface the retirement shape; \
             got {other:?}"
        ),
    }

    assert_eq!(
        pending_members(&tv).await,
        pending_before,
        "the doomed commit must enqueue NOTHING. A `Commit(seq)` member here would be a SECOND \
         pending member for the one decrement the concurrent retirement is already publishing; \
         `ref_count` would fall twice and the GC would delete H's meta+data underneath the \
         surviving layer that still points at it"
    );
    assert_eq!(
        ref_count(&tv, h).await,
        refcount_before,
        "the refusal itself must not move `ref_count` — a release is an ENQUEUE, and only the \
         GC sweep decrements. A drop here would mean the doomed path decremented in line, which \
         is the double-decrement this arm exists to prevent"
    );
    assert_eq!(
        watch_voided(&metrics, "update_layer", "committed_unwatched"),
        unwatched_before + 1,
        "the doomed commit must have been settled as PROVEN UNWATCHED exactly once. Zero means \
         the kill did not void the watch, in which case the `FailedPrecondition` above came from \
         the ordinary WATCH-window `superseded` guard — a commit that never landed, whose empty \
         pending set says nothing at all about the gate this arm exists to test"
    );

    // The commit LANDED. That is the whole premise of the suppression: a
    // proven-unwatched EXEC is not undone, so the doomed object now carries this
    // commit's write and owes the `surplus_acquire` release the gate is
    // deliberately withholding. Asserting the keyspace here is what separates
    // "the release was suppressed" from "there was no release to suppress" —
    // without it a build that refused BEFORE the EXEC (the ordinary guard shape)
    // satisfies every other assertion in this arm.
    assert_eq!(
        header_seq(&tv, doomed).await,
        Some(doomed_seq_before + 1),
        "the doomed object's header must advertise the sequence the unwatched EXEC wrote. An \
         unchanged seq would mean nothing committed, which makes the suppressed release \
         hypothetical and this arm vacuous"
    );
    assert_eq!(
        obj_ext_sorted(&tv, doomed).await,
        vec![(ext_hash_hex(0xAF), 1)],
        "the landed commit must leave `obj:ext` ABSOLUTELY correct for a same-hash rewrite: one \
         row still holds H, so the count stays 1. `obj:ext` is maintained inside the commit's own \
         atomic unit, so a 2 here would mean the unwatched EXEC double-counted the row it \
         rewrote — and the retirement release that is about to run for this doomed object reads \
         exactly this index to decide what it owes"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// 13. Cluster-gated siblings
// ═════════════════════════════════════════════════════════════════════════════

/// Cluster sibling of arm 1: the commit-derived release must survive the
/// three-way slot split.
///
/// One commit touches three different Cluster slots — `{object_id}` for the
/// value row and `obj:ext`, `{hash_hex}` for the blob's meta+data, and `{gc}`
/// for the pending set. No atomic unit can span them, which is exactly why the
/// release is enqueued AFTER the commit rather than inside it. A regression that
/// tried to fold the SADD into the commit's `MULTI` — or that dropped a hash tag
/// — raises `CROSSSLOT` here and passes everywhere else.
///
/// Gated on `LIVE_EDIT_TEST_CLUSTER_URL` and skipped with a loud line when it is
/// unset; CI's `test-cluster:` job is what runs the body.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_external_rewrite_release_crosses_the_object_hash_and_gc_slots() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    let tv = TestValkey::setup_with_url(cluster_url, 4).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "cluster URL required for the cluster release regression"
    );

    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, UpdateLayerBackend::Lua, &metrics);
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, "qpf-cluster-rewrite").await;
    let layer = "qpf-cluster-rewrite";
    let oid = ls
        .create_layer(
            &session_id,
            layer,
            "b-qpf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed on cluster");

    let a = ext_hash(0xB1);
    let b = ext_hash(0xC2);

    for (byte, hash, ts) in [(0xB1u8, a, 1u64), (0xC2, b, 2)] {
        let delta =
            delta_with_set_field_external(SECTION_ID, FIELD_A, ext_hash_vec(byte), EXT_SIZE, 1, ts);
        pre_store(&ext, &[hash]).await;
        ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", ts)
            .await
            .expect("the external commit must land on cluster");
    }

    assert_eq!(
        pending_members(&tv).await.len(),
        1,
        "the superseded hash must reach the `{{gc}}`-tagged pending set from a commit whose \
         other keys live on two other slots"
    );
    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.deleted, 1);
    assert_blob_gone(&tv, a, "the superseded hash, on its own shard").await;
    assert_blob_alive(&tv, b, "the live hash, on a possibly-different shard").await;
}

/// Cluster sibling of arm 6: two objects on (possibly) different shards retiring
/// on one hash still owe two distinct decrements.
///
/// The `{gc}` hash tag co-locates `pending` with `inflight` on one slot, so both
/// retirements' SADDs land on the same shard whichever shard owns either object.
/// What the cluster adds is the per-hash UNLINK: the decrement runs on `{gc}`'s
/// shard while `ext:meta` / `ext:data` are deleted on `{hash_hex}`'s, and a
/// regression that dropped the `{hash_hex}` tag would land the UNLINK on the
/// wrong shard and silently leak the entries.
///
/// Gated on `LIVE_EDIT_TEST_CLUSTER_URL`; CI-proven only.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cluster_two_objects_retiring_on_one_hash_owe_two_distinct_decrements() {
    let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
        eprintln!("SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)");
        return;
    };
    let tv = TestValkey::setup_with_url(cluster_url, 4).await;
    assert!(
        tv.client().pool.next().is_clustered(),
        "cluster URL required for the cluster collapse regression"
    );

    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, UpdateLayerBackend::Lua, &metrics);
    let ext = ext_store(&tv);
    let session_id = create_session(&tv, "qpf-cluster-collapse").await;

    let h = ext_hash(0xD3);
    let mut oids = Vec::new();
    for layer in ["qpf-cluster-collapse-one", "qpf-cluster-collapse-two"] {
        let oid = ls
            .create_layer(
                &session_id,
                layer,
                "b-qpf-user",
                &empty_state(),
                ext.clone(),
            )
            .await
            .expect("create_layer must succeed on cluster");
        let delta =
            delta_with_set_field_external(SECTION_ID, FIELD_A, ext_hash_vec(0xD3), EXT_SIZE, 1, 0);
        pre_store(&ext, &[h]).await;
        ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 1)
            .await
            .expect("the external commit must land on cluster");
        oids.push(oid);
    }
    assert_eq!(ref_count(&tv, h).await, Some(2));

    for layer in ["qpf-cluster-collapse-one", "qpf-cluster-collapse-two"] {
        ls.delete_layer(&session_id, layer, ext.clone())
            .await
            .expect("delete_layer must succeed on cluster");
    }
    ls.wait_for_cleanup(CLEANUP_JOIN_BUDGET).await;

    let mut expected = vec![retirement_member(h, oids[0]), retirement_member(h, oids[1])];
    expected.sort();
    assert_eq!(
        pending_members(&tv).await,
        expected,
        "two owed decrements must be two members on cluster too"
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 2);
    assert_blob_gone(
        &tv,
        h,
        "a hash retired by two objects, UNLINKed on its own shard",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 14. the leak signal — a release batch that cannot be enqueued
// ═════════════════════════════════════════════════════════════════════════════
//
// Every arm above assumes a healthy release channel — the enqueue succeeding is
// a precondition of what those arms were really testing, never the claim. The
// two bodies below make the channel's state an explicit parameter
// ([`PendingSetState`]) and run each of them over BOTH values: a `Planted` arm,
// where a wrong-typed key at the pending set's address makes every `SADD` fail,
// and a `Healthy` control on the same fixture and the same commit.
//
// The failure arm is the one condition under which b.qpf's fix silently does not
// happen, and therefore the one the operator has to be able to see. The control
// is what makes each failure reading a DIFFERENCE rather than an absolute: it
// pins the same owed batch being counted as releases on an intact channel, so
// neither "the failure counter moved by 2" nor "no release was counted" can be
// satisfied by a build that simply ticks the wrong counter unconditionally.
//
// ── How the failure is forced ────────────────────────────────────────────────
//
// By planting a STRING at `le:ext:{gc}:pending`, the address every release
// SADDs to. `SADD` against a string is `WRONGTYPE` — a standing condition, not a
// blip — so it fails on the first attempt and on the in-line retry, which is
// what makes the commit-path arm reach its terminal `Err` deterministically.
// Nothing is mocked and no test-only seam is added: the failure is a real fred
// error surfacing from a real Valkey, so the tick count these arms read is
// produced by exactly the code production runs.
//
// The plant is invisible to everything else the commit touches: the fast-path
// script never names the pending set (it lives on the `{gc}` slot, so it cannot
// be in a `{object_id}`-tagged script's KEYS at all — the script returns release
// TOKENS and the Rust caller enqueues them), the slow path's keyspace probe
// covers only the keys the commit itself writes, and the pre-flight
// `store_values` writes `ext:meta` / `ext:data`. So the only thing the plant can
// break is the enqueue, which is the point.
//
// ── What these arms pin, and what they deliberately do not ───────────────────
//
// They pin the b.qpf narrowing **for the COMMIT path**, which b.yda did not
// change: an enqueue failure there is VISIBLE (counted, in the per-reference
// unit) but not RECOVERABLE — nothing retries it beyond the single in-line
// attempt, and the commit has already landed, been reported and been published,
// so failing the caller is not available either. What separates an enqueue
// failure from a pre-schedule read failure is therefore MEASURABILITY, not
// recoverability: only here is the owed count known, and ticked.
//
// The RETIREMENT path's arms (section 15 below) used to make the same claim and
// no longer do. b.yda gave that path a durable retry carrier on the object's own
// Cluster slot, so its twice-failed enqueue is deferred rather than lost and its
// leak signal stays still; see section 15's own header. Nothing about the
// commit-path arms here changed with it — a commit has no object-slot teardown
// to sequence a carrier against, because the object it commits to is still very
// much alive.

/// Whether an arm plants the wrong-typed key that breaks the release enqueue.
#[derive(Clone, Copy, Debug)]
enum PendingSetState {
    /// A string sits at the pending set's address; every `SADD` is `WRONGTYPE`.
    Planted,
    /// The pending set is a real Set (or absent); releases enqueue normally.
    Healthy,
}

impl PendingSetState {
    fn label(self) -> &'static str {
        match self {
            PendingSetState::Planted => "planted",
            PendingSetState::Healthy => "healthy",
        }
    }
}

/// The string planted over the pending set. Its content is arbitrary; only its
/// TYPE matters.
const PLANTED_PENDING_STRING: &str = "planted-by-an-out-of-band-writer";

/// A commit superseding TWO hashes accounts for exactly two owed REFERENCES,
/// and lands either way: two ticks of
/// `live_edit_ext_ref_release_enqueue_failed_total` when the release channel is
/// broken, two ticks of `live_edit_ext_ref_released_total{reason="rewrite"}`
/// when it is intact.
///
/// Run over [`PendingSetState`] as well as over the backend. The two arms share
/// one fixture and one superseding commit, so the only thing that differs
/// between their readings is whether the pending set is writable.
///
/// Claims on the `Planted` arm, each of which has been wrong at some point in
/// this fix's history:
///
/// * **The commit is not failed for the enqueue.** By the time the enqueue runs
///   the commit has landed, been reported to the caller and been published to
///   subscribers; surfacing the failure would make the caller retry a delta that
///   is already applied. The `expect` on `update_layer` is that claim.
/// * **The tick is per REFERENCE, not per call.** The two counters have to be
///   readable in the same unit — an operator watches
///   `enqueue_failed / (released + enqueue_failed)` to see how much of the
///   release channel is being lost — and a per-call counter makes that ratio
///   meaningless. The arm supersedes TWO hashes in ONE commit precisely so the
///   two units disagree: a per-call tick reads 1 here, the correct one reads 2.
/// * **A failed enqueue must not be counted as a release.** `released_total` is
///   read as "references handed back"; ticking it for a batch that never
///   reached the channel would report reclamation that did not happen and hide
///   the leak the other counter is trying to show. Asserted across ALL THREE
///   reason cells, not just `rewrite` — a build that ticked the wrong cell would
///   satisfy a single-cell check.
///
/// Claims on the `Healthy` arm, each the mirror of one above: the failure
/// counter does not move at all (without it the `Planted` arm's `+2` is
/// satisfiable by a build that ticks on every commit), the same two owed
/// references ARE counted as two releases (without it the `Planted` arm's
/// `released == 0` is satisfiable by a build that never releases anything), both
/// owed decrements reach `pending` as their own members, and one GC tick
/// reclaims both blobs — the end-to-end proof that the references were really
/// handed back rather than merely counted.
///
/// The index assertion is shared by both arms and belongs to a different layer:
/// `obj:ext` is written INSIDE the commit's atomic unit, so it must reflect the
/// post-commit state whatever the enqueue that follows it did. An empty index
/// with a failed enqueue is precisely the leaked state — the object has given
/// the hashes up and nothing will ever hand the references back — and that is
/// what makes the counter the only remaining evidence.
///
/// Run on both backends: the two commit paths reach the enqueue by different
/// routes (the slow path passes `ext_plan.releases` directly, the fast path
/// decodes the script's release tokens first), so the batch handed to the
/// terminal `Err` arm is assembled twice and can be sized wrongly once.
///
/// **Fails pre-fix.** Before the per-reference correction the terminal arm added
/// `1` rather than `payload.len()`, so the `Planted` variant read
/// `failed_before + 1` against the required `+ 2`.
async fn release_enqueue_outcome_is_counted_once_per_owed_reference(
    backend: UpdateLayerBackend,
    pending: PendingSetState,
) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let layer = format!("qpf-enqueue-fail-{}", pending.label());
    let session_id = create_session(&tv, &layer).await;
    let oid = ls
        .create_layer(
            &session_id,
            &layer,
            "b-qpf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let a = ext_hash(0x1D);
    let b = ext_hash(0x2E);

    // ── Commit 1: two rows, two DISTINCT hashes, one pre-flight `+1` each ───
    let delta =
        delta_with_two_fields_holding_distinct_externals(ext_hash_vec(0x1D), ext_hash_vec(0x2E));
    assert_fast_path_eligible(&delta, "the two-hash external write");
    pre_store(&ext, &[a, b]).await;
    ls.update_layer_relayed(&session_id, &layer, oid, Some(delta), "qpf-writer", 1)
        .await
        .expect("the two-hash external commit must land");

    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0x1D), 1), (ext_hash_hex(0x2E), 1)],
        "fixture precondition: the object holds TWO distinct hashes, so the commit below owes \
         TWO references and a per-call tick is distinguishable from a per-reference one"
    );
    assert!(
        pending_members(&tv).await.is_empty(),
        "fixture precondition: first-time acquisitions owe nothing, so the pending set is empty \
         and safe to overwrite with the plant below"
    );

    // ── Break the release channel (or leave it alone, on the control) ───────
    let pending_key = tv.key_schema().ext_pending();
    if matches!(pending, PendingSetState::Planted) {
        let _: () = tv
            .client()
            .pool
            .set(&pending_key, PLANTED_PENDING_STRING, None, None, false)
            .await
            .expect("planting a string over the pending set must succeed");
    }

    let failed_before = ext_ref_release_enqueue_failed(&metrics);

    // ── Commit 2: BOTH rows go inline, superseding both hashes at once ──────
    let delta = delta_with_two_inline_fields(2);
    assert_fast_path_eligible(&delta, "the two-hash superseding rewrite");
    let (seq, _) = ls
        .update_layer_relayed(&session_id, &layer, oid, Some(delta), "qpf-writer", 2)
        .await
        .expect(
            "the commit must LAND even when its release enqueue cannot. By the time the enqueue \
             runs the commit is applied, reported and published, so failing the caller here would \
             make it retry a delta that is already in the keyspace — strictly worse than the leak",
        );

    assert!(
        obj_ext(&tv, oid).await.is_empty(),
        "both rows gave their hash up, so the index must be empty regardless of the enqueue. \
         `obj:ext` is written INSIDE the commit's atomic unit; the enqueue is a separate \
         publication that follows it and cannot reach back into the commit"
    );

    match pending {
        PendingSetState::Planted => {
            assert_eq!(
                ext_ref_release_enqueue_failed(&metrics),
                failed_before + 2,
                "TWO references were owed and none was recorded, so the leak signal must read 2. \
                 A 1 here is the per-CALL unit: it makes the counter incomparable with \
                 `live_edit_ext_ref_released_total`, which counts references, and an operator's \
                 loss ratio silently under-reports by the batch size"
            );
            for reason in ALL_RELEASE_REASONS {
                assert_eq!(
                    ext_ref_released(&metrics, reason),
                    0,
                    "no release reached the channel, so `reason=\"{reason}\"` must not move. \
                     Counting a failed batch as released would report reclamation that never \
                     happened and mask exactly the leak the failure counter is reporting"
                );
            }

            let survivor: Option<String> = tv.client().pool.get(&pending_key).await.expect(
                "GET must succeed against the planted key — a WRONGTYPE here would mean the \
                     store REPLACED the wrong-typed key with a Set of its own, which would make \
                     the enqueue succeed and this whole arm test nothing",
            );
            assert_eq!(
                survivor.as_deref(),
                Some(PLANTED_PENDING_STRING),
                "the store must leave an out-of-band writer's key exactly as it found it. \
                 Repairing it by DEL + SADD would destroy every owed decrement already queued \
                 there by other objects"
            );

            // Remove the plant before the fixture drops, so no later GC tick —
            // here or in a shared-keyspace neighbour — trips over it.
            let removed: u64 = tv
                .client()
                .pool
                .del(&pending_key)
                .await
                .expect("DEL of the planted key must succeed");
            assert_eq!(removed, 1, "the plant must actually have been removed");
        }
        PendingSetState::Healthy => {
            assert_eq!(
                ext_ref_release_enqueue_failed(&metrics),
                failed_before,
                "the control arm's enqueue succeeds, so the leak signal must not move at all. \
                 Without this the `Planted` arm's `+2` could be produced by a build that ticks \
                 the counter on every commit"
            );
            assert_eq!(
                ext_ref_released(&metrics, REASON_REWRITE),
                2,
                "the same two owed references, on an intact channel, are two RELEASES. This is \
                 what makes the `Planted` arm's `released == 0` a real difference rather than a \
                 build that never releases anything"
            );
            let mut expected = vec![commit_member(a, oid, seq), commit_member(b, oid, seq)];
            expected.sort();
            assert_eq!(
                pending_members(&tv).await,
                expected,
                "both owed decrements reach the channel when it is intact"
            );

            let stats = process_pending_gc(
                ext.as_ref(),
                None,
                live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            )
            .await;
            assert_eq!(stats.deleted, 2, "both blobs reach ref_count 0");
            assert_blob_gone(&tv, a, "the first superseded hash").await;
            assert_blob_gone(&tv, b, "the second superseded hash").await;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_a_failed_release_enqueue_is_counted_once_per_owed_reference() {
    release_enqueue_outcome_is_counted_once_per_owed_reference(
        UpdateLayerBackend::Lua,
        PendingSetState::Planted,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_a_failed_release_enqueue_is_counted_once_per_owed_reference() {
    release_enqueue_outcome_is_counted_once_per_owed_reference(
        UpdateLayerBackend::Txn,
        PendingSetState::Planted,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_an_intact_release_channel_counts_two_releases_and_no_failure() {
    release_enqueue_outcome_is_counted_once_per_owed_reference(
        UpdateLayerBackend::Lua,
        PendingSetState::Healthy,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_an_intact_release_channel_counts_two_releases_and_no_failure() {
    release_enqueue_outcome_is_counted_once_per_owed_reference(
        UpdateLayerBackend::Txn,
        PendingSetState::Healthy,
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 15. the retirement path — deferral when the channel is broken
// ═════════════════════════════════════════════════════════════════════════════
//
// Same two-arm shape as section 14, different terminal state on the failure
// arm. The retirement path is the one place where the owed count is known AND
// the object's own Cluster slot is still answering, so b.yda keeps the
// obligation there (`le:{oid}:obj:extret`, written by the retirement's begin
// before it enqueues since b.yuo Part R) rather than losing it. The failure
// arm therefore asserts a RECOVERY — a carrier and its attempt token that
// outlive the teardown, a work-list entry, a still leak signal — where section
// 14's asserts a loss.

/// A retirement owing TWO distinct hashes accounts for exactly two references,
/// and DELETES the object either way: two ticks of the DEFERRAL counter (and a
/// carrier holding both members) when the release channel is broken, two ticks
/// of `live_edit_ext_ref_released_total{reason="retirement"}` — and two
/// reclaimed blobs — when it is intact.
///
/// Run over [`PendingSetState`], for the reason the section header gives: the
/// `Planted` arm is the failure, the `Healthy` arm is the control that makes it
/// a difference. Both run the same three-row fixture and the same
/// `delete_layer`.
///
/// The counting unit is the load-bearing half of BOTH arms, and the fixture is
/// built to separate it from the plausible wrong one: the object has THREE value
/// rows but only TWO distinct hashes (two field rows share one hash, a time
/// sample holds the other). `ref_count` is per OBJECT per hash, so the object
/// owes two decrements, not three; a counter sized off rows would read 3 and
/// overstate the leak by 50% on exactly the workload — a mesh referenced from
/// several attributes — where the ratio matters most. The `Healthy` arm pins the
/// same unit on the success side, where the count is a literal argument
/// (`releases.len()`) rather than a batch length: with only single-hash
/// retirements elsewhere in this file, a success tick hard-coded to `1` would
/// otherwise pass the whole suite.
///
/// ── What the `Planted` arm pins (re-pointed by b.yda) ───────────────────────
///
/// It used to pin the b.qpf narrowing: a retirement enqueue failure was made
/// VISIBLE (`ext_ref_release_enqueue_failed_total`) but not RECOVERABLE, and the
/// `scan_object_keys` assertion proved the point by showing the structural
/// cleanup had removed every key that could have re-derived the owed hashes.
///
/// b.yda made it recoverable, so this arm now pins the recovery instead. The
/// owed members are on `le:{oid}:obj:extret` — on the OBJECT's Cluster slot,
/// which is the whole design: what failed is a `{gc}`-slot enqueue, and the
/// natural Cluster shape of that is "this object's slot answered every read,
/// the `{gc}` slot did not", so a second `{gc}`-slot write would recover
/// nothing. Since b.yuo Part R the retirement's begin writes them there, with
/// the attempt token on `le:{oid}:obj:extret:attempts`, BEFORE either attempt,
/// and a deferral simply leaves them. Neither key is in
/// `KeySchema::object_cleanup_keys` and neither carries a TTL, so the teardown
/// that follows leaves them standing, and the orphan sweep's retire-retry phase
/// settles them later.
///
/// Four claims, each of which a plausible wrong implementation satisfies but
/// not all four:
///
/// * **The carrier OUTLIVES the teardown.** The `scan_object_keys` assertion is
///   kept and inverted: `obj:extret` and its token HASH join `obj:stream` as
///   the only permitted survivors. A carrier listed in the cleanup inventory
///   would be UNLINKed in the same breath as the header, which is the pre-b.yda
///   leak wearing a new key name. (The `Healthy` arm pins the converse: after a
///   landed enqueue the retirement's own clear removed both, so only
///   `obj:stream` may survive.)
/// * **The leak signal does NOT move.** It is alertable at
///   `increase(...) > 0`, so ticking it for an obligation that is durably parked
///   pages an operator for a recovery in progress. This is the exact assertion
///   that read `failed_before + 2` before b.yda.
/// * **The deferral counter moves by the same per-REFERENCE unit** the old leak
///   signal used — two distinct hashes over three rows, so a row-sized count
///   reads 3 and a per-call one reads 1.
/// * **`released` stays still on every cell.** The obligation is parked, not
///   paid; a `retirement` tick here would report a reclamation that has not
///   happened yet, and `reconcile` is swept too so a build ticking the wrong
///   cell cannot pass.
///
/// The header is gone by the time these assertions run, so `ext_gc_scheduled`
/// cannot be read back here — the deferral's atomic flag write is pinned
/// instead by `storage::layer`'s own in-crate
/// `a_twice_failed_retirement_enqueue_parks_its_owed_references_on_the_carrier`,
/// which calls `schedule_old_object_ext_gc` directly and reads the surviving
/// header.
///
/// Not parameterised by backend: the retirement release is
/// `schedule_old_object_ext_gc`, which no commit backend participates in — a
/// second variant would re-run identical code under a different name.
///
/// **Fails without the b.yda fix.** The pre-b.yda build writes no carrier, ticks
/// the leak signal by 2 and leaves the work list empty — three of the four
/// claims above, in the opposite direction.
async fn retirement_release_enqueue_outcome_is_counted_once_per_distinct_owed_hash(
    pending: PendingSetState,
) {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-qpf"));
    let ls = layer_store(&tv, UpdateLayerBackend::Lua, &metrics);
    let ext = ext_store(&tv);
    let layer_name = format!("qpf-retire-enqueue-{}", pending.label());
    let layer = layer_name.as_str();
    let session_id = create_session(&tv, layer).await;
    let oid = ls
        .create_layer(
            &session_id,
            layer,
            "b-qpf-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let a = ext_hash(0x3B);
    let b = ext_hash(0x4D);

    // Three rows, two distinct hashes: both field rows hold A, the time sample
    // holds B.
    let mut delta = delta_with_two_fields_sharing_one_external(ext_hash_vec(0x3B));
    delta.time_samples =
        delta_with_time_sample_external(SECTION_ID, SAMPLE_TIME, ext_hash_vec(0x4D), EXT_SIZE, 2)
            .time_samples;
    assert_fast_path_eligible(&delta, "the three-row two-hash external write");
    pre_store(&ext, &[a, b]).await;
    ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "qpf-writer", 1)
        .await
        .expect("the external commit must land");

    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0x3B), 2), (ext_hash_hex(0x4D), 1)],
        "fixture precondition: THREE rows, TWO distinct hashes — the shape that tells a \
         per-distinct-hash count apart from a per-row one"
    );
    assert!(
        pending_members(&tv).await.is_empty(),
        "fixture precondition: first-time acquisitions owe nothing, so the pending set is empty — \
         safe to overwrite with the plant, and every member counted below comes from the \
         retirement"
    );

    // ── Break the release channel (or leave it alone, on the control) ───────
    let pending_key = tv.key_schema().ext_pending();
    if matches!(pending, PendingSetState::Planted) {
        let _: () = tv
            .client()
            .pool
            .set(&pending_key, PLANTED_PENDING_STRING, None, None, false)
            .await
            .expect("planting a string over the pending set must succeed");
    }

    let failed_before = ext_ref_release_enqueue_failed(&metrics);

    ls.delete_layer(&session_id, layer, ext.clone())
        .await
        .expect(
            "delete_layer must SUCCEED whatever the release channel did. All three retirement \
             callers treat the release enqueue as best-effort and go on to UNLINK: permanent \
             orphan RETENTION — an object whose keys survive because its ext-value bookkeeping \
             failed — is a worse failure than an inflated `ref_count`",
        );
    ls.wait_for_cleanup(CLEANUP_JOIN_BUDGET).await;

    // The object's keys that outlived the teardown, prefix stripped.
    let full_prefix = format!("{}{{{oid}}}:", tv.prefix());
    let mut survivors: Vec<String> = crate::common::scan_object_keys(tv.client(), tv.prefix(), oid)
        .await
        .into_iter()
        .map(|key| {
            let key = key.as_str_lossy().into_owned();
            key.strip_prefix(&full_prefix).unwrap_or(&key).to_string()
        })
        .collect();
    survivors.sort();

    match pending {
        PendingSetState::Planted => {
            assert_eq!(
                ext_ref_release_deferred(&metrics),
                2,
                "TWO distinct hashes were owed, so the deferral counter must read 2. A 3 would \
                 mean the count was taken off ROWS — `ref_count` is per object per hash, so the \
                 object owed two decrements however many of its rows named them, and a row-sized \
                 count overstates the obligation. A 0 means the deferral never happened at all, \
                 which is the pre-b.yda build"
            );
            assert_eq!(
                ext_ref_release_enqueue_failed(&metrics),
                failed_before,
                "and the LEAK signal must NOT move: the obligation is durably parked on the \
                 object's own carrier and the orphan sweep's retire-retry phase will re-drive it. \
                 The signal is alertable at `increase(...) > 0`, so a tick here pages an operator \
                 for a recovery in progress. `failed_before + 2` is the pre-b.yda reading, when \
                 the batch really was lost the moment the teardown landed"
            );
            for reason in ALL_RELEASE_REASONS {
                assert_eq!(
                    ext_ref_released(&metrics, reason),
                    0,
                    "nothing reached the channel, so `reason=\"{reason}\"` must not move. A \
                     deferred obligation is parked, not paid — counting it as released would \
                     report a reclamation that has not happened and would double-count once the \
                     sweep really does hand it back under `retirement`"
                );
            }

            // ── The carrier OUTLIVES the teardown, which is the whole fix ───
            let leaked: Vec<&String> = survivors
                .iter()
                .filter(|k| *k != "obj:stream" && *k != "obj:extret" && *k != "obj:extret:attempts")
                .collect();
            assert!(
                leaked.is_empty(),
                "the callers UNLINK the object's keys whether or not the enqueue succeeded, and \
                 after a deferral exactly three keys may outlive retirement: `obj:stream` \
                 (PEXPIRE'd with grace for lagging subscribers), `obj:extret`, the carrier holding \
                 the owed releases (b.yda), and `obj:extret:attempts`, the token of the attempt \
                 whose outcome is unknown (b.yuo Part R). Anything else is retained orphan state. \
                 Leaked: {leaked:?}"
            );
            assert!(
                survivors.iter().any(|k| k == "obj:extret")
                    && survivors.iter().any(|k| k == "obj:extret:attempts"),
                "and the carrier and its token HASH must actually BE among them. Both are \
                 deliberately excluded from `KeySchema::object_cleanup_keys` and carry no TTL, \
                 precisely so they outlive the object whose teardown just ran — with `obj:hdr`, \
                 `obj:ext` and every value row gone, the carrier is the only thing left that can \
                 name the two owed hashes, and the token is what lets the sweep ask whether either \
                 attempt ran before it sends them again. Survivors: {survivors:?}"
            );

            let mut expected_parked = vec![retirement_member(a, oid), retirement_member(b, oid)];
            expected_parked.sort();
            assert_eq!(
                retire_carrier_members(&tv, oid).await,
                expected_parked,
                "the carrier holds the formatted pending members VERBATIM, one per DISTINCT hash. \
                 The sweep's drain re-enqueues exactly these bytes — byte-identity is what makes a \
                 duplicate drain collapse in the pending Set rather than owe a second decrement"
            );
            assert_eq!(
                retire_retry_entries(&tv).await,
                vec![oid.to_string()],
                "and the object is on the `{{gc}}`-slot work list, so the sweep finds the carrier \
                 without a keyspace walk. The carrier is the record and this is only an index into \
                 it — hence carrier-then-work-list, never the reverse — but a missing entry means \
                 every drain waits on the backstop SCAN"
            );

            let survivor: Option<String> = tv.client().pool.get(&pending_key).await.expect(
                "GET must succeed against the planted key — the retirement path must not have \
                 replaced it",
            );
            assert_eq!(
                survivor.as_deref(),
                Some(PLANTED_PENDING_STRING),
                "an out-of-band writer's key must be left exactly as it was found"
            );

            let removed: u64 = tv
                .client()
                .pool
                .del(&pending_key)
                .await
                .expect("DEL of the planted key must succeed");
            assert_eq!(removed, 1, "the plant must actually have been removed");
        }
        PendingSetState::Healthy => {
            assert!(
                survivors.iter().all(|k| k == "obj:stream"),
                "after a retirement whose enqueue landed, only `obj:stream` (PEXPIRE'd with grace) \
                 may outlive the teardown: the retirement's own clear removed the carrier and the \
                 token HASH it wrote before the enqueue (b.yuo Part R). A surviving carrier would \
                 cost the sweep a resolve per retirement and count the obligation twice in the \
                 reconciler's owed walk. Survivors: {survivors:?}"
            );
            assert_eq!(
                ext_ref_release_enqueue_failed(&metrics),
                failed_before,
                "the control arm's enqueue succeeds, so the leak signal must not move at all. \
                 Without this the `Planted` arm's `+2` could be produced by a build that ticks the \
                 counter on every retirement"
            );
            assert_eq!(
                ext_ref_released(&metrics, REASON_RETIREMENT),
                2,
                "the same TWO owed references, on an intact channel, are two RELEASES. The success \
                 tick takes its count as a literal argument rather than a batch length, so a \
                 hard-coded `1` reads 1 here — and every other retirement in this file owes \
                 exactly one hash, which is precisely why that mutant needs a two-hash retirement \
                 to be caught at all"
            );
            let mut expected = vec![retirement_member(a, oid), retirement_member(b, oid)];
            expected.sort();
            assert_eq!(
                pending_members(&tv).await,
                expected,
                "one member per DISTINCT hash, each naming the retiring object and the `ret` \
                 event. Three members would mean the union counted A once per row and is about to \
                 over-release"
            );

            let stats = process_pending_gc(
                ext.as_ref(),
                None,
                live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            )
            .await;
            assert_eq!(
                stats.deleted, 2,
                "both blobs reach ref_count 0 — the counted releases really were handed back, not \
                 merely tallied"
            );
            assert_blob_gone(&tv, a, "the hash two of the retired object's rows held").await;
            assert_blob_gone(&tv, b, "the hash its time sample held").await;
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_retirement_enqueue_defers_each_owed_hash_onto_the_surviving_carrier() {
    retirement_release_enqueue_outcome_is_counted_once_per_distinct_owed_hash(
        PendingSetState::Planted,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_intact_retirement_channel_counts_one_release_per_distinct_owed_hash() {
    retirement_release_enqueue_outcome_is_counted_once_per_distinct_owed_hash(
        PendingSetState::Healthy,
    )
    .await;
}
