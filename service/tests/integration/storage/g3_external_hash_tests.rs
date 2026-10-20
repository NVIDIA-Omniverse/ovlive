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

//! G3 — an external value's hash is exactly 32 bytes, on the way in and on the
//! way out (Spec SDD SR-5.3).
//!
//! **Write side.** A `SetField` / `TimeSample` with `ext_value_size > 0` whose
//! `value_or_ext_hash` was not 32 bytes used to be zero-padded or truncated by
//! the Lua fast path's encoder, so the commit landed a value complete-mode
//! `apply_delta` rejects — and neither stream replay nor checkpoint compose could
//! reproduce the stored state. `validate_delta_bounds`, which
//! `LayerStore::update_layer` runs ahead of both backends, now refuses it
//! `InvalidArgument`, and the refusal leaves the object's keyspace
//! byte-identical: no row, no `obj:ext` change, no stream entry, no header change
//! (`seq` included). Pinned on both backends, for both row kinds, at 31 and 33
//! bytes (the pad and the truncate shapes). The blob the gRPC handler stores
//! ahead of `update_layer` is pinned in `grpc/update_layer_test.rs` and
//! `grpc/create_layer_test.rs`.
//!
//! **Read side.** A stored row's `hash` column was decoded by a pad-or-truncate
//! hex decoder that panicked on an odd-length or non-ASCII string and decoded
//! a `+`-carrying one to a wrong hash. It now decodes with the strict
//! `hash_from_hex`, and a malformed column fails the read `ServiceError::Internal`
//! — through the test-support reader `read_object_state_for_test` (which
//! shares the snapshot reads' row decoders), through `read_object_snapshot` on
//! the atomic route (catch-up, checkpoint), through the slim reads on both
//! routes (the structural update's unfenced fan-out, the structural cache's
//! cold start and audit), and through external-value clean-up's reference
//! walk (`storage::retirement::walk_object_references`, b.fsz MR-D4), which
//! must never release a hash it cannot decode. A row whose inline `value`
//! column is present still decodes as that value, whatever its `hash` column
//! holds.

use std::collections::HashMap;

use fred::interfaces::{HashesInterface, KeysInterface};
use oodf_ffi::OwnedDelta;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::{FieldValue, ObjectState};
use live_edit::error::ServiceError;
use live_edit::storage::diff_read::{
    read_object_snapshot, read_object_snapshot_slim, SlimSnapshotReadConsumer, SnapshotRead,
    SnapshotReadConsumer, SnapshotReadOf, SnapshotReadPath,
};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::is_fast_path;
use live_edit::storage::retirement::{walk_object_references, RetirementWalk};
use live_edit::storage::test_support::read_object_state_for_test;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::{delta_with_set_field_external, delta_with_time_sample_external};
use crate::storage::snapshot_read_atomic_test::{
    create, create_session, every_shape_state, layer_store, CUBE, WORLD,
};

// ═════════════════════════════════════════════════════════════════════════════
// Write side — a wrong-length hash is refused before anything is stored
// ═════════════════════════════════════════════════════════════════════════════

/// The section every write-side row hangs off.
const SECTION_ID: u64 = 1;
/// The field row's key.
const FIELD_KEY: &str = "tex";
/// The time-sample row's time.
const SAMPLE_TIME: f64 = 2.5;
/// The external value's declared size. Arbitrary but nonzero: zero would make
/// the bytes an inline value, which is never length-checked.
const EXT_SIZE: u64 = 1_024;
/// The two wrong lengths: one short (the pre-fix zero-pad shape) and one long
/// (the pre-fix truncate shape).
const WRONG_HASH_LENS: [usize; 2] = [31, 33];

#[derive(Clone, Copy, Debug)]
enum RowKind {
    Field,
    TimeSample,
}

impl RowKind {
    const ALL: [RowKind; 2] = [RowKind::Field, RowKind::TimeSample];

    fn external_delta(self, hash: Vec<u8>) -> OwnedDelta {
        match self {
            RowKind::Field => {
                delta_with_set_field_external(SECTION_ID, FIELD_KEY, hash, EXT_SIZE, 1, 0)
            }
            RowKind::TimeSample => {
                delta_with_time_sample_external(SECTION_ID, SAMPLE_TIME, hash, EXT_SIZE, 1)
            }
        }
    }
}

/// The shape `create_layer` materializes for a fresh layer whose content
/// arrives by `update_layer`.
fn empty_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        ..Default::default()
    }
}

/// Every object-scoped key a one-row fast-path or txn commit writes, plus the
/// row keys both deltas here would create.
///
/// Enumerated rather than `SCAN`ned, for the reason `b_459_tests::witness_keys`
/// gives: an explicit list names the keys a half-commit would have written.
fn witness_keys(keys: &KeySchema, oid: u64) -> Vec<String> {
    let oid_str = oid.to_string();
    let sid = SECTION_ID.to_string();
    vec![
        keys.obj_header(&oid_str),
        keys.layer_stream(oid),
        keys.obj_fields_index(&oid_str),
        keys.obj_time_samples_index(&oid_str),
        keys.obj_field(&oid_str, &sid, FIELD_KEY),
        keys.obj_time_sample(&oid_str, &sid, SAMPLE_TIME),
        keys.obj_ext_index(&oid_str),
        keys.obj_write_token(&oid_str),
        // The commit release ring: written beside the commit-token record by
        // a commit that owes external-value releases.
        keys.obj_write_releases(&oid_str),
        keys.obj_sections_index(&oid_str),
        keys.obj_children_index(&oid_str),
    ]
}

/// `DUMP` of every witness key: type-agnostic, `nil` for an absent key, so one
/// before/after equality covers "this hash did not change", "this key was not
/// created" and "no stream entry was appended".
async fn dump_witness(tv: &TestValkey, oid: u64) -> Vec<(String, Option<Vec<u8>>)> {
    let mut out = Vec::new();
    for key in witness_keys(tv.key_schema(), oid) {
        let dumped: Option<Vec<u8>> = tv
            .client()
            .pool
            .dump(&key)
            .await
            .expect("DUMP must succeed");
        out.push((key, dumped));
    }
    out
}

async fn header_seq(tv: &TestValkey, oid: u64) -> String {
    tv.client()
        .pool
        .hget(tv.key_schema().obj_header(&oid.to_string()), "seq")
        .await
        .expect("HGET obj:hdr seq must succeed")
}

async fn wrong_length_hash_is_refused_and_stores_nothing(backend: UpdateLayerBackend) {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend);
    let layer = "g3-write";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &empty_state()).await;

    for kind in RowKind::ALL {
        for len in WRONG_HASH_LENS {
            let what = format!("{backend:?}, {kind:?}, {len}-byte hash");
            let delta = kind.external_delta(vec![0xAB; len]);
            assert!(
                is_fast_path(&Some(delta.clone())),
                "{what}: fixture precondition — the delta must be fast-path eligible, or the \
                 Lua arm never reaches the encoder that used to pad it"
            );
            let before = dump_witness(&tv, oid).await;
            let seq_before = header_seq(&tv, oid).await;

            let result = ls
                .update_layer_relayed(&session_id, layer, oid, Some(delta), "g3-client", 7_000)
                .await;

            match result {
                Err(ServiceError::InvalidArgument(msg)) => assert_eq!(
                    msg,
                    format!(
                        "delta error: invalid external-value hash length: got {len} bytes, \
                         expected 32"
                    ),
                    "{what}: the refusal must be the hash-length one"
                ),
                Ok(v) => panic!(
                    "{what}: must be refused INVALID_ARGUMENT, never committed (pre-fix the fast \
                     path padded / truncated the hash and committed); got Ok({v:?})"
                ),
                Err(other) => panic!("{what}: expected InvalidArgument, got {other:?}"),
            }
            assert_eq!(
                header_seq(&tv, oid).await,
                seq_before,
                "{what}: `seq` must not advance"
            );
            assert_eq!(
                dump_witness(&tv, oid).await,
                before,
                "{what}: the object's keyspace must be byte-identical — no row, no `obj:ext` \
                 change, no stream entry, no header change"
            );
        }
    }

    // Control: the same row with a 32-byte hash commits and moves the witness,
    // so the byte-identical comparisons above are not vacuous.
    let before = dump_witness(&tv, oid).await;
    ls.update_layer_relayed(
        &session_id,
        layer,
        oid,
        Some(RowKind::Field.external_delta(vec![0xAB; 32])),
        "g3-client",
        7_001,
    )
    .await
    .unwrap_or_else(|e| panic!("{backend:?}: a 32-byte external hash must commit; got {e:?}"));
    assert_ne!(
        dump_witness(&tv, oid).await,
        before,
        "{backend:?}: the control commit must change the witnessed keys"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_wrong_length_external_hash_returns_invalid_argument_and_stores_nothing() {
    wrong_length_hash_is_refused_and_stores_nothing(UpdateLayerBackend::Lua).await;
}

/// The txn backend's `apply_delta` refused this delta even before the fix (its
/// `field_value_from_oodf` checks the length), so this arm is backend parity:
/// the same status and message as the fast path, before any write.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_wrong_length_external_hash_returns_invalid_argument_and_stores_nothing() {
    wrong_length_hash_is_refused_and_stores_nothing(UpdateLayerBackend::Txn).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Read side — a malformed stored `hash` column fails the read `Internal`
// ═════════════════════════════════════════════════════════════════════════════

/// The message fragment every route's `Internal` carries.
const MALFORMED_HASH_MSG: &str = "malformed external-value `hash` column";

/// Stored `hash` columns no writer produces. The odd-length and non-ASCII
/// strings panicked the old decoder (it sliced `hex[i..i + 2]`); the `+`
/// string decoded to a wrong hash (`u8::from_str_radix("+a", 16)` is `0x0a`).
fn malformed_hash_columns() -> [(&'static str, String); 4] {
    let valid = "ab".repeat(32);
    [
        ("odd length (63)", valid[..63].to_string()),
        ("non-hex digit", format!("zz{}", &valid[2..])),
        ("64 characters with a `+`", format!("+a{}", &valid[2..])),
        // 21 three-byte code points plus one ASCII digit: 64 bytes.
        (
            "64 bytes of non-ASCII",
            format!("{}a", "\u{20AC}".repeat(21)),
        ),
    ]
}

/// The two full-state readers, both on the atomic route. (The structural
/// `UpdateLayer` consumer's unfenced read is slim:
/// `SlimSnapshotReadConsumer::StructuralUpdate`. External-value clean-up's
/// reference walk is no snapshot read; it is [`walk_references`].)
const READ_CONSUMERS: [(SnapshotReadConsumer, SnapshotReadPath); 2] = [
    (SnapshotReadConsumer::Catchup, SnapshotReadPath::Atomic),
    (SnapshotReadConsumer::Checkpoint, SnapshotReadPath::Atomic),
];

/// External-value clean-up's reference walk of `oid`
/// (`storage::retirement::walk_object_references`, b.fsz MR-D4), which
/// decodes each value row's `hash` column by the readers' strict rule:
/// the hashes it found, or its failure.
async fn walk_references(
    tv: &TestValkey,
    oid: u64,
) -> Result<std::collections::HashSet<[u8; 32]>, ServiceError> {
    walk_object_references(
        tv.client(),
        tv.key_schema(),
        &oid.to_string(),
        &RetirementWalk::unreserved(),
    )
    .await
    .map(|walked| walked.hashes)
}

/// The structural write-planning consumers, whose reads are slim (they
/// decode into the compact form, keeping each row's validated hash): the
/// WATCH path's unfenced fan-out, and the cache's cold start and audit on the
/// atomic route, the script's `slim` flag set. Each validates a row's `hash`
/// column by the readers' rule, with the same error.
const SLIM_READ_CONSUMERS: [(SlimSnapshotReadConsumer, SnapshotReadPath); 3] = [
    (
        SlimSnapshotReadConsumer::StructuralUpdate,
        SnapshotReadPath::Unfenced,
    ),
    (
        SlimSnapshotReadConsumer::StructuralCache,
        SnapshotReadPath::Atomic,
    ),
    (
        SlimSnapshotReadConsumer::StructuralCacheAudit,
        SnapshotReadPath::Atomic,
    ),
];

/// A slim read of `oid` for `consumer`: its path, or the read's failure.
async fn read_slim(
    tv: &TestValkey,
    oid: u64,
    consumer: SlimSnapshotReadConsumer,
) -> Result<SnapshotReadPath, ServiceError> {
    match read_object_snapshot_slim(tv.client(), tv.key_schema(), &oid.to_string(), consumer)
        .await?
    {
        SnapshotReadOf::Snapshot(s) => Ok(s.path),
        SnapshotReadOf::TooLarge(t) => {
            panic!("{consumer:?} must read the object; got TOO_LARGE {t:?}")
        }
    }
}

/// The external rows [`every_shape_state`] carries, by kind.
#[derive(Clone, Copy, Debug)]
enum StoredRow {
    /// `(CUBE, "mesh")`.
    Field,
    /// `(CUBE, 1.5)`.
    TimeSample,
}

impl StoredRow {
    fn key(self, keys: &KeySchema, oid: u64) -> String {
        let (oid, sid) = (oid.to_string(), CUBE.to_string());
        match self {
            StoredRow::Field => keys.obj_field(&oid, &sid, "mesh"),
            StoredRow::TimeSample => keys.obj_time_sample(&oid, &sid, 1.5),
        }
    }
}

async fn set_hash_column(tv: &TestValkey, key: &str, value: &str) {
    let _: () = tv
        .client()
        .pool
        .hset(key, [("hash", value)])
        .await
        .expect("HSET hash must succeed");
}

/// Every route reads `oid` successfully, each consumer on the route it is
/// documented to take — the precondition that makes the failures below
/// attributable to the corrupted column and to the route named.
async fn assert_every_route_reads(tv: &TestValkey, oid: u64, what: &str) {
    read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .unwrap_or_else(|e| panic!("{what}: read_object_state_for_test must succeed; got {e:?}"));
    for (consumer, path) in READ_CONSUMERS {
        match read_object_snapshot(tv.client(), tv.key_schema(), &oid.to_string(), consumer).await {
            Ok(SnapshotRead::Snapshot(s)) => assert_eq!(
                s.path, path,
                "{what}: {consumer:?} must read through the {path:?} route"
            ),
            Ok(SnapshotRead::TooLarge(t)) => {
                panic!("{what}: {consumer:?} must read the object; got TOO_LARGE {t:?}")
            }
            Err(e) => panic!("{what}: {consumer:?} must read the object; got {e:?}"),
        }
    }
    for (consumer, path) in SLIM_READ_CONSUMERS {
        match read_slim(tv, oid, consumer).await {
            Ok(read) => assert_eq!(
                read, path,
                "{what}: {consumer:?} must read through the {path:?} route"
            ),
            Err(e) => panic!("{what}: {consumer:?} must read the object; got {e:?}"),
        }
    }
    walk_references(tv, oid)
        .await
        .unwrap_or_else(|e| panic!("{what}: the reference walk must read the object; got {e:?}"));
}

fn assert_malformed_hash_internal(result: Result<(), ServiceError>, what: &str) {
    match result {
        Err(ServiceError::Internal(msg)) => assert!(
            msg.contains(MALFORMED_HASH_MSG),
            "{what}: the Internal must name the malformed `hash` column; got {msg:?}"
        ),
        Ok(()) => panic!(
            "{what}: a malformed stored `hash` column must fail the read, never decode to a \
             padded or aliased hash"
        ),
        Err(other) => panic!("{what}: expected ServiceError::Internal, got {other:?}"),
    }
}

async fn malformed_hash_column_fails_every_read_internal(row: StoredRow) {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "g3-read";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    let row_key = row.key(tv.key_schema(), oid);
    let original: String = tv
        .client()
        .pool
        .hget(&row_key, "hash")
        .await
        .expect("fixture precondition: the external row must carry a `hash` column");
    assert_every_route_reads(&tv, oid, &format!("{row:?}, as created")).await;

    for (shape, column) in malformed_hash_columns() {
        set_hash_column(&tv, &row_key, &column).await;
        let what = format!("{row:?}, {shape} {column:?}");

        assert_malformed_hash_internal(
            read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
                .await
                .map(drop),
            &format!("{what}, read_object_state_for_test"),
        );
        for (consumer, path) in READ_CONSUMERS {
            assert_malformed_hash_internal(
                read_object_snapshot(tv.client(), tv.key_schema(), &oid.to_string(), consumer)
                    .await
                    .map(drop),
                &format!("{what}, {consumer:?} ({path:?} route)"),
            );
        }
        for (consumer, path) in SLIM_READ_CONSUMERS {
            assert_malformed_hash_internal(
                read_slim(&tv, oid, consumer).await.map(drop),
                &format!("{what}, {consumer:?} (slim, {path:?} route)"),
            );
        }
        // The clean-up's reference walk releases what the row names, so a
        // hash it cannot decode must stop the release, never release a
        // padded or aliased hash.
        assert_malformed_hash_internal(
            walk_references(&tv, oid).await.map(drop),
            &format!("{what}, the reference walk"),
        );
    }

    // Restoring the column restores every read: the failures above were the
    // column's, not a side effect of the corruption on anything else.
    set_hash_column(&tv, &row_key, &original).await;
    assert_every_route_reads(&tv, oid, &format!("{row:?}, restored")).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_reads_of_a_malformed_field_hash_column_return_internal() {
    malformed_hash_column_fails_every_read_internal(StoredRow::Field).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_reads_of_a_malformed_time_sample_hash_column_return_internal() {
    malformed_hash_column_fails_every_read_internal(StoredRow::TimeSample).await;
}

/// A row carrying its inline `value` column decodes as that value whatever
/// its `hash` column holds: `value` is checked first and the `hash` is never
/// decoded, so a garbage `hash` beside it cannot fail the read.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn snapshot_reads_of_an_inline_row_ignore_a_malformed_hash_column() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let layer = "g3-read-inline";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&ls, &session_id, layer, &every_shape_state()).await;
    // `every_shape_state`'s inline field: `(WORLD, "color")`.
    let row_key = tv
        .key_schema()
        .obj_field(&oid.to_string(), &WORLD.to_string(), "color");
    let row: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&row_key)
        .await
        .expect("HGETALL inline field row must succeed");
    assert!(
        row.contains_key("value") && !row.contains_key("hash"),
        "fixture precondition: the row must be inline, got {row:?}"
    );
    set_hash_column(&tv, &row_key, "zz").await;

    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .unwrap_or_else(|e| panic!("read_object_state_for_test must succeed; got {e:?}"));
    let color = snapshot
        .fields
        .iter()
        .find(|f| f.section_id == WORLD && f.key_name == "color")
        .expect("the inline field must be read");
    assert_eq!(
        color.value,
        FieldValue::Inline(vec![0xFF, 0x00, 0x80]),
        "the inline `value` column must win over the `hash` column"
    );
    assert_every_route_reads(&tv, oid, "inline row with a malformed `hash` beside it").await;
}
