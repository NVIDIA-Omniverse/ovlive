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

//! b.f4t sub-task 1 — **checkpoint restore is an authoritative REPLACE**, not
//! a merge into whatever survived.
//!
//! # The defect this file fences
//!
//! `apply_restore` materializes through `LayerStore::write_object`, the same
//! function `CreateLayer` uses. That materializer assumes a freshly-minted
//! `object_id` that owns nothing: it `SADD`s / `ZADD`s the restored members
//! into `obj:secs` / `obj:flds` / `obj:tss` rather than replacing those
//! indexes, it writes no tombstones at all, and it replaces a children list
//! only under a parent the restored state actually carries.
//!
//! The DR entry point violates that assumption by construction. It runs at a
//! KNOWN `object_id`, deliberately over a keyspace that may have partially
//! survived — the b.kf1 orphan class is exactly "header gone, everything else
//! alive". Written into such a keyspace, `write_object` MERGED: stale index
//! members unioned with the restored set and their per-record hashes read back
//! as live content the checkpoint never contained.
//!
//! Every test here therefore asserts the same shape from a different key
//! family: **after the restore, the object's index-reachable `obj:*` keyspace
//! equals exactly the restored state.** The pre-fix behaviour is what each
//! `must NOT survive` assertion catches.
//!
//! # Why these drive `apply_checkpoint` directly
//!
//! Two reasons, both load-bearing:
//!
//! 1. **The b.4et stream adjudication must not run.** It lives in
//!    `RecoveryHook`, upstream of the apply, and it legitimately `UNLINK`s a
//!    stale per-object stream. Driving through the hook would make
//!    [`restore_leaves_the_per_object_stream_untouched`] assert the
//!    adjudication's verdict rather than the wipe's leaving the stream alone
//!    (the shared key inventory never carries it), which is the thing under
//!    test.
//! 2. **No checkpoint store is needed.** The residue semantics are a property
//!    of the apply, not of restore-source selection, so a real
//!    `CheckpointStore` + leader-elected loop would be several seconds of
//!    fixture buying nothing. `e2e_stat_layer_recovery_test.rs` owns the
//!    handler-level path.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; see CLAUDE.md.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use fred::interfaces::{
    HashesInterface, KeysInterface, ListInterface, SetsInterface, SortedSetsInterface,
    StreamsInterface,
};

use live_edit::checkpoint::{apply_checkpoint, CheckpointError, CheckpointMeta};
use live_edit::domain::ompf::OmpfContainer;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::external_values::{
    process_pending_gc, ExtRelease, ExternalValueStore, ValkeyExternalValueStore,
};
use live_edit::storage::keys::{sha256_to_hex, KeySchema};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand, TimeSample};

use super::helpers::residue_wiped_cell;

/// A **re-export**, not a second `"test"` literal — the same aliasing
/// `helpers::SERVICE_LABEL` uses, and for the same reason. This file stamps the
/// label onto the registry it builds, and [`residue_wiped_cell`] reads the
/// `{service=...}` cell back out of that registry; re-typing either copy would
/// make every read here return `0` forever and every assertion built on it
/// vacuous. One address makes that drift unrepresentable rather than merely
/// unlikely.
const SERVICE_LABEL: &str = crate::common::RECOVERY_SERVICE_LABEL;
const SESSION_ID: &str = "residue-session";
const LAYER_NAME: &str = "residue-layer";

/// The one section the restored state carries. Every "stale" id seeded by a
/// test is deliberately different from this, so a surviving member is
/// distinguishable from a restored one by id alone.
const RESTORED_SECTION_ID: u64 = 42;

/// The one field the restored state carries, on [`RESTORED_SECTION_ID`]. Named
/// because both the fixture and the fields-family assertion have to spell the
/// same key name — the `obj:flds` member is `{section_id}:{hex(key_name)}`, so a
/// divergence would silently compare against a member no writer ever wrote.
const RESTORED_FIELD_KEY: &str = "restored_key";

/// The one time sample the restored state carries, on [`RESTORED_SECTION_ID`].
/// Same reasoning as [`RESTORED_FIELD_KEY`]: the `obj:tss` member is
/// `{section_id}:{hex_f64(time)}`, an IEEE-754 bit pattern, so the fixture and
/// the assertion must be reading one constant.
const RESTORED_SAMPLE_TIME: f64 = 1.0;

/// `snap_seq` fed to every restore here. Read by
/// [`restored_sequence_num`], which the ext-release assertion needs because a
/// `rewrite` release's pending-set member carries the committing sequence.
const SNAP_SEQ: u64 = 30;

/// The sequence the restore lands on, and therefore the `event` token of any
/// release `write_object` enqueues: `plan_ext_index` is handed
/// `state.sequence_num`, which is the same value `write_object` stamps into
/// `obj:hdr`'s `seq` (SR-2.5's `snap_seq + safety_margin` ratchet).
fn restored_sequence_num() -> u64 {
    SNAP_SEQ + crate::common::test_config().checkpoint_seq_safety_margin
}

/// A distinct 32-byte content hash. Two hashes tellable apart is what turns
/// "the restore released exactly the reference it gave up" into an assertable
/// claim rather than a count.
fn ext_hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

/// The hash the residue's value rows hold and the restored state does NOT —
/// the one reference the restore gives up, and therefore the one release it
/// owes.
const DROPPED_HASH_BYTE: u8 = 0xd0;

/// The hash held by BOTH the residue and the restored state. It must NOT be
/// released: the restored object still references it, so a decrement here is
/// over-release, and with GC behind it, meta+data deleted under a live
/// reference.
const KEPT_HASH_BYTE: u8 = 0x4e;

// =============================================================================
// Fixtures
// =============================================================================

/// A reset-delta OMPF carrying exactly one root-level section, one field on it,
/// and one time sample on it.
///
/// Non-empty on purpose, in all three families: an EMPTY restored state would
/// write no children list, no field and no time sample, so `write_object` would
/// touch none of `obj:ch:0` / `obj:flds` / `obj:tss` and the "restored set
/// replaced the surviving set" assertions would have no restored set to compare
/// against. The field and the time sample specifically make the two `ZADD`-union
/// families assertable in the same index-equals-restored-set shape the section
/// test uses, rather than the weaker "the index is simply absent" shape an empty
/// restored set forces.
fn restored_reset_ompf() -> Bytes {
    let delta = OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: RESTORED_SECTION_ID,
            section_name: "restored".to_string(),
            section_type: 1,
        })],
        set_fields: vec![SetField {
            section_id: RESTORED_SECTION_ID,
            key_name: RESTORED_FIELD_KEY.to_string(),
            value_or_ext_hash: b"restored-value".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        time_samples: vec![TimeSample {
            section_id: RESTORED_SECTION_ID,
            time: RESTORED_SAMPLE_TIME,
            value_or_ext_hash: b"restored-sample".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        source_format: "usda".to_string(),
    };
    let delta_bytes = oodf_ffi::build_delta(&delta).expect("build_delta must succeed");
    Bytes::from(OmpfContainer::serialize(&delta_bytes, &[]).expect("OMPF serialize must succeed"))
}

/// A reset-delta OMPF carrying one root-level section and one **external**
/// field on it, referencing `hash`.
///
/// A base checkpoint blob ships hash references with no Parts 1..N
/// (`assembler.rs::build_reset_ompf_from_state` keeps the hash and the
/// `ext_value_index`; the bytes do not travel), so `ext_value_index = 1`
/// against a zero-part container is the real shape this decoder sees — not a
/// malformed fixture. What matters downstream is only that
/// `field_value_from_oodf` reads `ext_value_size > 0` and mints a
/// `FieldValue::External`, because that is what `ext_row_counts` counts.
fn restored_reset_ompf_referencing(hash: &[u8; 32]) -> Bytes {
    let delta = OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: RESTORED_SECTION_ID,
            section_name: "restored".to_string(),
            section_type: 1,
        })],
        set_fields: vec![SetField {
            section_id: RESTORED_SECTION_ID,
            key_name: RESTORED_FIELD_KEY.to_string(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: 4_096,
            ext_value_index: 1,
            set_order: 0,
        }],
        time_samples: Vec::new(),
        source_format: "usda".to_string(),
    };
    let delta_bytes = oodf_ffi::build_delta(&delta).expect("build_delta must succeed");
    Bytes::from(OmpfContainer::serialize(&delta_bytes, &[]).expect("OMPF serialize must succeed"))
}

/// The `CheckpointMeta` whose `(session_id, layer_name)` identity the SR-2.4
/// guard resolves. Must match [`install_mapping`] or the guard refuses and no
/// test below reaches the wipe at all.
fn residue_meta() -> CheckpointMeta {
    CheckpointMeta {
        creator_replica_id: SERVICE_LABEL.to_string(),
        written_at_unix_ms: 1_700_000_000_000,
        snap_seq: Some(SNAP_SEQ),
        session_id: SESSION_ID.to_string(),
        layer_name: LAYER_NAME.to_string(),
        created_by: "residue-user".to_string(),
        server_ts: 1_700_000_000_000,
        ..CheckpointMeta::default()
    }
}

/// Point the live layer mapping at `object_id` so the SR-2.4 identity guard
/// passes. Without it every restore here would refuse `NotFound` before
/// reaching step 3b, and every assertion would be vacuous.
async fn install_mapping(tv: &TestValkey, object_id: u64) {
    let mapping_key = tv.key_schema().layer_mapping(SESSION_ID, LAYER_NAME);
    let _: () = tv
        .client()
        .pool
        .set(&mapping_key, object_id.to_string(), None, None, false)
        .await
        .expect("SET layer mapping so the SR-2.4 identity guard passes");
}

/// Stand up a `LayerStore` over `tv` with the process metrics handle wired on
/// BOTH sides — the client (so the residue wipe's `metrics_snapshot()` finds
/// it) and the store.
fn wired_layer_store(tv: &TestValkey, metrics: &SharedMetrics) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_metrics(metrics.clone())
}

/// Run the restore under test. Returns the `apply_checkpoint` result verbatim
/// so refusal tests can inspect it; the success tests `expect` it.
async fn run_restore(layer_store: &LayerStore, object_id: u64) -> Result<(), CheckpointError> {
    run_restore_of(layer_store, object_id, restored_reset_ompf()).await
}

/// [`run_restore`] with the restored payload chosen by the caller — the
/// external-value family needs a base blob that references a hash.
async fn run_restore_of(
    layer_store: &LayerStore,
    object_id: u64,
    ompf: Bytes,
) -> Result<(), CheckpointError> {
    apply_checkpoint(
        layer_store,
        object_id,
        ompf,
        SNAP_SEQ,
        &residue_meta(),
        None,
        &crate::common::test_config(),
    )
    .await
}

async fn key_exists(tv: &TestValkey, key: &str) -> bool {
    let n: u64 = tv
        .client()
        .pool
        .exists(key)
        .await
        .unwrap_or_else(|e| panic!("EXISTS {key} must succeed: {e}"));
    n > 0
}

async fn set_members(tv: &TestValkey, key: &str) -> Vec<String> {
    let mut members: Vec<String> = tv
        .client()
        .pool
        .smembers(key)
        .await
        .unwrap_or_else(|e| panic!("SMEMBERS {key} must succeed: {e}"));
    members.sort();
    members
}

/// Members of a sorted set, sorted. The score is deliberately NOT read: the
/// residue wipe consumes `ZRANGE` members only, so the member string is the
/// whole contract these tests pin.
async fn zset_members(tv: &TestValkey, key: &str) -> Vec<String> {
    let mut members: Vec<String> = tv
        .client()
        .pool
        .zrange(key, 0, -1, None, false, None, false)
        .await
        .unwrap_or_else(|e| panic!("ZRANGE {key} must succeed: {e}"));
    members.sort();
    members
}

/// Contents of a list key, in order. The children family is ORDER-bearing —
/// `obj:ch:{parent}` is the authoring order of that parent's children — so the
/// assertion compares the whole Vec rather than a set.
async fn list_items(tv: &TestValkey, key: &str) -> Vec<String> {
    tv.client()
        .pool
        .lrange(key, 0, -1)
        .await
        .unwrap_or_else(|e| panic!("LRANGE {key} must succeed: {e}"))
}

/// `HGETALL le:{oid}:obj:ext` as sorted `(hash_hex, count)` pairs — the
/// per-object external-value reference index, compared whole so hash iteration
/// order cannot matter.
async fn obj_ext_sorted(tv: &TestValkey, object_id: u64) -> Vec<(String, String)> {
    let raw: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().obj_ext_index(&object_id.to_string()))
        .await
        .expect("HGETALL obj:ext must succeed");
    let mut rows: Vec<(String, String)> = raw.into_iter().collect();
    rows.sort();
    rows
}

/// Every member of `le:ext:{gc}:pending`, sorted. Members are
/// `{hash_hex}:{object_id}:{event}`, so this is the whole owed-release ledger.
async fn pending_members(tv: &TestValkey) -> Vec<String> {
    let mut members: Vec<String> = tv
        .client()
        .pool
        .smembers(tv.key_schema().ext_pending())
        .await
        .expect("SMEMBERS ext_pending must succeed");
    members.sort();
    members
}

/// `HGET le:ext:meta:{hash_hex} ref_count`; `None` once GC has reclaimed the
/// blob.
async fn ref_count(tv: &TestValkey, hash: &[u8; 32]) -> Option<i64> {
    let raw: Option<String> = tv
        .client()
        .pool
        .hget(tv.key_schema().ext_meta(&sha256_to_hex(hash)), "ref_count")
        .await
        .expect("HGET ref_count must succeed");
    raw.map(|v| {
        v.parse()
            .unwrap_or_else(|e| panic!("ref_count must be an integer, got {v:?}: {e}"))
    })
}

/// `(meta_present, data_present)` for one blob — read as a pair because the two
/// are deleted atomically and half a blob surviving is a different failure from
/// the whole blob surviving.
async fn blob_present(tv: &TestValkey, hash: &[u8; 32]) -> (bool, bool) {
    let hex = sha256_to_hex(hash);
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

// The `live_edit_checkpoint_restore_residue_wiped_total` reader this file's arms
// use is `helpers::residue_wiped_cell`, the registry-typed core shared with the
// harness-typed `helpers::residue_wiped_metric`. This file drives `write_object`
// against a bare `TestValkey` + registry with no `CheckpointHarness` anywhere, so
// the registry-typed form is the one it can reach; that split is why the fourth
// copy this file used to carry is gone.

// =============================================================================
// 1. Index families: the restored set REPLACES the surviving one.
// =============================================================================

/// A section that survived in `obj:secs` — with its `obj:sec:{id}` hash alive
/// — must be gone after the restore, index member and record both.
///
/// Pre-fix this was the sharpest form of the defect: `write_object`'s `SADD`
/// unioned `{42}` into `{7}`, leaving `obj:secs = {7, 42}` with `obj:sec:7`
/// readable, so every subsequent snapshot read served a section the checkpoint
/// never contained as live content of the restored layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_drops_a_surviving_section_and_its_record() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let layer_store = wired_layer_store(&tv, &metrics);

    let oid: u64 = 9_401;
    let oid_str = oid.to_string();
    install_mapping(&tv, oid).await;

    // Residue: one stale section, index member + record hash, NO header.
    let stale_record = tv.key_schema().obj_section(&oid_str, "7");
    let _: () = tv
        .client()
        .pool
        .sadd(tv.key_schema().obj_sections_index(&oid_str), "7")
        .await
        .expect("SADD stale obj:secs member");
    let _: () = tv
        .client()
        .pool
        .hset(&stale_record, [("name", "stale"), ("parent_id", "0")])
        .await
        .expect("HSET stale obj:sec record");

    run_restore(&layer_store, oid)
        .await
        .expect("a restore over surviving residue must succeed");

    assert_eq!(
        set_members(&tv, &tv.key_schema().obj_sections_index(&oid_str)).await,
        vec![RESTORED_SECTION_ID.to_string()],
        "obj:secs must equal EXACTLY the restored set. A union ({{7, 42}}) is the pre-fix \
         merge: the surviving member is index-reachable, so every later snapshot read serves \
         section 7 as live content of a layer the checkpoint says has only section 42"
    );
    assert!(
        !key_exists(&tv, &stale_record).await,
        "the surviving section's record hash `{stale_record}` must be gone too. Dropping the \
         index member alone would leave an unreachable record that no retirement path can \
         enumerate — the b.9fn orphan shape"
    );
    assert_eq!(
        residue_wiped_cell(&metrics),
        1,
        "a restore that had to replace residue MUST tick \
         live_edit_checkpoint_restore_residue_wiped_total exactly once — it is the only signal \
         an operator has that state loss was PARTIAL rather than total"
    );
}

/// A field that survived in `obj:flds` — with its `obj:fld:{sid}:{hex(key)}`
/// hash alive — must be gone after the restore, index member and record both.
///
/// The sections family above is a `SADD` union; this one is a `ZADD` union, and
/// it reaches its record hash by a route the section family does not have. The
/// wipe cannot name `obj:fld:*` keys from the index key alone: it `ZRANGE`s
/// `obj:flds` for members and feeds each one back through
/// `KeySchema::obj_field_from_member`, which re-attaches the member VERBATIM
/// (`{prefix}{{oid}}:obj:fld:{member}`) precisely because the member already
/// carries `{section_id}:{hex(key_name)}` in the right order.
///
/// That makes the seeded member encoding load-bearing rather than cosmetic, and
/// it is why this test seeds via `KeySchema::field_member` / `KeySchema::obj_field`
/// rather than hand-spelling either string: a decoder that re-hex-encoded the
/// member, or an enumeration that read the ZSet `WITHSCORES` and mistook a score
/// for a member, would derive keys that do not exist, `UNLINK` nothing, and
/// strand the real record hash — while the object-level `obj:flds` key itself,
/// which is one of the twelve unconditional entries in the inventory, still
/// disappears and makes every index-side assertion pass.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_drops_a_surviving_field_and_its_record() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let layer_store = wired_layer_store(&tv, &metrics);

    let oid: u64 = 9_407;
    let oid_str = oid.to_string();
    install_mapping(&tv, oid).await;

    // Residue: one stale field on section 7, which the restored state does not
    // carry. Member + record hash, NO header.
    let stale_member = KeySchema::field_member("7", "stale_key");
    let stale_record = tv.key_schema().obj_field(&oid_str, "7", "stale_key");
    let _: () = tv
        .client()
        .pool
        .zadd(
            tv.key_schema().obj_fields_index(&oid_str),
            None,
            None,
            false,
            false,
            (11.0, stale_member.as_str()),
        )
        .await
        .expect("ZADD stale obj:flds member");
    let _: () = tv
        .client()
        .pool
        // The column shape a real inline-value writer lands: base64 `value`
        // plus the two orders and the version. `c3RhbGU=` is base64("stale").
        .hset(
            &stale_record,
            [
                ("value", "c3RhbGU="),
                ("set_order", "3"),
                ("authoring_set_order", "3"),
                ("version", "11"),
            ],
        )
        .await
        .expect("HSET stale obj:fld record");

    run_restore(&layer_store, oid)
        .await
        .expect("a restore over a surviving field must succeed");

    assert_eq!(
        zset_members(&tv, &tv.key_schema().obj_fields_index(&oid_str)).await,
        vec![KeySchema::field_member(
            &RESTORED_SECTION_ID.to_string(),
            RESTORED_FIELD_KEY
        )],
        "obj:flds must equal EXACTLY the restored set. `write_object` ZADDs rather than \
         replacing, so pre-fix the surviving `{stale_member}` was unioned with the restored \
         member and read back as live content of the restored layer"
    );
    assert!(
        !key_exists(&tv, &stale_record).await,
        "the surviving field's record hash `{stale_record}` must be gone too. This is the \
         assertion with teeth on the member-decoder path: the wipe reaches this key ONLY by \
         feeding the ZRANGE'd member back through `obj_field_from_member`, so a member-encoding \
         slip strands the record while the index-side assertion above still passes"
    );
}

/// A time sample that survived in `obj:tss` — with its
/// `obj:ts:{sid}:{hex_f64(time)}` hash alive — must be gone after the restore,
/// index member and record both.
///
/// The second `ZADD`-union family, and the second member decoder
/// (`KeySchema::obj_ts_from_member`). Its member encodes the sample time as a
/// raw IEEE-754 bit pattern, so it is the sharper of the two on encoding drift:
/// `hex_f64(12.5)` shares no substring with the decimal spelling, and a decoder
/// that re-encoded it would produce a key nothing has ever written.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_drops_a_surviving_time_sample_and_its_record() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let layer_store = wired_layer_store(&tv, &metrics);

    let oid: u64 = 9_408;
    let oid_str = oid.to_string();
    install_mapping(&tv, oid).await;

    // Residue: one stale sample on section 7 at a time the restored state does
    // not carry. Both coordinates differ from the restored sample, so a
    // survivor is distinguishable from a restored member by the member alone.
    const STALE_SAMPLE_TIME: f64 = 12.5;
    let stale_member = KeySchema::time_sample_member("7", STALE_SAMPLE_TIME);
    let stale_record = tv
        .key_schema()
        .obj_time_sample(&oid_str, "7", STALE_SAMPLE_TIME);
    let _: () = tv
        .client()
        .pool
        .zadd(
            tv.key_schema().obj_time_samples_index(&oid_str),
            None,
            None,
            false,
            false,
            (11.0, stale_member.as_str()),
        )
        .await
        .expect("ZADD stale obj:tss member");
    let _: () = tv
        .client()
        .pool
        // A time-sample record carries the value columns plus `version` — no
        // set orders. `c3RhbGU=` is base64("stale").
        .hset(&stale_record, [("value", "c3RhbGU="), ("version", "11")])
        .await
        .expect("HSET stale obj:ts record");

    run_restore(&layer_store, oid)
        .await
        .expect("a restore over a surviving time sample must succeed");

    assert_eq!(
        zset_members(&tv, &tv.key_schema().obj_time_samples_index(&oid_str)).await,
        vec![KeySchema::time_sample_member(
            &RESTORED_SECTION_ID.to_string(),
            RESTORED_SAMPLE_TIME
        )],
        "obj:tss must equal EXACTLY the restored set. Same ZADD-union defect as obj:flds: \
         pre-fix the surviving `{stale_member}` outlived the restore and every later snapshot \
         read served a sample at t={STALE_SAMPLE_TIME} the checkpoint never contained"
    );
    assert!(
        !key_exists(&tv, &stale_record).await,
        "the surviving sample's record hash `{stale_record}` must be gone too — reachable only \
         via `obj_ts_from_member`, whose hex-f64 member encoding is the thing this assertion \
         actually exercises"
    );
}

/// A surviving tombstone (`obj:dels` member + its `obj:del:{id}` hash) must be
/// dropped.
///
/// This family is worse than the index ones pre-fix, because `write_object`
/// writes **no tombstones at all**: there was no restored set to union with,
/// so a surviving deletion simply outlived the restore untouched and kept
/// suppressing a section the checkpoint carries as live.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_drops_a_surviving_tombstone_and_its_record() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let layer_store = wired_layer_store(&tv, &metrics);

    let oid: u64 = 9_402;
    let oid_str = oid.to_string();
    install_mapping(&tv, oid).await;

    // The tombstone deliberately names the section the restore DOES carry:
    // that is the case with teeth, because a surviving `obj:del:42` suppresses
    // the very section the checkpoint restored.
    let restored_sid = RESTORED_SECTION_ID.to_string();
    let tombstone_record = tv.key_schema().obj_deleted_section(&oid_str, &restored_sid);
    let _: () = tv
        .client()
        .pool
        .sadd(
            tv.key_schema().obj_deleted_sections_index(&oid_str),
            restored_sid.as_str(),
        )
        .await
        .expect("SADD stale obj:dels member");
    let _: () = tv
        .client()
        .pool
        .hset(&tombstone_record, [("version", "11")])
        .await
        .expect("HSET stale obj:del record");

    run_restore(&layer_store, oid)
        .await
        .expect("a restore over a surviving tombstone must succeed");

    assert!(
        !key_exists(&tv, &tv.key_schema().obj_deleted_sections_index(&oid_str)).await,
        "obj:dels must not survive: the restored state carries no tombstones, so an \
         authoritative replace leaves the index absent. Pre-fix `write_object` wrote no \
         tombstones at all — there was nothing to overwrite the survivor with, so a deletion \
         of section {RESTORED_SECTION_ID} outlived the restore that brought it back"
    );
    assert!(
        !key_exists(&tv, &tombstone_record).await,
        "the tombstone record `{tombstone_record}` must be gone too"
    );
}

/// A children list under a parent the restored state does NOT carry must be
/// dropped, along with its `obj:chs` index member.
///
/// b.zvd made children lists a replace, but only for parents the restored
/// state carries (`DEL` before `RPUSH`, inside the per-parent loop). A list
/// under a dangling parent had no loop iteration to clear it, so it survived
/// intact — and its `obj:chs` member with it, which is precisely what makes it
/// enumerable and therefore readable.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_drops_a_children_list_under_a_parent_it_does_not_carry() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let layer_store = wired_layer_store(&tv, &metrics);

    let oid: u64 = 9_403;
    let oid_str = oid.to_string();
    install_mapping(&tv, oid).await;

    // Parent 55 is dangling: the restored state's only section is 42, hanging
    // off the root, so `write_object`'s per-parent loop never visits 55.
    let dangling_list = tv.key_schema().obj_children(&oid_str, "55");
    let _: () = tv
        .client()
        .pool
        .rpush(&dangling_list, vec!["901", "902"])
        .await
        .expect("RPUSH stale children list");
    let _: () = tv
        .client()
        .pool
        .sadd(tv.key_schema().obj_children_index(&oid_str), "55")
        .await
        .expect("SADD stale obj:chs member");

    run_restore(&layer_store, oid)
        .await
        .expect("a restore over a dangling children list must succeed");

    assert!(
        !key_exists(&tv, &dangling_list).await,
        "`{dangling_list}` must be gone. b.zvd's DEL-before-RPUSH only covers parents the \
         restored state carries, and 55 is not one — so this list is exactly the case that \
         needed the pre-write wipe"
    );
    assert_eq!(
        set_members(&tv, &tv.key_schema().obj_children_index(&oid_str)).await,
        vec!["0".to_string()],
        "obj:chs must name exactly the restored state's parents — the root, whose list holds \
         section {RESTORED_SECTION_ID}. A surviving `55` member is what makes the dangling \
         list enumerable and therefore readable at all"
    );
}

/// A stale ROOT children list (`obj:ch:0`) must be replaced by the restored
/// root list, not appended to.
///
/// The sibling above covers the dangling-parent list, which the restore's own
/// per-parent loop never visits. This is the other half, and it is the half
/// with a second defence: parent `0` IS carried by the restored state, so
/// b.zvd's `DEL`-before-`RPUSH` clears the list inside `write_object` even
/// without the wipe. Asserting the list is *absent* would therefore be false
/// (the restore re-creates it) and asserting only `obj:chs == ["0"]` — which is
/// all the sibling does — cannot see a merge at all, because the index member
/// is `"0"` under both the stale and the restored spelling.
///
/// The list's CONTENTS are the only observation that separates them: a merge
/// reads back `["901", "902", "42"]`, an authoritative replace reads back
/// `["42"]`. The root list is where that distinction matters most — it is the
/// entry point of every snapshot reconstruction's tree walk, so a stale head
/// here surfaces two sections the checkpoint never contained as top-level
/// children of the restored layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_replaces_a_stale_root_children_list_rather_than_merging_into_it() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let layer_store = wired_layer_store(&tv, &metrics);

    let oid: u64 = 9_410;
    let oid_str = oid.to_string();
    install_mapping(&tv, oid).await;

    // The residue a real writer leaves: a root list plus the `obj:chs` member
    // that names it. Both ids differ from the restored section, so a survivor
    // is distinguishable from a restored child by the member alone.
    let root_list = tv.key_schema().obj_children(&oid_str, "0");
    let _: () = tv
        .client()
        .pool
        .rpush(&root_list, vec!["901", "902"])
        .await
        .expect("RPUSH stale root children list");
    let _: () = tv
        .client()
        .pool
        .sadd(tv.key_schema().obj_children_index(&oid_str), "0")
        .await
        .expect("SADD stale obj:chs root member");

    run_restore(&layer_store, oid)
        .await
        .expect("a restore over a stale root children list must succeed");

    assert_eq!(
        list_items(&tv, &root_list).await,
        vec![RESTORED_SECTION_ID.to_string()],
        "`{root_list}` must hold EXACTLY the restored root's children. A `[\"901\", \"902\", \
         \"42\"]` here is the merge: the root list is the tree walk's entry point, so two \
         children the checkpoint never contained read back as top-level content of the restored \
         layer"
    );
    assert_eq!(
        set_members(&tv, &tv.key_schema().obj_children_index(&oid_str)).await,
        vec!["0".to_string()],
        "obj:chs must still name exactly the restored state's one parent — the root"
    );
}

// =============================================================================
// 2. The two things the wipe must NOT do.
// =============================================================================

/// A keyspace with nothing under this `object_id` costs ONE `EXISTS` and
/// nothing else: no `WATCH`, no `UNLINK`, no counter tick.
///
/// This is the overwhelmingly common DR shape (whole-keyspace loss), so the
/// wipe being free here is a cost property worth pinning. The assertion is
/// made through observability rather than by counting round trips: the residue
/// counter is what ticks on a wipe, and the whole
/// `storage_watch_voided_total{path="restore_residue_wipe"}` row is what ticks
/// if any of the three watched-transaction defence layers reported anything —
/// which they cannot do if no `WATCH` was ever armed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_over_a_clean_keyspace_wipes_nothing_and_arms_no_watch() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let layer_store = wired_layer_store(&tv, &metrics);

    let oid: u64 = 9_404;
    install_mapping(&tv, oid).await;

    // Deliberately NO seeding: header-less and index-less, the whole-loss case.

    run_restore(&layer_store, oid)
        .await
        .expect("a restore over a clean keyspace must succeed");

    assert!(
        key_exists(&tv, &tv.key_schema().obj_header(&oid.to_string())).await,
        "non-vacuity: the restore must actually have materialized the object, or 'nothing was \
         wiped' would be trivially true of a restore that did nothing"
    );
    assert_eq!(
        residue_wiped_cell(&metrics),
        0,
        "the clean path must NOT tick the residue counter. A tick here makes the metric useless \
         as an alert — every ordinary DR restore would fire it — and means the common case is \
         paying a txn-pool checkout plus a WATCH it does not need"
    );

    let text = live_edit::observability::metrics::test_util::render(&metrics);
    for disposition in [
        "aborted_pre_exec",
        "committed_unwatched",
        "compensated",
        "witness_unavailable",
        "fingerprint_unavailable",
    ] {
        assert_eq!(
            live_edit::observability::metrics::test_util::sample_value(
                &text,
                "live_edit_storage_watch_voided_total",
                &[
                    ("path", "restore_residue_wipe"),
                    ("disposition", disposition)
                ],
            ),
            Some(0.0),
            "storage_watch_voided_total{{path=\"restore_residue_wipe\", \
             disposition=\"{disposition}\"}} must still read its zero seed — every one of \
             these is reported from INSIDE the watched section, so a non-zero value means a \
             WATCH was armed on a keyspace the probe should have short-circuited as clean"
        );
    }
}

/// The per-object stream survives the wipe.
///
/// The shared key inventory never carries the stream, and that is a
/// deliberate carve-out rather than an oversight: the stream belongs to the
/// b.4et adjudication (`probe_stream_high_water` / `unlink_stale_stream`),
/// whose high-water read IS the restore floor, and to the
/// `STREAM_DELETE_GRACE_MS` window that lets lagging subscribers still observe
/// a `deleted` sentinel. Wiping it here would destroy both.
///
/// Driven through `apply_checkpoint` precisely so the adjudication does not
/// run — see the module header. What is asserted is the apply's own
/// behaviour: it must leave the stream exactly as it found it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_leaves_the_per_object_stream_untouched() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let layer_store = wired_layer_store(&tv, &metrics);

    let oid: u64 = 9_405;
    let oid_str = oid.to_string();
    install_mapping(&tv, oid).await;

    // Residue alongside the stream, so the wipe genuinely runs — a `Clean`
    // short-circuit would leave the stream alone for the wrong reason. The
    // record hash is seeded too, because it is the non-vacuity witness at the
    // end: a member with no record would be removed from `obj:secs` by
    // `write_object`'s own SADD-then-read semantics and prove nothing.
    let stale_record = tv.key_schema().obj_section(&oid_str, "7");
    let _: () = tv
        .client()
        .pool
        .sadd(tv.key_schema().obj_sections_index(&oid_str), "7")
        .await
        .expect("SADD stale obj:secs member");
    let _: () = tv
        .client()
        .pool
        .hset(&stale_record, [("name", "stale"), ("parent_id", "0")])
        .await
        .expect("HSET stale obj:sec record");

    let stream_key = tv.key_schema().layer_stream(oid);
    let _: String = tv
        .client()
        .pool
        .xadd(&stream_key, false, None, "5-0", vec![("seq", "5")])
        .await
        .expect("XADD a surviving stream entry");

    run_restore(&layer_store, oid)
        .await
        .expect("a restore alongside a surviving stream must succeed");

    let len: u64 = tv
        .client()
        .pool
        .xlen(&stream_key)
        .await
        .expect("XLEN after the restore");
    assert_eq!(
        len, 1,
        "the per-object stream must survive the residue wipe untouched. It is owned by the \
         b.4et adjudication, whose high-water read is the restore floor, and by the \
         STREAM_DELETE_GRACE_MS window; unlinking it here destroys both"
    );
    assert!(
        !key_exists(&tv, &stale_record).await,
        "non-vacuity: the wipe must actually have run (the stale section record is gone), or \
         the stream surviving proves nothing about the wipe leaving it alone"
    );
}

// =============================================================================
// 3. The header as a probe key: a partial header with no other residue.
// =============================================================================

/// A header that is PRESENT but structurally incomplete, with nothing else
/// alive under the object, must be wiped and repaired.
///
/// `obj:hdr` is one of the eleven keys the residue probe `EXISTS`es, and this
/// is the case that makes it load-bearing rather than decorative: with the
/// header omitted from the probe, an object in this state short-circuits as
/// `Clean` and the partial header is left in place for `write_object` to HSET
/// over. That HSET stamps the 12 canonical fields but does NOT `DEL` the key
/// first, so any non-canonical field on the survivor outlives the restore
/// underneath the restored header — two of them with teeth, both asserted
/// below.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_wipes_and_repairs_a_lone_partial_header() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let layer_store = wired_layer_store(&tv, &metrics);

    let oid: u64 = 9_406;
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    install_mapping(&tv, oid).await;

    // `seq` present, `created_by` absent — what `stat_layer_for_object_id`
    // already answers NotFound on — plus the two stale flags with teeth.
    let _: () = tv
        .client()
        .pool
        .hset(
            &header_key,
            [
                ("seq", "100"),
                ("superseded", "1"),
                ("ext_gc_scheduled", "1"),
            ],
        )
        .await
        .expect("HSET the partial header");

    run_restore(&layer_store, oid)
        .await
        .expect("a structurally-partial header must drive a real restore, not a short-circuit");

    let header: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr after the restore");

    assert!(
        header.contains_key("created_by"),
        "the restore must repair the header — the missing `created_by` is what made \
         `stat_layer_for_object_id` answer NotFound and route the layer into recovery in the \
         first place. Got {header:?}"
    );
    assert_eq!(
        header.get("seq").map(String::as_str),
        Some((SNAP_SEQ + crate::common::test_config().checkpoint_seq_safety_margin).to_string())
            .as_deref(),
        "the repaired header must carry the seeded sequence, not the partial header's stale \
         100 — that is what distinguishes a restore that RAN from a short-circuit"
    );
    assert!(
        !header.contains_key("superseded"),
        "a surviving `superseded=1` makes the restored layer permanently UNWRITABLE: both \
         commit backends fail closed on it. `write_object`'s HSET cannot remove it — only the \
         step-3b wipe can. Got {header:?}"
    );
    assert!(
        !header.contains_key("ext_gc_scheduled"),
        "a surviving `ext_gc_scheduled=1` makes the restored object's later ext-GC schedule a \
         silent no-op, so its external-value refcounts are never released. Got {header:?}"
    );
}

// =============================================================================
// 4. The external-value reference index the wipe must LEAVE BEHIND.
// =============================================================================

/// A restore over residue releases exactly the external-value references the
/// residue held and the restored state does not — and nothing else.
///
/// # Why `obj:ext` is the one inventory member the wipe withholds
///
/// `obj:ext` is a member of the shared `KeySchema::object_cleanup_keys`
/// inventory, because the two RETIREMENT routes must remove it. This route is
/// not a retirement: the object is re-materialized at the same `object_id`
/// moments later, and `write_object`'s restore entry point reads this very key
/// (`HGETALL`, as `ext_before`) to compute the releases it owes —
/// `before − after` over the index, emitted as `rewrite` releases under
/// `SurplusPolicy::Suppress`.
///
/// So the ordering is load-bearing in a way no other family here is. The wipe
/// runs FIRST; if it had taken `obj:ext` with it, the `HGETALL` a few
/// statements later would answer empty, `before − after` would be empty too,
/// and **every** restore over residue would leak one `ref_count` per hash the
/// residue held and the restored state dropped — permanently, since nothing
/// re-derives that pre-image afterwards. That is the failure this test exists
/// for, and assertion (a) is the one that cannot pass without the carve-out.
///
/// # Why the fixture holds TWO hashes and one of them twice
///
/// Releasing the residue's whole hash set would be the other, worse bug:
/// `H_kept` is referenced by the restored state as well, so decrementing it is
/// an unmatched decrement against a live reference — and with GC behind it,
/// meta+data deleted under a layer that still reads the blob. Assertion (b) is
/// that canary, and it is why `H_kept` is here at all.
///
/// `H_kept` is seeded at a count of **2** (two residue rows held it) against
/// the restored state's **1**. That makes assertion (c) — the index equals the
/// restored state's ABSOLUTE row counts — a real observation rather than a
/// leftover that happened to match: a rebuild writes `1`, doing nothing leaves
/// `2`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restore_releases_only_the_external_references_the_residue_gave_up() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;
    let layer_store = wired_layer_store(&tv, &metrics);

    let oid: u64 = 9_409;
    let oid_str = oid.to_string();
    install_mapping(&tv, oid).await;

    let dropped = ext_hash(DROPPED_HASH_BYTE);
    let kept = ext_hash(KEPT_HASH_BYTE);

    // The real Valkey-backed store, so `ref_count` and the pending set below
    // are the production keys rather than a mock's in-memory maps. One
    // `store_values` per hash is the increment a live referencing object owns:
    // `ref_count` counts OBJECTS, so the residue object owes exactly one on
    // each hash however many of its rows hold it.
    let ext = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone());
    ext.store_values(&[
        (&dropped, b"residue-dropped-payload".as_slice()),
        (&kept, b"residue-kept-payload".as_slice()),
    ])
    .await
    .expect("seeding the two external blobs must succeed");
    assert_eq!(
        (ref_count(&tv, &dropped).await, ref_count(&tv, &kept).await),
        (Some(1), Some(1)),
        "fixture precondition: both blobs start at exactly one reference — the residue object's. \
         Anything else makes the release arithmetic below unreadable"
    );

    // Residue: three value rows on a section the restored state does not carry.
    // Two hold `kept`, one holds `dropped`; `obj:ext` counts them absolutely,
    // which is the shape every commit backend maintains.
    for (key_name, hash) in [
        ("kept_a", &kept),
        ("kept_b", &kept),
        ("dropped_row", &dropped),
    ] {
        let member = KeySchema::field_member("7", key_name);
        let _: () = tv
            .client()
            .pool
            .zadd(
                tv.key_schema().obj_fields_index(&oid_str),
                None,
                None,
                false,
                false,
                (11.0, member.as_str()),
            )
            .await
            .expect("ZADD stale obj:flds member");
        let _: () = tv
            .client()
            .pool
            // The column shape an external-value writer lands: `hash` + `size`,
            // no `value`. See `VALUE_HFIELDS` in `src/storage/layer.rs`.
            .hset(
                tv.key_schema().obj_field(&oid_str, "7", key_name),
                [
                    ("hash", sha256_to_hex(hash)),
                    ("size", "4096".to_owned()),
                    ("set_order", "3".to_owned()),
                    ("authoring_set_order", "3".to_owned()),
                    ("version", "11".to_owned()),
                ],
            )
            .await
            .expect("HSET stale obj:fld record");
    }
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_ext_index(&oid_str),
            [(sha256_to_hex(&kept), "2"), (sha256_to_hex(&dropped), "1")],
        )
        .await
        .expect("HSET the surviving obj:ext index");

    run_restore_of(&layer_store, oid, restored_reset_ompf_referencing(&kept))
        .await
        .expect("a restore over residue holding external references must succeed");

    // ── (a) exactly one `rewrite` release, for `dropped` ──────────────────
    assert_eq!(
        pending_members(&tv).await,
        vec![ExtRelease::commit(dropped, oid, restored_sequence_num()).member()],
        "the GC pending set must hold EXACTLY the one release the restore owes: `dropped`, \
         released by object {oid} at the restored sequence. An EMPTY set is the defect this test \
         exists for — it means the residue wipe UNLINKed `obj:ext` before `write_object` read it, \
         so the `before − after` pre-image was empty and the reference leaked with no reaper. A \
         member naming `kept` is the opposite failure: an unmatched decrement against a blob the \
         restored layer still references"
    );

    // ── (c) the index is REBUILT absolutely from the restored state ───────
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(sha256_to_hex(&kept), "1".to_owned())],
        "obj:ext must equal the restored state's absolute row counts: one row holds `kept`, \
         nothing holds `dropped`. A surviving `{}` = 2 means the index was left at the residue's \
         counts, and a surviving `dropped` field means the `HDEL` half of the plan never ran",
        sha256_to_hex(&kept)
    );

    // ── (b) one GC tick reclaims `dropped` and MUST NOT touch `kept` ──────
    process_pending_gc(
        &ext,
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;

    assert_eq!(
        ref_count(&tv, &dropped).await,
        None,
        "`dropped`'s reference was the last one, so the GC tick must take `ref_count` to zero and \
         reclaim the meta key. A surviving count is the leak; `ref_count` never reaches zero and \
         the blob is unreclaimable for the life of the store"
    );
    assert_eq!(
        blob_present(&tv, &dropped).await,
        (false, false),
        "meta AND data must both be gone — they are deleted atomically, and half a blob surviving \
         is a different failure from the whole blob surviving"
    );
    assert_eq!(
        ref_count(&tv, &kept).await,
        Some(1),
        "`kept`'s reference must be UNTOUCHED at exactly 1: the restored object still holds it. A \
         zero here is over-release — data loss rather than a leak, because the restored layer's \
         field would read back MetaNotFound forever"
    );
    assert_eq!(
        blob_present(&tv, &kept).await,
        (true, true),
        "and its blob must survive the tick"
    );
}
