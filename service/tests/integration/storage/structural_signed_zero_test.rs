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

//! A time-sample row's identity on the structural write path: the stored
//! keys of `0.0` and `-0.0` are two rows, and the compact state every
//! structural plan runs on keeps them apart, keyed by their stored bits
//! (b.mr6 PD1, kept for good by b.8jc PD4). This build writes no `-0.0` row
//! (`oodf_ffi::parse_delta` canonicalizes every time), so a `-0.0` row here is
//! a legacy one: planted by hand, or landed by [`commit_as_an_older_build`].
//!
//! # A stored pair (b.mr6)
//!
//! The object holds `(CUBE, 0.0)` and `(CUBE, -0.0)`, both external of one
//! hash, so `obj:ext` counts that hash twice. On both structural routes — the
//! `lua` backend's cached route (cold start, then the plan on the cache's
//! compact entry) and the `txn` backend's WATCH path (the slim fan-out):
//!
//! | Commit | What must hold after it | Fails if the plan conflates the two rows? |
//! |---|---|---|
//! | an unrelated structural commit (a new section) | `obj:ext` still counts the hash twice, and no release of it is pending | no: a commit that touches neither row settles no count for its hash on either form |
//! | a structural commit rewriting the `0.0` row to another external hash | `obj:ext` counts the signed-zero hash once and the new hash once, nothing is released, and the `0.0` and `-0.0` rows hold the new and the signed-zero hash | not on a plain index (the retarget of b.mr6's `-0.0` rewrite, which a `-0.0` delta can no longer reach); on the cached route over a drifted index, yes (next row) |
//! | a reset | no time-sample row of `CUBE` is stored: both signed-zero rows are deleted | under b.mr6, yes: a conflating plan deleted one key and left the other row stored; since rule E (below) a conflating plan names the other spelling too, so not on its own |
//! | a cascade (`CUBE` deleted) | both signed-zero rows are deleted, and `obj:ext` no longer counts their hash | as the reset |
//! | cached route only: a structural commit rewriting the `0.0` row to another external hash, over an `obj:ext` drifted to count the signed-zero hash once (the legacy under-count SDD §11.3.2 says earlier builds wrote) | `obj:ext` still counts the signed-zero hash once and the new hash once, nothing is released, and the `0.0` and `-0.0` rows hold the new and the signed-zero hash | yes: the commit's raise-only settle floors the hash's count on the plan's untouched-row count, which the untouched `-0.0` row makes 1; a conflating plan holds one row, which the rewrite touches, so the count falls to 0 and the hash is released while the `-0.0` row still holds it |
//!
//! The third column was checked under b.mr6 by putting the conflation back
//! (`CompactSamples::insert` folding `-0.0` into `0.0`): the reset and cascade
//! tests failed on both routes and the drifted `0.0` rewrite test on the
//! cached route (the count read 0); the unrelated-commit tests passed. It was
//! not re-run since rule E. The counted-hash half of the pre-image is also
//! pinned by the unit test
//! `untouched_ext_counts_on_a_compact_state_count_both_signed_zero_rows`
//! (`src/storage/structural.rs`).
//!
//! # A cache entry holding fewer spellings than the store (b.8jc rule E)
//!
//! A reset or cascade deletes a section's time 0 at both `+0.0` and `-0.0`,
//! and the commit checks each spelling's existence, so a spelling the entry
//! does not hold but the store does is caught (`CACHE_DIVERGED`, one reload)
//! instead of surviving the delete. All on the cached route, with the
//! structural cache's shadow check on (its default):
//!
//! | Test | Setup | Expected |
//! |---|---|---|
//! | the reviewer's sequence (test 12), reset and cascade | the store holds `CUBE` at `+0.0`; an older build's commit adds `(CUBE, -0.0)` and a stream record whose bytes carry `-0.0`; the entry's repair replays it at `+0.0` | one `CACHE_DIVERGED`, one `diverged` reload, both rows and members gone, `obj:ext` a recount, no release of a held hash. Fails with rule E reverted (the `-0.0` row survives) |
//! | a lone older-build row, then a this-build write (test 13) | the older commit is `CUBE`'s only time-0 row; a cached edit then writes `CUBE` at `+0.0`; a reset follows | one `CACHE_DIVERGED`, both rows gone. Fails with rule E reverted |
//! | an entry cold-started over a planted lone legacy row (test 14), reset and cascade | the entry holds `CUBE` at `-0.0` only, as the store does | commits on the first attempt: no `CACHE_DIVERGED`, no `INVALID_ARGUMENT`, no spelling left (binding refinement 1) |
//! | an entry holding the pair the merge then folded (test 8), reset and cascade | the entry holds both spellings; the reconciler's merge keeps `+0.0` | one `CACHE_DIVERGED`, then no `-0.0` key or member and no `+0.0` row left |
//!
//! And on the WATCH path (test 15): a reset and a cascade over `CUBE` holding
//! only `+0.0` plan the added `-0.0` delete, commit — through the keyspace
//! probe and the queued-command check — and leave no row of either spelling.
//!
//! # Replay of a this-build record carrying `-0.0` (test 6)
//!
//! Another replica of this build stores an ovlive `-0.0` sample at `+0.0`
//! and relays the writer's `-0.0` bytes; a warm entry's repair over that
//! record ends equal to the compact projection of a fresh read. (The
//! checkpoint tail twin is `checkpoint::compose`'s
//! `compose_tail_record_physically_carrying_negative_zero_composes_to_the_stored_state`.)

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use fred::interfaces::{
    HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface, StreamsInterface,
};
use ordered_float::OrderedFloat;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::{CompactState, FieldValue, ObjectState, SampleRows, TimeSample};
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::commit_chain::{commit_chain_fold, update_stream_payload};
use live_edit::storage::diff_read::{read_object_state_for_test, snapshot_to_object_state};
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::{LayerStore, HEADER_FIELD_COMMIT_CHAIN};
use live_edit::storage::notifications::{NOTIFICATION_PAYLOAD_FIELD, RELAY_OMPF_FIELD};
use live_edit::storage::structural_cache::{CacheLookup, StructuralCache};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::time_sample_merge::{merge_signed_zero_rows, negative_zero_sections};

use crate::common::owned_delta::{
    build_delta_with_negative_zero_times, reset_delta, NEGATIVE_ZERO_SENTINEL,
};
use crate::storage::snapshot_read_atomic_test::{
    create, create_cmd, create_session, delete_cmd, every_shape_state, structural, CUBE, ROOT,
    WORLD,
};
use crate::storage::structural_cache_cold_start_test::normalized_fresh_state;

/// The hash both signed-zero rows hold.
const SIGNED_ZERO_HASH: [u8; 32] = [0x5A; 32];
/// The hash the rewrite tests write to the `0.0` row.
const REWRITTEN_HASH: [u8; 32] = [0xC3; 32];
/// A section id `every_shape_state` does not use.
const NEW_SECTION: u64 = 40;
/// Another, for a second unrelated commit.
const OTHER_NEW_SECTION: u64 = 41;
const WRITER: &str = "signed-zero-writer";
/// The `client_id` an older build's commit carries.
const OLDER_WRITER: &str = "signed-zero-older-build";
/// The inline value an older build's `-0.0` commit writes.
const OLDER_VALUE: &[u8] = &[0x77];
/// A cache budget no fixture entry comes near.
const BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// The two structural routes.
#[derive(Debug, Clone, Copy)]
enum Route {
    /// `lua`: the cached route, planning on the structural cache's compact
    /// entry.
    Cached,
    /// `txn`: the WATCH path, planning on its slim fan-out read.
    Watch,
}

/// A store on `route`, and the structural cache it plans on (`Cached` only).
fn store(tv: &TestValkey, route: Route) -> (LayerStore, StructuralCache) {
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let cache = StructuralCache::new(BUDGET_BYTES);
    let ls = match route {
        Route::Cached => ls
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_structural_cache(cache.clone()),
        Route::Watch => ls.with_update_layer_backend(UpdateLayerBackend::Txn),
    };
    (ls, cache)
}

/// The lowercase hex `obj:ext` keys a hash by.
fn hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

/// `obj:ext`'s count of [`SIGNED_ZERO_HASH`], 0 when absent.
async fn ext_count(tv: &TestValkey, oid: u64) -> u64 {
    ext_count_of(tv, oid, &SIGNED_ZERO_HASH).await
}

/// `obj:ext`'s count of `hash`, 0 when absent.
async fn ext_count_of(tv: &TestValkey, oid: u64, hash: &[u8; 32]) -> u64 {
    let count: Option<String> = tv
        .client()
        .pool
        .hget(tv.key_schema().obj_ext_index(&oid.to_string()), hex(hash))
        .await
        .expect("HGET obj:ext must succeed");
    count.map_or(0, |c| c.parse().expect("a decimal count"))
}

/// The pending external-value releases.
async fn all_pending_releases(tv: &TestValkey) -> BTreeSet<String> {
    tv.client()
        .pool
        .smembers(tv.key_schema().ext_pending())
        .await
        .expect("SMEMBERS ext pending must succeed")
}

/// The pending external-value releases naming [`SIGNED_ZERO_HASH`] or
/// [`REWRITTEN_HASH`].
async fn pending_releases(tv: &TestValkey) -> BTreeSet<String> {
    let hashes = [hex(&SIGNED_ZERO_HASH), hex(&REWRITTEN_HASH)];
    all_pending_releases(tv)
        .await
        .into_iter()
        .filter(|m| hashes.iter().any(|h| m.contains(h)))
        .collect()
}

/// The `hash` column of `CUBE`'s stored time-sample row at `time` (`None`
/// when the row holds no `hash`).
async fn stored_row_hash(tv: &TestValkey, oid: u64, time: f64) -> Option<String> {
    tv.client()
        .pool
        .hget(
            tv.key_schema()
                .obj_time_sample(&oid.to_string(), &CUBE.to_string(), time),
            "hash",
        )
        .await
        .expect("HGET the row's hash must succeed")
}

/// The stored time-sample index members of `CUBE`.
async fn cube_sample_members(tv: &TestValkey, oid: u64) -> Vec<String> {
    let members: Vec<String> = tv
        .client()
        .pool
        .zrange(
            tv.key_schema().obj_time_samples_index(&oid.to_string()),
            0,
            -1,
            None,
            false,
            None,
            false,
        )
        .await
        .expect("ZRANGE obj:tss must succeed");
    members
        .into_iter()
        .filter(|m| m.starts_with(&format!("{CUBE}:")))
        .collect()
}

/// Whether `CUBE`'s time-sample row at `time` (by its bits) is stored.
async fn cube_row_exists(tv: &TestValkey, oid: u64, time: f64) -> bool {
    let exists: i64 = tv
        .client()
        .pool
        .exists(
            tv.key_schema()
                .obj_time_sample(&oid.to_string(), &CUBE.to_string(), time),
        )
        .await
        .expect("EXISTS must succeed");
    exists == 1
}

/// Assert no time-sample row or `obj:tss` member of `CUBE` at time 0 is left,
/// at either spelling.
async fn assert_no_cube_time_zero(tv: &TestValkey, oid: u64, context: &str) {
    for time in [-0.0f64, 0.0] {
        assert!(
            !cube_row_exists(tv, oid, time).await,
            "{context}: CUBE's row at {time:?} must be deleted"
        );
    }
    let members = cube_sample_members(tv, oid).await;
    for time in [-0.0f64, 0.0] {
        let member = KeySchema::time_sample_member(&CUBE.to_string(), time);
        assert!(
            !members.contains(&member),
            "{context}: CUBE's member {member} must be gone; got {members:?}"
        );
    }
}

/// `CUBE` at `(CUBE, 0.0)` external of [`SIGNED_ZERO_HASH`], created
/// through `ls` in an `every_shape_state` object.
fn state_with_cube_at_zero() -> ObjectState {
    let mut state = every_shape_state();
    state.time_samples.insert(
        (CUBE, OrderedFloat(0.0)),
        TimeSample {
            section_id: CUBE,
            time: OrderedFloat(0.0),
            value: FieldValue::External {
                hash: SIGNED_ZERO_HASH,
                size: 64,
                ext_value_index: 0,
            },
            version: 1,
        },
    );
    state
}

/// An `every_shape_state` object with `(CUBE, 0.0)` external of
/// [`SIGNED_ZERO_HASH`], created through `ls`, and then `(CUBE, -0.0)` planted
/// beside it as the same row under its own stored key (index member
/// included), with `obj:ext` raised to count both. Returns the session and
/// the object.
async fn signed_zero_object(tv: &TestValkey, ls: &LayerStore, layer: &str) -> (String, u64) {
    let session_id = create_session(tv, layer).await;
    let oid = create(ls, &session_id, layer, &state_with_cube_at_zero()).await;
    assert_eq!(ext_count(tv, oid).await, 1, "fixture: one row created");

    let (keys, pool, oid_str, sid) = (
        tv.key_schema(),
        &tv.client().pool,
        oid.to_string(),
        CUBE.to_string(),
    );
    let row: HashMap<String, String> = pool
        .hgetall(keys.obj_time_sample(&oid_str, &sid, 0.0))
        .await
        .expect("HGETALL the 0.0 row");
    assert_eq!(
        row.get("hash"),
        Some(&hex(&SIGNED_ZERO_HASH)),
        "fixture: the 0.0 row is external"
    );
    let _: () = pool
        .hset(keys.obj_time_sample(&oid_str, &sid, -0.0), row)
        .await
        .expect("HSET the -0.0 row");
    let index = keys.obj_time_samples_index(&oid_str);
    let score: f64 = pool
        .zscore(&index, KeySchema::time_sample_member(&sid, 0.0))
        .await
        .expect("ZSCORE the 0.0 member");
    let _: i64 = pool
        .zadd(
            &index,
            None,
            None,
            false,
            false,
            (score, KeySchema::time_sample_member(&sid, -0.0)),
        )
        .await
        .expect("ZADD the -0.0 member");
    let _: i64 = pool
        .hincrby(keys.obj_ext_index(&oid_str), hex(&SIGNED_ZERO_HASH), 1)
        .await
        .expect("HINCRBY obj:ext");
    assert_eq!(
        ext_count(tv, oid).await,
        2,
        "fixture: two rows hold the hash"
    );
    assert_eq!(
        cube_sample_members(tv, oid).await.len(),
        3,
        "fixture: CUBE's 1.5, 0.0 and -0.0 rows"
    );
    (session_id, oid)
}

/// Plant `(CUBE, -0.0)` alone, as a legacy inline row of `version` an older
/// build wrote before this one ran: its row and `obj:tss` member, no commit.
async fn plant_lone_legacy_row(tv: &TestValkey, oid: u64, version: u64) {
    let (keys, pool, oid_str, sid) = (
        tv.key_schema(),
        &tv.client().pool,
        oid.to_string(),
        CUBE.to_string(),
    );
    let _: () = pool
        .hset(
            keys.obj_time_sample(&oid_str, &sid, -0.0),
            [
                ("value", BASE64.encode(OLDER_VALUE)),
                ("version", version.to_string()),
            ],
        )
        .await
        .expect("HSET the legacy row");
    let _: i64 = pool
        .zadd(
            keys.obj_time_samples_index(&oid_str),
            None,
            None,
            false,
            false,
            (version as f64, KeySchema::time_sample_member(&sid, -0.0)),
        )
        .await
        .expect("ZADD the legacy member");
}

/// Land a commit on `oid` the way a build older than b.8jc landed an ovlive
/// time sample at `-0.0` on `CUBE`, bypassing this build's parser and
/// validation: the `(CUBE, -0.0)` row (inline [`OLDER_VALUE`], `version` the
/// new `seq`) and its `obj:tss` member at that score; a stream entry at
/// `{seq}-0` whose OMPF physically carries the `-0.0` time
/// ([`build_delta_with_negative_zero_times`]); and the header advanced to the
/// new `seq`, `commit_chain` folded over the entry's payload as the commit
/// script folds it. The precedent is `test_support::commit_without_fold`,
/// which leaves the chain unfolded on purpose; this one folds it, so the next
/// repair crosses the entry as an ordinary commit and replays it. Returns the
/// new `seq`.
async fn commit_as_an_older_build(tv: &TestValkey, oid: u64) -> u64 {
    let (keys, pool, oid_str, sid) = (
        tv.key_schema(),
        &tv.client().pool,
        oid.to_string(),
        CUBE.to_string(),
    );
    let header_key = keys.obj_header(&oid_str);
    let header: HashMap<String, String> = pool.hgetall(&header_key).await.expect("HGETALL obj:hdr");
    let seq: u64 = header["seq"].parse().expect("a decimal seq");
    let new_seq = seq + 1;

    let delta = oodf_ffi::OwnedDelta {
        time_samples: vec![oodf_ffi::TimeSample {
            section_id: CUBE,
            time: NEGATIVE_ZERO_SENTINEL,
            value_or_ext_hash: OLDER_VALUE.to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    };
    let delta_bytes = build_delta_with_negative_zero_times(&delta, &[NEGATIVE_ZERO_SENTINEL]);
    let ompf = OmpfContainer::serialize(&delta_bytes, &[]).expect("serialize OMPF");
    let server_ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or_default();
    let payload = update_stream_payload(oid, new_seq, OLDER_WRITER, 0, server_ts);
    let chain = commit_chain_fold(
        header
            .get(HEADER_FIELD_COMMIT_CHAIN)
            .map_or("", String::as_str),
        &payload,
    );

    let _: () = pool
        .hset(
            keys.obj_time_sample(&oid_str, &sid, -0.0),
            [
                ("value", BASE64.encode(OLDER_VALUE)),
                ("version", new_seq.to_string()),
            ],
        )
        .await
        .expect("HSET the older build's -0.0 row");
    let _: i64 = pool
        .zadd(
            keys.obj_time_samples_index(&oid_str),
            None,
            None,
            false,
            false,
            (new_seq as f64, KeySchema::time_sample_member(&sid, -0.0)),
        )
        .await
        .expect("ZADD its member");
    let fields: Vec<(&str, fred::types::Value)> = vec![
        (NOTIFICATION_PAYLOAD_FIELD, payload.as_str().into()),
        (RELAY_OMPF_FIELD, fred::types::Value::Bytes(ompf.into())),
    ];
    let _: () = pool
        .xadd(
            keys.layer_stream(oid),
            false,
            None::<()>,
            format!("{new_seq}-0").as_str(),
            fields,
        )
        .await
        .expect("XADD the older build's entry");
    let _: () = pool
        .hset(
            &header_key,
            [
                ("seq", new_seq.to_string()),
                (HEADER_FIELD_COMMIT_CHAIN, chain),
            ],
        )
        .await
        .expect("advance the header");
    new_seq
}

/// Assert `route`'s call landed where it plans: on the cached route the
/// object holds a cache entry afterwards.
fn assert_route(cache: &StructuralCache, oid: u64, route: Route, context: &str) {
    if let Route::Cached = route {
        assert!(
            matches!(cache.lookup(oid), CacheLookup::Hit(_)),
            "{context}: fixture: the cached route cold-started and kept an entry"
        );
    }
}

/// The cached entry's compact state (a hit, put back by the guard's drop).
fn entry_state(cache: &StructuralCache, oid: u64, context: &str) -> CompactState {
    match cache.lookup(oid) {
        CacheLookup::Hit(guard) => guard.entry().state.clone(),
        CacheLookup::Miss => panic!("{context}: the object must have a cached entry; got a miss"),
        CacheLookup::Uncacheable { reason } => {
            panic!("{context}: the object must have a cached entry; uncacheable {reason:?}")
        }
    }
}

/// Which spellings of `CUBE`'s time 0 `state` holds: `(+0.0, -0.0)`.
fn held_zero_spellings(state: &CompactState) -> (bool, bool) {
    (
        state.time_samples.contains(CUBE, 0.0f64.to_bits()),
        state.time_samples.contains(CUBE, (-0.0f64).to_bits()),
    )
}

/// `obj:ext` as `hash_hex → count`.
async fn ext_index(tv: &TestValkey, oid: u64) -> BTreeMap<String, u64> {
    let index: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(tv.key_schema().obj_ext_index(&oid.to_string()))
        .await
        .expect("HGETALL obj:ext must succeed");
    index
        .into_iter()
        .map(|(h, c)| (h, c.parse().expect("a decimal count")))
        .collect()
}

/// The object's external hashes counted over its stored rows — one count per
/// row, each time-sample row by its own stored key — from the fan-out
/// reference reader: what a recount of `obj:ext` writes.
async fn recount(tv: &TestValkey, oid: u64) -> BTreeMap<String, u64> {
    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("the reference read must succeed");
    let mut counts = BTreeMap::new();
    let values = snapshot
        .fields
        .iter()
        .map(|f| &f.value)
        .chain(snapshot.time_samples.iter().map(|t| &t.value));
    for value in values {
        if let FieldValue::External { hash, .. } = value {
            *counts.entry(hex(hash)).or_insert(0) += 1;
        }
    }
    counts
}

/// `obj:ext` equals a recount of the stored rows, and no pending release
/// names a hash a stored row still holds.
async fn assert_ext_index_sound(tv: &TestValkey, oid: u64, context: &str) {
    let held = recount(tv, oid).await;
    assert_eq!(
        ext_index(tv, oid).await,
        held,
        "{context}: obj:ext must equal a recount of the stored rows"
    );
    let pending = all_pending_releases(tv).await;
    for hash in held.keys() {
        assert!(
            !pending.iter().any(|m| m.contains(hash.as_str())),
            "{context}: no release may name {hash}, which a stored row still holds; pending \
             {pending:?}"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// A stored pair (b.mr6)
// ═════════════════════════════════════════════════════════════════════════════

/// An unrelated structural commit leaves both signed-zero rows counted:
/// `obj:ext` holds 2 for their hash and no release of it is pending.
///
/// What this establishes: a structural commit on a compact plan that touches
/// neither row neither releases nor recounts their hash, so the second row's
/// count is never mistaken for a surplus. What it does not: that the plan
/// holds the two rows apart. A plan that conflated them settles no count for
/// an untouched hash either, so this passes on such a plan too (checked by
/// putting the conflation back); the cached-route drifted `0.0` rewrite test
/// is the store-level one that tells them apart (the reset and cascade tests
/// did too, under b.mr6, before rule E).
async fn unrelated_commit_keeps_both_rows_counted(route: Route) {
    let tv = TestValkey::setup().await;
    let (ls, cache) = store(&tv, route);
    let layer = format!("signed-zero-unrelated-{route:?}");
    let (session_id, oid) = signed_zero_object(&tv, &ls, &layer).await;

    ls.update_layer_relayed(
        &session_id,
        &layer,
        oid,
        Some(structural(vec![create_cmd(ROOT, NEW_SECTION, "Unrelated")])),
        WRITER,
        0,
    )
    .await
    .unwrap_or_else(|e| panic!("{route:?}: the unrelated commit lands; got {e:?}"));

    assert_route(&cache, oid, route, "unrelated commit");
    assert_eq!(
        ext_count(&tv, oid).await,
        2,
        "{route:?}: both signed-zero rows still hold the hash"
    );
    assert_eq!(
        pending_releases(&tv).await,
        BTreeSet::new(),
        "{route:?}: nothing is released"
    );
    assert_eq!(cube_sample_members(&tv, oid).await.len(), 3, "{route:?}");
}

/// A structural commit (a new section) that rewrites the `(CUBE, 0.0)` row to
/// [`REWRITTEN_HASH`] beside the legacy `(CUBE, -0.0)` row moves exactly the
/// `0.0` row's count: `obj:ext` counts [`SIGNED_ZERO_HASH`] once (the `-0.0`
/// row) and [`REWRITTEN_HASH`] once, nothing is released, and the two stored
/// rows hold `REWRITTEN_HASH` and `SIGNED_ZERO_HASH`.
///
/// The retarget of b.mr6's `-0.0` rewrite: since b.8jc a delta cannot write
/// `-0.0` (the parser canonicalizes it and validation refuses it), so the
/// only rewrite of a time-0 row a client can make beside a legacy pair is at
/// `0.0`. What this establishes: the rewrite leaves the legacy row and its
/// count alone on both routes. It does not tell a conflating plan apart on a
/// plain index (a conflating plan's lone `0.0` row carries the same hash);
/// the cached-route drifted twin below does.
async fn rewrite_of_the_zero_row_moves_only_its_count(route: Route) {
    let tv = TestValkey::setup().await;
    let (ls, cache) = store(&tv, route);
    let layer = format!("signed-zero-rewrite-{route:?}");
    let (session_id, oid) = signed_zero_object(&tv, &ls, &layer).await;
    let mut delta = structural(vec![create_cmd(ROOT, NEW_SECTION, "Rewrite")]);
    delta.time_samples.push(oodf_ffi::TimeSample {
        section_id: CUBE,
        time: 0.0,
        value_or_ext_hash: REWRITTEN_HASH.to_vec(),
        ext_value_size: 64,
        ext_value_index: 1,
    });

    ls.update_layer_relayed(&session_id, &layer, oid, Some(delta), WRITER, 0)
        .await
        .unwrap_or_else(|e| panic!("{route:?}: the rewrite lands; got {e:?}"));

    assert_route(&cache, oid, route, "rewrite");
    assert_eq!(
        (
            stored_row_hash(&tv, oid, 0.0).await,
            stored_row_hash(&tv, oid, -0.0).await
        ),
        (Some(hex(&REWRITTEN_HASH)), Some(hex(&SIGNED_ZERO_HASH))),
        "{route:?}: the 0.0 row holds the rewritten hash and the -0.0 row keeps its own"
    );
    assert_eq!(
        (
            ext_count(&tv, oid).await,
            ext_count_of(&tv, oid, &REWRITTEN_HASH).await
        ),
        (1, 1),
        "{route:?}: obj:ext counts the signed-zero hash once (the -0.0 row) and the rewritten \
         hash once (the 0.0 row)"
    );
    assert_eq!(
        pending_releases(&tv).await,
        BTreeSet::new(),
        "{route:?}: nothing is released: the signed-zero hash is still held, and the \
         rewritten hash is a new reference"
    );
    assert_eq!(cube_sample_members(&tv, oid).await.len(), 3, "{route:?}");
}

/// On the cached route, over an `obj:ext` drifted to count
/// [`SIGNED_ZERO_HASH`] once while both signed-zero rows hold it — the legacy
/// under-count earlier builds wrote (SDD §11.3.2), set here out of band — a
/// structural commit (a new section) rewriting the `0.0` row to
/// [`REWRITTEN_HASH`] releases nothing: `obj:ext` still counts
/// `SIGNED_ZERO_HASH` once (the `-0.0` row's) and `REWRITTEN_HASH` once, no
/// release is pending, and the rows hold `REWRITTEN_HASH` (`0.0`) and
/// `SIGNED_ZERO_HASH` (`-0.0`).
///
/// What this establishes: on the cached route the counted-hash half of the
/// pre-image — the untouched-row counts the commit's raise-only settle floors
/// on — counts the `-0.0` row apart from the `0.0` row it rewrites. The
/// compact plan's untouched `-0.0` row gives `SIGNED_ZERO_HASH` an untouched
/// count of 1, so the floor holds the drifted count at 1 when the rewrite
/// takes it to 0. A plan that conflated the two rows holds one row, under the
/// `0.0` key, which the rewrite touches: the untouched count is 0, the count
/// falls to 0 and the hash is released while the `-0.0` row still holds it —
/// PD1's over-release (checked by putting the conflation back: the count
/// reads 0).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rewrite_of_the_zero_row_on_the_cached_route_over_a_drifted_index_releases_nothing() {
    let tv = TestValkey::setup().await;
    let route = Route::Cached;
    let (ls, cache) = store(&tv, route);
    let layer = "signed-zero-drifted-rewrite";
    let (session_id, oid) = signed_zero_object(&tv, &ls, layer).await;
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().obj_ext_index(&oid.to_string()),
            [(hex(&SIGNED_ZERO_HASH), "1")],
        )
        .await
        .expect("HSET obj:ext");
    assert_eq!(
        ext_count(&tv, oid).await,
        1,
        "fixture: obj:ext counts the hash once while two rows hold it"
    );
    let mut delta = structural(vec![create_cmd(ROOT, NEW_SECTION, "DriftedRewrite")]);
    delta.time_samples.push(oodf_ffi::TimeSample {
        section_id: CUBE,
        time: 0.0,
        value_or_ext_hash: REWRITTEN_HASH.to_vec(),
        ext_value_size: 64,
        ext_value_index: 1,
    });

    ls.update_layer_relayed(&session_id, layer, oid, Some(delta), WRITER, 0)
        .await
        .unwrap_or_else(|e| panic!("the drifted rewrite lands; got {e:?}"));

    assert_route(&cache, oid, route, "drifted rewrite");
    assert_eq!(
        (
            stored_row_hash(&tv, oid, 0.0).await,
            stored_row_hash(&tv, oid, -0.0).await
        ),
        (Some(hex(&REWRITTEN_HASH)), Some(hex(&SIGNED_ZERO_HASH))),
        "the 0.0 row holds the rewritten hash and the -0.0 row keeps the signed-zero one"
    );
    assert_eq!(
        ext_count(&tv, oid).await,
        1,
        "the floor holds the signed-zero hash's count at the -0.0 row's 1"
    );
    assert_eq!(
        ext_count_of(&tv, oid, &REWRITTEN_HASH).await,
        1,
        "the rewritten hash is counted once (the 0.0 row)"
    );
    assert_eq!(
        pending_releases(&tv).await,
        BTreeSet::new(),
        "nothing is released: the -0.0 row still holds the signed-zero hash"
    );
    assert_eq!(cube_sample_members(&tv, oid).await.len(), 3);
}

/// The commits that delete every row of `CUBE`.
#[derive(Debug, Clone, Copy)]
enum Deletion {
    /// A reset: the drain of every row.
    Reset,
    /// `CUBE` deleted: the cascade's purge of its rows.
    Cascade,
}

impl Deletion {
    fn delta(self) -> oodf_ffi::OwnedDelta {
        match self {
            Deletion::Reset => reset_delta("usda"),
            Deletion::Cascade => structural(vec![delete_cmd(WORLD, CUBE)]),
        }
    }
}

/// A reset or a cascade over `CUBE` deletes both signed-zero rows: no
/// time-sample row of `CUBE` is stored or indexed afterwards, and `obj:ext`
/// no longer counts their hash.
///
/// What this establishes: both stored keys of the pair are deleted, on both
/// routes. Under b.mr6 it also told a plan conflating the two rows apart: such
/// a plan deleted one key and left the other row stored (checked then by
/// putting the conflation back: the `-0.0` row stayed). Since b.8jc rule E has
/// a reset or cascade name the other spelling of every time 0 it deletes, so
/// a conflating plan names both keys too and this no longer discriminates it
/// on its own; the drifted `0.0` rewrite test and the unit tests on the
/// compact form do.
async fn deletion_deletes_both_rows(route: Route, deletion: Deletion) {
    let tv = TestValkey::setup().await;
    let (ls, cache) = store(&tv, route);
    let layer = format!("signed-zero-{deletion:?}-{route:?}");
    let (session_id, oid) = signed_zero_object(&tv, &ls, &layer).await;

    ls.update_layer_relayed(&session_id, &layer, oid, Some(deletion.delta()), WRITER, 0)
        .await
        .unwrap_or_else(|e| panic!("{route:?}: the {deletion:?} lands; got {e:?}"));

    assert_route(&cache, oid, route, &format!("{deletion:?}"));
    assert_no_cube_time_zero(&tv, oid, &format!("{route:?}")).await;
    assert_eq!(
        cube_sample_members(&tv, oid).await,
        Vec::<String>::new(),
        "{route:?}: no time-sample row of CUBE is indexed"
    );
    assert_eq!(
        ext_count(&tv, oid).await,
        0,
        "{route:?}: the hash is no row's"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unrelated_structural_commit_on_the_cached_route_keeps_both_signed_zero_rows_counted() {
    unrelated_commit_keeps_both_rows_counted(Route::Cached).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unrelated_structural_commit_on_the_watch_path_keeps_both_signed_zero_rows_counted() {
    unrelated_commit_keeps_both_rows_counted(Route::Watch).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rewrite_of_the_zero_row_beside_a_legacy_negative_zero_row_on_the_cached_route_moves_only_its_count(
) {
    rewrite_of_the_zero_row_moves_only_its_count(Route::Cached).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rewrite_of_the_zero_row_beside_a_legacy_negative_zero_row_on_the_watch_path_moves_only_its_count(
) {
    rewrite_of_the_zero_row_moves_only_its_count(Route::Watch).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_on_the_cached_route_deletes_both_signed_zero_rows() {
    deletion_deletes_both_rows(Route::Cached, Deletion::Reset).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_on_the_watch_path_deletes_both_signed_zero_rows() {
    deletion_deletes_both_rows(Route::Watch, Deletion::Reset).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cascade_on_the_cached_route_deletes_both_signed_zero_rows() {
    deletion_deletes_both_rows(Route::Cached, Deletion::Cascade).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cascade_on_the_watch_path_deletes_both_signed_zero_rows() {
    deletion_deletes_both_rows(Route::Watch, Deletion::Cascade).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Rule E on the cached route: an entry holding fewer spellings than the store
// ═════════════════════════════════════════════════════════════════════════════

/// A cached-route store with a metrics registry hooked onto it and its
/// client, and the object `state` created through it.
struct CachedFx {
    tv: TestValkey,
    metrics: Arc<Metrics>,
    cache: StructuralCache,
    ls: LayerStore,
    session_id: String,
    layer: String,
    oid: u64,
}

impl CachedFx {
    async fn new(layer: &str, state: &ObjectState) -> Self {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let cache = StructuralCache::new(BUDGET_BYTES);
        assert!(
            cache.shadow_check_for_test(),
            "fixture: the shadow check is on, its default"
        );
        let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_structural_cache(cache.clone())
            .with_metrics(Arc::clone(&metrics));
        let session_id = create_session(&tv, layer).await;
        let oid = create(&ls, &session_id, layer, state).await;
        Self {
            tv,
            metrics,
            cache,
            ls,
            session_id,
            layer: layer.to_owned(),
            oid,
        }
    }

    async fn update(&self, delta: oodf_ffi::OwnedDelta) -> Result<(), ServiceError> {
        self.ls
            .update_layer_relayed(
                &self.session_id,
                &self.layer,
                self.oid,
                Some(delta),
                WRITER,
                0,
            )
            .await
            .map(|_| ())
    }

    /// An unrelated structural commit creating `section`: a cold start the
    /// first time, a repair over anything committed since afterwards.
    async fn unrelated(&self, section: u64) {
        self.update(structural(vec![create_cmd(ROOT, section, "Unrelated")]))
            .await
            .expect("the unrelated structural commit lands");
    }

    fn entry_state(&self, context: &str) -> CompactState {
        entry_state(&self.cache, self.oid, context)
    }

    fn cell(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        sample_value(&render(&self.metrics), name, labels).unwrap_or(0.0) as u64
    }

    fn divergences(&self) -> u64 {
        self.cell("live_edit_structural_cache_divergence_total", &[])
    }

    fn diverged_reloads(&self) -> u64 {
        self.cell(
            "live_edit_structural_cache_reloads_total",
            &[("reason", "diverged")],
        )
    }
}

/// The reviewer's sequence (test 12): the store holds `CUBE` at `+0.0`, and
/// the cache entry holds it there; an older build's commit adds `(CUBE,
/// -0.0)`, its stream record physically carrying `-0.0`. The entry's repair
/// replays that record through this build's parser, which canonicalizes it
/// to `+0.0`, so the entry holds `CUBE`'s time 0 at `+0.0` only while the
/// store holds both rows (asserted, the premise). A reset — or the cascade
/// of deleting `CUBE` — then deletes `+0.0` (held) and, by rule E, `-0.0`
/// with an `absent` digest. The store holds that row, so the first attempt
/// answers `CACHE_DIVERGED`, the entry is reloaded from the store once and
/// the retry deletes both. Afterwards no row or member of `CUBE`'s time 0 is
/// left at either spelling, `obj:ext` equals a recount, and no release names
/// a hash a stored row holds.
///
/// Fails with rule E reverted (its two `emit_other_spellings` call sites in
/// `domain::delta`): the first attempt commits, deleting `+0.0` only, and the
/// `-0.0` row and member survive.
async fn the_reviewers_sequence_diverges_once_and_deletes_both_rows(deletion: Deletion) {
    let fx = CachedFx::new(
        &format!("signed-zero-reviewer-{deletion:?}"),
        &state_with_cube_at_zero(),
    )
    .await;
    fx.unrelated(NEW_SECTION).await;
    assert_eq!(
        held_zero_spellings(&fx.entry_state("warm-up")),
        (true, false),
        "fixture: the cold-started entry holds CUBE at +0.0"
    );

    commit_as_an_older_build(&fx.tv, fx.oid).await;
    fx.unrelated(OTHER_NEW_SECTION).await;
    assert_eq!(
        held_zero_spellings(&fx.entry_state("after the repair")),
        (true, false),
        "premise: the repair replayed the older build's -0.0 record at +0.0, so the entry \
         holds CUBE's time 0 at +0.0 only"
    );
    assert!(
        cube_row_exists(&fx.tv, fx.oid, 0.0).await && cube_row_exists(&fx.tv, fx.oid, -0.0).await,
        "premise: the store holds CUBE at both spellings"
    );
    let (divergences, reloads) = (fx.divergences(), fx.diverged_reloads());

    fx.update(deletion.delta())
        .await
        .unwrap_or_else(|e| panic!("the {deletion:?} commits after one reload; got {e:?}"));

    let context = format!("{deletion:?}");
    assert_no_cube_time_zero(&fx.tv, fx.oid, &context).await;
    assert_eq!(
        (
            fx.divergences() - divergences,
            fx.diverged_reloads() - reloads
        ),
        (1, 1),
        "{context}: the first attempt answers CACHE_DIVERGED and the entry is reloaded once"
    );
    assert_ext_index_sound(&fx.tv, fx.oid, &context).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_on_the_cached_route_after_an_older_builds_negative_zero_commit_diverges_once_and_deletes_both_rows(
) {
    the_reviewers_sequence_diverges_once_and_deletes_both_rows(Deletion::Reset).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cascade_on_the_cached_route_after_an_older_builds_negative_zero_commit_diverges_once_and_deletes_both_rows(
) {
    the_reviewers_sequence_diverges_once_and_deletes_both_rows(Deletion::Cascade).await;
}

/// Test 13: as the reviewer's sequence, but the older build's `-0.0` commit
/// is `CUBE`'s only time-0 row. The entry's repair replays it at `+0.0` — a
/// phantom row the store lacks — and a cached-route edit then writes `CUBE`
/// at `+0.0`, making it real. The store now holds both spellings and the
/// entry `+0.0` only, so a reset answers `CACHE_DIVERGED` once, reloads, and
/// deletes both rows.
///
/// Fails with rule E reverted: the reset deletes `+0.0` only, which the
/// entry rightly believes exists, and the `-0.0` row survives.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_after_a_lone_older_build_negative_zero_commit_and_a_this_build_write_diverges_once_and_deletes_both_rows(
) {
    let fx = CachedFx::new("signed-zero-lone-older", &every_shape_state()).await;
    fx.unrelated(NEW_SECTION).await;
    commit_as_an_older_build(&fx.tv, fx.oid).await;
    let mut edit = structural(vec![create_cmd(ROOT, OTHER_NEW_SECTION, "Edit")]);
    edit.time_samples.push(oodf_ffi::TimeSample {
        section_id: CUBE,
        time: 0.0,
        value_or_ext_hash: vec![0x01],
        ext_value_size: 0,
        ext_value_index: 0,
    });
    fx.update(edit).await.expect("the this-build write lands");
    assert_eq!(
        held_zero_spellings(&fx.entry_state("after the write")),
        (true, false),
        "premise: the entry holds CUBE's time 0 at +0.0 only"
    );
    assert!(
        cube_row_exists(&fx.tv, fx.oid, 0.0).await && cube_row_exists(&fx.tv, fx.oid, -0.0).await,
        "premise: the store holds CUBE at both spellings"
    );
    let (divergences, reloads) = (fx.divergences(), fx.diverged_reloads());

    fx.update(reset_delta("usda"))
        .await
        .unwrap_or_else(|e| panic!("the reset commits after one reload; got {e:?}"));

    assert_no_cube_time_zero(&fx.tv, fx.oid, "reset").await;
    assert_eq!(
        (
            fx.divergences() - divergences,
            fx.diverged_reloads() - reloads
        ),
        (1, 1),
        "the first attempt answers CACHE_DIVERGED and the entry is reloaded once"
    );
    assert_ext_index_sound(&fx.tv, fx.oid, "reset").await;
}

/// Test 14, binding refinement 1: a planted lone legacy `(CUBE, -0.0)` row
/// and an entry cold-started over it, which holds `CUBE`'s time 0 at `-0.0`
/// only, as the store does. A cached-route reset — or the cascade of
/// deleting `CUBE` — deletes `-0.0` (held, exists) and, by rule E, `+0.0`
/// (not held, absent in the store too), so it commits on its first attempt:
/// no `CACHE_DIVERGED`, no `INVALID_ARGUMENT` from the `-0.0` backstop (the
/// delete is a mutation, not a delta), and no spelling of `CUBE`'s time 0
/// left.
async fn a_legacy_row_under_a_cold_started_entry_commits_on_the_first_attempt(deletion: Deletion) {
    let tv_layer = format!("signed-zero-legacy-cold-{deletion:?}");
    let fx = CachedFx::new(&tv_layer, &every_shape_state()).await;
    plant_lone_legacy_row(&fx.tv, fx.oid, 1).await;
    fx.unrelated(NEW_SECTION).await;
    assert_eq!(
        held_zero_spellings(&fx.entry_state("cold start")),
        (false, true),
        "premise: the cold-started entry holds CUBE's time 0 at -0.0 only, keyed by bits"
    );
    let (divergences, reloads) = (fx.divergences(), fx.diverged_reloads());

    let committed = fx.update(deletion.delta()).await;

    let context = format!("{deletion:?}");
    assert!(
        committed.is_ok(),
        "{context}: a user's commit never fails because of a legacy row; got {committed:?}"
    );
    assert_eq!(
        (
            fx.divergences() - divergences,
            fx.diverged_reloads() - reloads
        ),
        (0, 0),
        "{context}: committed on the first attempt, with no CACHE_DIVERGED"
    );
    assert_no_cube_time_zero(&fx.tv, fx.oid, &context).await;
    assert_ext_index_sound(&fx.tv, fx.oid, &context).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_over_a_legacy_negative_zero_row_under_a_cold_started_entry_commits_on_the_first_attempt(
) {
    a_legacy_row_under_a_cold_started_entry_commits_on_the_first_attempt(Deletion::Reset).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cascade_over_a_legacy_negative_zero_row_under_a_cold_started_entry_commits_on_the_first_attempt(
) {
    a_legacy_row_under_a_cold_started_entry_commits_on_the_first_attempt(Deletion::Cascade).await;
}

/// Test 8: an entry cold-started over a stored pair holds both spellings of
/// `CUBE`'s time 0. The reconciler's signed-zero merge then folds the pair
/// (both rows of one version and value: rule B keeps `+0.0`), deleting the
/// `-0.0` row without moving the entry's cut. A reset — or the cascade of
/// deleting `CUBE` — deletes `-0.0` with an `exists` digest the store no
/// longer bears out, so the first attempt answers `CACHE_DIVERGED`, the entry
/// is reloaded once, and the retry commits: no `-0.0` key or member, and no
/// `+0.0` row orphaned.
async fn a_merge_under_a_cached_pair_diverges_once_and_deletes_every_row(deletion: Deletion) {
    let layer = format!("signed-zero-merged-{deletion:?}");
    let fx = CachedFx::new(&layer, &every_shape_state()).await;
    // The stored pair, as `signed_zero_object` builds it, on this fixture's
    // object: the `+0.0` row through a commit, the `-0.0` row beside it.
    let mut set_zero = structural(vec![create_cmd(ROOT, NEW_SECTION, "Warm")]);
    set_zero.time_samples.push(oodf_ffi::TimeSample {
        section_id: CUBE,
        time: 0.0,
        value_or_ext_hash: vec![0x01],
        ext_value_size: 0,
        ext_value_index: 0,
    });
    fx.update(set_zero).await.expect("the +0.0 write lands");
    let (pool, keys, oid_str) = (&fx.tv.client().pool, fx.tv.key_schema(), fx.oid.to_string());
    let row: HashMap<String, String> = pool
        .hgetall(keys.obj_time_sample(&oid_str, &CUBE.to_string(), 0.0))
        .await
        .expect("HGETALL the +0.0 row");
    let version: u64 = row["version"].parse().expect("a decimal version");
    plant_lone_legacy_row(&fx.tv, fx.oid, version).await;
    let _: () = pool
        .hset(keys.obj_time_sample(&oid_str, &CUBE.to_string(), -0.0), row)
        .await
        .expect("make the -0.0 row the +0.0 row's twin");
    // A fresh entry over the pair: drop the warm one, then cold-start.
    fx.cache.invalidate(fx.oid);
    fx.unrelated(OTHER_NEW_SECTION).await;
    assert_eq!(
        held_zero_spellings(&fx.entry_state("cold start over the pair")),
        (true, true),
        "premise: the entry holds both spellings"
    );

    let sections = negative_zero_sections(&cube_sample_members(&fx.tv, fx.oid).await);
    let merged = merge_signed_zero_rows(fx.tv.client(), keys, &oid_str, &sections, None).await;
    assert_eq!(
        merged.merged, 1,
        "premise: the merge kept the +0.0 row and dropped the -0.0 one"
    );
    assert!(!cube_row_exists(&fx.tv, fx.oid, -0.0).await);
    let (divergences, reloads) = (fx.divergences(), fx.diverged_reloads());

    fx.update(deletion.delta())
        .await
        .unwrap_or_else(|e| panic!("the {deletion:?} commits after one reload; got {e:?}"));

    let context = format!("{deletion:?}");
    assert_no_cube_time_zero(&fx.tv, fx.oid, &context).await;
    assert_eq!(
        (
            fx.divergences() - divergences,
            fx.diverged_reloads() - reloads
        ),
        (1, 1),
        "{context}: the entry's -0.0 row fails its existence digest once and is reloaded"
    );
    assert_ext_index_sound(&fx.tv, fx.oid, &context).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_on_a_cached_pair_after_the_merge_diverges_once_and_leaves_no_row() {
    a_merge_under_a_cached_pair_diverges_once_and_deletes_every_row(Deletion::Reset).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cascade_on_a_cached_pair_after_the_merge_diverges_once_and_leaves_no_row() {
    a_merge_under_a_cached_pair_diverges_once_and_deletes_every_row(Deletion::Cascade).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Replay of a this-build record whose bytes carry `-0.0`
// ═════════════════════════════════════════════════════════════════════════════

/// Test 6: another replica of this build commits an ovlive sample at `-0.0`
/// on `CUBE`: it stores the row at `+0.0` (its parser canonicalized the
/// time) and relays the writer's bytes, `-0.0` included, on the stream. This
/// replica's cache entry, warm from before, repairs over that record on its
/// next call: the replay goes through the same parser, so the entry ends
/// equal to the compact projection of a fresh read of the store — no `-0.0`
/// row believed, none stored.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn repair_over_a_this_build_record_carrying_negative_zero_equals_a_fresh_read() {
    let fx = CachedFx::new("signed-zero-replay", &every_shape_state()).await;
    fx.unrelated(NEW_SECTION).await;
    let delta = oodf_ffi::OwnedDelta {
        time_samples: vec![oodf_ffi::TimeSample {
            section_id: CUBE,
            time: NEGATIVE_ZERO_SENTINEL,
            value_or_ext_hash: vec![0x42],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    };
    let bytes = build_delta_with_negative_zero_times(&delta, &[NEGATIVE_ZERO_SENTINEL]);
    let relay = OmpfContainer::serialize(&bytes, &[]).expect("serialize OMPF");
    let parsed = oodf_ffi::parse_delta(&bytes).expect("the patched delta parses");
    let other_replica = LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua);
    other_replica
        .update_layer(
            &fx.session_id,
            &fx.layer,
            fx.oid,
            Some(parsed),
            Some(bytes::Bytes::from(relay)),
            "signed-zero-other-replica",
            0,
        )
        .await
        .expect("the other replica's commit lands");
    assert!(
        cube_row_exists(&fx.tv, fx.oid, 0.0).await && !cube_row_exists(&fx.tv, fx.oid, -0.0).await,
        "premise: the other replica stored the row at +0.0"
    );

    fx.unrelated(OTHER_NEW_SECTION).await;

    let entry = fx.entry_state("after the repair");
    assert_eq!(held_zero_spellings(&entry), (true, false));
    assert_eq!(
        entry,
        normalized_fresh_state(&fx.tv, fx.oid).await,
        "the repaired entry equals the compact projection of a fresh read"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Rule E on the WATCH path
// ═════════════════════════════════════════════════════════════════════════════

/// Test 15: on the WATCH path, a reset — or the cascade of deleting `CUBE` —
/// over `CUBE` holding time 0 only at `+0.0` plans rule E's added `-0.0`
/// delete (shown on the engine's plan of the same delta over the stored
/// state), and commits: the keyspace probe type-checks the added record's
/// absent key and the queued-command check counts its `DEL` and `ZREM`, so a
/// miscount in either refuses the commit. No row or member of `CUBE`'s time
/// 0 is left at either spelling; the added delete of an absent row changes
/// nothing else (`obj:ext` equals a recount).
async fn watch_path_deletion_plans_and_commits_the_added_delete(deletion: Deletion) {
    let tv = TestValkey::setup().await;
    let (ls, _cache) = store(&tv, Route::Watch);
    let layer = format!("signed-zero-watch-{deletion:?}");
    let session_id = create_session(&tv, &layer).await;
    let oid = create(&ls, &session_id, &layer, &state_with_cube_at_zero()).await;
    let stored = snapshot_to_object_state(
        &read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
            .await
            .expect("the reference read must succeed"),
    );
    let plan = live_edit::domain::delta::apply_delta(&mut stored.clone(), Some(deletion.delta()))
        .expect("the delta plans");
    let deleted_bits: Vec<u64> = plan
        .iter()
        .filter_map(|m| match m {
            live_edit::domain::delta::StorageMutation::DeleteTimeSample { section_id, time }
                if *section_id == CUBE && time.into_inner() == 0.0 =>
            {
                Some(time.into_inner().to_bits())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        deleted_bits,
        vec![0.0f64.to_bits(), (-0.0f64).to_bits()],
        "premise: the plan deletes CUBE's time 0 at +0.0, then at -0.0 (rule E)"
    );

    ls.update_layer_relayed(&session_id, &layer, oid, Some(deletion.delta()), WRITER, 0)
        .await
        .unwrap_or_else(|e| panic!("the WATCH-path {deletion:?} commits; got {e:?}"));

    let context = format!("WATCH {deletion:?}");
    assert_no_cube_time_zero(&tv, oid, &context).await;
    assert_ext_index_sound(&tv, oid, &context).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_on_the_watch_path_over_a_positive_zero_row_commits_with_rule_e_added_delete() {
    watch_path_deletion_plans_and_commits_the_added_delete(Deletion::Reset).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cascade_on_the_watch_path_over_a_positive_zero_row_commits_with_rule_e_added_delete() {
    watch_path_deletion_plans_and_commits_the_added_delete(Deletion::Cascade).await;
}
