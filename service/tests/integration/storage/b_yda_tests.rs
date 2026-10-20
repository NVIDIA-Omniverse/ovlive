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

// b.yda W1 — the `held` rule: a commit that rewrites a row with the hash that
// row was ALREADY holding hands its pre-flight `+1` straight back, even on an
// object whose `le:{oid}:obj:ext` index has never been seeded.
//
// ── The residual this closes ─────────────────────────────────────────────────
//
// b.qpf made `ref_count` count LIVE OBJECTS: one increment per object per
// distinct hash. The `surplus_acquire` release is what keeps that true when a
// commit re-references a hash the object already holds — the gRPC layer's
// pre-flight `store_values` pays a `+1` per distinct hash on EVERY delta, so a
// re-reference buys nothing and owes the increment back.
//
// Pre-b.yda that release fired on one proof only: `before > 0`, read from
// `le:{oid}:obj:ext`. An object materialized before b.qpf carries no index, so
// `before` reads 0 for a hash it genuinely holds — AND a same-hash rewrite nets
// `delta == 0`, so the commit never seeds the index either. Both proofs are
// unavailable at once, permanently: such an object leaked its pre-flight `+1` on
// every commit for its whole life, and retirement released exactly one. That is
// residual (c) on the ticket, and `a_same_hash_rewrite_on_an_unseeded_object_*`
// below is its regression test.
//
// b.yda adds the second proof. A row whose ORIGINAL pre-image equals its FINAL
// write proves the object held the hash without consulting any count — row-local
// evidence, and NOT a count clamp: it asserts nothing about how many rows hold
// the hash, which is the direction `KeySchema::obj_ext_index`'s `# Drift`
// section refuses.
//
// ── Why every arm runs on BOTH backends ──────────────────────────────────────
//
// The rule has two transcriptions that share one `ext:meta` `ref_count`:
// `fold_ext_row_transitions` + `plan_ext_index` (the definition, reached by the
// WATCH/MULTI/EXEC slow path and by `write_object`) and
// `scripts/update_layer_fast.lua`'s settle passes 6a/6b. Disagreeing about
// `held` is not a cosmetic divergence — it releases a `+1` that bought a real,
// NEW reference, which is a blob deleted under a live row. So each arm is one
// body with two `#[tokio::test]` wrappers, following the b.zvd / b.qpf shape,
// and each asserts the pending member set and the `obj:ext` index as WHOLE
// values so the two backends' answers are comparable literally rather than
// approximately.
//
// Three arms run a second axis on top of that: the ROW KIND. The `held` rule has
// one transcription per row kind at each commit site — the Lua script's `SF` and
// `TS` arms each call `ext_note_row`, and the txn path folds both through
// `fold_ext_row_transitions` — so a file that only ever wrote field rows would
// leave the two time-sample transcriptions deletable with the suite green.
//
// ── Why every arm also reads the `reason` cells ──────────────────────────────
//
// The release REASON reaches exactly one place: the `reason` label on
// `live_edit_ext_ref_released_total`. It is NOT in the pending member —
// `ExtRelease::member()` is `{hash_hex}:{object_id}:{event}` — so a `rewrite` and
// a `surplus_acquire` of the same hash, by the same object, on the same sequence
// spell the byte-identical member, and every keyspace assertion in this file
// passes under either. Both release exactly one reference, so the `ref_count` and
// GC assertions pass under either too. That counter is therefore the ONLY witness
// separating "the object gave up a hash it no longer holds" from "the object
// handed back an increment for a hash it still holds" — which is the whole of
// what the `held` rule decides, and the two spellings imply opposite things about
// `obj:ext`. Every arm reads all four cells as one vector (`release_reasons`),
// because a single-cell check has the same blind spot one step in: it cannot see
// the OTHER cell that moved.
//
// ── The ORIGINAL-vs-FINAL pairing, and the defect it guards ─────────────────
//
// Three arms write the same row TWICE in ONE delta. That is expressible —
// `storage::lua::encode_delta` does not dedup `SetField` records and
// `validate_delta_bounds` does not reject a repeat of one
// `(section_id, key_name)` — and it is the shape a per-RECORD fold gets wrong:
// the second record's pre-image read returns what the FIRST record of the same
// commit just wrote, so a row that acquired a brand-new hash reports it as
// `held` and hands back the increment that bought it. That was the b.yda round-3
// defect on the Lua side, and `a_row_written_twice_*` / `an_absent_row_*` are
// its regression tests: they end with a GC tick and a live-blob assertion,
// because the failure is data LOSS rather than a leak.
//
// The two backends reach the same answer by different routes and that is worth
// knowing when one of these fails: the Lua apply pass records
// `ext_row_original` on a row's FIRST touch only and settle pass 6a folds one
// `-original +final` step per row, while the txn path's mutation list cannot
// contain a duplicate row key at all (`apply_delta` builds it from
// `ObjectState`'s per-row maps, so the repeat collapses to the final write
// before `ext_rows_written` ever sees it). A regression that only breaks the
// recording discipline therefore shows up on the Lua wrapper alone.
//
// ── Deliberate leaks these arms PIN rather than fix ─────────────────────────
//
// `a_row_whose_original_and_final_differ_*` leaves one reference standing on the
// intermediate hash and asserts it: the pre-flight store pays a `+1` per
// DISTINCT hash the delta names, including a hash written only as an
// intermediate state of a doubly-written row, and that increment is neither
// matched nor handed back. Releasing it would need the fold to know whether some
// OTHER object holds the hash, which is not knowable from one object's rows —
// so it is `storage::ext_reconcile`'s to drain, and the assertion here is that
// the blob SURVIVES the GC tick. A build that "fixed" it in the commit path
// should fail that arm and be read as an over-release until proven otherwise.

use std::collections::HashMap;
use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface};

use live_edit::config::UpdateLayerBackend;
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::{process_pending_gc, ExtRelease};
use live_edit::storage::lua::is_fast_path;
use live_edit::storage::test_support::relay::RelayedUpdates;
use oodf_ffi::{OwnedDelta, SetField, TimeSample};

use crate::common::owned_delta::{
    delta_with_set_field_external, delta_with_set_field_inline, delta_with_time_sample_external,
    delta_with_time_sample_inline,
};
use crate::storage::b_qpf_tests::{
    assert_blob_alive, create_session, empty_state, ext_hash, ext_hash_hex, ext_hash_vec,
    ext_ref_released, ext_store, layer_store, obj_ext_sorted, pending_members, pre_store,
    ref_count, ALL_RELEASE_REASONS, EXT_SIZE, FIELD_A, REASON_REWRITE, REASON_SURPLUS, SECTION_ID,
};

// ═════════════════════════════════════════════════════════════════════════════
// Fixture helpers
// ═════════════════════════════════════════════════════════════════════════════

/// The time every time-sample row in this file carries.
///
/// The row key is `obj:ts:{section_id}:{hex_encode_f64(time)}` — a bit-exact
/// encoding of the `f64` — so two records built from this ONE constant land on
/// the same row key by construction. That is what makes a doubly-written
/// time-sample row a genuine repeat of one row rather than two rows a hair
/// apart, which is the whole shape arm 1 exists to test.
const SAMPLE_TIME: f64 = 2.5;

/// Which value-row shape an arm drives.
///
/// A real coverage axis, not a cosmetic one, and for the same reason it is one
/// in `b_qpf_tests`: the `held` rule is transcribed once per row kind at each of
/// the two commit sites — `update_layer_fast.lua` records `ext_row_original` on
/// a row's first touch from BOTH its `SF` and its `TS` arm, and the txn path
/// folds both through `fold_ext_row_transitions` — so a file that only ever
/// wrote a FIELD row would leave the two time-sample transcriptions deletable
/// with the suite green.
#[derive(Clone, Copy, Debug)]
enum RowKind {
    Field,
    TimeSample,
}

impl RowKind {
    /// Discriminator for layer names and assertion context, so a failure names
    /// the row shape as well as the backend.
    fn label(self) -> &'static str {
        match self {
            RowKind::Field => "field",
            RowKind::TimeSample => "ts",
        }
    }

    /// A one-row delta writing this kind with an INLINE value.
    ///
    /// `set_order` is a field-only concept — time samples are ordered by their
    /// `time` and carry no authoring rank — so it is accepted and ignored on the
    /// time-sample arm rather than being pushed onto every caller.
    fn inline_delta(self, value: Vec<u8>, set_order: u64) -> OwnedDelta {
        match self {
            RowKind::Field => delta_with_set_field_inline(SECTION_ID, FIELD_A, value, set_order),
            RowKind::TimeSample => delta_with_time_sample_inline(SECTION_ID, SAMPLE_TIME, value),
        }
    }

    /// A one-row delta writing this kind with an EXTERNAL hash — the SAME row
    /// [`RowKind::inline_delta`] writes, so the pair is a rewrite of one row.
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

    /// A delta writing ONE row of this kind TWICE, with `first` then `second`.
    ///
    /// `None` means an inline value. The row key is identical for both records —
    /// same `(section_id, key_name)` for a field, same `(section_id, time)` for a
    /// time sample — which is exactly the shape a per-RECORD fold misreads:
    /// `encode_delta` emits both records verbatim and the second one's pre-image
    /// `HGET` reads back what the first just wrote.
    fn delta_writing_one_row_twice(self, first: Option<u8>, second: Option<u8>) -> OwnedDelta {
        match self {
            RowKind::Field => OwnedDelta {
                set_fields: vec![field_record(first, 0), field_record(second, 1)],
                ..Default::default()
            },
            RowKind::TimeSample => OwnedDelta {
                time_samples: vec![time_sample_record(first), time_sample_record(second)],
                ..Default::default()
            },
        }
    }
}

/// One `SetField` record on the shared field row. `None` ⇒ an inline value.
fn field_record(hash_byte: Option<u8>, set_order: u64) -> SetField {
    match hash_byte {
        Some(byte) => SetField {
            section_id: SECTION_ID,
            key_name: FIELD_A.to_owned(),
            value_or_ext_hash: ext_hash_vec(byte),
            ext_value_size: EXT_SIZE,
            ext_value_index: 1,
            set_order,
        },
        None => SetField {
            section_id: SECTION_ID,
            key_name: FIELD_A.to_owned(),
            value_or_ext_hash: vec![0x11, 0x22],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        },
    }
}

/// One `TimeSample` record on the shared time-sample row. `None` ⇒ inline.
///
/// Both records carry [`SAMPLE_TIME`] verbatim, so `hex_encode_f64` spells one
/// row key for the pair.
fn time_sample_record(hash_byte: Option<u8>) -> TimeSample {
    match hash_byte {
        Some(byte) => TimeSample {
            section_id: SECTION_ID,
            time: SAMPLE_TIME,
            value_or_ext_hash: ext_hash_vec(byte),
            ext_value_size: EXT_SIZE,
            ext_value_index: 1,
        },
        None => TimeSample {
            section_id: SECTION_ID,
            time: SAMPLE_TIME,
            value_or_ext_hash: vec![0x11, 0x22],
            ext_value_size: 0,
            ext_value_index: 0,
        },
    }
}

/// The `live_edit_ext_ref_released_total` reading, as one whole tuple in
/// [`ALL_RELEASE_REASONS`] order.
///
/// Every arm asserts the WHOLE vector rather than the one cell it cares about.
/// The reason is the only thing that distinguishes a `rewrite` from a
/// `surplus_acquire` anywhere in the system — `ExtRelease::member()` is
/// `{hash_hex}:{object_id}:{event}` and carries no reason at all, so the pending
/// member an arm asserts whole is byte-identical under either spelling — and
/// both release exactly one reference, so every count-shaped assertion in this
/// file passes under either. A cell-at-a-time check has the same blind spot one
/// step in: it cannot see the OTHER cell that moved.
fn release_reasons(metrics: &Metrics) -> Vec<(&'static str, u64)> {
    ALL_RELEASE_REASONS
        .iter()
        .map(|reason| (*reason, ext_ref_released(metrics, reason)))
        .collect()
}

/// [`release_reasons`] with every cell at zero — the reading for an arm that
/// releases nothing.
fn no_releases() -> Vec<(&'static str, u64)> {
    ALL_RELEASE_REASONS.iter().map(|r| (*r, 0)).collect()
}

/// [`release_reasons`] with exactly `reason` at 1 and every other cell at zero.
fn one_release(reason: &str) -> Vec<(&'static str, u64)> {
    ALL_RELEASE_REASONS
        .iter()
        .map(|r| (*r, u64::from(*r == reason)))
        .collect()
}

/// Prove which backend the delta about to be dispatched will really run on.
///
/// `update_layer_once` dispatches on `backend == Lua && is_fast_path(&delta)`, so
/// under `Lua` a fast-path-INELIGIBLE delta silently falls through to the
/// WATCH/MULTI/EXEC path and the `_lua` wrapper re-runs its `_txn` sibling —
/// leaving `update_layer_fast.lua`'s settle passes, one of the two
/// transcriptions of the `held` rule, covered by nothing while the suite stays
/// green. Restated here rather than imported from `b_qpf_tests` because that
/// copy is private to it and the claim is one every arm in this file depends on.
fn assert_fast_path_eligible(delta: &OwnedDelta, ctx: &str) {
    assert!(
        is_fast_path(&Some(delta.clone())),
        "{ctx}: this delta must be fast-path ELIGIBLE, or the `_lua` wrapper of this arm is a \
         duplicate of the `_txn` one and the Lua transcription of the b.yda `held` rule is \
         untested"
    );
}

/// `HGETALL le:{object_id}:obj:ext`, as the raw map — used where the assertion
/// is that the index does not exist at all.
async fn obj_ext_raw(
    tv: &live_edit::storage::test_support::TestValkey,
    object_id: u64,
) -> HashMap<String, String> {
    tv.client()
        .pool
        .hgetall(tv.key_schema().obj_ext_index(&object_id.to_string()))
        .await
        .expect("HGETALL obj:ext must succeed")
}

/// The `le:ext:{gc}:pending` member a commit-derived release writes.
fn commit_member(hash: [u8; 32], object_id: u64, seq: u64) -> String {
    ExtRelease::commit(hash, object_id, seq).member()
}

fn backend_label(backend: UpdateLayerBackend) -> &'static str {
    match backend {
        UpdateLayerBackend::Lua => "lua",
        UpdateLayerBackend::LuaWatchStructural => "lua-watch-structural",
        UpdateLayerBackend::Txn => "txn",
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// 1. a row written twice in one delta, ending on a hash it did NOT hold
// ═════════════════════════════════════════════════════════════════════════════

/// A row that goes INLINE → B → B in one commit ACQUIRES B and releases
/// nothing.
///
/// The over-release canary for the `held` rule, and the one arm whose failure is
/// data loss rather than a leak. The row's ORIGINAL pre-image is inline — a
/// b.zvd row carries no `hash` column when its value is inline — so the object
/// did not hold B before this commit and the pre-flight `+1` bought a real, new
/// reference. Handing it back leaves `ref_count` at 0 for a hash a live row
/// still names, and the very next GC tick deletes `ext:meta` AND `ext:data`
/// under it.
///
/// **Fails without the b.yda round-3 correction.** A per-RECORD fold reads the
/// SECOND record's pre-image as `B` (what the first record of this same commit
/// wrote), declares `ext_held[B]`, and — with `before == 0` — takes the
/// `before > 0 or ext_held[hash]` arm and emits `surplus_acquire:B`. The pending
/// set then holds one member, the GC drains it, and `assert_blob_alive` below is
/// the assertion that fails.
///
/// Run on both row kinds. The time-sample variant is not a formality: the Lua
/// script's `TS` arm calls `ext_note_row` from its own branch, with its own
/// `old_hash` read, so a first-touch recording discipline correct in the `SF`
/// branch and wrong in the `TS` one is a live shape — and the time-sample row key
/// is derived from an `f64` rather than a name, which is a second way for two
/// records meant as one row to land on two keys and quietly turn this arm into a
/// two-row test that proves nothing.
async fn a_row_written_twice_with_one_new_hash_acquires_it_once(
    backend: UpdateLayerBackend,
    kind: RowKind,
) {
    let tv = live_edit::storage::test_support::TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-yda"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let layer = format!("yda-twice-new-{}-{}", kind.label(), backend_label(backend));
    let session_id = create_session(&tv, &layer).await;
    let oid = ls
        .create_layer(
            &session_id,
            &layer,
            "b-yda-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let b = ext_hash(0xB1);

    // ── Commit 1: the row exists, holding an INLINE value ───────────────────
    let inline = kind.inline_delta(vec![0x01, 0x02], 0);
    assert_fast_path_eligible(&inline, "the inline seed write");
    ls.update_layer_relayed(&session_id, &layer, oid, Some(inline), "yda-writer", 1)
        .await
        .expect("the inline seed commit must land");
    assert!(
        obj_ext_raw(&tv, oid).await.is_empty(),
        "fixture precondition: an inline row holds no external reference, so the index is empty \
         and `before` reads 0 for B below — which is what makes the `held` proof the only one in \
         play"
    );

    // ── Commit 2: the SAME row is written twice, both records naming B ──────
    //
    // One pre-flight `+1`, because `store_external_values` dedups per DISTINCT
    // hash per RPC. Getting that multiplicity wrong is the difference between
    // testing the rule and testing a fixture bug.
    let delta = kind.delta_writing_one_row_twice(Some(0xB1), Some(0xB1));
    assert_fast_path_eligible(&delta, "the doubly-written external row");
    pre_store(&ext, &[b]).await;
    assert_eq!(ref_count(&tv, b).await, Some(1));

    ls.update_layer_relayed(&session_id, &layer, oid, Some(delta), "yda-writer", 2)
        .await
        .expect("the doubly-written commit must land");

    assert!(
        pending_members(&tv).await.is_empty(),
        "NOTHING may be released. The row's ORIGINAL pre-image was inline, so the object did not \
         hold B and its `+1` bought a real reference; a `surplus_acquire:B` here hands back the \
         increment that paid for the value the row now holds. This is the per-RECORD fold defect, \
         and it is an OVER-release — the loss direction"
    );
    assert_eq!(
        release_reasons(&metrics),
        no_releases(),
        "and no `reason` cell may move either. The counter is read as well as the keyspace because \
         it is the only place the release REASON exists — the pending set above would look \
         identical for a `rewrite` and a `surplus_acquire`, and a build that enqueued one but \
         failed to SADD it would leave the set empty while ticking the counter"
    );
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xB1), 1)],
        "and the index records exactly ONE row holding B, not two: the two records are one row, \
         and a per-record `+1` would leave a count no retirement could ever reconcile"
    );
    assert_eq!(ref_count(&tv, b).await, Some(1));

    // The GC tick is what turns the over-release into an observable deletion.
    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(
        stats.processed, 0,
        "nothing was owed, so nothing is claimed"
    );
    assert_blob_alive(&tv, b, "the hash the doubly-written row acquired").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_field_a_row_written_twice_with_one_new_hash_acquires_it_once() {
    a_row_written_twice_with_one_new_hash_acquires_it_once(UpdateLayerBackend::Lua, RowKind::Field)
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_field_a_row_written_twice_with_one_new_hash_acquires_it_once() {
    a_row_written_twice_with_one_new_hash_acquires_it_once(UpdateLayerBackend::Txn, RowKind::Field)
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_ts_a_row_written_twice_with_one_new_hash_acquires_it_once() {
    a_row_written_twice_with_one_new_hash_acquires_it_once(
        UpdateLayerBackend::Lua,
        RowKind::TimeSample,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_ts_a_row_written_twice_with_one_new_hash_acquires_it_once() {
    a_row_written_twice_with_one_new_hash_acquires_it_once(
        UpdateLayerBackend::Txn,
        RowKind::TimeSample,
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 1b. the same, on a row that did not exist at all
// ═════════════════════════════════════════════════════════════════════════════

/// An ABSENT row written twice with B in one commit acquires B and releases
/// nothing.
///
/// The sibling of arm 1 with the pre-image removed entirely rather than merely
/// made inline. Kept as its own arm because the two reach the recording
/// discipline by different keyspace states — `HGET` on a row whose `hash` field
/// is absent versus `HGET` on a row key that does not exist — and a
/// transcription that special-cased one of them would leave the other green.
async fn an_absent_row_written_twice_with_one_new_hash_acquires_it_once(
    backend: UpdateLayerBackend,
) {
    let tv = live_edit::storage::test_support::TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-yda"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let layer = format!("yda-twice-absent-{}", backend_label(backend));
    let session_id = create_session(&tv, &layer).await;
    let oid = ls
        .create_layer(
            &session_id,
            &layer,
            "b-yda-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let b = ext_hash(0xB2);
    let delta = RowKind::Field.delta_writing_one_row_twice(Some(0xB2), Some(0xB2));
    assert_fast_path_eligible(&delta, "the doubly-written external field on a fresh row");
    pre_store(&ext, &[b]).await;

    ls.update_layer_relayed(&session_id, &layer, oid, Some(delta), "yda-writer", 1)
        .await
        .expect("the doubly-written commit must land");

    assert!(
        pending_members(&tv).await.is_empty(),
        "a row that did not exist has no pre-image at all, so nothing was held and the `+1` bought \
         a real reference"
    );
    assert_eq!(
        release_reasons(&metrics),
        no_releases(),
        "and no `reason` cell moves: the empty pending set above and these four zeros are two \
         independent readings of the same claim, and only the second can tell a release that was \
         never emitted from one that was emitted and lost on the way to the Set"
    );
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xB2), 1)],
        "one row holds B, however many records wrote it"
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 0);
    assert_blob_alive(&tv, b, "the hash the fresh doubly-written row acquired").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_an_absent_row_written_twice_with_one_new_hash_acquires_it_once() {
    an_absent_row_written_twice_with_one_new_hash_acquires_it_once(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_an_absent_row_written_twice_with_one_new_hash_acquires_it_once() {
    an_absent_row_written_twice_with_one_new_hash_acquires_it_once(UpdateLayerBackend::Txn).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 2. B → B on a SEEDED object — the `before > 0` control
// ═════════════════════════════════════════════════════════════════════════════

/// Rewriting a row with the hash it already holds, on an object whose index IS
/// seeded, hands the pre-flight `+1` back exactly once.
///
/// The control for arm 3. This case was already correct pre-b.yda — `before > 0`
/// sees it — so its value here is the difference it makes: arm 3 runs the
/// IDENTICAL commit against an object whose index has been removed, and without
/// this arm a build that never released anything at all would satisfy arm 3's
/// "the index is still absent" assertion just as well.
///
/// Parameterised over the row kind for the same reason arm 3 is: a control has
/// to be the control for the SAME commit its partner runs, or the difference the
/// pair isolates stops being the one variable it claims to be.
async fn a_same_hash_rewrite_hands_back_the_surplus_increment(
    backend: UpdateLayerBackend,
    kind: RowKind,
) {
    let tv = live_edit::storage::test_support::TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-yda"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let layer = format!(
        "yda-same-hash-seeded-{}-{}",
        kind.label(),
        backend_label(backend)
    );
    let session_id = create_session(&tv, &layer).await;
    let oid = ls
        .create_layer(
            &session_id,
            &layer,
            "b-yda-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let b = ext_hash(0xB3);

    let first = kind.external_delta(0xB3, 0);
    assert_fast_path_eligible(&first, "the acquiring write");
    pre_store(&ext, &[b]).await;
    ls.update_layer_relayed(&session_id, &layer, oid, Some(first), "yda-writer", 1)
        .await
        .expect("the acquiring commit must land");
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xB3), 1)],
        "fixture precondition: the index IS seeded, which is the whole difference from arm 3"
    );
    assert_eq!(ref_count(&tv, b).await, Some(1));

    let second = kind.external_delta(0xB3, 1);
    pre_store(&ext, &[b]).await;
    assert_eq!(
        ref_count(&tv, b).await,
        Some(2),
        "fixture precondition: the second RPC's pre-flight store paid a second `+1` for a \
         reference the object already held — that surplus increment is what the rule hands back"
    );
    let (seq, _) = ls
        .update_layer_relayed(&session_id, &layer, oid, Some(second), "yda-writer", 2)
        .await
        .expect("the same-hash rewrite must land");

    assert_eq!(
        pending_members(&tv).await,
        vec![commit_member(b, oid, seq)],
        "exactly one owed decrement, naming B, the committing object and the committing sequence"
    );
    assert_eq!(
        release_reasons(&metrics),
        one_release(REASON_SURPLUS),
        "and the one release must be spelled `surplus_acquire`, not `rewrite`. The member above \
         cannot tell them apart — it carries no reason — and both release exactly one reference, \
         so this counter is the entire difference between the two claims: `surplus_acquire` says \
         the object STILL holds B (which is why the index below keeps its entry), while `rewrite` \
         says it gave B up, and a build that emitted `rewrite` here would be describing a live \
         reference as retired"
    );
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xB3), 1)],
        "a net-zero row count must not rewrite the index"
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.deleted, 0, "one live object still references B");
    assert_eq!(
        ref_count(&tv, b).await,
        Some(1),
        "back to ONE, which is what `ref_count` counting live objects means"
    );
    assert_blob_alive(&tv, b, "the hash the object still holds").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_field_a_same_hash_rewrite_hands_back_the_surplus_increment() {
    a_same_hash_rewrite_hands_back_the_surplus_increment(UpdateLayerBackend::Lua, RowKind::Field)
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_field_a_same_hash_rewrite_hands_back_the_surplus_increment() {
    a_same_hash_rewrite_hands_back_the_surplus_increment(UpdateLayerBackend::Txn, RowKind::Field)
        .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_ts_a_same_hash_rewrite_hands_back_the_surplus_increment() {
    a_same_hash_rewrite_hands_back_the_surplus_increment(
        UpdateLayerBackend::Lua,
        RowKind::TimeSample,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_ts_a_same_hash_rewrite_hands_back_the_surplus_increment() {
    a_same_hash_rewrite_hands_back_the_surplus_increment(
        UpdateLayerBackend::Txn,
        RowKind::TimeSample,
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 3. B → B on an UNSEEDED object — residual (c)
// ═════════════════════════════════════════════════════════════════════════════

/// The same commit as arm 2, against an object carrying NO `obj:ext` index,
/// still hands the surplus increment back — and still does not seed the index.
///
/// The ticket's residual (c), reproduced exactly. `le:{oid}:obj:ext` is deleted
/// out of band after the acquiring commit, which is the shape every object
/// materialized before b.qpf is in permanently. The rewrite then nets
/// `delta == 0`, so the commit writes no index entry either, and both pre-b.yda
/// proofs are unavailable at once: the object leaks its pre-flight `+1` on every
/// commit for its whole life while retirement releases exactly one.
///
/// Two assertions carry the arm, and they are opposites:
///
/// * The release IS emitted, from `held` alone. **Fails pre-b.yda**: the rule
///   read `before > 0 && surplus == Release`, `before` was 0, and the pending
///   set was EMPTY here.
/// * The index is STILL absent afterwards. `held` changes the release arm only —
///   seeding an absolute count from a commit that saw one of the object's rows
///   would under-count every other row holding the hash, which is the count
///   clamp `KeySchema::obj_ext_index`'s `# Drift` section refuses. A build that
///   "helpfully" seeded here would pass the first assertion and fail this one.
///
/// Run on both row kinds. The row's own pre-image is the ONLY proof left once the
/// index is gone, and that pre-image is read by a different line of the commit
/// path per row kind — so a transcription that reads a field row's `hash` column
/// but not a time-sample row's would leave exactly this residual open on every
/// time-sample row in the fleet while the field arm stayed green.
async fn a_same_hash_rewrite_on_an_unseeded_object_hands_back_the_surplus(
    backend: UpdateLayerBackend,
    kind: RowKind,
) {
    let tv = live_edit::storage::test_support::TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-yda"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let layer = format!(
        "yda-same-hash-unseeded-{}-{}",
        kind.label(),
        backend_label(backend)
    );
    let session_id = create_session(&tv, &layer).await;
    let oid = ls
        .create_layer(
            &session_id,
            &layer,
            "b-yda-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let b = ext_hash(0xB4);

    let first = kind.external_delta(0xB4, 0);
    assert_fast_path_eligible(&first, "the acquiring write");
    pre_store(&ext, &[b]).await;
    ls.update_layer_relayed(&session_id, &layer, oid, Some(first), "yda-writer", 1)
        .await
        .expect("the acquiring commit must land");

    // ── Un-seed the index: the pre-b.qpf object shape ───────────────────────
    let index_key = tv.key_schema().obj_ext_index(&oid.to_string());
    let removed: u64 = tv
        .client()
        .pool
        .del(&index_key)
        .await
        .expect("DEL obj:ext must succeed");
    assert_eq!(
        removed, 1,
        "fixture precondition: the index must actually have existed and been removed — without \
         that this arm is a duplicate of the seeded control"
    );

    let second = kind.external_delta(0xB4, 1);
    pre_store(&ext, &[b]).await;
    assert_eq!(
        ref_count(&tv, b).await,
        Some(2),
        "fixture precondition: the second RPC paid its pre-flight `+1`. Pre-b.yda this 2 is where \
         the count STAYS — one release at retirement against two increments, for every commit the \
         object ever takes"
    );
    let (seq, _) = ls
        .update_layer_relayed(&session_id, &layer, oid, Some(second), "yda-writer", 2)
        .await
        .expect("the same-hash rewrite must land");

    assert_eq!(
        pending_members(&tv).await,
        vec![commit_member(b, oid, seq)],
        "the row's own pre-image is the only available proof that the object already held B — \
         `before` reads 0 because the index was never written, not because the object holds \
         nothing. Pre-b.yda this set was EMPTY and the increment leaked permanently"
    );
    assert_eq!(
        release_reasons(&metrics),
        one_release(REASON_SURPLUS),
        "spelled `surplus_acquire`, exactly as the seeded control spells it. This is the assertion \
         that makes the pair a pair: the two arms differ ONLY in whether `obj:ext` was there, so a \
         build that reached the release by some other route — reading the row as a `rewrite` of B \
         to B, say — would satisfy every other assertion in this arm while reporting the object as \
         having GIVEN UP the hash it still holds"
    );
    assert!(
        obj_ext_raw(&tv, oid).await.is_empty(),
        "and the index must still be ABSENT. The `held` rule changes the RELEASE arm only; the \
         index writes stay gated on `delta != 0`, and a same-hash rewrite nets zero. Seeding an \
         absolute count from a commit that saw only some of the object's rows would under-count \
         the rest"
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.deleted, 0, "the object still holds B");
    assert_eq!(
        ref_count(&tv, b).await,
        Some(1),
        "the surplus really is handed back, not merely enqueued: one live object, one reference"
    );
    assert_blob_alive(&tv, b, "the hash the unseeded object still holds").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_field_a_same_hash_rewrite_on_an_unseeded_object_hands_back_the_surplus() {
    a_same_hash_rewrite_on_an_unseeded_object_hands_back_the_surplus(
        UpdateLayerBackend::Lua,
        RowKind::Field,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_field_a_same_hash_rewrite_on_an_unseeded_object_hands_back_the_surplus() {
    a_same_hash_rewrite_on_an_unseeded_object_hands_back_the_surplus(
        UpdateLayerBackend::Txn,
        RowKind::Field,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_ts_a_same_hash_rewrite_on_an_unseeded_object_hands_back_the_surplus() {
    a_same_hash_rewrite_on_an_unseeded_object_hands_back_the_surplus(
        UpdateLayerBackend::Lua,
        RowKind::TimeSample,
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_ts_a_same_hash_rewrite_on_an_unseeded_object_hands_back_the_surplus() {
    a_same_hash_rewrite_on_an_unseeded_object_hands_back_the_surplus(
        UpdateLayerBackend::Txn,
        RowKind::TimeSample,
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// 4. A → A → C in one commit: the ORIGINAL pre-image, not an intermediate
// ═════════════════════════════════════════════════════════════════════════════

/// A row whose ORIGINAL pre-image is A and whose FINAL write is C releases A as
/// a `rewrite` — and nothing else — even though one of the commit's records
/// wrote A back.
///
/// The pairing under test is ORIGINAL against FINAL. A per-record fold sees
/// `A → A` on the first record, declares A `held`, and — since it also puts A in
/// the referenced set — emits `surplus_acquire:A` INSTEAD of `rewrite:A`. Both
/// spellings release one reference, and — this is the part it is easy to get
/// wrong — both spell the SAME pending member: `ExtRelease::member()` is
/// `{hash_hex}:{object_id}:{event}`, and the reason is not in it. So the whole
/// member assertion below does NOT distinguish them, and neither does anything
/// else in the keyspace. The reason reaches exactly one place,
/// `live_edit_ext_ref_released_total{reason}`, and the `release_reasons`
/// assertion is what pins it. It matters operationally (the `reason` cells are
/// read as a ratio) and structurally: `surplus_acquire` claims the object still
/// holds A, so a build that emits it would ALSO have to leave A in `obj:ext`, and
/// the index assertion below is the second, independent door on the same defect.
///
/// ── The reference this arm deliberately leaves standing ─────────────────────
///
/// A ends at `ref_count == 1` with no live object referencing it, and the blob
/// SURVIVES the GC tick. That is not an oversight: the commit's pre-flight store
/// pays a `+1` per DISTINCT hash the delta names, A included, and A folds out of
/// `referenced` entirely, so the increment is neither matched nor handed back.
/// Releasing it would need this fold to know whether some OTHER object holds A,
/// which is not knowable from one object's rows — it is
/// `storage::ext_reconcile`'s to drain. Asserted rather than tolerated so a
/// future build that closes it in the commit path has to come here and argue
/// for it, because the obvious closure is an over-release.
async fn a_row_whose_original_and_final_differ_releases_only_the_original(
    backend: UpdateLayerBackend,
) {
    let tv = live_edit::storage::test_support::TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-yda"));
    let ls = layer_store(&tv, backend, &metrics);
    let ext = ext_store(&tv);
    let layer = format!("yda-original-vs-final-{}", backend_label(backend));
    let session_id = create_session(&tv, &layer).await;
    let oid = ls
        .create_layer(
            &session_id,
            &layer,
            "b-yda-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    let a = ext_hash(0xA5);
    let c = ext_hash(0xC5);

    let first = RowKind::Field.external_delta(0xA5, 0);
    assert_fast_path_eligible(&first, "the acquiring write");
    pre_store(&ext, &[a]).await;
    ls.update_layer_relayed(&session_id, &layer, oid, Some(first), "yda-writer", 1)
        .await
        .expect("the acquiring commit must land");
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xA5), 1)]
    );
    assert_eq!(
        release_reasons(&metrics),
        no_releases(),
        "fixture precondition: the ACQUIRING commit released nothing, so every cell read after the \
         A-then-C commit below is that commit's own work rather than a running total"
    );

    // ── The doubly-written row: A (again) then C ────────────────────────────
    //
    // Both hashes are pre-stored, because `store_external_values` walks the
    // whole delta and pays one `+1` per distinct hash it names — including the
    // intermediate.
    let delta = RowKind::Field.delta_writing_one_row_twice(Some(0xA5), Some(0xC5));
    assert_fast_path_eligible(&delta, "the A-then-C doubly-written field");
    pre_store(&ext, &[a, c]).await;
    assert_eq!(
        (ref_count(&tv, a).await, ref_count(&tv, c).await),
        (Some(2), Some(1)),
        "fixture precondition: the commit's pre-flight paid for BOTH hashes it names"
    );

    let (seq, _) = ls
        .update_layer_relayed(&session_id, &layer, oid, Some(delta), "yda-writer", 2)
        .await
        .expect("the A-then-C commit must land");

    assert_eq!(
        pending_members(&tv).await,
        vec![commit_member(a, oid, seq)],
        "exactly ONE member, naming A, the committing object and the committing sequence. This \
         assertion pins the COUNT and the identity — not the reason, which is not in the member at \
         all; the next one is where the reason is pinned"
    );
    assert_eq!(
        release_reasons(&metrics),
        one_release(REASON_REWRITE),
        "and the reason must be `rewrite`. `surplus_acquire` is the per-record fold's answer — it \
         reads the first record's write back as the second record's pre-image, declares A held, \
         and hands the increment back as a surplus — and because the member above is \
         byte-identical under both spellings, this counter is the ONLY place the two answers \
         differ. `surplus_acquire` here would also be a lie about the keyspace: it claims the \
         object still holds A, which the `obj:ext` assertion below shows it does not"
    );
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(ext_hash_hex(0xC5), 1)],
        "A's field must be HDELed and C's set to 1. A surviving A entry would make the retirement \
         union release a reference this commit already released"
    );

    let stats = process_pending_gc(
        ext.as_ref(),
        None,
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 1, "one owed decrement, one claim");
    assert_eq!(
        stats.deleted, 0,
        "A's `ref_count` does NOT reach zero: the intermediate write's own pre-flight `+1` is \
         still standing"
    );
    assert_eq!(
        ref_count(&tv, a).await,
        Some(1),
        "the documented intermediate-hash leak, pinned. A hash written only as an intermediate \
         state of a doubly-written row folds out of `referenced` and out of `delta`, so its `+1` \
         is neither matched nor handed back here — `storage::ext_reconcile` drains it. Releasing \
         it from the commit path would need this fold to know whether some OTHER object holds A, \
         which one object's rows cannot say"
    );
    assert_blob_alive(
        &tv,
        a,
        "the intermediate hash whose surplus increment is still standing",
    )
    .await;
    assert_eq!(ref_count(&tv, c).await, Some(1));
    assert_blob_alive(&tv, c, "the hash the row ends holding").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_a_row_whose_original_and_final_differ_releases_only_the_original() {
    a_row_whose_original_and_final_differ_releases_only_the_original(UpdateLayerBackend::Lua).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_a_row_whose_original_and_final_differ_releases_only_the_original() {
    a_row_whose_original_and_final_differ_releases_only_the_original(UpdateLayerBackend::Txn).await;
}
