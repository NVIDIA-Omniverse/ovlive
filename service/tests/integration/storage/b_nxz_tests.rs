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

//! b.nxz — the txn backend's external-value pre-image read
//! (`LayerStore::read_ext_pre_image`) holds a bounded number of queued commands
//! in flight, whatever the commit's row count.
//!
//! The read sends one `HEXISTS <row> value` and one `HGET <row> hash` per
//! distinct written row, then one `HGET obj:ext <hash>` per external hash
//! those rows hold or receive. It used to send all of the row commands, with
//! one `HGETALL obj:ext`, in ONE data-pool pipeline: 2 × rows + 1 fred
//! commands queued on one connection at once, each holding its argument values
//! until its reply came back. It now sends pipelines of at most
//! `EXT_PRE_IMAGE_CHUNK_ROWS` rows, at most `EXT_PRE_IMAGE_CONCURRENCY` of
//! them in flight, and after them the count reads in pipelines and waves of
//! the same bound (b.26e Stage 2 replaced the whole-index `HGETALL`, whose
//! reply grew with the object's index, by those count reads).
//!
//! Nothing on the store tells the two shapes apart: both commit the same rows,
//! the same `obj:ext` and the same releases. The test therefore reads the shape
//! off `ExtPreImagePipelineObserver`, a test-only recorder of each pipeline the
//! read builds and of the most in flight at once, whose wave barrier also
//! holds each wave until all of its chunks are being polled together. It
//! pairs that with the store outcome, so a chunked read that answered wrongly
//! fails too.
//!
//! The pure plan (every row once, in order, within the bounds) and the
//! per-chunk decoder (a misaligned chunk reply refuses `Internal`) are unit
//! tests beside the functions in `src/storage/layer.rs`.
//!
//! b.26e Stage 2 adds the count phase's own tests:
//! [`fast_shaped_txn_rewrite_reads_only_its_own_hashes_counts_b26e`] — on an
//! object whose `obj:ext` holds 20,000 other hashes, one external rewrite
//! reads exactly its two hashes' counts, after a hold that covers the
//! replies, and the `obj:ext` replies are the same at 1,000 entries — and
//! [`fast_shaped_txn_count_phase_reads_its_rows_hashes_in_bounded_waves_b26e`]
//! — an inline rewrite reads nothing in `obj:ext`; more touched hashes than
//! one wave holds are read in bounded pipelines and waves after the rows; a
//! one-hash count phase decodes; a corrupt touched count reads as 0 and
//! ticks once, a corrupt untouched one is never read.
//!
//! # Two activation conditions, one of them silent
//!
//! Registered under `#[cfg(feature = "test-fault-injection")]`, the observer's
//! gate. Without `--features test-fault-injection` this file is compiled out
//! and the run reports green having covered none of it.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::HashesInterface;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::delta::MutationAllowance;
use live_edit::error::ServiceError;
use live_edit::grpc::upload_admission::{
    object_state_charge_for_test, object_state_rewrite_allowance_for_test,
};
use live_edit::storage::commit_charge::{CommitCharge, ObjectStateNeed};
use live_edit::storage::external_values::{ExtRelease, ValkeyExternalValueStore};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::{
    ExtPreImagePipelineObserver, ExtPreImagePipelineSend, EXT_PRE_IMAGE_CHUNK_ROWS,
    EXT_PRE_IMAGE_CONCURRENCY,
};
use live_edit::storage::lua::is_fast_path;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{OwnedDelta, SetField};

use crate::storage::b_qpf_tests::{
    create_session, empty_state, ext_hash, ext_hash_hex, ext_hash_vec, ext_ref_released, ext_store,
    layer_store, obj_ext_sorted, pending_members, pre_store, ref_count, EXT_SIZE, REASON_REWRITE,
    REASON_SURPLUS, SECTION_ID,
};
use crate::storage::snapshot_read_support::{monitor_fence, MonitorTap};

// ═════════════════════════════════════════════════════════════════════════════
// The fixture
// ═════════════════════════════════════════════════════════════════════════════

/// How many distinct rows the commit under test writes, **derived from the
/// production chunk size and concurrency, not restated beside them** (the
/// b.zvd chunk-span convention, `b_zvd_tests::CHUNK_SPAN_FIELD_COUNT`).
///
/// One full wave plus half a chunk, so the read takes two waves: a full one of
/// `EXT_PRE_IMAGE_CONCURRENCY` pipelines, then one short pipeline. At today's
/// values that is 4 × 512 + 256 = 2 304 rows, five pipelines. A fixture inside
/// one wave would leave the read's second wave exercised by nothing, and a
/// literal row count would collapse into one wave the moment either constant
/// was raised: green, and testing nothing.
const WAVE_SPAN_ROWS: usize =
    EXT_PRE_IMAGE_CONCURRENCY * EXT_PRE_IMAGE_CHUNK_ROWS + EXT_PRE_IMAGE_CHUNK_ROWS / 2;

/// The pipelines the read takes over [`WAVE_SPAN_ROWS`] rows.
const WAVE_SPAN_CHUNKS: usize = WAVE_SPAN_ROWS.div_ceil(EXT_PRE_IMAGE_CHUNK_ROWS);

// The fixture spans more than one wave, or the read's second wave would be
// covered by nothing. True by construction of [`WAVE_SPAN_ROWS`]; checked at
// compile time so a literal row count that broke it fails the build.
const _: () = assert!(WAVE_SPAN_CHUNKS > EXT_PRE_IMAGE_CONCURRENCY);

/// How long the commit under test may take before the test calls it hung.
///
/// A deadline, not a sleep: the correct commit returns in well under a second
/// against a local Valkey, and the whole test in about two. A read that awaits
/// a wave's chunks one at a time never gets past the wave barrier at all, so
/// any budget fails it; this one is far above the healthy run under CI load and
/// well inside nextest's 120 s hang-killer. That killer would report only a
/// bare timeout naming the test, not why it hung, and only after 120 s (plus a
/// retry); this budget fails with the cause.
const COMMIT_BUDGET: Duration = Duration::from_secs(30);

/// Decimal digit count of `n`, so [`ROW_NAME_WIDTH`] follows a derived row
/// count.
const fn decimal_width(mut n: usize) -> usize {
    let mut width = 1;
    while n >= 10 {
        n /= 10;
        width += 1;
    }
    width
}

/// The zero-padding width of [`row_name`]: wide enough for the greatest index.
/// Equal-length names keep the fixture's row order and its numeric order the
/// same, whatever a reader sorts by.
const ROW_NAME_WIDTH: usize = decimal_width(WAVE_SPAN_ROWS - 1);

/// The field name of row `i`.
fn row_name(i: usize) -> String {
    format!("f{i:0width$}", width = ROW_NAME_WIDTH)
}

/// The field the commit under test does not write. It holds [`KEPT_HASH`]
/// throughout, so a commit that miscounts that hash releases a reference a
/// live row still holds.
const KEEPER_FIELD: &str = "keeper";

/// The hash the keeper row holds, the hash every dual-shape row's stale `hash`
/// column names, and the hash the commit under test writes to every absent row.
const KEPT_HASH: u8 = 0xC0;

/// The hash the commit under test writes to every row that was inline before.
/// No row held it before, so it owes no release.
const NEW_HASH: u8 = 0xE0;

/// The first per-chunk hash: an external row in chunk `c` holds
/// `OLD_HASH_BASE + c`. One hash per chunk means a chunk the read lost or
/// misread leaves its own hash's count wrong, so the failure names the chunk.
const OLD_HASH_BASE: u8 = 0x10;

/// What a row of the commit under test held before that commit — what the
/// pre-image read must report about it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreImage {
    /// An external value: no `value` column, a `hash` column naming
    /// [`old_hash`]. Counted.
    External,
    /// An inline value. Not counted.
    Inline,
    /// A legacy pre-b.zvd dual-shape row: an inline `value` beside a stale
    /// `hash` naming [`KEPT_HASH`]. Not counted: the row holds no hash.
    DualShape,
    /// No row at all. Not counted.
    Absent,
}

/// Row `i`'s pre-image.
///
/// The first row of every chunk and the last row of every full chunk are
/// external, so external rows sit on both sides of every chunk boundary, the
/// wave boundary among them. The rest cycle through all four kinds, so the
/// short last chunk may end on any kind.
fn pre_image(i: usize) -> PreImage {
    let in_chunk = i % EXT_PRE_IMAGE_CHUNK_ROWS;
    if in_chunk == 0 || in_chunk == EXT_PRE_IMAGE_CHUNK_ROWS - 1 {
        return PreImage::External;
    }
    match i % 4 {
        0 => PreImage::External,
        1 => PreImage::Inline,
        2 => PreImage::DualShape,
        _ => PreImage::Absent,
    }
}

/// The per-chunk hash an external row `i` holds before the commit under test.
fn old_hash(i: usize) -> u8 {
    OLD_HASH_BASE + u8::try_from(i / EXT_PRE_IMAGE_CHUNK_ROWS).expect("chunk index fits a byte")
}

fn inline_field(key_name: String, set_order: u64) -> SetField {
    SetField {
        section_id: SECTION_ID,
        key_name,
        value_or_ext_hash: vec![0x01],
        ext_value_size: 0,
        ext_value_index: 0,
        set_order,
    }
}

fn external_field(key_name: String, hash: u8, set_order: u64) -> SetField {
    SetField {
        section_id: SECTION_ID,
        key_name,
        value_or_ext_hash: ext_hash_vec(hash),
        ext_value_size: EXT_SIZE,
        ext_value_index: 1,
        set_order,
    }
}

/// The seeding commit: every external row at its per-chunk hash, every inline
/// and dual-shape row inline (the stale `hash` is forged afterwards), the
/// keeper at [`KEPT_HASH`]. Absent rows are not written.
fn seed_delta() -> OwnedDelta {
    let mut set_fields = vec![external_field(KEEPER_FIELD.to_owned(), KEPT_HASH, 0)];
    for i in 0..WAVE_SPAN_ROWS {
        let order = 1 + i as u64;
        match pre_image(i) {
            PreImage::External => set_fields.push(external_field(row_name(i), old_hash(i), order)),
            PreImage::Inline | PreImage::DualShape => {
                set_fields.push(inline_field(row_name(i), order))
            }
            PreImage::Absent => {}
        }
    }
    OwnedDelta {
        set_fields,
        ..Default::default()
    }
}

/// The commit under test writes every one of the [`WAVE_SPAN_ROWS`] rows, in
/// row order, and leaves the keeper alone:
///
/// | Before | After | What the commit owes |
/// |---|---|---|
/// | external, per-chunk hash | inline | a `rewrite` release of each per-chunk hash, once its last row goes |
/// | inline | [`NEW_HASH`] | nothing: no row held it |
/// | dual shape | inline | nothing: the row held no hash |
/// | absent | [`KEPT_HASH`] | one `surplus_acquire` hand-back: the keeper already held it |
fn commit_delta() -> OwnedDelta {
    let set_order_base = 1 + WAVE_SPAN_ROWS as u64;
    let set_fields = (0..WAVE_SPAN_ROWS)
        .map(|i| {
            let order = set_order_base + i as u64;
            match pre_image(i) {
                PreImage::External | PreImage::DualShape => inline_field(row_name(i), order),
                PreImage::Inline => external_field(row_name(i), NEW_HASH, order),
                PreImage::Absent => external_field(row_name(i), KEPT_HASH, order),
            }
        })
        .collect();
    OwnedDelta {
        set_fields,
        ..Default::default()
    }
}

fn count(kind: PreImage) -> usize {
    (0..WAVE_SPAN_ROWS)
        .filter(|&i| pre_image(i) == kind)
        .count()
}

/// The `obj:ext` index the seeding commit leaves: each per-chunk hash at its
/// chunk's external-row count, and the keeper's one row on [`KEPT_HASH`].
/// The dual-shape rows' forged columns are not in it: they hold no hash.
fn seeded_index() -> Vec<(String, i64)> {
    let mut index: Vec<(String, i64)> = (0..WAVE_SPAN_CHUNKS)
        .map(|chunk| {
            let rows = (chunk * EXT_PRE_IMAGE_CHUNK_ROWS
                ..WAVE_SPAN_ROWS.min((chunk + 1) * EXT_PRE_IMAGE_CHUNK_ROWS))
                .filter(|&i| pre_image(i) == PreImage::External)
                .count();
            (
                ext_hash_hex(old_hash(chunk * EXT_PRE_IMAGE_CHUNK_ROWS)),
                rows as i64,
            )
        })
        .collect();
    index.push((ext_hash_hex(KEPT_HASH), 1));
    index.sort();
    index
}

/// The distinct hashes the commit under test touches, whose counts its
/// pre-image read reads: every per-chunk hash its external rows hold before
/// (one per chunk), plus the two it writes, [`KEPT_HASH`] and [`NEW_HASH`].
/// A dual-shape row's stale `hash` is not one: the row holds no hash.
const TOUCHED_HASHES: usize = WAVE_SPAN_CHUNKS + 2;

/// Every distinct hash the seeding commit references.
fn seeded_hashes() -> Vec<[u8; 32]> {
    let mut hashes: Vec<[u8; 32]> = (0..WAVE_SPAN_CHUNKS)
        .map(|chunk| ext_hash(old_hash(chunk * EXT_PRE_IMAGE_CHUNK_ROWS)))
        .collect();
    hashes.push(ext_hash(KEPT_HASH));
    hashes
}

/// The observer's records, split into one slice per `read_ext_pre_image`
/// call. A call's first pipeline is chunk 0, so each chunk 0 starts a call. A
/// retried attempt reads again and shows up as a second call rather than as
/// extra rows of the first.
fn calls(sent: &[ExtPreImagePipelineSend]) -> Vec<&[ExtPreImagePipelineSend]> {
    let starts: Vec<usize> = sent
        .iter()
        .enumerate()
        .filter(|(_, send)| send.chunk == 0)
        .map(|(i, _)| i)
        .collect();
    starts
        .iter()
        .zip(starts.iter().skip(1).chain(std::iter::once(&sent.len())))
        .map(|(&start, &end)| &sent[start..end])
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// The test
// ═════════════════════════════════════════════════════════════════════════════

/// A fast-shaped `txn` commit whose rows span two waves of the pre-image read
/// sends no pipeline of more than 2 × `EXT_PRE_IMAGE_CHUNK_ROWS` row commands,
/// builds waves of exactly `EXT_PRE_IMAGE_CONCURRENCY` chunks at their widest,
/// polls each wave's chunks concurrently, and still commits exactly the
/// `obj:ext` and releases its rows' pre-images owe.
///
/// **The bound.** Before b.nxz the read sent the whole commit's rows as one
/// pipeline: here 2 × [`WAVE_SPAN_ROWS`] row commands plus an `HGETALL` of the
/// index, which fails the per-pipeline assertion. The count reads that replaced
/// that `HGETALL` follow the row pipelines in their own (here one) pipeline.
///
/// **The wave width, asserted exactly.** The observer counts a chunk in flight
/// from the moment its future is built until that future finishes. The read
/// builds a whole wave's futures before it awaits any of them, and builds the
/// next wave only after the last one finished, so the peak is the width of the
/// widest wave, whatever the scheduling. Here that is a full wave of
/// `EXT_PRE_IMAGE_CONCURRENCY`. Building every chunk as one wave would peak at
/// all [`WAVE_SPAN_CHUNKS`] of them, and a narrower wave would peak lower. The
/// count says nothing about whether a wave's chunks then RUN together: a read
/// that built a wave and awaited its futures one at a time would still peak at
/// the wave's width. Equality holds only while one read at a time reports to
/// the observer, since every read on the store adds to one counter. So this
/// test sends one commit at a time through the observed store, and seeds
/// through a store without the observer. A retried attempt would read again
/// only after the previous one finished.
///
/// **Concurrent dispatch, proved by the wave barrier.** The observer is built
/// `with_wave_barrier`: every chunk future of a wave waits at a barrier sized
/// to that wave's chunk count before it queues anything. So no chunk of a wave
/// proceeds until all of them are being polled at once. A read that awaits a
/// wave's chunks one at a time never polls the second while the first waits,
/// and the commit hangs; [`COMMIT_BUDGET`] turns that into a failure naming the
/// cause. The fixture's first wave has `EXT_PRE_IMAGE_CONCURRENCY` (more than
/// one) chunks, so the barrier has something to hold.
///
/// **The answer.** The rows mix every pre-image kind (see [`pre_image`]), with
/// external rows on both sides of every chunk boundary, and each chunk's
/// external rows hold a hash of their own. So a chunk whose frames were lost,
/// skipped or paired with another chunk's rows leaves its own hash in
/// `obj:ext` or its release missing; a dual-shape row read as holding its
/// stale `hash` drops [`KEPT_HASH`]'s count below the keeper's live row; and
/// a count read decoded from anywhere but its own frame loses the keeper's
/// `surplus_acquire` hand-back, which depends on the index saying the object
/// already held the hash.
///
/// Slow for a storage test (two commits of a few thousand rows each, well
/// under a second against a local Valkey): the row count is what the test is
/// about, and it cannot be smaller than one wave.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_pre_image_read_spanning_two_waves_stays_within_its_in_flight_bound() {
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-nxz"));
    let seed_ls = layer_store(&tv, UpdateLayerBackend::Txn, &metrics);
    let observer = Arc::new(ExtPreImagePipelineObserver::with_wave_barrier());
    let ls = layer_store(&tv, UpdateLayerBackend::Txn, &metrics)
        .with_ext_pre_image_pipeline_observer_for_test(observer.clone());
    let ext = ext_store(&tv);
    let layer = "nxz-wave-span";
    let session_id = create_session(&tv, layer).await;
    let oid = seed_ls
        .create_layer(
            &session_id,
            layer,
            "b-nxz-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");

    // ── Seed every pre-image kind (on a store without the observer) ─────────
    let seed = seed_delta();
    assert!(
        is_fast_path(&Some(seed.clone())),
        "the seed must be fast-shaped"
    );
    pre_store(&ext, &seeded_hashes()).await;
    seed_ls
        .update_layer_relayed(&session_id, layer, oid, Some(seed), "nxz-writer", 1)
        .await
        .expect("the seeding commit must land");

    // Forge each dual-shape row: merge a `hash` column naming `KEPT_HASH` onto
    // its inline row, the shape an inline → external rewrite left before b.zvd.
    let oid_str = oid.to_string();
    let sid = SECTION_ID.to_string();
    let kept_hex = ext_hash_hex(KEPT_HASH);
    let size = EXT_SIZE.to_string();
    for i in (0..WAVE_SPAN_ROWS).filter(|&i| pre_image(i) == PreImage::DualShape) {
        let row = tv.key_schema().obj_field(&oid_str, &sid, &row_name(i));
        let _: () = tv
            .client()
            .pool
            .hset(&row, [("hash", kept_hex.as_str()), ("size", size.as_str())])
            .await
            .expect("forging a dual-shape row must succeed");
    }
    let first_dual = (0..WAVE_SPAN_ROWS)
        .find(|&i| pre_image(i) == PreImage::DualShape)
        .expect("the fixture has dual-shape rows");
    let first_dual_row = tv
        .key_schema()
        .obj_field(&oid_str, &sid, &row_name(first_dual));
    let (has_value, has_hash): (bool, bool) = (
        tv.client()
            .pool
            .hexists(&first_dual_row, "value")
            .await
            .expect("HEXISTS must succeed"),
        tv.client()
            .pool
            .hexists(&first_dual_row, "hash")
            .await
            .expect("HEXISTS must succeed"),
    );
    assert!(
        has_value && has_hash,
        "fixture: a dual-shape row carries both an inline `value` and a stale `hash`"
    );
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        seeded_index(),
        "fixture: the index counts each chunk's external rows under that chunk's hash and the \
         keeper under KEPT_HASH, and nothing for the dual-shape rows"
    );
    assert!(
        pending_members(&tv).await.is_empty(),
        "fixture: the seeding commit owes nothing"
    );

    // ── The commit under test ───────────────────────────────────────────────
    let delta = commit_delta();
    assert!(
        is_fast_path(&Some(delta.clone())),
        "the commit must be fast-shaped, or the txn backend takes the snapshot branch and the \
         pre-image read never runs"
    );
    let rewrite_before = ext_ref_released(&metrics, REASON_REWRITE);
    let surplus_before = ext_ref_released(&metrics, REASON_SURPLUS);
    pre_store(&ext, &[ext_hash(NEW_HASH), ext_hash(KEPT_HASH)]).await;
    let (seq, _) = tokio::time::timeout(
        COMMIT_BUDGET,
        ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "nxz-writer", 2),
    )
    .await
    .unwrap_or_else(|_| {
        panic!(
            "pre-image waves must be polled concurrently: the commit did not finish within \
             {COMMIT_BUDGET:?}. The wave barrier holds every chunk of a wave until all of them \
             are being polled at once, so a read that awaits a wave's chunks one at a time hangs \
             here"
        )
    })
    .expect("the wave-spanning commit must land");

    // ── The read's shape ────────────────────────────────────────────────────
    let sent = observer.sent();
    let max_row_commands = 2 * EXT_PRE_IMAGE_CHUNK_ROWS;
    for send in &sent {
        assert!(
            send.row_commands <= max_row_commands,
            "pipeline {send:?} queued {} row commands, above 2 x EXT_PRE_IMAGE_CHUNK_ROWS = \
             {max_row_commands}. One pipeline of every row's commands is the unbounded burst \
             b.nxz removed",
            send.row_commands
        );
    }
    assert_eq!(
        observer.peak_in_flight(),
        EXT_PRE_IMAGE_CONCURRENCY,
        "the read's widest wave must hold exactly EXT_PRE_IMAGE_CONCURRENCY chunks. Wider is an \
         unbounded fan-out; narrower is a read with more waves than the probe beside it, which \
         adds round trips inside the WATCH"
    );

    let calls = calls(&sent);
    assert!(
        !calls.is_empty(),
        "the observer recorded no read: the commit took a path without the txn pre-image read, \
         and every bound above held vacuously"
    );
    for call in &calls {
        assert_eq!(
            call.iter().map(|send| send.row_commands).sum::<usize>(),
            2 * WAVE_SPAN_ROWS,
            "each read must queue one HEXISTS and one HGET per distinct row, no more and no \
             fewer: {call:?}"
        );
        // Phase 1 is the rows' columns; phase 2, after it, the touched hashes'
        // counts (b.26e Stage 2: one `HGET obj:ext <hash>` per hash, no
        // whole-index `HGETALL`). No pipeline mixes the two.
        let (rows, counts): (Vec<&ExtPreImagePipelineSend>, Vec<&ExtPreImagePipelineSend>) =
            call.iter().partition(|send| send.count_reads == 0);
        assert!(
            counts.iter().all(|send| send.row_commands == 0),
            "a count-read pipeline carries no row command: {call:?}"
        );
        assert_eq!(
            rows.iter().map(|send| send.chunk).collect::<Vec<_>>(),
            (0..WAVE_SPAN_CHUNKS).collect::<Vec<_>>(),
            "each read must send its {WAVE_SPAN_CHUNKS} row pipelines once each, in chunk order"
        );
        assert_eq!(
            rows.iter().map(|send| send.wave).max(),
            Some(1),
            "the read must have taken a second row wave, or that wave's handling went untested: \
             {call:?}"
        );
        assert_eq!(
            counts.iter().map(|send| send.count_reads).sum::<usize>(),
            TOUCHED_HASHES,
            "each read must read exactly one count per hash its rows hold or receive — every \
             per-chunk hash, KEPT_HASH and NEW_HASH — and no whole index: {call:?}"
        );
        assert!(
            counts.iter().all(|send| {
                send.chunk >= WAVE_SPAN_CHUNKS && rows.iter().all(|row| row.wave < send.wave)
            }),
            "the count reads run after every row pipeline, in a later wave, their chunks \
             numbered on from phase 1's: {call:?}"
        );
    }

    // ── The read's answer, as committed ─────────────────────────────────────
    let mut expected_index = vec![
        (ext_hash_hex(KEPT_HASH), 1 + count(PreImage::Absent) as i64),
        (ext_hash_hex(NEW_HASH), count(PreImage::Inline) as i64),
    ];
    expected_index.sort();
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        expected_index,
        "after the commit the object holds KEPT_HASH on the keeper and every once-absent row, \
         NEW_HASH on every once-inline row, and no per-chunk hash: every external row was \
         rewritten inline. A per-chunk hash left here is a chunk the read lost or misread; \
         KEPT_HASH below 1 + absent rows is a dual-shape row read as holding its stale hash"
    );

    let mut expected_pending: Vec<String> = (0..WAVE_SPAN_CHUNKS)
        .map(|chunk| ExtRelease::commit(ext_hash(OLD_HASH_BASE + chunk as u8), oid, seq).member())
        .collect();
    expected_pending.push(ExtRelease::commit(ext_hash(KEPT_HASH), oid, seq).member());
    expected_pending.sort();
    assert_eq!(
        pending_members(&tv).await,
        expected_pending,
        "the commit owes one decrement per per-chunk hash it superseded, and one hand-back of \
         KEPT_HASH's pre-flight increment, which the keeper already held"
    );
    assert_eq!(
        ext_ref_released(&metrics, REASON_REWRITE),
        rewrite_before + WAVE_SPAN_CHUNKS as u64,
        "one rewrite release per per-chunk hash"
    );
    assert_eq!(
        ext_ref_released(&metrics, REASON_SURPLUS),
        surplus_before + 1,
        "one surplus_acquire hand-back, of KEPT_HASH"
    );
    assert_eq!(
        ref_count(&tv, ext_hash(KEPT_HASH)).await,
        Some(2),
        "KEPT_HASH: the seed's +1 and the commit's +1, its hand-back still pending"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// b.26e Stage 2: the count phase reads only the
// commit's own hashes
// ═════════════════════════════════════════════════════════════════════════════

/// A distinct content hash per `i`, beyond the 256 [`ext_hash`] spells.
fn numbered_hash(i: u32) -> [u8; 32] {
    let mut hash = [0x77u8; 32];
    hash[..4].copy_from_slice(&i.to_be_bytes());
    hash
}

/// An external field `key_name` holding `hash`.
fn external_field_of(key_name: String, hash: [u8; 32], set_order: u64) -> SetField {
    SetField {
        section_id: SECTION_ID,
        key_name,
        value_or_ext_hash: hash.to_vec(),
        ext_value_size: EXT_SIZE,
        ext_value_index: 1,
        set_order,
    }
}

/// [`pre_store`] for many hashes, a batch per call.
async fn pre_store_many(ext: &ValkeyExternalValueStore, hashes: &[[u8; 32]]) {
    use live_edit::storage::external_values::ExternalValueStore;
    for batch in hashes.chunks(512) {
        let pairs: Vec<(&[u8; 32], &[u8])> = batch
            .iter()
            .map(|hash| (hash, b"b-26e-external-payload".as_slice()))
            .collect();
        ext.store_values(&pairs)
            .await
            .expect("the pre-flight store_values must succeed");
    }
}

/// The non-script commands on `key` the `MONITOR` feed saw, from `tap`'s
/// start up to a fence issued now.
async fn monitored_on(tv: &TestValkey, tap: &mut MonitorTap, key: &str) -> Vec<Vec<String>> {
    use fred::interfaces::KeysInterface;
    let fence = monitor_fence(tv.prefix());
    let _: i64 = tv
        .client()
        .pool
        .exists(&fence)
        .await
        .expect("EXISTS the fence");
    tap.drain_until(&fence)
        .await
        .into_iter()
        .filter(|c| !c.issued_by_script() && c.args.get(1).map(String::as_str) == Some(key))
        .map(|c| c.args)
        .collect()
}

/// The `obj:ext` reads among `commands` (each its whole argument list).
fn index_reads(commands: &[Vec<String>]) -> Vec<Vec<String>> {
    const READS: [&str; 7] = [
        "HGET", "HGETALL", "HMGET", "HSCAN", "HKEYS", "HVALS", "HLEN",
    ];
    commands
        .iter()
        .filter(|args| READS.contains(&args[0].to_ascii_uppercase().as_str()))
        .cloned()
        .collect()
}

/// One hold a [`HoldLog`]ged charge took: its target, and how many pre-image
/// pipelines the observer had recorded when it was taken.
type Hold = (u64, usize);

/// A commit charge that never refuses, prices as production does, and logs
/// every hold beside the pre-image observer's pipeline count at that moment —
/// one shared order of holds and pipeline sends.
struct HoldLog {
    observer: Arc<ExtPreImagePipelineObserver>,
    held: u64,
    holds: Vec<Hold>,
}

#[async_trait::async_trait]
impl CommitCharge for HoldLog {
    fn held(&self) -> u64 {
        self.held
    }

    fn target(&self, need: &ObjectStateNeed) -> u64 {
        object_state_charge_for_test(need)
    }

    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
        object_state_rewrite_allowance_for_test(self.held, need)
    }

    fn try_hold(&mut self, target: u64) -> bool {
        self.holds.push((target, self.observer.sent().len()));
        self.held = target;
        true
    }

    async fn hold(&mut self, target: u64) -> Result<(), ServiceError> {
        self.holds.push((target, self.observer.sent().len()));
        self.held = target;
        Ok(())
    }
}

/// What one external rewrite on an object with `index_entries` other hashes
/// in its `obj:ext` showed: the holds, the pre-image pipelines, the `obj:ext`
/// commands and the bytes the read's `obj:ext` reads answer (replayed after
/// the commit, as the reply strings' lengths).
struct ExtRewriteCase {
    holds: Vec<Hold>,
    sent: Vec<ExtPreImagePipelineSend>,
    index_commands: Vec<Vec<String>>,
    index_reply_bytes: u64,
    old_hex: String,
    new_hex: String,
}

/// The external-rewrite fixture, at `index_entries`: one external field
/// holding `OLD`, its object's `obj:ext` seeded with `index_entries` other
/// hashes (written directly rather than by real commits), and a fast-shaped
/// `txn` commit rewriting the field to `NEW`, under a [`HoldLog`].
async fn ext_rewrite_case(index_entries: u32) -> ExtRewriteCase {
    use crate::storage::snapshot_read_support::test_valkey_addr;
    const OLD: u32 = u32::MAX;
    const NEW: u32 = u32::MAX - 1;
    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-26e-ext-counts"));
    let seed_ls = layer_store(&tv, UpdateLayerBackend::Txn, &metrics);
    let observer = Arc::new(ExtPreImagePipelineObserver::default());
    let ls = layer_store(&tv, UpdateLayerBackend::Txn, &metrics)
        .with_ext_pre_image_pipeline_observer_for_test(observer.clone());
    let ext = ext_store(&tv);
    let layer = "b26e-ext-counts";
    let session_id = create_session(&tv, layer).await;
    let oid = seed_ls
        .create_layer(
            &session_id,
            layer,
            "b-26e-user",
            &empty_state(),
            ext.clone(),
        )
        .await
        .expect("create_layer must succeed");
    let (old, new) = (numbered_hash(OLD), numbered_hash(NEW));
    pre_store_many(&ext, &[old]).await;
    seed_ls
        .update_layer_relayed(
            &session_id,
            layer,
            oid,
            Some(OwnedDelta {
                set_fields: vec![external_field_of("f".to_owned(), old, 1)],
                ..Default::default()
            }),
            "b26e-writer",
            1,
        )
        .await
        .expect("the seeding commit must land");
    let index_key = tv.key_schema().obj_ext_index(&oid.to_string());
    let seeded: Vec<(String, String)> = (0..index_entries)
        .map(|i| (sha256_to_hex(&numbered_hash(i)), "1".to_owned()))
        .collect();
    if !seeded.is_empty() {
        let _: () = tv
            .client()
            .pool
            .hset(&index_key, seeded)
            .await
            .expect("seeding obj:ext must succeed");
    }

    let delta = OwnedDelta {
        set_fields: vec![external_field_of("f".to_owned(), new, 2)],
        ..Default::default()
    };
    assert!(is_fast_path(&Some(delta.clone())), "fixture: fast-shaped");
    pre_store_many(&ext, &[new]).await;
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let mut charge = HoldLog {
        observer: Arc::clone(&observer),
        held: 0,
        holds: Vec::new(),
    };
    let delta = Some(delta);
    let relay = live_edit::storage::test_support::relay::relay_for(&delta);
    let (result, _summary) = tokio::time::timeout(
        COMMIT_BUDGET,
        ls.update_layer_with_summary(
            &session_id,
            layer,
            oid,
            Arc::new(delta),
            relay,
            "b26e-writer",
            2,
            None,
            &mut charge,
        ),
    )
    .await
    .expect("the commit finishes");
    result.expect("the rewrite commits");
    let index_commands = monitored_on(&tv, &mut tap, &index_key).await;

    // Replay the read's `obj:ext` reads for the size of what they answer.
    let mut index_reply_bytes = 0u64;
    for args in index_reads(&index_commands) {
        match args[0].to_ascii_uppercase().as_str() {
            "HGET" => {
                let reply: Option<String> = tv
                    .client()
                    .pool
                    .hget(&index_key, args[2].as_str())
                    .await
                    .expect("HGET replays");
                index_reply_bytes += reply.map_or(0, |r| r.len() as u64);
            }
            "HGETALL" => {
                let reply: std::collections::HashMap<String, String> = tv
                    .client()
                    .pool
                    .hgetall(&index_key)
                    .await
                    .expect("HGETALL replays");
                index_reply_bytes += reply
                    .iter()
                    .map(|(k, v)| (k.len() + v.len()) as u64)
                    .sum::<u64>();
            }
            other => panic!("the pre-image read issued {other} on obj:ext: {args:?}"),
        }
    }
    ExtRewriteCase {
        holds: charge.holds,
        sent: observer.sent(),
        index_commands,
        index_reply_bytes,
        old_hex: sha256_to_hex(&old),
        new_hex: sha256_to_hex(&new),
    }
}

/// The largest reply one pre-image command can answer, from the writers'
/// formats: a row's `hash` column (64 hex characters; `HEXISTS` answers an
/// integer) and a count (at most 19 decimal digits).
fn pre_image_reply_bound(send: &ExtPreImagePipelineSend) -> u64 {
    send.row_commands as u64 * 64 + send.count_reads as u64 * 19
}

/// A fast-shaped `txn` commit that
/// rewrites one external field to a new external value, on an object whose
/// `obj:ext` holds 20,000 other hashes.
///
/// - (c) The read issues no whole-index `HGETALL obj:ext` and exactly two
///   count reads, `HGET obj:ext <hash>` of the old and the new hash.
/// - (a) Every pre-image pipeline is sent after a hold whose object-state
///   target covers every reply the read's pipelines can bring (their
///   per-command bounds) and what its `obj:ext` reads answered, and no hold
///   after the first send shrinks below the hold in force at it.
/// - (b) The bytes the read's `obj:ext` reads answer are the same at 1,000
///   and at 20,000 index entries.
///
/// Before R9 the read sent one `HGETALL obj:ext`: ≥ 20,000 × 65 B ≈ 1.3 MB
/// at 20,000 entries against a hold of about 66 KB, growing with the index.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_shaped_txn_rewrite_reads_only_its_own_hashes_counts_b26e() {
    let large = ext_rewrite_case(20_000).await;
    let small = ext_rewrite_case(1_000).await;
    // Every claim is checked, and the violations reported together.
    let mut violations: Vec<String> = Vec::new();

    // (a)
    let replies: u64 = large.sent.iter().map(pre_image_reply_bound).sum();
    let first_send_hold = large
        .holds
        .iter()
        .rev()
        .find(|(_, sent)| *sent == 0)
        .map(|(target, _)| *target)
        .unwrap_or_else(|| panic!("a hold precedes the read: {:?}", large.holds));
    if first_send_hold < replies.max(large.index_reply_bytes) {
        violations.push(format!(
            "(a) the hold in force when the read is sent ({first_send_hold} B) does not cover \
             its replies ({replies} B bound, {} B of obj:ext reads)",
            large.index_reply_bytes
        ));
    }
    if large
        .holds
        .iter()
        .any(|(target, sent)| *sent > 0 && *target < first_send_hold)
    {
        violations.push(format!(
            "(a) a hold shrank while the read was in flight: {:?}",
            large.holds
        ));
    }

    // (b)
    if small.index_reply_bytes != large.index_reply_bytes {
        violations.push(format!(
            "(b) the read's obj:ext replies grow with the index: {} B at 1,000 entries, {} B at \
             20,000",
            small.index_reply_bytes, large.index_reply_bytes
        ));
    }

    // (c)
    let reads = index_reads(&large.index_commands);
    let mut counted: Vec<&str> = reads
        .iter()
        .filter(|args| args[0].eq_ignore_ascii_case("HGET"))
        .map(|args| args[2].as_str())
        .collect();
    counted.sort_unstable();
    let mut want = vec![large.old_hex.as_str(), large.new_hex.as_str()];
    want.sort_unstable();
    if reads.len() != 2 || counted != want {
        violations.push(format!(
            "(c) the read's obj:ext reads are not exactly the old and the new hash's counts: \
             {reads:?}"
        ));
    }
    if large.sent.iter().map(|s| s.count_reads).sum::<usize>() != 2 {
        violations.push(format!(
            "(c) the observer saw other than two count reads: {:?}",
            large.sent
        ));
    }
    assert!(violations.is_empty(), "{violations:#?}");
}

/// The count phase's shape and parity, on the commit's rows.
///
/// - An inline-over-inline commit reads and writes nothing in `obj:ext`
///   (no hash command on it; the keyspace probe may still type-check it)
///   and sends no count pipeline.
/// - A commit rewriting [`WAVE_SPAN_ROWS`] external rows each to a hash of
///   its own touches twice that many hashes: its count pipelines carry at
///   most 2 × `EXT_PRE_IMAGE_CHUNK_ROWS` reads each, at most
///   `EXT_PRE_IMAGE_CONCURRENCY` in flight, in waves after every row wave,
///   and the index ends with exactly the new hashes.
/// - A one-hash count phase decodes its one reply: a new row taking a hash
///   the object already holds hands the pre-flight increment back
///   (`surplus_acquire`) and counts 2.
/// - A non-integer stored count of a touched hash reads as 0 and ticks
///   `ext_ref_undecodable_total{site="ext_index_field"}` once; a corrupt
///   untouched field is neither read nor ticked, and is left as it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fast_shaped_txn_count_phase_reads_its_rows_hashes_in_bounded_waves_b26e() {
    use crate::storage::snapshot_read_support::test_valkey_addr;
    use live_edit::observability::metrics::test_util::{render, sample_value};

    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-26e-count-phase"));
    let seed_ls = layer_store(&tv, UpdateLayerBackend::Txn, &metrics);
    let ext = ext_store(&tv);
    let undecodable = || {
        sample_value(
            &render(&metrics),
            "live_edit_ext_ref_undecodable_total",
            &[("site", "ext_index_field")],
        )
        .unwrap_or(0.0) as u64
    };
    let new_object = |name: &'static str| {
        let (seed_ls, ext, tv) = (&seed_ls, &ext, &tv);
        async move {
            let session_id = create_session(tv, name).await;
            let oid = seed_ls
                .create_layer(&session_id, name, "b-26e-user", &empty_state(), ext.clone())
                .await
                .expect("create_layer must succeed");
            (session_id, oid)
        }
    };
    let observed = || {
        let observer = Arc::new(ExtPreImagePipelineObserver::default());
        let ls = layer_store(&tv, UpdateLayerBackend::Txn, &metrics)
            .with_ext_pre_image_pipeline_observer_for_test(observer.clone());
        (ls, observer)
    };

    // ── Inline over inline: no obj:ext command ──────────────────────────────
    let (session_id, oid) = new_object("b26e-inline").await;
    seed_ls
        .update_layer_relayed(
            &session_id,
            "b26e-inline",
            oid,
            Some(OwnedDelta {
                set_fields: vec![inline_field("f".to_owned(), 1)],
                ..Default::default()
            }),
            "b26e-writer",
            1,
        )
        .await
        .expect("the inline seed lands");
    let (ls, observer) = observed();
    let index_key = tv.key_schema().obj_ext_index(&oid.to_string());
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    ls.update_layer_relayed(
        &session_id,
        "b26e-inline",
        oid,
        Some(OwnedDelta {
            set_fields: vec![inline_field("f".to_owned(), 2)],
            ..Default::default()
        }),
        "b26e-writer",
        2,
    )
    .await
    .expect("the inline rewrite lands");
    let on_index = monitored_on(&tv, &mut tap, &index_key).await;
    assert!(
        on_index
            .iter()
            .all(|args| !args[0].to_ascii_uppercase().starts_with('H')),
        "an inline-over-inline commit reads and writes nothing in obj:ext: {on_index:?}"
    );
    assert!(
        observer.sent().iter().all(|s| s.count_reads == 0),
        "and sends no count pipeline: {:?}",
        observer.sent()
    );

    // ── Over two waves of count reads ───────────────────────────────────────
    let (session_id, oid) = new_object("b26e-many-hashes").await;
    let rows = WAVE_SPAN_ROWS as u32;
    let olds: Vec<[u8; 32]> = (0..rows).map(numbered_hash).collect();
    let news: Vec<[u8; 32]> = (rows..2 * rows).map(numbered_hash).collect();
    pre_store_many(&ext, &olds).await;
    seed_ls
        .update_layer_relayed(
            &session_id,
            "b26e-many-hashes",
            oid,
            Some(OwnedDelta {
                set_fields: olds
                    .iter()
                    .enumerate()
                    .map(|(i, h)| external_field_of(row_name(i), *h, 1 + i as u64))
                    .collect(),
                ..Default::default()
            }),
            "b26e-writer",
            1,
        )
        .await
        .expect("the external seed lands");
    pre_store_many(&ext, &news).await;
    let (ls, observer) = observed();
    let order = 1 + WAVE_SPAN_ROWS as u64;
    ls.update_layer_relayed(
        &session_id,
        "b26e-many-hashes",
        oid,
        Some(OwnedDelta {
            set_fields: news
                .iter()
                .enumerate()
                .map(|(i, h)| external_field_of(row_name(i), *h, order + i as u64))
                .collect(),
            ..Default::default()
        }),
        "b26e-writer",
        2,
    )
    .await
    .expect("the many-hash rewrite lands");
    let sent = observer.sent();
    let (row_sends, count_sends): (Vec<&ExtPreImagePipelineSend>, Vec<&ExtPreImagePipelineSend>) =
        sent.iter().partition(|send| send.count_reads == 0);
    assert_eq!(
        count_sends.iter().map(|s| s.count_reads).sum::<usize>(),
        2 * WAVE_SPAN_ROWS,
        "one count read per touched hash, old and new"
    );
    assert!(
        count_sends
            .iter()
            .all(|s| s.count_reads <= 2 * EXT_PRE_IMAGE_CHUNK_ROWS && s.row_commands == 0),
        "each count pipeline carries at most 2 × EXT_PRE_IMAGE_CHUNK_ROWS reads: {count_sends:?}"
    );
    let count_waves: std::collections::BTreeSet<usize> =
        count_sends.iter().map(|s| s.wave).collect();
    assert!(
        count_waves.len() >= 2,
        "fixture: the count phase spans two waves: {count_sends:?}"
    );
    for wave in &count_waves {
        assert!(
            count_sends.iter().filter(|s| s.wave == *wave).count() <= EXT_PRE_IMAGE_CONCURRENCY,
            "at most EXT_PRE_IMAGE_CONCURRENCY count pipelines in flight: {count_sends:?}"
        );
    }
    let last_row_wave = row_sends
        .iter()
        .map(|s| s.wave)
        .max()
        .expect("row pipelines");
    assert!(
        count_waves.iter().all(|wave| *wave > last_row_wave),
        "the count phase runs after the row phase"
    );
    assert_eq!(observer.peak_in_flight(), EXT_PRE_IMAGE_CONCURRENCY);
    let mut want: Vec<(String, i64)> = news.iter().map(|h| (sha256_to_hex(h), 1)).collect();
    want.sort();
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        want,
        "every old hash released, every new one counted once"
    );

    // ── A one-hash count phase ──────────────────────────────────────────────
    let (session_id, oid) = new_object("b26e-one-hash").await;
    let shared = numbered_hash(9_999_999);
    pre_store_many(&ext, &[shared]).await;
    seed_ls
        .update_layer_relayed(
            &session_id,
            "b26e-one-hash",
            oid,
            Some(OwnedDelta {
                set_fields: vec![external_field_of("keeper".to_owned(), shared, 1)],
                ..Default::default()
            }),
            "b26e-writer",
            1,
        )
        .await
        .expect("the keeper lands");
    pre_store_many(&ext, &[shared]).await;
    let surplus_before = ext_ref_released(&metrics, REASON_SURPLUS);
    let (ls, observer) = observed();
    ls.update_layer_relayed(
        &session_id,
        "b26e-one-hash",
        oid,
        Some(OwnedDelta {
            set_fields: vec![external_field_of("second".to_owned(), shared, 2)],
            ..Default::default()
        }),
        "b26e-writer",
        2,
    )
    .await
    .expect("the second row lands");
    assert_eq!(
        observer
            .sent()
            .iter()
            .map(|s| s.count_reads)
            .collect::<Vec<_>>()
            .iter()
            .filter(|&&n| n > 0)
            .collect::<Vec<_>>(),
        vec![&1],
        "one count pipeline of one read"
    );
    assert_eq!(
        obj_ext_sorted(&tv, oid).await,
        vec![(sha256_to_hex(&shared), 2)],
        "the one count decoded: 1 + 1"
    );
    assert_eq!(
        ext_ref_released(&metrics, REASON_SURPLUS),
        surplus_before + 1,
        "the object already held the hash: its pre-flight increment is handed back"
    );

    // ── A corrupt count ─────────────────────────────────────────────────────
    let (session_id, oid) = new_object("b26e-corrupt-count").await;
    let (old, new) = (numbered_hash(8_000_001), numbered_hash(8_000_002));
    pre_store_many(&ext, &[old]).await;
    seed_ls
        .update_layer_relayed(
            &session_id,
            "b26e-corrupt-count",
            oid,
            Some(OwnedDelta {
                set_fields: vec![external_field_of("f".to_owned(), old, 1)],
                ..Default::default()
            }),
            "b26e-writer",
            1,
        )
        .await
        .expect("the seed lands");
    let index_key = tv.key_schema().obj_ext_index(&oid.to_string());
    let untouched = sha256_to_hex(&numbered_hash(8_000_003));
    let _: () = tv
        .client()
        .pool
        .hset(
            &index_key,
            vec![
                (sha256_to_hex(&old), "not-a-count".to_owned()),
                (untouched.clone(), "also-not-a-count".to_owned()),
            ],
        )
        .await
        .expect("corrupting obj:ext must succeed");
    pre_store_many(&ext, &[new]).await;
    let ticks_before = undecodable();
    ls.update_layer_relayed(
        &session_id,
        "b26e-corrupt-count",
        oid,
        Some(OwnedDelta {
            set_fields: vec![external_field_of("f".to_owned(), new, 2)],
            ..Default::default()
        }),
        "b26e-writer",
        2,
    )
    .await
    .expect("the rewrite lands");
    assert_eq!(
        undecodable() - ticks_before,
        1,
        "the touched hash's corrupt count ticks once; the untouched one is never read"
    );
    let index: std::collections::HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&index_key)
        .await
        .expect("HGETALL obj:ext");
    let mut index: Vec<(String, String)> = index.into_iter().collect();
    index.sort();
    let mut want = vec![
        (sha256_to_hex(&new), "1".to_owned()),
        (untouched, "also-not-a-count".to_owned()),
    ];
    want.sort();
    assert_eq!(
        index, want,
        "the corrupt touched count read as 0 (the row's release drops it), the new hash \
         counted, the untouched field left as it was"
    );
}
