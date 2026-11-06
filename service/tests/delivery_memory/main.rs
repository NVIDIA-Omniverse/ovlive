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

//! The heap the delivery paths hold for a payload (b.ntm, SDD §12.6, §12.7,
//! §21.2): a relayed payload fanned out to many subscribers is one copy, its
//! chunks slices of it; a checkpoint tail capture's envelope is encoded at
//! its exact size beside the records it is encoded from; and a catch-up
//! restoring backup-only values holds one batch of them at a time (D1).
//!
//! A dedicated binary because it installs a counting `#[global_allocator]`,
//! which applies to the whole binary — in `integration` it would count every
//! other test's allocations too. It wraps the system allocator and counts the
//! bytes Rust requests, net of frees, and the highest that count reached since
//! a span began; glibc's own per-block overhead is not counted. No Valkey:
//! each test drives the code that holds the bytes directly, after its inputs
//! are built.
//!
//! **What each test measures, and what it does not.**
//!
//! - [`one_relay_fanned_out_to_many_subscribers_holds_one_copy_and_chunk_slices`]
//!   (T1): the reader's copy-out of one 8 MiB relay payload
//!   (`DeliveryLedger::relay_buf`) and every subscriber's chunk messages
//!   (`chunk_ompf_for_stream`), all held at once, as response queues full of
//!   them would hold them. Before b.ntm each chunk was a copy, so the same
//!   fan-out held a copy per subscriber. Not measured: the stream reader's
//!   reply buffer and the transport's frame buffers, which this test has
//!   none of.
//! - [`a_tail_envelope_is_encoded_at_its_exact_size_beside_its_records`]
//!   (T16): the encode of a tail capture of `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES`
//!   of records — the records held, as the capture holds them, and the
//!   envelope — against the peak the pod memory plan charges for a capture
//!   (`config::memory_plan::tail_capture_peak_bytes`: twice the byte cap, a
//!   per-entry overhead and one reply page), its per-entry term taken over
//!   the records held rather than the entry cap, so a tighter bound than the
//!   plan's. The records' sizes force an
//!   encoder that grows its buffer by doubling past that peak. Not measured:
//!   the capture's stream reads, whose one reply page the plan's term
//!   carries.
//! - [`a_tail_capture_at_its_entry_cap_peaks_within_the_plans_per_entry_charge`]
//!   (T16 at the entry cap): a capture of `LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES`
//!   tiny commits — the slice it read and the records classified from it,
//!   held while the envelope is encoded — against twice the envelope and the
//!   plan's per-entry overhead (`config::memory_plan::TAIL_ENTRY_OVERHEAD_BYTES`),
//!   which with tiny records is most of the charge; it prints the measured
//!   figure per entry. Not measured: the paging reads' replies and each
//!   page's per-entry field maps, dropped page by page.
//! - [`a_restore_of_backup_only_values_holds_one_batch_of_them_at_a_time`]
//!   (D1 T1): a catch-up's restore of a layer's backup-only values before its
//!   layout (`storage::ompf_payload::restore_in_batches`), against a store
//!   double whose every restore answers a fresh allocation and keeps nothing,
//!   holds one batch's values at a time, within the charge the batch
//!   reserved. Before D1 the restore returned every value's bytes at once.
//!   Not measured here: the checkpoint adapter's own transient (the
//!   response's collect) and the store's write-back buffers, which the
//!   double has none of and the catch-up's restore-only charge
//!   (`storage::ompf_payload::EXT_RESTORE_CHARGE_CENTI`, 4.24 per byte) is
//!   sized to cover. A release-build measurement of the same phase on the
//!   real store, with a local-disk adapter, found 2.0 heap bytes and at most
//!   3.39 resident bytes per restored byte at 4 MiB values, the charge being
//!   ⌈1.25 × 3.39 × 100⌉ hundredths (SDD §22.22, the post-restore catch-up's
//!   restore phase; record `restore-memory-envelope-tv3-20261022`). Built only with
//!   `test-support`: a restore takes a fetch-on-miss layout, which outside
//!   the miss protocol only `RestoreLayout::restore_for_test` builds.
//!
//! Every test takes [`MEASURE`] first, so under `cargo test` (one process,
//! parallel threads) no two measure at once; under nextest each runs in its
//! own process anyway.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use live_edit::checkpoint::tail_codec::{
    decode_tail_envelope, encode_tail_envelope, TailHeader, TailRecord, TAIL_ENVELOPE_MAGIC,
    TAIL_FORMAT_VERSION, TAIL_HEADER_SIZE, TAIL_RECORD_PREFIX_SIZE,
};
use live_edit::config::memory_plan;
use live_edit::grpc::proto::{subscribe_read_layer_response, SubscribeReadLayerResponse};
use live_edit::grpc::streaming::{chunk_ompf_for_stream, OmpfStreamHeader, OMPF_CHUNK_SIZE};
use live_edit::storage::notifications::LayerNotification;
use live_edit::storage::stream_mux::core::StreamId;
use live_edit::storage::stream_mux::delivery_ledger::DeliveryLedger;
use live_edit::storage::stream_mux::XREAD_REPLY_TARGET_BYTES;
use live_edit::storage::tail_slice::{TailEntryPayload, TailStreamEntry};
// D1 T1 drives the restore through `RestoreLayout::restore_for_test`, which
// only a `test-support` build has.
#[cfg(feature = "test-support")]
use live_edit::{
    error::ServiceError,
    storage::diff_read::RestoreLayout,
    storage::ext_release_gate::ExtReleaseGate,
    storage::external_values::{
        ExtRelease, ExtValueLengths, ExtValueProbe, ExtValueVerify, ExternalValueStore,
        GcBatchResult, GcPassState, GcPolicy, OnFetched, ReleaseAttemptOutcome,
    },
    storage::memory_budget::{MemoryBudget, CATCHUP_PAYLOAD_SERIES},
    storage::ompf_payload::{
        restore_charge_bytes, restore_in_batches, RestoredLength, EXT_RESTORE_CHARGE_CENTI,
        EXT_RESTORE_PARALLELISM,
    },
};

// ═════════════════════════════════════════════════════════════════════════════
// The counting allocator
// ═════════════════════════════════════════════════════════════════════════════

/// Bytes allocated and not yet freed, process-wide.
static LIVE: AtomicIsize = AtomicIsize::new(0);

/// The highest [`LIVE`] reached since the last [`span`] began.
static PEAK: AtomicIsize = AtomicIsize::new(0);

struct Counting;

/// Move [`LIVE`] by `delta` and raise [`PEAK`] to it.
fn account(delta: isize) {
    let now = LIVE.fetch_add(delta, Ordering::SeqCst) + delta;
    PEAK.fetch_max(now, Ordering::SeqCst);
}

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged and only updates the counters on the side.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded verbatim; the caller upholds `alloc`'s contract.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            account(layout.size() as isize);
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as `alloc`.
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            account(layout.size() as isize);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as `alloc`; `ptr` came from this allocator with `layout`.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size() as isize, Ordering::SeqCst);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: as `dealloc`.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            // A block that moves needs both copies for the copy: count the
            // new one before the old one goes, as the peak saw it.
            account(new_size as isize);
            LIVE.fetch_sub(layout.size() as isize, Ordering::SeqCst);
        }
        p
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn live() -> isize {
    LIVE.load(Ordering::SeqCst)
}

/// Begin a measured span: the peak restarts from what is live now, which is
/// returned.
fn span() -> isize {
    let now = live();
    PEAK.store(now, Ordering::SeqCst);
    now
}

/// The highest heap since the span began, above where it began.
fn peak_since(start: isize) -> isize {
    PEAK.load(Ordering::SeqCst) - start
}

/// Serializes the measuring tests within one process.
static MEASURE: Mutex<()> = Mutex::new(());

fn measure_lock() -> std::sync::MutexGuard<'static, ()> {
    MEASURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

const MIB: usize = 1 << 20;

// ═════════════════════════════════════════════════════════════════════════════
// T1: one relay, many subscribers
// ═════════════════════════════════════════════════════════════════════════════

/// The relayed payload: 8 MiB and a tail, so it is nine chunks, the last short.
const RELAY_BYTES: usize = 8 * MIB + 123;

/// Subscribers of the layer the relay is fanned out to.
const SUBSCRIBERS: usize = 16;

/// What the fan-out may hold beyond the one copy: every subscriber's ten
/// messages (a header and nine chunks, each a struct around a slice of the
/// shared copy, which allocates nothing) and the vectors holding them, a few
/// KiB in all. A copy per chunk, as before b.ntm, is the whole payload per
/// subscriber: 128 MiB here.
const FAN_OUT_SLACK: isize = 256 * 1024;

/// T1 (fails before b.ntm): one 8 MiB relay copied out of a reply once and
/// fanned out to [`SUBSCRIBERS`] subscribers, every one's chunk messages held
/// at once, grows the heap by one payload and [`FAN_OUT_SLACK`], not by a copy
/// per subscriber's chunk. Every subscriber's chunks are byte-identical to the
/// relay: exactly 1 MiB each but the last, `total_size` the relay's length.
#[test]
fn one_relay_fanned_out_to_many_subscribers_holds_one_copy_and_chunk_slices() {
    let _guard = measure_lock();
    let reply: Vec<u8> = (0..RELAY_BYTES).map(|i| (i % 253) as u8).collect();
    let ledger = DeliveryLedger::new(u64::MAX, Arc::new(Mutex::new(None)));
    let header = OmpfStreamHeader {
        object_id: 1,
        sequence_num: 2,
        server_timestamp: 3,
        client_timestamp: 4,
    };
    let mut held: Vec<Vec<SubscribeReadLayerResponse>> = Vec::with_capacity(SUBSCRIBERS);

    let start = span();
    let relay = ledger.relay_buf("le:obj:1:stream", StreamId { ms: 2, seq: 0 }, &reply);
    for _ in 0..SUBSCRIBERS {
        held.push(chunk_ompf_for_stream(relay.bytes().clone(), &header).collect());
    }
    let grown = live() - start;
    let peak = peak_since(start);

    assert!(
        grown >= RELAY_BYTES as isize,
        "fixture self-check: the copy-out is counted ({grown} B)"
    );
    assert!(
        peak <= RELAY_BYTES as isize + FAN_OUT_SLACK,
        "fanning one {RELAY_BYTES} B relay out to {SUBSCRIBERS} subscribers peaked at {peak} B: \
         more than the one copy and {FAN_OUT_SLACK} B, so the chunks are copies, not slices of it"
    );

    for (s, messages) in held.iter().enumerate() {
        let (first, chunks) = messages.split_first().expect("a header and chunks");
        match &first.response_type {
            Some(subscribe_read_layer_response::ResponseType::Header(h)) => {
                assert_eq!(h.total_size, RELAY_BYTES as u64, "subscriber {s}");
            }
            other => panic!("subscriber {s}: the first message is the header: {other:?}"),
        }
        assert_eq!(chunks.len(), RELAY_BYTES.div_ceil(OMPF_CHUNK_SIZE));
        let mut at = 0;
        for (i, message) in chunks.iter().enumerate() {
            let Some(subscribe_read_layer_response::ResponseType::Chunk(chunk)) =
                &message.response_type
            else {
                panic!("subscriber {s}: message {i} after the header is a chunk");
            };
            let expected_len = OMPF_CHUNK_SIZE.min(RELAY_BYTES - at);
            assert_eq!(chunk.data.len(), expected_len, "subscriber {s}, chunk {i}");
            assert!(
                chunk.data[..] == reply[at..at + expected_len],
                "subscriber {s}, chunk {i}: byte-identical to the relay"
            );
            at += expected_len;
        }
    }
    drop(held);
    drop(relay);
    assert_eq!(
        ledger.charged_bytes(),
        0,
        "the copy's charge is repaid with its last slice"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// T16: the tail envelope's exact size
// ═════════════════════════════════════════════════════════════════════════════

/// The capture's byte cap here: the records are this much, near enough.
const TAIL_MAX_BYTES: usize = 8 * MIB;

/// The small records ahead of the large last one: a buffer grown by
/// doubling from the header reaches about 6 MiB of capacity over them, and
/// the last record then doubles it to about 12 MiB for an 8 MiB envelope.
const SMALL_RECORDS: usize = 40;

/// Each small record's payload.
const SMALL_RECORD_BYTES: usize = 100_000;

/// The plan's per-entry overhead of a tail capture.
const TAIL_ENTRY_OVERHEAD_BYTES: usize = memory_plan::TAIL_ENTRY_OVERHEAD_BYTES as usize;

/// One reply page at the stream readers' reply target, which the plan's
/// tail term carries.
const REPLY_PAGE_BYTES: usize = XREAD_REPLY_TARGET_BYTES as usize;

/// The records of a capture of [`TAIL_MAX_BYTES`]: [`SMALL_RECORDS`] of
/// [`SMALL_RECORD_BYTES`], then one large one bringing the envelope to the
/// cap, each a verbatim commit with its own payload allocation; and the
/// header claiming exactly them.
fn capture_records() -> (TailHeader, Vec<TailRecord>) {
    let from_seq = 100;
    let mut sizes = vec![SMALL_RECORD_BYTES; SMALL_RECORDS];
    let used = TAIL_HEADER_SIZE
        + (SMALL_RECORDS + 1) * TAIL_RECORD_PREFIX_SIZE
        + SMALL_RECORDS * SMALL_RECORD_BYTES;
    sizes.push(TAIL_MAX_BYTES - used);
    let records: Vec<TailRecord> = sizes
        .iter()
        .enumerate()
        .map(|(i, &len)| TailRecord::Verbatim {
            sequence_num: from_seq + 1 + i as u64,
            server_ts: 1_000 + i as i64,
            ompf: Bytes::from(
                (0..len)
                    .map(|b| (b % 251) as u8 ^ i as u8)
                    .collect::<Vec<u8>>(),
            ),
        })
        .collect();
    let header = TailHeader {
        object_id: 7,
        base_snap_seq: from_seq,
        from_seq,
        to_seq: from_seq + records.len() as u64,
    };
    (header, records)
}

/// The envelope the documented v1 layout gives for `header` and `records`,
/// written field by field: the bytes the encoder has always produced.
fn layout_bytes(header: &TailHeader, records: &[TailRecord]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&TAIL_ENVELOPE_MAGIC);
    out.extend_from_slice(&TAIL_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&header.object_id.to_le_bytes());
    out.extend_from_slice(&header.base_snap_seq.to_le_bytes());
    out.extend_from_slice(&header.from_seq.to_le_bytes());
    out.extend_from_slice(&header.to_seq.to_le_bytes());
    out.extend_from_slice(&(records.len() as u32).to_le_bytes());
    assert_eq!(
        out.len(),
        TAIL_HEADER_SIZE,
        "fixture self-check: the header"
    );
    for record in records {
        let payload = record.ompf().expect("verbatim records");
        out.push(record.kind().as_u8());
        out.extend_from_slice(&record.sequence_num().to_le_bytes());
        out.extend_from_slice(&record.server_ts().to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
    }
    out
}

/// T16 (its first half fails before b.ntm): a tail capture of
/// [`TAIL_MAX_BYTES`] of records — held, as the capture holds them, while the
/// envelope is encoded — peaks at or below the pod memory plan's charge for
/// it, taken tighter here: twice the byte cap, [`TAIL_ENTRY_OVERHEAD_BYTES`]
/// for each of the 41 records it holds (the plan charges it for every entry
/// up to `LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES`) and one reply page. The
/// records end in a large one, which takes a buffer grown by doubling to
/// about 12 MiB of capacity for the 8 MiB envelope, past that charge; the
/// envelope encoded at its exact size is not. Its bytes are the documented
/// layout's — the encoder's output did not change — and it decodes to the
/// records.
#[test]
fn a_tail_envelope_is_encoded_at_its_exact_size_beside_its_records() {
    let _guard = measure_lock();
    let start = span();
    let (header, records) = capture_records();
    let held = live() - start;
    assert!(
        held >= (TAIL_MAX_BYTES - MIB) as isize,
        "fixture self-check: the records hold about the cap ({held} B)"
    );
    let expected = layout_bytes(&header, &records);
    let expected_len = expected.len();
    assert_eq!(
        expected_len, TAIL_MAX_BYTES,
        "fixture self-check: at the cap"
    );
    drop(expected);
    let charge = 2 * TAIL_MAX_BYTES + records.len() * TAIL_ENTRY_OVERHEAD_BYTES + REPLY_PAGE_BYTES;

    let encode_from = span();
    let envelope = encode_tail_envelope(&header, records.iter().cloned()).expect("encodes");
    let peak = peak_since(encode_from) + held;
    assert!(
        peak <= charge as isize,
        "the capture of {TAIL_MAX_BYTES} B of records peaked at {peak} B while its envelope was \
         encoded, past the {charge} B bound (tighter than the pod memory plan's charge: the \
         per-entry overhead is taken over the records held, not the entry cap): the envelope is \
         not allocated at its exact size"
    );

    assert_eq!(envelope.len(), expected_len);
    assert!(
        envelope[..] == layout_bytes(&header, &records)[..],
        "the envelope's bytes are the documented layout's"
    );
    let decoded = decode_tail_envelope(&envelope).expect("decodes");
    assert_eq!(decoded.records(), &records[..]);
}

/// The tail capture's entry cap at the chart's configuration
/// (`LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES`, 4,096).
const TAIL_ENTRY_CAP: usize = memory_plan::PlanInputs::reference().tail_max_entries as usize;

/// Each record's payload at the entry cap: tiny, so the plan's per-entry term
/// is most of its charge.
const TINY_RECORD_BYTES: usize = 64;

/// A client id of a realistic length (a UUID's).
const CLIENT_ID: &str = "8f14e45f-ceea-467f-a0e6-5c2b1a7d9e10";

/// What a tail capture of [`TAIL_ENTRY_CAP`] tiny commits holds while its
/// envelope is encoded (`checkpoint::tail_capture`): the slice it read —
/// each entry's stream id, its parsed notification and the payload copied
/// out of the reply — and the records classified from it, each sharing its
/// entry's payload; and the header claiming exactly them.
fn entry_cap_capture() -> (TailHeader, Vec<TailStreamEntry>, Vec<TailRecord>) {
    let from_seq = 100u64;
    let entries: Vec<TailStreamEntry> = (0..TAIL_ENTRY_CAP)
        .map(|i| {
            let seq = from_seq + 1 + i as u64;
            let server_ts = 1_000 + i as i64;
            let payload: Vec<u8> = (0..TINY_RECORD_BYTES)
                .map(|b| (b % 251) as u8 ^ i as u8)
                .collect();
            let notification = LayerNotification::parse(&format!(
                "update|7|{seq}|{CLIENT_ID}|{server_ts}|{server_ts}"
            ))
            .expect("an update notification parses")
            .with_relay_ompf(Some(Bytes::copy_from_slice(&payload)));
            TailStreamEntry {
                stream_id: format!("{seq}-0"),
                payload: TailEntryPayload::Notification(notification),
                payload_bytes: TINY_RECORD_BYTES as u64,
            }
        })
        .collect();
    let records: Vec<TailRecord> = entries
        .iter()
        .map(|entry| match &entry.payload {
            TailEntryPayload::Notification(LayerNotification::Update {
                sequence_num,
                server_ts,
                relay_ompf: Some(ompf),
                ..
            }) => TailRecord::Verbatim {
                sequence_num: *sequence_num,
                server_ts: *server_ts,
                ompf: ompf.clone(),
            },
            other => panic!("fixture: every entry is an update carrying its OMPF: {other:?}"),
        })
        .collect();
    let header = TailHeader {
        object_id: 7,
        base_snap_seq: from_seq,
        from_seq,
        to_seq: from_seq + records.len() as u64,
    };
    (header, entries, records)
}

/// T16 at the entry cap: a tail capture of [`TAIL_ENTRY_CAP`] tiny commits —
/// the slice it read and the records classified from it, held while the
/// envelope is encoded — peaks at or below the pod memory plan's charge for
/// it, its byte term taken over the envelope encoded rather than the byte cap:
/// twice the envelope, and [`TAIL_ENTRY_OVERHEAD_BYTES`] per entry. With tiny
/// records the per-entry term is most of that charge, so this pins the
/// plan's per-entry figure against what each entry holds — its stream id,
/// its parsed notification, its record and their allocations — which the
/// byte-cap case above cannot: there, 41 entries' worth of it is lost in the
/// byte term. No reply page is credited: nothing here reads one. Not
/// measured: the paging reads' replies and each page's per-entry field maps,
/// which a capture drops page by page as it reads.
#[test]
fn a_tail_capture_at_its_entry_cap_peaks_within_the_plans_per_entry_charge() {
    let _guard = measure_lock();
    let start = span();
    let (header, entries, records) = entry_cap_capture();
    let envelope = encode_tail_envelope(&header, records.iter().cloned()).expect("encodes");
    let peak = peak_since(start);

    let envelope_len = envelope.len();
    assert_eq!(
        envelope_len,
        TAIL_HEADER_SIZE + TAIL_ENTRY_CAP * (TAIL_RECORD_PREFIX_SIZE + TINY_RECORD_BYTES),
        "fixture self-check: the envelope is the records' framing and payloads"
    );
    let byte_term = 2 * envelope_len;
    let beyond = (peak as usize).saturating_sub(byte_term);
    assert!(
        beyond > byte_term,
        "fixture self-check: what the entries hold beyond twice the envelope ({beyond} B) is most \
         of the peak, over the byte term ({byte_term} B)"
    );
    let charge = byte_term + TAIL_ENTRY_CAP * TAIL_ENTRY_OVERHEAD_BYTES;
    let per_entry = beyond / TAIL_ENTRY_CAP;
    println!(
        "T16 at the entry cap: {TAIL_ENTRY_CAP} entries of {TINY_RECORD_BYTES} B peaked at {peak} \
         B, {per_entry} B per entry beyond twice the {envelope_len} B envelope; the plan charges \
         {TAIL_ENTRY_OVERHEAD_BYTES} B"
    );
    assert!(
        peak <= charge as isize,
        "a tail capture of {TAIL_ENTRY_CAP} entries of {TINY_RECORD_BYTES} B peaked at {peak} B \
         while its envelope was encoded, past the {charge} B the pod memory plan charges for it \
         (twice the {envelope_len} B envelope, and {TAIL_ENTRY_OVERHEAD_BYTES} B per entry): \
         {per_entry} B per entry"
    );

    let decoded = decode_tail_envelope(&envelope).expect("decodes");
    assert_eq!(decoded.records(), &records[..]);
    drop(entries);
}

// ═════════════════════════════════════════════════════════════════════════════
// D1 T1: a catch-up's restore of backup-only values, in charged batches
// ═════════════════════════════════════════════════════════════════════════════

/// The backup-only values one catch-up restores here.
#[cfg(feature = "test-support")]
const RESTORED_VALUES: usize = 16;

/// Each restored value's size.
#[cfg(feature = "test-support")]
const RESTORED_VALUE_BYTES: usize = 4 * MIB;

/// What the restore may hold beyond one batch's values: its bookkeeping (the
/// batch's hashes, charges and shares, the lengths answered) and the boxed
/// futures of its store calls, a few KiB in all.
#[cfg(feature = "test-support")]
const RESTORE_SLACK: isize = 1024 * 1024;

/// A store whose external values only the backup store holds: its length
/// probe finds every value missing, and each restore through the checkpoint
/// adapter answers a fresh allocation of the value — as the adapter's fetch
/// allocates it — and keeps nothing of it, so what the restore phase holds is
/// what [`restore_in_batches`] holds. Each restore yields once after it
/// allocates, so the values of one batch, restored at once, are all held
/// together before any is dropped. Every other method is unreachable: the
/// restore phase calls none of them.
#[cfg(feature = "test-support")]
struct BackupOnlyStore {
    value_bytes: usize,
}

#[cfg(feature = "test-support")]
#[async_trait::async_trait]
impl ExternalValueStore for BackupOnlyStore {
    async fn store_values(&self, _: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
        unreachable!("a restore phase stores through recover_value only")
    }

    async fn store_verified_values(&self, _: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
        unreachable!("a restore phase stores through recover_value only")
    }

    fn ext_value_verify(&self) -> ExtValueVerify<'_> {
        ExtValueVerify::ENFORCE_UNCOUNTED
    }

    async fn load_values(&self, _: &[&[u8; 32]]) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
        unreachable!("a restore phase reads no whole value")
    }

    async fn read_batch(
        &self,
        _: &[[u8; 32]],
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        unreachable!("a restore phase reads no whole value")
    }

    async fn probe_values(&self, _: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError> {
        unreachable!("a restore phase probes lengths only")
    }

    async fn probe_value_lengths(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<ExtValueLengths, ServiceError> {
        Ok(ExtValueLengths {
            lengths: std::collections::HashMap::new(),
            missing: hashes.to_vec(),
        })
    }

    async fn recover_value(
        &self,
        hash: &[u8; 32],
        on_fetched: OnFetched<'_>,
    ) -> Result<Bytes, ServiceError> {
        let fetched = Bytes::from(vec![hash[0]; self.value_bytes]);
        // Held across the yield that stands in for the write-back, as the
        // store holds it.
        let charge = on_fetched(fetched.len() as u64);
        tokio::task::yield_now().await;
        drop(charge);
        Ok(fetched)
    }

    async fn enqueue_retirement_attempt(
        &self,
        _: &str,
        _: &str,
        _: &[ExtRelease],
        _: std::time::Instant,
    ) -> Result<ReleaseAttemptOutcome, ServiceError> {
        unreachable!("a restore phase retires nothing")
    }

    async fn process_pending(
        &self,
        _: &GcPolicy,
        _: &mut GcPassState,
    ) -> Result<GcBatchResult, ServiceError> {
        unreachable!("a restore phase runs no GC")
    }

    fn ext_release_gate(&self) -> Option<&Arc<ExtReleaseGate>> {
        None
    }
}

/// D1 T1: a catch-up restoring [`RESTORED_VALUES`] backup-only values of
/// [`RESTORED_VALUE_BYTES`] each ([`restore_in_batches`], with the catch-up's
/// restore charge, [`EXT_RESTORE_CHARGE_CENTI`] hundredths per byte, against
/// the reference plan's catch-up payload budget
/// `D_c`) holds one batch of them at a time — at most
/// [`EXT_RESTORE_PARALLELISM`] values and [`RESTORE_SLACK`] — never the whole
/// layer's values at once, and that is within the charge the batch reserved.
/// Every value is answered at its length as restored, and every restored
/// byte is dropped once the restore phase returns. The self-check that the
/// peak reached one whole batch shows the batch's values were held together.
///
/// Fails without the change (in-place perturbation): restoring every value in
/// one batch peaks at all [`RESTORED_VALUES`] values.
#[cfg(feature = "test-support")]
#[tokio::test]
// The measuring lock is held across the restore on purpose: it keeps every
// other measuring test of the process out of the span. It is a std mutex no
// task of this current-thread runtime ever takes, so the await cannot block on
// it.
#[allow(clippy::await_holding_lock)]
async fn a_restore_of_backup_only_values_holds_one_batch_of_them_at_a_time() {
    let _guard = measure_lock();
    let store = BackupOnlyStore {
        value_bytes: RESTORED_VALUE_BYTES,
    };
    let layout = RestoreLayout::restore_for_test(
        (0..RESTORED_VALUES).map(|i| ([i as u8 + 1; 32], RESTORED_VALUE_BYTES as u64)),
    );
    let total = memory_plan::PodMemoryPlan::reference().catchup_payload_budget_bytes();
    let budget = Arc::new(MemoryBudget::with_series(
        total / 1024,
        Arc::new(Mutex::new(None)),
        &CATCHUP_PAYLOAD_SERIES,
    ));
    let batch_bytes = (EXT_RESTORE_PARALLELISM * RESTORED_VALUE_BYTES) as isize;
    let batch_charge = EXT_RESTORE_PARALLELISM as u64
        * restore_charge_bytes(RESTORED_VALUE_BYTES as u64, EXT_RESTORE_CHARGE_CENTI);
    assert!(
        batch_charge <= budget.total_bytes(),
        "fixture self-check: a whole batch's charge ({batch_charge} B) fits in D_c ({} B), so a \
         batch is EXT_RESTORE_PARALLELISM values",
        budget.total_bytes()
    );
    assert!(
        batch_bytes + RESTORE_SLACK <= batch_charge as isize,
        "fixture self-check: the bound asserted is within the batch's charge"
    );

    let start = span();
    let restored = restore_in_batches(
        &store,
        &layout,
        EXT_RESTORE_CHARGE_CENTI,
        budget.total_bytes(),
        |bytes| {
            let budget = Arc::clone(&budget);
            async move {
                budget.reserve(bytes, None).await.map_err(|refusal| {
                    ServiceError::Internal(format!("an idle budget refused: {refusal:?}"))
                })
            }
        },
    )
    .await
    .expect("every value restores");
    let peak = peak_since(start);
    let left = live() - start;

    assert!(
        peak >= batch_bytes,
        "fixture self-check: the restore held one whole batch's values at once ({peak} B, under \
         {batch_bytes} B), so the bound below is not vacuous"
    );
    assert!(
        peak <= batch_bytes + RESTORE_SLACK,
        "restoring {RESTORED_VALUES} values of {RESTORED_VALUE_BYTES} B peaked at {peak} B: more \
         than one batch of {EXT_RESTORE_PARALLELISM} values ({batch_bytes} B) and {RESTORE_SLACK} \
         B, so it held more than one batch's values at once (all of them is {} B)",
        RESTORED_VALUES * RESTORED_VALUE_BYTES
    );
    assert!(
        left <= RESTORE_SLACK,
        "every restored byte is dropped once the restore returns ({left} B still held)"
    );
    assert_eq!(restored.len(), RESTORED_VALUES);
    for value in layout.restores() {
        assert_eq!(
            restored.get(&value.hash()),
            Some(&RestoredLength {
                len: RESTORED_VALUE_BYTES as u64,
                restored: true,
            }),
            "each value answered at its restored length, restored by this catch-up"
        );
    }
    assert_eq!(
        (budget.reserved_bytes(), budget.overcommit_bytes()),
        (0, 0),
        "every batch's charge is repaid"
    );
}
