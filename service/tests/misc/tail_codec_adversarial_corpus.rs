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

//! Integration-test corpus of adversarial tail-envelope artifacts (b.dfh
//! Epic 2, Task 1).
//!
//! The oodf-ffi cargo-fuzz harness targets `oodf_ffi::parse_delta`, which
//! lives in the oodf-ffi crate. Tail-envelope decoding lives in live-edit
//! (`src/checkpoint/tail_codec.rs`) — adding a `fuzz_decode_tail_envelope`
//! target there would require either pulling `live-edit` (with tonic, tokio
//! and fred in its dependency graph) into the fuzz crate, or duplicating the
//! codec into oodf-ffi. Neither is desirable: the former bloats the fuzz image
//! dramatically, and the latter fragments the codebase. This is exactly the
//! position `src/domain/ompf.rs` is in, so the tail codec inherits the same
//! documented fallback — see `tests/misc/ompf_adversarial_corpus.rs`, whose
//! structure this file mirrors.
//!
//! This file is that fallback: a byte-level corpus of adversarial tail
//! shapes that exercise the surface a fuzzer would hit first — the
//! **pre-validation allocation** path, where a header or record prefix
//! claims a size before a single byte of that size has been seen.
//! Random-byte decode-never-panic coverage is provided by the proptest
//! `proptest_tail_envelope_decode_never_panics_on_arbitrary_bytes` in
//! `tests/proptests.rs` (plus its corrupted-valid-encoding sibling
//! `proptest_tail_envelope_decode_never_panics_on_corrupted_valid_encoding`).
//! Combined, these tests form the tail-codec equivalent of the
//! `fuzz_parse_delta` harness for oodf-ffi.
//!
//! # Scope split
//!
//! Semantic malformation with exact-variant assertions (bad magic, unknown
//! format version, unknown record kind, kind/payload aliasing, non-contiguous
//! sequences, trailing bytes, `RecordCountMismatch`, and a `0..len`
//! truncate-at-every-byte loop) is covered by the codec's own inline
//! `#[cfg(test)]` module in `src/checkpoint/tail_codec.rs`. **This** file
//! covers only resource-exhaustion shapes and the two cross-format
//! mis-routing directions. Cases deliberately *not* built here are recorded
//! as omission comments at their natural position below rather than silently
//! dropped.
//!
//! # Field widths are themselves a guard
//!
//! `record_count` and `payload_len` are both `u32` on the wire (44-byte
//! header, 21-byte record prefix — see the wire-format table in the codec's
//! module doc). The `u64::MAX` / `usize::MAX` variants of the classic
//! length-claim attacks are therefore *unrepresentable by construction*: the
//! worst a forged artifact can claim is `u32::MAX`, which is what every
//! builder below claims. That is a finding, not a gap, and it is why several
//! shapes named in the plan appear here as omission comments.

use bytes::Bytes;
use live_edit::checkpoint::{
    decode_tail_envelope, encode_tail_envelope, TailCodecError, TailHeader, TailRecord,
    TailRecordKind, TAIL_ENVELOPE_MAGIC, TAIL_FORMAT_VERSION, TAIL_HEADER_SIZE,
    TAIL_RECORD_PREFIX_SIZE,
};
use live_edit::domain::ompf::{OmpfContainer, OmpfError};

/// Arbitrary fixed timestamp. Nothing in the codec reads a clock, so the value
/// is irrelevant to every assertion here; it is fixed so the corpus bytes are
/// reproducible. Microseconds, matching `LayerNotification::Update`.
const TS: i64 = 1_764_000_000_000_000;

/// Arbitrary layer generation. Nothing here depends on its value.
const OBJ: u64 = 0x0123_4567_89AB_CDEF;

/// No adversarial artifact in this corpus is larger than this. Every builder
/// asserts against it, because "a tiny input claims a huge allocation" is the
/// whole hazard class: a shape that needed megabytes of input to provoke a
/// gigabyte allocation would not be a DoS worth guarding.
const MAX_CORPUS_ARTIFACT_BYTES: usize = 256;

// ---------------------------------------------------------------------------
// Byte-level builders. These deliberately do NOT go through
// `encode_tail_envelope` — the encoder refuses to emit every shape below, so
// the artifacts have to be hand-assembled to reach the decoder at all.
// ---------------------------------------------------------------------------

/// Assemble a raw 44-byte tail header with fully caller-controlled fields.
/// Magic and `format_version` are always the real ones, so every artifact
/// built on top of this reaches the decoder's structural checks rather than
/// stopping at `BadMagic` / `UnknownFormatVersion`.
fn tail_header_bytes(
    object_id: u64,
    base_snap_seq: u64,
    from_seq: u64,
    to_seq: u64,
    record_count: u32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(TAIL_HEADER_SIZE);
    bytes.extend_from_slice(&TAIL_ENVELOPE_MAGIC);
    bytes.extend_from_slice(&TAIL_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&object_id.to_le_bytes());
    bytes.extend_from_slice(&base_snap_seq.to_le_bytes());
    bytes.extend_from_slice(&from_seq.to_le_bytes());
    bytes.extend_from_slice(&to_seq.to_le_bytes());
    bytes.extend_from_slice(&record_count.to_le_bytes());
    assert_eq!(
        bytes.len(),
        TAIL_HEADER_SIZE,
        "corpus header must be exactly the fixed header size"
    );
    bytes
}

/// Assemble a raw 21-byte `Verbatim` record prefix with a caller-controlled
/// declared `payload_len`, and no payload bytes of its own. Callers append
/// payload bytes (or deliberately do not) themselves.
fn verbatim_record_prefix(sequence_num: u64, payload_len: u32) -> Vec<u8> {
    record_prefix(TailRecordKind::Verbatim, sequence_num, payload_len)
}

/// Assemble a raw 21-byte `EmptyDelta` record prefix. An empty-delta record
/// carries no payload, so its on-wire stride is exactly
/// [`TAIL_RECORD_PREFIX_SIZE`] — which is what makes it the right shape for
/// probing the size-derived `record_count` cap, whose divisor is that stride.
fn empty_delta_record_prefix(sequence_num: u64) -> Vec<u8> {
    record_prefix(TailRecordKind::EmptyDelta, sequence_num, 0)
}

fn record_prefix(kind: TailRecordKind, sequence_num: u64, payload_len: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(TAIL_RECORD_PREFIX_SIZE);
    bytes.push(kind.as_u8());
    bytes.extend_from_slice(&sequence_num.to_le_bytes());
    bytes.extend_from_slice(&TS.to_le_bytes());
    bytes.extend_from_slice(&payload_len.to_le_bytes());
    assert_eq!(
        bytes.len(),
        TAIL_RECORD_PREFIX_SIZE,
        "corpus record prefix must be exactly the fixed prefix size"
    );
    bytes
}

/// Build a pathological bare 44-byte tail header claiming
/// `record_count = u32::MAX` with zero record bytes following.
///
/// A naive decoder that did `Vec::with_capacity(record_count)` before bounding
/// the claim against the input size would reserve ~4.3 billion `TailRecord`
/// slots — at `size_of::<TailRecord>()` (five words plus a discriminant) that
/// is well over 100 GB, and the process is OOM-killed before any assertion in
/// the test runs. This is the tail-codec analogue of the OMPF
/// "DoS-with-8-bytes" case (`adversarial_oom_header`,
/// `tests/misc/ompf_adversarial_corpus.rs`), which reserves ~34 GB from an
/// 8-byte input.
///
/// The header range has to span the forged count too (`from_seq = 0`,
/// `to_seq = u32::MAX`): the decoder cross-checks `record_count` against
/// `to_seq - from_seq` *before* the allocation guard, so patching the count
/// alone lands on `RecordCountMismatch` (already covered inline by
/// `decode_record_count_disagreeing_with_header_range_is_rejected`) and never
/// reaches the allocation path this corpus exists to test.
fn adversarial_oom_record_count_header() -> Vec<u8> {
    tail_header_bytes(OBJ, 0, 0, u64::from(u32::MAX), u32::MAX)
}

/// Same forged `record_count = u32::MAX`, but with two records' worth of
/// trailing bytes so the size-derived cap is a small non-zero number rather
/// than zero.
///
/// A naive decoder that only special-cased the "no bytes at all" shape — or
/// that compared the claim against the *total* size instead of the bytes
/// remaining after the header — would sail past this one. It also pins that
/// the reported `max` is derived from the remaining bytes divided by the
/// fixed record-prefix stride, so a future stride change cannot silently
/// widen the guard.
fn adversarial_oom_record_count_with_partial_record_bytes() -> Vec<u8> {
    let mut bytes = adversarial_oom_record_count_header();
    bytes.extend_from_slice(&[0u8; 2 * TAIL_RECORD_PREFIX_SIZE]);
    bytes
}

/// Build a 65-byte artifact: a well-formed header declaring exactly one
/// record, followed by one `Verbatim` record prefix declaring
/// `payload_len = u32::MAX` with no payload bytes at all.
///
/// A naive decoder that copied the declared payload into an owned buffer —
/// `Vec::with_capacity(payload_len)` or `vec![0u8; payload_len]` — before
/// checking it against the bytes actually present would reserve 4 GiB from a
/// 65-byte input. The landed decoder instead slices the payload out of the
/// caller's `Bytes` and bounds `prefix_end + payload_len` against
/// `bytes.len()` with `checked_add` first, so the claim never sizes an
/// allocation.
///
/// Omission: the plan's `u64::MAX` and `usize::MAX as u64` payload-length
/// variants do not exist on this wire format — `payload_len` is a `u32`
/// field, so `u32::MAX` below IS the maximum claim expressible. The field
/// width is the structural guard.
fn adversarial_oom_record_length() -> Vec<u8> {
    let mut bytes = tail_header_bytes(OBJ, 0, 0, 1, 1);
    bytes.extend_from_slice(&verbatim_record_prefix(1, u32::MAX));
    bytes
}

/// Build a 90-byte artifact whose *first* record is entirely well-formed
/// (4-byte payload present) and whose *second* record declares
/// `payload_len = u32::MAX` with no payload bytes.
///
/// This is the running-offset case: the decoder has already advanced past a
/// real record, so the payload-end computation is `non_zero_offset +
/// u32::MAX` rather than `header_size + u32::MAX`. A decoder accumulating
/// offsets with plain `+` would panic in debug and wrap in release on a
/// 32-bit target, and a decoder that pre-summed all declared lengths to size
/// one buffer would allocate from the sum. The landed decoder reads records
/// one at a time, each bounded against `bytes.len()` via `checked_add`, and
/// never forms a sum at all.
fn adversarial_length_prefix_sum_overflow() -> Vec<u8> {
    let mut bytes = tail_header_bytes(OBJ, 0, 0, 2, 2);
    bytes.extend_from_slice(&verbatim_record_prefix(1, 4));
    bytes.extend_from_slice(b"ompf");
    bytes.extend_from_slice(&verbatim_record_prefix(2, u32::MAX));
    bytes
}

/// Build a 107-byte artifact declaring three records, each claiming
/// `payload_len = u32::MAX`, none carrying a payload byte.
///
/// The declared lengths sum to just under 12 GiB. On a 64-bit target that sum
/// does not overflow `usize` — it is simply a colossal allocation a naive
/// pre-summing decoder would attempt; on a 32-bit target the same sum wraps
/// `usize` and a decoder without checked arithmetic would then read
/// in-bounds-but-wrong bytes. Either way the artifact is 107 bytes.
///
/// Omission: a sum that overflows `u64` is unreachable here. `record_count`
/// is capped at `(len - 44) / 21` before any record is read, so an artifact
/// small enough to build carries at most a handful of records, and
/// `records * u32::MAX` cannot approach `u64::MAX`. The `u32` length field
/// combined with the size-derived record cap is the structural guard.
fn adversarial_every_record_claims_max_payload() -> Vec<u8> {
    let mut bytes = tail_header_bytes(OBJ, 0, 0, 3, 3);
    for sequence_num in 1..=3u64 {
        bytes.extend_from_slice(&verbatim_record_prefix(sequence_num, u32::MAX));
    }
    bytes
}

/// Records used to probe the size-derived `record_count` cap. Empty-delta
/// records are used because they carry no payload, so their on-wire stride is
/// exactly [`TAIL_RECORD_PREFIX_SIZE`] — the same value the decoder divides by
/// to derive the cap. That makes the cap land on a known integer
/// (`(len - 44) / 21`) with no remainder, which is what lets the pair of tests
/// below sit exactly on it and exactly one above it.
const CAP_PROBE_RECORDS: u64 = 3;

/// A fully valid tail whose declared `record_count` sits *exactly* on the
/// size-derived cap. Built with the real encoder: the artifact is legal, and
/// the decoder must accept it.
fn cap_probe_artifact_at_the_cap() -> Bytes {
    let header = TailHeader {
        object_id: OBJ,
        base_snap_seq: 0,
        from_seq: 0,
        to_seq: CAP_PROBE_RECORDS,
    };
    let records: Vec<TailRecord> = (1..=CAP_PROBE_RECORDS)
        .map(|sequence_num| TailRecord::EmptyDelta {
            sequence_num,
            server_ts: TS,
        })
        .collect();
    encode_tail_envelope(&header, records).expect("cap-probe fixture encodes")
}

/// The same artifact size, with `record_count` — and the header range that has
/// to agree with it — raised by exactly one. The bytes present are unchanged,
/// so the claim is the only thing that moved: it now sits one above the cap.
fn cap_probe_artifact_one_above_the_cap() -> Vec<u8> {
    let over = CAP_PROBE_RECORDS + 1;
    let mut bytes = tail_header_bytes(
        OBJ,
        0,
        0,
        over,
        u32::try_from(over).expect("cap probe count fits u32"),
    );
    for sequence_num in 1..=CAP_PROBE_RECORDS {
        bytes.extend_from_slice(&empty_delta_record_prefix(sequence_num));
    }
    bytes
}

// Omission: `adversarial_record_length_not_addressable_on_32_bit` (the
// analogue of OMPF's `OffsetNotAddressable`, `src/domain/ompf.rs`) has no
// representable shape. OMPF's offsets are `u64`, so a 32-bit host genuinely
// cannot address every value the field can hold; the tail codec's
// `payload_len` is a `u32`, and `usize::try_from(u32)` is infallible on every
// target Rust supports (`usize` is at least 32 bits). The decoder still spells
// the conversion as a fallible `usize::try_from` with a `PayloadTooLarge`
// fallback rather than `as usize` — that branch is enumerated by
// `tail_codec_error_payload_too_large_variant_is_enumerated` below, since no
// input can reach it.

// Omission: `adversarial_header_only_truncated_at_each_field_boundary` is
// already covered. The inline
// `decode_truncated_at_every_prefix_length_returns_typed_error_never_panics`
// (`src/checkpoint/tail_codec.rs`) slices a valid artifact at *every* byte
// offset in `0..len`, which includes all 44 header-internal boundaries, and
// asserts a typed `Truncated` / `RecordCountTooLarge` at each. Re-cutting the
// header at its field boundaries here would be a strict subset of that loop.

/// Build a real, valid, **multi-part** OMPF container (delta part plus two
/// external-value parts) to hand to the tail decoder. SR-2.2 is explicit that
/// a tail envelope is NOT an OMPF container, and a checkpoint OMPF is the blob
/// most likely to be mis-routed into this decoder by an adapter or restore
/// path.
///
/// Complements — rather than repeats — the inline
/// `decode_bad_magic_returns_bad_magic_error`, which feeds a *single-part*
/// container (no external values) alongside a zero-filled buffer and a
/// one-byte-off near miss. The multi-part shape is the larger, more
/// structurally elaborate artifact: it has a populated offset table and real
/// part bodies, so a decoder that sniffed structure instead of checking magic
/// first has more to latch onto.
fn adversarial_ompf_container_fed_to_tail_decoder() -> Vec<u8> {
    let external: [&[u8]; 2] = [b"external-value-a", b"external-value-b"];
    let bytes = OmpfContainer::serialize(b"flatbuffers-delta-data", &external)
        .expect("multi-part OMPF fixture serializes");
    // Prove the fixture is a genuinely valid foreign artifact rather than a
    // patched byte string: the OMPF parser must accept it and see all 3 parts.
    let parsed = OmpfContainer::parse(&bytes).expect("multi-part OMPF fixture is valid OMPF");
    assert_eq!(
        parsed.external_values.len(),
        2,
        "fixture is a multi-part container"
    );
    bytes
}

/// Build a real, valid tail envelope — one verbatim record and one empty-delta
/// record — via the actual encoder, to hand to `OmpfContainer::parse`.
///
/// The reverse mis-routing direction. A naive OMPF parser reading
/// `num_parts` out of bytes 4..8 without checking magic would read the tail's
/// `format_version` (`1`) as a part count and then read the tail's
/// `object_id` as an offset — a silent partial parse of a foreign artifact.
fn adversarial_tail_envelope_fed_to_ompf_parser() -> Bytes {
    let header = TailHeader {
        object_id: OBJ,
        base_snap_seq: 40,
        from_seq: 40,
        to_seq: 42,
    };
    let records = vec![
        TailRecord::Verbatim {
            sequence_num: 41,
            server_ts: TS,
            ompf: Bytes::from_static(b"opaque-payload"),
        },
        TailRecord::EmptyDelta {
            sequence_num: 42,
            server_ts: TS + 1,
        },
    ];
    encode_tail_envelope(&header, records).expect("tail fixture encodes")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn adversarial_oom_record_count_header_rejected_before_allocation() {
    let bytes = adversarial_oom_record_count_header();
    assert_eq!(bytes.len(), TAIL_HEADER_SIZE);

    // Reaching the assertions below at all is half the test: a decoder that
    // sized a `Vec` from the claim would have been OOM-killed on the line
    // above the assert, and the test would fail as a process abort rather
    // than as an assertion.
    let err = decode_tail_envelope(&Bytes::from(bytes)).unwrap_err();

    assert!(
        matches!(
            err,
            TailCodecError::RecordCountTooLarge {
                record_count: u32::MAX,
                max: 0,
                size: TAIL_HEADER_SIZE,
            }
        ),
        "expected RecordCountTooLarge DoS reject, got {err:?}"
    );
}

#[test]
fn adversarial_oom_record_count_with_partial_record_bytes_rejected_before_allocation() {
    let bytes = adversarial_oom_record_count_with_partial_record_bytes();
    let size = bytes.len();
    assert_eq!(size, TAIL_HEADER_SIZE + 2 * TAIL_RECORD_PREFIX_SIZE);

    let err = decode_tail_envelope(&Bytes::from(bytes)).unwrap_err();

    // `max` is the record-prefix-strided cap over the bytes *after* the
    // header, not over the whole artifact.
    assert!(
        matches!(
            err,
            TailCodecError::RecordCountTooLarge {
                record_count: u32::MAX,
                max: 2,
                size: reported,
            } if reported == size
        ),
        "expected RecordCountTooLarge with a size-derived cap, got {err:?}"
    );
}

#[test]
fn adversarial_oom_record_length_rejected_before_allocation() {
    let bytes = adversarial_oom_record_length();
    assert_eq!(bytes.len(), TAIL_HEADER_SIZE + TAIL_RECORD_PREFIX_SIZE);

    let err = decode_tail_envelope(&Bytes::from(bytes)).unwrap_err();

    // The claim is bounded against the bytes actually present (zero remain
    // after the prefix), so it is reported as truncation, never allocated.
    assert!(
        matches!(
            err,
            TailCodecError::Truncated {
                at: "record payload",
                needed,
                available: 0,
            } if needed == u32::MAX as usize
        ),
        "expected Truncated at the record payload, got {err:?}"
    );
}

#[test]
fn adversarial_length_prefix_sum_overflow_rejected_at_the_second_record() {
    let bytes = adversarial_length_prefix_sum_overflow();
    let size = bytes.len();
    assert_eq!(size, TAIL_HEADER_SIZE + 2 * TAIL_RECORD_PREFIX_SIZE + 4);

    let err = decode_tail_envelope(&Bytes::from(bytes)).unwrap_err();

    // The first record consumed real bytes, so this is the checked add from a
    // non-zero running offset; `available: 0` proves the decoder measured what
    // remained past the second prefix rather than reusing a stale remainder.
    assert!(
        matches!(
            err,
            TailCodecError::Truncated {
                at: "record payload",
                needed,
                available: 0,
            } if needed == u32::MAX as usize
        ),
        "expected Truncated at the second record's payload, got {err:?}"
    );
}

#[test]
fn adversarial_every_record_claims_max_payload_rejected_at_the_first_record() {
    let bytes = adversarial_every_record_claims_max_payload();
    let size = bytes.len();
    assert_eq!(size, TAIL_HEADER_SIZE + 3 * TAIL_RECORD_PREFIX_SIZE);

    let err = decode_tail_envelope(&Bytes::from(bytes)).unwrap_err();

    // Rejected on the very first record — the ~12 GiB running sum is never
    // formed, because each record is bounded independently as it is read.
    let expected_available = size - (TAIL_HEADER_SIZE + TAIL_RECORD_PREFIX_SIZE);
    assert!(
        matches!(
            err,
            TailCodecError::Truncated {
                at: "record payload",
                needed,
                available,
            } if needed == u32::MAX as usize && available == expected_available
        ),
        "expected Truncated at the first record's payload, got {err:?}"
    );
}

#[test]
fn record_count_exactly_on_the_size_derived_cap_is_accepted() {
    // The guard direction the `u32::MAX` cases above cannot see. Those claim a
    // count astronomically above the cap, so they pass whether the comparison
    // is `>` or `>=` — a guard tightened by one would reject every artifact
    // whose records exactly fill it (the densest legal tail: all empty-delta
    // commits, no payload slack) and make a whole class of real tails
    // permanently unreadable. That is silent data loss dressed as a DoS guard,
    // so the accepting side is pinned first.
    let bytes = cap_probe_artifact_at_the_cap();
    let size = bytes.len();
    assert_eq!(
        size,
        TAIL_HEADER_SIZE + (CAP_PROBE_RECORDS as usize * TAIL_RECORD_PREFIX_SIZE),
        "fixture: the records must exactly fill the artifact, leaving no slack"
    );
    assert_eq!(
        (size - TAIL_HEADER_SIZE) / TAIL_RECORD_PREFIX_SIZE,
        CAP_PROBE_RECORDS as usize,
        "fixture: the declared count must land exactly on the size-derived cap"
    );

    let envelope =
        decode_tail_envelope(&bytes).expect("an artifact exactly on the cap must decode");

    assert_eq!(envelope.len(), CAP_PROBE_RECORDS as usize);
}

#[test]
fn record_count_one_above_the_size_derived_cap_is_rejected_before_allocation() {
    // The rejecting side of the same boundary, one above rather than
    // `u32::MAX` above: a guard written `capacity > max + 1`, or one that
    // measured the cap against the whole artifact instead of the bytes after
    // the header, admits this and then reads a record that is not there.
    let bytes = cap_probe_artifact_one_above_the_cap();
    let size = bytes.len();
    assert_eq!(
        size,
        cap_probe_artifact_at_the_cap().len(),
        "fixture: only the claim moved, not the bytes present"
    );

    let err = decode_tail_envelope(&Bytes::from(bytes)).unwrap_err();

    assert!(
        matches!(
            err,
            TailCodecError::RecordCountTooLarge {
                record_count,
                max,
                size: reported,
            } if u64::from(record_count) == CAP_PROBE_RECORDS + 1
                && max == CAP_PROBE_RECORDS as usize
                && reported == size
        ),
        "expected RecordCountTooLarge one above the cap, got {err:?}"
    );
}

#[test]
fn adversarial_multi_part_ompf_container_rejected_by_tail_decoder() {
    let bytes = adversarial_ompf_container_fed_to_tail_decoder();

    let err = decode_tail_envelope(&Bytes::from(bytes)).unwrap_err();

    // A typed rejection on the magic, not a partial parse: none of the
    // container's offset table or part bodies is interpreted.
    assert!(
        matches!(err, TailCodecError::BadMagic { got } if got == *b"OMPF"),
        "expected BadMagic on a mis-routed OMPF container, got {err:?}"
    );
}

#[test]
fn adversarial_tail_envelope_rejected_by_ompf_parser() {
    let tail = adversarial_tail_envelope_fed_to_ompf_parser();
    // Sanity: the fixture really is a valid tail, so the rejection below is
    // about artifact family and not about a malformed input.
    decode_tail_envelope(&tail).expect("tail fixture decodes as a tail");

    // Matched rather than `expect_err`: `OmpfContainer` is not `Debug`, and an
    // accidental success is worth naming explicitly anyway — it would mean a
    // tail envelope silently partial-parsed as a container.
    let err = match OmpfContainer::parse(&tail) {
        Ok(container) => panic!(
            "a tail envelope parsed as an OMPF container with {} external value(s)",
            container.external_values.len()
        ),
        Err(err) => err,
    };

    let expected_magic = u32::from_le_bytes(TAIL_ENVELOPE_MAGIC);
    assert!(
        matches!(
            err,
            OmpfError::BadMagic { got, .. } if got == expected_magic
        ),
        "expected BadMagic on a mis-routed tail envelope, got {err:?}"
    );

    // The two families also disagree on the empty byte string, which is the
    // one input where a silent mis-route could otherwise look like success:
    // `OmpfContainer::parse` treats it as a valid empty container (SDD §8.4),
    // while the tail codec rejects it (a zero-length tail is a 44-byte
    // header, never zero bytes). The inline
    // `decode_empty_input_is_truncated_not_an_empty_tail` pins the tail-side
    // rule on its own; the point here is the asymmetry between the parsers.
    assert!(
        OmpfContainer::parse(&[]).is_ok(),
        "OMPF accepts empty input as an empty container"
    );
    let err = decode_tail_envelope(&Bytes::new()).unwrap_err();
    assert!(
        matches!(err, TailCodecError::Truncated { at: "magic", .. }),
        "tail decoder must reject empty input, got {err:?}"
    );
}

/// `PayloadTooLarge` is the fallible-conversion fallback the decoder uses
/// instead of `payload_len as usize`, and the encode-side guard for a payload
/// that does not fit the `u32` length field. Neither side is reachable from a
/// test: on decode, `usize::try_from(u32)` is infallible on every target Rust
/// supports; on encode, provoking it would require actually holding a
/// `> 4 GiB` payload in memory.
///
/// So this is a structural check in the same spirit as
/// `ompf_error_serialize_num_parts_overflow_variant_is_enumerated`
/// (`tests/misc/ompf_adversarial_corpus.rs`): pattern-matching the variant
/// proves it still exists with its field shape, so a future refactor that
/// "simplifies" the conversion into a truncating `as usize` cast — and drops
/// the variant — fails to compile this test rather than silently mis-framing
/// a payload onto the wrong bytes.
#[test]
fn tail_codec_error_payload_too_large_variant_is_enumerated() {
    let err = TailCodecError::PayloadTooLarge {
        index: 0,
        len: usize::MAX,
    };
    match err {
        TailCodecError::PayloadTooLarge { index, len } => {
            assert_eq!(index, 0);
            assert!(len > u32::MAX as usize);
        }
        other => unreachable!("constructed PayloadTooLarge, got {other:?}"),
    }
}

/// Scripted-fuzz smoke test: every corpus entry plus a grab-bag of noise must
/// produce a typed error rather than a panic, an abort, or an accidental
/// `Ok`. Mirrors `adversarial_corpus_never_panics` in the OMPF corpus — any
/// drift in decoder behaviour on these known-adversarial shapes fails loudly.
#[test]
fn adversarial_corpus_never_panics() {
    /// Expected outcome for a single corpus entry.
    enum Expect {
        /// Decode must fail with `RecordCountTooLarge` (the allocation guard).
        RecordCountTooLarge,
        /// Decode must fail with `Truncated` (a length claim bounded against
        /// the bytes present).
        Truncated,
        /// Decode must fail with `BadMagic` (a foreign artifact).
        BadMagic,
    }
    // Deliberately no `AnyTypedError` arm. Every shape below — including the
    // pure-noise ones — lands on a specific variant, and an `is_err()`
    // assertion would pass equally on a decoder that had started reporting
    // truncation as bad magic (or vice versa), which is precisely the drift
    // this smoke test exists to catch. Adding a shape whose variant is
    // genuinely undecidable means adding an arm for it here, with a reason.

    let corpus: Vec<(&str, Vec<u8>, Expect)> = vec![
        (
            "oom record_count, bare header",
            adversarial_oom_record_count_header(),
            Expect::RecordCountTooLarge,
        ),
        (
            "oom record_count, partial record bytes",
            adversarial_oom_record_count_with_partial_record_bytes(),
            Expect::RecordCountTooLarge,
        ),
        (
            "record_count one above the size-derived cap",
            cap_probe_artifact_one_above_the_cap(),
            Expect::RecordCountTooLarge,
        ),
        (
            "oom payload_len, no payload",
            adversarial_oom_record_length(),
            Expect::Truncated,
        ),
        (
            "length-prefix sum from a running offset",
            adversarial_length_prefix_sum_overflow(),
            Expect::Truncated,
        ),
        (
            "every record claims u32::MAX",
            adversarial_every_record_claims_max_payload(),
            Expect::Truncated,
        ),
        (
            "mis-routed multi-part OMPF container",
            adversarial_ompf_container_fed_to_tail_decoder(),
            Expect::BadMagic,
        ),
        // Noise: shapes a mis-routed or bit-rotted blob plausibly takes.
        // Both of these are shorter than the 4-byte magic, so they run out of
        // bytes before there is anything to compare — truncation, not a magic
        // mismatch.
        ("empty", vec![], Expect::Truncated),
        ("one zero byte", vec![0x00], Expect::Truncated),
        (
            "saturated bytes",
            vec![0xFF; TAIL_HEADER_SIZE],
            Expect::BadMagic,
        ),
        (
            "zero-filled header-sized region",
            vec![0x00; TAIL_HEADER_SIZE],
            Expect::BadMagic,
        ),
        // The magic matches, so this one gets past the first check and dies on
        // the `format_version` read — a different truncation site than the two
        // sub-magic entries above.
        (
            "magic only",
            TAIL_ENVELOPE_MAGIC.to_vec(),
            Expect::Truncated,
        ),
    ];

    for (label, bytes, expect) in corpus {
        assert!(
            bytes.len() <= MAX_CORPUS_ARTIFACT_BYTES,
            "{label}: corpus artifacts stay tiny — the hazard is a small input claiming a huge allocation"
        );
        let result = decode_tail_envelope(&Bytes::from(bytes));
        match expect {
            Expect::RecordCountTooLarge => assert!(
                matches!(result, Err(TailCodecError::RecordCountTooLarge { .. })),
                "{label}: expected RecordCountTooLarge, got {:?}",
                result.err()
            ),
            Expect::Truncated => assert!(
                matches!(result, Err(TailCodecError::Truncated { .. })),
                "{label}: expected Truncated, got {:?}",
                result.err()
            ),
            Expect::BadMagic => assert!(
                matches!(result, Err(TailCodecError::BadMagic { .. })),
                "{label}: expected BadMagic, got {:?}",
                result.err()
            ),
        }
    }
}
