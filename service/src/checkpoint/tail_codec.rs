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

//! Tail-envelope codec — the on-disk wire format for the stream-replay tail
//! artifact (SR-2.2, SR-6.1, SR-9.2).
//!
//! A *tail* is the immutable slice of a layer's commit history that sits
//! between a layer-state snapshot and some later sequence number. Capture
//! (Epic 3) writes one; replay (Epic 4) reads one and re-applies the commits it
//! carries on top of the snapshot the header names. This module owns the byte
//! format for that artifact and nothing else: it is a pure codec with no I/O,
//! no clock, and no knowledge of any adapter.
//!
//! # Wire format
//!
//! ```text
//! Header (fixed 44 bytes, little-endian throughout):
//!   off  size  field
//!   0    4     magic           = b"LETL"   (4 literal ASCII bytes, not a u32)
//!   4    4     format_version  u32 LE      (currently 1)
//!   8    8     object_id       u64 LE
//!   16   8     base_snap_seq   u64 LE
//!   24   8     from_seq        u64 LE
//!   32   8     to_seq          u64 LE
//!   40   4     record_count    u32 LE
//!
//! Record (fixed 21-byte prefix + payload), repeated record_count times:
//!   off  size  field
//!   0    1     kind            u8   (1 = Verbatim, 2 = EmptyDelta; 0 and >2 invalid)
//!   1    8     sequence_num    u64 LE
//!   9    8     server_ts       i64 LE
//!   17   4     payload_len     u32 LE   (Verbatim: MUST be > 0; EmptyDelta: MUST be 0)
//!   21   ..    payload         payload_len raw inbound OMPF bytes, untouched
//! ```
//!
//! # Format decisions
//!
//! - **Magic `b"LETL"`.** Not required by SR-2.2, added deliberately: SR-2.5's
//!   interim restore-selection invariant exists because handing a tail to
//!   `decode_checkpoint_ompf` (which hard-rejects anything that is not
//!   `is_diff == true && base_version == 0`) produces a confusing `Permanent`.
//!   A distinct magic makes cross-decoder misuse loud and self-describing in a
//!   hexdump. Written as 4 literal ASCII bytes rather than a `u32` so there is
//!   no endianness ambiguity about which byte lands first (`src/domain/ompf.rs`
//!   uses the `u32` form; that is deliberately *not* copied here).
//!
//! - **`format_version` is checked immediately after magic, before any other
//!   structural validation** (SR-6.1). A v2 artifact fed to a v1 reader must
//!   surface [`TailCodecError::UnknownFormatVersion`], never
//!   [`TailCodecError::Truncated`] or a mis-parse. This ordering is
//!   load-bearing, not stylistic.
//!
//! - **Explicit record-kind flag, never byte-shape inference.** SR-1.2 is
//!   explicit that empty commits are classified *by invariant, not by byte
//!   shape*, because the byte shapes alias (the `RELAY_OMPF_FIELD` absence
//!   list, `src/storage/notifications.rs`). The codec carries that
//!   classification forward as a first-class flag and forbids the two kinds
//!   from aliasing: a [`TailRecord::Verbatim`] with a zero-length payload is
//!   rejected on encode and on decode. Kind `0` is deliberately invalid so a
//!   zero-filled region cannot decode as a valid record.
//!
//! - **Canonical encoding — exactly one byte sequence per logical envelope.**
//!   Fixed field order, no optional fields, no padding, no maps, no wall-clock
//!   reads. This is SR-9.2's duplicate-leader safety property: two replicas
//!   capturing the same immutable stream slice at the same
//!   `(object_id, to_seq)` must produce byte-identical artifacts so the racing
//!   PUTs are idempotent.
//!
//! - **A zero-length tail is a 44-byte header with `record_count == 0` and
//!   `from_seq == to_seq`.** It is NOT an empty byte string. Contrast
//!   `OmpfContainer::parse` (`src/domain/ompf.rs`), which accepts empty input
//!   as a valid empty container — that precedent must not be copied here, and
//!   empty input decodes as [`TailCodecError::Truncated`].
//!
//! - **No CRC / checksum.** Adapters already write atomically (LocalDisk
//!   write-temp+rename; StorageApi by-address PUT), and every structural
//!   invariant enumerated by [`TailCodecError`] is self-checking. Recorded as a
//!   considered decision so a later reviewer does not re-litigate it silently.
//!
//! - **No `reserved` / `flags` field.** `format_version` is the
//!   forward-compatibility mechanism; an unread reserved field would be a
//!   second, unversioned one.
//!
//! # Determinism contract
//!
//! The encoder's output must depend on nothing but its inputs (SR-9.2), and the
//! decoder must be a total function of its input bytes. That is reviewable by
//! inspection because this module may not contain any of the following:
//!
//! - `HashMap` / `HashSet`, or any other collection with nondeterministic
//!   iteration order (`BTreeMap` / `Vec` only).
//! - `std::time`, `Instant`, `SystemTime`, or any other clock read. In
//!   particular `now_unix_ms()` (`src/checkpoint/mod.rs`) is exactly the kind of
//!   call that must never appear here — record timestamps are carried in from
//!   the captured stream entries, never read from the clock.
//! - Randomness of any kind.
//! - `async` functions or `await` points.
//! - `fred` (no Valkey), `cxx` / `oodf_ffi` (no FlatBuffers), or
//!   `OmpfContainer` (payload bytes stay opaque).
//! - `CheckpointStore` or any adapter type. [`CheckpointError`] is imported —
//!   it is the leaf error vocabulary declared alongside that trait, not an
//!   adapter type — solely for the classification impl at the bottom of this
//!   file.

use bytes::Bytes;
use thiserror::Error;

use crate::checkpoint::CheckpointError;

/// Envelope magic, `b"LETL"` ("Live Edit Tail"). Four literal ASCII bytes; see
/// the module doc for why this is not a `u32`.
pub const TAIL_ENVELOPE_MAGIC: [u8; 4] = *b"LETL";

/// Wire-format version stamped by [`TAIL_ENVELOPE_MAGIC`]-bearing artifacts
/// this build writes, and the only version this build can read.
pub const TAIL_FORMAT_VERSION: u32 = 1;

/// Fixed header size in bytes.
pub const TAIL_HEADER_SIZE: usize = 44;

/// Fixed per-record prefix size in bytes (everything before the payload).
pub const TAIL_RECORD_PREFIX_SIZE: usize = 21;

/// Byte offset of `format_version` in the header. Public so tests can forge an
/// unknown-version artifact by patching one field without hardcoding an offset.
pub const TAIL_FORMAT_VERSION_OFFSET: usize = 4;

/// Byte offset of `record_count`; the encoder backfills it here. Public so
/// tests can forge a record-count mismatch by patching one field without
/// hardcoding an offset.
pub const TAIL_RECORD_COUNT_OFFSET: usize = 40;

/// 2^53 - 1. Mirrors `F64_SAFE_INT_MAX` (`src/checkpoint/apply.rs`) and
/// `SEQUENCE_CEILING` (`src/domain/delta.rs`), duplicated rather than imported:
/// this codec is the lowest layer of the checkpoint subsystem and must not take
/// a dependency edge on the apply/restore layer above it.
pub const TAIL_MAX_SEQUENCE: u64 = 9_007_199_254_740_991;

/// Discriminant for the two record shapes a tail can carry.
///
/// The mapping to the on-wire `u8` is part of the format and must not change
/// without a [`TAIL_FORMAT_VERSION`] bump. `0` is reserved-invalid so a
/// zero-filled region cannot decode as a valid record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailRecordKind {
    /// A commit that carried OMPF bytes.
    Verbatim,
    /// A commit that advanced the sequence without changing state.
    EmptyDelta,
}

impl TailRecordKind {
    /// On-wire discriminant: `Verbatim` => 1, `EmptyDelta` => 2.
    pub fn as_u8(self) -> u8 {
        match self {
            TailRecordKind::Verbatim => 1,
            TailRecordKind::EmptyDelta => 2,
        }
    }

    /// Inverse of [`TailRecordKind::as_u8`]. `0` and anything above `2` are
    /// invalid and yield `None`.
    pub fn from_u8(raw: u8) -> Option<Self> {
        match raw {
            1 => Some(TailRecordKind::Verbatim),
            2 => Some(TailRecordKind::EmptyDelta),
            _ => None,
        }
    }
}

/// Logical lineage identity carried in-band in the envelope header.
///
/// # Lockstep with `CheckpointMeta` — a cross-module invariant
///
/// Field names and semantics deliberately match the lineage fields on
/// `CheckpointMeta` in **`src/checkpoint/store.rs`** (`kind`, `base_snap_seq`,
/// `from_seq`, `snap_seq`). Both exist on purpose: meta serves listing and
/// retention, this header is authoritative at replay time. This module does NOT
/// import `CheckpointMeta` — named in prose rather than as an intra-doc link so
/// neither module becomes a build-time dependency of the other's docs — which is
/// what let Task 1 and Task 2 land in either order.
///
/// The invariant lives in neither module alone, so it is recorded on **both**
/// sides (the shape SDD §22.11 uses for the `SessionCheckpoint` /
/// `SessionStore::create_once` coupling). `CheckpointMeta`'s doc carries the
/// reciprocal block, including the full mapping table. Changing one side's field
/// names or semantics without the other is a contract violation, not a refactor.
///
/// The correspondence, field by field:
///
/// - `base_snap_seq` and `from_seq` are spelled identically on both sides; here
///   they are plain `u64` (a tail always has a lineage), on `CheckpointMeta` they
///   are `Option<u64>` and are `Some` exactly when `kind` is `Tail`.
/// - **`to_seq` is the one name that differs.** It has no counterpart field on
///   `CheckpointMeta`: the number lands in that struct's `snap_seq`, and in
///   `CheckpointHandle::snap_seq` on the handle, so a tail and a base address and
///   sort through one field.
/// - `object_id` corresponds to `CheckpointHandle::object_id`, not to anything on
///   the meta.
/// - `kind` has **no** counterpart here, deliberately: an artifact carrying the
///   `LETL` magic *is* a tail, so a header-level kind byte would be a second,
///   redundant discriminator that could disagree with the magic.
///
/// This header carries **no session identity** (`session_id` / `layer_name`) —
/// also deliberate. Identity at replay time is base-sourced, read off the base
/// generation's meta; see `CheckpointMeta`'s lockstep block for why an
/// identity-empty tail meta is nonetheless forbidden.
///
/// `format_version` is NOT a field here: the encoder always stamps
/// [`TAIL_FORMAT_VERSION`], which structurally prevents a caller from writing an
/// artifact at a version this build cannot read. The decoded version is exposed
/// on [`TailEnvelope::format_version`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailHeader {
    /// The layer generation this tail belongs to.
    pub object_id: u64,
    /// Sequence number of the snapshot this tail is layered on top of.
    pub base_snap_seq: u64,
    /// Exclusive lower bound of the tail's `(from_seq, to_seq]` span — the
    /// sequence of the newest commit NOT carried; the first carried commit is
    /// `from_seq + 1`. Matches [`CheckpointMeta::from_seq`]'s semantics and the
    /// encoder's `expected = from_seq + 1 + count` contiguity check.
    ///
    /// [`CheckpointMeta::from_seq`]: crate::checkpoint::store::CheckpointMeta::from_seq
    pub from_seq: u64,
    /// Sequence number of the last commit carried by this tail.
    pub to_seq: u64,
}

/// One captured commit.
#[derive(Clone, PartialEq, Eq)]
pub enum TailRecord {
    /// A commit that carried OMPF bytes. `ompf` is the writer's raw inbound
    /// container, stored with no re-encode and no parse-and-rebuild (SR-2.2).
    /// This module never parses it — `oodf_ffi` and `OmpfContainer` are not
    /// imported here.
    Verbatim {
        /// The commit's sequence number.
        sequence_num: u64,
        /// Server-assigned microsecond timestamp from the stream entry.
        server_ts: i64,
        /// Raw inbound OMPF bytes, opaque to this module. Never empty.
        ompf: Bytes,
    },
    /// The §8.4 empty-commit shape: advanced the sequence without changing
    /// state. Replay applies `apply_delta(state, None)` for these (SR-3.2).
    EmptyDelta {
        /// The commit's sequence number.
        sequence_num: u64,
        /// Server-assigned microsecond timestamp from the stream entry.
        server_ts: i64,
    },
}

impl TailRecord {
    /// This record's on-wire kind.
    pub fn kind(&self) -> TailRecordKind {
        match self {
            TailRecord::Verbatim { .. } => TailRecordKind::Verbatim,
            TailRecord::EmptyDelta { .. } => TailRecordKind::EmptyDelta,
        }
    }

    /// The commit sequence number this record carries.
    pub fn sequence_num(&self) -> u64 {
        match self {
            TailRecord::Verbatim { sequence_num, .. }
            | TailRecord::EmptyDelta { sequence_num, .. } => *sequence_num,
        }
    }

    /// The server-assigned microsecond timestamp this record carries.
    ///
    /// `i64`, not `u64`, matching `LayerNotification::Update { server_ts, .. }`
    /// (`src/storage/notifications.rs`) so capture → replay round-trips the
    /// stream entry's value losslessly.
    pub fn server_ts(&self) -> i64 {
        match self {
            TailRecord::Verbatim { server_ts, .. } | TailRecord::EmptyDelta { server_ts, .. } => {
                *server_ts
            }
        }
    }

    /// The raw inbound OMPF bytes, or `None` for an empty-delta record.
    pub fn ompf(&self) -> Option<&Bytes> {
        match self {
            TailRecord::Verbatim { ompf, .. } => Some(ompf),
            TailRecord::EmptyDelta { .. } => None,
        }
    }
}

/// Manual `Debug` so a verbatim payload renders as a byte count, not its bytes.
///
/// Operational reason: `ompf` is the writer's raw inbound container, capped only
/// by `LIVE_EDIT_MAX_LAYER_SIZE` (100 MB default). The derived `Debug` for
/// `Bytes` escapes every byte, so a 10 MB payload becomes a ~40 MB single log
/// line — and [`TailEnvelope`]'s derived `Debug` multiplies that by the record
/// count. Any `{:?}` on a record or envelope (an error path, a `tracing` field,
/// a test panic message) would then flood the log pipeline. Bounding it here
/// keeps [`TailEnvelope`]'s derive safe to keep.
///
/// `EmptyDelta` carries no payload, so its rendering stays identical to the
/// derived form.
impl std::fmt::Debug for TailRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// Renders as `<N bytes>` — unquoted, so it reads as a summary rather
        /// than as a value that could be mistaken for the payload itself.
        struct ByteLen(usize);

        impl std::fmt::Debug for ByteLen {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "<{} bytes>", self.0)
            }
        }

        match self {
            TailRecord::Verbatim {
                sequence_num,
                server_ts,
                ompf,
            } => f
                .debug_struct("Verbatim")
                .field("sequence_num", sequence_num)
                .field("server_ts", server_ts)
                .field("ompf", &ByteLen(ompf.len()))
                .finish(),
            TailRecord::EmptyDelta {
                sequence_num,
                server_ts,
            } => f
                .debug_struct("EmptyDelta")
                .field("sequence_num", sequence_num)
                .field("server_ts", server_ts)
                .finish(),
        }
    }
}

/// Decoded, fully-validated envelope.
///
/// Fields are private and constructible only via the decoder, so an envelope in
/// hand always satisfies every structural invariant (SR-3.5 forbids partial
/// application; validating up-front is what makes that possible).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailEnvelope {
    format_version: u32,
    header: TailHeader,
    records: Vec<TailRecord>,
}

impl TailEnvelope {
    /// Module-private constructor. [`decode_tail_envelope`] is the only caller,
    /// which is what makes "an envelope in hand is fully validated" a type-level
    /// guarantee rather than a convention (SR-3.5).
    fn new(format_version: u32, header: TailHeader, records: Vec<TailRecord>) -> Self {
        TailEnvelope {
            format_version,
            header,
            records,
        }
    }

    /// The `format_version` read from the artifact. Always
    /// [`TAIL_FORMAT_VERSION`] for anything this build can decode — exposed so
    /// callers can log the version they acted on without re-parsing bytes.
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// The lineage header.
    pub fn header(&self) -> &TailHeader {
        &self.header
    }

    /// The carried commits, in ascending, contiguous sequence order.
    pub fn records(&self) -> &[TailRecord] {
        &self.records
    }

    /// Borrowing iterator over [`TailEnvelope::records`].
    pub fn iter(&self) -> std::slice::Iter<'_, TailRecord> {
        self.records.iter()
    }

    /// Consume the envelope, yielding the carried commits.
    pub fn into_records(self) -> Vec<TailRecord> {
        self.records
    }

    /// Number of carried commits.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True for a zero-length tail (a 44-byte header with `record_count == 0`).
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}

/// Encode a tail envelope. Byte-deterministic: identical `header` + identical
/// record sequence always produce identical bytes (SR-9.2), which is what makes
/// two replicas racing on the same `(object_id, to_seq)` write idempotent rather
/// than divergent artifacts.
///
/// Always stamps [`TAIL_FORMAT_VERSION`] — callers cannot select a version, so
/// this build can never mint an artifact it cannot read back.
///
/// The header's claimed range and the supplied records are cross-checked: an
/// envelope whose `(from_seq, to_seq]` span does not match exactly one
/// contiguous record per sequence number is rejected, never emitted. Claiming
/// coverage the records do not supply is the b.pw6 defect (SR-1.4) and is
/// structurally unrepresentable here.
///
/// Takes a plain [`IntoIterator`] rather than an [`ExactSizeIterator`]: the
/// `record_count` field sits at a fixed header offset, so the encoder writes a
/// placeholder, streams the records, and backfills the final count. Callers may
/// therefore stream records straight out of an adapter chain without collecting
/// them first, and no caller-supplied size hint can influence the output bytes.
pub fn encode_tail_envelope<I>(header: &TailHeader, records: I) -> Result<Bytes, TailCodecError>
where
    I: IntoIterator<Item = TailRecord>,
{
    if header.from_seq > header.to_seq {
        return Err(TailCodecError::InvalidRange {
            from_seq: header.from_seq,
            to_seq: header.to_seq,
        });
    }
    // A tail layers onto the snapshot it names: the first tail in a lineage has
    // `from_seq == base_snap_seq`, and each later tail has
    // `from_seq == previous.to_seq` (SR-2.1). Either way the base never sits
    // above the covered range.
    if header.base_snap_seq > header.from_seq {
        return Err(TailCodecError::BaseAfterRange {
            base_snap_seq: header.base_snap_seq,
            from_seq: header.from_seq,
        });
    }
    if header.to_seq > TAIL_MAX_SEQUENCE {
        return Err(TailCodecError::SequenceCeilingExceeded {
            sequence_num: header.to_seq,
            max: TAIL_MAX_SEQUENCE,
        });
    }

    // Capacity for the header only. Sizing from a caller-supplied hint would
    // put a value that cannot affect the output bytes within reach of doing so;
    // `Vec` growth is invisible to the encoded result.
    let mut out = Vec::with_capacity(TAIL_HEADER_SIZE);
    out.extend_from_slice(&TAIL_ENVELOPE_MAGIC);
    out.extend_from_slice(&TAIL_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&header.object_id.to_le_bytes());
    out.extend_from_slice(&header.base_snap_seq.to_le_bytes());
    out.extend_from_slice(&header.from_seq.to_le_bytes());
    out.extend_from_slice(&header.to_seq.to_le_bytes());
    // `record_count` placeholder, backfilled at TAIL_RECORD_COUNT_OFFSET below.
    out.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(out.len(), TAIL_HEADER_SIZE);

    let mut count: u64 = 0;
    for (index, record) in records.into_iter().enumerate() {
        // The envelope covers `(from_seq, to_seq]` with exactly one record per
        // sequence number — no gaps, no repeats (SR-1.2 contiguity).
        let expected = header.from_seq + 1 + count;
        let got = record.sequence_num();
        if got != expected {
            return Err(TailCodecError::NonContiguousSequence {
                index,
                expected,
                got,
            });
        }
        write_tail_record(&mut out, index, &record)?;
        count += 1;
    }

    let range_span = header.to_seq - header.from_seq;
    if count != range_span {
        return Err(TailCodecError::RecordCountMismatch {
            range_span,
            record_count: count,
        });
    }
    // Unreachable in practice: reaching a count above `u32::MAX` means the loop
    // above buffered more than 4 billion records at TAIL_RECORD_PREFIX_SIZE
    // bytes apiece (>85 GiB), so the allocation fails long first. Written as a
    // fallible conversion anyway because the alternative — `as u32` — would
    // silently truncate and mint a header that mis-frames every record on
    // decode. `RecordCountMismatch` is reused rather than minting a variant for
    // an unreachable state; its two fields are equal here (the mismatch check
    // just above passed), which only ever shows up in a log line that cannot be
    // emitted.
    let record_count = u32::try_from(count).map_err(|_| TailCodecError::RecordCountMismatch {
        range_span,
        record_count: count,
    })?;
    out[TAIL_RECORD_COUNT_OFFSET..TAIL_RECORD_COUNT_OFFSET + 4]
        .copy_from_slice(&record_count.to_le_bytes());

    Ok(Bytes::from(out))
}

/// Append one record's fixed [`TAIL_RECORD_PREFIX_SIZE`]-byte prefix followed by
/// its payload. `index` is carried only to name the offending record in errors.
///
/// On an error return `out` is left partially written; the caller drops the
/// whole buffer rather than returning it, so no truncated artifact escapes.
fn write_tail_record(
    out: &mut Vec<u8>,
    index: usize,
    record: &TailRecord,
) -> Result<(), TailCodecError> {
    // A zero-length verbatim payload would leave the two kinds distinguishable
    // only by the kind flag — exactly the aliasing SR-1.2 forbids, since a
    // commit carrying no OMPF bytes is an `EmptyDelta` by construction.
    let payload: &[u8] = match record.ompf() {
        Some(ompf) if ompf.is_empty() => {
            return Err(TailCodecError::EmptyVerbatimPayload { index })
        }
        Some(ompf) => ompf,
        None => &[],
    };
    // Explicit conversion, not `as u32`: a silent truncation here would emit a
    // record header whose declared length does not match its payload.
    let payload_len =
        u32::try_from(payload.len()).map_err(|_| TailCodecError::PayloadTooLarge {
            index,
            len: payload.len(),
        })?;

    out.reserve(TAIL_RECORD_PREFIX_SIZE + payload.len());
    out.push(record.kind().as_u8());
    out.extend_from_slice(&record.sequence_num().to_le_bytes());
    out.extend_from_slice(&record.server_ts().to_le_bytes());
    // Always present, so every record shares the same prefix stride;
    // `EmptyDelta` writes a zero here and no payload bytes.
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(payload);
    Ok(())
}

/// Parse only the fixed [`TAIL_HEADER_SIZE`]-byte header, returning the
/// artifact's `format_version` alongside its lineage header.
///
/// Reads at most the first [`TAIL_HEADER_SIZE`] bytes and allocates nothing, so
/// lineage gather (SR-3.1) and adapter-side listing can inspect a multi-MB tail's
/// identity without materializing its records. Takes `&[u8]` rather than
/// `&Bytes` because it only copies scalars out; a `&Bytes` coerces via `Deref`.
///
/// Every check the encoder applies to a header is re-applied here, and their
/// order is load-bearing: `format_version` is validated
/// immediately after the magic and *before* any length or semantic check, so a
/// future v2 artifact — whose header may be a different length entirely —
/// surfaces [`TailCodecError::UnknownFormatVersion`] rather than
/// [`TailCodecError::Truncated`] or a mis-parse (SR-6.1). A decoder that
/// accepted what the encoder refuses to emit would let a hand-forged or
/// bit-rotted artifact into replay, so the semantic checks are duplicated rather
/// than assumed.
///
/// Empty input is [`TailCodecError::Truncated`], never a valid empty envelope —
/// the `OmpfContainer::parse` precedent (`src/domain/ompf.rs`) is deliberately
/// not copied. A zero-length tail is a 44-byte header with `record_count == 0`.
pub fn peek_tail_header(bytes: &[u8]) -> Result<(u32, TailHeader), TailCodecError> {
    let (format_version, header, _record_count) = read_tail_header(bytes)?;
    Ok((format_version, header))
}

/// Fully decode and validate a tail envelope.
///
/// Eager and total: every structural defect surfaces here, before the caller's
/// first `apply_delta`. That is what makes SR-3.5's "no partial application in
/// any failure mode" achievable — a [`TailEnvelope`] in hand is already
/// known-good, so replay cannot begin and then abort mid-way on a malformed
/// record. A lazy record-by-record reader would defer errors into the middle of
/// a replay, which is exactly the shape the spec forbids, and is deliberately
/// not offered.
///
/// Zero-copy: [`TailRecord::Verbatim`] payloads are [`Bytes::slice`] views into
/// `bytes`, never re-allocated and never re-encoded. SR-2.2's "no re-encode, no
/// parse-and-rebuild" binds the read side as much as the write side, and a
/// multi-MB tail must not be copied once per record. This is why the parameter
/// is `&Bytes` and not `&[u8]`.
///
/// # Size policy
///
/// This codec imposes **no size cap of its own and takes no config knob**: every
/// allocation it makes is bounded by `bytes.len()`, via the `record_count`
/// allocation guard below. SR-1.4's `LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES` /
/// `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES` caps are capture-path policy and belong
/// to the writer, not here — a policy cap in the decoder would make an artifact
/// that was legal to write illegal to read back after a config change.
pub fn decode_tail_envelope(bytes: &Bytes) -> Result<TailEnvelope, TailCodecError> {
    let (format_version, header, record_count) = read_tail_header(bytes)?;

    // Bound `record_count` by the input size before allocating. A forged count
    // of `u32::MAX` on a 44-byte artifact would otherwise drive a multi-GB
    // `Vec::with_capacity`. Mirrors `OmpfError::NumPartsTooLarge`
    // (`src/domain/ompf.rs`): each record costs at least its fixed prefix, so
    // `remaining / TAIL_RECORD_PREFIX_SIZE` is a hard cap.
    let remaining = bytes.len().saturating_sub(TAIL_HEADER_SIZE);
    let max = remaining / TAIL_RECORD_PREFIX_SIZE;
    let too_large = || TailCodecError::RecordCountTooLarge {
        record_count,
        max,
        size: bytes.len(),
    };
    // `usize::try_from`, not `as usize`: on a hypothetical narrower target the
    // cast would wrap a forged count down into the accepted range.
    let capacity = usize::try_from(record_count).map_err(|_| too_large())?;
    if capacity > max {
        return Err(too_large());
    }

    let mut records = Vec::with_capacity(capacity);
    let mut offset = TAIL_HEADER_SIZE;
    for index in 0..capacity {
        let (record, next) = read_tail_record(bytes, offset, index, header.from_seq)?;
        records.push(record);
        offset = next;
    }

    // Canonical form admits no slack: trailing bytes mean these are not the
    // bytes the encoder would have produced for this envelope.
    let extra = bytes.len().saturating_sub(offset);
    if extra > 0 {
        return Err(TailCodecError::TrailingBytes { extra });
    }

    Ok(TailEnvelope::new(format_version, header, records))
}

/// Shared header parse behind [`peek_tail_header`] and [`decode_tail_envelope`].
///
/// Also yields the on-wire `record_count`, which is not a [`TailHeader`] field:
/// it is a framing detail the decoder needs and a peeking caller does not.
fn read_tail_header(bytes: &[u8]) -> Result<(u32, TailHeader, u32), TailCodecError> {
    let magic = bytes
        .get(0..TAIL_ENVELOPE_MAGIC.len())
        .and_then(|s| <[u8; 4]>::try_from(s).ok())
        .ok_or(TailCodecError::Truncated {
            at: "magic",
            needed: TAIL_ENVELOPE_MAGIC.len(),
            available: bytes.len(),
        })?;
    if magic != TAIL_ENVELOPE_MAGIC {
        return Err(TailCodecError::BadMagic { got: magic });
    }

    // Version before every remaining check (SR-6.1) — see the ordering note on
    // `peek_tail_header`.
    let format_version =
        le_u32(bytes, TAIL_FORMAT_VERSION_OFFSET).ok_or(TailCodecError::Truncated {
            at: "format_version",
            needed: TAIL_FORMAT_VERSION_OFFSET + 4,
            available: bytes.len(),
        })?;
    if format_version != TAIL_FORMAT_VERSION {
        return Err(TailCodecError::UnknownFormatVersion {
            got: format_version,
            supported: TAIL_FORMAT_VERSION,
        });
    }

    if bytes.len() < TAIL_HEADER_SIZE {
        return Err(TailCodecError::Truncated {
            at: "header",
            needed: TAIL_HEADER_SIZE,
            available: bytes.len(),
        });
    }
    // The length check above makes every read below in-bounds; the `ok_or` is a
    // total-function fallback, not a reachable branch.
    let short = || TailCodecError::Truncated {
        at: "header",
        needed: TAIL_HEADER_SIZE,
        available: bytes.len(),
    };
    let object_id = le_u64(bytes, 8).ok_or_else(short)?;
    let base_snap_seq = le_u64(bytes, 16).ok_or_else(short)?;
    let from_seq = le_u64(bytes, 24).ok_or_else(short)?;
    let to_seq = le_u64(bytes, 32).ok_or_else(short)?;
    let record_count = le_u32(bytes, TAIL_RECORD_COUNT_OFFSET).ok_or_else(short)?;

    // Mirrors `encode_tail_envelope` exactly. Accepting a header the encoder
    // would refuse to emit is how a hand-forged or bit-rotted artifact reaches
    // replay.
    if from_seq > to_seq {
        return Err(TailCodecError::InvalidRange { from_seq, to_seq });
    }
    if base_snap_seq > from_seq {
        return Err(TailCodecError::BaseAfterRange {
            base_snap_seq,
            from_seq,
        });
    }
    if to_seq > TAIL_MAX_SEQUENCE {
        return Err(TailCodecError::SequenceCeilingExceeded {
            sequence_num: to_seq,
            max: TAIL_MAX_SEQUENCE,
        });
    }

    // The header must not claim coverage its framing cannot supply — the b.pw6
    // defect class (SR-1.4), rejected symmetrically on both sides of the codec.
    let range_span = to_seq - from_seq;
    let framed = u64::from(record_count);
    if framed != range_span {
        return Err(TailCodecError::RecordCountMismatch {
            range_span,
            record_count: framed,
        });
    }

    Ok((
        format_version,
        TailHeader {
            object_id,
            base_snap_seq,
            from_seq,
            to_seq,
        },
        record_count,
    ))
}

/// Read one record starting at `offset`, returning it alongside the offset of
/// the next record. `index` names the offending record in errors; `from_seq` is
/// the header's, used to derive the sequence number this position must carry.
fn read_tail_record(
    bytes: &Bytes,
    offset: usize,
    index: usize,
    from_seq: u64,
) -> Result<(TailRecord, usize), TailCodecError> {
    let prefix_end =
        offset
            .checked_add(TAIL_RECORD_PREFIX_SIZE)
            .ok_or(TailCodecError::Truncated {
                at: "record prefix",
                needed: TAIL_RECORD_PREFIX_SIZE,
                available: bytes.len().saturating_sub(offset),
            })?;
    let prefix = bytes
        .get(offset..prefix_end)
        .ok_or(TailCodecError::Truncated {
            at: "record prefix",
            needed: TAIL_RECORD_PREFIX_SIZE,
            available: bytes.len().saturating_sub(offset),
        })?;

    // `prefix` is exactly TAIL_RECORD_PREFIX_SIZE bytes, so every read below is
    // in-bounds; the fallbacks are total-function plumbing, not live branches.
    let short = || TailCodecError::Truncated {
        at: "record prefix",
        needed: TAIL_RECORD_PREFIX_SIZE,
        available: bytes.len().saturating_sub(offset),
    };
    let kind_raw = prefix.first().copied().ok_or_else(short)?;
    // Unknown kinds are rejected, never skipped: skipping would silently
    // shorten the replay while the header still claims full coverage (b.pw6).
    let kind = TailRecordKind::from_u8(kind_raw).ok_or(TailCodecError::UnknownRecordKind {
        index,
        got: kind_raw,
    })?;
    let sequence_num = le_u64(prefix, 1).ok_or_else(short)?;
    let server_ts = le_i64(prefix, 9).ok_or_else(short)?;
    let payload_len = le_u32(prefix, 17).ok_or_else(short)?;

    // Records cover `(from_seq, to_seq]` with exactly one entry per sequence
    // number. `from_seq + 1 + index <= to_seq <= TAIL_MAX_SEQUENCE` given the
    // validated header, so the saturating adds cannot saturate; they are used
    // over `+` only so a future caller cannot turn this into a debug panic.
    let expected = from_seq
        .saturating_add(1)
        .saturating_add(u64::try_from(index).unwrap_or(u64::MAX));
    if sequence_num != expected {
        return Err(TailCodecError::NonContiguousSequence {
            index,
            expected,
            got: sequence_num,
        });
    }

    // The two kinds may never alias: a zero-length `Verbatim` and a non-empty
    // `EmptyDelta` are both a second byte sequence for a logical record that
    // already has one, so both are rejected (SR-1.2, canonical encoding).
    match kind {
        TailRecordKind::Verbatim if payload_len == 0 => {
            return Err(TailCodecError::EmptyVerbatimPayload { index })
        }
        TailRecordKind::EmptyDelta if payload_len != 0 => {
            return Err(TailCodecError::NonEmptyEmptyDeltaPayload {
                index,
                len: payload_len,
            })
        }
        _ => {}
    }

    // `usize::try_from` + `checked_add`, never `as`: a truncating cast here
    // would frame a payload onto the wrong bytes rather than reporting a
    // malformed artifact.
    // Unreachable wherever `usize` is at least 32 bits; `usize::MAX` is the
    // honest reported length on a target where it is not — "longer than this
    // platform can address".
    let payload_len_usize =
        usize::try_from(payload_len).map_err(|_| TailCodecError::PayloadTooLarge {
            index,
            len: usize::MAX,
        })?;
    let payload_end =
        prefix_end
            .checked_add(payload_len_usize)
            .ok_or(TailCodecError::Truncated {
                at: "record payload",
                needed: payload_len_usize,
                available: bytes.len().saturating_sub(prefix_end),
            })?;
    if bytes.len() < payload_end {
        return Err(TailCodecError::Truncated {
            at: "record payload",
            needed: payload_len_usize,
            available: bytes.len().saturating_sub(prefix_end),
        });
    }

    let record = match kind {
        // `Bytes::slice` — a refcounted view into the caller's buffer, no copy
        // and no re-encode (SR-2.2).
        TailRecordKind::Verbatim => TailRecord::Verbatim {
            sequence_num,
            server_ts,
            ompf: bytes.slice(prefix_end..payload_end),
        },
        TailRecordKind::EmptyDelta => TailRecord::EmptyDelta {
            sequence_num,
            server_ts,
        },
    };
    Ok((record, payload_end))
}

/// Little-endian `u32` at `at`, or `None` when `buf` is too short.
fn le_u32(buf: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    let raw = <[u8; 4]>::try_from(buf.get(at..end)?).ok()?;
    Some(u32::from_le_bytes(raw))
}

/// Little-endian `u64` at `at`, or `None` when `buf` is too short.
fn le_u64(buf: &[u8], at: usize) -> Option<u64> {
    let end = at.checked_add(8)?;
    let raw = <[u8; 8]>::try_from(buf.get(at..end)?).ok()?;
    Some(u64::from_le_bytes(raw))
}

/// Little-endian `i64` at `at`, or `None` when `buf` is too short. Signed to
/// match `LayerNotification::Update { server_ts, .. }`.
fn le_i64(buf: &[u8], at: usize) -> Option<i64> {
    let end = at.checked_add(8)?;
    let raw = <[u8; 8]>::try_from(buf.get(at..end)?).ok()?;
    Some(i64::from_le_bytes(raw))
}

/// Every way a tail artifact can be malformed.
///
/// Variants carry enough context to log without the caller adding more, and
/// every field is a `Copy` scalar or array — no `Bytes`, no `String` payload
/// copies — so the enum is `Send + Sync + 'static` and cheap to construct on
/// the failure path.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TailCodecError {
    /// The leading 4 bytes are not [`TAIL_ENVELOPE_MAGIC`]. Most likely a blob
    /// from a different codec (e.g. a checkpoint OMPF) handed to this decoder.
    #[error("bad tail magic: expected {expected:?}, got {got:?}", expected = TAIL_ENVELOPE_MAGIC)]
    BadMagic {
        /// The 4 bytes actually found at offset 0.
        got: [u8; 4],
    },

    /// The artifact was written at a format version this build cannot read.
    /// Checked immediately after magic, before any other structural validation.
    #[error("unsupported tail format version {got} (this build supports {supported})")]
    UnknownFormatVersion {
        /// Version read from the artifact.
        got: u32,
        /// Version this build supports ([`TAIL_FORMAT_VERSION`]).
        supported: u32,
    },

    /// The input ended before a structurally required field or payload.
    #[error("truncated tail at {at}: needed {needed} bytes, available {available}")]
    Truncated {
        /// Which structural element ran out of bytes (e.g. `"header"`).
        at: &'static str,
        /// Bytes required at this point.
        needed: usize,
        /// Bytes actually remaining.
        available: usize,
    },

    /// Bytes remain after the declared `record_count` records were consumed.
    /// Rejected rather than ignored: the encoding is canonical, so trailing
    /// bytes mean the artifact is not the one the encoder would have produced.
    #[error("trailing bytes after last tail record: {extra} extra bytes")]
    TrailingBytes {
        /// Number of unconsumed bytes.
        extra: usize,
    },

    /// A record's kind byte is neither `1` nor `2`.
    #[error("tail record {index}: unknown record kind {got}")]
    UnknownRecordKind {
        /// Zero-based record index.
        index: usize,
        /// The kind byte actually found.
        got: u8,
    },

    /// A `Verbatim` record declared a zero-length payload — the shape that
    /// would alias an `EmptyDelta`, which the explicit kind flag exists to
    /// prevent.
    #[error("tail record {index}: verbatim record has an empty payload")]
    EmptyVerbatimPayload {
        /// Zero-based record index.
        index: usize,
    },

    /// An `EmptyDelta` record declared a non-zero payload length.
    #[error("tail record {index}: empty-delta record has a {len}-byte payload")]
    NonEmptyEmptyDeltaPayload {
        /// Zero-based record index.
        index: usize,
        /// The declared payload length.
        len: u32,
    },

    /// `record_count` implies more records than the input could possibly hold.
    /// Checked before any allocation so a forged count cannot drive a large
    /// `Vec::with_capacity`.
    #[error("tail record_count {record_count} exceeds maximum {max} implied by size {size}")]
    RecordCountTooLarge {
        /// Declared record count.
        record_count: u32,
        /// Maximum records the remaining bytes could hold.
        max: usize,
        /// Total artifact size in bytes.
        size: usize,
    },

    /// The record count disagrees with the sequence range the header spans.
    #[error(
        "tail record count mismatch: {record_count} records, sequence range spans {range_span}"
    )]
    RecordCountMismatch {
        /// Records the header's `(from_seq, to_seq]` range implies.
        range_span: u64,
        /// Records actually accounted for — on decode the header's framed
        /// `record_count` field, on encode the number of records supplied.
        record_count: u64,
    },

    /// Records are not a contiguous ascending run from `from_seq`.
    #[error("tail record {index}: non-contiguous sequence, expected {expected}, got {got}")]
    NonContiguousSequence {
        /// Zero-based record index.
        index: usize,
        /// Sequence number required at this index.
        expected: u64,
        /// Sequence number actually found.
        got: u64,
    },

    /// `from_seq` / `to_seq` do not describe a well-formed range.
    #[error("invalid tail range: from_seq {from_seq}, to_seq {to_seq}")]
    InvalidRange {
        /// Header `from_seq`.
        from_seq: u64,
        /// Header `to_seq`.
        to_seq: u64,
    },

    /// `base_snap_seq` is not below the range the tail carries, so the tail
    /// does not layer onto the snapshot it names.
    #[error("tail base_snap_seq {base_snap_seq} is not below from_seq {from_seq}")]
    BaseAfterRange {
        /// Header `base_snap_seq`.
        base_snap_seq: u64,
        /// Header `from_seq`.
        from_seq: u64,
    },

    /// A sequence number exceeds the codebase-wide 2^53 - 1 ceiling.
    #[error("tail sequence_num {sequence_num} exceeds maximum {max}")]
    SequenceCeilingExceeded {
        /// The offending sequence number.
        sequence_num: u64,
        /// [`TAIL_MAX_SEQUENCE`].
        max: u64,
    },

    /// A payload does not fit the `u32` length field (encode side) or is
    /// otherwise unrepresentable on this platform.
    #[error("tail record {index}: payload of {len} bytes is too large to encode")]
    PayloadTooLarge {
        /// Zero-based record index.
        index: usize,
        /// The offending payload length.
        len: usize,
    },
}

/// Every codec failure is `Permanent` (SR-6.1): the bytes are durable, a retry
/// reads the same bytes. `Transient` is structurally impossible here — the
/// module does no I/O.
impl From<TailCodecError> for CheckpointError {
    fn from(err: TailCodecError) -> Self {
        CheckpointError::Permanent {
            source: Box::new(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixed timestamp: nothing in this codec reads a clock, so the value is
    /// arbitrary and every test uses the same one. Microseconds, matching
    /// `LayerNotification::Update { server_ts, .. }`.
    const TS: i64 = 1_764_000_000_000_000;

    /// Arbitrary layer generation; only its round-trip matters.
    const OBJ: u64 = 0x0123_4567_89AB_CDEF;

    /// Full-control record builders. The one-argument [`verbatim`] /
    /// [`empty_delta`] wrappers below delegate here, so there is exactly one
    /// struct literal per variant in this module.
    fn verbatim_with(sequence_num: u64, server_ts: i64, ompf: &[u8]) -> TailRecord {
        TailRecord::Verbatim {
            sequence_num,
            server_ts,
            ompf: Bytes::copy_from_slice(ompf),
        }
    }

    fn empty_delta_with(sequence_num: u64, server_ts: i64) -> TailRecord {
        TailRecord::EmptyDelta {
            sequence_num,
            server_ts,
        }
    }

    fn verbatim(sequence_num: u64) -> TailRecord {
        verbatim_with(sequence_num, TS, b"ompf-payload")
    }

    fn empty_delta(sequence_num: u64) -> TailRecord {
        empty_delta_with(sequence_num, TS)
    }

    /// Build a header and record list that are mutually consistent *by
    /// construction*: `to_seq = from_seq + kinds.len()`, one record per sequence
    /// number in `(from_seq, to_seq]`, `base_snap_seq <= from_seq`.
    ///
    /// Every happy-path test goes through this so a round-trip failure can never
    /// be an accidentally-invalid fixture. Payloads and timestamps vary with the
    /// record index, so a decoder that preserved counts while shuffling records
    /// fails rather than passing on interchangeable data.
    fn valid_envelope(
        object_id: u64,
        base_snap_seq: u64,
        from_seq: u64,
        kinds: &[TailRecordKind],
    ) -> (TailHeader, Vec<TailRecord>) {
        assert!(base_snap_seq <= from_seq, "fixture: base above range");
        let to_seq = from_seq + kinds.len() as u64;
        assert!(
            to_seq <= TAIL_MAX_SEQUENCE,
            "fixture: above sequence ceiling"
        );

        let records = kinds
            .iter()
            .enumerate()
            .map(|(index, kind)| {
                let seq = from_seq + 1 + index as u64;
                let ts = TS + i64::try_from(index).expect("test index fits i64");
                match kind {
                    TailRecordKind::Verbatim => {
                        verbatim_with(seq, ts, payload_for(index).as_bytes())
                    }
                    TailRecordKind::EmptyDelta => empty_delta_with(seq, ts),
                }
            })
            .collect();

        (
            TailHeader {
                object_id,
                base_snap_seq,
                from_seq,
                to_seq,
            },
            records,
        )
    }

    /// Index-dependent, variable-length payload so a decoder that mis-frames one
    /// record cannot accidentally produce the right bytes for the next.
    fn payload_for(index: usize) -> String {
        format!("ompf-record-{index}{}", "!".repeat(index))
    }

    /// Header field offsets the forging helpers below patch. Only
    /// `format_version` and `record_count` are exported by the module (tests are
    /// the only caller that needs the rest), so the remainder are named here
    /// rather than spelled as bare numbers at each patch site. Values come from
    /// the wire-format table in the module doc.
    const BASE_SNAP_SEQ_OFFSET: usize = 16;
    const FROM_SEQ_OFFSET: usize = 24;
    const TO_SEQ_OFFSET: usize = 32;

    /// Field offsets *within* a record, relative to the record's own start.
    const RECORD_KIND_OFFSET: usize = 0;
    const RECORD_SEQUENCE_OFFSET: usize = 1;
    const RECORD_PAYLOAD_LEN_OFFSET: usize = 17;

    /// Encode a valid artifact from [`valid_envelope`], returning its bytes
    /// alongside the records that produced them.
    ///
    /// Every decode-rejection test forges from a known-good encoding rather than
    /// hand-assembling bytes, so the only difference between the artifact under
    /// test and one the decoder accepts is the single field being corrupted —
    /// which is what makes "the decoder rejected it for *this* reason" a real
    /// assertion rather than an accident of an invalid fixture.
    fn encoded(
        base_snap_seq: u64,
        from_seq: u64,
        kinds: &[TailRecordKind],
    ) -> (Bytes, Vec<TailRecord>) {
        let (header, records) = valid_envelope(OBJ, base_snap_seq, from_seq, kinds);
        let bytes = encode_tail_envelope(&header, records.clone()).expect("fixture encodes");
        (bytes, records)
    }

    /// Copy `bytes`, overwriting `replacement.len()` bytes at `at`.
    fn patched(bytes: &Bytes, at: usize, replacement: &[u8]) -> Bytes {
        let mut raw = bytes.to_vec();
        raw[at..at + replacement.len()].copy_from_slice(replacement);
        Bytes::from(raw)
    }

    /// Copy `bytes` with `extra` appended.
    fn appended(bytes: &Bytes, extra: &[u8]) -> Bytes {
        let mut raw = bytes.to_vec();
        raw.extend_from_slice(extra);
        Bytes::from(raw)
    }

    /// Byte offset of each record in an encoding of `records`. Derived from the
    /// records themselves (prefix + payload stride) so a mixed-kind fixture can
    /// be patched at record `n` without hardcoding a layout arithmetic in every
    /// test.
    fn record_offsets(records: &[TailRecord]) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(records.len());
        let mut at = TAIL_HEADER_SIZE;
        for record in records {
            offsets.push(at);
            at += TAIL_RECORD_PREFIX_SIZE + record.ompf().map_or(0, |ompf| ompf.len());
        }
        offsets
    }

    /// The raw payload bytes of a verbatim record.
    fn ompf_bytes(record: &TailRecord) -> &[u8] {
        record.ompf().expect("record is verbatim")
    }

    /// The ordered kinds carried by a decoded envelope.
    fn kinds_of(envelope: &TailEnvelope) -> Vec<TailRecordKind> {
        envelope.iter().map(TailRecord::kind).collect()
    }

    /// Compile-time purity assertion backing the Task's "no fred / cxx / adapter
    /// types" criterion: a type carrying a `cxx::UniquePtr` or a fred client
    /// handle would not satisfy `Send + Sync`, so this fails to compile rather
    /// than failing at run time.
    const _: fn() = || {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TailHeader>();
        assert_send_sync::<TailRecord>();
        assert_send_sync::<TailRecordKind>();
        assert_send_sync::<TailEnvelope>();
        assert_send_sync::<TailCodecError>();
    };

    #[test]
    fn encode_rejects_inverted_range() {
        let header = TailHeader {
            object_id: 7,
            base_snap_seq: 4,
            from_seq: 9,
            to_seq: 4,
        };

        let err = encode_tail_envelope(&header, std::iter::empty()).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::InvalidRange {
                    from_seq: 9,
                    to_seq: 4
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn encode_rejects_base_snap_seq_above_from_seq() {
        let header = TailHeader {
            object_id: 7,
            base_snap_seq: 5,
            from_seq: 4,
            to_seq: 6,
        };

        let err = encode_tail_envelope(&header, vec![verbatim(5), verbatim(6)]).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::BaseAfterRange {
                    base_snap_seq: 5,
                    from_seq: 4
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn encode_rejects_to_seq_above_sequence_ceiling() {
        let over = TAIL_MAX_SEQUENCE + 1;
        let header = TailHeader {
            object_id: 7,
            base_snap_seq: over,
            from_seq: over,
            to_seq: over,
        };

        let err = encode_tail_envelope(&header, std::iter::empty()).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::SequenceCeilingExceeded {
                    sequence_num,
                    max: TAIL_MAX_SEQUENCE,
                } if sequence_num == over
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn encode_rejects_non_contiguous_records() {
        let header = TailHeader {
            object_id: 7,
            base_snap_seq: 4,
            from_seq: 4,
            to_seq: 6,
        };

        // A gap at sequence 6: the second record must be 6, not 7.
        let err = encode_tail_envelope(&header, vec![verbatim(5), verbatim(7)]).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::NonContiguousSequence {
                    index: 1,
                    expected: 6,
                    got: 7
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn encode_rejects_repeated_sequence_number() {
        let header = TailHeader {
            object_id: 7,
            base_snap_seq: 4,
            from_seq: 4,
            to_seq: 6,
        };

        let err = encode_tail_envelope(&header, vec![verbatim(5), verbatim(5)]).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::NonContiguousSequence {
                    index: 1,
                    expected: 6,
                    got: 5
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn encode_rejects_record_undersupply() {
        let header = TailHeader {
            object_id: 7,
            base_snap_seq: 4,
            from_seq: 4,
            to_seq: 7,
        };

        // Header claims three commits; only two are supplied. Claiming beyond
        // content is the b.pw6 defect (SR-1.4).
        let err = encode_tail_envelope(&header, vec![verbatim(5), empty_delta(6)]).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::RecordCountMismatch {
                    range_span: 3,
                    record_count: 2
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn encode_rejects_record_oversupply() {
        let header = TailHeader {
            object_id: 7,
            base_snap_seq: 4,
            from_seq: 4,
            to_seq: 5,
        };

        let err = encode_tail_envelope(&header, vec![verbatim(5), empty_delta(6)]).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::RecordCountMismatch {
                    range_span: 1,
                    record_count: 2
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn encode_rejects_zero_length_tail_carrying_records() {
        let header = TailHeader {
            object_id: 7,
            base_snap_seq: 4,
            from_seq: 4,
            to_seq: 4,
        };

        let err = encode_tail_envelope(&header, vec![verbatim(5)]).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::RecordCountMismatch {
                    range_span: 0,
                    record_count: 1
                }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn encode_decode_empty_tail_roundtrips_with_zero_records() {
        let (header, records) = valid_envelope(OBJ, 40, 40, &[]);
        assert!(records.is_empty(), "fixture carries no records");

        let bytes = encode_tail_envelope(&header, records).expect("encode empty tail");
        let decoded = decode_tail_envelope(&bytes).expect("decode empty tail");

        // A zero-length tail is a bare header, NOT an empty byte string.
        assert_eq!(
            bytes.len(),
            TAIL_HEADER_SIZE,
            "empty tail is exactly the fixed header"
        );
        assert_eq!(decoded.header(), &header, "header survives the round trip");
        assert_eq!(decoded.format_version(), TAIL_FORMAT_VERSION);
        assert!(decoded.records().is_empty(), "no records decoded");
        assert_eq!(decoded.len(), 0);
        assert!(decoded.is_empty());
        assert_eq!(decoded.iter().count(), 0, "iterator yields zero records");
    }

    #[test]
    fn encode_decode_verbatim_only_tail_roundtrips_byte_identically() {
        let kinds = [TailRecordKind::Verbatim; 3];
        let (header, records) = valid_envelope(OBJ, 100, 140, &kinds);

        let bytes = encode_tail_envelope(&header, records.clone()).expect("encode verbatim tail");
        let decoded = decode_tail_envelope(&bytes).expect("decode verbatim tail");

        assert_eq!(decoded.len(), records.len());
        for (index, (got, want)) in decoded.iter().zip(records.iter()).enumerate() {
            assert_eq!(
                ompf_bytes(got),
                ompf_bytes(want),
                "record {index}: OMPF bytes must survive byte-for-byte"
            );
            assert_eq!(
                got.sequence_num(),
                want.sequence_num(),
                "record {index}: sequence_num"
            );
            assert_eq!(
                got.server_ts(),
                want.server_ts(),
                "record {index}: server_ts"
            );
            assert_eq!(got.kind(), TailRecordKind::Verbatim, "record {index}: kind");
        }
    }

    #[test]
    fn encode_decode_empty_delta_only_tail_roundtrips() {
        let kinds = [TailRecordKind::EmptyDelta; 3];
        let (header, records) = valid_envelope(OBJ, 7, 7, &kinds);

        let bytes =
            encode_tail_envelope(&header, records.clone()).expect("encode empty-delta tail");
        let decoded = decode_tail_envelope(&bytes).expect("decode empty-delta tail");

        assert_eq!(decoded.len(), records.len());
        for (index, (got, want)) in decoded.iter().zip(records.iter()).enumerate() {
            assert_eq!(
                got.kind(),
                TailRecordKind::EmptyDelta,
                "record {index}: kind must not be reclassified"
            );
            assert_eq!(
                got.sequence_num(),
                want.sequence_num(),
                "record {index}: sequence_num"
            );
            assert_eq!(
                got.server_ts(),
                want.server_ts(),
                "record {index}: server_ts"
            );
            assert!(got.ompf().is_none(), "record {index}: carries no payload");
        }
    }

    #[test]
    fn encode_decode_mixed_kind_tail_preserves_per_record_kind_and_order() {
        let kinds = [
            TailRecordKind::Verbatim,
            TailRecordKind::EmptyDelta,
            TailRecordKind::Verbatim,
            TailRecordKind::EmptyDelta,
        ];
        let (header, records) = valid_envelope(OBJ, 3, 20, &kinds);

        let bytes = encode_tail_envelope(&header, records.clone()).expect("encode mixed tail");
        let decoded = decode_tail_envelope(&bytes).expect("decode mixed tail");

        // The full ordered kind sequence, so preserving the count while dropping
        // or reordering records fails.
        assert_eq!(
            kinds_of(&decoded),
            kinds.to_vec(),
            "per-record kinds must decode in input order"
        );
        assert_eq!(
            decoded
                .iter()
                .map(TailRecord::sequence_num)
                .collect::<Vec<_>>(),
            vec![21, 22, 23, 24],
            "records decode in ascending contiguous order"
        );
        assert_eq!(
            decoded.records(),
            records.as_slice(),
            "every record round-trips identically, in order"
        );
    }

    #[test]
    fn decode_header_fields_match_encoded_header() {
        // Every field a distinct value, so a field-order swap in the codec is
        // caught rather than aliasing onto an equal neighbour.
        let header = TailHeader {
            object_id: 0xDEAD_BEEF_0000_0042,
            base_snap_seq: 1_000,
            from_seq: 1_500,
            to_seq: 1_502,
        };
        let records = vec![verbatim(1_501), empty_delta(1_502)];

        let bytes = encode_tail_envelope(&header, records).expect("encode");
        let decoded = decode_tail_envelope(&bytes).expect("decode");
        let (peeked_version, peeked_header) = peek_tail_header(&bytes).expect("peek header");

        assert_eq!(decoded.format_version(), TAIL_FORMAT_VERSION);
        assert_eq!(decoded.header().object_id, 0xDEAD_BEEF_0000_0042);
        assert_eq!(decoded.header().base_snap_seq, 1_000);
        assert_eq!(decoded.header().from_seq, 1_500);
        assert_eq!(decoded.header().to_seq, 1_502);
        // The cheap header-only path must agree with the full decode.
        assert_eq!(peeked_version, TAIL_FORMAT_VERSION);
        assert_eq!(peeked_header, header, "peek agrees with full decode");
    }

    #[test]
    fn peek_tail_header_reads_only_the_fixed_header_of_a_record_bearing_tail() {
        // SR-3.1 lineage gather and adapter-side listing inspect a multi-MB
        // tail's identity without fetching its records, so `peek_tail_header`
        // must be satisfied by the first `TAIL_HEADER_SIZE` bytes alone. Every
        // other test hands it a whole artifact, so a peek that quietly needed
        // the record bytes would pass all of them while forcing gather to
        // download every candidate tail in full.
        let kinds = [TailRecordKind::Verbatim; 8];
        let (bytes, _) = encoded(100, 140, &kinds);
        assert!(
            bytes.len() > 2 * TAIL_HEADER_SIZE,
            "fixture carries substantially more than a bare header"
        );
        let whole = peek_tail_header(&bytes).expect("peek the whole artifact");

        let header_only = peek_tail_header(&bytes[..TAIL_HEADER_SIZE]);

        assert_eq!(
            header_only.expect("the fixed header alone must peek"),
            whole,
            "peeking the fixed header alone must agree with peeking the whole artifact"
        );
    }

    #[test]
    fn encode_accepts_offset_bytes_view_and_decodes_identically() {
        // A `Bytes` view at a non-zero offset into a larger allocation: the
        // classic bug is encoding from the underlying buffer instead of the view.
        let parent = Bytes::from_static(b"XXXXXXXXXXXXinner-ompf-payloadYYYYYYYYYYYY");
        let view = parent.slice(12..30);
        assert_eq!(
            &view[..],
            b"inner-ompf-payload",
            "fixture slice is the view"
        );

        let header = TailHeader {
            object_id: OBJ,
            base_snap_seq: 9,
            from_seq: 9,
            to_seq: 10,
        };
        let record = TailRecord::Verbatim {
            sequence_num: 10,
            server_ts: TS,
            ompf: view.clone(),
        };

        let bytes = encode_tail_envelope(&header, vec![record]).expect("encode offset view");
        let decoded = decode_tail_envelope(&bytes).expect("decode offset view");

        assert_eq!(decoded.len(), 1);
        assert_eq!(
            ompf_bytes(&decoded.records()[0]),
            &view[..],
            "decoded payload is the view, not the parent allocation"
        );
        assert_eq!(
            bytes.len(),
            TAIL_HEADER_SIZE + TAIL_RECORD_PREFIX_SIZE + view.len(),
            "only the view's bytes were encoded"
        );
    }

    #[test]
    fn boundary_sequence_and_timestamp_values_roundtrip() {
        // Case 1: every sequence-bearing field at its floor, object_id zero, and
        // the `from_seq == to_seq` zero-length-range shape.
        let (low_header, low_records) = valid_envelope(0, 0, 0, &[]);
        let low_bytes = encode_tail_envelope(&low_header, low_records).expect("encode floor");
        let low = decode_tail_envelope(&low_bytes).expect("decode floor");

        assert_eq!(low.header(), &low_header);
        assert_eq!(
            low.header().from_seq,
            low.header().to_seq,
            "zero-length range"
        );
        assert!(low.is_empty());

        // Case 2: object_id at `u64::MAX`, sequence fields pinned to the 2^53-1
        // ceiling (`u64::MAX` there is unencodable by design), and `server_ts` at
        // both `i64` extremes — microseconds must not be truncated or rescaled.
        let from_seq = TAIL_MAX_SEQUENCE - 2;
        let header = TailHeader {
            object_id: u64::MAX,
            base_snap_seq: from_seq,
            from_seq,
            to_seq: TAIL_MAX_SEQUENCE,
        };
        let records = vec![
            verbatim_with(from_seq + 1, i64::MIN, b"floor-ts"),
            empty_delta_with(TAIL_MAX_SEQUENCE, i64::MAX),
        ];

        let bytes = encode_tail_envelope(&header, records.clone()).expect("encode ceiling");
        let high = decode_tail_envelope(&bytes).expect("decode ceiling");

        assert_eq!(high.header(), &header, "boundary header round-trips");
        assert_eq!(high.header().to_seq, TAIL_MAX_SEQUENCE);
        assert_eq!(high.records(), records.as_slice());
        assert_eq!(
            high.records()[0].server_ts(),
            i64::MIN,
            "i64::MIN server_ts"
        );
        assert_eq!(
            high.records()[1].server_ts(),
            i64::MAX,
            "i64::MAX server_ts"
        );
    }

    #[test]
    fn large_verbatim_payload_roundtrips() {
        // Generated, not checked in (test-writing guide § Test Data
        // Construction). Deterministic byte pattern so head/tail spot-checks are
        // meaningful without a 1 MiB `assert_eq!` diff on failure.
        const PAYLOAD_LEN: usize = 1024 * 1024;
        let payload: Vec<u8> = (0..PAYLOAD_LEN).map(|i| (i % 251) as u8).collect();

        let header = TailHeader {
            object_id: OBJ,
            base_snap_seq: 0,
            from_seq: 0,
            to_seq: 1,
        };
        let bytes = encode_tail_envelope(&header, vec![verbatim_with(1, TS, &payload)])
            .expect("encode 1 MiB payload");
        let decoded = decode_tail_envelope(&bytes).expect("decode 1 MiB payload");

        assert_eq!(
            bytes.len(),
            TAIL_HEADER_SIZE + TAIL_RECORD_PREFIX_SIZE + PAYLOAD_LEN,
            "no padding or re-encode around a large payload"
        );
        assert_eq!(decoded.len(), 1);
        let got = ompf_bytes(&decoded.records()[0]);
        assert_eq!(got.len(), PAYLOAD_LEN, "payload length preserved");
        assert_eq!(&got[..64], &payload[..64], "leading bytes preserved");
        assert_eq!(
            &got[PAYLOAD_LEN - 64..],
            &payload[PAYLOAD_LEN - 64..],
            "trailing bytes preserved"
        );
    }

    #[test]
    fn into_records_yields_the_decoded_records() {
        let kinds = [TailRecordKind::Verbatim, TailRecordKind::EmptyDelta];
        let (header, records) = valid_envelope(OBJ, 2, 5, &kinds);

        let bytes = encode_tail_envelope(&header, records.clone()).expect("encode");
        let owned = decode_tail_envelope(&bytes).expect("decode").into_records();

        assert_eq!(owned, records, "consuming accessor yields the same records");
    }

    /// `TailRecord`'s manual `Debug` renders a verbatim payload as `<N bytes>`
    /// rather than escaping it. A derived `Debug` would turn one `{:?}` on an
    /// error path into a multi-megabyte log line (payload is capped only by
    /// `LIVE_EDIT_MAX_LAYER_SIZE`), so this pins the bound at the record *and*
    /// at `TailEnvelope`, whose derive inherits the property through the record.
    #[test]
    fn record_debug_output_is_bounded_and_never_dumps_payload_bytes() {
        /// Repeating printable, non-zero marker: `Bytes`'s own `Debug` renders
        /// printable ASCII verbatim, so finding this string in a rendering is
        /// unambiguous evidence that payload bytes leaked into it.
        const MARKER: &str = "OMPF-PAYLOAD-MARKER-";
        const PAYLOAD_LEN: usize = 4096;

        let payload: Vec<u8> = MARKER.bytes().cycle().take(PAYLOAD_LEN).collect();
        let record = verbatim_with(9, TS, &payload);

        // Positive control: the payload really is self-identifying under the
        // rendering the derive would have used. Without this, the "does not
        // contain MARKER" assertion below could pass vacuously.
        let unbounded = format!("{:?}", record.ompf().expect("fixture is verbatim"));
        assert!(
            unbounded.contains(MARKER) && unbounded.len() > PAYLOAD_LEN,
            "control: the payload's own Debug dumps its bytes"
        );

        let rendered = format!("{record:?}");
        assert!(
            rendered.len() < 200,
            "record Debug stays bounded regardless of payload size, got {} chars: {rendered}",
            rendered.len()
        );
        assert!(
            rendered.contains(&format!("<{PAYLOAD_LEN} bytes>")),
            "record Debug reports the payload length, got: {rendered}"
        );
        assert!(
            !rendered.contains(MARKER),
            "record Debug leaked payload bytes: {rendered}"
        );

        // The payload-free variant is unchanged from the derived form: still a
        // struct rendering that names both of its fields.
        let empty = format!("{:?}", empty_delta(9));
        assert!(
            empty.contains("sequence_num") && empty.contains("server_ts"),
            "empty-delta Debug still names its fields, got: {empty}"
        );

        let header = TailHeader {
            object_id: OBJ,
            base_snap_seq: 0,
            from_seq: 8,
            to_seq: 9,
        };
        let bytes = encode_tail_envelope(&header, vec![record]).expect("encode");
        let envelope = decode_tail_envelope(&bytes).expect("decode");

        let envelope_rendered = format!("{envelope:?}");
        assert!(
            envelope_rendered.len() < 400,
            "envelope Debug inherits the record's bound, got {} chars: {envelope_rendered}",
            envelope_rendered.len()
        );
        assert!(
            !envelope_rendered.contains(MARKER),
            "envelope Debug leaked payload bytes: {envelope_rendered}"
        );
    }

    // ---------------------------------------------------------------------
    // Decode-side rejection: semantic malformation.
    //
    // Every case below forges a *specific* corruption into an otherwise-valid
    // encoding and asserts the *specific* error variant. SR-6.1's contract is
    // "never a panic, never a guess": a decoder that fell back to a default
    // interpretation, or that reported a different structural defect than the
    // one present, would pass an `is_err()` check and fail these.
    //
    // Allocation/DoS-shaped inputs (forged `record_count` / `payload_len`
    // driving a large allocation) are deliberately NOT here — they live in the
    // adversarial-corpus tier.
    // ---------------------------------------------------------------------

    #[test]
    fn decode_unknown_format_version_returns_unsupported_version_error() {
        let (bytes, _) = encoded(
            3,
            20,
            &[TailRecordKind::Verbatim, TailRecordKind::EmptyDelta],
        );

        // The next version this build has never seen, and the saturated value a
        // bit-rotted field is most likely to land on.
        for forged in [TAIL_FORMAT_VERSION + 1, u32::MAX] {
            let artifact = patched(&bytes, TAIL_FORMAT_VERSION_OFFSET, &forged.to_le_bytes());

            let err = decode_tail_envelope(&artifact).unwrap_err();

            assert!(
                matches!(
                    err,
                    TailCodecError::UnknownFormatVersion { got, supported }
                        if got == forged && supported == TAIL_FORMAT_VERSION
                ),
                "version {forged}: unexpected error: {err:?}"
            );
            // The header-only path is the one lineage gather uses, so it must
            // refuse the same artifact rather than handing back a header parsed
            // under v1 assumptions.
            let peeked = peek_tail_header(&artifact).unwrap_err();
            assert!(
                matches!(
                    peeked,
                    TailCodecError::UnknownFormatVersion { got, .. } if got == forged
                ),
                "version {forged}: peek accepted what decode rejected: {peeked:?}"
            );
        }
    }

    #[test]
    fn decode_unknown_format_version_is_reported_before_truncation() {
        // A v2 artifact may have a header of a different length entirely, so
        // the version check runs before any length or semantic check (SR-6.1).
        // Reporting `Truncated` here would tell the operator the artifact is
        // corrupt when it is merely newer.
        let mut raw = Vec::new();
        raw.extend_from_slice(&TAIL_ENVELOPE_MAGIC);
        raw.extend_from_slice(&(TAIL_FORMAT_VERSION + 1).to_le_bytes());
        let artifact = Bytes::from(raw);
        assert!(
            artifact.len() < TAIL_HEADER_SIZE,
            "fixture is shorter than a v1 header"
        );

        let err = decode_tail_envelope(&artifact).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::UnknownFormatVersion {
                    got,
                    supported: TAIL_FORMAT_VERSION
                } if got == TAIL_FORMAT_VERSION + 1
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn decode_unknown_format_version_classifies_as_permanent() {
        // SR-6.1: the restore path (Epic 4) must step down to the next older
        // generation on an unreadable artifact, not retry it forever. That
        // behaviour hangs entirely on this classification.
        let (bytes, _) = encoded(0, 0, &[TailRecordKind::Verbatim]);
        let artifact = patched(
            &bytes,
            TAIL_FORMAT_VERSION_OFFSET,
            &(TAIL_FORMAT_VERSION + 1).to_le_bytes(),
        );
        let err = decode_tail_envelope(&artifact).unwrap_err();

        let classified = CheckpointError::from(err);

        assert!(
            classified.is_permanent(),
            "unknown format version must be Permanent, got {classified:?}"
        );
        assert!(
            !classified.is_transient(),
            "a Transient classification would make restore retry an artifact this build can never read"
        );
        assert!(!classified.is_not_found(), "got {classified:?}");
        assert!(!classified.is_conflict(), "got {classified:?}");
    }

    #[test]
    fn decode_errors_classify_as_permanent_never_transient() {
        // The whole codec error vocabulary shares one classification: the bytes
        // are durable, so a retry reads the same bytes. Driven from real forged
        // artifacts rather than hand-constructed enum values, so a future
        // variant-specific mapping in the `From` impl is caught on a live path.
        let (bytes, records) =
            encoded(3, 20, &[TailRecordKind::Verbatim, TailRecordKind::Verbatim]);
        let offsets = record_offsets(&records);
        let forged: Vec<(&str, Bytes)> = vec![
            ("bad magic", patched(&bytes, 0, b"OMPF")),
            (
                "unknown version",
                patched(&bytes, TAIL_FORMAT_VERSION_OFFSET, &7u32.to_le_bytes()),
            ),
            ("truncated", bytes.slice(..TAIL_HEADER_SIZE + 4)),
            ("trailing bytes", appended(&bytes, b"!")),
            (
                "unknown record kind",
                patched(&bytes, offsets[1] + RECORD_KIND_OFFSET, &[9]),
            ),
            (
                "record count mismatch",
                patched(&bytes, TAIL_RECORD_COUNT_OFFSET, &1u32.to_le_bytes()),
            ),
            (
                "non-contiguous sequence",
                patched(
                    &bytes,
                    offsets[1] + RECORD_SEQUENCE_OFFSET,
                    &99u64.to_le_bytes(),
                ),
            ),
        ];

        for (label, artifact) in forged {
            let err = decode_tail_envelope(&artifact)
                .map(|_| ())
                .expect_err(&format!("{label}: forged artifact must not decode"));

            let classified = CheckpointError::from(err);

            assert!(
                classified.is_permanent() && !classified.is_transient(),
                "{label}: expected Permanent, got {classified:?}"
            );
        }
    }

    #[test]
    fn checkpoint_error_conversion_preserves_the_codec_message_as_its_source() {
        // `CheckpointError::Permanent` renders as
        // "permanent checkpoint-store error: {source}", so the codec's own
        // sentence is what an operator actually reads when a restore refuses an
        // artifact. The classification tests above prove only the *variant*: a
        // `From` impl that boxed a generic stand-in would satisfy every one of
        // them while leaving the log silent about which structural check failed.
        let (bytes, _) = encoded(0, 0, &[TailRecordKind::Verbatim]);
        let artifact = patched(&bytes, TAIL_FORMAT_VERSION_OFFSET, &9u32.to_le_bytes());
        let err = decode_tail_envelope(&artifact).unwrap_err();
        let codec_message = err.to_string();
        assert!(
            codec_message.contains('9'),
            "fixture: the codec message must name the forged version, got {codec_message:?}"
        );

        let rendered = CheckpointError::from(err).to_string();

        assert!(
            rendered.ends_with(&codec_message),
            "the checkpoint error must render the codec's own message verbatim; \
             got {rendered:?}, which does not carry {codec_message:?}"
        );
    }

    // ---------------------------------------------------------------------
    // Operator-facing `Display` surface.
    //
    // The `#[error(...)]` attributes are what a restore failure is diagnosed
    // from — `CheckpointError::Permanent` renders the codec message verbatim
    // (asserted just above). Two hazards live here that no variant-specific
    // decode test can see, because each of those asserts on the *value* via
    // `matches!` and never renders it: a copy-pasted attribute (two variants
    // rendering the same sentence, so a log line names the wrong defect), and
    // an attribute that drops or transposes the fields it was given.
    // ---------------------------------------------------------------------

    /// Number of [`TailCodecError`] variants. Pinned so a variant added to the
    /// enum but not to [`every_error_variant`] fails the fixture guard there
    /// rather than silently escaping both tests below.
    const ERROR_VARIANT_COUNT: usize = 14;

    /// One instance of every [`TailCodecError`] variant.
    ///
    /// Field values are distinct across the whole list and distinct within each
    /// variant, so a message that renders the wrong field — or renders two
    /// fields in the wrong order — is visible rather than aliasing onto an equal
    /// neighbour.
    fn every_error_variant() -> Vec<TailCodecError> {
        let variants = vec![
            TailCodecError::BadMagic { got: *b"OMPF" },
            TailCodecError::UnknownFormatVersion {
                got: 37,
                supported: 41,
            },
            TailCodecError::Truncated {
                at: "record payload",
                needed: 43,
                available: 47,
            },
            TailCodecError::TrailingBytes { extra: 53 },
            TailCodecError::UnknownRecordKind { index: 59, got: 61 },
            TailCodecError::EmptyVerbatimPayload { index: 67 },
            TailCodecError::NonEmptyEmptyDeltaPayload { index: 71, len: 73 },
            TailCodecError::RecordCountTooLarge {
                record_count: 79,
                max: 83,
                size: 89,
            },
            TailCodecError::RecordCountMismatch {
                range_span: 97,
                record_count: 101,
            },
            TailCodecError::NonContiguousSequence {
                index: 103,
                expected: 107,
                got: 109,
            },
            TailCodecError::InvalidRange {
                from_seq: 113,
                to_seq: 127,
            },
            TailCodecError::BaseAfterRange {
                base_snap_seq: 131,
                from_seq: 137,
            },
            TailCodecError::SequenceCeilingExceeded {
                sequence_num: 139,
                max: 149,
            },
            TailCodecError::PayloadTooLarge {
                index: 151,
                len: 157,
            },
        ];
        assert_eq!(
            variants.len(),
            ERROR_VARIANT_COUNT,
            "every TailCodecError variant must be enumerated here, or the \
             Display tests below silently skip it"
        );
        variants
    }

    /// The context `err`'s `Display` must name, in the order it must appear.
    ///
    /// Written as an exhaustive `match`: adding a variant to [`TailCodecError`]
    /// without deciding what its operator-facing message has to say fails to
    /// compile here.
    fn required_display_context(err: &TailCodecError) -> Vec<String> {
        match err {
            TailCodecError::BadMagic { got } => {
                vec![format!("{TAIL_ENVELOPE_MAGIC:?}"), format!("{got:?}")]
            }
            TailCodecError::UnknownFormatVersion { got, supported } => {
                vec![got.to_string(), supported.to_string()]
            }
            TailCodecError::Truncated {
                at,
                needed,
                available,
            } => vec![at.to_string(), needed.to_string(), available.to_string()],
            TailCodecError::TrailingBytes { extra } => vec![extra.to_string()],
            TailCodecError::UnknownRecordKind { index, got } => {
                vec![index.to_string(), got.to_string()]
            }
            TailCodecError::EmptyVerbatimPayload { index } => vec![index.to_string()],
            TailCodecError::NonEmptyEmptyDeltaPayload { index, len } => {
                vec![index.to_string(), len.to_string()]
            }
            TailCodecError::RecordCountTooLarge {
                record_count,
                max,
                size,
            } => vec![record_count.to_string(), max.to_string(), size.to_string()],
            TailCodecError::RecordCountMismatch {
                range_span,
                record_count,
            } => {
                // Rendered record-count-first; this list is order-sensitive.
                vec![record_count.to_string(), range_span.to_string()]
            }
            TailCodecError::NonContiguousSequence {
                index,
                expected,
                got,
            } => vec![index.to_string(), expected.to_string(), got.to_string()],
            TailCodecError::InvalidRange { from_seq, to_seq } => {
                vec![from_seq.to_string(), to_seq.to_string()]
            }
            TailCodecError::BaseAfterRange {
                base_snap_seq,
                from_seq,
            } => vec![base_snap_seq.to_string(), from_seq.to_string()],
            TailCodecError::SequenceCeilingExceeded { sequence_num, max } => {
                vec![sequence_num.to_string(), max.to_string()]
            }
            TailCodecError::PayloadTooLarge { index, len } => {
                vec![index.to_string(), len.to_string()]
            }
        }
    }

    #[test]
    fn error_display_names_every_field_in_declaration_order() {
        for err in every_error_variant() {
            let message = err.to_string();

            // Walked with a moving cursor rather than independent `contains`
            // calls, so a message that renders its fields transposed — the
            // shape a copy-pasted attribute produces — fails here too.
            let mut cursor = 0usize;
            for fragment in required_display_context(&err) {
                let at = message[cursor..].find(&fragment).unwrap_or_else(|| {
                    panic!(
                        "{err:?}: Display message {message:?} does not name \
                         {fragment:?} at or after byte {cursor}"
                    )
                });
                cursor += at + fragment.len();
            }
        }
    }

    #[test]
    fn error_display_is_distinct_for_every_variant() {
        let mut seen: Vec<(String, String)> = Vec::with_capacity(ERROR_VARIANT_COUNT);

        for err in every_error_variant() {
            let message = err.to_string();
            if let Some((other, _)) = seen.iter().find(|(_, m)| *m == message) {
                panic!(
                    "{err:?} and {other} render the identical message {message:?}; \
                     an operator reading a log line could not tell which \
                     structural check failed"
                );
            }
            seen.push((format!("{err:?}"), message));
        }

        assert_eq!(seen.len(), ERROR_VARIANT_COUNT);
    }

    #[test]
    fn decode_format_version_zero_rejected() {
        // `0` is not a sentinel for "unversioned" or "current": it is simply not
        // the supported version. Pinned so a zero-filled or partially-written
        // field can never be read as an implicit v1.
        let (bytes, _) = encoded(0, 0, &[TailRecordKind::EmptyDelta]);
        let artifact = patched(&bytes, TAIL_FORMAT_VERSION_OFFSET, &0u32.to_le_bytes());

        let err = decode_tail_envelope(&artifact).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::UnknownFormatVersion {
                    got: 0,
                    supported: TAIL_FORMAT_VERSION
                }
            ),
            "unexpected error: {err:?}"
        );
        assert_eq!(TAIL_FORMAT_VERSION, 1, "supported version is 1, not 0");
    }

    #[test]
    fn decode_empty_input_is_truncated_not_an_empty_tail() {
        // `OmpfContainer::parse` accepts empty input as a valid empty container;
        // that precedent is deliberately not copied. A zero-length tail is a
        // 44-byte header with `record_count == 0`, so a zero-byte blob — a
        // truncated write, a missing object served as empty — must never be
        // mistaken for "this tail carried no commits", which would silently skip
        // replay of commits the artifact was supposed to hold.
        let empty = Bytes::new();

        let err = decode_tail_envelope(&empty).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::Truncated {
                    at: "magic",
                    available: 0,
                    ..
                }
            ),
            "unexpected error: {err:?}"
        );
        let peeked = peek_tail_header(&empty).unwrap_err();
        assert!(
            matches!(peeked, TailCodecError::Truncated { at: "magic", .. }),
            "peek gave {peeked:?}"
        );
    }

    #[test]
    fn decode_truncated_at_every_prefix_length_returns_typed_error_never_panics() {
        // One loop covers header truncation, mid-record-prefix truncation, and
        // mid-payload truncation without hand-enumerating the cut points. Mixed
        // kinds and variable-length payloads so the cuts land at every kind of
        // structural boundary.
        let kinds = [
            TailRecordKind::Verbatim,
            TailRecordKind::EmptyDelta,
            TailRecordKind::Verbatim,
            TailRecordKind::EmptyDelta,
        ];
        let (bytes, _) = encoded(3, 20, &kinds);
        assert!(
            bytes.len() > TAIL_HEADER_SIZE,
            "fixture carries records to truncate through"
        );

        for cut in 0..bytes.len() {
            let artifact = bytes.slice(..cut);

            let err = match decode_tail_envelope(&artifact) {
                Ok(envelope) => panic!(
                    "cut at {cut}/{}: a prefix of a valid artifact decoded as {envelope:?}",
                    bytes.len()
                ),
                Err(err) => err,
            };

            // Two shapes are reachable: the input ran out mid-field, or the
            // declared record count no longer fits the surviving bytes (the
            // allocation guard fires before the per-record read). Both are
            // truncation reported at a different structural level; anything
            // else means the decoder mis-diagnosed the defect.
            assert!(
                matches!(
                    err,
                    TailCodecError::Truncated { .. } | TailCodecError::RecordCountTooLarge { .. }
                ),
                "cut at {cut}: unexpected error: {err:?}"
            );
        }
    }

    #[test]
    fn decode_bad_magic_returns_bad_magic_error() {
        // An OMPF container is the blob most likely to be mis-fed to this
        // decoder (SR-2.2: a tail is explicitly NOT an OMPF container), so it is
        // built here with the real serializer rather than a stand-in. This is
        // the codec's only `#[cfg(test)]`-scoped reference to `OmpfContainer` —
        // the encode/decode paths themselves never name it, and payload bytes
        // stay opaque to them.
        let ompf = crate::domain::ompf::OmpfContainer::serialize(b"flatbuffers-delta-data", &[])
            .expect("serialize OMPF fixture");
        let ompf = Bytes::from(ompf);
        assert_eq!(&ompf[..4], b"OMPF", "fixture is a real OMPF container");

        let (valid, _) = encoded(0, 0, &[TailRecordKind::Verbatim]);
        let cases: [(&str, Bytes, [u8; 4]); 3] = [
            ("OMPF container", ompf, *b"OMPF"),
            (
                "zero-filled region",
                Bytes::from(vec![0u8; TAIL_HEADER_SIZE]),
                [0; 4],
            ),
            // A single flipped magic byte on an otherwise-perfect artifact: the
            // near-miss a partial overwrite or a bit flip produces.
            ("one byte off", patched(&valid, 3, b"M"), *b"LETM"),
        ];

        for (label, artifact, expected) in cases {
            let err = decode_tail_envelope(&artifact).unwrap_err();

            assert!(
                matches!(err, TailCodecError::BadMagic { got } if got == expected),
                "{label}: unexpected error: {err:?}"
            );
            // The version field of a foreign blob must never be read: magic is
            // checked first, so a mis-fed OMPF cannot masquerade as a future
            // tail version.
            let peeked = peek_tail_header(&artifact).unwrap_err();
            assert!(
                matches!(peeked, TailCodecError::BadMagic { .. }),
                "{label}: peek gave {peeked:?}"
            );
        }
    }

    #[test]
    fn decode_unknown_record_kind_flag_returns_typed_error() {
        // SR-6.1's "never guess", applied per record: an unrecognized kind must
        // not fall back to the verbatim interpretation, and must not be skipped
        // (skipping would shorten the replay while the header still claims full
        // coverage — the b.pw6 shape).
        let (bytes, records) = encoded(
            3,
            20,
            &[TailRecordKind::EmptyDelta, TailRecordKind::EmptyDelta],
        );
        let offsets = record_offsets(&records);

        // `0` is the reserved-invalid discriminant, `3` the next unused one,
        // `255` the saturated byte.
        for forged_kind in [0u8, 3, 255] {
            for (index, offset) in offsets.iter().enumerate() {
                let artifact = patched(&bytes, offset + RECORD_KIND_OFFSET, &[forged_kind]);

                let err = decode_tail_envelope(&artifact).unwrap_err();

                assert!(
                    matches!(
                        err,
                        TailCodecError::UnknownRecordKind { index: at, got }
                            if at == index && got == forged_kind
                    ),
                    "kind {forged_kind} at record {index}: unexpected error: {err:?}"
                );
            }
        }
    }

    #[test]
    fn decode_empty_delta_record_carrying_payload_bytes_is_rejected() {
        // Case 1: a verbatim record's kind byte flipped to empty-delta. The
        // payload bytes are really there, so this cannot be mistaken for
        // truncation — the record is internally inconsistent, full stop.
        let (bytes, records) = encoded(0, 0, &[TailRecordKind::Verbatim]);
        let payload_len = records[0].ompf().expect("verbatim").len() as u32;
        assert!(payload_len > 0, "fixture carries payload bytes");
        let flipped = patched(
            &bytes,
            TAIL_HEADER_SIZE + RECORD_KIND_OFFSET,
            &[TailRecordKind::EmptyDelta.as_u8()],
        );

        let err = decode_tail_envelope(&flipped).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::NonEmptyEmptyDeltaPayload { index: 0, len }
                    if len == payload_len
            ),
            "kind-flip: unexpected error: {err:?}"
        );

        // Case 2: a genuine empty-delta record whose declared length is
        // non-zero, with the declared bytes actually appended so the artifact is
        // not short. Rejected on the inconsistency alone.
        let (empty_bytes, _) = encoded(0, 0, &[TailRecordKind::EmptyDelta]);
        let relabelled = patched(
            &empty_bytes,
            TAIL_HEADER_SIZE + RECORD_PAYLOAD_LEN_OFFSET,
            &3u32.to_le_bytes(),
        );
        let with_bytes = appended(&relabelled, b"abc");

        let err = decode_tail_envelope(&with_bytes).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::NonEmptyEmptyDeltaPayload { index: 0, len: 3 }
            ),
            "declared-length forge: unexpected error: {err:?}"
        );
    }

    #[test]
    fn decode_verbatim_record_with_zero_length_payload_is_rejected() {
        // The kinds must never alias: a zero-length verbatim record is an
        // empty-delta record spelled a second way, and a canonical encoding
        // admits exactly one spelling per logical record (SR-1.2).
        let (bytes, _) = encoded(0, 0, &[TailRecordKind::EmptyDelta]);
        let artifact = patched(
            &bytes,
            TAIL_HEADER_SIZE + RECORD_KIND_OFFSET,
            &[TailRecordKind::Verbatim.as_u8()],
        );

        let err = decode_tail_envelope(&artifact).unwrap_err();

        assert!(
            matches!(err, TailCodecError::EmptyVerbatimPayload { index: 0 }),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn encode_rejects_verbatim_record_with_empty_payload() {
        // Encode-side twin of the test above: the aliasing shape is refused at
        // both ends, so it can never be minted in the first place.
        let header = TailHeader {
            object_id: OBJ,
            base_snap_seq: 4,
            from_seq: 4,
            to_seq: 5,
        };

        let err = encode_tail_envelope(&header, vec![verbatim_with(5, TS, b"")]).unwrap_err();

        assert!(
            matches!(err, TailCodecError::EmptyVerbatimPayload { index: 0 }),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn codec_reports_the_offending_record_index_not_zero() {
        // Every kind/payload aliasing rejection above is asserted at record 0,
        // where a hardcoded `index: 0` — or an index taken from the wrong loop
        // variable — is indistinguishable from the real one. That index is how
        // an operator locates the damaged record in a tail carrying thousands,
        // so it is pinned at a non-zero position here, on both sides of the
        // codec.
        let kinds = [
            TailRecordKind::Verbatim,
            TailRecordKind::EmptyDelta,
            TailRecordKind::Verbatim,
        ];
        let (bytes, records) = encoded(3, 20, &kinds);
        let offsets = record_offsets(&records);
        let payload_len = ompf_bytes(&records[2]).len() as u32;
        assert!(payload_len > 0, "fixture record 2 carries payload bytes");

        // Decode, case 1: record 2's kind flipped to empty-delta while its
        // payload bytes are still present.
        let mislabelled = patched(
            &bytes,
            offsets[2] + RECORD_KIND_OFFSET,
            &[TailRecordKind::EmptyDelta.as_u8()],
        );
        let err = decode_tail_envelope(&mislabelled).unwrap_err();
        assert!(
            matches!(
                err,
                TailCodecError::NonEmptyEmptyDeltaPayload { index: 2, len } if len == payload_len
            ),
            "mislabelled record 2: unexpected error: {err:?}"
        );

        // Decode, case 2: record 1's kind flipped to verbatim while it carries
        // no payload.
        let mislabelled = patched(
            &bytes,
            offsets[1] + RECORD_KIND_OFFSET,
            &[TailRecordKind::Verbatim.as_u8()],
        );
        let err = decode_tail_envelope(&mislabelled).unwrap_err();
        assert!(
            matches!(err, TailCodecError::EmptyVerbatimPayload { index: 1 }),
            "mislabelled record 1: unexpected error: {err:?}"
        );

        // Encode: the same index discipline on the write side, where the index
        // comes from the record iterator rather than a byte offset.
        let (header, mut supplied) = valid_envelope(OBJ, 3, 20, &kinds);
        supplied[2] = verbatim_with(23, TS, b"");
        let err = encode_tail_envelope(&header, supplied).unwrap_err();
        assert!(
            matches!(err, TailCodecError::EmptyVerbatimPayload { index: 2 }),
            "encode of an empty verbatim at record 2: unexpected error: {err:?}"
        );
    }

    #[test]
    fn decode_record_length_overrunning_buffer_returns_typed_error() {
        // Declared length one byte past the end of the buffer: the framing is
        // off by the smallest possible amount, which is exactly the case a
        // bounds check written with `<=` instead of `<` would let through.
        let (bytes, records) = encoded(0, 0, &[TailRecordKind::Verbatim]);
        let actual = records[0].ompf().expect("verbatim").len();
        let artifact = patched(
            &bytes,
            TAIL_HEADER_SIZE + RECORD_PAYLOAD_LEN_OFFSET,
            &(actual as u32 + 1).to_le_bytes(),
        );

        let err = decode_tail_envelope(&artifact).unwrap_err();

        assert!(
            matches!(
                err,
                TailCodecError::Truncated {
                    at: "record payload",
                    needed,
                    available,
                } if needed == actual + 1 && available == actual
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn decode_trailing_bytes_after_last_record_returns_typed_error() {
        // A decoder that stopped at `record_count` and ignored the rest would
        // silently accept a corrupted or concatenated artifact. The encoding is
        // canonical, so slack means these are not the bytes the encoder would
        // have produced.
        let (with_records, _) = encoded(
            3,
            20,
            &[TailRecordKind::Verbatim, TailRecordKind::EmptyDelta],
        );
        let (empty_tail, _) = encoded(40, 40, &[]);

        for (label, valid) in [
            ("record-bearing", with_records),
            ("zero-length", empty_tail),
        ] {
            let artifact = appended(&valid, b"!");

            let err = decode_tail_envelope(&artifact).unwrap_err();

            assert!(
                matches!(err, TailCodecError::TrailingBytes { extra: 1 }),
                "{label}: unexpected error: {err:?}"
            );

            // A whole extra record's worth of slack is the same defect, and is
            // the shape a truncated-then-reappended artifact would take.
            let artifact = appended(&valid, &[0u8; TAIL_RECORD_PREFIX_SIZE]);

            let err = decode_tail_envelope(&artifact).unwrap_err();

            assert!(
                matches!(
                    err,
                    TailCodecError::TrailingBytes { extra } if extra == TAIL_RECORD_PREFIX_SIZE
                ),
                "{label}: unexpected error: {err:?}"
            );
        }
    }

    #[test]
    fn decode_record_count_disagreeing_with_header_range_is_rejected() {
        // SR-1.4 / SR-2.1: the header's `(from_seq, to_seq]` span and the framed
        // record count are two statements of the same fact, cross-checked before
        // a single record is read. A header claiming coverage its framing cannot
        // supply is the b.pw6 defect.
        let kinds = [TailRecordKind::Verbatim; 4];
        let (bytes, _) = encoded(3, 20, &kinds);
        let range_span = kinds.len() as u64;

        for forged in [range_span - 1, range_span + 1, 0] {
            let artifact = patched(
                &bytes,
                TAIL_RECORD_COUNT_OFFSET,
                &(forged as u32).to_le_bytes(),
            );

            let err = decode_tail_envelope(&artifact).unwrap_err();

            assert!(
                matches!(
                    err,
                    TailCodecError::RecordCountMismatch {
                        range_span: span,
                        record_count
                    } if span == range_span && record_count == forged
                ),
                "record_count {forged}: unexpected error: {err:?}"
            );
        }
    }

    #[test]
    fn decode_record_sequence_outside_header_range_is_rejected() {
        // Records cover `(from_seq, to_seq]` with exactly one entry per sequence
        // number, strictly ascending. A record whose sequence falls outside the
        // claimed run — above it, below it, or repeating its predecessor — makes
        // the header's range claim a lie, so it is rejected rather than replayed.
        let (bytes, records) = encoded(
            3,
            20,
            &[
                TailRecordKind::Verbatim,
                TailRecordKind::Verbatim,
                TailRecordKind::Verbatim,
            ],
        );
        let offsets = record_offsets(&records);

        // (record index, forged sequence, the sequence the position requires)
        let cases = [
            (1, 99u64, 22u64), // above the claimed range
            (1, 21, 22),       // repeats its predecessor: not strictly ascending
            (0, 5, 21),        // below `from_seq` entirely
            (2, 22, 23),       // descending relative to its predecessor
        ];

        for (index, forged, expected) in cases {
            let artifact = patched(
                &bytes,
                offsets[index] + RECORD_SEQUENCE_OFFSET,
                &forged.to_le_bytes(),
            );

            let err = decode_tail_envelope(&artifact).unwrap_err();

            assert!(
                matches!(
                    err,
                    TailCodecError::NonContiguousSequence {
                        index: at,
                        expected: want,
                        got
                    } if at == index && want == expected && got == forged
                ),
                "record {index} forged to {forged}: unexpected error: {err:?}"
            );
        }
    }

    #[test]
    fn decode_rejects_forged_header_the_encoder_would_refuse_to_emit() {
        // The decoder re-applies every semantic check the encoder applies rather
        // than trusting that the artifact came from this encoder. A hand-forged
        // or bit-rotted header that the encoder could never have produced must
        // not reach replay.
        let (bytes, _) = encoded(40, 40, &[]);

        // `from_seq` above `to_seq`: not a well-formed range.
        let inverted = patched(&bytes, TO_SEQ_OFFSET, &39u64.to_le_bytes());
        let err = decode_tail_envelope(&inverted).unwrap_err();
        assert!(
            matches!(
                err,
                TailCodecError::InvalidRange {
                    from_seq: 40,
                    to_seq: 39
                }
            ),
            "inverted range: unexpected error: {err:?}"
        );

        // Base snapshot above the range: the tail does not layer onto the
        // snapshot its own header names.
        let floating = patched(&bytes, BASE_SNAP_SEQ_OFFSET, &41u64.to_le_bytes());
        let err = decode_tail_envelope(&floating).unwrap_err();
        assert!(
            matches!(
                err,
                TailCodecError::BaseAfterRange {
                    base_snap_seq: 41,
                    from_seq: 40
                }
            ),
            "base above range: unexpected error: {err:?}"
        );

        // Above the 2^53-1 ceiling the rest of the pipeline (Lua's f64 integer
        // range) cannot represent.
        let over = TAIL_MAX_SEQUENCE + 1;
        let above_ceiling = patched(&bytes, FROM_SEQ_OFFSET, &over.to_le_bytes());
        let above_ceiling = patched(&above_ceiling, TO_SEQ_OFFSET, &over.to_le_bytes());
        let err = decode_tail_envelope(&above_ceiling).unwrap_err();
        assert!(
            matches!(
                err,
                TailCodecError::SequenceCeilingExceeded {
                    sequence_num,
                    max: TAIL_MAX_SEQUENCE
                } if sequence_num == over
            ),
            "above ceiling: unexpected error: {err:?}"
        );
    }

    // ---------------------------------------------------------------------
    // Determinism, versioning, and the committed golden layout.
    //
    // SR-9.2's duplicate-leader safety rests on byte-determinism: two replicas
    // capturing the same immutable stream slice at the same
    // `(object_id, to_seq)` must produce byte-identical artifacts, so the
    // racing PUTs are idempotent rather than conflicting. SR-6.1/SR-6.3 make
    // the layout itself a versioned compatibility surface — a silent layout
    // change without a `TAIL_FORMAT_VERSION` bump would make every
    // already-persisted tail undecodable, or worse, mis-decodable.
    // ---------------------------------------------------------------------

    /// Golden fixture: `object_id` — a value whose eight bytes are all distinct
    /// so a field-order or endianness change cannot alias onto an equal
    /// neighbour.
    const GOLDEN_OBJECT_ID: u64 = 0x0123_4567_89AB_CDEF;
    /// Golden fixture: `base_snap_seq`, below `from_seq` as the format requires.
    const GOLDEN_BASE_SNAP_SEQ: u64 = 0x1111;
    /// Golden fixture: `from_seq`. Distinct from every other header field so a
    /// swapped write order is visible in the bytes.
    const GOLDEN_FROM_SEQ: u64 = 0x2222;
    /// Golden fixture: `to_seq` — two above `from_seq`, so the envelope carries
    /// exactly the two records below.
    const GOLDEN_TO_SEQ: u64 = 0x2224;
    /// Golden fixture: `server_ts` of the first record. Byte-distinct for the
    /// same reason as [`GOLDEN_OBJECT_ID`]; the second record carries this + 1.
    /// Nothing in this codec reads a clock, so the value is arbitrary and fixed.
    const GOLDEN_TS: i64 = 0x0102_0304_0506_0708;
    /// Golden fixture: the verbatim record's payload. Opaque to the codec — it
    /// is deliberately not a real OMPF container, because this module must
    /// never parse one.
    const GOLDEN_PAYLOAD: &[u8] = b"ompf";

    /// The exact bytes [`golden_envelope`] must encode to under
    /// [`TAIL_FORMAT_VERSION`] `== 1`. Grouped one field per line, little-endian
    /// throughout, matching the wire-format table in the module doc.
    ///
    /// Total = 44-byte header + (21-byte prefix + 4-byte payload) + 21-byte
    /// prefix = 90 bytes.
    #[rustfmt::skip]
    const GOLDEN_V1_BYTES: &[u8] = &[
        // -- header (44 bytes) --
        // off  0: magic = b"LETL" (4 literal ASCII bytes, not a u32)
        0x4C, 0x45, 0x54, 0x4C,
        // off  4: format_version = 1 (u32 LE)
        0x01, 0x00, 0x00, 0x00,
        // off  8: object_id = 0x0123_4567_89AB_CDEF (u64 LE)
        0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01,
        // off 16: base_snap_seq = 0x1111 = 4_369 (u64 LE)
        0x11, 0x11, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // off 24: from_seq = 0x2222 = 8_738 (u64 LE)
        0x22, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // off 32: to_seq = 0x2224 = 8_740 (u64 LE)
        0x24, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // off 40: record_count = 2 (u32 LE)
        0x02, 0x00, 0x00, 0x00,
        // -- record 0, Verbatim (21-byte prefix + 4-byte payload) --
        // +0:  kind = 1 (Verbatim)
        0x01,
        // +1:  sequence_num = 0x2223 = 8_739 (u64 LE) — from_seq + 1
        0x23, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // +9:  server_ts = 0x0102_0304_0506_0708 (i64 LE)
        0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
        // +17: payload_len = 4 (u32 LE)
        0x04, 0x00, 0x00, 0x00,
        // +21: payload = b"ompf", stored byte-for-byte with no re-encode
        0x6F, 0x6D, 0x70, 0x66,
        // -- record 1, EmptyDelta (21-byte prefix, no payload) --
        // +0:  kind = 2 (EmptyDelta)
        0x02,
        // +1:  sequence_num = 0x2224 = 8_740 (u64 LE)
        0x24, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // +9:  server_ts = 0x0102_0304_0506_0709 (i64 LE)
        0x09, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01,
        // +17: payload_len = 0 (u32 LE) — an EmptyDelta always writes a zero
        //      here, so every record shares the same prefix stride
        0x00, 0x00, 0x00, 0x00,
    ];

    /// The remedy a golden-layout failure must name. Spelled once and asserted
    /// in every direction, so whoever trips it is told what to do instead of
    /// being tempted to regenerate the constant.
    const GOLDEN_LAYOUT_DRIFT: &str =
        "the v1 tail-envelope wire layout changed — bump `TAIL_FORMAT_VERSION` \
         and add a decode path for the old version (SR-6.1/SR-6.3), or revert \
         the layout change.";

    /// The logical content of [`GOLDEN_V1_BYTES`]: one verbatim record with a
    /// short known payload followed by one empty-delta record.
    ///
    /// Deliberately built from explicit constants rather than [`valid_envelope`]
    /// — the golden must not move when a fixture helper changes.
    fn golden_envelope() -> (TailHeader, Vec<TailRecord>) {
        let header = TailHeader {
            object_id: GOLDEN_OBJECT_ID,
            base_snap_seq: GOLDEN_BASE_SNAP_SEQ,
            from_seq: GOLDEN_FROM_SEQ,
            to_seq: GOLDEN_TO_SEQ,
        };
        let records = vec![
            verbatim_with(GOLDEN_FROM_SEQ + 1, GOLDEN_TS, GOLDEN_PAYLOAD),
            empty_delta_with(GOLDEN_TO_SEQ, GOLDEN_TS + 1),
        ];
        (header, records)
    }

    #[test]
    fn encode_is_byte_identical_across_repeated_calls() {
        // Any wall-clock read, RNG draw, or address-derived byte in the encoder
        // shows up as a difference between two encodes of the same input.
        let kinds = [
            TailRecordKind::Verbatim,
            TailRecordKind::EmptyDelta,
            TailRecordKind::Verbatim,
            TailRecordKind::Verbatim,
        ];
        let (header, records) = valid_envelope(OBJ, 4_000, 4_200, &kinds);

        let first = encode_tail_envelope(&header, records.clone()).expect("first encode");
        // Repeated rather than paired: a once-per-process value (a lazily
        // seeded RNG, a cached `Instant`) would survive a two-call comparison.
        for attempt in 1..8 {
            let again = encode_tail_envelope(&header, records.clone()).expect("repeat encode");
            assert_eq!(
                again, first,
                "encode attempt {attempt} differs from the first — the encoder \
                 depends on something other than its inputs, which breaks \
                 SR-9.2 duplicate-leader idempotence"
            );
        }

        // Round-tripping through the decoder re-encodes from `Bytes::slice`
        // views with entirely different allocation provenance; the bytes must
        // still be identical.
        let decoded = decode_tail_envelope(&first).expect("decode");
        let decoded_header = *decoded.header();
        let re_encoded =
            encode_tail_envelope(&decoded_header, decoded.into_records()).expect("re-encode");
        assert_eq!(
            re_encoded, first,
            "re-encoding a decoded envelope must reproduce the same bytes"
        );
    }

    #[test]
    fn encode_is_byte_identical_for_independently_constructed_equal_input() {
        use std::collections::VecDeque;

        let kinds = [
            TailRecordKind::Verbatim,
            TailRecordKind::EmptyDelta,
            TailRecordKind::Verbatim,
        ];
        // Two separate constructions: `verbatim_with` copies each payload onto
        // a fresh allocation, so the second list shares no memory with the
        // first even though it is logically equal.
        let (header_a, records_a) = valid_envelope(OBJ, 12, 900, &kinds);
        let (header_b, records_b) = valid_envelope(OBJ, 12, 900, &kinds);
        assert_eq!(header_a, header_b, "fixture: headers are logically equal");
        assert_eq!(records_a, records_b, "fixture: records are logically equal");
        assert_ne!(
            records_a.as_ptr(),
            records_b.as_ptr(),
            "fixture: the two record lists must be separate allocations"
        );

        let from_vec = encode_tail_envelope(&header_a, records_a.clone()).expect("encode from Vec");

        // Different container type feeding the same records. The encoder takes
        // a plain `IntoIterator`, so nothing about the container — including
        // its `size_hint` — may reach the output bytes.
        let mut deque: VecDeque<TailRecord> = records_b.iter().cloned().collect();
        let from_deque =
            encode_tail_envelope(&header_b, deque.drain(..)).expect("encode from drain");
        assert_eq!(
            from_deque, from_vec,
            "a VecDeque drain and a Vec must encode identically"
        );

        // A lazy iterator with an unknown length (`size_hint == (0, None)`),
        // proving the backfilled `record_count` comes from the records actually
        // yielded and not from a caller-supplied size hint.
        let mut pending: VecDeque<TailRecord> = records_b.into_iter().collect();
        let from_lazy =
            encode_tail_envelope(&header_b, std::iter::from_fn(move || pending.pop_front()))
                .expect("encode from lazy iterator");
        assert_eq!(
            from_lazy, from_vec,
            "an unsized lazy iterator must encode identically to a Vec"
        );

        // Payload provenance must not leak either: a `Bytes` promoted from a
        // `'static` slice and one copied onto the heap have different internal
        // representations and different addresses, but identical contents.
        let header = TailHeader {
            object_id: OBJ,
            base_snap_seq: 1,
            from_seq: 1,
            to_seq: 2,
        };
        let static_payload = encode_tail_envelope(
            &header,
            vec![TailRecord::Verbatim {
                sequence_num: 2,
                server_ts: TS,
                ompf: Bytes::from_static(b"ompf-payload"),
            }],
        )
        .expect("encode with a static payload");
        let heap_payload =
            encode_tail_envelope(&header, vec![verbatim_with(2, TS, b"ompf-payload")])
                .expect("encode with a heap payload");
        assert_eq!(
            static_payload, heap_payload,
            "the `Bytes` representation of a payload must not reach the output"
        );
    }

    #[test]
    fn encoded_v1_layout_matches_golden_bytes() {
        let (header, records) = golden_envelope();

        let bytes = encode_tail_envelope(&header, records).expect("golden fixture encodes");

        assert_eq!(bytes.as_ref(), GOLDEN_V1_BYTES, "{GOLDEN_LAYOUT_DRIFT}");
        // Cross-check the framing constants against the golden's own length, so
        // a header or prefix resize cannot be "fixed" by editing one constant.
        assert_eq!(
            GOLDEN_V1_BYTES.len(),
            TAIL_HEADER_SIZE
                + (TAIL_RECORD_PREFIX_SIZE + GOLDEN_PAYLOAD.len())
                + TAIL_RECORD_PREFIX_SIZE,
            "{GOLDEN_LAYOUT_DRIFT}"
        );
    }

    #[test]
    fn decode_of_golden_bytes_yields_the_expected_logical_content() {
        // The reverse direction of the golden: regenerating the constant from a
        // broken encoder would satisfy `encoded_v1_layout_matches_golden_bytes`
        // but break the literal expectations below.
        let decoded =
            decode_tail_envelope(&Bytes::from_static(GOLDEN_V1_BYTES)).expect("golden decodes");

        assert_eq!(
            decoded.format_version(),
            1,
            "the golden is a v1 artifact; {GOLDEN_LAYOUT_DRIFT}"
        );
        assert_eq!(decoded.header().object_id, 0x0123_4567_89AB_CDEF);
        assert_eq!(decoded.header().base_snap_seq, 0x1111);
        assert_eq!(decoded.header().from_seq, 0x2222);
        assert_eq!(decoded.header().to_seq, 0x2224);

        assert_eq!(decoded.len(), 2, "the golden carries two records");
        assert_eq!(
            kinds_of(&decoded),
            vec![TailRecordKind::Verbatim, TailRecordKind::EmptyDelta],
            "record kinds, in order"
        );

        let first = &decoded.records()[0];
        assert_eq!(first.sequence_num(), 0x2223);
        assert_eq!(first.server_ts(), 0x0102_0304_0506_0708);
        assert_eq!(ompf_bytes(first), b"ompf".as_slice());

        let second = &decoded.records()[1];
        assert_eq!(second.sequence_num(), 0x2224);
        assert_eq!(second.server_ts(), 0x0102_0304_0506_0709);
        assert!(second.ompf().is_none(), "an EmptyDelta carries no payload");

        // And the whole decoded value equals the fixture the golden encodes
        // from, closing the loop in both directions.
        let (header, records) = golden_envelope();
        assert_eq!(decoded.header(), &header);
        assert_eq!(decoded.records(), records.as_slice());

        // `peek_tail_header` is the SR-3.1 lineage-gather entry point and reads
        // the same fixed header without materializing records, so a layout
        // change that moved a header field must fail here too, not only in the
        // full decode.
        let (peeked_version, peeked_header) =
            peek_tail_header(GOLDEN_V1_BYTES).expect("golden header peeks");
        assert_eq!(peeked_version, 1, "{GOLDEN_LAYOUT_DRIFT}");
        assert_eq!(peeked_header, header, "{GOLDEN_LAYOUT_DRIFT}");
    }

    #[test]
    fn current_format_version_is_one() {
        // Pinned so a bump is a deliberate, visible edit that also forces the
        // golden above to be revisited (SR-6.1/SR-6.3).
        assert_eq!(
            TAIL_FORMAT_VERSION, 1,
            "TAIL_FORMAT_VERSION changed: every already-persisted tail was \
             written at the old version, so a bump must be accompanied by a \
             decode path for it and a second golden fixture."
        );
        // The remaining wire constants, pinned alongside it for the same reason.
        assert_eq!(TAIL_ENVELOPE_MAGIC, *b"LETL", "{GOLDEN_LAYOUT_DRIFT}");
        assert_eq!(TAIL_HEADER_SIZE, 44, "{GOLDEN_LAYOUT_DRIFT}");
        assert_eq!(TAIL_RECORD_PREFIX_SIZE, 21, "{GOLDEN_LAYOUT_DRIFT}");
        assert_eq!(TAIL_FORMAT_VERSION_OFFSET, 4, "{GOLDEN_LAYOUT_DRIFT}");
        assert_eq!(TAIL_RECORD_COUNT_OFFSET, 40, "{GOLDEN_LAYOUT_DRIFT}");

        // The encoder stamps the constant rather than accepting one from the
        // caller, so this build can never mint an artifact it cannot read back.
        let (header, records) = golden_envelope();
        let bytes = encode_tail_envelope(&header, records).expect("encode");
        let stamped = u32::from_le_bytes(
            bytes[TAIL_FORMAT_VERSION_OFFSET..TAIL_FORMAT_VERSION_OFFSET + 4]
                .try_into()
                .expect("4 bytes"),
        );
        assert_eq!(
            stamped, TAIL_FORMAT_VERSION,
            "the encoder must stamp the current format version"
        );
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    #[test]
    fn source_has_no_wall_clock_or_io_or_ffi_references() {
        // Mechanical source-scanning guard, following the in-repo precedent at
        // `src/domain/ompf.rs` (`source_has_no_try_into_unwrap_on_input_slices`,
        // which uses `include_str!`). Determinism and no-I/O are Task Success
        // Criteria: they are what makes two replicas' artifacts for the same
        // stream slice byte-identical, and they are reviewable by inspection
        // only if the module genuinely never names these things.
        //
        // Each banned token, and why:
        const BANNED: &[(&str, &str)] = &[
            // An await point puts the codec on an executor and opens the door
            // to I/O; this module is a pure function of its inputs.
            ("async fn", "no async: the codec must stay a pure function"),
            // A wall-clock read would make two replicas encoding the same
            // immutable slice produce different bytes, breaking SR-9.2's
            // duplicate-leader idempotence. Record timestamps are carried in
            // from the captured stream entries, never read from a clock.
            (
                "SystemTime",
                "no clock reads: output must depend only on inputs",
            ),
            (
                "Instant::now",
                "no clock reads: output must depend only on inputs",
            ),
            // The in-crate clock helper the module doc names as *the* hazard
            // (`src/checkpoint/mod.rs`). It is a plain function call, so
            // neither `SystemTime` nor `Instant::now` above would catch it.
            (
                "now_unix_ms",
                "no clock reads: record timestamps come from the captured \
                 stream entries",
            ),
            // Randomness is the third clause of the determinism contract, and
            // was the one clause with no guard.
            ("rand::", "no randomness: output must depend only on inputs"),
            // Persistence belongs to the adapters; this module owns bytes only.
            (
                "std::fs",
                "no filesystem: adapters own persistence, not the codec",
            ),
            ("tokio::", "no runtime: the codec is sync and I/O-free"),
            // No backing store: a codec that could reach Valkey would no longer
            // be a total function of its input bytes.
            ("fred::", "no Valkey client in the codec"),
            // Verbatim payloads stay opaque (SR-2.2: no re-encode, no
            // parse-and-rebuild), so the FlatBuffers FFI must not appear here.
            //
            // `cxx::` alone is close to vacuous for this crate: `oodf-ffi`'s
            // bridge is `pub(crate)` and the `internal-bridge` feature is NOT
            // enabled on live-edit's dependency edge, so live-edit code cannot
            // name `cxx::` even if it tried. The reachable spellings are the
            // ones below, which is how a future edit would actually reach the
            // OODF stack from here.
            ("cxx::", "no FlatBuffers FFI: payload bytes stay opaque"),
            (
                "oodf_ffi",
                "no OODF parse/build: a tail stores the writer's raw bytes",
            ),
            (
                "OmpfContainer",
                "no OMPF parsing: payload bytes stay opaque to this codec",
            ),
            // The adapter seam. `CheckpointError` — the leaf error vocabulary
            // declared alongside it — is imported on purpose for the
            // classification impl; the trait and its implementors are not.
            (
                "CheckpointStore",
                "no adapter types: the codec owns bytes, adapters own I/O",
            ),
            // The nondeterministic-iteration hazard the Task names explicitly.
            // `BTreeMap` or a `Vec` is fine; a hash container is not, because
            // its iteration order is not a function of its contents.
            (
                "HashMap",
                "nondeterministic iteration order; use BTreeMap or Vec",
            ),
            (
                "HashSet",
                "nondeterministic iteration order; use BTreeSet or Vec",
            ),
        ];

        // Scan only the pre-test-module slice: the `#[cfg(test)] mod tests`
        // block legitimately names some of these (this very list, and the
        // decode-rejection tests' use of `OmpfContainer`), and test-only code
        // is not part of the shipped codec.
        let src = include_str!("tail_codec.rs");
        let cut = src
            .find("#[cfg(test)]")
            .expect("the module must have a #[cfg(test)] block to cut at");
        let non_test = &src[..cut];

        // Strip `//`, `///` and `//!` comments before matching: the module's own
        // determinism-contract doc *names* every banned token in order to forbid
        // it, and a naive substring scan would trip on the prose.
        assert!(
            !non_test.contains("/*"),
            "this guard strips line comments only; a block comment would slip \
             banned tokens past it"
        );
        let code: String = non_test
            .lines()
            .map(|line| line.split("//").next().unwrap_or(line))
            .collect::<Vec<_>>()
            .join("\n");

        for (token, reason) in BANNED {
            assert!(
                !code.contains(token),
                "tail_codec.rs must not reference `{token}` outside its test \
                 module ({reason}). Determinism and no-I/O are Task Success \
                 Criteria for this codec."
            );
        }

        // Sanity: the stripper must not have eaten the code it is meant to
        // scan, or the assertions above would pass vacuously.
        assert!(
            code.contains("pub fn encode_tail_envelope"),
            "comment stripping removed real code; the scan would be vacuous"
        );
        assert!(
            code.contains("pub fn decode_tail_envelope"),
            "comment stripping removed real code; the scan would be vacuous"
        );
    }

    #[test]
    fn record_kind_discriminants_are_pinned_and_total_over_every_byte() {
        // The decode-rejection test samples three invalid kind bytes through
        // the decoder; this closes the mapping over all 256 of them, so no
        // other byte can ever be admitted as a record kind. The two valid
        // discriminants are part of the wire format and cannot change without
        // a `TAIL_FORMAT_VERSION` bump.
        assert_eq!(TailRecordKind::Verbatim.as_u8(), 1, "{GOLDEN_LAYOUT_DRIFT}");
        assert_eq!(
            TailRecordKind::EmptyDelta.as_u8(),
            2,
            "{GOLDEN_LAYOUT_DRIFT}"
        );

        for raw in 0u8..=u8::MAX {
            match TailRecordKind::from_u8(raw) {
                Some(kind) => {
                    assert!(
                        raw == 1 || raw == 2,
                        "byte {raw} decoded as {kind:?}; only 1 and 2 are valid \
                         record kinds. {GOLDEN_LAYOUT_DRIFT}"
                    );
                    assert_eq!(kind.as_u8(), raw, "byte {raw}: as_u8 must invert from_u8");
                }
                None => assert!(
                    raw != 1 && raw != 2,
                    "byte {raw} must map to a record kind. {GOLDEN_LAYOUT_DRIFT}"
                ),
            }
        }
    }

    // ---------------------------------------------------------------------
    // Real-traffic corpus (SR-2.2 byte-transparency, SR-6.2 real fixtures)
    //
    // The tests above build minimal hand-made payloads to pin edge and
    // boundary shapes. These prove the same contract against the bytes the
    // service actually receives: `tools/fixtures/gze_inbound_deltas.json` is
    // the committed capture of 36 real inbound Kit messages (b.gze / b.zju).
    // Real captured traffic is non-optional here: synthetic single-section
    // fixtures are exactly what hid both of those defects.
    //
    // None of these tests parse a payload. `OmpfContainer::parse` and
    // `oodf_ffi` are deliberately absent: SR-2.2 says the tail stores the
    // writer's raw bytes "with no re-encode and no parse-and-rebuild", so the
    // property under test is that the codec is byte-transparent for opaque
    // input, and reaching into the OODF stack here would both slow the unit
    // test and couple it to a layer this module must never touch.
    // ---------------------------------------------------------------------

    /// Timestamp stamped on records built from the real corpus. The fixture's
    /// own `ts` field is a wall-clock time-of-day *string* from the capture
    /// tool, not the microsecond `server_ts` a stream entry carries, so these
    /// tests supply a deterministic synthetic value rather than inventing a
    /// parse the format does not define.
    const REAL_TRAFFIC_TS: i64 = 1_700_000_000_000_000;

    /// The raw inbound OMPF payloads of the committed real-traffic capture, in
    /// capture order.
    ///
    /// The fixture is embedded with `include_str!`, so these stay unit tests:
    /// no runtime filesystem I/O, no dependence on the process's working
    /// directory. `serde_json` and `base64` are normal (not dev-only)
    /// dependencies of this crate, so no new dependency edge is created.
    ///
    /// Every guard below is an anti-vacuity guard. A fixture-path typo cannot
    /// compile at all; a schema change, a re-capture that empties the corpus,
    /// or a truncated `b64` field must fail loudly here rather than silently
    /// reducing the tests that call this to assertions over an empty `Vec`.
    fn real_inbound_ompf_payloads() -> Vec<Vec<u8>> {
        use base64::{engine::general_purpose::STANDARD as B64, Engine};

        const FIXTURE: &str = include_str!("../../tools/fixtures/gze_inbound_deltas.json");

        let entries: Vec<serde_json::Value> = serde_json::from_str(FIXTURE)
            .expect("real-traffic fixture must parse as a JSON array of capture entries");
        assert!(
            !entries.is_empty(),
            "real-traffic fixture is empty — SR-6.2 requires these tests to run \
             against real captured writer traffic, and an empty corpus makes \
             every assertion below vacuous"
        );

        let payloads: Vec<Vec<u8>> = entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                let b64 = entry["b64"]
                    .as_str()
                    .unwrap_or_else(|| panic!("fixture entry {index}: missing `b64` string field"));
                let declared = entry["len"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("fixture entry {index}: missing `len` number field"));
                let bytes = B64
                    .decode(b64)
                    .unwrap_or_else(|err| panic!("fixture entry {index}: base64 decode: {err}"));
                // The recorded `len` is the capture tool's independent record of
                // how many bytes went over the wire, so a mismatch means the
                // corpus or its schema changed under us.
                assert_eq!(
                    bytes.len() as u64,
                    declared,
                    "fixture entry {index}: decoded payload length disagrees with \
                     the capture's recorded `len`"
                );
                assert!(
                    !bytes.is_empty(),
                    "fixture entry {index}: a real inbound OMPF message is never \
                     zero bytes, and a Verbatim record may not carry an empty \
                     payload"
                );
                bytes
            })
            .collect();

        assert!(
            payloads.len() >= 30,
            "real-traffic corpus shrank to {} payloads; the committed capture \
             carries 36 and these tests are sized against it",
            payloads.len()
        );
        // Variable-length payloads are what make a mis-framing decoder fail:
        // with a fixed stride, reading record n+1 at the wrong offset could
        // still yield the right bytes.
        let shortest = payloads.iter().map(Vec::len).min().expect("non-empty");
        let longest = payloads.iter().map(Vec::len).max().expect("non-empty");
        assert!(
            shortest < longest,
            "real-traffic corpus payloads are all {shortest} bytes; a uniform \
             corpus cannot catch a record mis-framing"
        );

        payloads
    }

    /// One `Verbatim` record per corpus payload, numbered contiguously from
    /// `from_seq` per the envelope's `(from_seq, to_seq]` coverage rule, with
    /// the matching header.
    fn real_traffic_envelope(
        base_snap_seq: u64,
        from_seq: u64,
        payloads: &[Vec<u8>],
    ) -> (TailHeader, Vec<TailRecord>) {
        let records: Vec<TailRecord> = payloads
            .iter()
            .enumerate()
            .map(|(index, payload)| {
                verbatim_with(from_seq + 1 + index as u64, REAL_TRAFFIC_TS, payload)
            })
            .collect();
        let header = TailHeader {
            object_id: OBJ,
            base_snap_seq,
            from_seq,
            to_seq: from_seq + records.len() as u64,
        };
        (header, records)
    }

    #[test]
    fn real_captured_ompf_payloads_roundtrip_byte_identically() {
        let payloads = real_inbound_ompf_payloads();
        // `from_seq == 0`, so record n carries `sequence_num == n + 1`.
        let (header, records) = real_traffic_envelope(0, 0, &payloads);

        let bytes =
            encode_tail_envelope(&header, records.clone()).expect("real captured corpus encodes");
        let decoded = decode_tail_envelope(&bytes).expect("real captured corpus decodes");

        assert_eq!(
            decoded.len(),
            payloads.len(),
            "every captured message must survive as its own record"
        );
        assert_eq!(decoded.header(), &header, "lineage header round-trips");

        for (index, (record, payload)) in decoded.iter().zip(payloads.iter()).enumerate() {
            assert_eq!(
                record.kind(),
                TailRecordKind::Verbatim,
                "record {index}: kind"
            );
            assert_eq!(
                record.sequence_num(),
                index as u64 + 1,
                "record {index}: sequence_num"
            );
            assert_eq!(record.server_ts(), REAL_TRAFFIC_TS, "record {index}: ts");
            // The SR-2.2 contract, measured on real bytes: what comes out is
            // what went in, byte for byte.
            assert_eq!(
                ompf_bytes(record),
                payload.as_slice(),
                "record {index}: real captured OMPF bytes were altered by the \
                 codec — SR-2.2 requires storage with no re-encode and no \
                 parse-and-rebuild"
            );
        }

        // Independently: the artifact is exactly framing plus payloads. A
        // codec that re-encoded, compressed, or padded a payload would still
        // satisfy per-record equality only by accident, but cannot satisfy
        // this size identity.
        let payload_bytes: usize = payloads.iter().map(Vec::len).sum();
        assert_eq!(
            bytes.len(),
            TAIL_HEADER_SIZE + payloads.len() * TAIL_RECORD_PREFIX_SIZE + payload_bytes,
            "encoded size must be the fixed framing plus the payloads verbatim"
        );
    }

    #[test]
    fn real_captured_corpus_encoding_is_deterministic() {
        // SR-9.2's duplicate-leader property is about real tails, not 3-byte
        // ones: two replicas capturing the same immutable stream slice must
        // produce byte-identical artifacts so the racing PUTs are idempotent.
        let payloads = real_inbound_ompf_payloads();
        let (header, records) = real_traffic_envelope(120, 900, &payloads);

        let first = encode_tail_envelope(&header, records.clone()).expect("first encode");
        let second = encode_tail_envelope(&header, records).expect("second encode");
        assert_eq!(
            second, first,
            "two encodes of the real captured corpus differ — the encoder \
             depends on something other than its inputs, which breaks SR-9.2 \
             duplicate-leader idempotence"
        );

        // A second, independent load of the corpus: the payload `Bytes` are
        // freshly allocated and share no memory with the first run's, standing
        // in for the other replica that never saw this process's allocations.
        let (other_header, other_records) =
            real_traffic_envelope(120, 900, &real_inbound_ompf_payloads());
        let other_replica =
            encode_tail_envelope(&other_header, other_records).expect("independent encode");
        assert_eq!(
            other_replica, first,
            "an independently constructed but logically equal corpus must \
             encode to the same bytes"
        );
    }

    #[test]
    fn real_captured_corpus_mixed_with_empty_delta_records_roundtrips() {
        // Real histories interleave the §8.4 empty-commit shape (SR-1.2)
        // between content-bearing commits, so the mixed sequence is the
        // realistic tail shape, not the verbatim-only one.
        let payloads = real_inbound_ompf_payloads();
        let base_snap_seq = 4_096;
        let from_seq = 4_096;

        let mut records = Vec::with_capacity(payloads.len() * 2);
        let mut expected_kinds = Vec::with_capacity(payloads.len() * 2);
        for (index, payload) in payloads.iter().enumerate() {
            let verbatim_seq = from_seq + 1 + (index as u64 * 2);
            // Per-record timestamps vary, so a decoder that preserved the
            // record count while shuffling records fails here too.
            let ts = REAL_TRAFFIC_TS + index as i64;
            records.push(verbatim_with(verbatim_seq, ts, payload));
            expected_kinds.push(TailRecordKind::Verbatim);
            records.push(empty_delta_with(verbatim_seq + 1, ts));
            expected_kinds.push(TailRecordKind::EmptyDelta);
        }
        let header = TailHeader {
            object_id: OBJ,
            base_snap_seq,
            from_seq,
            to_seq: from_seq + records.len() as u64,
        };

        let bytes =
            encode_tail_envelope(&header, records.clone()).expect("mixed real corpus encodes");
        let decoded = decode_tail_envelope(&bytes).expect("mixed real corpus decodes");

        assert_eq!(
            kinds_of(&decoded),
            expected_kinds,
            "the full ordered kind sequence must survive: an empty delta may \
             never be reclassified as a verbatim commit or vice versa"
        );
        assert_eq!(
            decoded
                .iter()
                .map(TailRecord::sequence_num)
                .collect::<Vec<_>>(),
            (from_seq + 1..=header.to_seq).collect::<Vec<_>>(),
            "records decode in ascending contiguous order across both kinds"
        );

        for (index, payload) in payloads.iter().enumerate() {
            let record = &decoded.records()[index * 2];
            assert_eq!(
                ompf_bytes(record),
                payload.as_slice(),
                "record {}: real captured OMPF bytes must survive interleaving \
                 with empty-delta records",
                index * 2
            );
            assert_eq!(
                record.server_ts(),
                REAL_TRAFFIC_TS + index as i64,
                "record {}: server_ts",
                index * 2
            );
            assert!(
                decoded.records()[index * 2 + 1].ompf().is_none(),
                "record {}: an empty delta carries no payload",
                index * 2 + 1
            );
        }

        // And the whole record list, in order, as one comparison.
        assert_eq!(decoded.records(), records.as_slice());
    }
}
