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

//! The chunked base artifact (LEBC): how a checkpoint base too large for one
//! OODF delta is written and read back.
//!
//! # When it is used
//!
//! A base checkpoint is a whole layer serialized as one reset delta
//! (`is_diff = true`, `base_version = 0`). One OODF delta carries at most
//! [`MAX_DELTA_ENTRIES`] entries (struct commands + set fields + time
//! samples), because `oodf_ffi::parse_delta` — the only decoder of untrusted
//! bytes, which restore also uses — refuses anything larger. A layer has no
//! size cap, so its base can exceed that.
//!
//! [`encode_reset_base`] therefore writes:
//!
//! - **at or below [`MAX_DELTA_ENTRIES`] entries** — the single OMPF-wrapped
//!   reset delta every base has always been, byte for byte;
//! - **above it** — a chunked base: the same reset delta cut into consecutive
//!   runs of at most [`MAX_DELTA_ENTRIES`] entries, each built as its own OODF
//!   delta, framed by the codec below.
//!
//! The three producers (the checkpoint assembler, the creation-time bootstrap
//! and the re-base) reach it through `assembler::build_reset_ompf_from_state`;
//! every reader of base bytes (the restore decode and compose) reads it
//! through `apply::decode_checkpoint_ompf`, which dispatches on the leading
//! magic. Adapters, artifact names, sidecars, retention and reconciliation move
//! the bytes opaquely and do not distinguish the two shapes.
//!
//! A chunked base decodes, but on-access restore does not restore it. On both
//! arms, the layer recovery hook refuses it as `Permanent` before decoding or
//! writing anything (SR-14.7, `recovery::refuse_over_limit_base_on_access`),
//! once the layer's identity verdict finds it still mapped; a layer deleted or
//! replaced during the state loss is answered not-found instead
//! (`RecoveryHook::adjudicate_over_limit_base_on_access`).
//! Because a chunked base now decodes, refusing it before the decode is what
//! keeps it away from the restore's single write. There is no step-down,
//! following SR-14.7's rule. In the jumped-base shape of SDD §22.20.3, that
//! gives up an older base the selection would otherwise have reached, even when
//! that base is restorable.
//!
//! An earlier release's plain OMPF base over the limit is refused the same
//! way, with no step-down. The base-only arm fails it at apply's decode. The
//! composed arm refuses it when the composition's base decode reports the
//! parser's typed over-limit error (`apply::is_over_entry_limit_base`).
//!
//! # Older releases
//!
//! A release older than this format cannot read a chunked base: it does not
//! know the magic and refuses the artifact as a malformed OMPF. Nothing it
//! could restore is lost by that — a base over the cap was never restorable by
//! those releases, whose decoder refuses any delta over the cap.
//!
//! # Decoded state equals the unchunked decode
//!
//! The cut follows the plan's own order — every struct command, then every set
//! field, then every time sample — and each run is built by the same
//! `build_delta` with the whole-state external-value index map, so each entry
//! is encoded exactly as an unchunked base would encode it. The first chunk
//! carries the layer's `source_format`; later chunks carry an empty one.
//!
//! [`decode_chunked_reset_delta`] parses every chunk through the unchanged,
//! capped `parse_delta`, checks each is a reset delta, and concatenates the
//! three vectors in chunk order into one [`OwnedDelta`] — exactly the delta
//! the unchunked base would have carried. The caller applies it **once**, so
//! version stamps, the re-derived `set_order` cursor and children order come
//! out identical to an unchunked decode. (Applying chunk by chunk would not:
//! each apply bumps the version.) That one delta can exceed
//! [`MAX_DELTA_ENTRIES`], which is why `decode_checkpoint_ompf` applies it in
//! `ApplyMode::RestoreBase`.
//!
//! # Wire format (version 1)
//!
//! ```text
//! Header (fixed 12 bytes, little-endian):
//!   off  size  field
//!   0    4     magic           = b"LEBC"   (4 literal ASCII bytes, not a u32)
//!   4    4     format_version  u32 LE      (currently 1)
//!   8    4     chunk_count     u32 LE      (MUST be > 0)
//!
//! Chunk (4-byte prefix + payload), repeated chunk_count times:
//!   0    4     chunk_len       u32 LE      (MUST be > 0)
//!   4    ..    chunk           chunk_len bytes: one OODF delta (FlatBuffers,
//!                              "OODF" identifier), no OMPF framing, carrying
//!                              at most MAX_DELTA_ENTRIES entries
//! ```
//!
//! A base carries no external-value parts (the ext-refs sidecar does), so the
//! chunks are bare OODF deltas.
//!
//! # Format decisions
//!
//! - **Its own magic, `b"LEBC"`** ("Live Edit Base, Chunked"), distinct from
//!   OMPF's `b"OMPF"` and the tail envelope's `b"LETL"`, so the decoder's
//!   dispatch is unambiguous and a hexdump is self-describing. Literal bytes,
//!   as in `tail_codec`.
//! - **`format_version` is checked immediately after the magic, before any
//!   length check**, so a future version surfaces
//!   [`ChunkedBaseError::UnknownFormatVersion`], never a truncation or a
//!   mis-parse.
//! - **Canonical encoding.** Fixed field order, no optional fields, no padding,
//!   empty chunks refused on both sides, trailing bytes refused on decode. The
//!   chunk boundaries depend only on the plan's total order and the per-chunk
//!   limit, so equal state yields byte-identical artifacts — the property the
//!   duplicate-leader re-base relies on.
//! - **No allocation from a claimed count.** `chunk_count` is bounded by the
//!   bytes that remain before anything is allocated.
//! - **Eager and total decode.** Every framing defect and every chunk's parse
//!   or shape failure surfaces before the caller applies anything. Each
//!   failure inside a chunk — truncation, an empty chunk, a parse failure, an
//!   over-limit chunk, a non-reset chunk, a misplaced source format — names the
//!   chunk's zero-based index; header defects and trailing bytes describe the
//!   artifact as a whole.
//! - **No CRC**, for the same reason as `tail_codec`: adapters write atomically
//!   and every structural invariant here is self-checking.
//!
//! [`MAX_DELTA_ENTRIES`]: crate::domain::limits::MAX_DELTA_ENTRIES

use std::collections::HashMap;
use std::ops::Range;

use oodf_ffi::{OodfError, OwnedDelta};
use thiserror::Error;

use crate::checkpoint::store::CheckpointError;
use crate::domain::ompf::OmpfContainer;
use crate::domain::types::{Field, Section, TimeSample};
use crate::storage::diff_read::{build_delta, DeltaBuildParams};

/// Chunked-base magic, `b"LEBC"`. Four literal ASCII bytes.
pub const CHUNKED_BASE_MAGIC: [u8; 4] = *b"LEBC";

/// The format version this build writes, and the only one it reads.
pub const CHUNKED_BASE_FORMAT_VERSION: u32 = 1;

/// Fixed header size in bytes.
pub const CHUNKED_BASE_HEADER_SIZE: usize = 12;

/// Byte offset of `format_version` in the header. Public so tests can forge an
/// unknown version by patching one field.
pub const CHUNKED_BASE_FORMAT_VERSION_OFFSET: usize = 4;

/// Byte offset of `chunk_count` in the header. Public so tests can forge a
/// count by patching one field.
pub const CHUNKED_BASE_CHUNK_COUNT_OFFSET: usize = 8;

/// Fixed per-chunk prefix size in bytes (the `u32` length).
pub const CHUNKED_BASE_CHUNK_PREFIX_SIZE: usize = 4;

/// Whether `bytes` carry the chunked-base magic. Anything else — including
/// empty input — is left to the OMPF decoder, exactly as before this format
/// existed.
pub fn is_chunked_base(bytes: &[u8]) -> bool {
    bytes.starts_with(&CHUNKED_BASE_MAGIC)
}

/// Frame `chunks` (each one OODF delta's bytes) as a chunked base.
///
/// Byte-deterministic: the output depends only on the chunk bytes and their
/// order. Refuses an empty chunk list, an empty chunk, and a chunk or count
/// that does not fit its `u32` field — never emits what the decoder would
/// refuse for its framing.
pub fn encode_chunked_base(chunks: &[Vec<u8>]) -> Result<Vec<u8>, ChunkedBaseError> {
    if chunks.is_empty() {
        return Err(ChunkedBaseError::NoChunks);
    }
    let chunk_count =
        u32::try_from(chunks.len()).map_err(|_| ChunkedBaseError::ChunkCountTooLarge {
            chunk_count: u64::try_from(chunks.len()).unwrap_or(u64::MAX),
            max: u32::MAX as usize,
            size: 0,
        })?;
    let payload: usize = chunks
        .iter()
        .map(|c| c.len().saturating_add(CHUNKED_BASE_CHUNK_PREFIX_SIZE))
        .fold(0usize, usize::saturating_add);
    let mut out = Vec::with_capacity(CHUNKED_BASE_HEADER_SIZE.saturating_add(payload));
    out.extend_from_slice(&CHUNKED_BASE_MAGIC);
    out.extend_from_slice(&CHUNKED_BASE_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&chunk_count.to_le_bytes());
    debug_assert_eq!(out.len(), CHUNKED_BASE_HEADER_SIZE);
    for (index, chunk) in chunks.iter().enumerate() {
        if chunk.is_empty() {
            return Err(ChunkedBaseError::EmptyChunk { index });
        }
        // Explicit conversion: a truncating cast would mis-frame every later
        // chunk.
        let len = u32::try_from(chunk.len()).map_err(|_| ChunkedBaseError::ChunkTooLarge {
            index,
            len: chunk.len(),
        })?;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(chunk);
    }
    Ok(out)
}

/// Decode a chunked base's framing, returning each chunk's bytes (views into
/// `bytes`) in order.
///
/// Checks, in this order: the magic, then `format_version` (before any length
/// check), then the header length, then `chunk_count` against the bytes that
/// remain (before allocating), then each chunk's prefix and payload, then that
/// no bytes trail the last chunk. The chunk bytes themselves are not parsed
/// here — see [`decode_chunked_reset_delta`].
pub fn decode_chunked_base(bytes: &[u8]) -> Result<Vec<&[u8]>, ChunkedBaseError> {
    let magic = bytes
        .get(0..CHUNKED_BASE_MAGIC.len())
        .and_then(|s| <[u8; 4]>::try_from(s).ok())
        .ok_or(ChunkedBaseError::Truncated {
            at: "magic",
            needed: CHUNKED_BASE_MAGIC.len(),
            available: bytes.len(),
        })?;
    if magic != CHUNKED_BASE_MAGIC {
        return Err(ChunkedBaseError::BadMagic { got: magic });
    }

    let format_version =
        le_u32(bytes, CHUNKED_BASE_FORMAT_VERSION_OFFSET).ok_or(ChunkedBaseError::Truncated {
            at: "format_version",
            needed: CHUNKED_BASE_FORMAT_VERSION_OFFSET + 4,
            available: bytes.len(),
        })?;
    if format_version != CHUNKED_BASE_FORMAT_VERSION {
        return Err(ChunkedBaseError::UnknownFormatVersion {
            got: format_version,
            supported: CHUNKED_BASE_FORMAT_VERSION,
        });
    }

    let chunk_count =
        le_u32(bytes, CHUNKED_BASE_CHUNK_COUNT_OFFSET).ok_or(ChunkedBaseError::Truncated {
            at: "header",
            needed: CHUNKED_BASE_HEADER_SIZE,
            available: bytes.len(),
        })?;
    if chunk_count == 0 {
        return Err(ChunkedBaseError::NoChunks);
    }

    // Each chunk costs at least its prefix plus one payload byte, so the
    // remaining bytes cap the count before anything is allocated.
    let remaining = bytes.len().saturating_sub(CHUNKED_BASE_HEADER_SIZE);
    let max = remaining / (CHUNKED_BASE_CHUNK_PREFIX_SIZE + 1);
    let too_large = || ChunkedBaseError::ChunkCountTooLarge {
        chunk_count: u64::from(chunk_count),
        max,
        size: bytes.len(),
    };
    let capacity = usize::try_from(chunk_count).map_err(|_| too_large())?;
    if capacity > max {
        return Err(too_large());
    }

    let mut chunks = Vec::with_capacity(capacity);
    let mut offset = CHUNKED_BASE_HEADER_SIZE;
    for index in 0..capacity {
        let available = bytes.len().saturating_sub(offset);
        let len = le_u32(bytes, offset).ok_or(ChunkedBaseError::ChunkTruncated {
            index,
            at: "chunk prefix",
            needed: CHUNKED_BASE_CHUNK_PREFIX_SIZE,
            available,
        })?;
        if len == 0 {
            return Err(ChunkedBaseError::EmptyChunk { index });
        }
        let start = offset + CHUNKED_BASE_CHUNK_PREFIX_SIZE;
        let len = usize::try_from(len).map_err(|_| ChunkedBaseError::ChunkTooLarge {
            index,
            len: usize::MAX,
        })?;
        let chunk = start
            .checked_add(len)
            .and_then(|end| bytes.get(start..end))
            .ok_or(ChunkedBaseError::ChunkTruncated {
                index,
                at: "chunk payload",
                needed: len,
                available: bytes.len().saturating_sub(start),
            })?;
        chunks.push(chunk);
        offset = start + len;
    }

    let extra = bytes.len().saturating_sub(offset);
    if extra > 0 {
        return Err(ChunkedBaseError::TrailingBytes { extra });
    }
    Ok(chunks)
}

/// Little-endian `u32` at `at`, or `None` when `buf` is too short.
fn le_u32(buf: &[u8], at: usize) -> Option<u32> {
    let end = at.checked_add(4)?;
    let raw = <[u8; 4]>::try_from(buf.get(at..end)?).ok()?;
    Some(u32::from_le_bytes(raw))
}

/// Every way a chunked base can be malformed.
///
/// Framing: the header-level variants (`BadMagic`, `UnknownFormatVersion`,
/// `Truncated`, `NoChunks`, `ChunkCountTooLarge`) and `TrailingBytes` (after
/// the last chunk) describe the artifact as a whole; `ChunkTruncated`,
/// `EmptyChunk` and `ChunkTooLarge` name the chunk's zero-based `index`.
///
/// Content, raised by [`decode_chunked_reset_delta`] once the framing has
/// decoded: `ChunkParseFailed`, `ChunkOverEntryLimit`, `NonResetChunk` and
/// `SourceFormatAfterFirstChunk` name the chunk's zero-based `index` and the
/// artifact's chunk `count`.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ChunkedBaseError {
    /// The leading 4 bytes are not [`CHUNKED_BASE_MAGIC`].
    #[error("bad chunked-base magic: expected {expected:?}, got {got:?}", expected = CHUNKED_BASE_MAGIC)]
    BadMagic {
        /// The 4 bytes found at offset 0.
        got: [u8; 4],
    },

    /// Written at a format version this build cannot read. Checked right
    /// after the magic, before any length check.
    #[error("unsupported chunked-base format version {got} (this build supports {supported})")]
    UnknownFormatVersion {
        /// Version read from the artifact.
        got: u32,
        /// [`CHUNKED_BASE_FORMAT_VERSION`].
        supported: u32,
    },

    /// The input ended inside the fixed header (magic, `format_version` or
    /// `chunk_count`).
    #[error("truncated chunked base at {at}: needed {needed} bytes, available {available}")]
    Truncated {
        /// Which header element ran out of bytes.
        at: &'static str,
        /// Bytes required at this point.
        needed: usize,
        /// Bytes actually remaining.
        available: usize,
    },

    /// The input ended inside chunk `index`: its length prefix or its payload.
    #[error(
        "truncated chunked base in chunk {index} at {at}: needed {needed} bytes, available \
         {available}"
    )]
    ChunkTruncated {
        /// Zero-based chunk index.
        index: usize,
        /// Which part of the chunk ran out of bytes (`"chunk prefix"` or
        /// `"chunk payload"`).
        at: &'static str,
        /// Bytes required at this point.
        needed: usize,
        /// Bytes actually remaining.
        available: usize,
    },

    /// `chunk_count` is zero, or (encode side) no chunks were supplied. A base
    /// with no chunk carries no reset and cannot describe any state.
    #[error("chunked base carries no chunks")]
    NoChunks,

    /// `chunk_count` claims more chunks than the input could hold (or, encode
    /// side, more than the `u32` field can count).
    #[error("chunked-base chunk_count {chunk_count} exceeds maximum {max} implied by size {size}")]
    ChunkCountTooLarge {
        /// Declared (or supplied) chunk count.
        chunk_count: u64,
        /// The most chunks the bytes could hold.
        max: usize,
        /// Total artifact size in bytes (0 on the encode side).
        size: usize,
    },

    /// A chunk has length zero.
    #[error("chunked-base chunk {index} is empty")]
    EmptyChunk {
        /// Zero-based chunk index.
        index: usize,
    },

    /// A chunk does not fit its `u32` length field.
    #[error("chunked-base chunk {index} of {len} bytes is too large to encode")]
    ChunkTooLarge {
        /// Zero-based chunk index.
        index: usize,
        /// The chunk's length.
        len: usize,
    },

    /// Bytes remain after the last declared chunk.
    #[error("trailing bytes after the last chunked-base chunk: {extra} extra bytes")]
    TrailingBytes {
        /// Unconsumed bytes.
        extra: usize,
    },

    /// `oodf_ffi::parse_delta` refused chunk `index` for a reason other than
    /// its entry count (that is [`ChunkedBaseError::ChunkOverEntryLimit`]).
    #[error(
        "OODF parse failed for chunk {index} of {count} of a chunked checkpoint base: {reason}"
    )]
    ChunkParseFailed {
        /// Zero-based chunk index.
        index: usize,
        /// Chunks in the artifact.
        count: usize,
        /// The parser's error, rendered.
        reason: String,
    },

    /// Chunk `index` carries more entries than one delta may
    /// ([`MAX_DELTA_ENTRIES`](crate::domain::limits::MAX_DELTA_ENTRIES)), so
    /// the capped `parse_delta` refused it.
    #[error(
        "OODF parse failed for chunk {index} of {count} of a chunked checkpoint base: it carries \
         {entries} entries; one delta may carry at most {max_entries} entries in total"
    )]
    ChunkOverEntryLimit {
        /// Zero-based chunk index.
        index: usize,
        /// Chunks in the artifact.
        count: usize,
        /// The chunk's struct commands + set fields + time samples.
        entries: u64,
        /// The parser's per-delta limit.
        max_entries: u64,
    },

    /// Chunk `index` is not a reset delta (`is_diff = true` and
    /// `base_version = 0`).
    #[error(
        "chunk {index} of {count} of a chunked checkpoint base carries a non-reset delta \
         (is_diff={is_diff}, base_version={base_version}); every chunk must be is_diff=true AND \
         base_version=0"
    )]
    NonResetChunk {
        /// Zero-based chunk index.
        index: usize,
        /// Chunks in the artifact.
        count: usize,
        /// The chunk's `is_diff`.
        is_diff: bool,
        /// The chunk's `base_version`.
        base_version: u64,
    },

    /// Chunk `index`, after the first, carries a `source_format`; only chunk 0
    /// may.
    #[error(
        "chunk {index} of {count} of a chunked checkpoint base carries a source_format; only \
         chunk 0 may"
    )]
    SourceFormatAfterFirstChunk {
        /// Zero-based chunk index (never 0).
        index: usize,
        /// Chunks in the artifact.
        count: usize,
    },
}

/// Every chunked-base failure is `Permanent`: the bytes are durable, and a
/// retry reads the same bytes.
impl From<ChunkedBaseError> for CheckpointError {
    fn from(err: ChunkedBaseError) -> Self {
        CheckpointError::Permanent {
            source: Box::new(err),
        }
    }
}

/// The planned content of a reset base, in the plan's order: what
/// `assembler::build_reset_ompf_from_state` hands [`encode_reset_base`].
pub(crate) struct ResetBaseParts<'a> {
    /// The layer's `source_format`; carried by the first (or only) delta.
    pub(crate) source_format: &'a str,
    /// Sections, in the plan's order; each becomes one struct command.
    pub(crate) sections: &'a [Section],
    /// Fields, in the plan's order, `set_order` already remapped.
    pub(crate) fields: &'a [Field],
    /// Time samples, in the plan's order.
    pub(crate) time_samples: &'a [TimeSample],
    /// Whole-state external-value hash → 1-based index map, shared by every
    /// chunk so each entry encodes exactly as it would unchunked.
    pub(crate) hash_to_index: &'a HashMap<[u8; 32], u32>,
}

/// Encode a reset base: one OMPF-wrapped OODF delta when the entries fit in
/// `max_entries_per_chunk`, a chunked base otherwise (see the module docs).
///
/// Production passes [`MAX_DELTA_ENTRIES`](crate::domain::limits::MAX_DELTA_ENTRIES);
/// the parameter exists so tests can force chunking at a small size. A value
/// of `0` is treated as `1`.
///
/// # Errors
///
/// [`CheckpointError::Permanent`] if `build_delta` fails (OODF FFI allocator
/// failure) or the OMPF / chunked framing refuses its input. None is
/// retryable: the same input fails the same way.
pub(crate) fn encode_reset_base(
    parts: &ResetBaseParts<'_>,
    max_entries_per_chunk: u64,
) -> Result<Vec<u8>, CheckpointError> {
    let per_chunk = usize::try_from(max_entries_per_chunk)
        .unwrap_or(usize::MAX)
        .max(1);
    let sections = parts.sections.len();
    let fields = parts.fields.len();
    let samples = parts.time_samples.len();
    let total = sections.saturating_add(fields).saturating_add(samples);

    if total <= per_chunk {
        // The unchunked base, byte-identical to every base written before the
        // chunked format existed.
        let delta_bytes =
            build_reset_chunk(parts, 0..total, Some(parts.source_format)).map_err(|e| {
                CheckpointError::Permanent {
                    source: format!("OODF delta build failed during checkpoint assembly: {e}")
                        .into(),
                }
            })?;
        // No external-value parts — the base carries hash refs only; the blob
        // bytes travel via the ext-refs sidecar.
        return OmpfContainer::serialize(&delta_bytes, &[]).map_err(|e| {
            CheckpointError::Permanent {
                source: format!("OMPF serialize failed during checkpoint assembly: {e}").into(),
            }
        });
    }

    let mut chunks = Vec::with_capacity(total.div_ceil(per_chunk));
    let mut start = 0usize;
    while start < total {
        let end = start.saturating_add(per_chunk).min(total);
        let index = chunks.len();
        // Only the first chunk carries the source format.
        let source_format = (index == 0).then_some(parts.source_format);
        let chunk = build_reset_chunk(parts, start..end, source_format).map_err(|e| {
            CheckpointError::Permanent {
                source: format!(
                    "OODF delta build failed during checkpoint assembly (chunk {index}): {e}"
                )
                .into(),
            }
        })?;
        chunks.push(chunk);
        start = end;
    }
    Ok(encode_chunked_base(&chunks)?)
}

/// Build the reset delta for entries `range` of the plan's
/// commands → fields → samples sequence.
fn build_reset_chunk(
    parts: &ResetBaseParts<'_>,
    range: Range<usize>,
    source_format: Option<&str>,
) -> Result<Vec<u8>, crate::error::ServiceError> {
    let sections = parts.sections.len();
    let fields = parts.fields.len();
    let section_range = sub_range(&range, 0, sections);
    let field_range = sub_range(&range, sections, fields);
    let sample_range = sub_range(&range, sections + fields, parts.time_samples.len());
    build_delta(&DeltaBuildParams {
        is_diff: true,
        base_version: 0,
        source_format,
        new_sections: &parts.sections[section_range],
        modified_sections: &[],
        deleted_sections: &[],
        fields: &parts.fields[field_range],
        time_samples: &parts.time_samples[sample_range],
        hash_to_index: parts.hash_to_index,
    })
}

/// The part of the global entry `range` that falls in a block of `len`
/// entries starting at global position `offset`, as indices into that block.
fn sub_range(range: &Range<usize>, offset: usize, len: usize) -> Range<usize> {
    let lo = range.start.saturating_sub(offset).min(len);
    let hi = range.end.saturating_sub(offset).min(len);
    lo..hi
}

/// Decode a chunked base into the single reset delta an unchunked base of the
/// same state would have carried (see the module docs).
///
/// Every chunk goes through `oodf_ffi::parse_delta`, unchanged and capped, and
/// must be a reset delta (`is_diff = true`, `base_version = 0`); every chunk
/// after the first must carry an empty `source_format`. Decoding is eager:
/// nothing is returned until every chunk has parsed and passed.
///
/// # Errors
///
/// [`CheckpointError::Permanent`] carrying a [`ChunkedBaseError`] for any
/// framing defect, any chunk that fails to parse or is over the per-delta
/// entry limit, and any chunk of the wrong shape; every chunk-level variant
/// names the chunk's zero-based index.
pub(crate) fn decode_chunked_reset_delta(bytes: &[u8]) -> Result<OwnedDelta, CheckpointError> {
    let chunks = decode_chunked_base(bytes)?;
    let count = chunks.len();
    let mut merged: Option<OwnedDelta> = None;
    for (index, chunk) in chunks.into_iter().enumerate() {
        let delta = oodf_ffi::parse_delta(chunk).map_err(|e| match e {
            OodfError::TooManyTables {
                struct_commands,
                set_fields,
                time_samples,
                max_entries,
            } => ChunkedBaseError::ChunkOverEntryLimit {
                index,
                count,
                entries: struct_commands
                    .saturating_add(set_fields)
                    .saturating_add(time_samples),
                max_entries,
            },
            other => ChunkedBaseError::ChunkParseFailed {
                index,
                count,
                reason: other.to_string(),
            },
        })?;
        if !(delta.is_diff && delta.base_version == 0) {
            return Err(ChunkedBaseError::NonResetChunk {
                index,
                count,
                is_diff: delta.is_diff,
                base_version: delta.base_version,
            }
            .into());
        }
        match merged.as_mut() {
            None => merged = Some(delta),
            Some(acc) => {
                if !delta.source_format.is_empty() {
                    return Err(
                        ChunkedBaseError::SourceFormatAfterFirstChunk { index, count }.into(),
                    );
                }
                let OwnedDelta {
                    struct_commands,
                    set_fields,
                    time_samples,
                    ..
                } = delta;
                acc.struct_commands.extend(struct_commands);
                acc.set_fields.extend(set_fields);
                acc.time_samples.extend(time_samples);
            }
        }
    }
    // `decode_chunked_base` refuses a zero chunk count, so this is total-function
    // plumbing rather than a reachable branch.
    merged.ok_or_else(|| ChunkedBaseError::NoChunks.into())
}

#[cfg(test)]
pub(crate) mod tests {
    //! The chunked base: the encoder's choice of shape, decode fidelity against
    //! the unchunked base and against the original state, byte determinism, and
    //! every refusal of the decoder.
    //!
    //! Chunking is forced at a small size through [`encode_reset_base`]'s
    //! `max_entries_per_chunk` seam, so most tests run on a 14-entry fixture.
    //! The real boundary is pinned through the production entry point
    //! `build_reset_ompf_from_state`, by two tests that build a state of about
    //! `MAX_DELTA_ENTRIES` entries each (about a second each in a debug build):
    //! one at exactly the cap, one an entry over it.
    //!
    //! Plain `#[test]`: no Valkey, no runtime. `encode_reset_base` and
    //! `decode_checkpoint_ompf` are `pub(crate)`, so these live in the crate.
    //! `pub(crate)` so the compose and recovery tests can build a chunked base
    //! of their own fixtures through [`chunked_base_of`].
    //!
    //! Not covered: `ChunkedBaseError::ChunkTooLarge` (a chunk over `u32::MAX`
    //! bytes on encode, unreachable on decode on a 64-bit target) and the encode
    //! side of `ChunkCountTooLarge` (over `u32::MAX` chunks). Both need more
    //! than 4 GiB of input.

    use super::*;
    use crate::checkpoint::apply::decode_checkpoint_ompf;
    use crate::checkpoint::assembler::build_reset_ompf_from_state;
    use crate::domain::diff::{collect_external_value_hashes, DiffEngine, DiffPlan};
    use crate::domain::limits::MAX_DELTA_ENTRIES;
    use crate::domain::types::{FieldValue, ObjectState};
    use ordered_float::OrderedFloat;

    const OBJECT_ID: u64 = 1;
    const OMPF_MAGIC: &[u8; 4] = b"OMPF";
    /// [`MAX_DELTA_ENTRIES`] as a length.
    const CAP: usize = MAX_DELTA_ENTRIES as usize;

    // ── Fixture ─────────────────────────────────────────────────────────────

    fn section(id: u64, parent_id: u64, section_order: u32) -> Section {
        Section {
            section_id: id,
            parent_id,
            name: format!("sec_{id}"),
            section_type: 0,
            section_order,
            first_version: 1,
            last_version: 9,
        }
    }

    fn external(hash_byte: u8) -> FieldValue {
        FieldValue::External {
            hash: [hash_byte; 32],
            size: 1024,
            ext_value_index: 0,
        }
    }

    fn field(
        section_id: u64,
        key: &str,
        version: u64,
        cumulative_set_order: u64,
        authoring_set_order: u64,
        value: FieldValue,
    ) -> Field {
        Field {
            section_id,
            key_name: key.to_string(),
            value,
            set_order: cumulative_set_order,
            authoring_set_order,
            version,
        }
    }

    fn sample(section_id: u64, time: f64, version: u64, value: FieldValue) -> TimeSample {
        TimeSample {
            section_id,
            time: OrderedFloat(time),
            value,
            version,
        }
    }

    /// A 14-entry layer (5 sections, 5 fields, 4 time samples) exercising
    /// every surface a chunk boundary could disturb:
    ///
    /// - a children list out of id order (`1 → [3, 2]`, keyed by
    ///   `section_order`), so a decode that rebuilt lists by id would differ;
    /// - an orphan section (dangling parent 900) in the plan's orphan tail;
    /// - fields whose cumulative `set_order` (the 900s) differs from their
    ///   authoring rank, so the re-derived cursor depends on application order;
    /// - external values on fields and on time samples, with a hash repeated
    ///   across them, so a per-chunk external-value index map (instead of the
    ///   whole-state one) would renumber `ext_value_index`.
    ///
    /// `reverse` flips the insertion order of every `HashMap`, so two builds
    /// differ in layout while being logically identical (the
    /// `assembler.rs` byte-identity fixture's discipline).
    fn fixture_state(reverse: bool) -> ObjectState {
        let mut state = ObjectState {
            object_id: OBJECT_ID,
            sequence_num: 9,
            first_version: 1,
            source_format: "usda".to_string(),
            next_set_order: 908,
            ..Default::default()
        };
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![3, 2]);
        state.children.insert(2, vec![4]);

        let mut sections = vec![
            section(1, 0, 0),
            section(3, 1, 0),
            section(2, 1, 1),
            section(4, 2, 0),
            section(50, 900, 0),
        ];
        let mut fields = vec![
            field(1, "a", 5, 900, 2, FieldValue::Inline(vec![0xA])),
            field(2, "b", 6, 901, 1, external(0xA1)),
            field(3, "c", 6, 902, 1, external(0xA2)),
            field(4, "d", 7, 903, 0, FieldValue::Inline(vec![0xD])),
            field(1, "e", 6, 905, 1, external(0xA1)),
        ];
        let mut samples = vec![
            sample(2, 1.0, 7, external(0xB1)),
            sample(2, 2.0, 7, FieldValue::Inline(vec![0x2])),
            sample(3, 0.5, 8, external(0xA2)),
            sample(4, 3.0, 9, FieldValue::Inline(vec![0x4])),
        ];
        if reverse {
            sections.reverse();
            fields.reverse();
            samples.reverse();
        }
        for s in sections {
            state.sections.insert(s.section_id, s);
        }
        for f in fields {
            state.fields.insert((f.section_id, f.key_name.clone()), f);
        }
        for ts in samples {
            state.time_samples.insert((ts.section_id, ts.time), ts);
        }
        state
    }

    /// Entries in [`fixture_state`].
    const FIXTURE_ENTRIES: u64 = 14;

    /// The planned base `build_reset_ompf_from_state` hands
    /// [`encode_reset_base`], rebuilt here so a test can call the encoder at
    /// any chunk size. The test
    /// `encode_reset_base_at_the_production_limit_matches_the_assembler_byte_for_byte`
    /// anchors it to the production builder, so it cannot drift from what the
    /// assembler encodes.
    struct PlannedBase {
        source_format: String,
        sections: Vec<Section>,
        fields: Vec<Field>,
        time_samples: Vec<TimeSample>,
        hash_to_index: HashMap<[u8; 32], u32>,
    }

    impl PlannedBase {
        fn of(state: &ObjectState) -> Self {
            let plan = DiffEngine::plan(state, &[], 0, 0, None).expect("the fixture must plan");
            let DiffPlan::FullContent {
                sections,
                mut fields,
                time_samples,
            } = plan
            else {
                panic!("have_version 0 always plans FullContent");
            };
            for f in &mut fields {
                f.set_order = f.authoring_set_order;
            }
            let scan = collect_external_value_hashes(&fields, &time_samples);
            Self {
                source_format: state.source_format.clone(),
                sections,
                fields,
                time_samples,
                hash_to_index: scan.hash_to_index,
            }
        }

        fn parts(&self) -> ResetBaseParts<'_> {
            ResetBaseParts {
                source_format: &self.source_format,
                sections: &self.sections,
                fields: &self.fields,
                time_samples: &self.time_samples,
                hash_to_index: &self.hash_to_index,
            }
        }

        fn encode(&self, max_entries_per_chunk: u64) -> Vec<u8> {
            encode_reset_base(&self.parts(), max_entries_per_chunk)
                .expect("the fixture must encode")
        }

        /// The base exactly as the assembler wrote it before the chunked
        /// format existed: the whole plan built as one delta by `build_delta`
        /// and wrapped in an OMPF with no external-value parts. Spelled out
        /// here, independently of [`encode_reset_base`], as the byte oracle
        /// for "at or below the cap, bytes are unchanged".
        fn pre_chunking_ompf(&self) -> Vec<u8> {
            let delta_bytes = build_delta(&DeltaBuildParams {
                is_diff: true,
                base_version: 0,
                source_format: Some(&self.source_format),
                new_sections: &self.sections,
                modified_sections: &[],
                deleted_sections: &[],
                fields: &self.fields,
                time_samples: &self.time_samples,
                hash_to_index: &self.hash_to_index,
            })
            .expect("the fixture delta must build");
            OmpfContainer::serialize(&delta_bytes, &[]).expect("the fixture OMPF must serialize")
        }
    }

    /// `state`'s base as a chunked base of at most `max_entries_per_chunk`
    /// entries per chunk — the bytes the production encoder writes for a
    /// layer over the cap, at a size small enough for a fixture. Panics
    /// unless the result really is chunked, so a caller cannot silently test
    /// a plain OMPF.
    pub(crate) fn chunked_base_of(state: &ObjectState, max_entries_per_chunk: u64) -> Vec<u8> {
        let bytes = PlannedBase::of(state).encode(max_entries_per_chunk);
        assert!(
            is_chunked_base(&bytes),
            "the fixture state must exceed {max_entries_per_chunk} entries, or its base is not \
             chunked"
        );
        let chunks = decode_chunked_base(&bytes).expect("the encoder's framing must decode");
        assert!(
            chunks.len() >= 2,
            "a chunked fixture base carries at least two chunks"
        );
        bytes
    }

    /// The state the real base decoder rebuilds from `bytes`.
    fn decode(bytes: &[u8]) -> ObjectState {
        decode_checkpoint_ompf(OBJECT_ID, bytes).expect("the base must decode")
    }

    // ── Encoder shape ───────────────────────────────────────────────────────

    /// At the production limit the test's planned base encodes to exactly the
    /// bytes `build_reset_ompf_from_state` writes, and both equal the base the
    /// assembler wrote before the chunked format existed: the single OMPF
    /// every base has always been. This anchors [`PlannedBase`] to the
    /// production plan, so the chunked tests below exercise the parts the
    /// assembler really hands the encoder.
    #[test]
    fn encode_reset_base_at_the_production_limit_matches_the_assembler_byte_for_byte() {
        let state = fixture_state(false);
        let (assembled, _) = build_reset_ompf_from_state(&state).expect("must assemble");
        let planned = PlannedBase::of(&state);

        let encoded = planned.encode(MAX_DELTA_ENTRIES);

        assert!(
            assembled.starts_with(OMPF_MAGIC),
            "an in-limit base is an OMPF"
        );
        assert_eq!(encoded, assembled);
        assert_eq!(
            assembled,
            planned.pre_chunking_ompf(),
            "an in-limit base must be byte-identical to the pre-chunking assembler's"
        );
    }

    /// Above the per-chunk limit the encoder writes the chunked form, cut
    /// into ⌈entries / limit⌉ chunks; at or above the entry count it writes
    /// the OMPF. A limit of `0` behaves as `1`.
    #[test]
    fn encode_reset_base_chunks_exactly_when_the_entries_exceed_the_per_chunk_limit() {
        let planned = PlannedBase::of(&fixture_state(false));

        for n in [FIXTURE_ENTRIES, FIXTURE_ENTRIES + 1] {
            let bytes = planned.encode(n);
            assert!(
                bytes.starts_with(OMPF_MAGIC) && !is_chunked_base(&bytes),
                "{FIXTURE_ENTRIES} entries at a limit of {n} must stay one OMPF"
            );
        }
        for n in [1, 4, 5, FIXTURE_ENTRIES - 1] {
            let bytes = planned.encode(n);
            assert!(
                is_chunked_base(&bytes),
                "{FIXTURE_ENTRIES} entries at a limit of {n} must be chunked"
            );
            let chunks = decode_chunked_base(&bytes).expect("the encoder's framing must decode");
            assert_eq!(
                chunks.len() as u64,
                FIXTURE_ENTRIES.div_ceil(n),
                "chunks at limit {n}"
            );
        }
        assert_eq!(
            planned.encode(0),
            planned.encode(1),
            "a per-chunk limit of 0 must behave as 1"
        );
    }

    /// A layer at the real cap, shaped like a real one: a small section tree
    /// with a children list out of id order, `fields` inline fields (one of
    /// them external) spread over the sections, and three time samples (one
    /// external). Every field's cumulative `set_order` and every entry's
    /// version differ from what a reset decode re-derives, so a decode that
    /// carried them instead of re-deriving them would show.
    fn cap_scale_state(fields: usize) -> ObjectState {
        let mut state = ObjectState {
            object_id: OBJECT_ID,
            sequence_num: 40,
            first_version: 3,
            source_format: "usda".to_string(),
            next_set_order: 9_000_000,
            ..Default::default()
        };
        for s in [
            section(1, 0, 0),
            section(3, 1, 0),
            section(2, 1, 1),
            section(4, 2, 0),
        ] {
            state.sections.insert(s.section_id, s);
        }
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![3, 2]);
        state.children.insert(2, vec![4]);
        for i in 0..fields {
            let value = if i == 7 {
                external(0xE7)
            } else {
                FieldValue::Inline((i as u32).to_le_bytes().to_vec())
            };
            let f = field(
                1 + (i % 4) as u64,
                &format!("k{i:06}"),
                5 + (i % 3) as u64,
                1_000_000 + i as u64,
                (i % 3) as u64,
                value,
            );
            state.fields.insert((f.section_id, f.key_name.clone()), f);
        }
        for ts in [
            sample(2, 1.0, 30, FieldValue::Inline(vec![0x21])),
            sample(3, 0.5, 31, external(0xE8)),
            sample(4, 2.5, 32, FieldValue::Inline(vec![0x43])),
        ] {
            state.time_samples.insert((ts.section_id, ts.time), ts);
        }
        state
    }

    /// Entries in [`cap_scale_state`] besides its fields: four sections and
    /// three time samples.
    const CAP_SCALE_NON_FIELD_ENTRIES: usize = 7;

    /// The base an earlier release wrote for a layer one entry over
    /// `MAX_DELTA_ENTRIES`: [`cap_scale_state`] encoded as one plain OMPF delta
    /// (the pre-chunking assembler's bytes), which the capped parser refuses
    /// with `OodfError::TooManyTables`. This release never writes it; restore
    /// can still meet it in a store an earlier release filled. Building it
    /// takes about a second in a debug build.
    pub(crate) fn plain_over_cap_base() -> Vec<u8> {
        let planned = PlannedBase::of(&cap_scale_state(CAP + 1 - CAP_SCALE_NON_FIELD_ENTRIES));
        assert_eq!(
            planned.sections.len() + planned.fields.len() + planned.time_samples.len(),
            CAP + 1,
            "the fixture sits one entry over the cap"
        );
        let bytes = planned.pre_chunking_ompf();
        assert!(
            bytes.starts_with(OMPF_MAGIC) && !is_chunked_base(&bytes),
            "an earlier release's base is a plain OMPF"
        );
        bytes
    }

    /// The real boundary through the production builder, below it: a state of
    /// exactly `MAX_DELTA_ENTRIES` entries is written as the one OMPF the
    /// assembler wrote before the chunked format existed, byte for byte.
    #[test]
    fn build_reset_ompf_from_state_at_exactly_the_cap_writes_the_pre_chunking_ompf() {
        let state = cap_scale_state(CAP - CAP_SCALE_NON_FIELD_ENTRIES);
        let planned = PlannedBase::of(&state);
        assert_eq!(
            planned.sections.len() + planned.fields.len() + planned.time_samples.len(),
            CAP,
            "the fixture sits exactly at the cap"
        );

        let (at_cap, _) = build_reset_ompf_from_state(&state).expect("must assemble");

        assert!(
            at_cap.starts_with(OMPF_MAGIC) && !is_chunked_base(&at_cap),
            "exactly MAX_DELTA_ENTRIES entries must stay one OMPF"
        );
        assert!(
            at_cap == planned.pre_chunking_ompf(),
            "a base at the cap must be byte-identical to the pre-chunking assembler's ({} bytes)",
            at_cap.len()
        );
    }

    /// The real boundary through the production builder, above it: one entry
    /// over `MAX_DELTA_ENTRIES` is a chunked base of two chunks — the first
    /// filled to the cap and carrying the source format, the second carrying
    /// the one remaining entry — and `decode_checkpoint_ompf` rebuilds the
    /// original layer from it.
    ///
    /// "The original layer" is every column a reset base carries, compared
    /// against the state the base was written from: the section tree with its
    /// out-of-id-order children list, every field value (an external one by
    /// hash and size) and authoring rank, every time sample. The columns a
    /// reset decode re-derives are pinned to the re-derived value: every
    /// version stamp is the one reset's `1` (a chunk-by-chunk apply would bump
    /// it, and each chunk being a reset would wipe its predecessors), and the
    /// `set_order` cursor is the sum over the fields of `authoring + 1`. Each
    /// chunk goes through the capped `parse_delta`, so a first chunk of
    /// `MAX_DELTA_ENTRIES + 1` entries would be refused here.
    #[test]
    fn build_reset_ompf_from_state_one_over_the_cap_writes_a_chunked_base_that_decodes_to_the_original(
    ) {
        let original = cap_scale_state(CAP + 1 - CAP_SCALE_NON_FIELD_ENTRIES);

        let (over, ext_hashes) = build_reset_ompf_from_state(&original).expect("must assemble");

        assert!(
            is_chunked_base(&over),
            "one entry over MAX_DELTA_ENTRIES must be chunked"
        );
        assert_eq!(
            ext_hashes,
            vec![[0xE7; 32], [0xE8; 32]],
            "the ext-refs sidecar list is unchanged by chunking"
        );
        let chunks = decode_chunked_base(&over).expect("the framing must decode");
        assert_eq!(chunks.len(), 2);
        let first = oodf_ffi::parse_delta(chunks[0]).expect("chunk 0 must parse");
        let second = oodf_ffi::parse_delta(chunks[1]).expect("chunk 1 must parse");
        assert_eq!(
            first.struct_commands.len() + first.set_fields.len() + first.time_samples.len(),
            CAP,
            "chunk 0 fills the cap"
        );
        assert_eq!(
            second.struct_commands.len() + second.set_fields.len() + second.time_samples.len(),
            1,
            "chunk 1 carries the one remaining entry"
        );
        assert_eq!(
            first.source_format, "usda",
            "chunk 0 carries the source format"
        );
        assert_eq!(second.source_format, "", "later chunks carry none");

        let restored = decode(&over);

        assert_eq!(restored.object_id, OBJECT_ID);
        assert_eq!(restored.source_format, "usda");
        assert_eq!(
            (restored.sequence_num, restored.first_version),
            (0, 0),
            "the decoder leaves both to the caller's checkpoint meta"
        );

        // The section tree.
        assert_eq!(restored.sections.len(), original.sections.len());
        for (id, s) in &original.sections {
            let r = restored
                .sections
                .get(id)
                .unwrap_or_else(|| panic!("section {id} must be restored"));
            assert_eq!(
                (&r.name, r.parent_id, r.section_type, r.section_order),
                (&s.name, s.parent_id, s.section_type, s.section_order),
                "section {id}"
            );
            assert_eq!(
                (r.first_version, r.last_version),
                (1, 1),
                "section {id}: one reset stamps version 1"
            );
        }
        assert_eq!(
            restored.children, original.children,
            "children lists, in order"
        );

        // Every field.
        assert_eq!(restored.fields.len(), original.fields.len());
        assert_eq!(restored.fields.len(), CAP + 1 - CAP_SCALE_NON_FIELD_ENTRIES);
        for (key, f) in &original.fields {
            let r = restored
                .fields
                .get(key)
                .unwrap_or_else(|| panic!("field {key:?} must be restored"));
            assert_same_value(&r.value, &f.value, &format!("field {key:?}"));
            assert_eq!(
                r.authoring_set_order, f.authoring_set_order,
                "field {key:?} authoring rank"
            );
            assert_eq!(r.version, 1, "field {key:?}: one reset stamps version 1");
        }
        let expected_cursor: u64 = original
            .fields
            .values()
            .map(|f| f.authoring_set_order + 1)
            .sum();
        assert_eq!(
            restored.next_set_order, expected_cursor,
            "the set_order cursor re-derived over every field, once"
        );

        // Every time sample.
        assert_eq!(restored.time_samples.len(), original.time_samples.len());
        for (key, ts) in &original.time_samples {
            let r = restored
                .time_samples
                .get(key)
                .unwrap_or_else(|| panic!("time sample {key:?} must be restored"));
            assert_same_value(&r.value, &ts.value, &format!("time sample {key:?}"));
            assert_eq!(
                r.version, 1,
                "time sample {key:?}: one reset stamps version 1"
            );
        }
    }

    /// `FieldValue` equality over what a base carries: inline bytes, or an
    /// external value's hash and size (`ext_value_index` is re-derived over
    /// the whole state).
    fn assert_same_value(restored: &FieldValue, original: &FieldValue, what: &str) {
        match (restored, original) {
            (FieldValue::Inline(a), FieldValue::Inline(b)) => assert_eq!(a, b, "{what}: bytes"),
            (
                FieldValue::External {
                    hash: ra, size: rs, ..
                },
                FieldValue::External {
                    hash: oa, size: os, ..
                },
            ) => assert_eq!((ra, rs), (oa, os), "{what}: external hash and size"),
            (r, o) => panic!("{what}: restored {r:?}, original {o:?}"),
        }
    }

    // ── Decode fidelity and determinism ─────────────────────────────────────

    /// A chunked base decodes to the same `ObjectState` the unchunked base of
    /// the same state decodes to, wherever the chunk boundaries fall — inside
    /// the sections, on the sections/fields seam, inside the fields, on the
    /// fields/samples seam, inside the samples. The chunks are merged into one
    /// delta and applied once, so the version stamps (every section's
    /// `first_version` / `last_version` and every field's `version`), the
    /// re-derived `set_order` cursor and every children list match; applying
    /// chunk by chunk would bump the version per chunk and fail here. (The
    /// state-level `first_version` is not a pin: the decoder zeroes it on every
    /// path.)
    #[test]
    fn chunked_base_decodes_to_the_state_the_unchunked_base_decodes_to() {
        let state = fixture_state(false);
        let (unchunked, _) = build_reset_ompf_from_state(&state).expect("must assemble");
        let expected = decode(&unchunked);

        // Non-vacuity: the baseline really carries the fixture.
        assert_eq!(
            (
                expected.sections.len(),
                expected.fields.len(),
                expected.time_samples.len()
            ),
            (5, 5, 4)
        );
        assert_eq!(
            expected.children.get(&1).map(Vec::as_slice),
            Some([3u64, 2].as_slice()),
            "the baseline keeps the out-of-id-order children list"
        );

        let planned = PlannedBase::of(&state);
        for n in [1, 2, 4, 5, 6, 10, 13] {
            let chunked = planned.encode(n);
            assert!(is_chunked_base(&chunked), "limit {n} must chunk");

            let got = decode(&chunked);

            for (id, s) in &got.sections {
                assert_eq!(
                    (s.first_version, s.last_version),
                    (1, 1),
                    "limit {n}: section {id} must be stamped version 1, as one reset stamps it"
                );
            }
            for (key, f) in &got.fields {
                assert_eq!(f.version, 1, "limit {n}: field {key:?} version stamp");
            }
            assert_eq!(
                got.next_set_order, expected.next_set_order,
                "limit {n}: the re-derived set_order cursor"
            );
            assert_eq!(got.children, expected.children, "limit {n}: children order");
            assert_eq!(got, expected, "limit {n}: the whole decoded state");
        }
    }

    /// Two independently built, logically identical states give byte-identical
    /// chunked artifacts — the property the duplicate-leader re-base relies on
    /// (mirrors the assembler's
    /// `build_reset_ompf_from_state_is_byte_identical_for_equal_state`, which
    /// covers the unchunked form).
    #[test]
    fn chunked_base_is_byte_identical_for_equal_state() {
        let a = PlannedBase::of(&fixture_state(false)).encode(4);
        let b = PlannedBase::of(&fixture_state(true)).encode(4);

        assert!(
            is_chunked_base(&a),
            "the fixture must be chunked for this to cover the chunked form"
        );
        assert_eq!(
            decode_chunked_base(&a).expect("must frame").len() as u64,
            FIXTURE_ENTRIES.div_ceil(4)
        );
        assert_eq!(
            a.len(),
            b.len(),
            "chunked base LENGTH diverged for equal states"
        );
        assert_eq!(a, b, "chunked base bytes diverged for equal states");
    }

    /// The decoder concatenates the three vectors in chunk order into one
    /// reset delta carrying chunk 0's source format.
    #[test]
    fn decode_chunked_reset_delta_merges_chunks_in_order() {
        let artifact = encode_chunked_base(&[
            reset_chunk(true, 0, "usda", 0..2),
            reset_chunk(true, 0, "", 2..3),
            reset_chunk(true, 0, "", 3..5),
        ])
        .expect("must frame");

        let merged = decode_chunked_reset_delta(&artifact).expect("must decode");

        assert!(merged.is_diff && merged.base_version == 0, "a reset delta");
        assert_eq!(merged.source_format, "usda");
        assert_eq!(
            merged
                .set_fields
                .iter()
                .map(|f| f.key_name.as_str())
                .collect::<Vec<_>>(),
            ["k0", "k1", "k2", "k3", "k4"]
        );
    }

    /// b.8jc, the chunked twin of `apply::tests`'
    /// `decode_checkpoint_ompf_base_carrying_negative_zero_time_restores_the_sample_at_positive_zero`:
    /// a chunked base whose first and second chunks each physically carry a
    /// `-0.0` sample time decodes, and the restored state holds both samples
    /// at `+0.0` with their values intact. The decoder parses each chunk
    /// through `parse_delta`, which canonicalizes the time before the merged
    /// delta reaches `validate_delta_content` under `ApplyMode::RestoreBase`;
    /// without it the base is refused with `TimeSampleNegativeZeroTime`.
    #[test]
    fn chunked_base_carrying_negative_zero_times_in_every_chunk_restores_them_at_positive_zero() {
        use crate::checkpoint::apply::tests::{
            assert_sample_restored_at_positive_zero, build_delta_with_negative_zero_time,
            NEGATIVE_ZERO_SENTINEL,
        };
        use oodf_ffi::{CommandCreateSection, StructCommand};

        const FIRST_VALUE: [u8; 1] = [0x07];
        const SECOND_VALUE: [u8; 2] = [0x08, 0x09];
        let negative_zero_sample = |section_id: u64, value: &[u8]| oodf_ffi::TimeSample {
            section_id,
            time: NEGATIVE_ZERO_SENTINEL,
            value_or_ext_hash: value.to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
        };
        let create = |section_id: u64| {
            StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id,
                section_name: format!("sec_{section_id}"),
                section_type: 0,
            })
        };
        let first = build_delta_with_negative_zero_time(
            &OwnedDelta {
                is_diff: true,
                base_version: 0,
                source_format: "usda".to_string(),
                struct_commands: vec![create(7), create(8)],
                time_samples: vec![negative_zero_sample(7, &FIRST_VALUE)],
                ..Default::default()
            },
            NEGATIVE_ZERO_SENTINEL,
        );
        let second = build_delta_with_negative_zero_time(
            &OwnedDelta {
                is_diff: true,
                base_version: 0,
                time_samples: vec![negative_zero_sample(8, &SECOND_VALUE)],
                ..Default::default()
            },
            NEGATIVE_ZERO_SENTINEL,
        );
        let artifact = encode_chunked_base(&[first, second]).expect("must frame");
        assert!(
            is_chunked_base(&artifact),
            "precondition: this is the chunked base path"
        );

        let state = decode_checkpoint_ompf(OBJECT_ID, &artifact)
            .expect("a chunked base whose bytes carry -0.0 times must restore, not be refused");

        assert_eq!(
            state.time_samples.len(),
            2,
            "one sample from each chunk, neither dropped nor merged"
        );
        assert_sample_restored_at_positive_zero(&state, 7, &FIRST_VALUE);
        assert_sample_restored_at_positive_zero(&state, 8, &SECOND_VALUE);
    }

    // ── Framing codec ───────────────────────────────────────────────────────

    /// One OODF delta's bytes: `is_diff` / `base_version` / `source_format` as
    /// given, carrying inline set fields `k{i}` for `i` in `keys`.
    fn reset_chunk(
        is_diff: bool,
        base_version: u64,
        source_format: &str,
        keys: Range<usize>,
    ) -> Vec<u8> {
        oodf_ffi::build_delta(&OwnedDelta {
            is_diff,
            base_version,
            source_format: source_format.to_string(),
            set_fields: keys
                .map(|i| oodf_ffi::SetField {
                    section_id: 1,
                    key_name: format!("k{i}"),
                    value_or_ext_hash: vec![1],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                })
                .collect(),
            ..Default::default()
        })
        .expect("build_delta cannot fail here")
    }

    /// A valid two-chunk artifact and its two chunks' lengths.
    fn two_chunk_artifact() -> (Vec<u8>, usize, usize) {
        let first = reset_chunk(true, 0, "usda", 0..2);
        let second = reset_chunk(true, 0, "", 2..3);
        let (l0, l1) = (first.len(), second.len());
        (
            encode_chunked_base(&[first, second]).expect("must frame"),
            l0,
            l1,
        )
    }

    /// A header: magic, `format_version`, `chunk_count`.
    fn header(format_version: u32, chunk_count: u32) -> Vec<u8> {
        let mut out = CHUNKED_BASE_MAGIC.to_vec();
        out.extend_from_slice(&format_version.to_le_bytes());
        out.extend_from_slice(&chunk_count.to_le_bytes());
        out
    }

    #[test]
    fn decode_chunked_base_returns_the_chunks_encode_framed() {
        let chunks = vec![vec![1u8, 2, 3], vec![4u8], vec![5u8, 6]];

        let framed = encode_chunked_base(&chunks).expect("must frame");

        assert_eq!(
            &framed[..CHUNKED_BASE_HEADER_SIZE],
            header(CHUNKED_BASE_FORMAT_VERSION, 3).as_slice()
        );
        assert_eq!(
            framed.len(),
            CHUNKED_BASE_HEADER_SIZE + 3 * CHUNKED_BASE_CHUNK_PREFIX_SIZE + 6,
            "header, then each chunk's u32 length and bytes, and nothing else"
        );
        let decoded = decode_chunked_base(&framed).expect("must decode");
        assert_eq!(
            decoded,
            chunks.iter().map(Vec::as_slice).collect::<Vec<_>>()
        );
    }

    /// `is_chunked_base` keys on the magic alone: an OMPF, empty input and a
    /// prefix of the magic are left to the OMPF decoder.
    #[test]
    fn is_chunked_base_keys_on_the_full_magic_only() {
        let (artifact, _, _) = two_chunk_artifact();
        let (ompf, _) = build_reset_ompf_from_state(&fixture_state(false)).expect("must assemble");

        assert!(is_chunked_base(&artifact));
        assert!(is_chunked_base(&CHUNKED_BASE_MAGIC));
        assert!(!is_chunked_base(&ompf), "an OMPF base");
        assert!(!is_chunked_base(&[]), "empty input");
        assert!(!is_chunked_base(b"LEB"), "a prefix of the magic");
    }

    #[test]
    fn decode_chunked_base_refuses_a_foreign_magic() {
        let (ompf, _) = build_reset_ompf_from_state(&fixture_state(false)).expect("must assemble");

        assert_eq!(
            decode_chunked_base(&ompf),
            Err(ChunkedBaseError::BadMagic { got: *OMPF_MAGIC })
        );
    }

    /// An unknown `format_version` is reported as such even when nothing
    /// follows it — the version is checked before any length — and also when
    /// a whole artifact carries it.
    #[test]
    fn decode_chunked_base_unknown_format_version_is_refused_before_any_length_check() {
        let mut bare = CHUNKED_BASE_MAGIC.to_vec();
        bare.extend_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            decode_chunked_base(&bare),
            Err(ChunkedBaseError::UnknownFormatVersion {
                got: 2,
                supported: CHUNKED_BASE_FORMAT_VERSION,
            }),
            "8 bytes with an unknown version must not read as a truncated header"
        );

        let (mut artifact, _, _) = two_chunk_artifact();
        artifact[CHUNKED_BASE_FORMAT_VERSION_OFFSET..CHUNKED_BASE_FORMAT_VERSION_OFFSET + 4]
            .copy_from_slice(&7u32.to_le_bytes());
        assert_eq!(
            decode_chunked_base(&artifact),
            Err(ChunkedBaseError::UnknownFormatVersion {
                got: 7,
                supported: CHUNKED_BASE_FORMAT_VERSION,
            })
        );
    }

    #[test]
    fn decode_chunked_base_truncated_header_names_the_missing_element() {
        assert_eq!(
            decode_chunked_base(b"LEB"),
            Err(ChunkedBaseError::Truncated {
                at: "magic",
                needed: 4,
                available: 3,
            })
        );

        let mut short_version = CHUNKED_BASE_MAGIC.to_vec();
        short_version.extend_from_slice(&[1, 0]);
        assert_eq!(
            decode_chunked_base(&short_version),
            Err(ChunkedBaseError::Truncated {
                at: "format_version",
                needed: CHUNKED_BASE_FORMAT_VERSION_OFFSET + 4,
                available: 6,
            })
        );

        let mut short_count = header(CHUNKED_BASE_FORMAT_VERSION, 1);
        short_count.truncate(CHUNKED_BASE_HEADER_SIZE - 2);
        assert_eq!(
            decode_chunked_base(&short_count),
            Err(ChunkedBaseError::Truncated {
                at: "header",
                needed: CHUNKED_BASE_HEADER_SIZE,
                available: CHUNKED_BASE_HEADER_SIZE - 2,
            })
        );
    }

    #[test]
    fn decode_chunked_base_truncated_chunk_prefix_names_the_chunk() {
        let (artifact, l0, _) = two_chunk_artifact();
        let chunk1_prefix = CHUNKED_BASE_HEADER_SIZE + CHUNKED_BASE_CHUNK_PREFIX_SIZE + l0;

        let cut = &artifact[..chunk1_prefix + 2];

        assert_eq!(
            decode_chunked_base(cut),
            Err(ChunkedBaseError::ChunkTruncated {
                index: 1,
                at: "chunk prefix",
                needed: CHUNKED_BASE_CHUNK_PREFIX_SIZE,
                available: 2,
            })
        );
    }

    #[test]
    fn decode_chunked_base_truncated_chunk_payload_names_the_chunk() {
        let (artifact, _, l1) = two_chunk_artifact();

        let cut = &artifact[..artifact.len() - 1];

        assert_eq!(
            decode_chunked_base(cut),
            Err(ChunkedBaseError::ChunkTruncated {
                index: 1,
                at: "chunk payload",
                needed: l1,
                available: l1 - 1,
            })
        );
    }

    #[test]
    fn decode_chunked_base_refuses_trailing_bytes() {
        let (mut artifact, _, _) = two_chunk_artifact();
        artifact.extend_from_slice(&[0, 0, 0]);

        assert_eq!(
            decode_chunked_base(&artifact),
            Err(ChunkedBaseError::TrailingBytes { extra: 3 })
        );
    }

    #[test]
    fn chunked_base_with_no_chunks_is_refused_on_both_sides() {
        assert_eq!(
            decode_chunked_base(&header(CHUNKED_BASE_FORMAT_VERSION, 0)),
            Err(ChunkedBaseError::NoChunks)
        );
        assert_eq!(encode_chunked_base(&[]), Err(ChunkedBaseError::NoChunks));
    }

    #[test]
    fn chunked_base_with_an_empty_chunk_is_refused_on_both_sides_naming_it() {
        let good = reset_chunk(true, 0, "usda", 0..1);
        let mut forged = header(CHUNKED_BASE_FORMAT_VERSION, 2);
        forged.extend_from_slice(&(good.len() as u32).to_le_bytes());
        forged.extend_from_slice(&good);
        forged.extend_from_slice(&0u32.to_le_bytes());

        assert_eq!(
            decode_chunked_base(&forged),
            Err(ChunkedBaseError::EmptyChunk { index: 1 })
        );
        assert_eq!(
            encode_chunked_base(&[good, Vec::new()]),
            Err(ChunkedBaseError::EmptyChunk { index: 1 })
        );
    }

    /// A claimed `chunk_count` the remaining bytes cannot hold is refused
    /// before anything is allocated for it — up to `u32::MAX`, which would
    /// otherwise reserve tens of GiB.
    #[test]
    fn decode_chunked_base_refuses_a_chunk_count_the_bytes_cannot_hold() {
        let chunk = reset_chunk(true, 0, "usda", 0..1);
        for claimed in [1_000u32, u32::MAX] {
            let mut forged = header(CHUNKED_BASE_FORMAT_VERSION, claimed);
            forged.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
            forged.extend_from_slice(&chunk);
            let remaining = forged.len() - CHUNKED_BASE_HEADER_SIZE;
            let max = remaining / (CHUNKED_BASE_CHUNK_PREFIX_SIZE + 1);
            assert!(
                (claimed as usize) > max,
                "precondition: {remaining} bytes cannot hold {claimed} chunks"
            );

            assert_eq!(
                decode_chunked_base(&forged),
                Err(ChunkedBaseError::ChunkCountTooLarge {
                    chunk_count: u64::from(claimed),
                    max,
                    size: forged.len(),
                }),
                "claimed {claimed}"
            );
        }
    }

    // ── Decoder refusals, as the restore path sees them ─────────────────────

    /// The typed `ChunkedBaseError` inside the `Permanent` a decode returned,
    /// or a panic naming what came instead.
    fn chunked_error<T: std::fmt::Debug>(result: Result<T, CheckpointError>) -> ChunkedBaseError {
        match result {
            Err(CheckpointError::Permanent { source }) => match source
                .downcast::<ChunkedBaseError>()
            {
                Ok(err) => *err,
                Err(other) => panic!("expected a typed ChunkedBaseError, got Permanent({other})"),
            },
            Err(other) => panic!("expected Permanent, got {other:?}"),
            Ok(decoded) => panic!("expected Permanent, but decoded {decoded:?}"),
        }
    }

    /// Every framing refusal reaches the restore path as `Permanent`, carrying
    /// the typed `ChunkedBaseError` — through the chunked decoder and through
    /// `decode_checkpoint_ompf`'s magic dispatch alike.
    #[test]
    fn framing_refusals_are_permanent_and_typed_through_both_decoders() {
        let (mut artifact, _, _) = two_chunk_artifact();
        artifact.push(0);
        let expected = ChunkedBaseError::TrailingBytes { extra: 1 };

        assert_eq!(
            chunked_error(decode_chunked_reset_delta(&artifact)),
            expected
        );
        assert_eq!(
            chunked_error(decode_checkpoint_ompf(OBJECT_ID, &artifact)),
            expected,
            "the LEBC magic must route to the chunked decoder, not the OMPF parser"
        );
    }

    #[test]
    fn decode_chunked_reset_delta_refuses_a_non_reset_chunk_naming_it() {
        let not_a_diff = encode_chunked_base(&[
            reset_chunk(true, 0, "usda", 0..1),
            reset_chunk(false, 0, "", 1..2),
        ])
        .expect("must frame");
        assert_eq!(
            chunked_error(decode_chunked_reset_delta(&not_a_diff)),
            ChunkedBaseError::NonResetChunk {
                index: 1,
                count: 2,
                is_diff: false,
                base_version: 0,
            }
        );

        let based = encode_chunked_base(&[
            reset_chunk(true, 5, "usda", 0..1),
            reset_chunk(true, 0, "", 1..2),
        ])
        .expect("must frame");
        assert_eq!(
            chunked_error(decode_chunked_reset_delta(&based)),
            ChunkedBaseError::NonResetChunk {
                index: 0,
                count: 2,
                is_diff: true,
                base_version: 5,
            }
        );
    }

    #[test]
    fn decode_chunked_reset_delta_refuses_a_source_format_after_chunk_zero_naming_the_chunk() {
        let artifact = encode_chunked_base(&[
            reset_chunk(true, 0, "usda", 0..1),
            reset_chunk(true, 0, "", 1..2),
            reset_chunk(true, 0, "usda", 2..3),
        ])
        .expect("must frame");

        assert_eq!(
            chunked_error(decode_chunked_reset_delta(&artifact)),
            ChunkedBaseError::SourceFormatAfterFirstChunk { index: 2, count: 3 }
        );
    }

    #[test]
    fn decode_chunked_reset_delta_refuses_a_chunk_that_fails_to_parse_naming_it() {
        let artifact = encode_chunked_base(&[
            reset_chunk(true, 0, "usda", 0..1),
            b"definitely not an OODF delta".to_vec(),
        ])
        .expect("must frame");

        let err = chunked_error(decode_chunked_reset_delta(&artifact));

        assert!(
            matches!(
                &err,
                ChunkedBaseError::ChunkParseFailed { index: 1, count: 2, reason }
                    if !reason.is_empty()
            ),
            "{err:?}"
        );
    }

    /// Every chunk is still held to the per-delta entry limit by the capped
    /// `parse_delta`: a chunk over it is refused, naming the chunk and its
    /// entry count, rather than decoded by some uncapped path.
    #[test]
    fn decode_chunked_reset_delta_refuses_a_chunk_over_the_entry_limit_naming_it() {
        let artifact = encode_chunked_base(&[
            reset_chunk(true, 0, "usda", 0..1),
            reset_chunk(true, 0, "", 1..CAP + 2),
        ])
        .expect("must frame");

        let err = chunked_error(decode_chunked_reset_delta(&artifact));

        assert_eq!(
            err,
            ChunkedBaseError::ChunkOverEntryLimit {
                index: 1,
                count: 2,
                entries: MAX_DELTA_ENTRIES + 1,
                max_entries: MAX_DELTA_ENTRIES,
            }
        );
        assert!(
            err.to_string()
                .contains(&format!("at most {MAX_DELTA_ENTRIES} entries")),
            "the message names the limit: {err}"
        );
    }
}
