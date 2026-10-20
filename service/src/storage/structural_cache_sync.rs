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

//! The structural cache's repair read: `scripts/structural_cache_sync.lua`,
//! its `KEYS` / `ARGV` encoder and its reply decoder (Spec SDD SR-4.1).
//!
//! One read-only `EVALSHA` on the data pool reports what a replica's cached
//! decoded state of one object needs to come current:
//!
//! - the header facts the caller maps (presence, a decimal `seq`,
//!   `commit_chain`, `superseded`, `chs_complete`);
//! - the call's commit-token record (`le:{oid}:obj:wtok`), as data;
//! - outside probe mode, a bounded chunk of the layer-stream entries after the
//!   cached `seq`, each with its `payload` text and its `ompf` raw bytes;
//! - in probe mode, the object's five index cardinalities
//!   ([`SyncIndexCardinalities`]): the ones `object_snapshot_read.lua`'s
//!   step 2 reads, so the cold start can compute that script's item bound and
//!   choose its route before any full read
//!   (`storage::structural_cache_cold_start`). They cost the probe no round
//!   trip: five O(1) `TYPE` and `SCARD` / `ZCARD` pairs inside the same
//!   script, on five more declared keys that only a probe-mode call carries.
//!   Like the commit-token record they are data: an index key of a foreign
//!   type reports no cardinalities (`SyncChunk::Probe(None)`) instead of
//!   failing the probe, so its header facts and token record still answer.
//!
//! **Format v2.** The reply's format tag is `structural-cache-sync-v2`; v2
//! added the probe chunk's cardinalities. A reply in any other format is
//! refused as `Internal`.
//!
//! Nothing here interprets the answer. Mapping the header facts to errors,
//! matching the commit token, checking the entries and replaying them belong
//! to the caller; this module only builds the call and decodes the reply into
//! owned types ([`StructuralSyncReply`]).
//!
//! **Bytes.** A stream entry's `ompf` field is the writer's binary OMPF, so it
//! is decoded as raw bytes and never through a UTF-8 `String`, which would fail
//! the whole reply on the first OMPF that is not valid UTF-8 (the
//! `tail_slice` / `stream_mux` rule). Only `payload` is decoded as text.

use bytes::Bytes;
use fred::types::Value;

use crate::error::ServiceError;
use crate::storage::client::{
    map_transient_fred_err, FredErrOrigin, ValkeyClient, POOL_LABEL_DATA,
};
use crate::storage::diff_read::SnapshotCardinalities;
use crate::storage::keys::KeySchema;
use crate::storage::layer::CHS_COMPLETE_MARKED;
use crate::storage::lua::STRUCTURAL_CACHE_SYNC_SCRIPT;
use crate::storage::notifications::{NOTIFICATION_PAYLOAD_FIELD, RELAY_OMPF_FIELD};

/// The most stream entries one repair-read call returns (Spec SDD SR-4.1).
pub const STRUCTURAL_REPAIR_MAX_ENTRIES: u64 = 256;

/// The most stream-entry bytes one repair-read call returns (Spec SDD SR-4.1):
/// an entry's id plus every field name and value, its full OMPF included.
/// The hard cap on a repair read's byte bound: a repair's reads start at
/// [`STRUCTURAL_REPAIR_INITIAL_BYTES`] and grow toward it only for a larger
/// pending entry or a chunk cut short by its byte bound
/// (`storage::structural_cache_repair`, "The loop").
///
/// A first pending entry alone above a read's bound answers
/// [`SyncChunk::TooLarge`]; above this cap the repair reloads `too_large`.
///
/// 4 MiB holds the large-external-value fixture's 2 MiB value (an entry of
/// 2,097,428 bytes) in one chunk, and one call that returns a full 4 MiB
/// blocks the shard for about 4.6 ms on a private Valkey 9.0.3
/// (`docs-internal/validation/structural-edits-b2wo-20261002/epic4-measurements.md`).
pub const STRUCTURAL_REPAIR_MAX_BYTES: u64 = 4 * 1024 * 1024;

/// The byte bound of a stream repair's first read: 256 KiB [spec choice].
/// A repair holds its read's reply in the structural call's commit charge
/// before it issues the read (RP0, SDD §5.4), so its reads ask for what is
/// usually pending rather than the 4 MiB cap. A read's bound grows, up to
/// [`STRUCTURAL_REPAIR_MAX_BYTES`], and the repair keeps the grown size for
/// its later reads: a pending entry larger than the bound grows the next read
/// to that entry's size, and a chunk that stops at the bound with entries
/// still pending doubles it, so a backlog of large records (screenshot-sized,
/// 0.15 to 0.6 MB) is read in batches rather than one round trip each.
pub const STRUCTURAL_REPAIR_INITIAL_BYTES: u64 = 256 * 1024;

/// The entry bound of a stream repair's first read: 91, at most
/// [`STRUCTURAL_REPAIR_MAX_ENTRIES`]; the later reads of the repair ask up to
/// that full bound. RP0 prices a read's reply by its entry bound and its byte
/// bound (`grpc::upload_admission::REPAIR_REPLY_ENTRY_BYTES`,
/// `REPAIR_REPLY_BYTE_CENTI`), and the first read's RP0 is the charge every
/// cache hit holds before it learns what is pending, so its two bounds are set
/// by a fixed rule against 1 MiB: the full 256 entries at
/// [`STRUCTURAL_REPAIR_INITIAL_BYTES`] if that fits, else the largest entry
/// bound in [64, 256] that does, else 128 KiB with the entry bound chosen the
/// same way. Derived, never fitted: at the wired reply constants (864 B per
/// entry, 3.70 B per byte, which cover every member of the T7 reply fit;
/// `docs-internal/validation/repair-charge-t7-b-fsz-20261010/fits.json`)
/// 256 entries price 1,191,117 B and 91 entries 1,048,557 B, the largest
/// within 1,048,576 B — about twice the ~0.5 MiB per call the repair's
/// charge was planned at, which the rule's 1 MiB limit caps. It moves
/// whenever either reply constant is refitted: the unit test
/// `first_read_bound_is_the_rules_step` recomputes the step from the wired
/// constants and fails until this follows.
pub const STRUCTURAL_REPAIR_INITIAL_ENTRIES: u64 = 91;

/// Entries per `XRANGE` page inside the script.
///
/// The script reads the chunk a page at a time and stops at the entry that
/// would cross the call's byte bound (at most
/// [`STRUCTURAL_REPAIR_MAX_BYTES`]), so one call copies at most the byte bound
/// plus one page into Lua. At one entry per page that is the bound plus one
/// entry. A larger page issues fewer `XRANGE` calls but can copy up to one
/// page minus one entry more than that.
///
/// Kept at 1 after measurement (`epic4-measurements.md`, as above). Larger
/// pages save at most about 0.8 ms on a call of 256 small entries (1.25 ms
/// median at one per page, about 0.5 ms from eight up) and nothing on a
/// byte-bound chunk. But each extra entry a page may copy can be as large as
/// one upload (`LIVE_EDIT_MAX_LAYER_SIZE`, 100 MB by default), so the
/// worst-case shard block grows with the page size.
pub const STRUCTURAL_REPAIR_PAGE_ENTRIES: u64 = 1;

/// The reply format tag the script puts first in every reply. A reply carrying
/// another tag came from a script this binary does not carry and is refused as
/// `Internal` rather than decoded.
const STRUCTURAL_CACHE_SYNC_FORMAT: &str = "structural-cache-sync-v2";

/// What a repair-read call reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncMode {
    /// Header facts, the commit-token record and the index cardinalities; the
    /// stream is not read.
    Probe,
    /// Header facts, the commit-token record, and the stream entries
    /// `{from_seq+1}-0 ..= {end}-0`, where `end` is the header `seq`, or
    /// `target_seq` when that is lower.
    Repair {
        /// The `seq` the caller's state is at.
        from_seq: u64,
        /// Caps the range below the header `seq` (a repair to a given cut).
        target_seq: Option<u64>,
    },
}

/// One repair-read call: which object, for which `client_id`, in which mode,
/// under which bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralSyncRequest<'a> {
    /// The object read.
    pub object_id: u64,
    /// The call's `client_id`: the commit-token record's field.
    pub client_id: &'a str,
    /// What to read.
    pub mode: SyncMode,
    /// At most this many stream entries per call.
    pub max_entries: u64,
    /// At most this many stream-entry bytes per call.
    pub max_bytes: u64,
    /// Entries per `XRANGE` page inside the script.
    pub page_entries: u64,
}

impl<'a> StructuralSyncRequest<'a> {
    /// A probe-mode call: header facts, the commit-token record and the index
    /// cardinalities.
    pub fn probe(object_id: u64, client_id: &'a str) -> Self {
        Self::with_mode(object_id, client_id, SyncMode::Probe)
    }

    /// A repair call from `from_seq` up to the header `seq`, under the
    /// production bounds.
    pub fn repair(object_id: u64, client_id: &'a str, from_seq: u64) -> Self {
        Self::with_mode(
            object_id,
            client_id,
            SyncMode::Repair {
                from_seq,
                target_seq: None,
            },
        )
    }

    /// A repair call from `from_seq` up to `target_seq` (or the header `seq`
    /// when that is lower), under the production bounds.
    pub fn repair_to(object_id: u64, client_id: &'a str, from_seq: u64, target_seq: u64) -> Self {
        Self::with_mode(
            object_id,
            client_id,
            SyncMode::Repair {
                from_seq,
                target_seq: Some(target_seq),
            },
        )
    }

    fn with_mode(object_id: u64, client_id: &'a str, mode: SyncMode) -> Self {
        Self {
            object_id,
            client_id,
            mode,
            max_entries: STRUCTURAL_REPAIR_MAX_ENTRIES,
            max_bytes: STRUCTURAL_REPAIR_MAX_BYTES,
            page_entries: STRUCTURAL_REPAIR_PAGE_ENTRIES,
        }
    }

    /// The script's `KEYS`: the header, the layer stream and the commit-token
    /// record; in probe mode also the sections, tombstones, fields,
    /// time-samples and children-list indexes, whose cardinalities only a
    /// probe reads. All under the object's `{oid}` hash tag.
    pub fn keys(&self, keys: &KeySchema) -> Vec<String> {
        let oid = self.object_id.to_string();
        let mut declared = vec![
            keys.obj_header(&oid),
            keys.layer_stream(self.object_id),
            keys.obj_write_token(&oid),
        ];
        if self.mode == SyncMode::Probe {
            declared.extend([
                keys.obj_sections_index(&oid),
                keys.obj_deleted_sections_index(&oid),
                keys.obj_fields_index(&oid),
                keys.obj_time_samples_index(&oid),
                keys.obj_children_index(&oid),
            ]);
        }
        declared
    }

    /// The script's `ARGV` (see `scripts/structural_cache_sync.lua`). Every
    /// position is always present; an empty string is "not given".
    pub fn argv(&self) -> Vec<String> {
        let (from_seq, target_seq) = match self.mode {
            SyncMode::Probe => (String::new(), String::new()),
            SyncMode::Repair {
                from_seq,
                target_seq,
            } => (
                from_seq.to_string(),
                target_seq.map(|t| t.to_string()).unwrap_or_default(),
            ),
        };
        vec![
            from_seq,
            self.max_entries.to_string(),
            self.max_bytes.to_string(),
            self.client_id.to_owned(),
            target_seq,
            self.page_entries.to_string(),
        ]
    }
}

/// The decoded reply of one repair-read call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuralSyncReply {
    /// The header facts, read in the same script as everything below.
    pub header: SyncHeader,
    /// The call's commit-token record.
    pub commit_token: SyncCommitToken,
    /// The stream chunk, or why there is none.
    pub chunk: SyncChunk,
}

/// The object header as the script found it. Reported, not mapped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncHeader {
    /// The header key holds no field.
    Absent,
    /// The header key holds at least one field.
    Present(SyncHeaderFacts),
}

/// The header columns the caller maps and the repair checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncHeaderFacts {
    /// The `seq` column, when it is a decimal of at most 2^53 - 1 (the
    /// ceiling both commit paths enforce). `None` when it is absent or
    /// anything else.
    pub seq: Option<u64>,
    /// The `commit_chain` column; `""` when absent (an object materialized
    /// before the column existed).
    pub commit_chain: String,
    /// The `superseded` column, verbatim; `None` when absent.
    pub superseded: Option<String>,
    /// The `chs_complete` column, verbatim; `None` when absent.
    pub chs_complete: Option<String>,
}

impl SyncHeaderFacts {
    /// Whether the header marks the object's children-list index complete
    /// (`chs_complete` = [`CHS_COMPLETE_MARKED`]).
    pub fn chs_complete_marked(&self) -> bool {
        self.chs_complete.as_deref() == Some(CHS_COMPLETE_MARKED)
    }
}

/// The call's commit-token record, as the script read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncCommitToken {
    /// `TYPE le:{oid}:obj:wtok`: `"none"`, `"hash"`, or a foreign type.
    pub key_type: String,
    /// `HGET le:{oid}:obj:wtok <client_id>`: `None` when the field is absent,
    /// and always `None` for a foreign type, where the script does not read it.
    pub record: Option<String>,
}

impl SyncCommitToken {
    /// Whether the record key holds a type other than a hash (or nothing).
    pub fn is_wrong_type(&self) -> bool {
        self.key_type != "hash" && self.key_type != "none"
    }
}

/// The index cardinalities a probe-mode call reads: the five
/// `object_snapshot_read.lua`'s step 2 reads, so
/// [`Self::snapshot_cardinalities`] gives that script's item bound
/// ([`crate::storage::diff_read::snapshot_item_bound`]) without a full read.
/// Each is 0 for an absent key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SyncIndexCardinalities {
    /// `SCARD obj:secs`.
    pub sections: u64,
    /// `SCARD obj:dels`.
    pub tombstones: u64,
    /// `ZCARD obj:flds`.
    pub fields: u64,
    /// `ZCARD obj:tss`.
    pub time_samples: u64,
    /// `SCARD obj:chs`, read whatever the header says; it counts toward the
    /// item bound only on a marked object.
    pub children_index: u64,
}

impl SyncIndexCardinalities {
    /// The cardinalities as the atomic read's step 2 reports them: the
    /// children-list index counted only when `chs_complete_marked` (the
    /// probe's header facts, [`SyncHeaderFacts::chs_complete_marked`]), as
    /// that script reads it only then.
    pub fn snapshot_cardinalities(self, chs_complete_marked: bool) -> SnapshotCardinalities {
        SnapshotCardinalities {
            sections: self.sections,
            tombstones: self.tombstones,
            fields: self.fields,
            time_samples: self.time_samples,
            children_index: chs_complete_marked.then_some(self.children_index),
        }
    }
}

/// The stream part of a reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SyncChunk {
    /// A probe-mode call: the stream was not read; the index cardinalities
    /// were. `None` when any of the five index keys holds a foreign type, so
    /// its count could not be read; the probe's other answers are unaffected.
    Probe(Option<SyncIndexCardinalities>),
    /// A repair call that had no range to read: the header is absent or its
    /// `seq` is not a decimal ([`SyncHeaderFacts::seq`] is `None`).
    NoRange,
    /// Entries `{from_seq+1}-0 ..= {end_seq}-0`, as many as fit the bounds.
    Entries {
        /// The range's last `seq`: the header `seq`, or the target `seq` when
        /// that is lower.
        end_seq: u64,
        /// In stream-id order.
        entries: Vec<SyncStreamEntry>,
        /// Whether the chunk covers the range through `end_seq`: its last
        /// entry is `{end_seq}-0`, or the stream had no more entries in the
        /// range. `false` when the entry or byte bound stopped it first; the
        /// caller continues from the last entry.
        reached_end: bool,
        /// The chunk's byte tally (ids plus every field name and value).
        bytes: u64,
    },
    /// The first pending entry alone is larger than the byte bound; no entry
    /// is returned.
    TooLarge {
        /// As in [`SyncChunk::Entries`].
        end_seq: u64,
        /// The oversize entry's stream id.
        entry_id: String,
        /// Its size, counted as the byte bound counts it.
        entry_bytes: u64,
    },
}

/// One layer-stream entry of a chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncStreamEntry {
    /// The stream id, verbatim (`{seq}-0` on every entry a writer appends).
    pub id: String,
    /// The `payload` field as text. `None` when the field is absent or not
    /// valid UTF-8: either way the entry is unparseable to the caller.
    pub payload: Option<String>,
    /// The `ompf` field's raw bytes; `None` when the field is absent.
    pub ompf: Option<Bytes>,
}

/// Run one repair-read call on the data pool and decode its reply.
///
/// A failure of the `EVALSHA` (transport, `NOSCRIPT` reload, `-BUSY`, a tagged
/// in-script refusal such as a wrong-typed header) is classified by
/// [`map_transient_fred_err`]: transient → `Aborted`, the in-band retry signal
/// of the caller's retry loop; permanent → `Internal`. A reply that does not
/// have the script's shape is `Internal`.
pub async fn read_structural_cache_sync(
    client: &ValkeyClient,
    keys: &KeySchema,
    request: &StructuralSyncRequest<'_>,
) -> Result<StructuralSyncReply, ServiceError> {
    let reply = STRUCTURAL_CACHE_SYNC_SCRIPT
        .evalsha_with_reload::<Value, _, _>(client.pool.next(), request.keys(keys), request.argv())
        .await
        .map_err(|e| {
            map_transient_fred_err(
                FredErrOrigin::client(client, POOL_LABEL_DATA),
                "EVALSHA structural_cache_sync",
                e,
            )
        })?;
    decode_structural_cache_sync_reply(request.object_id, reply)
}

/// Decode the script's reply (shapes in `scripts/structural_cache_sync.lua`).
///
/// Strict: every element must have exactly the script's type and arity, so a
/// short or reshaped reply is `Internal` rather than decoded as empty facts.
pub fn decode_structural_cache_sync_reply(
    object_id: u64,
    reply: Value,
) -> Result<StructuralSyncReply, ServiceError> {
    let d = Decoder { object_id };
    let [format, header, token, chunk] = d.fixed::<4>(reply, "reply")?;
    let format = d.text(format, "format tag")?;
    if format != STRUCTURAL_CACHE_SYNC_FORMAT {
        return Err(ServiceError::Internal(format!(
            "structural_cache_sync.lua answered for object {object_id} in format {format:?}, \
             expected {STRUCTURAL_CACHE_SYNC_FORMAT:?}: the script that ran is not the one this \
             binary carries"
        )));
    }
    Ok(StructuralSyncReply {
        header: d.header(header)?,
        commit_token: d.commit_token(token)?,
        chunk: d.chunk(chunk)?,
    })
}

/// The reply decoder's element readers, carrying the object id for messages.
struct Decoder {
    object_id: u64,
}

impl Decoder {
    /// The `Internal` error for a reply that does not have the script's shape.
    fn malformed(&self, what: impl std::fmt::Display) -> ServiceError {
        ServiceError::Internal(format!(
            "structural_cache_sync.lua reply for object {} is malformed: {what}",
            self.object_id
        ))
    }

    /// An array element.
    fn array(&self, value: Value, what: &str) -> Result<Vec<Value>, ServiceError> {
        match value {
            Value::Array(items) => Ok(items),
            other => Err(self.malformed(format!("{what} is not an array: {other:?}"))),
        }
    }

    /// An array element of exactly `N` elements.
    fn fixed<const N: usize>(&self, value: Value, what: &str) -> Result<[Value; N], ServiceError> {
        let items = self.array(value, what)?;
        let len = items.len();
        items
            .try_into()
            .map_err(|_| self.malformed(format!("{what} has {len} elements, expected {N}")))
    }

    /// A bulk-string element as raw bytes. Never through UTF-8.
    fn bytes(&self, value: Value, what: &str) -> Result<Bytes, ServiceError> {
        match value {
            Value::Bytes(b) => Ok(b),
            Value::String(s) => Ok(s.into_inner()),
            other => Err(self.malformed(format!("{what} is not a string: {other:?}"))),
        }
    }

    /// A bulk-string element that must be UTF-8 text.
    fn text(&self, value: Value, what: &str) -> Result<String, ServiceError> {
        let raw = self.bytes(value, what)?;
        String::from_utf8(raw.to_vec())
            .map_err(|_| self.malformed(format!("{what} is not valid UTF-8")))
    }

    /// A non-negative integer element.
    fn integer(&self, value: Value, what: &str) -> Result<u64, ServiceError> {
        match value {
            Value::Integer(i) => {
                u64::try_from(i).map_err(|_| self.malformed(format!("{what} is negative: {i}")))
            }
            other => Err(self.malformed(format!("{what} is not an integer: {other:?}"))),
        }
    }

    /// A `0` / `1` flag element.
    fn flag(&self, value: Value, what: &str) -> Result<bool, ServiceError> {
        match self.integer(value, what)? {
            0 => Ok(false),
            1 => Ok(true),
            n => Err(self.malformed(format!("{what} is {n}, expected 0 or 1"))),
        }
    }

    /// A decimal `seq` carried as a string.
    fn seq(&self, value: Value, what: &str) -> Result<u64, ServiceError> {
        let s = self.text(value, what)?;
        s.parse::<u64>()
            .map_err(|_| self.malformed(format!("{what} {s:?} is not a decimal u64")))
    }

    /// An optional text element: `{}` → `None`, `{text}` → `Some`.
    fn optional_text(&self, value: Value, what: &str) -> Result<Option<String>, ServiceError> {
        let mut items = self.array(value, what)?;
        match items.len() {
            0 => Ok(None),
            1 => self.text(items.remove(0), what).map(Some),
            n => Err(self.malformed(format!("{what} has {n} elements, expected 0 or 1"))),
        }
    }

    /// An optional integer element: `{}` → `None`, `{n}` → `Some(n)`.
    fn optional_integer(&self, value: Value, what: &str) -> Result<Option<u64>, ServiceError> {
        let mut items = self.array(value, what)?;
        match items.len() {
            0 => Ok(None),
            1 => self.integer(items.remove(0), what).map(Some),
            n => Err(self.malformed(format!("{what} has {n} elements, expected 0 or 1"))),
        }
    }

    /// `{exists, seq, commit_chain, superseded, chs_complete}`.
    fn header(&self, value: Value) -> Result<SyncHeader, ServiceError> {
        let [exists, seq, commit_chain, superseded, chs_complete] =
            self.fixed::<5>(value, "header")?;
        let exists = self.flag(exists, "header presence")?;
        let seq = self
            .optional_text(seq, "header seq")?
            .map(|s| {
                s.parse::<u64>()
                    .map_err(|_| self.malformed(format!("header seq {s:?} is not a decimal u64")))
            })
            .transpose()?;
        let facts = SyncHeaderFacts {
            seq,
            commit_chain: self.text(commit_chain, "header commit_chain")?,
            superseded: self.optional_text(superseded, "header superseded")?,
            chs_complete: self.optional_text(chs_complete, "header chs_complete")?,
        };
        if exists {
            return Ok(SyncHeader::Present(facts));
        }
        let empty = SyncHeaderFacts {
            seq: None,
            commit_chain: String::new(),
            superseded: None,
            chs_complete: None,
        };
        if facts != empty {
            return Err(self.malformed(format!("an absent header carries columns: {facts:?}")));
        }
        Ok(SyncHeader::Absent)
    }

    /// `{type, record}`.
    fn commit_token(&self, value: Value) -> Result<SyncCommitToken, ServiceError> {
        let [key_type, record] = self.fixed::<2>(value, "commit-token record")?;
        let token = SyncCommitToken {
            key_type: self.text(key_type, "commit-token record type")?,
            record: self.optional_text(record, "commit-token record value")?,
        };
        if token.is_wrong_type() && token.record.is_some() {
            return Err(self.malformed(format!(
                "a commit-token record of type {:?} carries a value",
                token.key_type
            )));
        }
        Ok(token)
    }

    /// `{'PROBE', cards}`, `{'NO_RANGE'}`,
    /// `{'OK', end, entries, reached, bytes}` or
    /// `{'TOO_LARGE', end, id, entry_bytes}`.
    fn chunk(&self, value: Value) -> Result<SyncChunk, ServiceError> {
        let mut parts = self.array(value, "chunk")?.into_iter();
        let arity = parts.len();
        let status = match parts.next() {
            Some(status) => self.text(status, "chunk status")?,
            None => return Err(self.malformed("chunk is empty")),
        };
        let expected = match status.as_str() {
            "NO_RANGE" => 1,
            "PROBE" => 2,
            "OK" => 5,
            "TOO_LARGE" => 4,
            other => return Err(self.malformed(format!("unknown chunk status {other:?}"))),
        };
        if arity != expected {
            return Err(self.malformed(format!(
                "a {status} chunk has {arity} elements, expected {expected}"
            )));
        }
        let mut next = || parts.next().expect("arity checked");
        Ok(match status.as_str() {
            "PROBE" => {
                let [sections, tombstones, fields, time_samples, children_index] =
                    self.fixed::<5>(next(), "probe cardinalities")?;
                // Every element is decoded, so a malformed one is refused
                // even beside a foreign-typed index.
                let sections = self.optional_integer(sections, "probe sections cardinality")?;
                let tombstones =
                    self.optional_integer(tombstones, "probe tombstones cardinality")?;
                let fields = self.optional_integer(fields, "probe fields cardinality")?;
                let time_samples =
                    self.optional_integer(time_samples, "probe time-samples cardinality")?;
                let children_index =
                    self.optional_integer(children_index, "probe children-index cardinality")?;
                SyncChunk::Probe(
                    match (sections, tombstones, fields, time_samples, children_index) {
                        (
                            Some(sections),
                            Some(tombstones),
                            Some(fields),
                            Some(time_samples),
                            Some(children_index),
                        ) => Some(SyncIndexCardinalities {
                            sections,
                            tombstones,
                            fields,
                            time_samples,
                            children_index,
                        }),
                        _ => None,
                    },
                )
            }
            "NO_RANGE" => SyncChunk::NoRange,
            "OK" => {
                let end_seq = self.seq(next(), "chunk end seq")?;
                let entries = self
                    .array(next(), "chunk entries")?
                    .into_iter()
                    .map(|entry| self.stream_entry(entry))
                    .collect::<Result<Vec<_>, _>>()?;
                SyncChunk::Entries {
                    end_seq,
                    entries,
                    reached_end: self.flag(next(), "chunk reached flag")?,
                    bytes: self.integer(next(), "chunk byte tally")?,
                }
            }
            _ => SyncChunk::TooLarge {
                end_seq: self.seq(next(), "TOO_LARGE end seq")?,
                entry_id: self.text(next(), "TOO_LARGE entry id")?,
                entry_bytes: self.integer(next(), "TOO_LARGE entry size")?,
            },
        })
    }

    /// One `XRANGE` entry: `{id, {field, value, ...}}`. `payload` is read as
    /// text and `ompf` as raw bytes; other fields are ignored. Where a field
    /// name repeats, its first value is kept.
    fn stream_entry(&self, value: Value) -> Result<SyncStreamEntry, ServiceError> {
        let [id, fields] = self.fixed::<2>(value, "stream entry")?;
        let id = self.text(id, "stream entry id")?;
        let fields = self.array(fields, "stream entry fields")?;
        if fields.len() % 2 != 0 {
            return Err(self.malformed(format!(
                "stream entry {id} has {} field elements, expected an even number",
                fields.len()
            )));
        }
        let mut payload: Option<Option<String>> = None;
        let mut ompf: Option<Bytes> = None;
        let mut fields = fields.into_iter();
        while let (Some(name), Some(value)) = (fields.next(), fields.next()) {
            let name = self.bytes(name, "stream entry field name")?;
            let value = self.bytes(value, "stream entry field value")?;
            if name.as_ref() == NOTIFICATION_PAYLOAD_FIELD.as_bytes() && payload.is_none() {
                payload = Some(String::from_utf8(value.to_vec()).ok());
            } else if name.as_ref() == RELAY_OMPF_FIELD.as_bytes() && ompf.is_none() {
                ompf = Some(value);
            }
        }
        Ok(SyncStreamEntry {
            id,
            payload: payload.flatten(),
            ompf,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::layer::{
        HEADER_FIELD_CHS_COMPLETE, HEADER_FIELD_COMMIT_CHAIN, HEADER_FIELD_SEQ,
        HEADER_FIELD_SUPERSEDED,
    };

    // ── The script's source (Spec SDD SR-4.1) ───────────────────────────────
    //
    // "Every key declared, read-only, reads only through `pcall`" is asserted
    // here over the script's source and the Rust side's `KEYS` / `ARGV`.

    const STRUCTURAL_CACHE_SYNC_LUA: &str = include_str!("scripts/structural_cache_sync.lua");

    /// The script's code with `--` comments removed.
    fn script_code() -> String {
        STRUCTURAL_CACHE_SYNC_LUA
            .lines()
            .map(|line| line.split("--").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// `(what, command, key expression)` for every call of the script's
    /// per-key `call(what, cmd, key, ...)` wrapper, `what` and `command`
    /// unquoted.
    fn wrapper_calls() -> Vec<(String, String, String)> {
        let code = script_code();
        let mut out = Vec::new();
        let mut rest = code.as_str();
        while let Some(at) = rest.find("call(") {
            let before = &rest[..at];
            let args = &rest[at + "call(".len()..];
            rest = args;
            // `redis.pcall(`, `pcall(read)` and the wrapper's own definition
            // are not calls of the wrapper.
            if before.ends_with("function ")
                || before
                    .chars()
                    .last()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '.')
            {
                continue;
            }
            let mut depth = 0usize;
            let mut parts = vec![String::new()];
            for c in args.chars() {
                match c {
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' if depth > 0 => depth -= 1,
                    ')' => break,
                    ',' if depth == 0 => {
                        parts.push(String::new());
                        continue;
                    }
                    _ => {}
                }
                parts.last_mut().expect("never empty").push(c);
            }
            let parts: Vec<String> = parts.iter().map(|p| p.trim().to_owned()).collect();
            assert!(
                parts.len() >= 3,
                "call({args}) must pass what it reads, a command and a key"
            );
            let unquote = |s: &str| {
                s.strip_prefix('\'')
                    .and_then(|s| s.strip_suffix('\''))
                    .unwrap_or_else(|| panic!("{s} must be a quoted literal"))
                    .to_owned()
            };
            out.push((unquote(&parts[0]), unquote(&parts[1]), parts[2].clone()));
        }
        out
    }

    /// Every key the script reads is one of its declared `KEYS`, named by the
    /// local the label says; it reads no other `KEYS` entry and builds no
    /// key; and the Rust side's `KEYS` are exactly the header, stream and
    /// commit-token keys `KeySchema` builds — plus, in probe mode only, the
    /// five index keys — all under the object's `{oid}` hash tag. So the
    /// script runs in the object's slot and touches nothing outside it.
    #[test]
    fn structural_cache_sync_reads_only_its_declared_keys_under_the_objects_hash_tag() {
        let code = script_code();
        for declared in [
            "local hdr_key = KEYS[1]",
            "local stream_key = KEYS[2]",
            "local wtok_key = KEYS[3]",
            "local secs_key = KEYS[4]",
            "local dels_key = KEYS[5]",
            "local flds_key = KEYS[6]",
            "local tss_key = KEYS[7]",
            "local chs_key = KEYS[8]",
            "local key_count = 3",
            "if probe then key_count = 8 end",
            "if #KEYS ~= key_count then",
        ] {
            assert!(code.contains(declared), "the script must say `{declared}`");
        }
        assert_eq!(
            code.matches("KEYS[").count(),
            8,
            "the script reads KEYS[1..8] once each, into the eight named locals"
        );

        let calls = wrapper_calls();
        let mut read = std::collections::BTreeSet::new();
        for (what, command, key) in &calls {
            let expected = match what.as_str() {
                "header" => "hdr_key",
                "commit_token" => "wtok_key",
                "stream" => "stream_key",
                "sections_index" => "secs_key",
                "deleted_sections_index" => "dels_key",
                "fields_index" => "flds_key",
                "time_samples_index" => "tss_key",
                "children_index" => "chs_key",
                other => panic!("call label {other:?} names no declared key"),
            };
            assert_eq!(
                key, expected,
                "call('{what}', '{command}', {key}) must read the declared key its label names"
            );
            read.insert(what.as_str());
        }
        assert_eq!(
            read.into_iter().collect::<Vec<_>>(),
            [
                "children_index",
                "commit_token",
                "deleted_sections_index",
                "fields_index",
                "header",
                "sections_index",
                "stream",
                "time_samples_index",
            ],
            "the script reads all eight keys: {calls:?}"
        );

        let keys = KeySchema::new("test_tag:");
        let request = StructuralSyncRequest::repair(1234, "client-a", 7);
        let built = request.keys(&keys);
        assert_eq!(
            built,
            [
                keys.obj_header("1234"),
                keys.layer_stream(1234),
                keys.obj_write_token("1234"),
            ],
            "a repair call's KEYS[1..3] are the header, the layer stream and the commit-token \
             record"
        );
        let probe = StructuralSyncRequest::probe(1234, "client-a").keys(&keys);
        assert_eq!(
            probe,
            [
                keys.obj_header("1234"),
                keys.layer_stream(1234),
                keys.obj_write_token("1234"),
                keys.obj_sections_index("1234"),
                keys.obj_deleted_sections_index("1234"),
                keys.obj_fields_index("1234"),
                keys.obj_time_samples_index("1234"),
                keys.obj_children_index("1234"),
            ],
            "a probe declares the same three, then the five indexes in the order of its \
             cardinalities"
        );
        for key in built.iter().chain(&probe) {
            assert!(
                key.starts_with("test_tag:{1234}:obj:"),
                "{key} must carry the object's hash tag"
            );
        }
    }

    /// Read-only, unflagged, and every read through `redis.pcall`: no write
    /// command, no `#!lua` shebang (so it runs in compat mode, SDD §7.5.1, and
    /// keeps running on a store over `maxmemory`), and no `redis.call`, whose
    /// error would abort the script untagged instead of naming what was read.
    #[test]
    fn structural_cache_sync_is_read_only_unflagged_and_reads_through_pcall() {
        assert!(!STRUCTURAL_CACHE_SYNC_LUA.starts_with("#!"), "no shebang");
        let code = script_code();
        assert!(
            !code.contains("redis.call("),
            "every read goes through redis.pcall"
        );
        assert_eq!(
            code.matches("redis.pcall(").count(),
            1,
            "the one redis.pcall is the per-key wrapper's, so every command goes through it"
        );
        const READS: [&str; 6] = ["HGETALL", "TYPE", "HGET", "XRANGE", "SCARD", "ZCARD"];
        let calls = wrapper_calls();
        assert!(
            calls.len() >= 4,
            "the call scan found too few calls: {calls:?}"
        );
        for (what, command, key) in calls {
            assert!(
                READS.contains(&command.as_str()),
                "call('{what}', '{command}', {key}) is not a read"
            );
        }
    }

    /// The header columns the script reads by literal name are the columns the
    /// Rust side names, and its format tag is the one the decoder expects.
    #[test]
    fn structural_cache_sync_reads_the_named_header_columns_and_tags_its_format() {
        let code = script_code();
        for column in [
            HEADER_FIELD_SEQ,
            HEADER_FIELD_COMMIT_CHAIN,
            HEADER_FIELD_SUPERSEDED,
            HEADER_FIELD_CHS_COMPLETE,
        ] {
            assert!(
                code.contains(&format!("cols['{column}']")),
                "the script must read the header column {column:?}"
            );
        }
        assert!(
            code.contains(&format!("local FORMAT = '{STRUCTURAL_CACHE_SYNC_FORMAT}'")),
            "the script's format tag must be the decoder's"
        );
    }

    /// `ARGV` positions: entry `seq` ('' in probe mode), entry bound, byte
    /// bound, `client_id`, target `seq` ('' when absent), page size — the six
    /// the script requires.
    #[test]
    fn structural_cache_sync_argv_puts_every_argument_in_its_position() {
        assert!(script_code().contains("if #ARGV ~= 6 then"));
        let bounds = [
            STRUCTURAL_REPAIR_MAX_ENTRIES.to_string(),
            STRUCTURAL_REPAIR_MAX_BYTES.to_string(),
        ];
        let page = STRUCTURAL_REPAIR_PAGE_ENTRIES.to_string();
        assert_eq!(
            StructuralSyncRequest::probe(9, "client-a").argv(),
            ["", &bounds[0], &bounds[1], "client-a", "", &page]
        );
        assert_eq!(
            StructuralSyncRequest::repair(9, "client-a", 41).argv(),
            ["41", &bounds[0], &bounds[1], "client-a", "", &page]
        );
        assert_eq!(
            StructuralSyncRequest::repair_to(9, "client-a", 41, 50).argv(),
            ["41", &bounds[0], &bounds[1], "client-a", "50", &page]
        );
        let tight = StructuralSyncRequest {
            max_entries: 3,
            max_bytes: 100,
            page_entries: 2,
            ..StructuralSyncRequest::repair(9, "c", 0)
        };
        assert_eq!(tight.argv(), ["0", "3", "100", "c", "", "2"]);
    }

    // ── The reply decoder ───────────────────────────────────────────────────

    fn text(s: &str) -> Value {
        Value::String(s.into())
    }

    fn int(n: i64) -> Value {
        Value::Integer(n)
    }

    fn arr(items: Vec<Value>) -> Value {
        Value::Array(items)
    }

    /// A present header at `seq` 7, chain `"c0ffee"`, marked, not superseded.
    fn header_at_7() -> Value {
        arr(vec![
            int(1),
            arr(vec![text("7")]),
            text("c0ffee"),
            arr(vec![]),
            arr(vec![text("1")]),
        ])
    }

    fn absent_header() -> Value {
        arr(vec![
            int(0),
            arr(vec![]),
            text(""),
            arr(vec![]),
            arr(vec![]),
        ])
    }

    fn hash_token(record: &str) -> Value {
        arr(vec![text("hash"), arr(vec![text(record)])])
    }

    /// Probe cardinalities `{secs 1, dels 2, flds 3, tss 4, chs 5}`, each in
    /// its optional `{n}` form.
    fn probe_cards() -> Value {
        arr((1..=5).map(|n| arr(vec![int(n)])).collect())
    }

    fn probe_chunk() -> Value {
        arr(vec![text("PROBE"), probe_cards()])
    }

    fn reply(format: &str, header: Value, token: Value, chunk: Value) -> Value {
        arr(vec![text(format), header, token, chunk])
    }

    fn decode(reply: Value) -> Result<StructuralSyncReply, ServiceError> {
        decode_structural_cache_sync_reply(42, reply)
    }

    #[track_caller]
    fn assert_internal(result: Result<StructuralSyncReply, ServiceError>, why: &str) {
        match result {
            Err(ServiceError::Internal(_)) => {}
            other => panic!("{why}: expected Internal, got {other:?}"),
        }
    }

    /// An `ompf` value that is not valid UTF-8 decodes byte for byte; `payload`
    /// decodes as text; the chunk's end, reached flag and tally come through.
    #[test]
    fn an_ok_chunk_decodes_binary_ompf_intact_and_payload_as_text() {
        // Invalid UTF-8 from its first byte (0xff never occurs in UTF-8).
        let binary: &'static [u8] = &[0xff, 0xfe, 0x00, 0x80, b'O', 0xc3];
        let entry = arr(vec![
            text("8-0"),
            arr(vec![
                text("payload"),
                text("update|42|8|client-a|5|6"),
                text("ompf"),
                Value::Bytes(Bytes::from_static(binary)),
            ]),
        ]);
        // A writer's OMPF that happens to be valid UTF-8 reaches fred as a
        // string; it decodes to the same bytes.
        let utf8_ompf = arr(vec![
            text("9-0"),
            arr(vec![
                text("ompf"),
                text("OMPF"),
                text("payload"),
                text("update|42|9|c|5|6"),
            ]),
        ]);
        let chunk = arr(vec![
            text("OK"),
            text("9"),
            arr(vec![entry, utf8_ompf]),
            int(1),
            int(77),
        ]);
        let decoded = decode(reply(
            STRUCTURAL_CACHE_SYNC_FORMAT,
            header_at_7(),
            hash_token("tok:7:100"),
            chunk,
        ))
        .expect("a well-formed reply decodes");

        assert_eq!(
            decoded.chunk,
            SyncChunk::Entries {
                end_seq: 9,
                entries: vec![
                    SyncStreamEntry {
                        id: "8-0".to_owned(),
                        payload: Some("update|42|8|client-a|5|6".to_owned()),
                        ompf: Some(Bytes::from_static(binary)),
                    },
                    SyncStreamEntry {
                        id: "9-0".to_owned(),
                        payload: Some("update|42|9|c|5|6".to_owned()),
                        ompf: Some(Bytes::from_static(b"OMPF")),
                    },
                ],
                reached_end: true,
                bytes: 77,
            }
        );
        assert_eq!(
            decoded.header,
            SyncHeader::Present(SyncHeaderFacts {
                seq: Some(7),
                commit_chain: "c0ffee".to_owned(),
                superseded: None,
                chs_complete: Some("1".to_owned()),
            })
        );
        assert_eq!(
            decoded.commit_token,
            SyncCommitToken {
                key_type: "hash".to_owned(),
                record: Some("tok:7:100".to_owned()),
            }
        );
    }

    /// An entry with no `ompf` (an empty commit, the `1-0` seed, a `deleted`
    /// tombstone) and an entry whose `payload` is absent or not UTF-8 decode
    /// as data, not as a malformed reply: they are the caller's to classify.
    #[test]
    fn entries_missing_fields_or_with_a_non_utf8_payload_decode_as_data() {
        let chunk = arr(vec![
            text("OK"),
            text("4"),
            arr(vec![
                arr(vec![
                    text("2-0"),
                    arr(vec![text("payload"), text("deleted")]),
                ]),
                arr(vec![text("3-0"), arr(vec![text("ompf"), text("x")])]),
                arr(vec![
                    text("4-0"),
                    arr(vec![
                        text("payload"),
                        Value::Bytes(Bytes::from_static(&[0xff])),
                    ]),
                ]),
            ]),
            int(0),
            int(30),
        ]);
        let decoded = decode(reply(
            STRUCTURAL_CACHE_SYNC_FORMAT,
            header_at_7(),
            hash_token("t"),
            chunk,
        ))
        .expect("entries missing fields are data");
        let SyncChunk::Entries {
            entries,
            reached_end,
            ..
        } = decoded.chunk
        else {
            panic!("an OK chunk decodes as Entries");
        };
        assert!(!reached_end);
        assert_eq!(entries[0].payload.as_deref(), Some("deleted"));
        assert_eq!(entries[0].ompf, None);
        assert_eq!(entries[1].payload, None, "absent payload");
        assert_eq!(entries[2].payload, None, "non-UTF-8 payload");
    }

    #[test]
    fn a_probe_reply_and_its_header_and_token_facts_decode() {
        let probe = probe_chunk();
        let decoded = decode(reply(
            STRUCTURAL_CACHE_SYNC_FORMAT,
            arr(vec![
                int(1),
                arr(vec![]),
                text(""),
                arr(vec![text("1")]),
                arr(vec![]),
            ]),
            arr(vec![text("string"), arr(vec![])]),
            probe,
        ))
        .expect("a probe reply decodes");
        let cards = SyncIndexCardinalities {
            sections: 1,
            tombstones: 2,
            fields: 3,
            time_samples: 4,
            children_index: 5,
        };
        assert_eq!(
            decoded.chunk,
            SyncChunk::Probe(Some(cards)),
            "the five cardinalities decode in KEYS[4..8] order"
        );
        // Any index reported as a foreign type (`{}`) leaves no
        // cardinalities, and the rest of the reply still decodes.
        for refused in 0..5 {
            let elements = (1..=5)
                .map(|n| {
                    if n - 1 == refused {
                        arr(vec![])
                    } else {
                        arr(vec![int(n)])
                    }
                })
                .collect();
            let decoded = decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![text("PROBE"), arr(elements)]),
            ))
            .unwrap_or_else(|e| panic!("index {refused} refused: the reply decodes; got {e:?}"));
            assert_eq!(
                (decoded.chunk, decoded.commit_token.record.as_deref()),
                (SyncChunk::Probe(None), Some("t")),
                "index {refused} refused: no cardinalities, the token record intact"
            );
        }
        // The children-list index counts only on a marked object, as the
        // atomic read's step 2 reads it only then.
        assert_eq!(cards.snapshot_cardinalities(true).children_index, Some(5));
        assert_eq!(cards.snapshot_cardinalities(false).children_index, None);
        assert_eq!(
            (
                cards.snapshot_cardinalities(false).sections,
                cards.snapshot_cardinalities(false).tombstones,
                cards.snapshot_cardinalities(false).fields,
                cards.snapshot_cardinalities(false).time_samples,
            ),
            (1, 2, 3, 4)
        );
        let SyncHeader::Present(facts) = decoded.header else {
            panic!("a header with fields is present");
        };
        assert_eq!(facts.seq, None, "no decimal seq");
        assert_eq!(facts.commit_chain, "", "an absent commit_chain is empty");
        assert_eq!(facts.superseded.as_deref(), Some("1"));
        assert!(!facts.chs_complete_marked());
        assert!(
            decoded.commit_token.is_wrong_type(),
            "a foreign record type is reported as data"
        );
        assert_eq!(decoded.commit_token.record, None);
    }

    #[test]
    fn an_absent_header_a_no_range_chunk_and_a_too_large_chunk_decode() {
        let decoded = decode(reply(
            STRUCTURAL_CACHE_SYNC_FORMAT,
            absent_header(),
            arr(vec![text("none"), arr(vec![])]),
            arr(vec![text("NO_RANGE")]),
        ))
        .expect("an absent header decodes");
        assert_eq!(decoded.header, SyncHeader::Absent);
        assert_eq!(decoded.chunk, SyncChunk::NoRange);
        assert!(!decoded.commit_token.is_wrong_type());

        let decoded = decode(reply(
            STRUCTURAL_CACHE_SYNC_FORMAT,
            header_at_7(),
            hash_token("t"),
            arr(vec![
                text("TOO_LARGE"),
                text("7"),
                text("5-0"),
                int(5_000_000),
            ]),
        ))
        .expect("a TOO_LARGE chunk decodes");
        assert_eq!(
            decoded.chunk,
            SyncChunk::TooLarge {
                end_seq: 7,
                entry_id: "5-0".to_owned(),
                entry_bytes: 5_000_000,
            }
        );
    }

    #[test]
    fn a_reply_in_another_format_is_internal() {
        assert_internal(
            decode(reply(
                "structural-cache-sync-v0",
                header_at_7(),
                hash_token("t"),
                probe_chunk(),
            )),
            "a format tag this binary does not carry",
        );
        assert_internal(
            decode(arr(vec![
                int(1),
                header_at_7(),
                hash_token("t"),
                probe_chunk(),
            ])),
            "a format tag that is not a string",
        );
    }

    #[test]
    fn a_reply_of_the_wrong_arity_is_internal() {
        let ok = || arr(vec![text("OK"), text("7"), arr(vec![]), int(1), int(0)]);
        assert_internal(
            decode(arr(vec![
                text(STRUCTURAL_CACHE_SYNC_FORMAT),
                header_at_7(),
                hash_token("t"),
            ])),
            "a three-element reply",
        );
        assert_internal(
            decode(arr(vec![
                text(STRUCTURAL_CACHE_SYNC_FORMAT),
                header_at_7(),
                hash_token("t"),
                ok(),
                int(0),
            ])),
            "a five-element reply",
        );
        assert_internal(
            decode(text(STRUCTURAL_CACHE_SYNC_FORMAT)),
            "a reply that is not an array",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                arr(vec![int(1), arr(vec![text("7")]), text(""), arr(vec![])]),
                hash_token("t"),
                ok(),
            )),
            "a four-element header",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                arr(vec![text("hash")]),
                ok(),
            )),
            "a one-element token",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![text("OK"), text("7"), arr(vec![]), int(1)]),
            )),
            "a four-element OK chunk",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![text("PROBE")]),
            )),
            "a PROBE chunk without its cardinalities",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![
                    text("PROBE"),
                    arr((1..=4).map(|n| arr(vec![int(n)])).collect()),
                ]),
            )),
            "four probe cardinalities",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![
                    text("PROBE"),
                    arr(vec![int(1), int(2), int(3), int(4), int(5)]),
                ]),
            )),
            "bare cardinalities, not in their optional form",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![
                    text("PROBE"),
                    arr(vec![
                        arr(vec![int(1), int(1)]),
                        arr(vec![int(2)]),
                        arr(vec![int(3)]),
                        arr(vec![int(4)]),
                        arr(vec![int(5)]),
                    ]),
                ]),
            )),
            "a cardinality of two elements",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![
                    text("PROBE"),
                    arr(vec![
                        arr(vec![]),
                        arr(vec![int(-2)]),
                        arr(vec![int(3)]),
                        arr(vec![int(4)]),
                        arr(vec![int(5)]),
                    ]),
                ]),
            )),
            "a negative cardinality beside a foreign-typed index",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![text("PROBE"), probe_cards(), int(0)]),
            )),
            "a three-element PROBE chunk",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![text("TOO_LARGE"), text("7"), text("5-0")]),
            )),
            "a three-element TOO_LARGE chunk",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![
                    text("OK"),
                    text("7"),
                    arr(vec![arr(vec![text("5-0")])]),
                    int(1),
                    int(0),
                ]),
            )),
            "a stream entry without its fields",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![
                    text("OK"),
                    text("7"),
                    arr(vec![arr(vec![text("5-0"), arr(vec![text("payload")])])]),
                    int(1),
                    int(0),
                ]),
            )),
            "a stream entry with an odd field list",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![text("DONE")]),
            )),
            "an unknown chunk status",
        );
    }

    /// The script vouches for these values, so one that is off means the script
    /// that ran is not this binary's.
    #[test]
    fn values_the_script_vouches_for_are_checked() {
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                arr(vec![
                    int(1),
                    arr(vec![text("seven")]),
                    text(""),
                    arr(vec![]),
                    arr(vec![]),
                ]),
                hash_token("t"),
                probe_chunk(),
            )),
            "a reported seq that is not decimal",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                arr(vec![
                    int(0),
                    arr(vec![text("7")]),
                    text(""),
                    arr(vec![]),
                    arr(vec![]),
                ]),
                hash_token("t"),
                probe_chunk(),
            )),
            "an absent header carrying a seq",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                arr(vec![
                    int(2),
                    arr(vec![]),
                    text(""),
                    arr(vec![]),
                    arr(vec![]),
                ]),
                hash_token("t"),
                probe_chunk(),
            )),
            "a presence flag that is not 0 or 1",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                arr(vec![text("list"), arr(vec![text("t")])]),
                probe_chunk(),
            )),
            "a foreign-typed record carrying a value",
        );
        assert_internal(
            decode(reply(
                STRUCTURAL_CACHE_SYNC_FORMAT,
                header_at_7(),
                hash_token("t"),
                arr(vec![text("OK"), text("7"), arr(vec![]), int(1), int(-1)]),
            )),
            "a negative byte tally",
        );
    }
}
