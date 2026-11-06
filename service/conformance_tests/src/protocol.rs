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

//! The wire protocol as a client sees it, written out here so the suite stays
//! independent of the service code it checks:
//!
//! - the rich error model a refusal carries in `grpc-status-details-bin`
//!   ([`error_info`], [`carries_detail`]), read through tonic-types;
//! - the OMPF container ([`pack_ompf`], [`unpack_ompf`]);
//! - minimal FlatBuffer writers for deltas the suite builds rather than ships
//!   ([`delta_of_entries`], [`delta_of_children_list`], [`delta_deleting_section`]);
//! - the external-value [`content_hash`];
//! - an in-process OODF decode to a canonical [`Signature`] that validates
//!   every external reference ([`decode`]), through `oodf_ffi::parse_delta`;
//! - catch-up checks and the client-side fold ([`snapshot`],
//!   [`apply_snapshot`], [`field_order`]).
//!
//! No server state logic: expected state is computed by folding signatures,
//! never by the service's own engine.

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tonic_types::StatusExt;

use crate::check::{CaseError, CaseResult};

/// The `ErrorInfo.domain` of the service's refusals.
pub const ERROR_DOMAIN: &str = "nvidia.omniverse.live.v1alpha";

/// Block size of the external-value content hash: a fixed protocol constant.
pub const CONTENT_HASH_BLOCK_BYTES: usize = 1 << 20;

/// The `google.rpc.ErrorInfo` a refusal carries.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ErrorInfo {
    /// The refusal's reason, e.g. `PREDECESSOR_NOT_COMMITTED`.
    pub reason: String,
    /// Normally [`ERROR_DOMAIN`].
    pub domain: String,
    /// The reason's metadata.
    pub metadata: BTreeMap<String, String>,
}

impl ErrorInfo {
    /// An `ErrorInfo` to compare against: `reason` in `domain` with
    /// `metadata` pairs.
    pub fn new(reason: &str, domain: &str, metadata: &[(&str, &str)]) -> Self {
        Self {
            reason: reason.to_string(),
            domain: domain.to_string(),
            metadata: metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }
}

/// The `type_url` of a `google.rpc.ErrorInfo` detail.
pub const ERROR_INFO_TYPE: &str = "type.googleapis.com/google.rpc.ErrorInfo";

/// The `type_url` of a `google.rpc.RetryInfo` detail.
pub const RETRY_INFO_TYPE: &str = "type.googleapis.com/google.rpc.RetryInfo";

/// The `ErrorInfo` in `status`'s `grpc-status-details-bin`, or `None` when it
/// carries none (or the details do not decode). For checking that a status
/// carries no `ErrorInfo`, use [`carries_detail`] with [`ERROR_INFO_TYPE`]:
/// it fails on details that do not decode rather than reading them as none.
pub fn error_info(status: &tonic::Status) -> Option<ErrorInfo> {
    let info = status.get_details_error_info()?;
    Some(ErrorInfo {
        reason: info.reason,
        domain: info.domain,
        metadata: info.metadata.into_iter().collect(),
    })
}

/// Whether any detail in `status`'s `grpc-status-details-bin` has the
/// `type_url` `type_url` ([`ERROR_INFO_TYPE`], [`RETRY_INFO_TYPE`]), whatever
/// its value holds. No details is `false`; details that do not decode as a
/// `google.rpc.Status` fail the check, so a malformed detail never reads as
/// absent.
pub fn carries_detail(status: &tonic::Status, type_url: &str) -> CaseResult<bool> {
    let details = status.details();
    if details.is_empty() {
        return Ok(false);
    }
    let decoded = <tonic_types::pb::Status as prost::Message>::decode(details).map_err(|_| {
        CaseError::new("grpc-status-details-bin does not decode as a google.rpc.Status")
    })?;
    Ok(decoded
        .details
        .iter()
        .any(|detail| detail.type_url == type_url))
}

/// The OMPF magic, the container's first four bytes.
const OMPF_MAGIC: &[u8; 4] = b"OMPF";

/// An OMPF container holding `delta` as its only part.
pub fn pack_ompf(delta: &[u8]) -> Vec<u8> {
    // The magic and the part count, then the one part's offset: 16.
    let header: u64 = 8 + 8;
    let mut out = Vec::with_capacity(16 + delta.len());
    out.extend_from_slice(OMPF_MAGIC);
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&header.to_le_bytes());
    out.extend_from_slice(delta);
    out
}

/// The parts of an OMPF container: part 0 the OODF delta, then each external
/// value. Fails unless `payload` is a complete container: the `OMPF` magic, a
/// part count from 1 to what fits, and in-order offsets past the offset table.
pub fn unpack_ompf(payload: &[u8]) -> CaseResult<Vec<&[u8]>> {
    if payload.len() < 16 || &payload[..4] != OMPF_MAGIC {
        return Err(CaseError::new("Response is not a complete OMPF container"));
    }
    let count = u32::from_le_bytes(payload[4..8].try_into().expect("4 bytes")) as usize;
    if !(1..=(payload.len() - 8) / 8).contains(&count) {
        return Err(CaseError::new("Invalid OMPF part count"));
    }
    let invalid_offsets = || CaseError::new("Invalid OMPF part offsets");
    let mut offsets = Vec::with_capacity(count + 1);
    for part in 0..count {
        let at = 8 + 8 * part;
        let offset = u64::from_le_bytes(payload[at..at + 8].try_into().expect("8 bytes"));
        offsets.push(usize::try_from(offset).map_err(|_| invalid_offsets())?);
    }
    offsets.push(payload.len());
    if offsets[0] < 8 + count * 8 || offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(invalid_offsets());
    }
    Ok(offsets
        .windows(2)
        .map(|pair| &payload[pair[0]..pair[1]])
        .collect())
}

/// Just enough of a FlatBuffers writer for the deltas below.
///
/// Laid out front to back: the root offset and the file identifier, then each
/// table after its vtable, then what the table points at, so every unsigned
/// offset points forward as the format requires. Little-endian, every scalar
/// aligned to its size, as the verifier checks.
struct FlatBuffer {
    data: Vec<u8>,
}

impl FlatBuffer {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Pads with zero bytes until the length is `remainder` modulo `size`.
    fn align(&mut self, size: usize, remainder: usize) {
        while self.data.len() % size != remainder {
            self.data.push(0);
        }
    }

    /// Appends `raw` and returns where it starts.
    fn put(&mut self, raw: &[u8]) -> usize {
        let position = self.data.len();
        self.data.extend_from_slice(raw);
        position
    }

    /// Points the unsigned offset at `at` to `target`, which lies after it.
    fn offset_to(&mut self, at: usize, target: usize) {
        assert!(target > at, "FlatBuffers offsets point forward");
        let offset = u32::try_from(target - at).expect("offset fits in 32 bits");
        self.data[at..at + 4].copy_from_slice(&offset.to_le_bytes());
    }

    /// A vtable naming each field's inline slot (0 = absent), then its
    /// table, which starts with its offset back to the vtable. Returns the
    /// table's position; the caller writes the inline fields after it.
    fn table(&mut self, slots: &[u16], inline_size: u16) -> usize {
        self.align(4, 0);
        let vtable_size = u16::try_from(4 + 2 * slots.len()).expect("small vtable");
        let mut vtable_bytes = Vec::with_capacity(usize::from(vtable_size));
        vtable_bytes.extend_from_slice(&vtable_size.to_le_bytes());
        vtable_bytes.extend_from_slice(&inline_size.to_le_bytes());
        for slot in slots {
            vtable_bytes.extend_from_slice(&slot.to_le_bytes());
        }
        let vtable = self.put(&vtable_bytes);
        self.align(4, 0);
        let table = self.put(&0i32.to_le_bytes());
        let back = i32::try_from(table - vtable).expect("vtable offset fits in 32 bits");
        self.data[table..table + 4].copy_from_slice(&back.to_le_bytes());
        table
    }

    fn into_bytes(self) -> Vec<u8> {
        self.data
    }
}

// Union discriminants of OmniObjectDelta's StructCommand (schema order).
const COMMAND_DELETE_SECTION: u8 = 2;
const COMMAND_REORDER_CHILDREN: u8 = 4;

/// The root offset placeholder and the `OODF` file identifier.
fn preamble(buf: &mut FlatBuffer) {
    buf.put(&0u32.to_le_bytes());
    buf.put(b"OODF");
}

/// An OODF delta of `struct_commands` `DeleteSection` commands and
/// `set_fields` `SetField`s, every element of a kind pointing at one shared
/// all-default table: a valid FlatBuffer whose vector lengths are the counts,
/// small enough to send, which a server refuses for its counts before it
/// walks any element.
pub fn delta_of_entries(struct_commands: u32, set_fields: u32) -> Vec<u8> {
    let commands_len = struct_commands as usize;
    let fields_len = set_fields as usize;
    let mut buf = FlatBuffer::new();
    preamble(&mut buf);
    // Delta fields: isDiff, baseVersion, structCommands_type, structCommands, setFields.
    let delta = buf.table(&[0, 0, 4, 8, 12], 16);
    buf.put(&[0u8; 12]);
    buf.offset_to(0, delta);

    buf.align(4, 0);
    let types = buf.put(&struct_commands.to_le_bytes());
    buf.put(&vec![COMMAND_DELETE_SECTION; commands_len]);
    buf.align(4, 0);
    let commands = buf.put(&struct_commands.to_le_bytes());
    buf.put(&vec![0u8; 4 * commands_len]);
    buf.align(4, 0);
    let fields = buf.put(&set_fields.to_le_bytes());
    buf.put(&vec![0u8; 4 * fields_len]);

    let shared = buf.table(&[], 4);
    for (vector, count) in [(commands, commands_len), (fields, fields_len)] {
        for i in 0..count {
            buf.offset_to(vector + 4 + 4 * i, shared);
        }
    }
    buf.offset_to(delta + 4, types);
    buf.offset_to(delta + 8, commands);
    buf.offset_to(delta + 12, fields);
    buf.into_bytes()
}

/// An OODF delta of one `ReorderChildren` command under section 0 whose
/// children list names the ids `1..=children`.
pub fn delta_of_children_list(children: u32) -> Vec<u8> {
    let mut buf = FlatBuffer::new();
    preamble(&mut buf);
    // Delta fields: isDiff, baseVersion, structCommands_type, structCommands.
    let delta = buf.table(&[0, 0, 4, 8], 12);
    buf.put(&[0u8; 8]);
    buf.offset_to(0, delta);

    buf.align(4, 0);
    let types = buf.put(&1u32.to_le_bytes());
    buf.put(&[COMMAND_REORDER_CHILDREN]);
    buf.align(4, 0);
    let commands = buf.put(&1u32.to_le_bytes());
    buf.put(&0u32.to_le_bytes());
    // ReorderChildren fields: sectionId, childrenListId, childrenList.
    let reorder = buf.table(&[0, 0, 4], 8);
    buf.put(&0u32.to_le_bytes());
    buf.align(8, 4); // the u64 elements after the length word are 8-aligned
    let children_list = buf.put(&children.to_le_bytes());
    let mut ids = Vec::with_capacity(8 * children as usize);
    for id in 1..=u64::from(children) {
        ids.extend_from_slice(&id.to_le_bytes());
    }
    buf.put(&ids);

    buf.offset_to(delta + 4, types);
    buf.offset_to(delta + 8, commands);
    buf.offset_to(commands + 4, reorder);
    buf.offset_to(reorder + 4, children_list);
    buf.into_bytes()
}

/// An ordinary OODF delta of one `DeleteSection` command removing
/// `section_id` from under `parent_id`.
pub fn delta_deleting_section(parent_id: u64, section_id: u64) -> Vec<u8> {
    let mut buf = FlatBuffer::new();
    preamble(&mut buf);
    // Delta fields: isDiff, baseVersion, structCommands_type, structCommands.
    let delta = buf.table(&[0, 0, 4, 8], 12);
    buf.put(&[0u8; 8]);
    buf.offset_to(0, delta);

    buf.align(4, 0);
    let types = buf.put(&1u32.to_le_bytes());
    buf.put(&[COMMAND_DELETE_SECTION]);
    buf.align(4, 0);
    let commands = buf.put(&1u32.to_le_bytes());
    buf.put(&0u32.to_le_bytes());
    // DeleteSection fields: parentId, sectionId. The vtable starts 8-aligned
    // and is 8 bytes long, so the table does too, and its u64 fields after the
    // 4-byte offset and 4 bytes of padding are 8-aligned.
    buf.align(8, 0);
    let delete = buf.table(&[8, 16], 24);
    buf.put(&0u32.to_le_bytes());
    buf.put(&parent_id.to_le_bytes());
    buf.put(&section_id.to_le_bytes());

    buf.offset_to(delta + 4, types);
    buf.offset_to(delta + 8, commands);
    buf.offset_to(commands + 4, delete);
    buf.into_bytes()
}

/// Lowercase hex of `bytes`.
fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// The raw 32-byte content hash of `value`, as an OODF `valueOrExtHash`
/// carries it (see [`content_hash`]).
pub(crate) fn content_digest(value: &[u8]) -> [u8; 32] {
    let mut outer = Sha256::new();
    for block in value.chunks(CONTENT_HASH_BLOCK_BYTES) {
        outer.update(Sha256::digest(block));
    }
    outer.finalize().into()
}

/// The lowercase-hex content hash an external value's `valueOrExtHash` must
/// carry: SHA-256 over the concatenated SHA-256 digests of the value's
/// consecutive [`CONTENT_HASH_BLOCK_BYTES`] blocks; an empty value hashes to
/// SHA-256 of the empty string. The clients' definition, not plain SHA-256.
pub fn content_hash(value: &[u8]) -> String {
    hex(&content_digest(value))
}

/// The canonical logical signature of an OODF delta, in the shape the
/// acceptance oracle and `captured-golden.json` use (camelCase keys). Arrays
/// are in canonical order: set fields by (section, key, set order), time
/// samples by (section, time), struct commands by their key-sorted JSON
/// rendering. OMPF positions (`ext_value_index`) are dropped.
///
/// Fields are declared in key order, so serializing a signature writes its
/// keys sorted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Signature {
    pub base_version: u64,
    /// Plain SHA-256 (hex, sorted) of each external part: a fingerprint of
    /// the blobs, not the wire's content hash.
    pub external_value_hashes: Vec<String>,
    pub is_diff: bool,
    pub set_fields: Vec<SignatureField>,
    pub source_format: String,
    pub struct_commands: Vec<SignatureCommand>,
    pub time_samples: Vec<SignatureSample>,
}

/// One `SetField` of a [`Signature`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignatureField {
    pub ext_value_size: u64,
    pub key_name: String,
    pub section_id: u64,
    pub set_order: u64,
    /// Lowercase hex of the inline value, or of the external value's content
    /// hash when `ext_value_size` is non-zero.
    pub value_or_ext_hash: String,
}

/// One `TimeSample` of a [`Signature`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SignatureSample {
    pub ext_value_size: u64,
    pub section_id: u64,
    pub time: f64,
    /// As [`SignatureField::value_or_ext_hash`].
    pub value_or_ext_hash: String,
}

/// One struct command of a [`Signature`], tagged by its `type`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SignatureCommand {
    #[serde(rename_all = "camelCase")]
    CommandCreateSection {
        parent_id: u64,
        section_id: u64,
        section_name: String,
        section_type: u8,
    },
    #[serde(rename_all = "camelCase")]
    CommandDeleteSection { parent_id: u64, section_id: u64 },
    #[serde(rename_all = "camelCase")]
    CommandMoveSection {
        new_name: String,
        new_parent_id: u64,
        old_parent_id: u64,
        section_id: u64,
    },
    #[serde(rename_all = "camelCase")]
    CommandReorderChildren {
        children_list: Vec<u64>,
        children_list_id: u8,
        section_id: u64,
    },
    #[serde(rename_all = "camelCase")]
    CommandDiffChangeSection {
        parent_id: u64,
        section_id: u64,
        section_name: String,
        section_order: u32,
        section_type: u8,
    },
    #[serde(rename_all = "camelCase")]
    CommandDiffDeleteSection { section_id: u64 },
}

impl SignatureCommand {
    /// The command's `type` tag, e.g. `CommandDiffChangeSection`.
    fn type_name(&self) -> &'static str {
        match self {
            Self::CommandCreateSection { .. } => "CommandCreateSection",
            Self::CommandDeleteSection { .. } => "CommandDeleteSection",
            Self::CommandMoveSection { .. } => "CommandMoveSection",
            Self::CommandReorderChildren { .. } => "CommandReorderChildren",
            Self::CommandDiffChangeSection { .. } => "CommandDiffChangeSection",
            Self::CommandDiffDeleteSection { .. } => "CommandDiffDeleteSection",
        }
    }

    /// The command as compact, key-sorted JSON: the key struct commands are
    /// sorted by, rendered as `decode_ompf --json` renders it. Written by
    /// hand so the key order does not depend on serde_json's map features.
    fn canonical_json(&self) -> String {
        let kind = json_string(self.type_name());
        match self {
            Self::CommandCreateSection {
                parent_id,
                section_id,
                section_name,
                section_type,
            } => format!(
                "{{\"parentId\":{parent_id},\"sectionId\":{section_id},\"sectionName\":{},\"sectionType\":{section_type},\"type\":{kind}}}",
                json_string(section_name),
            ),
            Self::CommandDeleteSection {
                parent_id,
                section_id,
            } => format!("{{\"parentId\":{parent_id},\"sectionId\":{section_id},\"type\":{kind}}}"),
            Self::CommandMoveSection {
                new_name,
                new_parent_id,
                old_parent_id,
                section_id,
            } => format!(
                "{{\"newName\":{},\"newParentId\":{new_parent_id},\"oldParentId\":{old_parent_id},\"sectionId\":{section_id},\"type\":{kind}}}",
                json_string(new_name),
            ),
            Self::CommandReorderChildren {
                children_list,
                children_list_id,
                section_id,
            } => format!(
                "{{\"childrenList\":[{}],\"childrenListId\":{children_list_id},\"sectionId\":{section_id},\"type\":{kind}}}",
                children_list
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            Self::CommandDiffChangeSection {
                parent_id,
                section_id,
                section_name,
                section_order,
                section_type,
            } => format!(
                "{{\"parentId\":{parent_id},\"sectionId\":{section_id},\"sectionName\":{},\"sectionOrder\":{section_order},\"sectionType\":{section_type},\"type\":{kind}}}",
                json_string(section_name),
            ),
            Self::CommandDiffDeleteSection { section_id } => {
                format!("{{\"sectionId\":{section_id},\"type\":{kind}}}")
            }
        }
    }
}

impl From<&oodf_ffi::StructCommand> for SignatureCommand {
    fn from(command: &oodf_ffi::StructCommand) -> Self {
        use oodf_ffi::StructCommand as C;
        match command {
            C::CreateSection(x) => Self::CommandCreateSection {
                parent_id: x.parent_id,
                section_id: x.section_id,
                section_name: x.section_name.clone(),
                section_type: x.section_type,
            },
            C::DeleteSection(x) => Self::CommandDeleteSection {
                parent_id: x.parent_id,
                section_id: x.section_id,
            },
            C::MoveSection(x) => Self::CommandMoveSection {
                new_name: x.new_name.clone(),
                new_parent_id: x.new_parent_id,
                old_parent_id: x.old_parent_id,
                section_id: x.section_id,
            },
            C::ReorderChildren(x) => Self::CommandReorderChildren {
                children_list: x.children_list.clone(),
                children_list_id: x.children_list_id,
                section_id: x.section_id,
            },
            C::DiffChangeSection(x) => Self::CommandDiffChangeSection {
                parent_id: x.parent_id,
                section_id: x.section_id,
                section_name: x.section_name.clone(),
                section_order: x.section_order,
                section_type: x.section_type,
            },
            C::DiffDeleteSection(x) => Self::CommandDiffDeleteSection {
                section_id: x.section_id,
            },
        }
    }
}

/// `s` as a JSON string literal, escaped exactly as `decode_ompf --json`
/// escapes it, so struct commands sort in the order that decoder gives them.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Validates the delta's external references against the OMPF parts before
/// the signature drops their positions: every reference (a non-zero
/// `ext_value_size`) must name, by its 1-based `ext_value_index`, a part that
/// is exactly that many bytes long and whose content hash is its
/// `value_or_ext_hash`.
fn validate_externals(delta: &oodf_ffi::OwnedDelta, externals: &[&[u8]]) -> CaseResult<()> {
    let references = delta
        .set_fields
        .iter()
        .map(|v| (v.ext_value_index, v.ext_value_size, &v.value_or_ext_hash))
        .chain(
            delta
                .time_samples
                .iter()
                .map(|v| (v.ext_value_index, v.ext_value_size, &v.value_or_ext_hash)),
        );
    for (index, size, hash) in references {
        if size == 0 {
            continue;
        }
        let blob = index
            .checked_sub(1)
            .and_then(|i| usize::try_from(i).ok())
            .and_then(|i| externals.get(i))
            .ok_or_else(|| {
                CaseError::new("OODF decoder failed: external value index is outside OMPF parts")
            })?;
        if blob.len() as u64 != size || content_digest(blob).as_slice() != hash.as_slice() {
            return Err(CaseError::new(
                "OODF decoder failed: external value size or hash differs from referenced OMPF part",
            ));
        }
    }
    Ok(())
}

/// The canonical signature of a parsed delta and its external parts.
fn signature_of(delta: &oodf_ffi::OwnedDelta, externals: &[&[u8]]) -> Signature {
    let mut set_fields: Vec<SignatureField> = delta
        .set_fields
        .iter()
        .map(|f| SignatureField {
            ext_value_size: f.ext_value_size,
            key_name: f.key_name.clone(),
            section_id: f.section_id,
            set_order: f.set_order,
            value_or_ext_hash: hex(&f.value_or_ext_hash),
        })
        .collect();
    set_fields.sort_by(|a, b| {
        (a.section_id, &a.key_name, a.set_order).cmp(&(b.section_id, &b.key_name, b.set_order))
    });

    let mut time_samples: Vec<SignatureSample> = delta
        .time_samples
        .iter()
        .map(|t| SignatureSample {
            ext_value_size: t.ext_value_size,
            section_id: t.section_id,
            time: t.time,
            value_or_ext_hash: hex(&t.value_or_ext_hash),
        })
        .collect();
    time_samples.sort_by(|a, b| {
        a.section_id
            .cmp(&b.section_id)
            .then(a.time.partial_cmp(&b.time).unwrap_or(Ordering::Equal))
    });

    let mut struct_commands: Vec<(String, SignatureCommand)> = delta
        .struct_commands
        .iter()
        .map(|c| {
            let command = SignatureCommand::from(c);
            (command.canonical_json(), command)
        })
        .collect();
    struct_commands.sort_by(|a, b| a.0.cmp(&b.0));

    let mut external_value_hashes: Vec<String> =
        externals.iter().map(|b| hex(&Sha256::digest(b))).collect();
    external_value_hashes.sort();

    Signature {
        base_version: delta.base_version,
        external_value_hashes,
        is_diff: delta.is_diff,
        set_fields,
        source_format: delta.source_format.clone(),
        struct_commands: struct_commands.into_iter().map(|(_, c)| c).collect(),
        time_samples,
    }
}

/// Decodes an OMPF payload in process: `oodf_ffi::parse_delta` on part 0,
/// then every external reference (a field or sample with a non-zero
/// `ext_value_size`) must name an existing part of exactly that size whose
/// [`content_hash`] is its `valueOrExtHash`. Fails for a payload that is not
/// a complete OMPF container, an empty part 0, a delta the parser refuses, or
/// a missing or corrupt external value.
pub fn decode(payload: &[u8]) -> CaseResult<Signature> {
    let parts = unpack_ompf(payload)?;
    let (delta_bytes, externals) = parts.split_first().expect("at least one OMPF part");
    if delta_bytes.is_empty() {
        return Err(CaseError::new(
            "OODF decoder failed: the delta part is empty",
        ));
    }
    let delta = oodf_ffi::parse_delta(delta_bytes)
        .map_err(|error| CaseError::new(format!("OODF decoder failed: {error}")))?;
    validate_externals(&delta, externals)?;
    let signature = signature_of(&delta, externals);

    // The blob cross-check: every external reference's claim names a part of
    // its size by content, whatever index it gives.
    let blobs: HashMap<String, &[u8]> = externals
        .iter()
        .map(|part| (content_hash(part), *part))
        .collect();
    let references = signature
        .set_fields
        .iter()
        .map(|v| (v.ext_value_size, &v.value_or_ext_hash))
        .chain(
            signature
                .time_samples
                .iter()
                .map(|v| (v.ext_value_size, &v.value_or_ext_hash)),
        );
    for (size, claim) in references {
        if size != 0 {
            let present = blobs
                .get(claim)
                .is_some_and(|blob| blob.len() as u64 == size);
            if !present {
                return Err(CaseError::new("Missing/corrupt external value"));
            }
        }
    }
    Ok(signature)
}

/// Decodes a catch-up delivery and checks its envelope ([`check_snapshot`]).
pub fn snapshot(payload: &[u8], base_version: u64) -> CaseResult<Signature> {
    check_snapshot(decode(payload)?, base_version)
}

/// Checks a decoded catch-up delivery: `isDiff` true, `baseVersion` equal to
/// `base_version`, and no two set fields sharing a `setOrder` slot (slots
/// need only be unique and keep authoring order, not be dense).
pub fn check_snapshot(signature: Signature, base_version: u64) -> CaseResult<Signature> {
    if !signature.is_diff || signature.base_version != base_version {
        return Err(CaseError::new("Invalid catch-up envelope"));
    }
    // Absolute counters may differ across implementations; clients require
    // globally unique slots and relative authoring order, not dense numbering.
    let mut orders = HashSet::new();
    if !signature
        .set_fields
        .iter()
        .all(|field| orders.insert(field.set_order))
    {
        return Err(CaseError::new("Catch-up field ordering slots collide"));
    }
    Ok(signature)
}

/// A section of a folded [`Scene`]: (parent id, name, section type, section
/// order).
pub type SceneSection = (u64, String, u8, u32);

/// A field or sample value of a folded [`Scene`]: (`valueOrExtHash`,
/// `extValueSize`).
pub type SceneValue = (String, u64);

/// The key of a time sample in a [`Scene`]: (section id, the time's
/// `f64::to_bits`). Build it with [`sample_key`].
pub type SampleKey = (u64, u64);

/// The [`SampleKey`] of the sample at `time` in `section_id`.
pub fn sample_key(section_id: u64, time: f64) -> SampleKey {
    (section_id, time.to_bits())
}

/// A client's view of a layer after folding catch-up diffs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Scene {
    /// By section id.
    pub sections: BTreeMap<u64, SceneSection>,
    /// By (section id, key name).
    pub fields: BTreeMap<(u64, String), SceneValue>,
    /// By [`SampleKey`].
    pub samples: BTreeMap<SampleKey, SceneValue>,
}

/// Folds a catch-up diff onto `baseline` (an empty scene when `None`), as a
/// client applies it: `CommandDiffDeleteSection` removes the section and its
/// fields and samples, `CommandDiffChangeSection` upserts the section, then
/// every set field and time sample upserts its value. Any other command
/// fails: catch-up commands are disjoint upserts and retractions, and raw
/// inbound command streams, whose order matters, are not folded.
pub fn apply_snapshot(signature: &Signature, baseline: Option<&Scene>) -> CaseResult<Scene> {
    let mut scene = baseline.cloned().unwrap_or_default();
    for command in &signature.struct_commands {
        match command {
            SignatureCommand::CommandDiffDeleteSection { section_id } => {
                let sid = *section_id;
                scene.sections.remove(&sid);
                scene.fields.retain(|key, _| key.0 != sid);
                scene.samples.retain(|key, _| key.0 != sid);
            }
            SignatureCommand::CommandDiffChangeSection {
                parent_id,
                section_id,
                section_name,
                section_order,
                section_type,
            } => {
                scene.sections.insert(
                    *section_id,
                    (
                        *parent_id,
                        section_name.clone(),
                        *section_type,
                        *section_order,
                    ),
                );
            }
            other => {
                return Err(CaseError::new(format!(
                    "Unexpected catch-up command: {}",
                    other.type_name()
                )))
            }
        }
    }
    for value in &signature.set_fields {
        scene.fields.insert(
            (value.section_id, value.key_name.clone()),
            (value.value_or_ext_hash.clone(), value.ext_value_size),
        );
    }
    for value in &signature.time_samples {
        scene.samples.insert(
            sample_key(value.section_id, value.time),
            (value.value_or_ext_hash.clone(), value.ext_value_size),
        );
    }
    Ok(scene)
}

/// The relative authoring order of a signature's set fields: (section id,
/// key name) sorted by `setOrder`.
pub fn field_order(signature: &Signature) -> Vec<(u64, String)> {
    let mut fields: Vec<&SignatureField> = signature.set_fields.iter().collect();
    fields.sort_by_key(|field| field.set_order);
    fields
        .into_iter()
        .map(|field| (field.section_id, field.key_name.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use oodf_ffi::{
        CommandDeleteSection, CommandReorderChildren, OwnedDelta, SetField, StructCommand,
    };
    use tonic::codegen::Bytes;
    use tonic::Code;
    use tonic_types::ErrorDetails;

    use super::*;
    use crate::fixtures::{capture_golden, fixture, Fixture};

    // --- Protobuf, written out by hand as the Python suite's test did, so the
    // reading is checked against bytes rather than against tonic-types'
    // own encoder.

    fn varint(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    /// One length-delimited protobuf field.
    fn field(number: u64, payload: &[u8]) -> Vec<u8> {
        [
            varint(number << 3 | 2),
            varint(payload.len() as u64),
            payload.to_vec(),
        ]
        .concat()
    }

    /// A `google.rpc.Status` (code ABORTED, a message) with `details` as its
    /// `Any` entries, each a (`type_url`, value) pair.
    fn rpc_status(details: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut status = vec![0x08, 10];
        status.extend(field(2, b"refused"));
        for (type_url, value) in details {
            status.extend(field(
                3,
                &[field(1, type_url.as_bytes()), field(2, value)].concat(),
            ));
        }
        status
    }

    fn status_with_details(details: Vec<u8>) -> tonic::Status {
        tonic::Status::with_details(Code::Aborted, "refused", Bytes::from(details))
    }

    fn predecessor_not_committed() -> Vec<u8> {
        [
            field(1, b"PREDECESSOR_NOT_COMMITTED"),
            field(2, b"nvidia.omniverse.live.v1alpha"),
            field(
                3,
                &[field(1, b"last_record"), field(2, b"operation")].concat(),
            ),
            field(
                3,
                &[field(1, b"last_operation_id"), field(2, b"op-1")].concat(),
            ),
        ]
        .concat()
    }

    /// An `ErrorInfo` value that does not decode: its first field claims 5
    /// bytes and carries 1.
    const UNDECODABLE_ERROR_INFO: [u8; 3] = [0x0a, 0x05, b'a'];

    #[test]
    fn error_info_reads_the_refusal_detail_of_grpc_status_details_bin() {
        let status = status_with_details(rpc_status(&[(
            ERROR_INFO_TYPE,
            predecessor_not_committed(),
        )]));
        assert_eq!(
            error_info(&status),
            Some(ErrorInfo::new(
                "PREDECESSOR_NOT_COMMITTED",
                ERROR_DOMAIN,
                &[("last_record", "operation"), ("last_operation_id", "op-1")],
            ))
        );
        assert_eq!(carries_detail(&status, ERROR_INFO_TYPE), Ok(true));
        assert_eq!(carries_detail(&status, RETRY_INFO_TYPE), Ok(false));
    }

    #[test]
    fn error_info_without_details_is_none() {
        let status = tonic::Status::aborted("refused");
        assert_eq!(error_info(&status), None);
        assert_eq!(carries_detail(&status, ERROR_INFO_TYPE), Ok(false));
    }

    #[test]
    fn carries_detail_undecodable_details_fail_rather_than_read_as_none() {
        // Not a google.rpc.Status at all.
        let status = status_with_details(vec![0xff]);
        assert!(carries_detail(&status, ERROR_INFO_TYPE).is_err());
        assert_eq!(error_info(&status), None);
    }

    #[test]
    fn carries_detail_matching_type_url_is_true_even_when_its_value_does_not_decode() {
        // The case that `error_info` alone would read as "no ErrorInfo".
        let status = status_with_details(rpc_status(&[(
            ERROR_INFO_TYPE,
            UNDECODABLE_ERROR_INFO.to_vec(),
        )]));
        assert_eq!(carries_detail(&status, ERROR_INFO_TYPE), Ok(true));
        assert_eq!(error_info(&status), None);
    }

    #[test]
    fn carries_detail_type_urls_match_what_tonic_types_writes() {
        let retry = tonic::Status::with_error_details(
            Code::Unavailable,
            "retry",
            ErrorDetails::with_retry_info(Some(Duration::from_secs(1))),
        );
        assert_eq!(carries_detail(&retry, RETRY_INFO_TYPE), Ok(true));
        assert_eq!(carries_detail(&retry, ERROR_INFO_TYPE), Ok(false));

        let refused = tonic::Status::with_error_details(
            Code::InvalidArgument,
            "too large",
            ErrorDetails::with_error_info(
                "DELTA_TOO_LARGE",
                ERROR_DOMAIN,
                [("max_entries".to_string(), "99999".to_string())],
            ),
        );
        assert_eq!(carries_detail(&refused, ERROR_INFO_TYPE), Ok(true));
        assert_eq!(
            error_info(&refused),
            Some(ErrorInfo::new(
                "DELTA_TOO_LARGE",
                ERROR_DOMAIN,
                &[("max_entries", "99999")]
            ))
        );
    }

    // --- OMPF.

    #[test]
    fn pack_ompf_one_part_unpacks_to_that_part() {
        let delta = b"OODF-delta-bytes".to_vec();
        assert_eq!(
            unpack_ompf(&pack_ompf(&delta)).unwrap(),
            vec![delta.as_slice()]
        );
    }

    #[test]
    fn unpack_ompf_incomplete_containers_are_refused() {
        let mut bad_offset = pack_ompf(b"delta");
        bad_offset[8] = 4; // the part starts inside the offset table
        let mut zero_parts = pack_ompf(b"delta");
        zero_parts[4] = 0;
        let mut too_many_parts = pack_ompf(b"delta");
        too_many_parts[4] = 200;
        for payload in [
            b"not-ompf".to_vec(),
            b"OMPF\x01\x00\x00\x00".to_vec(),
            [b"OMPF".to_vec(), vec![0xff; 40]].concat(),
            [b"OMPX".to_vec(), pack_ompf(b"delta")[4..].to_vec()].concat(),
            bad_offset,
            zero_parts,
            too_many_parts,
        ] {
            assert!(unpack_ompf(&payload).is_err(), "{payload:?}");
        }
    }

    // --- The external-value content hash.

    #[test]
    fn content_hash_matches_the_clients_vectors() {
        // Over bytes(i % 251), from the C++ and Rust client libraries' own
        // suites: the 1 MiB block hash, not plain SHA-256 of the bytes.
        for (length, expected) in [
            (
                0,
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            ),
            (
                1,
                "1406e05881e299367766d313e26c05564ec91bf721d31726bd6e46e60689539a",
            ),
            (
                1_048_577,
                "a9c574ce937d2371daf87cdd0e75396b096d7f74a66c764e4c656bd12b2e7bf9",
            ),
        ] {
            let value: Vec<u8> = (0..length).map(|i| (i % 251) as u8).collect();
            assert_eq!(content_hash(&value), expected, "length {length}");
        }
    }

    #[test]
    fn content_hash_keys_the_external_fixtures_value() {
        let parts = unpack_ompf(fixture(Fixture::External)).unwrap();
        let hashes: Vec<String> = parts[1..].iter().map(|part| content_hash(part)).collect();
        assert_eq!(
            hashes,
            ["368a4df03f4e059ae6f4392ba381e03c261d21f064fa1caa48e72e6b14c9e84d"]
        );
    }

    // --- The FlatBuffer builders, read back as the FlatBuffers layout
    // defines it rather than through a FlatBuffers reader.

    fn u16_at(bytes: &[u8], at: usize) -> u16 {
        u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
    }

    fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    }

    fn i32_at(bytes: &[u8], at: usize) -> i32 {
        i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    }

    fn u64_at(bytes: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
    }

    /// The table at `table`'s vtable: its position and size.
    fn vtable_of(bytes: &[u8], table: usize) -> (usize, usize) {
        let vtable = usize::try_from(table as i64 - i64::from(i32_at(bytes, table))).unwrap();
        (vtable, usize::from(u16_at(bytes, vtable)))
    }

    /// The file identifier of an OODF delta and, by vtable byte position
    /// from 8 on, each vector field's (length word position, length): the
    /// root offset at 0, the root table's signed offset back to its vtable,
    /// each vector field's unsigned offset to its length word.
    fn delta_vectors(delta: &[u8]) -> (&[u8], BTreeMap<usize, (usize, u32)>) {
        let root = u32_at(delta, 0) as usize;
        let (vtable, vtable_size) = vtable_of(delta, root);
        let mut vectors = BTreeMap::new();
        for slot in (8..vtable_size).step_by(2) {
            let field = usize::from(u16_at(delta, vtable + slot));
            if field != 0 {
                let at = root + field;
                let start = at + u32_at(delta, at) as usize;
                vectors.insert(slot, (start, u32_at(delta, start)));
            }
        }
        (&delta[4..8], vectors)
    }

    fn lengths(vectors: &BTreeMap<usize, (usize, u32)>) -> BTreeMap<usize, u32> {
        vectors.iter().map(|(slot, (_, n))| (*slot, *n)).collect()
    }

    #[test]
    fn delta_of_entries_writes_its_counts_where_a_reader_finds_them() {
        // structCommands_type (vtable byte 8), structCommands (10), setFields (12).
        let delta = delta_of_entries(60_000, 59_000);
        let (identifier, vectors) = delta_vectors(&delta);
        assert_eq!(identifier, b"OODF");
        assert_eq!(
            lengths(&vectors),
            BTreeMap::from([(8, 60_000), (10, 60_000), (12, 59_000)])
        );
        assert_eq!(
            unpack_ompf(&pack_ompf(&delta)).unwrap(),
            vec![delta.as_slice()]
        );
    }

    #[test]
    fn delta_of_children_list_writes_one_aligned_list_of_the_ids() {
        let delta = delta_of_children_list(7);
        let (identifier, vectors) = delta_vectors(&delta);
        assert_eq!(identifier, b"OODF");
        assert_eq!(lengths(&vectors), BTreeMap::from([(8, 1), (10, 1)]));
        // The list's u64 elements follow an 8-aligned length word + 4: the
        // last 7 × 8 bytes are the ids 1..=7, little-endian.
        let ids: Vec<u8> = (1..=7u64).flat_map(u64::to_le_bytes).collect();
        assert_eq!(&delta[delta.len() - 56..], ids.as_slice());
        assert_eq!((delta.len() - 56) % 8, 0);
        assert_eq!(
            unpack_ompf(&pack_ompf(&delta)).unwrap(),
            vec![delta.as_slice()]
        );
    }

    #[test]
    fn delta_deleting_section_writes_one_aligned_delete_where_a_reader_finds_it() {
        let delta = delta_deleting_section(3, 0x0102_0304_0506_0708);
        let (identifier, vectors) = delta_vectors(&delta);
        assert_eq!(identifier, b"OODF");
        // One struct-command type (vtable byte 8) and one struct command (10).
        assert_eq!(lengths(&vectors), BTreeMap::from([(8, 1), (10, 1)]));
        let (types, _) = vectors[&8];
        assert_eq!(delta[types + 4], 2, "the one command is a DeleteSection");

        let (commands, _) = vectors[&10];
        let at = commands + 4;
        let table = at + u32_at(&delta, at) as usize;
        let (vtable, vtable_size) = vtable_of(&delta, table);
        let slots: Vec<usize> = (4..vtable_size)
            .step_by(2)
            .map(|byte| usize::from(u16_at(&delta, vtable + byte)))
            .collect();
        // parentId, sectionId: each a u64 the verifier finds 8-aligned.
        assert_eq!(slots.len(), 2);
        assert!(
            slots.iter().all(|slot| (table + slot).is_multiple_of(8)),
            "{slots:?}"
        );
        assert_eq!(
            slots
                .iter()
                .map(|slot| u64_at(&delta, table + slot))
                .collect::<Vec<_>>(),
            [3, 0x0102_0304_0506_0708]
        );
        assert_eq!(
            unpack_ompf(&pack_ompf(&delta)).unwrap(),
            vec![delta.as_slice()]
        );
    }

    #[test]
    fn parse_delta_accepts_the_delete_section_delta() {
        let parsed = oodf_ffi::parse_delta(&delta_deleting_section(3, 5))
            .expect("the verifier accepts the hand-written delta");
        assert_eq!(
            parsed.struct_commands,
            [StructCommand::DeleteSection(CommandDeleteSection {
                parent_id: 3,
                section_id: 5,
            })]
        );
        assert!(parsed.set_fields.is_empty() && parsed.time_samples.is_empty());
    }

    #[test]
    fn parse_delta_accepts_the_children_list_delta() {
        let parsed = oodf_ffi::parse_delta(&delta_of_children_list(7))
            .expect("the verifier accepts the hand-written delta");
        assert_eq!(
            parsed.struct_commands,
            [StructCommand::ReorderChildren(CommandReorderChildren {
                section_id: 0,
                children_list_id: 0,
                children_list: (1..=7).collect(),
            })]
        );
    }

    // --- In-process decode.

    /// An OMPF container of `parts` (part 0 the delta).
    fn ompf_of(parts: &[&[u8]]) -> Vec<u8> {
        let mut out = b"OMPF".to_vec();
        out.extend_from_slice(&(parts.len() as u32).to_le_bytes());
        let mut offset = (8 + 8 * parts.len()) as u64;
        for part in parts {
            out.extend_from_slice(&offset.to_le_bytes());
            offset += part.len() as u64;
        }
        for part in parts {
            out.extend_from_slice(part);
        }
        out
    }

    fn decode_error(payload: &[u8]) -> String {
        match decode(payload) {
            Err(error) => error.message().to_string(),
            Ok(signature) => panic!("decoded as {signature:?}"),
        }
    }

    #[test]
    fn decode_initial_fixture_gives_its_canonical_signature() {
        let section =
            |section_id, parent_id, name: &str| SignatureCommand::CommandDiffChangeSection {
                parent_id,
                section_id,
                section_name: name.to_string(),
                section_order: 0,
                section_type: 0,
            };
        assert_eq!(
            decode(fixture(Fixture::Initial)).unwrap(),
            Signature {
                base_version: 0,
                external_value_hashes: Vec::new(),
                is_diff: true,
                set_fields: vec![SignatureField {
                    ext_value_size: 0,
                    key_name: "value".into(),
                    section_id: 5,
                    set_order: 0,
                    value_or_ext_hash: hex(b"initial"),
                }],
                source_format: "usda".into(),
                struct_commands: vec![section(1, 0, ""), section(5, 1, "probe")],
                time_samples: Vec::new(),
            }
        );
    }

    #[test]
    fn decode_external_fixture_validates_and_fingerprints_its_value() {
        let parts = unpack_ompf(fixture(Fixture::External)).unwrap();
        let signature = decode(fixture(Fixture::External)).unwrap();
        assert_eq!(signature.set_fields.len(), 1);
        assert_eq!(
            signature.set_fields[0].ext_value_size,
            parts[1].len() as u64
        );
        assert_eq!(
            signature.set_fields[0].value_or_ext_hash,
            content_hash(parts[1])
        );
        assert_eq!(
            signature.external_value_hashes,
            [hex(&Sha256::digest(parts[1]))]
        );
    }

    #[test]
    fn decode_corrupt_external_value_fails() {
        let parts = unpack_ompf(fixture(Fixture::External)).unwrap();
        let mut corrupt = parts[1].to_vec();
        *corrupt.last_mut().unwrap() ^= 0x01;
        let message = decode_error(&ompf_of(&[parts[0], corrupt.as_slice()]));
        assert!(message.contains("size or hash differs"), "{message}");
    }

    #[test]
    fn decode_truncated_external_value_fails() {
        let parts = unpack_ompf(fixture(Fixture::External)).unwrap();
        let short = &parts[1][..parts[1].len() - 1];
        let message = decode_error(&ompf_of(&[parts[0], short]));
        assert!(message.contains("size or hash differs"), "{message}");
    }

    #[test]
    fn decode_missing_external_part_fails() {
        let parts = unpack_ompf(fixture(Fixture::External)).unwrap();
        let message = decode_error(&ompf_of(&[parts[0]]));
        assert!(message.contains("outside OMPF parts"), "{message}");
    }

    /// An OMPF of one external `value` referenced by a set field claiming
    /// `claim` as its hash.
    fn external_reference(value: &[u8], claim: Vec<u8>) -> Vec<u8> {
        let delta = oodf_ffi::build_delta(&OwnedDelta {
            is_diff: true,
            source_format: "usda".into(),
            set_fields: vec![SetField {
                section_id: 5,
                key_name: "value".into(),
                value_or_ext_hash: claim,
                ext_value_size: value.len() as u64,
                ext_value_index: 1,
                set_order: 0,
            }],
            ..Default::default()
        })
        .unwrap();
        ompf_of(&[&delta, value])
    }

    #[test]
    fn decode_external_reference_by_plain_sha256_fails() {
        let value = vec![0x43; 8192];
        // The positive control: the same reference by its content hash.
        assert!(decode(&external_reference(&value, content_digest(&value).to_vec())).is_ok());
        let message = decode_error(&external_reference(&value, Sha256::digest(&value).to_vec()));
        assert!(message.contains("size or hash differs"), "{message}");
    }

    #[test]
    fn decode_refuses_what_is_not_an_oodf_delta() {
        assert!(decode(b"not-ompf").is_err());
        let message = decode_error(&ompf_of(&[b""]));
        assert!(message.contains("delta part is empty"), "{message}");
        let message = decode_error(&ompf_of(&[b"garbage-that-is-no-flatbuffer"]));
        assert!(message.contains("OODF decoder failed"), "{message}");
    }

    #[test]
    fn decode_delete_section_delta_gives_its_command() {
        let signature = decode(&pack_ompf(&delta_deleting_section(3, 5))).unwrap();
        assert_eq!(
            signature.struct_commands,
            [SignatureCommand::CommandDeleteSection {
                parent_id: 3,
                section_id: 5
            }]
        );
    }

    // --- The catch-up checks and the client-side fold.

    /// A catch-up diff of section 5 with one field and one sample, as the
    /// Python fold control's `full`.
    fn full_diff() -> Signature {
        Signature {
            base_version: 0,
            external_value_hashes: Vec::new(),
            is_diff: true,
            set_fields: vec![SignatureField {
                ext_value_size: 0,
                key_name: "value".into(),
                section_id: 5,
                set_order: 0,
                value_or_ext_hash: "42".into(),
            }],
            source_format: "usda".into(),
            struct_commands: vec![SignatureCommand::CommandDiffChangeSection {
                parent_id: 1,
                section_id: 5,
                section_name: "probe".into(),
                section_order: 0,
                section_type: 0,
            }],
            time_samples: vec![SignatureSample {
                ext_value_size: 0,
                section_id: 5,
                time: 1.5,
                value_or_ext_hash: "43".into(),
            }],
        }
    }

    #[test]
    fn apply_snapshot_full_diff_gives_the_section_field_and_sample() {
        let scene = apply_snapshot(&full_diff(), None).unwrap();
        assert_eq!(
            scene.sections,
            BTreeMap::from([(5, (1, "probe".to_string(), 0, 0))])
        );
        assert_eq!(
            scene.fields,
            BTreeMap::from([((5, "value".to_string()), ("42".to_string(), 0))])
        );
        assert_eq!(
            scene.samples,
            BTreeMap::from([(sample_key(5, 1.5), ("43".to_string(), 0))])
        );
    }

    #[test]
    fn apply_snapshot_lost_content_changes_the_folded_scene() {
        let expected = apply_snapshot(&full_diff(), None).unwrap();
        let mut broken: Vec<(&str, Signature)> = Vec::new();
        let mut lost = full_diff();
        lost.struct_commands.clear();
        broken.push(("no struct commands", lost));
        let mut lost = full_diff();
        lost.set_fields.clear();
        broken.push(("no set fields", lost));
        let mut lost = full_diff();
        lost.time_samples.clear();
        broken.push(("no time samples", lost));
        for corrupt in ["parentId", "sectionOrder"] {
            let mut changed = full_diff();
            if let SignatureCommand::CommandDiffChangeSection {
                parent_id,
                section_order,
                ..
            } = &mut changed.struct_commands[0]
            {
                match corrupt {
                    "parentId" => *parent_id = 99,
                    _ => *section_order = 99,
                }
            }
            broken.push((corrupt, changed));
        }
        let mut changed = full_diff();
        changed.set_fields[0].value_or_ext_hash = "corrupt".into();
        broken.push(("valueOrExtHash", changed));
        let mut changed = full_diff();
        changed.time_samples[0].time = 2.5;
        broken.push(("time", changed));

        for (what, signature) in broken {
            assert_ne!(
                apply_snapshot(&signature, None).unwrap(),
                expected,
                "{what}"
            );
        }
    }

    #[test]
    fn apply_snapshot_diff_delete_section_removes_it_with_its_fields_and_samples() {
        let baseline = apply_snapshot(&full_diff(), None).unwrap();
        let deleted = Signature {
            set_fields: Vec::new(),
            struct_commands: vec![SignatureCommand::CommandDiffDeleteSection { section_id: 5 }],
            time_samples: Vec::new(),
            ..full_diff()
        };
        assert_eq!(
            apply_snapshot(&deleted, Some(&baseline)).unwrap(),
            Scene::default()
        );
    }

    #[test]
    fn apply_snapshot_raw_structural_command_is_refused() {
        let raw = Signature {
            struct_commands: vec![SignatureCommand::CommandDeleteSection {
                parent_id: 1,
                section_id: 5,
            }],
            ..full_diff()
        };
        let error = apply_snapshot(&raw, None).unwrap_err();
        assert!(
            error.message().contains("CommandDeleteSection"),
            "{}",
            error.message()
        );
    }

    #[test]
    fn check_snapshot_golden_signature_passes_its_envelope_and_order_checks() {
        let golden = capture_golden();
        let base_version = golden.base_version;
        assert!(check_snapshot(golden, base_version).is_ok());
    }

    #[test]
    fn check_snapshot_envelope_and_order_negative_controls_fail() {
        let golden = capture_golden();
        assert!(
            golden.set_fields.len() >= 2,
            "the golden signature has fields to collide"
        );
        let base_version = golden.base_version;

        let mut not_diff = golden.clone();
        not_diff.is_diff = false;
        let mut other_base = golden.clone();
        other_base.base_version = base_version + 42;
        let mut colliding = golden.clone();
        for field in &mut colliding.set_fields {
            field.set_order = 0;
        }
        for (what, signature, reason) in [
            ("isDiff", not_diff, "Invalid catch-up envelope"),
            ("baseVersion", other_base, "Invalid catch-up envelope"),
            ("duplicateOrder", colliding, "slots collide"),
        ] {
            let error = check_snapshot(signature, base_version).unwrap_err();
            assert!(
                error.message().contains(reason),
                "{what}: {}",
                error.message()
            );
        }
    }

    #[test]
    fn field_order_detects_reordered_fields() {
        let original = capture_golden();
        let mut swapped = original.clone();
        let first = swapped.set_fields[0].set_order;
        swapped.set_fields[0].set_order = swapped.set_fields[1].set_order;
        swapped.set_fields[1].set_order = first;
        assert_ne!(field_order(&original), field_order(&swapped));
    }

    #[test]
    fn field_order_follows_set_order_not_position() {
        let mut signature = full_diff();
        signature.set_fields = vec![
            SignatureField {
                key_name: "b".into(),
                set_order: 9,
                ..full_diff().set_fields[0].clone()
            },
            SignatureField {
                key_name: "a".into(),
                set_order: 3,
                ..full_diff().set_fields[0].clone()
            },
        ];
        assert_eq!(
            field_order(&signature),
            [(5, "a".to_string()), (5, "b".to_string())]
        );
    }
}
