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

// Shared name-validation logic for sessions and layers, plus delta-payload
// bounds validation, enforced by the upload handlers before the processing
// charge and again at the top of `apply_delta` (see `domain::limits`).
//
// Both session names (PRD §3.1.1) and layer names (SDD §23.1) follow identical
// rules. This module provides a single implementation to avoid duplication.

use oodf_ffi::{OwnedDelta, StructCommand};

use crate::domain::delta::{own_element_count, DeltaError};
use crate::domain::limits::{
    MAX_CHILDREN_LIST_LEN, MAX_DELTA_ENTRIES, MAX_DELTA_SET_ORDER, MAX_NAME_BYTES,
};
use crate::error::ServiceError;

/// Enforce hard caps on a parsed delta before `apply_delta` starts mutating
/// `ObjectState`. See [`crate::domain::limits`] for the constants and rationale.
///
/// Checks the delta's entries, struct commands + set fields + time samples
/// summed ([`own_element_count`]), `<= MAX_DELTA_ENTRIES`, else
/// [`DeltaError::TooManyEntries`] with the three counts; then every per-item
/// rule of [`validate_delta_content`].
///
/// Every apply runs this except a checkpoint base decode, which runs
/// [`validate_delta_content`] alone (see `ApplyMode::RestoreBase`).
pub fn validate_delta_bounds(delta: &OwnedDelta) -> Result<(), DeltaError> {
    if own_element_count(delta) > MAX_DELTA_ENTRIES {
        let count = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
        return Err(DeltaError::TooManyEntries {
            struct_commands: count(delta.struct_commands.len()),
            set_fields: count(delta.set_fields.len()),
            time_samples: count(delta.time_samples.len()),
            max_entries: MAX_DELTA_ENTRIES,
        });
    }
    validate_delta_content(delta)
}

/// Every per-item rule [`validate_delta_bounds`] enforces, without its
/// per-delta entry-count cap.
///
/// Checks:
/// - For every struct command carrying a name (`CreateSection::section_name`,
///   `MoveSection::new_name`, `DiffChangeSection::section_name`):
///   byte-length `<= MAX_NAME_BYTES`, no control characters, and no
///   self-parenting.
/// - For every `SetField`: `key_name.len() <= MAX_NAME_BYTES` (bytes), no
///   control characters, and `set_order <= MAX_DELTA_SET_ORDER`.
/// - For every `CommandReorderChildren::children_list`:
///   `children_list.len() <= MAX_CHILDREN_LIST_LEN`, else
///   [`DeltaError::ChildrenListTooLong`].
/// - For every `SetField` / `TimeSample` with `ext_value_size > 0`:
///   `value_or_ext_hash` is exactly 32 bytes, else
///   [`DeltaError::InvalidExternalHashLength`].
/// - Every `TimeSample` time is finite, else
///   [`DeltaError::TimeSampleNonFiniteTime`].
/// - Every `TimeSample` time is not `-0.0`, else
///   [`DeltaError::TimeSampleNegativeZeroTime`] (a backstop:
///   `oodf_ffi::parse_delta` canonicalizes `-0.0`, so only a delta that
///   bypassed the parser can carry it).
///
/// Byte length is checked, not `chars().count()` — downstream storage treats
/// names as bytes, and a multi-MB name of single-code-point characters would
/// pass a character-count check while still blowing storage budgets.
pub fn validate_delta_content(delta: &OwnedDelta) -> Result<(), DeltaError> {
    for cmd in &delta.struct_commands {
        match cmd {
            StructCommand::CreateSection(c) => {
                check_name_len("CreateSection.section_name", &c.section_name)?;
                check_name_chars("CreateSection.section_name", &c.section_name)?;
                // b.39x medium: reject self-parenting before the apply
                // engine has a chance to wire a section to itself as its
                // own parent. Self-parented sections produce cycles in
                // `state.children` that break tree walks (cascade-delete,
                // diff reconstruction) with stack overflow or infinite
                // loop. parent_id == 0 is the root marker and is fine.
                if c.parent_id != 0 && c.parent_id == c.section_id {
                    return Err(DeltaError::SelfParenting {
                        command: "CreateSection",
                        section_id: c.section_id,
                    });
                }
            }
            StructCommand::MoveSection(c) => {
                check_name_len("MoveSection.new_name", &c.new_name)?;
                check_name_chars("MoveSection.new_name", &c.new_name)?;
                // b.39x medium: same self-parenting check for MoveSection.
                // `new_parent_id == 0` is the "move to root" case and is
                // valid; reject only when the new parent is the section
                // itself.
                if c.new_parent_id != 0 && c.new_parent_id == c.section_id {
                    return Err(DeltaError::SelfParenting {
                        command: "MoveSection",
                        section_id: c.section_id,
                    });
                }
                // b.39x PM follow-up: ALSO reject `old_parent_id ==
                // section_id`. A section cannot be its own parent — past
                // or present — so a command claiming it is corrupt.
                // `apply_move_section` takes its bookkeeping from the
                // section's *stored* parent and uses the carried
                // `old_parent_id` only to remove the section from that id's
                // children list as well; with `old_parent_id == section_id`
                // that list is the section's own children list, which it
                // would treat as a parent's (and mark for a rewrite).
                // Rejecting the claim keeps a self-referencing parent id out
                // of the engine altogether, as the `new_parent_id` branch
                // above does. `old_parent_id == 0` is the "moved from root"
                // case and is valid.
                if c.old_parent_id != 0 && c.old_parent_id == c.section_id {
                    return Err(DeltaError::SelfParenting {
                        command: "MoveSection",
                        section_id: c.section_id,
                    });
                }
            }
            StructCommand::DiffChangeSection(c) => {
                check_name_len("DiffChangeSection.section_name", &c.section_name)?;
                check_name_chars("DiffChangeSection.section_name", &c.section_name)?;
                // b.39x medium: same self-parenting check for
                // DiffChangeSection (the diff-reconstruction analog of
                // CreateSection / MoveSection).
                if c.parent_id != 0 && c.parent_id == c.section_id {
                    return Err(DeltaError::SelfParenting {
                        command: "DiffChangeSection",
                        section_id: c.section_id,
                    });
                }
            }
            StructCommand::ReorderChildren(c) => {
                if c.children_list.len() > MAX_CHILDREN_LIST_LEN {
                    return Err(DeltaError::ChildrenListTooLong {
                        limit: MAX_CHILDREN_LIST_LEN as u64,
                        actual: c.children_list.len() as u64,
                    });
                }
            }
            StructCommand::DeleteSection(_) | StructCommand::DiffDeleteSection(_) => {}
        }
    }

    for sf in &delta.set_fields {
        check_name_len("SetField.key_name", &sf.key_name)?;
        check_name_chars("SetField.key_name", &sf.key_name)?;
        // b.qw6 item 2: cap `delta.set_order` at the Lua f64 integer-
        // precision ceiling so a single malicious delta cannot blow
        // past the 2^53 ceiling in one shot. The cumulative case (a
        // long-lived object whose `next_set_order` cursor crosses the
        // ceiling over many legitimate writes) is caught by the
        // runtime ceiling check in `apply_set_field`.
        if sf.set_order > MAX_DELTA_SET_ORDER {
            return Err(DeltaError::DeltaBoundsExceeded {
                field: "SetField.set_order",
                limit: MAX_DELTA_SET_ORDER,
                actual: sf.set_order,
            });
        }
        check_external_hash_len(sf.ext_value_size, &sf.value_or_ext_hash)?;
    }

    for ts in &delta.time_samples {
        check_external_hash_len(ts.ext_value_size, &ts.value_or_ext_hash)?;
    }

    // b.39x H-dom: reject non-finite time-sample keys. `OrderedFloat<f64>`
    // wraps the raw `f64` for use as a HashMap key, but NaN values violate
    // the reflexivity contract `OrderedFloat` papers over via total-ordering
    // — round-tripping a NaN key through Valkey (which stringifies the f64)
    // produces the literal `"nan"`, which Lua scripts then compare as a
    // string. Infinities have undefined storage semantics too. Reject every
    // non-finite value at validation time so the apply engine and Lua fast
    // path both see only finite f64s.
    //
    // Then refuse a `-0.0` time. `-0.0` and `0.0` are one USD time code but
    // two stored keys (the row key spells the time's bits), so a `-0.0` write
    // would leave a second row for one time. `oodf_ffi::parse_delta`
    // canonicalizes `-0.0` to `+0.0`, so a parsed delta never carries it;
    // this fails closed on a delta that bypassed the parser.
    for (idx, ts) in delta.time_samples.iter().enumerate() {
        if !ts.time.is_finite() {
            return Err(DeltaError::TimeSampleNonFiniteTime {
                index: idx,
                time: ts.time,
            });
        }
        if ts.time == 0.0 && ts.time.is_sign_negative() {
            return Err(DeltaError::TimeSampleNegativeZeroTime { index: idx });
        }
    }

    Ok(())
}

/// Byte length of an external value's content hash (SHA-256, SDD §11.1).
const EXTERNAL_HASH_LEN: usize = 32;

/// Reject an external value (`ext_value_size > 0`) whose `value_or_ext_hash`
/// is not exactly [`EXTERNAL_HASH_LEN`] bytes. An inline value
/// (`ext_value_size == 0`) carries its payload in the same bytes and is never
/// length-checked here.
///
/// Refused, never padded or truncated: a padded hash would store a value that
/// complete-mode `apply_delta` rejects, so neither stream replay nor checkpoint
/// compose could reproduce the stored state. `LayerStore::update_layer` runs
/// this validator ahead of both backends, and the Lua fast path's encoder
/// writes the hash verbatim on the strength of it.
fn check_external_hash_len(
    ext_value_size: u64,
    value_or_ext_hash: &[u8],
) -> Result<(), DeltaError> {
    if ext_value_size > 0 && value_or_ext_hash.len() != EXTERNAL_HASH_LEN {
        return Err(DeltaError::InvalidExternalHashLength {
            got: value_or_ext_hash.len(),
            expected: EXTERNAL_HASH_LEN,
        });
    }
    Ok(())
}

fn check_name_len(field: &'static str, name: &str) -> Result<(), DeltaError> {
    if name.len() > MAX_NAME_BYTES {
        return Err(DeltaError::DeltaBoundsExceeded {
            field,
            limit: MAX_NAME_BYTES as u64,
            actual: name.len() as u64,
        });
    }
    Ok(())
}

/// Reject control characters (NUL + ASCII 0x01..=0x1F) in delta name fields
/// (b.39x medium). Sister rule to [`check_name_len`]; the two run together
/// on every name carried by a delta primitive (`section_name`, `new_name`,
/// `key_name`).
///
/// Threat model — names flow into Valkey hash keys, structured tracing
/// spans, and per-commit notification payloads. A name containing NUL
/// corrupts the Valkey key encoding on some clients; a name containing
/// `\x1b` (ESC, the ANSI introducer) echoed into operator logs can
/// overwrite terminal state. This check matches the `validate_client_id` /
/// `validate_base_path` policy so a single rule applies across every
/// untrusted-string ingress.
fn check_name_chars(field: &'static str, name: &str) -> Result<(), DeltaError> {
    if name.bytes().any(|b| b == 0 || (0x01..=0x1F).contains(&b)) {
        return Err(DeltaError::NameContainsControlChar { field });
    }
    Ok(())
}

/// Max byte length of a session or layer name. Byte-aligned rather than
/// character-aligned because downstream storage (Valkey keys, hash fields)
/// treats names as bytes. A 255-codepoint name of 4-byte characters would
/// pass a char-count check but exceed every byte-aligned budget downstream.
const MAX_SESSION_OR_LAYER_NAME_BYTES: usize = 255;

/// Validate a name (session or layer) per PRD §3.1.1 and SDD §23.1.
///
/// Rules:
/// - Non-blank (not empty or whitespace-only)
/// - Valid UTF-8 (guaranteed by Rust `&str`)
/// - Max 255 bytes (byte length, not character count — downstream storage
///   treats names as bytes)
/// - No control characters (ASCII 0-31)
/// - No path traversal (`..`)
/// - No UNC path prefixes (`\\` or `//`)
/// - No Windows absolute path patterns (e.g. `C:`)
///
/// `kind` is used in error messages (e.g. "session" or "layer").
fn validate_name(name: &str, kind: &str) -> Result<(), ServiceError> {
    if name.trim().is_empty() {
        return Err(ServiceError::InvalidArgument(format!(
            "{kind} name must not be blank"
        )));
    }

    if name.len() > MAX_SESSION_OR_LAYER_NAME_BYTES {
        return Err(ServiceError::InvalidArgument(format!(
            "{kind} name exceeds {MAX_SESSION_OR_LAYER_NAME_BYTES} byte limit"
        )));
    }

    if name.chars().any(|c| c.is_control()) {
        return Err(ServiceError::InvalidArgument(format!(
            "{kind} name contains control characters"
        )));
    }

    if name.contains("..") {
        return Err(ServiceError::InvalidArgument(format!(
            "{kind} name contains path traversal sequence '..'"
        )));
    }

    if name.starts_with("\\\\") || name.starts_with("//") {
        return Err(ServiceError::InvalidArgument(format!(
            "{kind} name starts with UNC path prefix"
        )));
    }

    if name.len() >= 2 && name.as_bytes()[0].is_ascii_alphabetic() && name.as_bytes()[1] == b':' {
        return Err(ServiceError::InvalidArgument(format!(
            "{kind} name contains Windows absolute path pattern"
        )));
    }

    Ok(())
}

/// Validate a session name per PRD §3.1.1 and SDD §23.1.
pub fn validate_session_name(name: &str) -> Result<(), ServiceError> {
    validate_name(name, "session")
}

/// Validate a layer name per PRD §3.1.1 and SDD §23.1.
pub fn validate_layer_name(name: &str) -> Result<(), ServiceError> {
    validate_name(name, "layer")
}

/// Maximum byte length of a session/layer `base_path` (b.91j H-grpc-5). The
/// limit is byte-aligned (not character-aligned) because the value is
/// hex-encoded into Valkey keys (`le:path:{hex(base_path)}`) and logged
/// verbatim — a pathological multi-MB path would blow both budgets. 4096 B
/// matches the `MAX_NAME_BYTES` delta-validation cap so a single policy
/// applies across session/layer metadata.
pub const MAX_BASE_PATH_BYTES: usize = 4096;

/// Maximum byte length of a `client_id` (b.jr8 H-grpc-6). 256 bytes is
/// loose enough for every sane UUID / opaque token scheme but tight enough
/// that a malicious subscriber cannot stuff unbounded strings into tracing
/// output or memory. The cap is byte-aligned (not char-aligned) to match
/// every other string-length cap in this module.
pub const MAX_CLIENT_ID_BYTES: usize = 256;

/// Validate a `client_id` carried by `SubscribeReadLayer` / `UpdateLayer`
/// (b.jr8 H-grpc-6).
///
/// Proto semantic: an empty `client_id` disables self-echo suppression
/// (SDD §12.5). We therefore **accept empty** — rejecting would break
/// callers that deliberately opt out of the echo filter.
///
/// Rules:
/// - Byte length `<= MAX_CLIENT_ID_BYTES` (256). Byte-aligned, not
///   character-aligned, because the value is echoed into the layer
///   notification payload, into tracing output, and compared byte-for-byte
///   in `is_self_echo`. A multi-MB client_id would poison all three.
/// - No control characters (ASCII 0x00..=0x1F including NUL). Control
///   chars in `client_id` pollute structured tracing output and, via
///   ANSI-escape injection on `\x1b`, could overwrite operator terminal
///   state. NUL is caught here too; no separate null-byte check is needed.
///
/// Errors map to `ServiceError::InvalidArgument` → gRPC `INVALID_ARGUMENT`.
pub fn validate_client_id(client_id: &str) -> Result<(), ServiceError> {
    if client_id.is_empty() {
        return Ok(());
    }

    if client_id.len() > MAX_CLIENT_ID_BYTES {
        return Err(ServiceError::InvalidArgument(format!(
            "client_id exceeds {MAX_CLIENT_ID_BYTES} byte limit"
        )));
    }

    if client_id
        .bytes()
        .any(|b| b == 0 || (0x01..=0x1F).contains(&b))
    {
        return Err(ServiceError::InvalidArgument(
            "client_id must not contain control characters".to_owned(),
        ));
    }

    // b.k1r medium: reject '|' because it is the delimiter in the
    // notification stream payload
    // (`update|{oid}|{seq}|{client_id}|{client_ts}|{server_ts}`). A
    // client_id containing `|` otherwise confuses `LayerNotification::parse`'s
    // `splitn(6, '|')` and produces spurious `WrongFieldCount` errors or,
    // worse, mis-parsed client_ts / server_ts values because the split
    // consumes part of the client_id. Rejecting at ingress is strictly
    // cheaper than escaping per-XADD on the write path and keeps the
    // wire format grammar-free. Matches the b.jr8 precedent of rejecting
    // control characters here rather than escaping downstream.
    if client_id.contains('|') {
        return Err(ServiceError::InvalidArgument(
            "client_id must not contain '|' — it is reserved as the notification payload delimiter"
                .to_owned(),
        ));
    }

    Ok(())
}

/// Maximum byte length of an `operation_id` carried by an `UpdateLayer` /
/// `CreateLayer` header. 64 bytes holds the recommended value — a random
/// UUIDv4 in its canonical 36-character text — with room to spare.
pub const MAX_OPERATION_ID_BYTES: usize = 64;

/// Validate the client-supplied `operation_id` of an `UpdateLayer` /
/// `CreateLayer` header (SDD §7.5.3, §23.1).
///
/// Empty means absent and is accepted: the call then behaves exactly as a call
/// from a client that predates the field. Otherwise the id is 1 to
/// [`MAX_OPERATION_ID_BYTES`] bytes, each one of the RFC 3986 unreserved
/// characters `A–Z a–z 0–9 - . _ ~`, and it is compared byte for byte.
///
/// The charset is what lets the id ride the two operation records with no
/// escaping: `:` delimits the `le:{oid}:obj:wtok` record and the
/// `le:{sid}:createop:{operation_id}` value, `{` / `}` would change the
/// latter key's Cluster hash tag, and the id is logged verbatim at INFO.
///
/// Errors map to `ServiceError::InvalidArgument` → gRPC `INVALID_ARGUMENT`.
pub fn validate_operation_id(operation_id: &str) -> Result<(), ServiceError> {
    if operation_id.is_empty() {
        return Ok(());
    }

    if operation_id.len() > MAX_OPERATION_ID_BYTES {
        return Err(ServiceError::InvalidArgument(format!(
            "operation_id exceeds {MAX_OPERATION_ID_BYTES} byte limit"
        )));
    }

    if !operation_id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~'))
    {
        return Err(ServiceError::InvalidArgument(
            "operation_id may contain only ASCII letters, digits, '-', '.', '_' and '~'".to_owned(),
        ));
    }

    Ok(())
}

/// Maximum byte length of a `session_id` (b.39x H). 256 bytes mirrors the
/// `client_id` cap and is generous for every sane UUID / opaque-token
/// scheme. Byte-aligned because the value is hex-injected into Valkey hash
/// tags (`le:{session_id}:sess`, `le:presence:events:{sid}`, …) and logged
/// verbatim through `tracing::info!(session_id = %session_id, …)`.
pub const MAX_SESSION_ID_BYTES: usize = 256;

/// Validate a `session_id` carried by every gRPC handler that takes one off
/// the wire (b.39x H). This validator owns the empty-string check — the
/// per-callsite `is_empty()` guards in `grpc/{layer,session,subscribe,presence}.rs`
/// were removed and folded into the first branch below — and piles on
/// length + control-character rules so the value is safe to:
///
/// - Embed in Valkey keys (`le:{session_id}:sess`, `le:path:...:names`,
///   `le:{object_id}:obj:*` is keyed by `object_id` but is reached via a
///   `(session_id, layer_name)` lookup whose key includes `session_id`).
/// - Publish over Valkey pub/sub channels (`le:presence:events:{sid}`,
///   `le:session:deleted`).
/// - Echo into tracing spans / structured fields. A multi-MB session_id
///   would blow log-aggregator budgets; an ANSI-escape (`\x1b`) could
///   overwrite operator terminal state.
///
/// Rules — identical to `validate_client_id` minus the `|` delimiter rule
/// (intentionally omitted because `session_id` does not flow into the
/// notification wire payload):
/// - Non-empty (this validator is the sole owner of the empty-string check).
/// - Byte length `<= MAX_SESSION_ID_BYTES` (256). Byte-aligned, not
///   character-aligned, because every downstream consumer treats the value
///   as bytes.
/// - No control characters (ASCII 0x00..=0x1F, including NUL).
///
/// Errors map to `ServiceError::InvalidArgument` → gRPC `INVALID_ARGUMENT`.
pub fn validate_session_id(session_id: &str) -> Result<(), ServiceError> {
    if session_id.is_empty() {
        return Err(ServiceError::InvalidArgument(
            "session_id must not be empty".to_owned(),
        ));
    }

    if session_id.len() > MAX_SESSION_ID_BYTES {
        return Err(ServiceError::InvalidArgument(format!(
            "session_id exceeds {MAX_SESSION_ID_BYTES} byte limit"
        )));
    }

    if session_id
        .bytes()
        .any(|b| b == 0 || (0x01..=0x1F).contains(&b))
    {
        return Err(ServiceError::InvalidArgument(
            "session_id must not contain control characters".to_owned(),
        ));
    }

    Ok(())
}

/// Maximum byte length of a `user_application` carried by `JoinSession`
/// (b.39x H). 256 bytes matches `MAX_CLIENT_ID_BYTES` — `user_application`
/// is the same shape of caller-asserted identifier and follows the same
/// threat model (it is HSET into `presence_user`, broadcast on
/// `PeerJoined`, and replayed on every membership replay so any other peer
/// would observe it).
pub const MAX_USER_APPLICATION_BYTES: usize = 256;

/// Validate the `user_application` field on a `JoinSessionHeader`
/// (b.39x H). Empty strings are accepted — `user_application` is an
/// optional descriptive string ("Omniverse Composer 2025.1.0", …) and a
/// caller is free to omit it.
///
/// Pre-fix the value flowed unsanitized into `presence_user` HSET,
/// `PeerJoined` broadcasts, and the membership-replay payload. A multi-MB
/// string would poison tracing output and every other peer's
/// `PeerJoined` view; an ANSI-escape would overwrite operator terminals;
/// a NUL byte would corrupt log aggregators.
///
/// Rules — identical to `validate_client_id`'s rule set (length cap,
/// control-character rejection) so the same operator mental model applies
/// across every untrusted ingress string. The `|` rejection from
/// `validate_client_id` is intentionally NOT included here because
/// `user_application` is not embedded in the `LayerNotification` wire
/// payload — its only consumers are HSET fields and gRPC response
/// messages, neither of which use `|` as a delimiter.
pub fn validate_user_application(user_application: &str) -> Result<(), ServiceError> {
    if user_application.is_empty() {
        return Ok(());
    }

    if user_application.len() > MAX_USER_APPLICATION_BYTES {
        return Err(ServiceError::InvalidArgument(format!(
            "user_application exceeds {MAX_USER_APPLICATION_BYTES} byte limit"
        )));
    }

    if user_application
        .bytes()
        .any(|b| b == 0 || (0x01..=0x1F).contains(&b))
    {
        return Err(ServiceError::InvalidArgument(
            "user_application must not contain control characters".to_owned(),
        ));
    }

    Ok(())
}

/// Validate a CreateSession / ListSessions `base_path` (b.91j H-grpc-5).
///
/// Rules:
/// - Non-empty (callers already short-circuit empty-string, but we enforce
///   here too so the rule is co-located with the rest of the validation).
/// - Byte length `<= MAX_BASE_PATH_BYTES` (4096). Byte-aligned, not
///   character-aligned — downstream storage hex-encodes the value and logs
///   tag it verbatim (see `tracing::info!(%base_path)` in
///   `handle_create_session`).
/// - No null bytes — null-terminators corrupt Valkey keys on some clients
///   and poison logs.
/// - No control characters (ASCII 0..=0x1F). Control chars (including tab,
///   newline, BEL, etc.) in a base_path pollute tracing output and, via
///   ANSI-escape injection on `\x1b`, can overwrite operator terminal state.
///
/// Errors map to `ServiceError::InvalidArgument` → gRPC `INVALID_ARGUMENT`.
pub fn validate_base_path(path: &str) -> Result<(), ServiceError> {
    if path.is_empty() {
        return Err(ServiceError::InvalidArgument(
            "base_path must not be empty".to_owned(),
        ));
    }

    if path.len() > MAX_BASE_PATH_BYTES {
        return Err(ServiceError::InvalidArgument(format!(
            "base_path exceeds {MAX_BASE_PATH_BYTES} byte limit"
        )));
    }

    if path.contains('\0') {
        return Err(ServiceError::InvalidArgument(
            "base_path must not contain null bytes".to_owned(),
        ));
    }

    if path.chars().any(|c| c.is_control()) {
        return Err(ServiceError::InvalidArgument(
            "base_path must not contain control characters".to_owned(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Rejection tests ─────────────────────────────────────────────

    #[test]
    fn validate_layer_name_empty_string_returns_error() {
        let err = validate_layer_name("").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("layer")));
    }

    #[test]
    fn validate_layer_name_whitespace_only_returns_error() {
        let err = validate_layer_name("   ").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("blank")));
    }

    #[test]
    fn validate_layer_name_exceeds_255_chars_returns_error() {
        let long_name = "a".repeat(256);
        let err = validate_layer_name(&long_name).unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("255")));
    }

    #[test]
    fn validate_layer_name_null_byte_returns_error() {
        let err = validate_layer_name("hello\0world").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("control")));
    }

    #[test]
    fn validate_layer_name_newline_returns_error() {
        let err = validate_layer_name("hello\nworld").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("control")));
    }

    #[test]
    fn validate_layer_name_carriage_return_returns_error() {
        let err = validate_layer_name("hello\rworld").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("control")));
    }

    #[test]
    fn validate_layer_name_tab_returns_error() {
        let err = validate_layer_name("hello\tworld").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("control")));
    }

    #[test]
    fn validate_layer_name_bell_control_char_returns_error() {
        let err = validate_layer_name("hello\x07world").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("control")));
    }

    #[test]
    fn validate_layer_name_double_dot_returns_error() {
        let err = validate_layer_name("foo..bar").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("..")));
    }

    #[test]
    fn validate_layer_name_backslash_prefix_returns_error() {
        let err = validate_layer_name("\\\\server\\share").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("UNC")));
    }

    #[test]
    fn validate_layer_name_double_slash_prefix_returns_error() {
        let err = validate_layer_name("//server/share").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("UNC")));
    }

    #[test]
    fn validate_layer_name_windows_drive_uppercase_returns_error() {
        let err = validate_layer_name("C:\\folder").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("Windows")));
    }

    #[test]
    fn validate_layer_name_windows_drive_lowercase_returns_error() {
        let err = validate_layer_name("c:\\folder").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("Windows")));
    }

    // ── Error messages say "layer", not "session" ───────────────────

    #[test]
    fn validate_layer_name_error_messages_contain_layer_not_session() {
        let err = validate_layer_name("").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("layer"), "expected 'layer' in error: {msg}");
        assert!(
            !msg.contains("session"),
            "unexpected 'session' in error: {msg}"
        );
    }

    // ── Acceptance tests ────────────────────────────────────────────

    #[test]
    fn validate_layer_name_normal_ascii_succeeds() {
        assert!(validate_layer_name("my-layer").is_ok());
    }

    #[test]
    fn validate_layer_name_utf8_multibyte_succeeds() {
        assert!(validate_layer_name("レイヤー").is_ok());
    }

    #[test]
    fn validate_layer_name_exactly_255_chars_succeeds() {
        let name = "a".repeat(255);
        assert!(validate_layer_name(&name).is_ok());
    }

    #[test]
    fn validate_layer_name_single_slash_succeeds() {
        assert!(validate_layer_name("foo/bar").is_ok());
    }

    #[test]
    fn validate_layer_name_single_backslash_in_middle_succeeds() {
        assert!(validate_layer_name("foo\\bar").is_ok());
    }

    #[test]
    fn validate_layer_name_single_dot_succeeds() {
        assert!(validate_layer_name("foo.bar").is_ok());
    }

    // ── b.55p: byte-vs-char length for session/layer names ───────────────────
    //
    // Pre-fix: `name.chars().count() > 255`. Post-fix: `name.len() > 255`
    // (byte length). An emoji is 4 UTF-8 bytes, so 64 emojis (64 chars, 256
    // bytes) now rejects where it previously passed. This is load-bearing:
    // downstream Valkey keys, HashMap shards, and gRPC metadata all count
    // bytes, and a 255-codepoint name of 4-byte chars would blow every
    // byte-aligned budget.

    /// 📦 is a 4-byte UTF-8 code point (U+1F4E6). Repeating it `n` times
    /// gives exactly `4 * n` bytes and `n` chars, so we can target byte
    /// thresholds precisely without approximating.
    const PACKAGE_EMOJI: &str = "📦";

    #[test]
    fn validate_layer_name_emoji_255_bytes_succeeds() {
        // 63 emojis (252 bytes) + "abc" (3 bytes) = 255 bytes — at the cap.
        let name = format!("{}abc", PACKAGE_EMOJI.repeat(63));
        assert_eq!(name.len(), 255);
        assert_eq!(name.chars().count(), 66);
        assert!(
            validate_layer_name(&name).is_ok(),
            "name at exactly 255 bytes must succeed"
        );
    }

    #[test]
    fn validate_layer_name_emoji_256_bytes_returns_error() {
        // 64 emojis = 256 bytes, 64 chars. Pre-b.55p this would pass the
        // `chars().count() > 255` check; post-fix it rejects on bytes.
        let name = PACKAGE_EMOJI.repeat(64);
        assert_eq!(name.len(), 256);
        assert_eq!(name.chars().count(), 64);
        let err = validate_layer_name(&name).unwrap_err();
        assert!(
            matches!(err, ServiceError::InvalidArgument(ref msg) if msg.contains("byte")),
            "expected byte-limit error, got {err:?}"
        );
    }

    #[test]
    fn validate_layer_name_emoji_well_under_cap_succeeds() {
        // Sanity: 64 chars at 1-4 bytes per char must still pass when bytes ≤ 255.
        let name = PACKAGE_EMOJI.repeat(10); // 40 bytes, 10 chars.
        assert!(validate_layer_name(&name).is_ok());
    }

    // ── b.55p H-dom-1: delta bounds enforcement ─────────────────────────────
    //
    // `validate_delta_bounds` runs at the top of `apply_delta` and caps a
    // delta's entries + per-name byte lengths + each children list. Before
    // b.55p the OODF verifier + OMPF 100 MB cap only bounded total size; a
    // single 100 MB layer could carry one 99 MB `key_name` or millions of
    // `ReorderChildren` ids without being rejected. The name and list tests
    // below each construct a delta that exceeds exactly one cap by one and
    // assert the error variant identifies it.
    //
    // The entry cap is ONE combined limit (b.tbw): struct commands + set
    // fields + time samples, summed by `own_element_count`, at most
    // `MAX_DELTA_ENTRIES` (99,999 — the OODF verifier's table cap less the
    // root table). It replaced three per-kind caps of 100,000 each, which
    // admitted deltas the parser could never accept. The entry tests pin the
    // boundary on the SUM: at the cap in any split is accepted, one over in
    // any split is refused with the exact three counts, and a split under the
    // old per-kind caps whose sum is over is refused.

    use crate::domain::limits::{MAX_CHILDREN_LIST_LEN, MAX_DELTA_ENTRIES, MAX_NAME_BYTES};

    /// The combined cap as a `usize` count.
    const CAP: usize = MAX_DELTA_ENTRIES as usize;

    /// Assert `err` is the combined entry cap refusing exactly these counts.
    #[track_caller]
    fn assert_too_many_entries(err: DeltaError, counts: (usize, usize, usize)) {
        match err {
            DeltaError::TooManyEntries {
                struct_commands,
                set_fields,
                time_samples,
                max_entries,
            } => {
                assert_eq!(
                    (struct_commands, set_fields, time_samples),
                    (counts.0 as u64, counts.1 as u64, counts.2 as u64),
                    "the refusal carries the delta's own three counts"
                );
                assert_eq!(max_entries, MAX_DELTA_ENTRIES, "and the combined cap");
            }
            other => panic!("expected TooManyEntries {counts:?}, got {other:?}"),
        }
    }

    fn delete_section_cmd(section_id: u64) -> StructCommand {
        StructCommand::DeleteSection(oodf_ffi::CommandDeleteSection {
            parent_id: 0,
            section_id,
        })
    }

    /// A delta of `commands` delete-section commands, `fields` one-byte inline
    /// set fields and `samples` one-byte inline time samples — the cheapest
    /// entry of each kind, so only the entry count can trip.
    fn entries_delta(commands: usize, fields: usize, samples: usize) -> OwnedDelta {
        OwnedDelta {
            struct_commands: (0..commands as u64).map(delete_section_cmd).collect(),
            set_fields: (0..fields)
                .map(|_| oodf_ffi::SetField {
                    section_id: 1,
                    key_name: "k".to_string(),
                    value_or_ext_hash: vec![0u8],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                })
                .collect(),
            time_samples: (0..samples)
                .map(|i| oodf_ffi::TimeSample {
                    section_id: 1,
                    time: i as f64,
                    value_or_ext_hash: vec![0u8],
                    ext_value_size: 0,
                    ext_value_index: 0,
                })
                .collect(),
            ..Default::default()
        }
    }

    /// One entry over the combined cap, all of one kind and split across the
    /// three kinds, is refused with the exact counts. The single-kind rows are
    /// 100,000 of a kind — what the old per-kind caps admitted.
    #[test]
    fn validate_delta_bounds_one_entry_over_the_combined_cap_in_any_split_returns_too_many_entries()
    {
        let third = CAP / 3;
        for counts in [
            (CAP + 1, 0, 0),
            (0, CAP + 1, 0),
            (0, 0, CAP + 1),
            (third + 1, third, CAP - 2 * third),
        ] {
            let delta = entries_delta(counts.0, counts.1, counts.2);
            assert_eq!(
                own_element_count(&delta),
                MAX_DELTA_ENTRIES + 1,
                "{counts:?}"
            );
            let err = validate_delta_bounds(&delta)
                .expect_err("one entry over the combined cap must be refused");
            assert_too_many_entries(err, counts);
        }
    }

    /// 60,000 struct commands + 60,000 set fields: each kind under the old
    /// per-kind cap of 100,000, their sum over the combined one — the shape
    /// the per-kind caps admitted and the parser refused.
    #[test]
    fn validate_delta_bounds_kinds_each_under_the_old_per_kind_cap_but_over_in_sum_returns_too_many_entries(
    ) {
        let err = validate_delta_bounds(&entries_delta(60_000, 60_000, 0))
            .expect_err("120,000 entries in total must be refused");
        assert_too_many_entries(err, (60_000, 60_000, 0));
    }

    /// Exactly the combined cap is accepted in any split — all of one kind or
    /// spread over two. Paired with the test above's one-over rows, it pins
    /// the comparison to `>` on the sum.
    #[test]
    fn validate_delta_bounds_exactly_the_combined_cap_in_any_split_is_accepted() {
        for counts in [(CAP, 0, 0), (0, CAP, 0), (0, 0, CAP), (CAP - 1, 0, 1)] {
            let delta = entries_delta(counts.0, counts.1, counts.2);
            assert_eq!(own_element_count(&delta), MAX_DELTA_ENTRIES, "{counts:?}");
            assert!(
                validate_delta_bounds(&delta).is_ok(),
                "{counts:?}: exactly the cap must be accepted"
            );
        }
    }

    /// The refusal's text names the total, each count and the cap.
    #[test]
    fn too_many_entries_display_names_the_total_each_count_and_the_cap() {
        let err = validate_delta_bounds(&entries_delta(60_000, 60_000, 0)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "delta carries 120000 entries (60000 struct commands, 60000 set fields, 0 time \
             samples); one delta may carry at most 99999 entries in total"
        );
    }

    #[test]
    fn validate_delta_bounds_create_section_name_over_byte_cap_returns_error() {
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(
                oodf_ffi::CommandCreateSection {
                    parent_id: 0,
                    section_id: 1,
                    section_name: "a".repeat(MAX_NAME_BYTES + 1),
                    section_type: 0,
                },
            )],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        match err {
            DeltaError::DeltaBoundsExceeded {
                field,
                limit,
                actual,
            } => {
                assert_eq!(field, "CreateSection.section_name");
                assert_eq!(limit, MAX_NAME_BYTES as u64);
                assert_eq!(actual, (MAX_NAME_BYTES + 1) as u64);
            }
            other => {
                panic!("expected DeltaBoundsExceeded(CreateSection.section_name), got {other:?}")
            }
        }
    }

    #[test]
    fn validate_delta_bounds_move_section_new_name_over_byte_cap_returns_error() {
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::MoveSection(oodf_ffi::CommandMoveSection {
                old_parent_id: 0,
                new_parent_id: 0,
                section_id: 1,
                new_name: "a".repeat(MAX_NAME_BYTES + 1),
            })],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        match err {
            DeltaError::DeltaBoundsExceeded { field, .. } => {
                assert_eq!(field, "MoveSection.new_name");
            }
            other => panic!("expected DeltaBoundsExceeded(MoveSection.new_name), got {other:?}"),
        }
    }

    #[test]
    fn validate_delta_bounds_diff_change_section_name_over_byte_cap_returns_error() {
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 1,
            struct_commands: vec![StructCommand::DiffChangeSection(
                oodf_ffi::CommandDiffChangeSection {
                    parent_id: 0,
                    section_id: 1,
                    section_name: "a".repeat(MAX_NAME_BYTES + 1),
                    section_type: 0,
                    section_order: 0,
                },
            )],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        match err {
            DeltaError::DeltaBoundsExceeded { field, .. } => {
                assert_eq!(field, "DiffChangeSection.section_name");
            }
            other => panic!(
                "expected DeltaBoundsExceeded(DiffChangeSection.section_name), got {other:?}"
            ),
        }
    }

    #[test]
    fn validate_delta_bounds_set_field_key_name_over_byte_cap_returns_error() {
        let delta = OwnedDelta {
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "a".repeat(MAX_NAME_BYTES + 1),
                value_or_ext_hash: vec![0u8],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        match err {
            DeltaError::DeltaBoundsExceeded { field, .. } => {
                assert_eq!(field, "SetField.key_name");
            }
            other => panic!("expected DeltaBoundsExceeded(SetField.key_name), got {other:?}"),
        }
    }

    #[test]
    fn validate_delta_bounds_reorder_children_list_over_cap_returns_error() {
        // Build a reorder list with MAX_CHILDREN_LIST_LEN + 1 ids. The bounds
        // validator runs before `apply_delta` looks at `state.sections`, so
        // the ids don't need to correspond to real sections — we're testing
        // the cap on list length alone.
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::ReorderChildren(
                oodf_ffi::CommandReorderChildren {
                    section_id: 1,
                    children_list_id: 0,
                    children_list: (0..(MAX_CHILDREN_LIST_LEN + 1) as u64).collect(),
                },
            )],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        match &err {
            DeltaError::ChildrenListTooLong { limit, actual } => {
                assert_eq!(*limit, MAX_CHILDREN_LIST_LEN as u64);
                assert_eq!(*actual, (MAX_CHILDREN_LIST_LEN + 1) as u64);
            }
            other => panic!("expected ChildrenListTooLong, got {other:?}"),
        }
        // Its own variant now, but its text is the per-field bound's, byte
        // for byte, so a log or message reader sees no change.
        assert_eq!(
            err.to_string(),
            "delta bounds exceeded: ReorderChildren.children_list.len limit=100000, actual=100001"
        );
    }

    #[test]
    fn validate_delta_bounds_reorder_children_list_at_cap_is_accepted() {
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::ReorderChildren(
                oodf_ffi::CommandReorderChildren {
                    section_id: 1,
                    children_list_id: 0,
                    children_list: (0..MAX_CHILDREN_LIST_LEN as u64).collect(),
                },
            )],
            ..Default::default()
        };
        assert!(validate_delta_bounds(&delta).is_ok());
    }

    #[test]
    fn validate_delta_bounds_create_section_name_emoji_byte_length_not_char_count() {
        // Byte-vs-char: 1025 📦 emojis = 4100 bytes, 1025 chars. Pre-fix a
        // char-count-based check (hypothetical) would accept; post-fix the
        // byte-based cap rejects at 4097 bytes (MAX_NAME_BYTES = 4096).
        let name = PACKAGE_EMOJI.repeat(1025);
        assert!(name.len() > MAX_NAME_BYTES);
        assert!(name.chars().count() < MAX_NAME_BYTES);
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(
                oodf_ffi::CommandCreateSection {
                    parent_id: 0,
                    section_id: 1,
                    section_name: name,
                    section_type: 0,
                },
            )],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        assert!(matches!(
            err,
            DeltaError::DeltaBoundsExceeded {
                field: "CreateSection.section_name",
                ..
            }
        ));
    }

    // ── b.jr8 H-grpc-6: validate_client_id ────────────────────────────────
    //
    // Contract (see docstring on `validate_client_id`):
    //   * Empty string is accepted (proto semantic: disables self-echo).
    //   * > 256 bytes rejected with InvalidArgument("... 256 byte ...").
    //   * Control bytes 0x00..=0x1F rejected with InvalidArgument("... control ...").
    //   * Ordinary ASCII / UTF-8 accepted up to and including 256 bytes.
    //
    // Tests are structured one-assertion-per-test so a regression on any
    // single rule produces a targeted failure rather than a grab-bag of red.

    #[test]
    fn validate_client_id_empty_accepted() {
        // Empty disables self-echo suppression (SDD §12.5). Proto calls
        // this out; the validator must not break callers who opt out.
        assert!(validate_client_id("").is_ok());
    }

    #[test]
    fn validate_client_id_happy_path_ascii_accepted() {
        assert!(validate_client_id("client-A").is_ok());
    }

    #[test]
    fn validate_client_id_happy_path_uuid_accepted() {
        // Typical UUID4 — representative of the common production shape.
        assert!(validate_client_id("5b7c4e6c-0a3a-4f6a-8b01-a7c9e3d2f1ab").is_ok());
    }

    #[test]
    fn validate_client_id_happy_path_utf8_accepted() {
        // Multi-byte UTF-8 is fine — only control bytes are rejected, and
        // `char::is_control` is not applied (we match raw ASCII-range
        // control bytes only, so legitimate UTF-8 stays permitted).
        assert!(validate_client_id("user-みく").is_ok());
    }

    #[test]
    fn validate_client_id_at_cap_256_bytes_accepted() {
        // Exactly at the cap — boundary check.
        let id = "a".repeat(MAX_CLIENT_ID_BYTES);
        assert_eq!(id.len(), MAX_CLIENT_ID_BYTES);
        assert!(
            validate_client_id(&id).is_ok(),
            "client_id at exactly MAX_CLIENT_ID_BYTES must be accepted"
        );
    }

    #[test]
    fn validate_client_id_one_over_cap_257_bytes_rejected() {
        let id = "a".repeat(MAX_CLIENT_ID_BYTES + 1);
        let err = validate_client_id(&id).unwrap_err();
        match err {
            ServiceError::InvalidArgument(msg) => {
                assert!(
                    msg.contains(&MAX_CLIENT_ID_BYTES.to_string()),
                    "error message must cite the byte cap: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn validate_client_id_far_over_cap_1024_bytes_rejected() {
        // A multi-KB client_id would pollute tracing output and layer
        // notifications; the byte check must reject regardless of
        // character composition.
        let id = "x".repeat(1024);
        let err = validate_client_id(&id).unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_client_id_null_byte_rejected() {
        // NUL is part of the control-char range, so a separate null-byte
        // rule isn't needed — but we pin it here so a future refactor can't
        // silently let `\0` through.
        let err = validate_client_id("abc\0def").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_client_id_newline_rejected() {
        let err = validate_client_id("line1\nline2").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_client_id_carriage_return_rejected() {
        let err = validate_client_id("a\rb").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_client_id_tab_rejected() {
        let err = validate_client_id("a\tb").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_client_id_bell_control_char_rejected() {
        let err = validate_client_id("a\x07b").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_client_id_ansi_escape_rejected() {
        // ESC (0x1B) is the ANSI-escape introducer. A client_id containing
        // ESC echoed into `tracing::warn!(%client_id, ...)` could overwrite
        // operator terminal state — the docstring on `validate_client_id`
        // calls this out, so we pin the behavior.
        let err = validate_client_id("a\x1b[31mred").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_client_id_low_boundary_0x01_rejected() {
        // 0x01 — the lowest non-NUL control byte. Boundary guard for the
        // `0x01..=0x1F` range.
        let err = validate_client_id("a\x01b").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_client_id_high_boundary_0x1f_rejected() {
        // 0x1F — the highest byte in the rejected control range. If the
        // range boundary regresses (`..0x1F` vs `..=0x1F`), this test
        // catches it.
        let err = validate_client_id("a\x1fb").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_client_id_space_0x20_accepted() {
        // Space (0x20) sits just above the control-byte range. The
        // validator must NOT reject it — the control check is `0x01..=0x1F`
        // plus NUL, nothing else. Pinning this boundary keeps the rule
        // from silently drifting to "all non-graphic bytes".
        assert!(validate_client_id("has space").is_ok());
    }

    #[test]
    fn validate_client_id_error_maps_to_invalid_argument_status() {
        // End-to-end: the error returned by the validator must map to
        // gRPC INVALID_ARGUMENT when converted via `From<ServiceError>`.
        let err = validate_client_id("\0").unwrap_err();
        let status: tonic::Status = err.into();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    // ── b.v6d T11: validate_operation_id ────────────────────────────────────
    //
    // Contract (see the docstring on `validate_operation_id`):
    //   * Empty is accepted: it is the field's absent value.
    //   * 1..=64 bytes of `A–Z a–z 0–9 - . _ ~` are accepted.
    //   * Over 64 bytes → InvalidArgument("operation_id exceeds 64 byte limit").
    //   * Any other byte → InvalidArgument naming the allowed set. That set is
    //     what lets the id ride the two operation records unescaped: `:` is
    //     their delimiter, `{` / `}` would re-tag `le:{sid}:createop:{id}`'s
    //     Cluster slot, and the id is logged verbatim at INFO.

    /// The charset refusal's exact message — the wire contract the proto
    /// comment and the conformance suite quote.
    const OPERATION_ID_CHARSET_MESSAGE: &str =
        "operation_id may contain only ASCII letters, digits, '-', '.', '_' and '~'";

    #[test]
    fn validate_operation_id_empty_accepted() {
        assert!(
            validate_operation_id("").is_ok(),
            "empty is the field's absent value"
        );
    }

    #[test]
    fn validate_operation_id_every_allowed_character_accepted() {
        // Every member of the set, in one id (exactly 64 bytes: 26 + 26 + 10 + 2).
        let id = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-.";
        assert!(validate_operation_id(id).is_ok());
        assert!(validate_operation_id("_~").is_ok());
        assert!(
            validate_operation_id("5b7c4e6c-0a3a-4f6a-8b01-a7c9e3d2f1ab").is_ok(),
            "the recommended UUIDv4 text form must be accepted"
        );
    }

    #[test]
    fn validate_operation_id_at_cap_64_bytes_accepted() {
        let id = "a".repeat(MAX_OPERATION_ID_BYTES);
        assert_eq!(MAX_OPERATION_ID_BYTES, 64, "the wire contract says 64");
        assert!(validate_operation_id(&id).is_ok());
    }

    #[test]
    fn validate_operation_id_one_over_cap_65_bytes_rejected() {
        let id = "a".repeat(MAX_OPERATION_ID_BYTES + 1);
        match validate_operation_id(&id) {
            Err(ServiceError::InvalidArgument(msg)) => {
                assert_eq!(msg, "operation_id exceeds 64 byte limit")
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn validate_operation_id_each_disallowed_character_rejected() {
        // `:` and `|` are delimiters, space and `{` are the hash-tag / logging
        // hazards, a control byte reaches tracing output, and non-ASCII is
        // outside the byte-compared ASCII set even though it is valid UTF-8.
        for (bad, what) in [
            ("op:1", "a colon"),
            ("op|1", "a pipe"),
            ("op 1", "a space"),
            ("op{1", "an opening brace"),
            ("op}1", "a closing brace"),
            ("op\x071", "a control byte"),
            ("op\n1", "a newline"),
            ("opé1", "a non-ASCII character"),
        ] {
            match validate_operation_id(bad) {
                Err(ServiceError::InvalidArgument(msg)) => assert_eq!(
                    msg, OPERATION_ID_CHARSET_MESSAGE,
                    "{what} must be refused with the charset message"
                ),
                other => panic!("{what}: expected InvalidArgument, got {other:?}"),
            }
        }
    }

    #[test]
    fn validate_operation_id_length_is_checked_before_charset() {
        // A 65-byte id that is ALSO full of bad bytes reports the length: the
        // length bound is checked before the per-byte scan.
        let id = ":".repeat(MAX_OPERATION_ID_BYTES + 1);
        assert!(matches!(
            validate_operation_id(&id),
            Err(ServiceError::InvalidArgument(ref m)) if m.contains("64 byte limit")
        ));
    }

    #[test]
    fn validate_operation_id_error_maps_to_invalid_argument_status() {
        let status: tonic::Status = validate_operation_id("a:b").unwrap_err().into();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    // ── b.k1r medium: '|' is the notification payload delimiter ─────────────
    //
    // `LayerNotification::parse` uses `splitn(6, '|')` to unpack the payload
    // `update|{oid}|{seq}|{client_id}|{client_ts}|{server_ts}`. A client_id
    // containing a literal `|` shifts later fields left — `client_ts`
    // absorbs the trailing suffix of `client_id`, `server_ts` absorbs the
    // original `client_ts`, and the payload either fails parse with
    // `WrongFieldCount` (good, but the produce-side XADD already happened)
    // or silently mis-parses timestamps (bad). The fix rejects `|` at
    // ingress so the wire format stays grammar-free.
    #[test]
    fn validate_client_id_pipe_delimiter_rejected() {
        let err = validate_client_id("client|hack").unwrap_err();
        match err {
            ServiceError::InvalidArgument(msg) => assert!(
                msg.contains('|'),
                "rejection message must name the disallowed delimiter so operators \
                 triaging a rejected client can fix the caller; got: {msg}"
            ),
            other => panic!("expected InvalidArgument for pipe char, got {other:?}"),
        }
    }

    #[test]
    fn validate_client_id_leading_pipe_rejected() {
        // Edge-case: a `|` at the head would set `parts[0]` empty and shift
        // every following field by one — validate the rule still fires.
        let err = validate_client_id("|starts-with-pipe").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_client_id_trailing_pipe_rejected() {
        // Edge-case: a trailing `|` produces a 7th empty slot under
        // `splitn(6, '|')`, which doesn't corrupt timestamp parsing but
        // still violates the grammar. The rule fires here too.
        let err = validate_client_id("ends-with-pipe|").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    // ── b.91j H-grpc-5: validate_base_path ─────────────────────────────────
    //
    // Covers the two fields flagged in the deep review: no length cap, no
    // control-char rejection. The fix adds both (4096-byte cap + reject
    // every `char::is_control()` code point + explicit null-byte check).
    // Each test below lands exactly one rejection category so a regression
    // on one check cannot be masked by another.

    #[test]
    fn validate_base_path_empty_returns_error() {
        let err = validate_base_path("").unwrap_err();
        match err {
            ServiceError::InvalidArgument(msg) => {
                assert!(msg.contains("empty"), "message must mention empty: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn validate_base_path_exactly_4096_bytes_succeeds() {
        let path = "/".to_string() + "a".repeat(MAX_BASE_PATH_BYTES - 1).as_str();
        assert_eq!(path.len(), MAX_BASE_PATH_BYTES);
        assert!(
            validate_base_path(&path).is_ok(),
            "path at exactly MAX_BASE_PATH_BYTES must succeed"
        );
    }

    #[test]
    fn validate_base_path_4097_bytes_returns_error() {
        let path = "/".to_string() + "a".repeat(MAX_BASE_PATH_BYTES).as_str();
        assert_eq!(path.len(), MAX_BASE_PATH_BYTES + 1);
        let err = validate_base_path(&path).unwrap_err();
        match err {
            ServiceError::InvalidArgument(msg) => {
                assert!(
                    msg.contains(&MAX_BASE_PATH_BYTES.to_string()),
                    "message must mention the byte limit: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn validate_base_path_null_byte_returns_error() {
        let err = validate_base_path("/stage/foo\0bar").unwrap_err();
        match err {
            ServiceError::InvalidArgument(msg) => {
                // `\0` is both a null byte AND a control character. The
                // fix orders the null-byte check before the control-char
                // check so operators get the more specific message; we
                // assert the more-specific "null" text here to pin that
                // ordering.
                assert!(msg.contains("null"), "message must mention null: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn validate_base_path_ansi_escape_returns_error() {
        // ESC (0x1b) is the ANSI-escape introducer. The validator comment
        // explicitly calls it out because a `base_path` containing ESC
        // could overwrite operator terminal state via `tracing::info!(%base_path)`.
        let err = validate_base_path("/stage/\x1b[31mred").unwrap_err();
        match err {
            ServiceError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("control"),
                    "message must mention control char: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn validate_base_path_bell_returns_error() {
        let err = validate_base_path("/stage/\x07bell").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_base_path_newline_returns_error() {
        let err = validate_base_path("/stage/line1\nline2").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_base_path_carriage_return_returns_error() {
        let err = validate_base_path("/stage/line1\rline2").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_base_path_tab_returns_error() {
        // The engineer chose `char::is_control()` to stay consistent with
        // `validate_session_name`, so tab is rejected even though the
        // ticket hedged "except maybe \t". This test pins that choice —
        // if the policy is ever relaxed, this assertion is the signal.
        let err = validate_base_path("/stage/a\tb").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_base_path_del_0x7f_returns_error() {
        // DEL (0x7F) is a control char above the usual 0x00..=0x1F range.
        // `char::is_control()` catches it; a naive `< 0x20` check would
        // miss it. This pins the full-range policy.
        let err = validate_base_path("/stage/\x7fdel").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_base_path_unicode_control_u0080_returns_error() {
        // U+0080 (C1 control). `char::is_control()` matches C0 (0x00-0x1F),
        // DEL (0x7F), AND C1 (0x80-0x9F) — we assert C1 is also rejected so
        // UTF-8 encoded control characters can't slip through a byte-range
        // check.
        let err = validate_base_path("/stage/\u{0080}c1").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_base_path_happy_path_ascii_succeeds() {
        assert!(validate_base_path("/stage/assets/scene-01").is_ok());
    }

    #[test]
    fn validate_base_path_happy_path_utf8_succeeds() {
        // Non-control, non-null UTF-8 must pass — `base_path` is not a
        // layer/session name and doesn't impose ASCII constraints.
        assert!(validate_base_path("/stage/ステージ/shot-01").is_ok());
    }

    #[test]
    fn validate_base_path_single_slash_succeeds() {
        assert!(validate_base_path("/").is_ok());
    }

    #[test]
    fn validate_base_path_spaces_allowed() {
        // Space (0x20) is not a control char; must be accepted.
        assert!(validate_base_path("/stage/my project/shot 01").is_ok());
    }

    #[test]
    fn validate_base_path_emoji_byte_length_cap() {
        // 📦 is 4 bytes; at 1024 emojis we sit at 4096 bytes, exactly on
        // the cap. Add one character to cross it without relying on the
        // `MAX_BASE_PATH_BYTES + 1` literal.
        const PACKAGE: &str = "📦";
        let at_cap = PACKAGE.repeat(MAX_BASE_PATH_BYTES / PACKAGE.len());
        assert_eq!(at_cap.len(), MAX_BASE_PATH_BYTES);
        assert!(
            validate_base_path(&at_cap).is_ok(),
            "exact-cap emoji path must succeed"
        );

        let over_cap = format!("{at_cap}a");
        assert!(over_cap.len() > MAX_BASE_PATH_BYTES);
        let err = validate_base_path(&over_cap).unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains(&MAX_BASE_PATH_BYTES.to_string())
        ));
    }

    #[test]
    fn validate_delta_bounds_exact_caps_accepted() {
        // Sanity: a delta that sits exactly at every cap must be accepted —
        // `MAX_DELTA_ENTRIES` entries in total, a third of each kind.
        let third = MAX_DELTA_ENTRIES as usize / 3;
        assert_eq!(3 * third, MAX_DELTA_ENTRIES as usize);
        let mut struct_commands: Vec<StructCommand> = Vec::with_capacity(third);
        for i in 0..third as u64 {
            struct_commands.push(delete_section_cmd(i));
        }
        let mut set_fields: Vec<oodf_ffi::SetField> = Vec::with_capacity(third);
        for _ in 0..third {
            set_fields.push(oodf_ffi::SetField {
                section_id: 1,
                key_name: "a".repeat(MAX_NAME_BYTES), // at cap
                value_or_ext_hash: vec![0u8],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            });
        }
        let mut time_samples: Vec<oodf_ffi::TimeSample> = Vec::with_capacity(third);
        for i in 0..third {
            time_samples.push(oodf_ffi::TimeSample {
                section_id: 1,
                time: i as f64,
                value_or_ext_hash: vec![0u8],
                ext_value_size: 0,
                ext_value_index: 0,
            });
        }
        let delta = OwnedDelta {
            struct_commands,
            set_fields,
            time_samples,
            ..Default::default()
        };
        assert!(validate_delta_bounds(&delta).is_ok());
    }

    // ── b.39x item 1: validate_session_id ───────────────────────────────────
    //
    // session_id flows into Valkey hash tags (`le:{session_id}:sess`,
    // `le:presence:events:{sid}`, …), tracing spans, and pub/sub channel
    // names. The validator rejects empty, > MAX_SESSION_ID_BYTES, and
    // control characters (NUL + ASCII 0x01..=0x1F). Each rule lands its own
    // test so a regression on one cannot be masked by another.

    #[test]
    fn validate_session_id_empty_returns_error() {
        let err = validate_session_id("").unwrap_err();
        match err {
            ServiceError::InvalidArgument(msg) => {
                assert!(msg.contains("empty"), "message must mention empty: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn validate_session_id_happy_path_uuid_accepted() {
        // Typical session_id shape — UUID4. Representative of the common
        // production input.
        assert!(validate_session_id("5b7c4e6c-0a3a-4f6a-8b01-a7c9e3d2f1ab").is_ok());
    }

    #[test]
    fn validate_session_id_at_cap_256_bytes_accepted() {
        let id = "a".repeat(MAX_SESSION_ID_BYTES);
        assert_eq!(id.len(), MAX_SESSION_ID_BYTES);
        assert!(
            validate_session_id(&id).is_ok(),
            "session_id at exactly MAX_SESSION_ID_BYTES must be accepted"
        );
    }

    #[test]
    fn validate_session_id_one_over_cap_257_bytes_rejected() {
        let id = "a".repeat(MAX_SESSION_ID_BYTES + 1);
        let err = validate_session_id(&id).unwrap_err();
        match err {
            ServiceError::InvalidArgument(msg) => {
                assert!(
                    msg.contains(&MAX_SESSION_ID_BYTES.to_string()),
                    "error message must cite the byte cap: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn validate_session_id_null_byte_rejected() {
        let err = validate_session_id("abc\0def").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_session_id_newline_rejected() {
        let err = validate_session_id("a\nb").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_session_id_ansi_escape_rejected() {
        // ESC (0x1B) echoed into operator logs could overwrite terminal
        // state — pin the rejection so a future relaxation cannot regress.
        let err = validate_session_id("a\x1b[31mred").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_session_id_low_boundary_0x01_rejected() {
        let err = validate_session_id("a\x01b").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_id_high_boundary_0x1f_rejected() {
        let err = validate_session_id("a\x1fb").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_id_space_0x20_accepted() {
        // Pin the boundary: space (0x20) sits just above the control range
        // and must be accepted. If the rule ever drifts to "all non-graphic
        // bytes" this test catches it.
        assert!(validate_session_id("has space").is_ok());
    }

    #[test]
    fn validate_session_id_utf8_multibyte_accepted() {
        // Multi-byte UTF-8 must pass — the rule examines raw ASCII control
        // bytes only, not Unicode general categories.
        assert!(validate_session_id("session-セッション").is_ok());
    }

    #[test]
    fn validate_session_id_error_maps_to_invalid_argument_status() {
        // End-to-end: the error must map to gRPC INVALID_ARGUMENT.
        let err = validate_session_id("\0").unwrap_err();
        let status: tonic::Status = err.into();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    // ── b.39x item 2: validate_user_application ─────────────────────────────
    //
    // user_application is HSET into `presence_user`, broadcast on
    // `PeerJoined`, and replayed on membership replay. Same threat model as
    // client_id minus the '|' delimiter (user_application is not in the
    // LayerNotification wire payload). Empty is accepted (it is optional
    // descriptive metadata).

    #[test]
    fn validate_user_application_empty_accepted() {
        // user_application is optional — accept empty so callers can omit it.
        assert!(validate_user_application("").is_ok());
    }

    #[test]
    fn validate_user_application_happy_path_accepted() {
        // Typical real-world shape — Omniverse client identifier.
        assert!(validate_user_application("Omniverse Composer 2025.1.0").is_ok());
    }

    #[test]
    fn validate_user_application_at_cap_256_bytes_accepted() {
        let app = "a".repeat(MAX_USER_APPLICATION_BYTES);
        assert_eq!(app.len(), MAX_USER_APPLICATION_BYTES);
        assert!(
            validate_user_application(&app).is_ok(),
            "user_application at exactly MAX_USER_APPLICATION_BYTES must be accepted"
        );
    }

    #[test]
    fn validate_user_application_one_over_cap_257_bytes_rejected() {
        let app = "a".repeat(MAX_USER_APPLICATION_BYTES + 1);
        let err = validate_user_application(&app).unwrap_err();
        match err {
            ServiceError::InvalidArgument(msg) => {
                assert!(
                    msg.contains(&MAX_USER_APPLICATION_BYTES.to_string()),
                    "error message must cite the byte cap: {msg}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn validate_user_application_null_byte_rejected() {
        let err = validate_user_application("app\0name").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_user_application_newline_rejected() {
        let err = validate_user_application("line1\nline2").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_user_application_ansi_escape_rejected() {
        let err = validate_user_application("\x1b[31m").unwrap_err();
        assert!(matches!(
            err,
            ServiceError::InvalidArgument(ref m) if m.contains("control")
        ));
    }

    #[test]
    fn validate_user_application_low_boundary_0x01_rejected() {
        let err = validate_user_application("a\x01b").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_user_application_high_boundary_0x1f_rejected() {
        let err = validate_user_application("a\x1fb").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_user_application_pipe_accepted() {
        // user_application is NOT in the LayerNotification wire payload —
        // unlike client_id, the '|' delimiter rule does not apply. Pin the
        // carve-out so a future copy-paste of the client_id validator does
        // not accidentally tighten the rule here.
        assert!(validate_user_application("Pipes|are|fine|here").is_ok());
    }

    #[test]
    fn validate_user_application_utf8_multibyte_accepted() {
        assert!(validate_user_application("Composer・2025").is_ok());
    }

    #[test]
    fn validate_user_application_error_maps_to_invalid_argument_status() {
        let err = validate_user_application("\0").unwrap_err();
        let status: tonic::Status = err.into();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
    }

    // ── b.39x item 3: validate_delta_bounds — check_name_chars rejection ────
    //
    // Apply-engine-side coverage of the NUL/control-char rule lives in
    // `domain::delta::tests`; here we exercise the validator directly so a
    // regression in `check_name_chars` produces a targeted failure rather
    // than coupling to apply-engine state setup.

    #[test]
    fn validate_delta_bounds_create_section_name_with_control_char_returns_error() {
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(
                oodf_ffi::CommandCreateSection {
                    parent_id: 0,
                    section_id: 1,
                    section_name: "bad\x07name".to_string(),
                    section_type: 0,
                },
            )],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        assert!(
            matches!(err, DeltaError::NameContainsControlChar { field } if field == "CreateSection.section_name"),
            "expected NameContainsControlChar(CreateSection.section_name), got {err:?}"
        );
    }

    #[test]
    fn validate_delta_bounds_set_field_key_name_with_null_returns_error() {
        let delta = OwnedDelta {
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "k\0bad".to_string(),
                value_or_ext_hash: vec![0u8],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        assert!(
            matches!(err, DeltaError::NameContainsControlChar { field } if field == "SetField.key_name"),
            "expected NameContainsControlChar(SetField.key_name), got {err:?}"
        );
    }

    // ── b.39x item 4: validate_delta_bounds — non-finite time rejection ─────

    #[test]
    fn validate_delta_bounds_time_sample_nan_returns_non_finite_time() {
        let delta = OwnedDelta {
            time_samples: vec![oodf_ffi::TimeSample {
                section_id: 1,
                time: f64::NAN,
                value_or_ext_hash: vec![0u8],
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        match err {
            DeltaError::TimeSampleNonFiniteTime { index, time } => {
                assert_eq!(index, 0);
                assert!(time.is_nan());
            }
            other => panic!("expected TimeSampleNonFiniteTime, got {other:?}"),
        }
    }

    #[test]
    fn validate_delta_bounds_time_sample_positive_infinity_returns_non_finite_time() {
        let delta = OwnedDelta {
            time_samples: vec![oodf_ffi::TimeSample {
                section_id: 1,
                time: f64::INFINITY,
                value_or_ext_hash: vec![0u8],
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        assert!(matches!(
            err,
            DeltaError::TimeSampleNonFiniteTime { time, .. } if time == f64::INFINITY
        ));
    }

    #[test]
    fn validate_delta_bounds_time_sample_negative_infinity_returns_non_finite_time() {
        let delta = OwnedDelta {
            time_samples: vec![oodf_ffi::TimeSample {
                section_id: 1,
                time: f64::NEG_INFINITY,
                value_or_ext_hash: vec![0u8],
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        assert!(matches!(
            err,
            DeltaError::TimeSampleNonFiniteTime { time, .. } if time == f64::NEG_INFINITY
        ));
    }

    // ── b.8jc: validate_delta_bounds — the `-0.0` time backstop ─────────────

    /// A delta whose time samples sit at `times`, in order, on section 1.
    fn time_samples_at(times: &[f64]) -> OwnedDelta {
        OwnedDelta {
            time_samples: times
                .iter()
                .map(|&time| oodf_ffi::TimeSample {
                    section_id: 1,
                    time,
                    value_or_ext_hash: vec![0u8],
                    ext_value_size: 0,
                    ext_value_index: 0,
                })
                .collect(),
            ..Default::default()
        }
    }

    /// A `-0.0` time is refused with the backstop variant, naming the index of
    /// the offending sample — here the third, behind two canonical ones.
    #[test]
    fn validate_delta_bounds_time_sample_negative_zero_returns_negative_zero_time_at_its_index() {
        let delta = time_samples_at(&[1.0, 0.0, -0.0, 2.0]);
        let err = validate_delta_bounds(&delta).unwrap_err();
        assert!(
            matches!(err, DeltaError::TimeSampleNegativeZeroTime { index: 2 }),
            "expected TimeSampleNegativeZeroTime at index 2, got {err:?}"
        );
    }

    /// `+0.0` is the canonical spelling of time zero and is accepted, alone
    /// and beside other times.
    #[test]
    fn validate_delta_bounds_time_sample_positive_zero_is_accepted() {
        validate_delta_bounds(&time_samples_at(&[0.0])).expect("+0.0 is canonical");
        validate_delta_bounds(&time_samples_at(&[-1.0, 0.0, 1.0]))
            .expect("+0.0 beside other times is canonical");
    }

    // ── b.39x item 5: validate_delta_bounds — self-parenting rejection ──────

    #[test]
    fn validate_delta_bounds_create_section_self_parenting_returns_self_parenting() {
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(
                oodf_ffi::CommandCreateSection {
                    parent_id: 5,
                    section_id: 5,
                    section_name: "Cycle".to_string(),
                    section_type: 0,
                },
            )],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::SelfParenting {
                    command: "CreateSection",
                    section_id: 5,
                }
            ),
            "expected SelfParenting(CreateSection, 5), got {err:?}"
        );
    }

    #[test]
    fn validate_delta_bounds_move_section_self_parenting_returns_self_parenting() {
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::MoveSection(oodf_ffi::CommandMoveSection {
                old_parent_id: 0,
                new_parent_id: 6,
                section_id: 6,
                new_name: "Cycle".to_string(),
            })],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::SelfParenting {
                    command: "MoveSection",
                    section_id: 6,
                }
            ),
            "expected SelfParenting(MoveSection, 6), got {err:?}"
        );
    }

    #[test]
    fn validate_delta_bounds_move_section_old_parent_self_parenting_returns_self_parenting() {
        // b.39x PM follow-up: `old_parent_id == section_id` must also fail
        // self-parenting validation. The narrative is "a section cannot be
        // its own parent — past or present"; without this branch the apply
        // engine in `apply_move_section` would still execute a
        // `remove_child_from_parent(section_id, section_id, …)` /
        // `insert_child_into_parent(new_parent_id, section_id, …)` pair
        // that leaves a spurious self-cycle in `state.children` for a tick
        // before being corrected by the move. Reject at the validator.
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::MoveSection(oodf_ffi::CommandMoveSection {
                old_parent_id: 9,
                new_parent_id: 12,
                section_id: 9,
                new_name: "MovedOutOfSelf".to_string(),
            })],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::SelfParenting {
                    command: "MoveSection",
                    section_id: 9,
                }
            ),
            "expected SelfParenting(MoveSection, 9) on old_parent_id == section_id, got {err:?}"
        );
    }

    #[test]
    fn validate_delta_bounds_move_section_old_parent_zero_allowed() {
        // Carve-out twin to the existing parent_zero test: when
        // `old_parent_id == 0` (the "moved from root" case) and
        // `section_id == 0`, the validator must still pass — the rule only
        // fires when `old_parent_id != 0`.
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::MoveSection(oodf_ffi::CommandMoveSection {
                old_parent_id: 0,
                new_parent_id: 12,
                section_id: 0,
                new_name: "MovedFromRoot".to_string(),
            })],
            ..Default::default()
        };
        assert!(
            validate_delta_bounds(&delta).is_ok(),
            "old_parent_id == 0 must not trigger SelfParenting even when section_id == 0"
        );
    }

    #[test]
    fn validate_delta_bounds_diff_change_section_self_parenting_returns_self_parenting() {
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 1,
            struct_commands: vec![StructCommand::DiffChangeSection(
                oodf_ffi::CommandDiffChangeSection {
                    parent_id: 7,
                    section_id: 7,
                    section_name: "Cycle".to_string(),
                    section_type: 0,
                    section_order: 0,
                },
            )],
            ..Default::default()
        };
        let err = validate_delta_bounds(&delta).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::SelfParenting {
                    command: "DiffChangeSection",
                    section_id: 7,
                }
            ),
            "expected SelfParenting(DiffChangeSection, 7), got {err:?}"
        );
    }

    #[test]
    fn validate_delta_bounds_parent_zero_section_zero_not_self_parenting() {
        // Carve-out: parent_id == 0 is the root marker and is allowed even
        // when section_id == 0 (the rule only fires on parent_id != 0).
        // Pin the carve-out so a tightening of the validator catches this
        // boundary test rather than slipping through.
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(
                oodf_ffi::CommandCreateSection {
                    parent_id: 0,
                    section_id: 0,
                    section_name: "EdgeCase".to_string(),
                    section_type: 0,
                },
            )],
            ..Default::default()
        };
        // The validator must not reject; downstream apply-engine logic
        // separately handles section_id == 0 (root) semantics — the rule
        // we're pinning here is purely "parent != section" enforcement.
        assert!(
            validate_delta_bounds(&delta).is_ok(),
            "parent_id == 0 must not trigger SelfParenting even when section_id == 0"
        );
    }

    // ── G3: validate_delta_bounds — external hash length ────────────────────
    //
    // An external value (`ext_value_size > 0`) names its blob by a 32-byte
    // SHA-256 hash in `value_or_ext_hash`. Any other length is refused with
    // `InvalidExternalHashLength` rather than padded or truncated into a hash
    // the client never sent. An inline value (`ext_value_size == 0`) carries
    // its payload in the same bytes and is never length-checked.

    /// Hash lengths that must be refused: empty, one short, one long, and the
    /// 64-character hex *string* length a client might send by mistake.
    const WRONG_EXTERNAL_HASH_LENS: [usize; 4] = [0, 31, 33, 64];

    /// A nonzero external-value size, so the external branch is taken.
    const EXT_SIZE: u64 = 1024;

    fn ext_set_field(value_or_ext_hash: Vec<u8>, ext_value_size: u64) -> OwnedDelta {
        OwnedDelta {
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "k".to_string(),
                value_or_ext_hash,
                ext_value_size,
                ext_value_index: 1,
                set_order: 0,
            }],
            ..Default::default()
        }
    }

    fn ext_time_sample(value_or_ext_hash: Vec<u8>, ext_value_size: u64) -> OwnedDelta {
        OwnedDelta {
            time_samples: vec![oodf_ffi::TimeSample {
                section_id: 1,
                time: 1.0,
                value_or_ext_hash,
                ext_value_size,
                ext_value_index: 1,
            }],
            ..Default::default()
        }
    }

    fn assert_invalid_hash_length(delta: &OwnedDelta, len: usize, what: &str) {
        let err = validate_delta_bounds(delta).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::InvalidExternalHashLength { got, expected: 32 } if got == len
            ),
            "{what}: a {len}-byte external hash must be refused \
             InvalidExternalHashLength {{ got: {len}, expected: 32 }}, got {err:?}"
        );
    }

    #[test]
    fn validate_delta_bounds_set_field_external_hash_wrong_length_returns_invalid_hash_length() {
        for len in WRONG_EXTERNAL_HASH_LENS {
            let delta = ext_set_field(vec![0xAB; len], EXT_SIZE);
            assert_invalid_hash_length(&delta, len, "SetField");
        }
    }

    #[test]
    fn validate_delta_bounds_time_sample_external_hash_wrong_length_returns_invalid_hash_length() {
        for len in WRONG_EXTERNAL_HASH_LENS {
            let delta = ext_time_sample(vec![0xAB; len], EXT_SIZE);
            assert_invalid_hash_length(&delta, len, "TimeSample");
        }
    }

    #[test]
    fn validate_delta_bounds_external_hash_32_bytes_accepted() {
        validate_delta_bounds(&ext_set_field(vec![0xAB; 32], EXT_SIZE))
            .expect("a 32-byte SetField external hash must pass");
        validate_delta_bounds(&ext_time_sample(vec![0xAB; 32], EXT_SIZE))
            .expect("a 32-byte TimeSample external hash must pass");
    }

    #[test]
    fn validate_delta_bounds_inline_value_any_length_accepted() {
        // `ext_value_size == 0` marks an inline value: its bytes are the
        // payload, not a hash, so the hash-length rule must not apply.
        for len in [0, 31, 32, 33, 64] {
            validate_delta_bounds(&ext_set_field(vec![0xAB; len], 0))
                .unwrap_or_else(|e| panic!("inline SetField of {len} bytes must pass: {e:?}"));
            validate_delta_bounds(&ext_time_sample(vec![0xAB; len], 0))
                .unwrap_or_else(|e| panic!("inline TimeSample of {len} bytes must pass: {e:?}"));
        }
    }
}
