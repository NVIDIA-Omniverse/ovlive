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

//! The per-delta size limits at the FFI boundary: `parse_delta`'s entry-count
//! probe (`probe_entry_counts` in `cpp/oodf.cpp`), which runs after the magic
//! check and before the FlatBuffers verifier, and the parser's defensive cap
//! on one `ReorderChildren` children list.
//!
//! A delta is exactly one root table plus one table per struct command, set
//! field and time sample, and the verifier admits at most
//! [`MAX_DELTA_TABLES`] tables. So one delta carries at most
//! `MAX_DELTA_TABLES - 1` (99,999) entries in total, summed across the three
//! kinds. The probe refuses a buffer over that with
//! [`OodfError::TooManyTables`], carrying the buffer's own three vector
//! lengths and the limit. A buffer the probe cannot count — a root offset, a
//! root table or a vector header that fails its bounds check, or unpaired
//! union vectors — falls through to the verifier and keeps the
//! `VerifierFailed` it would have had without the probe.
//!
//! What each test discriminates:
//!
//! | Test | Fails if |
//! |---|---|
//! | exactly the limit, mixed kinds, parses | the probe compares with `>=`, or counts one kind twice |
//! | one over the limit, mixed, returns the exact counts | the probe forgets the root table (compares the sum with `MAX_DELTA_TABLES`), or mixes up the three counts |
//! | 60,000 commands + 60,000 fields | there is no probe (the parser before it answered `VerifierFailed`), or the probe checks per-kind caps instead of the total — each kind here is under any per-kind cap |
//! | 100,000 of one kind, for each kind | the probe checks per-kind caps at the old documented limit of 100,000, which admit the buffer and leave the verifier to answer `VerifierFailed` |
//! | a malformed element over the limit | the probe walks elements instead of reading only headers |
//! | out-of-bounds vector length, unpaired command vectors, an out-of-bounds root offset, truncation after the root table | the probe trusts an unverified header and claims a count it cannot have read |
//! | a children list one past the parser's cap / at it | the parser's cap answers the opaque `VerifierFailed` instead of the typed `ChildrenListTooLong` with the list's length, or refuses the cap itself |
//!
//! Every buffer is built by the real `build_delta`, so the counts are the
//! counts a client's encoder would produce. The fixtures carry up to 120,000
//! entries or 1,000,001 children ids; each test builds and parses a few
//! megabytes.

use oodf_ffi::{
    build_delta, parse_delta, CommandDeleteSection, CommandReorderChildren, OodfError, OwnedDelta,
    SetField, StructCommand, TimeSample, MAX_DELTA_TABLES,
};

/// The most entries one delta may carry: the verifier's table cap less the
/// root table.
const MAX_ENTRIES: usize = MAX_DELTA_TABLES - 1;

/// The parser's defensive cap on one `ReorderChildren` children list
/// (`kMaxCommands` in `cpp/oodf.cpp`): ten times the service's own
/// 100,000-id limit on one list.
const PARSER_CHILDREN_CAP: usize = 1_000_000;

// `Delta` vtable slots (`Delta::VT_*` in the generated header): the byte
// offset of each field's entry inside the root table's vtable. Field order in
// `schema/OmniObjectDelta.fbs` is isDiff, baseVersion, structCommands (a union
// vector, so two slots: `_type` then the values), setFields, timeSamples,
// sourceFormat; slot `i` sits at `4 + 2 * i`. `cpp/tests/round_trip.cpp`'s
// `wire_layout_split_structcommands_vtable_slots` pins the two-slot union.
const VT_STRUCTCOMMANDS_TYPE: usize = 8;
const VT_SETFIELDS: usize = 12;

/// A delta of `commands` delete-section commands, `fields` inline set fields
/// and `samples` inline time samples — the cheapest entry of each kind.
fn delta_of(commands: usize, fields: usize, samples: usize) -> OwnedDelta {
    OwnedDelta {
        struct_commands: (0..commands)
            .map(|i| {
                StructCommand::DeleteSection(CommandDeleteSection {
                    parent_id: 0,
                    section_id: i as u64 + 1,
                })
            })
            .collect(),
        set_fields: (0..fields)
            .map(|i| SetField {
                section_id: 1,
                key_name: format!("k{i}"),
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            })
            .collect(),
        time_samples: (0..samples)
            .map(|i| TimeSample {
                section_id: 1,
                time: i as f64,
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
            })
            .collect(),
        ..Default::default()
    }
}

/// The wire bytes of [`delta_of`].
fn encoded(commands: usize, fields: usize, samples: usize) -> Vec<u8> {
    build_delta(&delta_of(commands, fields, samples)).expect("build_delta cannot fail here")
}

/// Assert `result` is the entry-count refusal with exactly these counts and the
/// real limit.
#[track_caller]
fn assert_too_many_tables(
    result: Result<OwnedDelta, OodfError>,
    struct_commands: usize,
    set_fields: usize,
    time_samples: usize,
) {
    match result {
        Err(OodfError::TooManyTables {
            struct_commands: sc,
            set_fields: sf,
            time_samples: ts,
            max_entries,
        }) => {
            assert_eq!(
                (sc, sf, ts),
                (
                    struct_commands as u64,
                    set_fields as u64,
                    time_samples as u64
                ),
                "the refusal must carry the buffer's own three vector lengths"
            );
            assert_eq!(
                max_entries, MAX_ENTRIES as u64,
                "the refusal must carry the limit: the verifier's table cap less the root"
            );
        }
        Err(other) => panic!("expected TooManyTables, got {other:?}"),
        Ok(d) => panic!(
            "expected TooManyTables, but the buffer parsed ({} + {} + {} entries)",
            d.struct_commands.len(),
            d.set_fields.len(),
            d.time_samples.len()
        ),
    }
}

/// Assert `result` is the verifier's refusal, not the probe's.
#[track_caller]
fn assert_verifier_failed(result: Result<OwnedDelta, OodfError>, what: &str) {
    match result {
        Err(OodfError::VerifierFailed(_)) => {}
        Err(other) => panic!(
            "{what}: the probe cannot count this buffer, so it must fall through to the \
             verifier's VerifierFailed; got {other:?}"
        ),
        Ok(_) => panic!("{what}: a malformed buffer parsed"),
    }
}

fn u16_at(buf: &[u8], at: usize) -> usize {
    usize::from(u16::from_le_bytes([buf[at], buf[at + 1]]))
}

fn u32_at(buf: &[u8], at: usize) -> usize {
    u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]) as usize
}

fn i32_at(buf: &[u8], at: usize) -> i64 {
    i64::from(i32::from_le_bytes([
        buf[at],
        buf[at + 1],
        buf[at + 2],
        buf[at + 3],
    ]))
}

/// `(root table position, root vtable position)` of a buffer `build_delta`
/// produced. FlatBuffers layout: the root `uoffset_t` at byte 0 points at the
/// root table, whose first four bytes are a signed back-offset to its vtable.
fn root_and_vtable(buf: &[u8]) -> (usize, usize) {
    let root = u32_at(buf, 0);
    let vtable = usize::try_from(root as i64 - i32_at(buf, root)).expect("vtable inside buffer");
    (root, vtable)
}

/// Byte position of the length word of the root table's vector field at
/// vtable slot `vt`.
fn vector_length_pos(buf: &[u8], vt: usize) -> usize {
    let (root, vtable) = root_and_vtable(buf);
    let field = u16_at(buf, vtable + vt);
    assert_ne!(
        field, 0,
        "the fixture must carry the vector at vtable slot {vt}"
    );
    let slot = root + field;
    slot + u32_at(buf, slot)
}

/// One past the root table's last inline byte (its vtable's second word is the
/// table's inline size).
fn root_table_end(buf: &[u8]) -> usize {
    let (root, vtable) = root_and_vtable(buf);
    root + u16_at(buf, vtable + 2)
}

fn put_u32(buf: &mut [u8], at: usize, value: u32) {
    buf[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

// ── The boundary ─────────────────────────────────────────────────────────────

/// The constant is the verifier's cap: one delta's entries are the tables less
/// the root, 99,999. A change to either side moves every number below.
#[test]
fn max_delta_tables_is_the_verifier_cap_of_one_hundred_thousand() {
    assert_eq!(MAX_DELTA_TABLES, 100_000);
    assert_eq!(MAX_ENTRIES, 99_999);
}

/// `MAX_DELTA_TABLES - 1` entries spread across all three kinds parse: the
/// probe admits the limit itself. A probe comparing with `>=` refuses this.
#[test]
fn parse_delta_at_exactly_the_entry_limit_mixed_across_kinds_parses() {
    let third = MAX_ENTRIES / 3;
    let (commands, fields, samples) = (third, third, MAX_ENTRIES - 2 * third);
    assert_eq!(commands + fields + samples, MAX_ENTRIES);

    let parsed = parse_delta(&encoded(commands, fields, samples))
        .unwrap_or_else(|e| panic!("a delta of exactly {MAX_ENTRIES} entries must parse: {e:?}"));

    assert_eq!(
        (
            parsed.struct_commands.len(),
            parsed.set_fields.len(),
            parsed.time_samples.len()
        ),
        (commands, fields, samples),
        "every entry must come back"
    );
}

/// One entry over the limit is refused `TooManyTables` with the exact three
/// counts. Paired with the test above it pins the threshold to the entry: a
/// probe that forgot the root table (comparing the sum with
/// `MAX_DELTA_TABLES`) would admit this buffer, and the verifier would then
/// refuse its 100,001 tables as `VerifierFailed`.
#[test]
fn parse_delta_one_entry_over_the_limit_mixed_across_kinds_returns_too_many_tables_with_exact_counts(
) {
    let third = MAX_ENTRIES / 3;
    let (commands, fields, samples) = (third + 1, third, MAX_ENTRIES - 2 * third);
    assert_eq!(commands + fields + samples, MAX_ENTRIES + 1);

    let result = parse_delta(&encoded(commands, fields, samples));

    assert_too_many_tables(result, commands, fields, samples);
}

/// 60,000 struct commands and 60,000 set fields — each kind inside the old
/// per-kind limit of 100,000, their sum over the verifier's table cap — are
/// refused `TooManyTables` naming both counts. The parser before the probe
/// answered the verifier's opaque `VerifierFailed` here.
#[test]
fn parse_delta_sixty_thousand_commands_and_sixty_thousand_fields_returns_too_many_tables() {
    let result = parse_delta(&encoded(60_000, 60_000, 0));

    assert_too_many_tables(result, 60_000, 60_000, 0);
}

/// 100,000 entries of any single kind — one over the limit, and the old
/// documented per-kind maximum, which was never reachable — are refused
/// `TooManyTables`, for each kind.
#[test]
fn parse_delta_one_hundred_thousand_of_a_single_kind_returns_too_many_tables() {
    const N: usize = MAX_ENTRIES + 1;
    for (commands, fields, samples) in [(N, 0, 0), (0, N, 0), (0, 0, N)] {
        let result = parse_delta(&encoded(commands, fields, samples));

        assert_too_many_tables(result, commands, fields, samples);
    }
}

// ── What the probe reads, and what it does not ───────────────────────────────

/// The probe reads vector headers only: an over-limit buffer whose first set
/// field's element offset points far outside the buffer is still refused
/// `TooManyTables`, not `VerifierFailed`. This is the documented order — the
/// count is the cheaper and more useful refusal, and both are
/// `INVALID_ARGUMENT` at the service — and it shows the probe never follows
/// an element offset.
#[test]
fn parse_delta_over_the_limit_with_a_malformed_element_returns_too_many_tables() {
    let mut buf = encoded(60_000, 60_000, 0);
    let len_pos = vector_length_pos(&buf, VT_SETFIELDS);
    assert_eq!(
        u32_at(&buf, len_pos),
        60_000,
        "located the setFields header"
    );
    // Element 0's `uoffset_t` sits right after the length word.
    put_u32(&mut buf, len_pos + 4, u32::MAX - 16);

    let result = parse_delta(&buf);

    assert_too_many_tables(result, 60_000, 60_000, 0);
}

/// An over-limit buffer whose `setFields` length word claims `u32::MAX`
/// elements fails the probe's `VerifyVector` and falls through to the
/// verifier, which refuses it `VerifierFailed`. A probe that read the length
/// without bounds-checking the vector would report a count of four billion.
#[test]
fn parse_delta_over_the_limit_with_an_out_of_bounds_vector_length_returns_verifier_failed() {
    let mut buf = encoded(60_000, 60_000, 0);
    let len_pos = vector_length_pos(&buf, VT_SETFIELDS);
    assert_eq!(
        u32_at(&buf, len_pos),
        60_000,
        "located the setFields header"
    );
    put_u32(&mut buf, len_pos, u32::MAX);

    assert_verifier_failed(
        parse_delta(&buf),
        "setFields length word patched to u32::MAX",
    );
}

/// An over-limit buffer whose `structCommands_type` vector is shorter than its
/// `structCommands` vector cannot be counted — the union's two vectors are
/// unpaired — so the probe leaves the verdict to the verifier, whose union
/// check refuses it `VerifierFailed`.
#[test]
fn parse_delta_over_the_limit_with_unpaired_command_vectors_returns_verifier_failed() {
    let mut buf = encoded(60_000, 60_000, 0);
    let len_pos = vector_length_pos(&buf, VT_STRUCTCOMMANDS_TYPE);
    assert_eq!(
        u32_at(&buf, len_pos),
        60_000,
        "located the structCommands_type header"
    );
    put_u32(&mut buf, len_pos, 1);

    assert_verifier_failed(
        parse_delta(&buf),
        "structCommands_type shortened to one element",
    );
}

/// An over-limit buffer whose root offset (the `uoffset_t` at byte 0) points
/// past the end of the buffer cannot be counted: the probe's first bounds
/// check fails, and the verifier refuses the buffer `VerifierFailed`. Bytes
/// 4..8 (the file identifier) are left alone, so the magic check passes and
/// the probe is reached.
#[test]
fn parse_delta_over_the_limit_with_an_out_of_bounds_root_offset_returns_verifier_failed() {
    let mut buf = encoded(60_000, 60_000, 0);
    let past_end = u32::try_from(buf.len()).expect("a few-MB buffer") + 64;
    put_u32(&mut buf, 0, past_end);
    assert_eq!(&buf[4..8], b"OODF", "the file identifier is untouched");

    assert_verifier_failed(
        parse_delta(&buf),
        "root offset patched past the end of the buffer",
    );
}

/// An over-limit buffer truncated right after its root table keeps a valid
/// root table and vtable, but every vector offset now points past the end, so
/// the probe's offset check fails and the verifier refuses the buffer
/// `VerifierFailed` — the outcome a truncated upload has always had.
#[test]
fn parse_delta_over_the_limit_truncated_after_its_root_table_returns_verifier_failed() {
    let full = encoded(60_000, 60_000, 0);
    let end = root_table_end(&full);
    assert!(
        end >= 8 && end < full.len(),
        "the root table must sit near the front of a {}-byte buffer, ending at {end}",
        full.len()
    );
    let truncated = &full[..end];
    assert_eq!(
        &truncated[4..8],
        b"OODF",
        "the truncation keeps the file identifier, so the magic check passes"
    );

    assert_verifier_failed(
        parse_delta(truncated),
        "truncated just after the root table",
    );
}

// ── The children-list cap ────────────────────────────────────────────────────

/// A delta whose one `ReorderChildren` command names `n` children.
fn reorder_of(n: usize) -> Vec<u8> {
    build_delta(&OwnedDelta {
        struct_commands: vec![StructCommand::ReorderChildren(CommandReorderChildren {
            section_id: 1,
            children_list_id: 0,
            children_list: (1..=n as u64).collect(),
        })],
        ..Default::default()
    })
    .expect("build_delta cannot fail here")
}

/// A children list one id past the parser's defensive cap is refused as the
/// typed `ChildrenListTooLong` carrying the list's length — which the service
/// answers `CHILDREN_LIST_TOO_LONG` — not as the opaque `VerifierFailed` it
/// was answered with before. The list is a `[ulong]` vector, which the
/// verifier does not count as tables, so the entry-count probe never sees it.
#[test]
fn parse_delta_children_list_one_past_the_parser_cap_returns_children_list_too_long_with_its_length(
) {
    let result = parse_delta(&reorder_of(PARSER_CHILDREN_CAP + 1));

    match result {
        Err(OodfError::ChildrenListTooLong { len }) => {
            assert_eq!(len, PARSER_CHILDREN_CAP as u64 + 1, "the list's own length");
        }
        other => panic!("expected ChildrenListTooLong, got {other:?}"),
    }
}

/// A children list exactly at the parser's cap parses (the service's own,
/// lower, 100,000-id limit refuses it later, in the domain). Paired with the
/// test above it pins the cap's comparison to `>`.
#[test]
fn parse_delta_children_list_at_the_parser_cap_parses() {
    let parsed = parse_delta(&reorder_of(PARSER_CHILDREN_CAP)).unwrap_or_else(|e| {
        panic!("a list of exactly {PARSER_CHILDREN_CAP} ids must parse: {e:?}")
    });

    match parsed.struct_commands.as_slice() {
        [StructCommand::ReorderChildren(c)] => {
            assert_eq!(c.children_list.len(), PARSER_CHILDREN_CAP);
        }
        other => panic!("expected one ReorderChildren, got {} commands", other.len()),
    }
}
