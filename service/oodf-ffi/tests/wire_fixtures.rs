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

//! Task 5.2 — byte-level regression suite for the Task 5.1 fixture library.
//!
//! Each synthetic fixture under `tests/fixtures/oodf/` + the one real C++
//! fixture at `oodf-ffi/tests/fixtures/postman_chunk.bin` is exercised:
//!
//! 1. `parse_delta(bytes)` → assert Ok + structural expectations.
//! 2. `build_delta(parsed)` → `parse_delta(rebuilt)` → `PartialEq` with the
//!    first parse. Byte-level equality is NOT asserted because FlatBuffers
//!    offset ordering is not guaranteed to be byte-stable across builds.
//!
//! A meta-test walks the fixture directory and fails loudly if a file is
//! present that is not referenced by the named assertion set — this prevents
//! a new fixture from being silently added without a corresponding regression
//! test.

use std::collections::BTreeSet;
use std::path::PathBuf;

use oodf_ffi::{build_delta, parse_delta, OwnedDelta, StructCommand};

// ── Fixture byte handles ────────────────────────────────────────────────────

const POSTMAN_REAL_CPP: &[u8] = include_bytes!("fixtures/postman_chunk.bin");
const DIFF_DELETE_SECTION: &[u8] =
    include_bytes!("../../tests/fixtures/oodf/diff_delete_section.oodf");
const CREATE_AND_DELETE: &[u8] = include_bytes!("../../tests/fixtures/oodf/create_and_delete.oodf");
const MOVE_SECTION: &[u8] = include_bytes!("../../tests/fixtures/oodf/move_section.oodf");
const REORDER_CHILDREN: &[u8] = include_bytes!("../../tests/fixtures/oodf/reorder_children.oodf");
const SET_FIELD_EXTERNAL: &[u8] =
    include_bytes!("../../tests/fixtures/oodf/set_field_external.oodf");
const TIME_SAMPLE_EXTERNAL: &[u8] =
    include_bytes!("../../tests/fixtures/oodf/time_sample_external.oodf");
const MIXED_DELTA: &[u8] = include_bytes!("../../tests/fixtures/oodf/mixed_delta.oodf");
const LARGE_DELTA_1MB: &[u8] = include_bytes!("../../tests/fixtures/oodf/large_delta_1mb.oodf");
// Real-C++-captured fixtures (see bug b.v4v closure). Unlike the synthetic
// set above, these bytes were captured from a deployed C++ reference service
// (tests/fixtures/oodf/CAPTURE.md §3).
const REAL_TIME_SAMPLE: &[u8] = include_bytes!("../../tests/fixtures/oodf/time_sample.oodf");
const REAL_SET_FIELD_LARGE_INLINE: &[u8] =
    include_bytes!("../../tests/fixtures/oodf/set_field_large_inline.oodf");
const REAL_DIFF_AFTER_DELETE: &[u8] =
    include_bytes!("../../tests/fixtures/oodf/diff_after_delete.oodf");

/// The set of fixture file names the meta-test expects to find under
/// `tests/fixtures/oodf/`. Keep in sync with the per-fixture tests below —
/// adding a fixture to the directory without adding it here is a test failure.
const NAMED_SYNTHETIC_FIXTURES: &[&str] = &[
    "diff_delete_section.oodf",
    "create_and_delete.oodf",
    "move_section.oodf",
    "reorder_children.oodf",
    "set_field_external.oodf",
    "time_sample_external.oodf",
    "mixed_delta.oodf",
    "large_delta_1mb.oodf",
    // Real-C++-captured (see tests/fixtures/oodf/README.md → "Fixture
    // provenance"). Shared meta-test treats these identically to the
    // synthetic set — every .oodf file on disk must be named here.
    "time_sample.oodf",
    "set_field_large_inline.oodf",
    "diff_after_delete.oodf",
];

// ── Shared round-trip helper ────────────────────────────────────────────────

/// Parse → build → re-parse. Asserts both parses succeed and the two
/// `OwnedDelta`s are `PartialEq`-equal. Returns the first parse so callers
/// can run further structural assertions without re-parsing.
fn round_trip_semantic(bytes: &[u8], label: &str) -> OwnedDelta {
    let first = parse_delta(bytes).unwrap_or_else(|e| panic!("{label}: first parse failed: {e}"));
    let rebuilt = build_delta(&first)
        .unwrap_or_else(|e| panic!("{label}: build_delta on parsed delta failed: {e}"));
    let second = parse_delta(&rebuilt)
        .unwrap_or_else(|e| panic!("{label}: re-parse after build failed: {e}"));
    assert_eq!(
        first, second,
        "{label}: round-tripped OwnedDelta must be PartialEq-equal to the original parse"
    );
    first
}

// ── Per-fixture byte-level regression tests ─────────────────────────────────

#[test]
fn postman_real_cpp_round_trips_and_has_two_diff_change_sections() {
    // Real C++ fixture — the only non-synthetic buffer in this suite.
    assert_eq!(POSTMAN_REAL_CPP.len(), 280, "Postman fixture is 280 bytes");
    let delta = round_trip_semantic(POSTMAN_REAL_CPP, "postman_chunk.bin");
    assert!(delta.is_diff, "Postman fixture is a diff delta");
    assert_eq!(delta.struct_commands.len(), 2);
    assert_eq!(delta.set_fields.len(), 1);
    assert_eq!(delta.time_samples.len(), 0);
    for cmd in &delta.struct_commands {
        assert!(
            matches!(cmd, StructCommand::DiffChangeSection(_)),
            "Postman fixture struct_commands must all be DiffChangeSection, got {cmd:?}"
        );
    }
    // Postman capture: second DiffChangeSection defines section 5 ("Prim5")
    // and the SetField writes a 64-char inline value under key "field0" on
    // that same section.
    if let StructCommand::DiffChangeSection(c) = &delta.struct_commands[1] {
        assert_eq!(c.section_id, 5);
        assert_eq!(c.section_name, "Prim5");
    }
    assert_eq!(delta.set_fields[0].section_id, 5);
    assert_eq!(delta.set_fields[0].key_name, "field0");
    assert_eq!(delta.set_fields[0].ext_value_size, 0);
}

#[test]
fn diff_delete_section_round_trips() {
    let delta = round_trip_semantic(DIFF_DELETE_SECTION, "diff_delete_section.oodf");
    assert!(delta.is_diff);
    assert_eq!(delta.struct_commands.len(), 1);
    assert_eq!(delta.set_fields.len(), 0);
    assert_eq!(delta.time_samples.len(), 0);
    match &delta.struct_commands[0] {
        StructCommand::DiffDeleteSection(c) => {
            assert_eq!(c.section_id, 42);
        }
        other => panic!("expected DiffDeleteSection, got {other:?}"),
    }
}

#[test]
fn create_and_delete_round_trips() {
    let delta = round_trip_semantic(CREATE_AND_DELETE, "create_and_delete.oodf");
    assert!(!delta.is_diff);
    assert_eq!(delta.struct_commands.len(), 2);
    assert_eq!(delta.source_format, "usda");
    match &delta.struct_commands[0] {
        StructCommand::CreateSection(c) => {
            assert_eq!(c.parent_id, 1);
            assert_eq!(c.section_id, 2);
            assert_eq!(c.section_name, "child");
            assert_eq!(c.section_type, 3);
        }
        other => panic!("expected CreateSection at index 0, got {other:?}"),
    }
    match &delta.struct_commands[1] {
        StructCommand::DeleteSection(c) => {
            assert_eq!(c.parent_id, 1);
            assert_eq!(c.section_id, 2);
        }
        other => panic!("expected DeleteSection at index 1, got {other:?}"),
    }
}

#[test]
fn move_section_round_trips() {
    let delta = round_trip_semantic(MOVE_SECTION, "move_section.oodf");
    assert_eq!(delta.struct_commands.len(), 1);
    match &delta.struct_commands[0] {
        StructCommand::MoveSection(c) => {
            assert_eq!(c.old_parent_id, 10);
            assert_eq!(c.new_parent_id, 20);
            assert_eq!(c.section_id, 30);
            assert_eq!(c.new_name, "moved-node");
        }
        other => panic!("expected MoveSection, got {other:?}"),
    }
}

#[test]
fn reorder_children_round_trips() {
    let delta = round_trip_semantic(REORDER_CHILDREN, "reorder_children.oodf");
    assert_eq!(delta.struct_commands.len(), 1);
    match &delta.struct_commands[0] {
        StructCommand::ReorderChildren(c) => {
            assert_eq!(c.section_id, 100);
            assert_eq!(c.children_list_id, 1);
            assert_eq!(c.children_list, vec![101, 102, 103, 104, 105]);
        }
        other => panic!("expected ReorderChildren, got {other:?}"),
    }
}

#[test]
fn set_field_external_round_trips() {
    let delta = round_trip_semantic(SET_FIELD_EXTERNAL, "set_field_external.oodf");
    assert_eq!(delta.struct_commands.len(), 0);
    assert_eq!(delta.set_fields.len(), 1);
    let f = &delta.set_fields[0];
    assert_eq!(f.section_id, 500);
    assert_eq!(f.key_name, "vertices");
    assert_eq!(f.value_or_ext_hash.len(), 32, "external hash is 32 bytes");
    let expected_hash: Vec<u8> = (0..32u8).collect();
    assert_eq!(f.value_or_ext_hash, expected_hash);
    assert_eq!(f.ext_value_size, 1_048_576);
    assert_eq!(f.ext_value_index, 1);
}

#[test]
fn time_sample_external_round_trips() {
    let delta = round_trip_semantic(TIME_SAMPLE_EXTERNAL, "time_sample_external.oodf");
    assert_eq!(delta.time_samples.len(), 1);
    let ts = &delta.time_samples[0];
    assert_eq!(ts.section_id, 600);
    assert_eq!(ts.time, 0.5);
    assert_eq!(ts.value_or_ext_hash.len(), 32);
    assert_eq!(ts.ext_value_size, 2048);
    assert_eq!(ts.ext_value_index, 2);
}

#[test]
fn mixed_delta_round_trips() {
    let delta = round_trip_semantic(MIXED_DELTA, "mixed_delta.oodf");
    assert!(delta.is_diff);
    assert_eq!(delta.base_version, 7);
    assert_eq!(delta.source_format, "usdc");
    assert_eq!(delta.struct_commands.len(), 4);
    assert_eq!(delta.set_fields.len(), 2);
    assert_eq!(delta.time_samples.len(), 1);

    assert!(matches!(
        delta.struct_commands[0],
        StructCommand::CreateSection(_)
    ));
    assert!(matches!(
        delta.struct_commands[1],
        StructCommand::MoveSection(_)
    ));
    assert!(matches!(
        delta.struct_commands[2],
        StructCommand::ReorderChildren(_)
    ));
    assert!(matches!(
        delta.struct_commands[3],
        StructCommand::DiffDeleteSection(_)
    ));

    // First SetField is inline ("name" = b"hello"); second is external with a
    // 32-byte hash and ext_value_size > 0.
    assert_eq!(delta.set_fields[0].key_name, "name");
    assert_eq!(delta.set_fields[0].ext_value_size, 0);
    assert_eq!(delta.set_fields[0].value_or_ext_hash, b"hello");
    assert_eq!(delta.set_fields[1].key_name, "mesh");
    assert_eq!(delta.set_fields[1].ext_value_size, 65_536);
    assert_eq!(delta.set_fields[1].value_or_ext_hash.len(), 32);

    assert_eq!(delta.time_samples[0].time, 1.25);
}

#[test]
fn large_delta_1mb_round_trips() {
    // >1 MB payload, so this test is heavier than the others — still runs in
    // well under the 5s suite budget per Task 5.2's acceptance criteria.
    assert!(
        LARGE_DELTA_1MB.len() > 1_000_000,
        "large_delta_1mb.oodf must be at least 1 MB to exercise OMPF multi-chunk"
    );
    let delta = round_trip_semantic(LARGE_DELTA_1MB, "large_delta_1mb.oodf");
    assert_eq!(delta.set_fields.len(), 1100);
    assert_eq!(delta.source_format, "usda");
    // Spot-check first, middle, last entry to catch ordering regressions.
    assert_eq!(delta.set_fields[0].section_id, 10_000);
    assert_eq!(delta.set_fields[0].key_name, "bulk_key_0000");
    assert_eq!(delta.set_fields[0].value_or_ext_hash.len(), 1024);
    assert_eq!(delta.set_fields[550].section_id, 10_550);
    assert_eq!(delta.set_fields[1099].section_id, 11_099);
    assert_eq!(delta.set_fields[1099].key_name, "bulk_key_1099");
}

// ── Real-C++-captured fixtures ──────────────────────────────────────────────
//
// These fixtures were captured from a deployed C++ reference service —
// distinct from the synthetic set above, whose bytes come from
// `oodf_ffi::build_delta`. Each test asserts the authentic payload
// round-trips through our parse/build.

#[test]
fn real_time_sample_round_trips() {
    let delta = round_trip_semantic(REAL_TIME_SAMPLE, "time_sample.oodf");
    assert!(delta.is_diff);
    assert_eq!(delta.time_samples.len(), 1, "capture has one TimeSample");
    let ts = &delta.time_samples[0];
    assert_eq!(ts.section_id, 5, "seed attached the time-sample to sid=5");
    assert_eq!(ts.time, 1.5, "seed time was 1.5");
    assert_eq!(
        ts.ext_value_size, 0,
        "emitted inline (server does not promote to external on CreateLayer catchup)"
    );
}

#[test]
fn real_set_field_large_inline_round_trips() {
    let delta = round_trip_semantic(REAL_SET_FIELD_LARGE_INLINE, "set_field_large_inline.oodf");
    assert!(delta.is_diff);
    assert_eq!(delta.set_fields.len(), 1);
    let f = &delta.set_fields[0];
    assert_eq!(f.key_name, "bigfield");
    assert_eq!(
        f.value_or_ext_hash.len(),
        2 * 1024 * 1024,
        "2 MiB inline — the server echoed our seed without promoting to external"
    );
    assert_eq!(f.ext_value_size, 0, "inline: ext_value_size must be 0");
}

#[test]
fn real_diff_after_delete_round_trips() {
    // Captured after seeding 3 child sections, UpdateLayer-deleting sid=10,
    // and re-subscribing from seq=0 (full-state catchup). The emitted delta
    // is a RESET catchup reflecting the post-delete state, not a diff with
    // DiffDeleteSection. SDD §9.1 caveat: subscribers joining after a
    // delete see the current state; DiffDeleteSection only appears to
    // subscribers that were already streaming when the delete occurred.
    let delta = round_trip_semantic(REAL_DIFF_AFTER_DELETE, "diff_after_delete.oodf");
    assert!(
        delta.is_diff,
        "catchup is diff-shaped (reset against base_version=0)"
    );
    assert_eq!(
        delta.struct_commands.len(),
        2,
        "after deleting sid=10, the server retains 2 sections"
    );
    for cmd in &delta.struct_commands {
        assert!(
            matches!(cmd, StructCommand::DiffChangeSection(_)),
            "catchup emits DiffChangeSection for surviving sections, got {cmd:?}"
        );
    }
}

// ── Meta-test: no orphan fixtures ───────────────────────────────────────────

#[test]
fn every_fixture_in_directory_has_a_named_test() {
    // Locate `tests/fixtures/oodf/` relative to this test file's crate
    // manifest. CARGO_MANIFEST_DIR is the `oodf-ffi/` dir; go up one to hit
    // the workspace root.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest
        .parent()
        .expect("oodf-ffi has a workspace parent")
        .join("tests")
        .join("fixtures")
        .join("oodf");

    let mut on_disk: BTreeSet<String> = BTreeSet::new();
    for entry in std::fs::read_dir(&dir).expect("fixture dir must exist") {
        let entry = entry.expect("readable fixture dir entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        // Only track `.oodf` payloads; `.md` and other metadata are allowed.
        if name.ends_with(".oodf") {
            on_disk.insert(name);
        }
    }

    let named: BTreeSet<String> = NAMED_SYNTHETIC_FIXTURES
        .iter()
        .map(|s| s.to_string())
        .collect();

    let orphans: Vec<&String> = on_disk.difference(&named).collect();
    assert!(
        orphans.is_empty(),
        "fixture file(s) present without a named test in wire_fixtures.rs: {orphans:?}. \
         Add an explicit round-trip test and include the file name in NAMED_SYNTHETIC_FIXTURES."
    );

    let missing: Vec<&String> = named.difference(&on_disk).collect();
    assert!(
        missing.is_empty(),
        "NAMED_SYNTHETIC_FIXTURES entries without a backing file on disk: {missing:?}"
    );
}

// ── Meta-test: all fixtures parse ───────────────────────────────────────────

#[test]
fn every_fixture_parses_cleanly() {
    // Belt-and-suspenders companion to the per-fixture tests: even if a
    // structural assertion block is removed by mistake, this test guarantees
    // that every file in the directory still yields Ok from parse_delta.
    for name in NAMED_SYNTHETIC_FIXTURES {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let path = manifest
            .parent()
            .unwrap()
            .join("tests")
            .join("fixtures")
            .join("oodf")
            .join(name);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        parse_delta(&bytes).unwrap_or_else(|e| panic!("parse_delta({name}) failed: {e}"));
    }
}
