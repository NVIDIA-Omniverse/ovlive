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

//! Task 5.1 — synthetic OODF fixture generator.
//!
//! Deviation note: these fixtures are synthesized via `oodf_ffi::build_delta`
//! because access to a deployed C++ reference service was not available
//! in the environment that generated them. Real C++ capture is
//! deferred — see `tests/fixtures/oodf/README.md` for the TODO. The only
//! real C++ fixture in the repo is `oodf-ffi/tests/fixtures/postman_chunk.bin`
//! (280 bytes).
//!
//! Running `cargo run -p oodf-ffi --example generate_fixtures` from the
//! workspace root rewrites every `.oodf` file under `tests/fixtures/oodf/`.
//! The output is deterministic: identical inputs produce identical bytes
//! (the C++ `FlatBufferBuilder` lays out fields left-to-right in a single
//! pass, so byte output is stable for a given oodf-ffi version).

use std::fs;
use std::path::{Path, PathBuf};

use oodf_ffi::{
    build_delta, CommandCreateSection, CommandDeleteSection, CommandDiffDeleteSection,
    CommandMoveSection, CommandReorderChildren, OwnedDelta, SetField, StructCommand, TimeSample,
};

fn fixtures_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<workspace>/oodf-ffi`. Pop one level and append
    // `tests/fixtures/oodf/`.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("oodf-ffi/ has a parent (workspace root)")
        .join("tests")
        .join("fixtures")
        .join("oodf")
}

fn write_fixture(dir: &Path, name: &str, delta: &OwnedDelta) {
    let bytes = build_delta(delta).expect("fixture deltas are idiomatic, build cannot fail");
    let path = dir.join(name);
    fs::write(&path, &bytes).unwrap_or_else(|e| panic!("failed to write {path:?}: {e}"));
    println!("wrote {} ({} bytes)", path.display(), bytes.len());
}

fn diff_delete_section() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands: vec![StructCommand::DiffDeleteSection(CommandDiffDeleteSection {
            section_id: 42,
        })],
        ..Default::default()
    }
}

fn create_and_delete() -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![
            StructCommand::CreateSection(CommandCreateSection {
                parent_id: 1,
                section_id: 2,
                section_name: "child".to_string(),
                section_type: 3,
            }),
            StructCommand::DeleteSection(CommandDeleteSection {
                parent_id: 1,
                section_id: 2,
            }),
        ],
        source_format: "usda".to_string(),
        ..Default::default()
    }
}

fn move_section() -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::MoveSection(CommandMoveSection {
            old_parent_id: 10,
            new_parent_id: 20,
            section_id: 30,
            new_name: "moved-node".to_string(),
        })],
        ..Default::default()
    }
}

fn reorder_children() -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::ReorderChildren(CommandReorderChildren {
            section_id: 100,
            children_list_id: 1,
            children_list: vec![101, 102, 103, 104, 105],
        })],
        ..Default::default()
    }
}

fn set_field_external() -> OwnedDelta {
    // 32-byte external value content hash (SHA-256 shape). The actual bytes
    // don't need to match a real blob — this fixture exercises the
    // ext_value_size > 0 + 32-byte hash branch.
    let hash: Vec<u8> = (0..32u8).collect();
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        set_fields: vec![SetField {
            section_id: 500,
            key_name: "vertices".to_string(),
            value_or_ext_hash: hash,
            ext_value_size: 1_048_576, // 1 MB nominal external blob
            ext_value_index: 1,
            set_order: 1,
        }],
        ..Default::default()
    }
}

fn time_sample_external() -> OwnedDelta {
    let hash: Vec<u8> = (10..42u8).collect();
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        time_samples: vec![TimeSample {
            section_id: 600,
            time: 0.5,
            value_or_ext_hash: hash,
            ext_value_size: 2048,
            ext_value_index: 2,
        }],
        ..Default::default()
    }
}

fn mixed_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 7,
        struct_commands: vec![
            StructCommand::CreateSection(CommandCreateSection {
                parent_id: 1,
                section_id: 2,
                section_name: "root-child".to_string(),
                section_type: 1,
            }),
            StructCommand::MoveSection(CommandMoveSection {
                old_parent_id: 1,
                new_parent_id: 3,
                section_id: 2,
                new_name: "relocated".to_string(),
            }),
            StructCommand::ReorderChildren(CommandReorderChildren {
                section_id: 3,
                children_list_id: 0,
                children_list: vec![2, 4, 5],
            }),
            StructCommand::DiffDeleteSection(CommandDiffDeleteSection { section_id: 99 }),
        ],
        set_fields: vec![
            SetField {
                section_id: 2,
                key_name: "name".to_string(),
                value_or_ext_hash: b"hello".to_vec(),
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 1,
            },
            SetField {
                section_id: 2,
                key_name: "mesh".to_string(),
                value_or_ext_hash: (0..32u8).collect(),
                ext_value_size: 65_536,
                ext_value_index: 1,
                set_order: 2,
            },
        ],
        time_samples: vec![TimeSample {
            section_id: 2,
            time: 1.25,
            value_or_ext_hash: vec![0xaa, 0xbb, 0xcc, 0xdd],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        source_format: "usdc".to_string(),
    }
}

fn large_delta_1mb() -> OwnedDelta {
    // Produce ~1 MB of SetFields with 1 KB inline values each. With ~1100
    // entries of 1 KB plus per-entry overhead, the resulting FlatBuffers
    // payload comfortably exceeds 1 MB so the SubscribeReadLayer path has to
    // wrap it into multi-chunk OMPF frames.
    const COUNT: u64 = 1100;
    const VALUE_SIZE: usize = 1024;

    let mut set_fields = Vec::with_capacity(COUNT as usize);
    for i in 0..COUNT {
        // Each value is a deterministic 1 KB byte pattern keyed on the index
        // so the fixture remains reproducible across runs.
        let mut value = Vec::with_capacity(VALUE_SIZE);
        for j in 0..VALUE_SIZE {
            value.push(((i as usize + j) % 256) as u8);
        }
        set_fields.push(SetField {
            section_id: 10_000 + i,
            key_name: format!("bulk_key_{i:04}"),
            value_or_ext_hash: value,
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: i,
        });
    }

    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![],
        set_fields,
        time_samples: vec![],
        source_format: "usda".to_string(),
    }
}

fn main() {
    let dir = fixtures_dir();
    fs::create_dir_all(&dir).expect("create tests/fixtures/oodf/");

    write_fixture(&dir, "diff_delete_section.oodf", &diff_delete_section());
    write_fixture(&dir, "create_and_delete.oodf", &create_and_delete());
    write_fixture(&dir, "move_section.oodf", &move_section());
    write_fixture(&dir, "reorder_children.oodf", &reorder_children());
    write_fixture(&dir, "set_field_external.oodf", &set_field_external());
    write_fixture(&dir, "time_sample_external.oodf", &time_sample_external());
    write_fixture(&dir, "mixed_delta.oodf", &mixed_delta());
    write_fixture(&dir, "large_delta_1mb.oodf", &large_delta_1mb());

    println!("Done. {} fixtures written to {}", 8, dir.display());
}
