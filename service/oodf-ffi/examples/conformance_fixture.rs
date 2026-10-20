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

//! Emit deterministic, synthetic protocol fixtures as raw OODF.
//! No server-domain code participates in fixture construction.

use std::io::Write;

use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection, CommandMoveSection,
    CommandReorderChildren, OwnedDelta, SetField, StructCommand, TimeSample,
};
use sha2::{Digest, Sha256};

/// Block size of the external-value content hash: a fixed protocol constant.
const CONTENT_HASH_BLOCK_BYTES: usize = 1 << 20;

/// The content hash an external value's `valueOrExtHash` must carry: SHA-256
/// over the concatenated SHA-256 digests of the value's consecutive 1 MiB
/// blocks (an empty value hashes to SHA-256 of the empty string). This is the
/// clients' definition, not plain SHA-256 of the bytes. Written out here rather
/// than imported, so the fixtures stay independent of the service code they
/// test.
fn content_hash(bytes: &[u8]) -> [u8; 32] {
    let mut outer = Sha256::new();
    for block in bytes.chunks(CONTENT_HASH_BLOCK_BYTES) {
        outer.update(Sha256::digest(block));
    }
    outer.finalize().into()
}

fn section(id: u64, parent: u64, name: &str) -> StructCommand {
    StructCommand::DiffChangeSection(CommandDiffChangeSection {
        section_id: id,
        parent_id: parent,
        section_name: name.into(),
        section_type: 0,
        section_order: 0,
    })
}

fn fixture(name: &str) -> Result<OwnedDelta, String> {
    let mut delta = OwnedDelta {
        is_diff: true,
        base_version: 0,
        struct_commands: vec![section(1, 0, ""), section(5, 1, "probe")],
        source_format: "usda".into(),
        ..Default::default()
    };
    match name {
        "initial" | "large" | "external" => {
            let (value, size) = match name {
                "large" => (vec![0x42; 2 * 1024 * 1024], 0),
                "external" => (content_hash(&[0x43; 8192]).to_vec(), 8192),
                _ => (b"initial".to_vec(), 0),
            };
            delta.set_fields.push(SetField {
                section_id: 5,
                key_name: "value".into(),
                value_or_ext_hash: value,
                ext_value_size: size,
                ext_value_index: if size == 0 { 0 } else { 1 },
                set_order: 0,
            });
        }
        "time_sample" => delta.time_samples.push(TimeSample {
            section_id: 5,
            time: 1.5,
            value_or_ext_hash: b"animated".to_vec(),
            ..Default::default()
        }),
        "structure" => {
            delta.is_diff = false;
            delta.base_version = 1;
            delta.struct_commands = vec![
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 1,
                    section_id: 6,
                    section_name: "second".into(),
                    section_type: 0,
                }),
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 5,
                    section_id: 7,
                    section_name: "child".into(),
                    section_type: 0,
                }),
                StructCommand::MoveSection(CommandMoveSection {
                    old_parent_id: 5,
                    new_parent_id: 6,
                    section_id: 7,
                    new_name: "moved".into(),
                }),
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 1,
                    section_id: 8,
                    section_name: "removed".into(),
                    section_type: 0,
                }),
                StructCommand::DeleteSection(CommandDeleteSection {
                    parent_id: 1,
                    section_id: 8,
                }),
                StructCommand::ReorderChildren(CommandReorderChildren {
                    section_id: 1,
                    children_list_id: 0,
                    children_list: vec![6, 5],
                }),
            ];
        }
        "update" | "update_time" => {
            delta.is_diff = false;
            delta.base_version = 1;
            delta.struct_commands.clear();
            if name == "update" {
                delta.set_fields.push(SetField {
                    section_id: 5,
                    key_name: "value".into(),
                    value_or_ext_hash: b"updated".to_vec(),
                    ..Default::default()
                });
            } else {
                delta.time_samples.push(TimeSample {
                    section_id: 5,
                    time: 1.5,
                    value_or_ext_hash: b"animated".to_vec(),
                    ..Default::default()
                });
            }
        }
        _ => return Err(format!("unknown fixture: {name}")),
    }
    Ok(delta)
}

fn main() {
    let name = std::env::args().nth(1).unwrap_or_default();
    let delta = fixture(&name).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    std::io::stdout()
        .write_all(&oodf_ffi::build_delta(&delta).expect("valid fixture"))
        .expect("stdout write");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_roundtrip_through_wire_codec() {
        for name in [
            "initial",
            "large",
            "external",
            "time_sample",
            "update",
            "update_time",
            "structure",
        ] {
            let delta = fixture(name).unwrap();
            let bytes = oodf_ffi::build_delta(&delta).unwrap();
            assert_eq!(oodf_ffi::parse_delta(&bytes).unwrap(), delta, "{name}");
        }
        assert!(fixture("misspelled").is_err());
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The local content hash agrees with the clients' own vectors, so the
    /// `external` fixture's claim is one the service admits.
    #[test]
    fn content_hash_matches_the_clients_vectors() {
        // The C++ client library's test vector: one byte of 0x00.
        assert_eq!(
            hex(&content_hash(&[0x00])),
            "1406e05881e299367766d313e26c05564ec91bf721d31726bd6e46e60689539a"
        );
        // The Rust client library's test vector: 1 MiB + 1 bytes of `i % 251`,
        // which spans two blocks.
        let two_blocks: Vec<u8> = (0..(1 << 20) + 1).map(|i| (i % 251) as u8).collect();
        assert_eq!(
            hex(&content_hash(&two_blocks)),
            "a9c574ce937d2371daf87cdd0e75396b096d7f74a66c764e4c656bd12b2e7bf9"
        );
    }

    /// The `external` fixture claims the content hash of its 8192 bytes of
    /// 0x43, not their plain SHA-256.
    #[test]
    fn external_fixture_claims_the_content_hash_of_its_value() {
        let delta = fixture("external").unwrap();
        assert_eq!(
            hex(&delta.set_fields[0].value_or_ext_hash),
            "368a4df03f4e059ae6f4392ba381e03c261d21f064fa1caa48e72e6b14c9e84d"
        );
        assert_ne!(
            delta.set_fields[0].value_or_ext_hash,
            Sha256::digest([0x43; 8192]).to_vec(),
            "plain SHA-256 is not the content hash"
        );
    }
}
