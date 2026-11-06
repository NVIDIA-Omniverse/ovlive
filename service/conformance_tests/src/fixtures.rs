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

//! The suite's wire fixtures.
//!
//! - **Synthetic** ([`Fixture`]): eight small OMPF payloads built at run time
//!   from their definitions through `oodf_ffi::build_delta`, each wrapped as a
//!   one- or two-part OMPF container and pinned by the SHA-256 of its bytes
//!   ([`Fixture::sha256`]). [`verify`] builds and checks all eight in the
//!   preflight, before any case runs. The first seven pins are the bytes the
//!   retired Python suite sent, so the port sends the same payloads;
//!   `sibling_names` was added after the port.
//! - **Captured** ([`capture`]): Kit client traffic, a byte copy of
//!   `tools/fixtures/gze_inbound_deltas.json`, and **golden**
//!   ([`capture_golden`]): the C++ reference catch-up signature, a byte copy
//!   of `tools/fixtures/gze_cpp_catchup_oracle.json`. Both are embedded in the
//!   binary from `conformance_tests/fixtures/`.
//! - **Build manifest** ([`build_manifest`]): the contract name and the
//!   SHA-256 of the public proto, the OODF schema and the captured and golden
//!   files, recorded in `summary.json`'s `build` key
//!   (`conformance_tests/fixtures/build.json`). The synthetic fixtures are
//!   pinned by their own hashes above, not here.
//!
//! Every fixture is NVIDIA data: synthetic, or a capture approved for
//! publication. None needs a third-party license.

use std::sync::OnceLock;

use base64::Engine as _;
use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection, CommandMoveSection,
    CommandReorderChildren, OwnedDelta, SetField, StructCommand, TimeSample,
};
use sha2::{Digest, Sha256};

use crate::check::{CaseError, CaseResult};
use crate::protocol::{content_digest, Signature};

/// The captured Kit uploads, embedded.
const CAPTURED_JSON: &str = include_str!("../fixtures/captured.json");
/// The C++ reference catch-up result, embedded.
const CAPTURED_GOLDEN_JSON: &str = include_str!("../fixtures/captured-golden.json");
/// The build manifest, embedded.
const BUILD_JSON: &str = include_str!("../fixtures/build.json");

/// The length of the `external` fixture's one external value.
const EXTERNAL_VALUE_BYTES: usize = 8192;
/// The byte the `external` fixture's external value repeats: `C`.
const EXTERNAL_VALUE_BYTE: u8 = 0x43;
/// The length of the `large` fixture's inline value: 2 MiB.
const LARGE_VALUE_BYTES: usize = 2 * 1024 * 1024;
/// The byte the `large` fixture's inline value repeats: `B`.
const LARGE_VALUE_BYTE: u8 = 0x42;

/// A synthetic fixture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Fixture {
    /// Sections 1 and 5 (`probe`), field `value` = `initial`.
    Initial,
    /// A non-diff update setting `value` = `updated`.
    Update,
    /// Sections 1 and 5 with a time sample at 1.5 = `animated`.
    TimeSample,
    /// A non-diff update setting the time sample at 1.5 = `animated`.
    UpdateTime,
    /// Sections 1 and 5, `value` an 8192-byte external value of `0x43`.
    External,
    /// Sections 1 and 5, `value` a 2 MiB inline value of `0x42`.
    Large,
    /// A structural update: create, move, delete and reorder sections.
    Structure,
    /// A structural update walking the sibling-name rules under a new prim
    /// `P` (10): a prim `x` (11) and an attribute `x` (12) created side by
    /// side, a relationship `x` (13) created onto the attribute, a prim `y`
    /// (14) with an attribute child (15) created, then the prim `x` renamed
    /// `y`, and renamed `y` again.
    SiblingNames,
}

impl Fixture {
    /// All eight, in a fixed order.
    pub const ALL: [Fixture; 8] = [
        Fixture::Initial,
        Fixture::Update,
        Fixture::TimeSample,
        Fixture::UpdateTime,
        Fixture::External,
        Fixture::Large,
        Fixture::Structure,
        Fixture::SiblingNames,
    ];

    /// The fixture's name: `initial`, `update`, `time_sample`, `update_time`,
    /// `external`, `large`, `structure` or `sibling_names`.
    pub fn name(self) -> &'static str {
        match self {
            Fixture::Initial => "initial",
            Fixture::Update => "update",
            Fixture::TimeSample => "time_sample",
            Fixture::UpdateTime => "update_time",
            Fixture::External => "external",
            Fixture::Large => "large",
            Fixture::Structure => "structure",
            Fixture::SiblingNames => "sibling_names",
        }
    }

    /// The pinned lowercase-hex SHA-256 of the fixture's OMPF bytes.
    pub fn sha256(self) -> &'static str {
        match self {
            Fixture::Initial => "bcd78884d600c00aff1654db3bb6b1bbafa1b2694a00c4391b7fb0efa6040a7d",
            Fixture::Update => "3c72bfbb07377caa1945bbe444c78f6a06eb2736fd887e65c0f00ef38d401eb1",
            Fixture::TimeSample => {
                "026c2702c67c1895b3c6cb107143575b9fa2b2f176bc14afb124aeb6832eda52"
            }
            Fixture::UpdateTime => {
                "dac2cefd66868e965b901a9b0310f818a37782060615cc2c868abf016cfff382"
            }
            Fixture::External => "2f17e61fc125a70d6e61de2f006901ee7e4da75a0acb9d6a9286a37f4ff6ffc0",
            Fixture::Large => "f53b666fae7d6ad668c2c0be3bbef8d361592f0579b06156b873755725c31545",
            Fixture::Structure => {
                "699ee102c678c132acab86d8eda8d1f84574e9055526cc83acf890c140a715f9"
            }
            Fixture::SiblingNames => {
                "43dfabc37f9a4e60edf3cdd9a6aae785edc32543d5965101ad473a4a4e889a2c"
            }
        }
    }

    /// The fixture's position in [`Fixture::ALL`].
    fn index(self) -> usize {
        Fixture::ALL
            .iter()
            .position(|fixture| *fixture == self)
            .expect("every fixture is in ALL")
    }

    /// The fixture's OODF delta, the definition the retired Python suite's
    /// fixtures were generated from.
    fn delta(self) -> OwnedDelta {
        let mut delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: vec![section(1, 0, ""), section(5, 1, "probe")],
            source_format: "usda".into(),
            ..Default::default()
        };
        match self {
            Fixture::Initial | Fixture::Large | Fixture::External => {
                let (value, size) = match self {
                    Fixture::Large => (vec![LARGE_VALUE_BYTE; LARGE_VALUE_BYTES], 0),
                    Fixture::External => (
                        content_digest(&external_value()).to_vec(),
                        EXTERNAL_VALUE_BYTES as u64,
                    ),
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
            Fixture::TimeSample => delta.time_samples.push(TimeSample {
                section_id: 5,
                time: 1.5,
                value_or_ext_hash: b"animated".to_vec(),
                ..Default::default()
            }),
            Fixture::Structure => {
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
            Fixture::SiblingNames => {
                delta.is_diff = false;
                delta.base_version = 1;
                let create = |parent_id: u64, section_id: u64, name: &str, section_type: u8| {
                    StructCommand::CreateSection(CommandCreateSection {
                        parent_id,
                        section_id,
                        section_name: name.into(),
                        section_type,
                    })
                };
                let rename = |section_id: u64, new_name: &str| {
                    StructCommand::MoveSection(CommandMoveSection {
                        old_parent_id: 10,
                        new_parent_id: 10,
                        section_id,
                        new_name: new_name.into(),
                    })
                };
                // `SdfSpecType`s: attribute 1, prim 6, relationship 8.
                delta.struct_commands = vec![
                    create(1, 10, "P", 6),
                    create(10, 11, "x", 6),
                    create(10, 12, "x", 1),
                    create(10, 13, "x", 8),
                    create(10, 14, "y", 6),
                    create(14, 15, "c", 1),
                    rename(11, "y"),
                    rename(11, "y"),
                ];
            }
            Fixture::Update | Fixture::UpdateTime => {
                delta.is_diff = false;
                delta.base_version = 1;
                delta.struct_commands.clear();
                if self == Fixture::Update {
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
        }
        delta
    }

    /// Builds the fixture's OMPF bytes: the delta through
    /// `oodf_ffi::build_delta` as part 0, then (for `external`) its external
    /// value as part 1, in the layout the pinned hashes were taken over.
    fn build(self) -> Result<Vec<u8>, String> {
        let delta = oodf_ffi::build_delta(&self.delta())
            .map_err(|error| format!("building synthetic fixture {}: {error}", self.name()))?;
        let mut parts = vec![delta];
        if self == Fixture::External {
            parts.push(external_value());
        }
        Ok(ompf(&parts))
    }
}

/// A `DiffChangeSection` of section `id` under `parent` named `name`.
fn section(id: u64, parent: u64, name: &str) -> StructCommand {
    StructCommand::DiffChangeSection(CommandDiffChangeSection {
        section_id: id,
        parent_id: parent,
        section_name: name.into(),
        section_type: 0,
        section_order: 0,
    })
}

/// The `external` fixture's external value.
fn external_value() -> Vec<u8> {
    vec![EXTERNAL_VALUE_BYTE; EXTERNAL_VALUE_BYTES]
}

/// An OMPF container of `parts`: the magic, the part count, each part's
/// offset (the first just past the offset table), then the parts in order.
fn ompf(parts: &[Vec<u8>]) -> Vec<u8> {
    let table_end = 8 + 8 * parts.len();
    let mut out = Vec::with_capacity(table_end + parts.iter().map(Vec::len).sum::<usize>());
    out.extend_from_slice(b"OMPF");
    out.extend_from_slice(&u32::try_from(parts.len()).expect("few parts").to_le_bytes());
    let mut offset = table_end as u64;
    for part in parts {
        out.extend_from_slice(&offset.to_le_bytes());
        offset += part.len() as u64;
    }
    for part in parts {
        out.extend_from_slice(part);
    }
    out
}

/// Lowercase-hex SHA-256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Each fixture's built bytes, or why building them failed, by
/// [`Fixture::index`]: built once per process.
static BUILT: [OnceLock<Result<Vec<u8>, String>>; 8] = [const { OnceLock::new() }; 8];

/// The fixture's built bytes, building them on first use.
fn built(fixture: Fixture) -> &'static Result<Vec<u8>, String> {
    BUILT[fixture.index()].get_or_init(|| fixture.build())
}

/// Builds every synthetic fixture and checks each against its pinned hash.
/// The error names the first fixture that does not match.
pub fn verify() -> Result<(), String> {
    for fixture in Fixture::ALL {
        let bytes = built(fixture).as_ref().map_err(Clone::clone)?;
        let actual = sha256_hex(bytes);
        if actual != fixture.sha256() {
            return Err(format!(
                "synthetic fixture {} does not match its pinned SHA-256 {} (built {actual})",
                fixture.name(),
                fixture.sha256()
            ));
        }
    }
    Ok(())
}

/// The OMPF bytes of `fixture`, built once per process. The preflight's
/// [`verify`] has already checked them, so this does not fail during a run.
pub fn fixture(fixture: Fixture) -> &'static [u8] {
    match built(fixture) {
        Ok(bytes) => bytes,
        // Unreachable after a passing preflight: `verify` built this fixture.
        Err(error) => panic!("{error}"),
    }
}

/// One captured client upload.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct CapturedRow {
    /// When it was captured (`HH:MM:SS.mmm`).
    pub ts: String,
    /// `CreateLayer` or `UpdateLayer`.
    pub kind: String,
    /// The payload's length in bytes.
    pub len: u64,
    /// The OMPF payload, base64.
    pub b64: String,
}

impl CapturedRow {
    /// The decoded OMPF payload.
    pub fn payload(&self) -> CaseResult<Vec<u8>> {
        base64::engine::general_purpose::STANDARD
            .decode(&self.b64)
            .map_err(|error| {
                CaseError::new(format!(
                    "captured row at {} is not base64: {error}",
                    self.ts
                ))
            })
    }
}

/// The captured Kit client uploads, in capture order.
pub fn capture() -> Vec<CapturedRow> {
    serde_json::from_str(CAPTURED_JSON).expect("the embedded captured.json is a list of rows")
}

/// The C++ reference implementation's catch-up signature after the captured
/// replay (`normalized_signature` of the golden file).
pub fn capture_golden() -> Signature {
    #[derive(serde::Deserialize)]
    struct Golden {
        normalized_signature: Signature,
    }
    serde_json::from_str::<Golden>(CAPTURED_GOLDEN_JSON)
        .expect("the embedded captured-golden.json carries a normalized_signature")
        .normalized_signature
}

/// The embedded build manifest: the contract name and the SHA-256 of the
/// inputs the fixtures were made from, as `summary.json`'s `build` value.
pub fn build_manifest() -> serde_json::Value {
    serde_json::from_str(BUILD_JSON).expect("the embedded build.json is JSON")
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::protocol::{decode, unpack_ompf};

    // The inputs `build.json` and the embedded copies are pinned against,
    // embedded at compile time from the repository.
    const PROTO: &[u8] =
        include_bytes!("../../proto/nvidia/omniverse/live/v1alpha/live_service.proto");
    const OODF_SCHEMA: &[u8] = include_bytes!("../../oodf-ffi/schema/OmniObjectDelta.fbs");
    const TOOLS_CAPTURED: &[u8] = include_bytes!("../../tools/fixtures/gze_inbound_deltas.json");
    const TOOLS_GOLDEN: &[u8] = include_bytes!("../../tools/fixtures/gze_cpp_catchup_oracle.json");

    #[test]
    fn verify_every_synthetic_fixture_matches_its_pinned_sha256() {
        assert_eq!(verify(), Ok(()));
        for fixture in Fixture::ALL {
            assert_eq!(
                sha256_hex(super::fixture(fixture)),
                fixture.sha256(),
                "{}",
                fixture.name()
            );
        }
    }

    #[test]
    fn fixture_pins_are_eight_distinct_hashes() {
        let pins: BTreeSet<&str> = Fixture::ALL.iter().map(|f| f.sha256()).collect();
        assert_eq!(pins.len(), 8);
        let names: BTreeSet<&str> = Fixture::ALL.iter().map(|f| f.name()).collect();
        assert_eq!(names.len(), 8);
    }

    #[test]
    fn every_synthetic_fixture_is_an_ompf_container_that_decodes() {
        for fixture in Fixture::ALL {
            let parts = unpack_ompf(super::fixture(fixture)).unwrap();
            assert!(!parts[0].is_empty(), "{}", fixture.name());
            let expected_parts = if fixture == Fixture::External { 2 } else { 1 };
            assert_eq!(parts.len(), expected_parts, "{}", fixture.name());
            assert!(
                decode(super::fixture(fixture)).is_ok(),
                "{}",
                fixture.name()
            );
        }
    }

    #[test]
    fn build_manifest_hashes_equal_the_files_they_name() {
        let manifest = build_manifest();
        assert_eq!(
            manifest["contract"],
            "nvidia.omniverse.live.v1alpha.LiveService"
        );
        let hashes = manifest["sha256"].as_object().expect("a sha256 object");
        // The synthetic fixtures are pinned by their own hashes, not here.
        assert_eq!(
            hashes.keys().map(String::as_str).collect::<BTreeSet<_>>(),
            BTreeSet::from(["captured", "captured_golden", "oodf_schema", "proto"])
        );
        for (key, bytes) in [
            ("proto", PROTO),
            ("oodf_schema", OODF_SCHEMA),
            ("captured", CAPTURED_JSON.as_bytes()),
            ("captured_golden", CAPTURED_GOLDEN_JSON.as_bytes()),
        ] {
            assert_eq!(hashes[key], sha256_hex(bytes), "{key}");
        }
    }

    #[test]
    fn captured_and_golden_copies_are_byte_equal_to_tools_fixtures() {
        assert!(CAPTURED_JSON.as_bytes() == TOOLS_CAPTURED);
        assert!(CAPTURED_GOLDEN_JSON.as_bytes() == TOOLS_GOLDEN);
    }

    #[test]
    fn capture_rows_decode_to_their_recorded_lengths() {
        let rows = capture();
        assert!(!rows.is_empty());
        for row in &rows {
            assert!(
                row.kind == "CreateLayer" || row.kind == "UpdateLayer",
                "{}",
                row.kind
            );
            assert_eq!(row.payload().unwrap().len() as u64, row.len, "{}", row.ts);
        }
    }

    #[test]
    fn capture_golden_is_a_catch_up_signature() {
        let golden = capture_golden();
        assert!(golden.is_diff);
        assert!(!golden.set_fields.is_empty());
    }
}
