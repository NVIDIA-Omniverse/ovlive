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

//! Conversion layer between the cxx bridge types (`crate::bridge::ffi::*`)
//! and the idiomatic Rust mirrors (`crate::types::*`).
//!
//! **Visibility:** this module is `pub(crate)` in production builds; the
//! `internal-bridge` cargo feature re-exposes it as `pub` + `#[doc(hidden)]`
//! for the crate's own integration tests. It is NEVER part of the stable
//! public API. Callers should use the idiomatic types in `crate::types` and
//! go through the Task 3.3 public wrappers instead of invoking `From`/
//! `TryFrom` directly (b.vvs item 7).
//!
//! - `From<idiomatic> for bridge`: always infallible — the idiomatic side is
//!   strongly typed, so every Rust enum variant maps unambiguously onto a
//!   bridge tag + payload.
//! - `TryFrom<bridge> for idiomatic`: fallible because the C++ side produces
//!   bridge values whose `tag` could in principle be out of range (a bridge
//!   bug, not user input). We fail with
//!   `OodfError::UnknownCommandTagAt { index, tag }` rather than panicking,
//!   preserving the Epic 3 invariant that no panic reaches public callers.
//!   The index carried in the error is the position of the bad command
//!   within the `struct_commands` vector, produced by walking the vector
//!   with `enumerate()` during the `OwnedDelta` conversion.
//!
//! The module has no I/O, no async, no `unsafe`, and no `unwrap`/`expect`
//! outside of `#[cfg(test)]` code.

use crate::bridge::ffi;
use crate::error::OodfError;
use crate::types;

// ── Command payloads: idiomatic → bridge ────────────────────────────────────

impl From<types::CommandCreateSection> for ffi::CommandCreateSection {
    fn from(v: types::CommandCreateSection) -> Self {
        ffi::CommandCreateSection {
            parent_id: v.parent_id,
            section_id: v.section_id,
            section_name: v.section_name,
            section_type: v.section_type,
        }
    }
}

impl From<ffi::CommandCreateSection> for types::CommandCreateSection {
    fn from(v: ffi::CommandCreateSection) -> Self {
        types::CommandCreateSection {
            parent_id: v.parent_id,
            section_id: v.section_id,
            section_name: v.section_name,
            section_type: v.section_type,
        }
    }
}

impl From<types::CommandDeleteSection> for ffi::CommandDeleteSection {
    fn from(v: types::CommandDeleteSection) -> Self {
        ffi::CommandDeleteSection {
            parent_id: v.parent_id,
            section_id: v.section_id,
        }
    }
}

impl From<ffi::CommandDeleteSection> for types::CommandDeleteSection {
    fn from(v: ffi::CommandDeleteSection) -> Self {
        types::CommandDeleteSection {
            parent_id: v.parent_id,
            section_id: v.section_id,
        }
    }
}

impl From<types::CommandMoveSection> for ffi::CommandMoveSection {
    fn from(v: types::CommandMoveSection) -> Self {
        ffi::CommandMoveSection {
            old_parent_id: v.old_parent_id,
            new_parent_id: v.new_parent_id,
            section_id: v.section_id,
            new_name: v.new_name,
        }
    }
}

impl From<ffi::CommandMoveSection> for types::CommandMoveSection {
    fn from(v: ffi::CommandMoveSection) -> Self {
        types::CommandMoveSection {
            old_parent_id: v.old_parent_id,
            new_parent_id: v.new_parent_id,
            section_id: v.section_id,
            new_name: v.new_name,
        }
    }
}

impl From<types::CommandReorderChildren> for ffi::CommandReorderChildren {
    fn from(v: types::CommandReorderChildren) -> Self {
        ffi::CommandReorderChildren {
            section_id: v.section_id,
            children_list_id: v.children_list_id,
            children_list: v.children_list,
        }
    }
}

impl From<ffi::CommandReorderChildren> for types::CommandReorderChildren {
    fn from(v: ffi::CommandReorderChildren) -> Self {
        types::CommandReorderChildren {
            section_id: v.section_id,
            children_list_id: v.children_list_id,
            children_list: v.children_list,
        }
    }
}

impl From<types::CommandDiffChangeSection> for ffi::CommandDiffChangeSection {
    fn from(v: types::CommandDiffChangeSection) -> Self {
        ffi::CommandDiffChangeSection {
            parent_id: v.parent_id,
            section_id: v.section_id,
            section_name: v.section_name,
            section_type: v.section_type,
            section_order: v.section_order,
        }
    }
}

impl From<ffi::CommandDiffChangeSection> for types::CommandDiffChangeSection {
    fn from(v: ffi::CommandDiffChangeSection) -> Self {
        types::CommandDiffChangeSection {
            parent_id: v.parent_id,
            section_id: v.section_id,
            section_name: v.section_name,
            section_type: v.section_type,
            section_order: v.section_order,
        }
    }
}

impl From<types::CommandDiffDeleteSection> for ffi::CommandDiffDeleteSection {
    fn from(v: types::CommandDiffDeleteSection) -> Self {
        ffi::CommandDiffDeleteSection {
            section_id: v.section_id,
        }
    }
}

impl From<ffi::CommandDiffDeleteSection> for types::CommandDiffDeleteSection {
    fn from(v: ffi::CommandDiffDeleteSection) -> Self {
        types::CommandDiffDeleteSection {
            section_id: v.section_id,
        }
    }
}

// ── SetField / TimeSample ───────────────────────────────────────────────────

impl From<types::SetField> for ffi::SetField {
    fn from(v: types::SetField) -> Self {
        ffi::SetField {
            section_id: v.section_id,
            key_name: v.key_name,
            value_or_ext_hash: v.value_or_ext_hash,
            ext_value_size: v.ext_value_size,
            ext_value_index: v.ext_value_index,
            set_order: v.set_order,
        }
    }
}

impl From<ffi::SetField> for types::SetField {
    fn from(v: ffi::SetField) -> Self {
        types::SetField {
            section_id: v.section_id,
            key_name: v.key_name,
            value_or_ext_hash: v.value_or_ext_hash,
            ext_value_size: v.ext_value_size,
            ext_value_index: v.ext_value_index,
            set_order: v.set_order,
        }
    }
}

impl From<types::TimeSample> for ffi::TimeSample {
    fn from(v: types::TimeSample) -> Self {
        ffi::TimeSample {
            section_id: v.section_id,
            time: v.time,
            value_or_ext_hash: v.value_or_ext_hash,
            ext_value_size: v.ext_value_size,
            ext_value_index: v.ext_value_index,
        }
    }
}

/// The parse direction canonicalizes the time code (`-0.0` → `+0.0`, see
/// [`types::canonical_sample_time`]): every byte-derived delta crosses this
/// conversion, so no owned delta the parser returns carries `-0.0`. The build
/// direction above writes the time as given.
impl From<ffi::TimeSample> for types::TimeSample {
    fn from(v: ffi::TimeSample) -> Self {
        types::TimeSample {
            section_id: v.section_id,
            time: types::canonical_sample_time(v.time),
            value_or_ext_hash: v.value_or_ext_hash,
            ext_value_size: v.ext_value_size,
            ext_value_index: v.ext_value_index,
        }
    }
}

// ── StructCommand (tagged) ──────────────────────────────────────────────────

impl From<types::StructCommand> for ffi::StructCommand {
    fn from(v: types::StructCommand) -> Self {
        // Every arm sets exactly one payload field and leaves the others as
        // their `Default`. The FFI contract (matching oodf.h) is that the
        // inactive payloads carry no meaning; this is also what the C++ side
        // emits from `parse_delta`.
        let mut out = ffi::StructCommand::default();
        match v {
            types::StructCommand::CreateSection(p) => {
                out.tag = ffi::CommandTag::CreateSection;
                out.create_section = p.into();
            }
            types::StructCommand::DeleteSection(p) => {
                out.tag = ffi::CommandTag::DeleteSection;
                out.delete_section = p.into();
            }
            types::StructCommand::MoveSection(p) => {
                out.tag = ffi::CommandTag::MoveSection;
                out.move_section = p.into();
            }
            types::StructCommand::ReorderChildren(p) => {
                out.tag = ffi::CommandTag::ReorderChildren;
                out.reorder_children = p.into();
            }
            types::StructCommand::DiffChangeSection(p) => {
                out.tag = ffi::CommandTag::DiffChangeSection;
                out.diff_change_section = p.into();
            }
            types::StructCommand::DiffDeleteSection(p) => {
                out.tag = ffi::CommandTag::DiffDeleteSection;
                out.diff_delete_section = p.into();
            }
        }
        out
    }
}

/// Convert a single bridge `StructCommand` at position `index` in its
/// containing vector. Producing the index here means a bad tag surfaces as
/// `UnknownCommandTagAt { index, tag }`, so callers can tell *which* command
/// was malformed without re-walking the vector.
fn struct_command_from_bridge(
    index: usize,
    v: ffi::StructCommand,
) -> Result<types::StructCommand, OodfError> {
    // `CommandTag` is declared `#[repr(u8)]` in bridge.rs. cxx represents
    // shared enums as an opaque struct wrapping a runtime u8 field (`repr:
    // u8`) with named-constant accessors — NOT as a native Rust enum with
    // exhaustively-checked variants. C++ can therefore place any u8 into
    // the `repr` slot, including values outside the declared 0..=6 range.
    // Matching on `v.tag.repr` rather than on the wrapper type forces us to
    // handle every u8, so out-of-range tags surface as
    // `UnknownCommandTagAt` instead of silently selecting a default arm.
    match v.tag.repr {
        r if r == ffi::CommandTag::CreateSection.repr => {
            Ok(types::StructCommand::CreateSection(v.create_section.into()))
        }
        r if r == ffi::CommandTag::DeleteSection.repr => {
            Ok(types::StructCommand::DeleteSection(v.delete_section.into()))
        }
        r if r == ffi::CommandTag::MoveSection.repr => {
            Ok(types::StructCommand::MoveSection(v.move_section.into()))
        }
        r if r == ffi::CommandTag::ReorderChildren.repr => Ok(
            types::StructCommand::ReorderChildren(v.reorder_children.into()),
        ),
        r if r == ffi::CommandTag::DiffChangeSection.repr => Ok(
            types::StructCommand::DiffChangeSection(v.diff_change_section.into()),
        ),
        r if r == ffi::CommandTag::DiffDeleteSection.repr => Ok(
            types::StructCommand::DiffDeleteSection(v.diff_delete_section.into()),
        ),
        other => Err(OodfError::UnknownCommandTagAt { index, tag: other }),
    }
}

impl TryFrom<ffi::StructCommand> for types::StructCommand {
    type Error = OodfError;

    /// Standalone conversion without a containing index — reports the bad tag
    /// as `UnknownCommandTagAt { index: 0, tag }`. Prefer going through
    /// `TryFrom<ffi::OwnedDelta>`, which walks the vector and records the
    /// real index.
    fn try_from(v: ffi::StructCommand) -> Result<Self, Self::Error> {
        struct_command_from_bridge(0, v)
    }
}

// ── OwnedDelta ──────────────────────────────────────────────────────────────

/// Borrowed conversion used by the public `build_delta` (b.vvs item 6).
///
/// The previous owned conversion (`From<types::OwnedDelta>`) required callers
/// to deep-clone the entire `OwnedDelta` (and with it every external-value
/// byte vector) before the FFI could consume it — a wasted copy on top of the
/// unavoidable copy that crosses the FlatBuffers builder. The borrowed form
/// clones the leaf payloads (strings, byte vectors, `Vec<u64>` children) on
/// the way in but skips the wrapping `OwnedDelta` clone. For a 100 MB
/// `SetField.value_or_ext_hash` this halves total bytes copied between the
/// idiomatic and bridge representations on the build path.
///
/// The owned variant is intentionally **not** provided: keeping both forms
/// would let a caller write `let bridge: ffi::OwnedDelta = my_owned.into()`
/// and silently double-clone the entire delta. Callers that hold an owned
/// `OwnedDelta` and no longer need it after the FFI hop should still go
/// through `(&owned).into()` — the bridge type owns its own copy anyway, and
/// the leaf-by-leaf clone is strictly cheaper than the outer-plus-leaf clone
/// an owned `From` would perform.
impl From<&types::OwnedDelta> for ffi::OwnedDelta {
    fn from(v: &types::OwnedDelta) -> Self {
        ffi::OwnedDelta {
            is_diff: v.is_diff,
            base_version: v.base_version,
            struct_commands: v.struct_commands.iter().cloned().map(Into::into).collect(),
            set_fields: v.set_fields.iter().cloned().map(Into::into).collect(),
            time_samples: v.time_samples.iter().cloned().map(Into::into).collect(),
            source_format: v.source_format.clone(),
        }
    }
}

impl TryFrom<ffi::OwnedDelta> for types::OwnedDelta {
    type Error = OodfError;

    fn try_from(v: ffi::OwnedDelta) -> Result<Self, Self::Error> {
        let struct_commands = v
            .struct_commands
            .into_iter()
            .enumerate()
            .map(|(i, sc)| struct_command_from_bridge(i, sc))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(types::OwnedDelta {
            is_diff: v.is_diff,
            base_version: v.base_version,
            struct_commands,
            set_fields: v.set_fields.into_iter().map(Into::into).collect(),
            time_samples: v.time_samples.into_iter().map(Into::into).collect(),
            source_format: v.source_format,
        })
    }
}

#[cfg(test)]
mod tests {
    //! Compile-smoke: the real variant coverage, invalid-tag handling, and
    //! per-direction round-trip tests live in `oodf-ffi/tests/conversion.rs`
    //! (added by Task 3.2.B's Test Writer).

    use super::*;

    #[test]
    fn owned_delta_round_trips_through_bridge_default() {
        let original = types::OwnedDelta::default();
        let bridge: ffi::OwnedDelta = (&original).into();
        let back: types::OwnedDelta = bridge.try_into().expect("default round-trips cleanly");
        assert_eq!(original, back);
    }

    #[test]
    fn invalid_tag_none_surfaces_as_error() {
        let bridge = ffi::StructCommand::default(); // tag = NONE
        let err = types::StructCommand::try_from(bridge).expect_err("NONE must be rejected");
        assert!(matches!(
            err,
            OodfError::UnknownCommandTagAt { index: 0, tag: 0 }
        ));
    }
}
