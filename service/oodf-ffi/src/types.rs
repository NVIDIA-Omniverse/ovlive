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

//! Idiomatic Rust types mirroring the OODF FlatBuffers schema
//! (`oodf-ffi/schema/OmniObjectDelta.fbs`).
//!
//! These are the types Rust consumers will see after the FFI boundary
//! converts to/from the generated FlatBuffers representation. No `cxx`,
//! no `flatbuffers`, no `unsafe`, no I/O — pure data.

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommandCreateSection {
    pub parent_id: u64,
    pub section_id: u64,
    pub section_name: String,
    pub section_type: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommandDeleteSection {
    pub parent_id: u64,
    pub section_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommandMoveSection {
    pub old_parent_id: u64,
    pub new_parent_id: u64,
    pub section_id: u64,
    pub new_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommandReorderChildren {
    pub section_id: u64,
    pub children_list_id: u8,
    pub children_list: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommandDiffChangeSection {
    pub parent_id: u64,
    pub section_id: u64,
    pub section_name: String,
    pub section_type: u8,
    pub section_order: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommandDiffDeleteSection {
    pub section_id: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructCommand {
    CreateSection(CommandCreateSection),
    DeleteSection(CommandDeleteSection),
    MoveSection(CommandMoveSection),
    ReorderChildren(CommandReorderChildren),
    DiffChangeSection(CommandDiffChangeSection),
    DiffDeleteSection(CommandDiffDeleteSection),
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SetField {
    pub section_id: u64,
    pub key_name: String,
    pub value_or_ext_hash: Vec<u8>,
    pub ext_value_size: u64,
    pub ext_value_index: u32,
    pub set_order: u64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct TimeSample {
    pub section_id: u64,
    /// The sample's USD time code. [`crate::parse_delta`] canonicalizes it
    /// with [`canonical_sample_time`], so a parsed delta never carries `-0.0`:
    /// `-0.0` and `0.0` are one USD time code, and the service keys stored
    /// rows by the time's exact bits. NaN and ±∞ pass through unchanged for
    /// the service's validation to refuse. A value built in Rust is not
    /// canonicalized, and [`crate::build_delta`] writes it as given.
    pub time: f64,
    pub value_or_ext_hash: Vec<u8>,
    pub ext_value_size: u64,
    pub ext_value_index: u32,
}

/// The canonical form of a time-sample time code: `-0.0` becomes `+0.0`, and
/// every other value, NaN and ±∞ included, is returned with its bits
/// unchanged.
///
/// `-0.0` and `0.0` are the same USD time code but differ in their IEEE-754
/// bits, which the service uses to key a stored time-sample row. Mapping both
/// to `+0.0` gives one time code exactly one row.
///
/// This is deliberately not `t + 0.0`: that also turns a signalling NaN
/// quiet, changing bits that must pass through unchanged so validation sees
/// what the client sent.
#[must_use]
#[inline]
pub fn canonical_sample_time(t: f64) -> f64 {
    if t == 0.0 {
        0.0
    } else {
        t
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct OwnedDelta {
    pub is_diff: bool,
    pub base_version: u64,
    pub struct_commands: Vec<StructCommand>,
    pub set_fields: Vec<SetField>,
    pub time_samples: Vec<TimeSample>,
    pub source_format: String,
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<OwnedDelta>();
    assert_send_sync::<StructCommand>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn struct_command_variants_are_distinct() {
        let create = StructCommand::CreateSection(CommandCreateSection {
            parent_id: 1,
            section_id: 2,
            section_name: "a".to_string(),
            section_type: 0,
        });
        let delete = StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 1,
            section_id: 2,
        });
        let move_sec = StructCommand::MoveSection(CommandMoveSection {
            old_parent_id: 1,
            new_parent_id: 3,
            section_id: 2,
            new_name: "b".to_string(),
        });
        let reorder = StructCommand::ReorderChildren(CommandReorderChildren {
            section_id: 2,
            children_list_id: 0,
            children_list: vec![4, 5],
        });
        let diff_change = StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 1,
            section_id: 2,
            section_name: "a".to_string(),
            section_type: 0,
            section_order: 7,
        });
        let diff_delete =
            StructCommand::DiffDeleteSection(CommandDiffDeleteSection { section_id: 2 });

        assert_ne!(create, delete);
        assert_ne!(create, move_sec);
        assert_ne!(create, reorder);
        assert_ne!(create, diff_change);
        assert_ne!(create, diff_delete);
        assert_ne!(delete, move_sec);
        assert_ne!(delete, reorder);
        assert_ne!(delete, diff_change);
        assert_ne!(delete, diff_delete);
        assert_ne!(move_sec, reorder);
        assert_ne!(move_sec, diff_change);
        assert_ne!(move_sec, diff_delete);
        assert_ne!(reorder, diff_change);
        assert_ne!(reorder, diff_delete);
        assert_ne!(diff_change, diff_delete);
    }

    #[test]
    fn owned_delta_default_is_sane() {
        let delta = OwnedDelta::default();
        assert!(!delta.is_diff);
        assert_eq!(delta.base_version, 0);
        assert!(delta.struct_commands.is_empty());
        assert!(delta.set_fields.is_empty());
        assert!(delta.time_samples.is_empty());
        assert!(delta.source_format.is_empty());
    }

    #[test]
    fn all_types_clone_and_compare_equal() {
        let create = CommandCreateSection {
            parent_id: 10,
            section_id: 11,
            section_name: "sec".to_string(),
            section_type: 3,
        };
        assert_eq!(create, create.clone());

        let delete = CommandDeleteSection {
            parent_id: 10,
            section_id: 11,
        };
        assert_eq!(delete, delete.clone());

        let move_sec = CommandMoveSection {
            old_parent_id: 1,
            new_parent_id: 2,
            section_id: 3,
            new_name: "new".to_string(),
        };
        assert_eq!(move_sec, move_sec.clone());

        let reorder = CommandReorderChildren {
            section_id: 4,
            children_list_id: 1,
            children_list: vec![5, 6, 7],
        };
        assert_eq!(reorder, reorder.clone());

        let diff_change = CommandDiffChangeSection {
            parent_id: 8,
            section_id: 9,
            section_name: "d".to_string(),
            section_type: 2,
            section_order: 100,
        };
        assert_eq!(diff_change, diff_change.clone());

        let diff_delete = CommandDiffDeleteSection { section_id: 9 };
        assert_eq!(diff_delete, diff_delete.clone());

        let struct_cmd = StructCommand::CreateSection(create.clone());
        assert_eq!(struct_cmd, struct_cmd.clone());

        let set_field = SetField {
            section_id: 12,
            key_name: "key".to_string(),
            value_or_ext_hash: vec![0xde, 0xad, 0xbe, 0xef],
            ext_value_size: 4,
            ext_value_index: 0,
            set_order: 1,
        };
        assert_eq!(set_field, set_field.clone());

        let time_sample = TimeSample {
            section_id: 13,
            time: 1.5,
            value_or_ext_hash: vec![1, 2, 3],
            ext_value_size: 3,
            ext_value_index: 0,
        };
        assert_eq!(time_sample, time_sample.clone());

        let owned = OwnedDelta {
            is_diff: true,
            base_version: 42,
            struct_commands: vec![struct_cmd.clone()],
            set_fields: vec![set_field.clone()],
            time_samples: vec![time_sample.clone()],
            source_format: "usda".to_string(),
        };
        assert_eq!(owned, owned.clone());
    }
}
