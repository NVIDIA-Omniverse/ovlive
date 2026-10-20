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

//! Acceptance-criterion 4 regression test for b.vvs item 7.
//!
//! Asserts that the `oodf-ffi` cxx bridge module is not nameable from a
//! `live-edit` `use` statement. The actual visibility check happens at
//! compile time — if a future change accidentally re-exposes
//! `oodf_ffi::bridge`, the doctest in the comment below would compile and
//! this file's regression contract would silently weaken.
//!
//! Because Rust integration tests cannot easily express a "this MUST NOT
//! compile" assertion without trybuild (an extra heavyweight dev-dep), the
//! check here is the lightweight version: import only the public surface
//! and confirm that surface is non-empty. The companion compile-fail
//! demonstration lives in the comment block.
//!
//! Manual compile-fail reproduction:
//!
//! ```text
//! // Drop the following into any live-edit .rs file and run `cargo check`:
//! //
//! //   use oodf_ffi::bridge::ffi;
//! //
//! // Expected error:
//! //   error[E0603]: module `bridge` is private
//! //     |
//! //     |     pub(crate) mod bridge;
//! //     |     ^^^^^^^^^^^^^^^^^^^^^
//! ```
//!
//! The `cargo check --workspace` build path does not enable
//! `oodf-ffi/internal-bridge`, so `oodf_ffi::bridge` resolves to
//! `pub(crate) mod bridge` and the import above fails to compile from
//! anywhere outside the `oodf-ffi` crate itself.

// Importing only the public surface succeeds — these are the types
// live-edit code is allowed to name. If a future refactor breaks any of
// these public re-exports, this test stops compiling and the regression
// is caught at CI time.
use oodf_ffi::{
    build_delta, parse_delta, CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection,
    CommandDiffDeleteSection, CommandMoveSection, CommandReorderChildren, OodfError, OwnedDelta,
    SetField, StructCommand, TimeSample,
};

#[test]
fn public_surface_round_trips() {
    // Mirror of the test in `oodf-ffi/src/lib.rs` — restated here through
    // the workspace-consumer dependency view to prove the public API is
    // complete enough to round-trip a delta without naming `bridge`.
    let original = OwnedDelta {
        is_diff: false,
        base_version: 7,
        struct_commands: vec![
            StructCommand::CreateSection(CommandCreateSection {
                parent_id: 1,
                section_id: 2,
                section_name: "sec".into(),
                section_type: 3,
            }),
            StructCommand::DeleteSection(CommandDeleteSection {
                parent_id: 1,
                section_id: 2,
            }),
            StructCommand::MoveSection(CommandMoveSection {
                old_parent_id: 1,
                new_parent_id: 4,
                section_id: 2,
                new_name: "new".into(),
            }),
            StructCommand::ReorderChildren(CommandReorderChildren {
                section_id: 4,
                children_list_id: 1,
                children_list: vec![10, 20, 30],
            }),
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id: 1,
                section_id: 5,
                section_name: "ch".into(),
                section_type: 0,
                section_order: 7,
            }),
            StructCommand::DiffDeleteSection(CommandDiffDeleteSection { section_id: 5 }),
        ],
        set_fields: vec![SetField {
            section_id: 2,
            key_name: "k".into(),
            value_or_ext_hash: vec![1, 2, 3, 4],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        time_samples: vec![TimeSample {
            section_id: 2,
            time: 1.5,
            value_or_ext_hash: vec![0xAB],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        source_format: "usda".into(),
    };
    let bytes: Vec<u8> =
        build_delta(&original).expect("build_delta cannot fail on idiomatic input");
    let parsed: OwnedDelta = parse_delta(&bytes).expect("round-trip parse");
    assert_eq!(parsed, original);
}

#[test]
fn oodf_error_implements_std_error() {
    // OodfError must be usable with `?` from anywhere live-edit links a
    // ServiceError chain. This requires the trait-bound surface to be
    // public even when the bridge module is private.
    fn assert_error<E: std::error::Error + Send + Sync + 'static>() {}
    assert_error::<OodfError>();
}

// Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
#[test]
fn oodf_ffi_lib_rs_gates_bridge_module_behind_internal_bridge_feature() {
    // Source-level pin (b.vvs item 7 / AC4) on the visibility contract.
    //
    // The positive-import test above proves the *public* API surface is
    // complete enough to round-trip without naming `bridge`. The
    // companion negative property — "`oodf_ffi::bridge` is NOT nameable
    // from a `live-edit` `use` statement" — would normally need
    // `trybuild` to verify at compile time, which is a heavyweight
    // dev-dep for a single regression. Instead we pin the load-bearing
    // declaration shape in `oodf-ffi/src/lib.rs` directly: a regression
    // that re-exposes `pub mod bridge` unconditionally (the pre-fix
    // shape) would change the source string in an obvious way and trip
    // these asserts.
    //
    // Reading `oodf-ffi/src/lib.rs` via a relative include from this
    // test file requires the file to live at a stable path; we use the
    // workspace-relative path resolved via `CARGO_MANIFEST_DIR`. If the
    // file is ever moved, this test fires loudly rather than silently
    // missing the pin.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let lib_rs_path = format!("{manifest_dir}/oodf-ffi/src/lib.rs");
    let source = std::fs::read_to_string(&lib_rs_path)
        .unwrap_or_else(|e| panic!("could not read {lib_rs_path}: {e}"));

    // Load-bearing assertions: the gating must use the cfg pattern
    // `#[cfg(not(feature = "internal-bridge"))]` followed by
    // `pub(crate) mod bridge;`. A regression that drops the cfg or
    // changes `pub(crate)` back to `pub` is what AC4 must catch.
    assert!(
        source.contains(r#"#[cfg(not(feature = "internal-bridge"))]"#),
        "b.vvs item 7 regression: oodf-ffi/src/lib.rs must gate `bridge` visibility \
         on `#[cfg(not(feature = \"internal-bridge\"))]`; gating gone in:\n{source}"
    );
    assert!(
        source.contains("pub(crate) mod bridge;"),
        "b.vvs item 7 regression: oodf-ffi/src/lib.rs must declare `bridge` as \
         `pub(crate) mod bridge;` in the not-internal-bridge cfg branch — a \
         regression that flips it back to `pub mod bridge;` unconditionally would \
         re-expose cxx types and `extend_byte_vec` as part of the stable public API. \
         Source:\n{source}"
    );
    // The matching `convert` module carries the same contract (cxx
    // conversions must not leak either) — pin it too.
    assert!(
        source.contains("pub(crate) mod convert;"),
        "b.vvs item 7: oodf-ffi/src/lib.rs must gate `convert` as `pub(crate)` in \
         the default-features cfg branch; regression in:\n{source}"
    );
}
