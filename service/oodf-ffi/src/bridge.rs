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

//! cxx bridge between Rust and the C++ OODF parse/build implementation.
//!
//! **Visibility:** this module is `pub(crate)` by default — production consumers
//! (e.g. the `live-edit` parent crate) MUST go through `crate::types` and the
//! public wrappers in `crate::lib` and cannot import bridge types directly.
//! The `internal-bridge` cargo feature re-exposes it as `pub` so the
//! integration tests under `oodf-ffi/tests/` can construct bridge values and
//! poke at FFI representations — that feature is intentionally NOT a default
//! feature, so a downstream `[dependencies] oodf-ffi = { path = "…" }` cannot
//! accidentally enable it (b.vvs item 7).
//!
//! # Shared-type naming
//!
//! cxx shared structs require matching names on both sides of the bridge.
//! The C++ side defines `oodf::OwnedDelta`, `oodf::StructCommand`,
//! `oodf::CommandTag`, `oodf::ParseErrorKind`, `oodf::ParseResult`, etc.
//! (see `oodf-ffi/include/oodf.h`), so the Rust bridge types reuse those
//! exact names inside this `ffi` submodule. The idiomatic Rust mirrors live
//! in `crate::types` with the same short names (`OwnedDelta`,
//! `StructCommand`, …); the conversion layer in `crate::convert` disambiguates
//! by referring to the bridge types as `crate::bridge::ffi::OwnedDelta` and
//! the idiomatic types as `crate::types::OwnedDelta`.
//!
//! # Shared structs and `oodf.h`
//!
//! These types are declared here as cxx *shared* structs — cxx emits their
//! C++ definitions into `bridge.rs.h` in namespace `oodf`. Epic 2's
//! `oodf-ffi/include/oodf.h` therefore no longer hand-defines them; it only
//! includes the generated header and re-exports a namespace alias so the
//! public include path (`#include "oodf.h"`) stays stable for existing Epic
//! 2 tests and `oodf.cpp`.

#[cxx::bridge(namespace = "oodf")]
pub mod ffi {
    // ── Command-payload shared structs ──────────────────────────────────────
    // Field names and types mirror `oodf::CommandCreateSection` as originally
    // hand-written in oodf.h (now deleted — cxx owns the C++ definition).

    #[derive(Default)]
    pub struct CommandCreateSection {
        pub parent_id: u64,
        pub section_id: u64,
        pub section_name: String,
        pub section_type: u8,
    }

    #[derive(Default)]
    pub struct CommandDeleteSection {
        pub parent_id: u64,
        pub section_id: u64,
    }

    #[derive(Default)]
    pub struct CommandMoveSection {
        pub old_parent_id: u64,
        pub new_parent_id: u64,
        pub section_id: u64,
        pub new_name: String,
    }

    #[derive(Default)]
    pub struct CommandReorderChildren {
        pub section_id: u64,
        pub children_list_id: u8,
        pub children_list: Vec<u64>,
    }

    #[derive(Default)]
    pub struct CommandDiffChangeSection {
        pub parent_id: u64,
        pub section_id: u64,
        pub section_name: String,
        pub section_type: u8,
        pub section_order: u32,
    }

    #[derive(Default)]
    pub struct CommandDiffDeleteSection {
        pub section_id: u64,
    }

    // ── Discriminant for `StructCommand` ────────────────────────────────────
    // Numeric values MUST stay in sync with Epic 2's reference layout — the
    // tags here are the exact `CommandTag` values that oodf.cpp matches on
    // when writing / reading flatc-generated union discriminants.
    #[repr(u8)]
    pub enum CommandTag {
        NONE = 0,
        CreateSection = 1,
        DeleteSection = 2,
        MoveSection = 3,
        ReorderChildren = 4,
        DiffChangeSection = 5,
        DiffDeleteSection = 6,
    }

    // ── Tagged mirror of the FlatBuffers union ──────────────────────────────
    // Exactly one payload is meaningful per instance (the one named by `tag`).
    // Inactive payloads are default-constructed and MUST be ignored.
    #[derive(Default)]
    pub struct StructCommand {
        pub tag: CommandTag,
        pub create_section: CommandCreateSection,
        pub delete_section: CommandDeleteSection,
        pub move_section: CommandMoveSection,
        pub reorder_children: CommandReorderChildren,
        pub diff_change_section: CommandDiffChangeSection,
        pub diff_delete_section: CommandDiffDeleteSection,
    }

    #[derive(Default)]
    pub struct SetField {
        pub section_id: u64,
        pub key_name: String,
        pub value_or_ext_hash: Vec<u8>,
        pub ext_value_size: u64,
        pub ext_value_index: u32,
        pub set_order: u64,
    }

    #[derive(Default)]
    pub struct TimeSample {
        pub section_id: u64,
        pub time: f64,
        pub value_or_ext_hash: Vec<u8>,
        pub ext_value_size: u64,
        pub ext_value_index: u32,
    }

    #[derive(Default)]
    pub struct OwnedDelta {
        pub is_diff: bool,
        pub base_version: u64,
        pub struct_commands: Vec<StructCommand>,
        pub set_fields: Vec<SetField>,
        pub time_samples: Vec<TimeSample>,
        pub source_format: String,
    }

    // ── Parse error taxonomy ────────────────────────────────────────────────
    // Values MUST stay in sync with oodf.cpp's error-returning paths.
    #[repr(u32)]
    pub enum ParseErrorKind {
        Ok = 0,
        EmptyInput = 1,
        TooShort = 2,
        BadMagic = 3,
        VerifierFailed = 4,
        UnknownCommandType = 5,
        AllocationFailed = 6,
        InternalError = 7,
        // Raised when a FlatBuffers string field contains bytes that are not
        // valid UTF-8. The FlatBuffers verifier treats strings as opaque byte
        // vectors, so this check is performed explicitly in oodf.cpp before
        // any `rust::String` is constructed. `error_message` names the schema
        // field that tripped the check and the byte offset of the first
        // invalid byte, to aid in debugging the producer.
        InvalidUtf8 = 8,
        // Raised by oodf.cpp's entry-count probe, before the full verifier
        // runs, when the delta's struct commands + set fields + time samples
        // sum to more than one buffer may carry (`kVerifierMaxTables - 1`:
        // the verifier counts the root table too). The counts and the limit
        // travel in `ParseResult`'s count fields, never in `error_message`.
        TooManyTables = 9,
        // Raised by oodf.cpp when a `ReorderChildren` command's children list
        // is longer than the parser's defensive cap (`kMaxCommands`). The
        // list's length travels in `ParseResult::children_list_len`.
        ChildrenListTooLong = 10,
    }

    #[derive(Default)]
    pub struct ParseResult {
        pub delta: OwnedDelta,
        pub error: ParseErrorKind,
        pub error_message: String,
        // Set only when `error == TooManyTables` (zero otherwise): the entry
        // counts the probe read from the buffer's three vector headers, and
        // the most entries one delta may carry in total.
        pub struct_command_count: u64,
        pub set_field_count: u64,
        pub time_sample_count: u64,
        pub max_entries: u64,
        // Set only when `error == ChildrenListTooLong` (zero otherwise): the
        // length of the children list the parser refused.
        pub children_list_len: u64,
    }

    // ── Build error taxonomy ────────────────────────────────────────────────
    // Build errors raised on the C++ side now flow back as data instead of
    // `std::abort()` (b.vvs item 8). The previous abort blew away every
    // concurrent in-flight RPC sharing the process when a single bridge
    // caller (a) emitted an out-of-range CommandTag or (b) hit allocator
    // exhaustion while serializing a buffer. Both are now surfaced as
    // BuildResult values so the Rust side can map them to a clean
    // `OodfError`.
    #[repr(u32)]
    pub enum BuildErrorKind {
        Ok = 0,
        // Reached when a `StructCommand` has `tag.repr` outside the declared
        // `CommandTag` range (e.g. a `NONE` or a value > DiffDeleteSection).
        // The Rust idiomatic types prevent this at the type system, but the
        // cxx bridge allows raw `repr: u8`, so the C++ side defends against
        // a buggy or malicious bridge caller.
        UnknownCommandTag = 1,
        // `std::bad_alloc` from the FlatBufferBuilder or any byte-vector
        // copy.
        AllocationFailed = 2,
        // Any other `std::exception` escaping the build path. Should not
        // normally fire — the existing implementation has no throwing
        // accessors — but kept as a catch-all so we never reach
        // `std::abort()`.
        InternalError = 3,
    }

    #[derive(Default)]
    pub struct BuildResult {
        pub bytes: Vec<u8>,
        pub error: BuildErrorKind,
        pub error_message: String,
    }

    unsafe extern "C++" {
        include!("oodf.h");

        fn parse_delta(bytes: &[u8]) -> ParseResult;
        fn build_delta(delta: &OwnedDelta) -> BuildResult;
    }

    extern "Rust" {
        // Bulk-copy helper invoked from C++ to extend a `rust::Vec<u8>` in a
        // single bridge call. Previously oodf.cpp's hot paths (parse's
        // external-value marshalling and build_delta's final buffer copy-out)
        // appended bytes via `rust::Vec::push_back`, which on cxx 1.0.194
        // performs one `reserve_total` bridge call per byte. For 1 MB+
        // external values that is a measurable overhead. Exposing an
        // idiomatic `Vec::extend_from_slice` across the bridge reduces
        // the whole copy to one bridge invocation, which benchmarked to
        // ~8× faster on 1 MB+ payloads during the b.8uk review (M-series
        // release build).
        fn extend_byte_vec(dst: &mut Vec<u8>, src: &[u8]);

        // Same bulk-copy idiom for `u64` vectors. Used by `parse_delta` to
        // marshal `CommandReorderChildren.children_list` without the
        // per-element bridge round-trip that the previous `push_back` loop
        // incurred (b.vvs item 4). On a `ReorderChildren` command with a
        // million-element `childrenList` this collapses ~1M bridge calls into
        // one.
        fn extend_u64_vec(dst: &mut Vec<u64>, src: &[u64]);
    }
}

/// `extern "Rust"` impl for the cxx bridge function.
#[doc(hidden)]
pub fn extend_byte_vec(dst: &mut Vec<u8>, src: &[u8]) {
    dst.extend_from_slice(src);
}

/// `extern "Rust"` impl for the u64 bulk-copy bridge function (b.vvs item 4).
#[doc(hidden)]
pub fn extend_u64_vec(dst: &mut Vec<u64>, src: &[u64]) {
    dst.extend_from_slice(src);
}

// cxx does not synthesize `Default` on shared enums declared as C++-style
// `enum class`. Provide the canonical zero-value explicitly so shared struct
// defaults (`StructCommand::default()`, `ParseResult::default()`) work.
//
// Picking `NONE` (the FlatBuffers-level discriminant for "no union payload")
// as the default is intentional: `StructCommand::default()` must land in a
// value that is universally detected as invalid by
// `crate::convert::struct_command_from_bridge`, not in a silently
// well-formed variant. `NONE` has no associated payload and is not used
// anywhere in the OODF schema — any `StructCommand` reaching idiomatic Rust
// with `tag = NONE` therefore guarantees surfacing as
// `OodfError::UnknownCommandTagAt { .., tag: 0 }` rather than an empty but
// "plausible" command.
impl Default for ffi::CommandTag {
    fn default() -> Self {
        ffi::CommandTag::NONE
    }
}

impl Default for ffi::ParseErrorKind {
    fn default() -> Self {
        ffi::ParseErrorKind::Ok
    }
}

impl Default for ffi::BuildErrorKind {
    fn default() -> Self {
        ffi::BuildErrorKind::Ok
    }
}
