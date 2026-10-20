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

//! `oodf-ffi` — Rust bindings to the Omniverse OODF C++ library.
//!
//! ## Architecture
//!
//! ```text
//! user code
//!   ↓  (OwnedDelta, StructCommand, ...)
//! parse_delta / build_delta   ← this lib.rs
//!   ↓  (convert.rs: From/TryFrom)
//! ffi::OwnedDelta / ffi::ParseResult   ← bridge.rs (#[cxx::bridge])
//!   ↓  (cxx FFI)
//! C++ oodf::parse_delta / oodf::build_delta   ← cpp/oodf.cpp
//! ```
//!
//! Idiomatic Rust types live in [`types`]. The private [`bridge`] module holds
//! the cxx shared structs. [`convert`] translates between them. This module
//! exposes the two public functions [`parse_delta`] and [`build_delta`] that
//! external callers use.
//!
//! ## `bridge` visibility
//!
//! `bridge` is `pub(crate)` in production builds so downstream consumers
//! (e.g. the `live-edit` parent crate) cannot import `oodf_ffi::bridge::ffi`
//! or any cxx type directly — they must go through the idiomatic Rust types
//! and the public wrappers in this module. Enabling the `internal-bridge`
//! cargo feature re-exposes the module as `pub` so `oodf-ffi`'s own
//! integration tests (which need to construct bridge values to exercise FFI
//! edge cases) can compile. The feature is intentionally NOT a default
//! feature, so a `[dependencies] oodf-ffi = { path = "..." }` declaration
//! (as in the workspace root `Cargo.toml`) does not enable it (b.vvs item
//! 7). Run the crate tests with `cargo test -p oodf-ffi --features
//! internal-bridge`.

#[cfg(feature = "internal-bridge")]
#[doc(hidden)]
pub mod bridge;
#[cfg(not(feature = "internal-bridge"))]
pub(crate) mod bridge;

#[cfg(feature = "internal-bridge")]
#[doc(hidden)]
pub mod convert;
#[cfg(not(feature = "internal-bridge"))]
pub(crate) mod convert;

pub mod error;
pub mod types;

pub use error::OodfError;
pub use types::{
    canonical_sample_time, CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection,
    CommandDiffDeleteSection, CommandMoveSection, CommandReorderChildren, OwnedDelta, SetField,
    StructCommand, TimeSample,
};

use bridge::ffi;

/// The most FlatBuffers tables one OODF buffer may hold: the verifier's
/// `max_tables` option, `kVerifierMaxTables` in `cpp/oodf.cpp`. The two values
/// must stay equal; changing either needs the security review that constant's
/// comment asks for.
///
/// The OODF schema has no nested tables, so a delta is exactly one root table
/// plus one table per struct command, set field and time sample. One delta
/// therefore carries at most `MAX_DELTA_TABLES - 1` entries in total, summed
/// across the three kinds. [`parse_delta`] refuses a buffer over that limit
/// with [`OodfError::TooManyTables`] before the verifier walks it.
///
/// The limit bounds one delta a client sends, never a layer: a layer is built
/// from as many deltas as its writers send, so a whole layer can hold far
/// more entries than any one delta, and the service's catch-up of such a
/// layer is one delta of that size, which this parser refuses.
pub const MAX_DELTA_TABLES: usize = 100_000;

/// Parse an OODF FlatBuffers byte stream into an owned delta.
///
/// Calls into the hardened C++ FlatBuffers verifier + accessor layer.
/// Never panics, even on arbitrary or malicious input — every error path
/// returns an [`OodfError`].
///
/// A buffer whose struct commands, set fields and time samples sum to more
/// than `MAX_DELTA_TABLES - 1` is refused with [`OodfError::TooManyTables`],
/// carrying the three counts, before the full verifier runs — so an over-cap
/// buffer whose elements are also malformed gets `TooManyTables`, not
/// [`OodfError::VerifierFailed`]. A buffer the probe cannot count (root table
/// or vector headers fail a bounds check, or the two command vectors are
/// unpaired) falls through to the verifier and keeps the outcome it would
/// have had without the probe. A `ReorderChildren` children list longer than
/// the parser's defensive cap is refused with
/// [`OodfError::ChildrenListTooLong`], carrying its length.
///
/// Every time sample's `time` is canonicalized with [`canonical_sample_time`]:
/// a buffer carrying `-0.0` parses to `+0.0`. `build_delta` followed by
/// `parse_delta` is therefore the identity only for deltas whose times are
/// already canonical.
pub fn parse_delta(bytes: &[u8]) -> Result<OwnedDelta, OodfError> {
    let result = ffi::parse_delta(bytes);
    if result.error.repr == ffi::ParseErrorKind::Ok.repr {
        // Decompose the ParseResult by move so we avoid needing Clone on the
        // cxx shared struct (cxx does not auto-derive Clone for shared structs).
        let ffi::ParseResult { delta, .. } = result;
        // Return the original `OodfError` unchanged so callers see e.g. the
        // precise `OodfError::UnknownCommandTagAt { index, tag }` instead of
        // a stringified version stuffed inside `InternalError`. The C++
        // parser already rejects unknown discriminants at parse time with
        // `UnknownCommandType`, so this arm firing means either the flatc-
        // generated verifier and oodf.cpp disagree on the legal tag set or
        // the bridge is returning a tag the Rust enum does not know about —
        // both are bridge inconsistencies, not "Internal" errors.
        OwnedDelta::try_from(delta)
    } else {
        Err(OodfError::from_ffi(&result))
    }
}

/// Build an OODF FlatBuffers byte stream from an owned delta.
///
/// The Rust idiomatic types make every well-formed `OwnedDelta` representable,
/// so callers will not normally see an error. The function still returns
/// `Result<_, OodfError>` because the cxx bridge admits raw discriminants and
/// allocator failure can in principle occur on the C++ side; surfacing those
/// as `OodfError` rather than aborting the process is part of b.vvs item 8 (a
/// `std::abort()` in the parser blew away every concurrent in-flight RPC
/// sharing the process).
///
/// `delta` is borrowed: the conversion layer (`From<&types::OwnedDelta>`) does
/// the work of copying owned fields into bridge values, so the previous
/// `delta.clone()` (which deep-copied every external value byte twice) is no
/// longer required (b.vvs item 6).
pub fn build_delta(delta: &OwnedDelta) -> Result<Vec<u8>, OodfError> {
    let ffi_delta: ffi::OwnedDelta = delta.into();
    let result = ffi::build_delta(&ffi_delta);
    if result.error.repr == ffi::BuildErrorKind::Ok.repr {
        Ok(result.bytes)
    } else {
        Err(OodfError::from_build_ffi(&result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_empty_input_returns_empty_input_err() {
        let err = parse_delta(&[]).unwrap_err();
        assert!(matches!(err, OodfError::EmptyInput), "got {err:?}");
    }

    #[test]
    fn round_trip_default_owned_delta() {
        let original = OwnedDelta::default();
        let bytes = build_delta(&original).expect("build_delta of default delta must succeed");
        let parsed = parse_delta(&bytes).expect("round-trip parse failed");
        assert_eq!(parsed, original);
    }

    #[test]
    fn round_trip_non_trivial_owned_delta() {
        let original = OwnedDelta {
            is_diff: false,
            base_version: 7,
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 1,
                section_id: 2,
                section_name: "hello".into(),
                section_type: 3,
            })],
            set_fields: vec![SetField {
                section_id: 2,
                key_name: "k".into(),
                value_or_ext_hash: vec![1, 2, 3, 4],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            time_samples: vec![],
            source_format: String::new(),
        };
        let bytes = build_delta(&original).expect("build_delta of well-formed delta must succeed");
        let parsed = parse_delta(&bytes).expect("round-trip parse failed");
        assert_eq!(parsed, original);
    }
}
