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

//! Public error type for the `oodf-ffi` crate.
//!
//! `OodfError` covers every failure mode the public `parse_delta` /
//! `build_delta` API can surface to callers. The variants mirror
//! `crate::bridge::ffi::ParseErrorKind` (emitted by the C++ core from Epic 2)
//! plus two bridge-layer variants (`UnknownCommandTagAt`, `InternalError`)
//! that guard against invariant violations at the cxx boundary.
//!
//! Epic 4 will add `From<OodfError>` impls mapping into the service's
//! `DeltaError` / `ServiceError`. The crate itself does not depend on any
//! service-level types, keeping `oodf-ffi` self-contained.

use crate::bridge::ffi;

/// Errors returned by the public `oodf-ffi` API.
///
/// Every variant renders a human-readable message via `thiserror`'s
/// `#[error(...)]` attribute; callers can use `Display` for logs and
/// `std::error::Error` for error-chain integration.
#[derive(Debug, thiserror::Error)]
pub enum OodfError {
    /// The input byte slice was empty.
    #[error("empty input")]
    EmptyInput,

    /// The input was too short to contain the OODF root.
    #[error("buffer too short for OODF root")]
    TooShort,

    /// The 4-byte file identifier did not match `OODF`.
    #[error("wrong file identifier; expected OODF")]
    BadMagic,

    /// The FlatBuffers verifier rejected the buffer. Payload is the
    /// C++-side message describing where verification failed.
    #[error("FlatBuffers verifier failed: {0}")]
    VerifierFailed(String),

    /// A `StructCommand` arrived with a tag/discriminant the C++ parser did
    /// not recognize. Payload is the observed discriminant value — a `u8`
    /// because the FlatBuffers union-discriminant vector is `[ubyte]` per
    /// `OmniObjectDelta.fbs`, so any legitimately-encoded discriminant fits
    /// in a `u8`. Kept distinct from `UnknownCommandTagAt` because the two
    /// detectors sit on opposite sides of the bridge: this variant fires in
    /// `oodf.cpp` before the discriminant crosses the FFI, while
    /// `UnknownCommandTagAt` fires in `crate::convert` after.
    #[error("unknown command type: {0}")]
    UnknownCommandType(u8),

    /// The C++ side failed to allocate memory while building `OwnedDelta`.
    #[error("C++-side allocation failed")]
    AllocationFailed,

    /// Catch-all for bridge invariants (C++ emitted a value the Rust side
    /// could not map without panicking). Payload is a descriptive message.
    #[error("internal OODF error: {0}")]
    InternalError(String),

    /// Reserved for future FFI-layer issues that don't fit the existing
    /// taxonomy. Not currently produced by `parse_delta` / `build_delta`.
    #[error("FFI error: {0}")]
    Ffi(String),

    /// A `StructCommand` arrived from C++ whose `CommandTag` value is out of
    /// range for the Rust-side enum. Distinct from `UnknownCommandType`:
    /// this is raised by the bridge-to-idiomatic conversion in
    /// `crate::convert`, while `UnknownCommandType` is raised by the C++
    /// parser itself before the bridge. `index` is the position of the bad
    /// tag within the `struct_commands` vector.
    #[error("unknown StructCommand tag {tag} at index {index}")]
    UnknownCommandTagAt { index: usize, tag: u8 },

    /// A FlatBuffers string field in the parsed buffer contained bytes that
    /// were not valid UTF-8. The C++ parser detects this before any
    /// `rust::String` is constructed, so invalid input cannot cross the FFI
    /// boundary as a panic. Payload is the C++-side message describing the
    /// field and, when available, the byte offset of the first bad byte.
    #[error("invalid UTF-8 in FlatBuffers string field: {0}")]
    InvalidUtf8(String),

    /// `build_delta` reached a `StructCommand` whose `CommandTag` was out of
    /// range (raised by the C++ side after `b.vvs` item 8 replaced
    /// `std::abort()` with a return-by-value error). The idiomatic Rust
    /// types prevent this at the type system, so this variant fires only
    /// when a bridge caller smuggles a raw out-of-range `repr: u8`.
    /// Payload is the C++-side message including the observed discriminant
    /// and command index.
    #[error("build_delta saw unknown CommandTag: {0}")]
    BuildUnknownCommandTag(String),

    /// The delta carries more entries than one OODF buffer may hold: its
    /// struct commands, set fields and time samples sum to more than
    /// `max_entries` (`MAX_DELTA_TABLES - 1`, see [`crate::MAX_DELTA_TABLES`]).
    /// Raised by `parse_delta`'s entry-count probe, which reads the three
    /// vector lengths after a bounds-checked look at the root table and runs
    /// before the full verifier, so an over-limit buffer is refused without
    /// the verifier walking its tables. The counts are the buffer's own vector
    /// lengths; they travel in the bridge's `ParseResult` fields, never parsed
    /// out of the message.
    #[error(
        "delta carries {} entries ({struct_commands} struct commands, {set_fields} set fields, \
         {time_samples} time samples); one delta may carry at most {max_entries} entries in total",
        entry_total(.struct_commands, .set_fields, .time_samples)
    )]
    TooManyTables {
        struct_commands: u64,
        set_fields: u64,
        time_samples: u64,
        max_entries: u64,
    },

    /// A `ReorderChildren` command's children list is longer than the
    /// parser's defensive cap (1,000,000 ids, ten times the service's own
    /// per-list limit), so the parser refused it before copying it. `len` is
    /// the list's length, read from the bridge's `ParseResult`.
    #[error("ReorderChildren children list of {len} ids exceeds the parser's defensive cap")]
    ChildrenListTooLong { len: u64 },
}

/// Sum of the three per-kind counts [`OodfError::TooManyTables`] renders,
/// saturating, so a hand-built error with absurd counts still renders.
fn entry_total(struct_commands: &u64, set_fields: &u64, time_samples: &u64) -> u64 {
    struct_commands
        .saturating_add(*set_fields)
        .saturating_add(*time_samples)
}

impl OodfError {
    /// Build an `OodfError` from a completed `ffi::ParseResult` that reports
    /// a non-`Ok` `error` kind. Consumes both the error discriminant and
    /// `error_message`, so payload-carrying variants (`VerifierFailed`,
    /// `InternalError`, `Ffi`) keep the C++-side explanation.
    /// `TooManyTables` and `ChildrenListTooLong` take their numbers from the
    /// result's structured fields, not from the message.
    ///
    /// For `UnknownCommandType` the C++ parser reports the observed
    /// discriminant textually inside `error_message`; we attempt to parse the
    /// first ASCII integer out of the message and fall back to `0` if no
    /// integer is present.
    ///
    /// Calling this with `result.error == ParseErrorKind::Ok` is a
    /// programming error on the Rust side; we return `InternalError` in that
    /// case rather than panicking.
    pub fn from_ffi(result: &ffi::ParseResult) -> Self {
        match result.error.repr {
            r if r == ffi::ParseErrorKind::Ok.repr => OodfError::InternalError(format!(
                "from_ffi called on successful ParseResult (message: {:?})",
                result.error_message
            )),
            r if r == ffi::ParseErrorKind::EmptyInput.repr => OodfError::EmptyInput,
            r if r == ffi::ParseErrorKind::TooShort.repr => OodfError::TooShort,
            r if r == ffi::ParseErrorKind::BadMagic.repr => OodfError::BadMagic,
            r if r == ffi::ParseErrorKind::VerifierFailed.repr => {
                OodfError::VerifierFailed(result.error_message.clone())
            }
            r if r == ffi::ParseErrorKind::UnknownCommandType.repr => {
                let observed = result
                    .error_message
                    .split(|c: char| !c.is_ascii_digit())
                    .find(|s| !s.is_empty())
                    .and_then(|s| s.parse::<u8>().ok())
                    .unwrap_or(0);
                OodfError::UnknownCommandType(observed)
            }
            r if r == ffi::ParseErrorKind::AllocationFailed.repr => OodfError::AllocationFailed,
            r if r == ffi::ParseErrorKind::InternalError.repr => {
                OodfError::InternalError(result.error_message.clone())
            }
            r if r == ffi::ParseErrorKind::InvalidUtf8.repr => {
                OodfError::InvalidUtf8(result.error_message.clone())
            }
            r if r == ffi::ParseErrorKind::TooManyTables.repr => OodfError::TooManyTables {
                struct_commands: result.struct_command_count,
                set_fields: result.set_field_count,
                time_samples: result.time_sample_count,
                max_entries: result.max_entries,
            },
            r if r == ffi::ParseErrorKind::ChildrenListTooLong.repr => {
                OodfError::ChildrenListTooLong {
                    len: result.children_list_len,
                }
            }
            _ => OodfError::Ffi(format!(
                "unknown ParseErrorKind discriminant {}: {}",
                result.error.repr, result.error_message
            )),
        }
    }

    /// Build an `OodfError` from a `ffi::BuildResult` whose `error` field is
    /// non-`Ok`. Mirror of `from_ffi` for the build path (b.vvs item 8).
    /// Returns `InternalError` if called with an `Ok` result — that is a
    /// Rust-side logic bug, but we surface it instead of panicking so the
    /// caller can log and continue.
    pub fn from_build_ffi(result: &ffi::BuildResult) -> Self {
        match result.error.repr {
            r if r == ffi::BuildErrorKind::Ok.repr => OodfError::InternalError(format!(
                "from_build_ffi called on successful BuildResult (message: {:?})",
                result.error_message
            )),
            r if r == ffi::BuildErrorKind::UnknownCommandTag.repr => {
                OodfError::BuildUnknownCommandTag(result.error_message.clone())
            }
            r if r == ffi::BuildErrorKind::AllocationFailed.repr => OodfError::AllocationFailed,
            r if r == ffi::BuildErrorKind::InternalError.repr => {
                OodfError::InternalError(result.error_message.clone())
            }
            _ => OodfError::Ffi(format!(
                "unknown BuildErrorKind discriminant {}: {}",
                result.error.repr, result.error_message
            )),
        }
    }
}

/// Best-effort mapping from the raw FFI discriminant. Payload-carrying
/// variants get placeholders; callers that have access to the full
/// `ParseResult` should prefer [`OodfError::from_ffi`] so error messages
/// from the C++ side are preserved.
impl From<ffi::ParseErrorKind> for OodfError {
    fn from(kind: ffi::ParseErrorKind) -> Self {
        match kind.repr {
            r if r == ffi::ParseErrorKind::Ok.repr => {
                OodfError::InternalError("From<ParseErrorKind> called on Ok".to_string())
            }
            r if r == ffi::ParseErrorKind::EmptyInput.repr => OodfError::EmptyInput,
            r if r == ffi::ParseErrorKind::TooShort.repr => OodfError::TooShort,
            r if r == ffi::ParseErrorKind::BadMagic.repr => OodfError::BadMagic,
            r if r == ffi::ParseErrorKind::VerifierFailed.repr => {
                OodfError::VerifierFailed(String::new())
            }
            r if r == ffi::ParseErrorKind::UnknownCommandType.repr => {
                OodfError::UnknownCommandType(0)
            }
            r if r == ffi::ParseErrorKind::AllocationFailed.repr => OodfError::AllocationFailed,
            r if r == ffi::ParseErrorKind::InternalError.repr => {
                OodfError::InternalError(String::new())
            }
            r if r == ffi::ParseErrorKind::InvalidUtf8.repr => {
                OodfError::InvalidUtf8(String::new())
            }
            // The numbers travel in `ParseResult`, which this conversion does
            // not see: zero counts and length are placeholders; the limit is
            // the real one.
            r if r == ffi::ParseErrorKind::TooManyTables.repr => OodfError::TooManyTables {
                struct_commands: 0,
                set_fields: 0,
                time_samples: 0,
                max_entries: crate::MAX_DELTA_TABLES as u64 - 1,
            },
            r if r == ffi::ParseErrorKind::ChildrenListTooLong.repr => {
                OodfError::ChildrenListTooLong { len: 0 }
            }
            other => OodfError::Ffi(format!("unknown ParseErrorKind discriminant {other}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types;

    // Compile-time assertions that the public error and delta types satisfy
    // the trait bounds consumers (including Epic 4's tonic handlers) rely on.
    const _: fn() = || {
        fn assert_send_sync_error<T: Send + Sync + std::error::Error>() {}
        assert_send_sync_error::<OodfError>();

        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<types::OwnedDelta>();
    };

    #[test]
    fn display_is_non_empty_for_every_variant() {
        let cases: Vec<OodfError> = vec![
            OodfError::EmptyInput,
            OodfError::TooShort,
            OodfError::BadMagic,
            OodfError::VerifierFailed("bad root table".to_string()),
            OodfError::UnknownCommandType(42),
            OodfError::AllocationFailed,
            OodfError::InternalError("something broke".to_string()),
            OodfError::Ffi("bridge weirdness".to_string()),
            OodfError::UnknownCommandTagAt { index: 3, tag: 9 },
            OodfError::InvalidUtf8("sectionName @ offset 7".to_string()),
            OodfError::BuildUnknownCommandTag("build saw tag 9 at index 0".to_string()),
        ];
        for err in cases {
            let rendered = format!("{err}");
            assert!(!rendered.is_empty(), "Display for {err:?} was empty");
        }
    }

    #[test]
    fn display_contains_variant_identifying_substrings() {
        assert!(format!("{}", OodfError::EmptyInput).contains("empty"));
        assert!(format!("{}", OodfError::TooShort).contains("too short"));
        assert!(format!("{}", OodfError::BadMagic).contains("OODF"));
        assert!(format!("{}", OodfError::VerifierFailed("X".to_string())).contains("verifier"));
        assert!(format!("{}", OodfError::UnknownCommandType(7)).contains("unknown command type"));
        assert!(format!("{}", OodfError::AllocationFailed).contains("allocation"));
        assert!(format!("{}", OodfError::InternalError("X".to_string())).contains("internal"));
        assert!(format!("{}", OodfError::Ffi("X".to_string())).contains("FFI"));
        assert!(
            format!("{}", OodfError::UnknownCommandTagAt { index: 4, tag: 3 })
                .contains("StructCommand tag")
        );
        assert!(format!("{}", OodfError::InvalidUtf8("sectionName".to_string())).contains("UTF-8"));
    }

    #[test]
    fn display_carries_payload_values() {
        assert!(format!("{}", OodfError::VerifierFailed("root".into())).contains("root"));
        assert!(format!("{}", OodfError::UnknownCommandType(42)).contains("42"));
        assert!(format!("{}", OodfError::InternalError("oops".into())).contains("oops"));
        assert!(format!("{}", OodfError::Ffi("weird".into())).contains("weird"));
        let rendered = format!("{}", OodfError::UnknownCommandTagAt { index: 4, tag: 9 });
        assert!(rendered.contains("9"));
        assert!(rendered.contains("4"));
        assert!(format!(
            "{}",
            OodfError::InvalidUtf8("keyName @ offset 2".to_string())
        )
        .contains("keyName"));
    }

    #[test]
    fn from_parse_error_kind_maps_every_variant() {
        assert!(matches!(
            OodfError::from(ffi::ParseErrorKind::EmptyInput),
            OodfError::EmptyInput
        ));
        assert!(matches!(
            OodfError::from(ffi::ParseErrorKind::TooShort),
            OodfError::TooShort
        ));
        assert!(matches!(
            OodfError::from(ffi::ParseErrorKind::BadMagic),
            OodfError::BadMagic
        ));
        assert!(matches!(
            OodfError::from(ffi::ParseErrorKind::VerifierFailed),
            OodfError::VerifierFailed(_)
        ));
        assert!(matches!(
            OodfError::from(ffi::ParseErrorKind::UnknownCommandType),
            OodfError::UnknownCommandType(_)
        ));
        assert!(matches!(
            OodfError::from(ffi::ParseErrorKind::AllocationFailed),
            OodfError::AllocationFailed
        ));
        assert!(matches!(
            OodfError::from(ffi::ParseErrorKind::InternalError),
            OodfError::InternalError(_)
        ));
        // Ok is a logic bug on the Rust side — it maps to InternalError
        // rather than panicking, so callers can observe it and report it.
        assert!(matches!(
            OodfError::from(ffi::ParseErrorKind::Ok),
            OodfError::InternalError(_)
        ));
    }

    #[test]
    fn from_ffi_preserves_verifier_message() {
        let result = ffi::ParseResult {
            delta: ffi::OwnedDelta::default(),
            error: ffi::ParseErrorKind::VerifierFailed,
            error_message: "offset 0x40 past end of buffer".to_string(),
            ..Default::default()
        };
        match OodfError::from_ffi(&result) {
            OodfError::VerifierFailed(msg) => assert!(msg.contains("offset 0x40")),
            other => panic!("expected VerifierFailed, got {other:?}"),
        }
    }

    #[test]
    fn from_ffi_extracts_unknown_command_type_discriminant() {
        let result = ffi::ParseResult {
            delta: ffi::OwnedDelta::default(),
            error: ffi::ParseErrorKind::UnknownCommandType,
            error_message: "saw discriminant 42 during parse".to_string(),
            ..Default::default()
        };
        assert!(matches!(
            OodfError::from_ffi(&result),
            OodfError::UnknownCommandType(42)
        ));
    }

    #[test]
    fn from_ffi_falls_back_to_zero_when_message_has_no_integer() {
        let result = ffi::ParseResult {
            delta: ffi::OwnedDelta::default(),
            error: ffi::ParseErrorKind::UnknownCommandType,
            error_message: "no integer here".to_string(),
            ..Default::default()
        };
        assert!(matches!(
            OodfError::from_ffi(&result),
            OodfError::UnknownCommandType(0)
        ));
    }

    #[test]
    fn from_ffi_preserves_internal_error_message() {
        let result = ffi::ParseResult {
            delta: ffi::OwnedDelta::default(),
            error: ffi::ParseErrorKind::InternalError,
            error_message: "cxx.rs: allocation vector overflow".to_string(),
            ..Default::default()
        };
        match OodfError::from_ffi(&result) {
            OodfError::InternalError(msg) => assert!(msg.contains("vector overflow")),
            other => panic!("expected InternalError, got {other:?}"),
        }
    }

    #[test]
    fn from_ffi_on_ok_reports_logic_bug() {
        let result = ffi::ParseResult {
            delta: ffi::OwnedDelta::default(),
            error: ffi::ParseErrorKind::Ok,
            error_message: String::new(),
            ..Default::default()
        };
        assert!(matches!(
            OodfError::from_ffi(&result),
            OodfError::InternalError(_)
        ));
    }

    /// The probe's refusal takes its numbers from the result's structured
    /// fields; the message is for logs only, so a message carrying other
    /// numbers changes nothing.
    #[test]
    fn from_ffi_builds_too_many_tables_from_the_result_fields_not_the_message() {
        let result = ffi::ParseResult {
            error: ffi::ParseErrorKind::TooManyTables,
            error_message: "delta carries 7 entries; at most 8 fit in one delta".to_string(),
            struct_command_count: 60_000,
            set_field_count: 40_000,
            time_sample_count: 1,
            max_entries: 99_999,
            ..Default::default()
        };
        match OodfError::from_ffi(&result) {
            OodfError::TooManyTables {
                struct_commands,
                set_fields,
                time_samples,
                max_entries,
            } => assert_eq!(
                (struct_commands, set_fields, time_samples, max_entries),
                (60_000, 40_000, 1, 99_999)
            ),
            other => panic!("expected TooManyTables, got {other:?}"),
        }
    }

    /// The parser's children-list refusal carries the list's length from
    /// `children_list_len`, not from the message.
    #[test]
    fn from_ffi_builds_children_list_too_long_from_its_length_field() {
        let result = ffi::ParseResult {
            error: ffi::ParseErrorKind::ChildrenListTooLong,
            error_message: "ReorderChildren.childrenList size 5 exceeds the defensive cap of 4"
                .to_string(),
            children_list_len: 1_000_001,
            ..Default::default()
        };
        assert!(matches!(
            OodfError::from_ffi(&result),
            OodfError::ChildrenListTooLong { len: 1_000_001 }
        ));
    }

    /// The bare-kind conversion has no `ParseResult` to read counts from:
    /// zero counts and length, but the real limit.
    #[test]
    fn parse_error_kind_conversion_gives_the_size_refusals_placeholder_counts_and_the_real_limit() {
        assert!(matches!(
            OodfError::from(ffi::ParseErrorKind::TooManyTables),
            OodfError::TooManyTables {
                struct_commands: 0,
                set_fields: 0,
                time_samples: 0,
                max_entries: 99_999,
            }
        ));
        assert!(matches!(
            OodfError::from(ffi::ParseErrorKind::ChildrenListTooLong),
            OodfError::ChildrenListTooLong { len: 0 }
        ));
    }

    /// The refusal's text names the total, each count and the limit; the
    /// total saturates, so even a hand-built error with absurd counts renders.
    #[test]
    fn too_many_tables_display_names_the_total_each_count_and_the_limit() {
        let err = OodfError::TooManyTables {
            struct_commands: 60_000,
            set_fields: 60_000,
            time_samples: 0,
            max_entries: 99_999,
        };
        assert_eq!(
            err.to_string(),
            "delta carries 120000 entries (60000 struct commands, 60000 set fields, 0 time \
             samples); one delta may carry at most 99999 entries in total"
        );
        let absurd = OodfError::TooManyTables {
            struct_commands: u64::MAX,
            set_fields: 1,
            time_samples: 1,
            max_entries: 99_999,
        };
        assert!(
            absurd
                .to_string()
                .starts_with(&format!("delta carries {} entries", u64::MAX)),
            "{absurd}"
        );
    }

    #[test]
    fn from_ffi_covers_simple_variants() {
        let mk = |kind: ffi::ParseErrorKind| ffi::ParseResult {
            delta: ffi::OwnedDelta::default(),
            error: kind,
            error_message: String::new(),
            ..Default::default()
        };
        assert!(matches!(
            OodfError::from_ffi(&mk(ffi::ParseErrorKind::EmptyInput)),
            OodfError::EmptyInput
        ));
        assert!(matches!(
            OodfError::from_ffi(&mk(ffi::ParseErrorKind::TooShort)),
            OodfError::TooShort
        ));
        assert!(matches!(
            OodfError::from_ffi(&mk(ffi::ParseErrorKind::BadMagic)),
            OodfError::BadMagic
        ));
        assert!(matches!(
            OodfError::from_ffi(&mk(ffi::ParseErrorKind::AllocationFailed)),
            OodfError::AllocationFailed
        ));
    }
}
