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

//! Session-checkpoint wire payload (SR-2.7).
//!
//! Defines [`SessionCheckpoint`], the JSON-serialized envelope the Epic 2
//! session-and-layer-mapping DR path persists via
//! [`CheckpointStore::put_session_checkpoint`](crate::checkpoint::CheckpointStore::put_session_checkpoint)
//! and reconstructs via
//! [`CheckpointStore::get_session_checkpoint`](crate::checkpoint::CheckpointStore::get_session_checkpoint).
//! The struct mirrors the Valkey state a single session contributes:
//!
//! - The four `le:{sid}:sess` hash fields (`name`, `base_path`, `created_by`,
//!   `created_at`) — verified field-for-field against
//!   `SessionStore::create_once` (`src/storage/session.rs`).
//! - The session's layer-name → object_id mapping (`le:{sid}:layer:{hex(name)}`
//!   reverse-decoded to the raw name).
//! - The session's `le:{sid}:layers` set membership (carried independently of
//!   the mapping so the applier does not have to derive it).
//! - The path indices the session contributes to: the legacy
//!   session-id-only `le:path:{hex(base_path)}` set and the composite
//!   `le:path:{hex(base_path)}:names` set.
//!
//! Wire format is non-pretty JSON via `serde_json`; `to_bytes` / `from_bytes`
//! are the only sanctioned codec surface. Parse failures classify as
//! [`CheckpointError::Permanent`] — a malformed stored blob will not heal on
//! retry (matches `in_memory.rs` ext-refs-sidecar handling at line ~351).
//!
//! This module owns only the wire-shape types and serde plumbing. Adapter
//! wire-up, assembly from live Valkey state, write-path emission, and the
//! recovery applier land in sibling Tasks under Epic `t1.fgj.7v`.

use std::collections::{BTreeMap, BTreeSet};

use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::checkpoint::CheckpointError;
use crate::storage::keys::KeySchema;

/// Schema version stamped into every [`SessionCheckpoint`] JSON envelope.
///
/// Bumped when the payload's field set or semantics change in a way that the
/// applier needs to branch on. The Epic 2 wire shape is version `1`.
pub const SESSION_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

/// Default used by `#[serde(default = "default_schema_version")]` so an
/// envelope without an explicit `schema_version` field decodes as the current
/// version. Forward-compatibility insurance for blobs written by a future
/// build that omits the field (or by a test fixture).
fn default_schema_version() -> u32 {
    SESSION_CHECKPOINT_SCHEMA_VERSION
}

/// Serializable wire shape for a single session's checkpoint blob (SR-2.7).
///
/// One `SessionCheckpoint` corresponds to one `session_id`. The
/// `layer_mappings` / `layers_index` / `path_*` collections name the keys and
/// members the session contributes to its peers' state; cross-session keys
/// (e.g. another session's `le:path:{hex(base_path)}:names` membership) are
/// the responsibility of that session's own checkpoint.
///
/// Field ordering and collection types are chosen for **deterministic
/// serialization**: `BTreeMap` and `BTreeSet` produce identical JSON for
/// identical input across runs, which is necessary for byte-equality tests
/// and content-addressed reasoning in downstream Tasks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionCheckpoint {
    /// JSON-envelope schema version. See
    /// [`SESSION_CHECKPOINT_SCHEMA_VERSION`].
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,

    /// Session id this checkpoint reconstructs. Matches the `session_id`
    /// argument passed to `put_session_checkpoint`.
    pub session_id: String,

    /// `le:{sid}:sess` hash field — session display name.
    pub name: String,

    /// `le:{sid}:sess` hash field — `stage:/<path>` the session was created
    /// against.
    pub base_path: String,

    /// `le:{sid}:sess` hash field — caller identity at `CreateSession` time.
    pub created_by: String,

    /// `le:{sid}:sess` hash field — `now_micros()` at `CreateSession` time
    /// (microseconds since Unix epoch).
    pub created_at: u64,

    /// Layer-name → object_id mapping. Keys are the **unhexed** layer names
    /// (the values stored under `le:{sid}:layer:{hex(name)}` keys, with the
    /// hex-encoded suffix decoded back to the raw name).
    pub layer_mappings: BTreeMap<String, u64>,

    /// Members of `le:{sid}:layers`. In a well-formed snapshot this equals
    /// `layer_mappings.keys()`, but it is carried independently so the
    /// applier can re-`SADD` the index without deriving it (and so a torn
    /// snapshot's divergence is observable to the applier).
    pub layers_index: BTreeSet<String>,

    /// `name\0session_id` composite members the session contributes to its
    /// path's `le:path:{hex(base_path)}:names` set. Construct via
    /// [`SessionCheckpoint::add_path_name_member`] — never by ad-hoc
    /// `format!`.
    pub path_name_members: BTreeSet<String>,

    /// The bare `session_id` member the session contributes to the legacy
    /// `le:path:{hex(base_path)}` session-id-only set.
    pub path_index_session_id_member: String,
}

impl SessionCheckpoint {
    /// Serialize to non-pretty JSON wrapped in [`Bytes`].
    ///
    /// Failures classify as [`CheckpointError::Permanent`] — a
    /// `serde_json::to_vec` failure on an in-memory struct is a programmer
    /// bug (non-UTF-8 string content, etc.), not a retryable I/O fault.
    pub fn to_bytes(&self) -> Result<Bytes, CheckpointError> {
        serde_json::to_vec(self)
            .map(Bytes::from)
            .map_err(|err| CheckpointError::Permanent {
                source: Box::new(err),
            })
    }

    /// Deserialize from JSON [`Bytes`].
    ///
    /// Failures classify as [`CheckpointError::Permanent`] — a malformed
    /// stored checkpoint blob will not heal on retry, so the applier should
    /// surface it to the operator rather than retry. Matches the
    /// `serde_json::from_slice` → `Permanent` pattern in
    /// `src/checkpoint/adapters/in_memory.rs` (ext-refs-sidecar parsing).
    pub fn from_bytes(body: &Bytes) -> Result<Self, CheckpointError> {
        serde_json::from_slice(body.as_ref()).map_err(|err| CheckpointError::Permanent {
            source: Box::new(err),
        })
    }

    /// Push a composite `name\0session_id` member into
    /// [`Self::path_name_members`], routed through the sanctioned
    /// [`KeySchema::path_name_member`] constructor.
    ///
    /// This wrapper is the **only** call site (in this module) that
    /// constructs a path-names composite; downstream producers in Tasks 3
    /// (assembler) and 4 (write-path) MUST go through it so the
    /// helper-only-construction rule from the Task body is centralized
    /// here. NUL in `name` is forbidden by `validate_session_name` on the
    /// inbound `CreateSession` path, so the separator never aliases name
    /// content.
    pub fn add_path_name_member(&mut self, name: &str, session_id: &str) {
        self.path_name_members
            .insert(KeySchema::path_name_member(name, session_id));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a fully-populated `SessionCheckpoint` covering every field
    /// the Engineer's struct declares. The four `session_meta` hash
    /// fields (`name`, `base_path`, `created_by`, `created_at`) are
    /// cross-referenced against `SessionStore::create_once` in
    /// `src/storage/session.rs` (the `HSET` block at lines ~386-394).
    fn sample_checkpoint() -> SessionCheckpoint {
        let mut sc = SessionCheckpoint {
            schema_version: SESSION_CHECKPOINT_SCHEMA_VERSION,
            session_id: "sess-1".to_string(),
            name: "stage-name".to_string(),
            base_path: "stage:/projects/alpha".to_string(),
            created_by: "alice@example.com".to_string(),
            created_at: 1_700_000_000_000_000,
            layer_mappings: BTreeMap::from([
                ("root.usda".to_string(), 42_u64),
                ("sublayer.usda".to_string(), 43_u64),
            ]),
            layers_index: BTreeSet::from(["root.usda".to_string(), "sublayer.usda".to_string()]),
            path_name_members: BTreeSet::new(),
            path_index_session_id_member: "sess-1".to_string(),
        };
        // Exercise the wrapper the design centers on, NOT inline concat
        // and NOT a direct `KeySchema::path_name_member` call.
        sc.add_path_name_member("stage-name", "sess-1");
        sc
    }

    #[test]
    fn session_checkpoint_to_bytes_from_bytes_round_trip_preserves_all_fields() {
        let original = sample_checkpoint();
        let bytes = original.to_bytes().expect("to_bytes succeeds");
        let decoded = SessionCheckpoint::from_bytes(&bytes).expect("from_bytes succeeds");
        assert_eq!(decoded, original);
        // Spot-check every field individually so a future field addition
        // that breaks `PartialEq` semantics or that gets silently dropped
        // by a missing `#[serde(default)]` shows up here, not in the
        // catch-all `assert_eq!` above.
        assert_eq!(decoded.schema_version, SESSION_CHECKPOINT_SCHEMA_VERSION);
        assert_eq!(decoded.session_id, "sess-1");
        assert_eq!(decoded.name, "stage-name");
        assert_eq!(decoded.base_path, "stage:/projects/alpha");
        assert_eq!(decoded.created_by, "alice@example.com");
        assert_eq!(decoded.created_at, 1_700_000_000_000_000);
        assert_eq!(decoded.layer_mappings.len(), 2);
        assert_eq!(decoded.layer_mappings.get("root.usda"), Some(&42));
        assert_eq!(decoded.layer_mappings.get("sublayer.usda"), Some(&43));
        assert_eq!(decoded.layers_index.len(), 2);
        assert!(decoded.layers_index.contains("root.usda"));
        assert!(decoded.layers_index.contains("sublayer.usda"));
        assert_eq!(decoded.path_name_members.len(), 1);
        assert!(decoded
            .path_name_members
            .contains(&KeySchema::path_name_member("stage-name", "sess-1")));
        assert_eq!(decoded.path_index_session_id_member, "sess-1");
    }

    #[test]
    fn session_checkpoint_decode_known_good_fixture_matches_expected_struct() {
        // The `path_name_members` composite is rendered with its literal
        // six-character ` ` JSON escape (backslash, lowercase u, four
        // zero digits) so the on-the-wire encoding of
        // `KeySchema::path_name_member`'s `name\0session_id` output is
        // locked. Any future change to the NUL escape (e.g. emitting a
        // raw byte vs the JSON ` ` escape) will fail this test.
        //
        // The fixture is a regular (non-raw) string literal so the `\\`
        // escape sequence below is unambiguous: at compile time the
        // resulting `&str` contains the literal six characters
        // ` ` — exactly what `serde_json` emits for a NUL byte
        // inside a JSON string.
        const FIXTURE: &str = "{\"schema_version\":1,\"session_id\":\"sess-1\",\"name\":\"stage-name\",\"base_path\":\"stage:/projects/alpha\",\"created_by\":\"alice@example.com\",\"created_at\":1700000000000000,\"layer_mappings\":{\"root.usda\":42,\"sublayer.usda\":43},\"layers_index\":[\"root.usda\",\"sublayer.usda\"],\"path_name_members\":[\"stage-name\\u0000sess-1\"],\"path_index_session_id_member\":\"sess-1\"}";

        // Sanity-check that the fixture literally contains the
        // six-character ` ` escape (and NOT a raw NUL byte). If a
        // future edit accidentally collapses the escape to a literal
        // NUL, this assertion will fail before the decode panics on a
        // confusing JSON error.
        assert!(
            FIXTURE.contains("\\u0000"),
            "fixture must carry literal `\\u0000` JSON escape, not a raw NUL byte"
        );
        assert!(
            !FIXTURE.contains('\u{0000}'),
            "fixture must not contain a raw NUL byte"
        );

        let bytes = Bytes::from(FIXTURE.as_bytes().to_vec());
        let decoded = SessionCheckpoint::from_bytes(&bytes).expect("fixture decodes");

        assert_eq!(decoded.schema_version, 1);
        assert_eq!(decoded.session_id, "sess-1");
        assert_eq!(decoded.name, "stage-name");
        assert_eq!(decoded.base_path, "stage:/projects/alpha");
        assert_eq!(decoded.created_by, "alice@example.com");
        assert_eq!(decoded.created_at, 1_700_000_000_000_000);
        assert_eq!(decoded.layer_mappings.get("root.usda"), Some(&42));
        assert_eq!(decoded.layer_mappings.get("sublayer.usda"), Some(&43));
        assert_eq!(decoded.layer_mappings.len(), 2);
        assert!(decoded.layers_index.contains("root.usda"));
        assert!(decoded.layers_index.contains("sublayer.usda"));
        assert_eq!(decoded.layers_index.len(), 2);
        // The composite must decode to exactly the bytes
        // `KeySchema::path_name_member` produces — i.e. a raw NUL
        // between `name` and `session_id`. The JSON ` ` escape on
        // the wire round-trips into a literal NUL byte in the Rust
        // string.
        let expected_composite = KeySchema::path_name_member("stage-name", "sess-1");
        assert_eq!(expected_composite, "stage-name\u{0000}sess-1");
        assert!(decoded.path_name_members.contains(&expected_composite));
        assert_eq!(decoded.path_name_members.len(), 1);
        assert_eq!(decoded.path_index_session_id_member, "sess-1");
    }

    #[test]
    fn session_checkpoint_decode_with_unknown_field_tolerates_and_ignores() {
        // Includes every required field PLUS a `future_field_xyz` that
        // the current struct does not declare. Pins that no future
        // refactor turns on `#[serde(deny_unknown_fields)]` — that
        // would silently break older replicas reading newer blobs.
        const FIXTURE: &str = r#"{"schema_version":1,"session_id":"sess-1","name":"stage-name","base_path":"stage:/projects/alpha","created_by":"alice@example.com","created_at":1700000000000000,"layer_mappings":{},"layers_index":[],"path_name_members":[],"path_index_session_id_member":"sess-1","future_field_xyz":42}"#;

        let bytes = Bytes::from_static(FIXTURE.as_bytes());
        let decoded = SessionCheckpoint::from_bytes(&bytes)
            .expect("unknown field is tolerated, not rejected");
        assert_eq!(decoded.schema_version, 1);
        assert_eq!(decoded.session_id, "sess-1");
        assert_eq!(decoded.name, "stage-name");
        assert_eq!(decoded.base_path, "stage:/projects/alpha");
        assert_eq!(decoded.created_by, "alice@example.com");
        assert_eq!(decoded.created_at, 1_700_000_000_000_000);
        assert!(decoded.layer_mappings.is_empty());
        assert!(decoded.layers_index.is_empty());
        assert!(decoded.path_name_members.is_empty());
        assert_eq!(decoded.path_index_session_id_member, "sess-1");
    }

    #[test]
    fn session_checkpoint_decode_missing_schema_version_defaults_to_current() {
        // Omit `schema_version` entirely. The `#[serde(default = ...)]`
        // attribute MUST default it to `SESSION_CHECKPOINT_SCHEMA_VERSION`.
        const FIXTURE: &str = r#"{"session_id":"sess-1","name":"stage-name","base_path":"stage:/projects/alpha","created_by":"alice@example.com","created_at":1700000000000000,"layer_mappings":{},"layers_index":[],"path_name_members":[],"path_index_session_id_member":"sess-1"}"#;

        let bytes = Bytes::from_static(FIXTURE.as_bytes());
        let decoded = SessionCheckpoint::from_bytes(&bytes)
            .expect("missing schema_version defaults via #[serde(default)]");
        assert_eq!(decoded.schema_version, SESSION_CHECKPOINT_SCHEMA_VERSION);
    }

    #[test]
    fn session_checkpoint_from_bytes_garbage_returns_permanent_error() {
        let body = Bytes::from_static(b"not json");
        let err = SessionCheckpoint::from_bytes(&body).expect_err("garbage input must not decode");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "expected Permanent variant for malformed JSON, got: {err:?}"
        );
    }

    #[test]
    fn session_checkpoint_from_bytes_empty_bytes_returns_permanent_error() {
        // The adapter contract says `Ok(None)` for an absent session
        // (NotFound is returned at the store-level, not the codec
        // level). An empty-bytes payload reaching `from_bytes` is
        // therefore corruption — Permanent, not Transient, not NotFound.
        let body = Bytes::new();
        let err = SessionCheckpoint::from_bytes(&body).expect_err("empty bytes must not decode");
        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "expected Permanent variant for empty bytes, got: {err:?}"
        );
    }

    #[test]
    fn session_checkpoint_to_bytes_is_deterministic_for_equal_inputs() {
        // `BTreeMap` / `BTreeSet` serialize in sorted key order. Two
        // encodes of the same struct MUST produce byte-identical
        // output. This is the contract the known-good-fixture test
        // (#2) depends on and the foundation for any future
        // content-addressing of session-checkpoint blobs.
        let sc = sample_checkpoint();
        let a = sc.to_bytes().expect("first encode");
        let b = sc.to_bytes().expect("second encode");
        assert_eq!(a, b);
    }

    #[test]
    fn session_checkpoint_path_name_member_round_trips_through_split_helper() {
        // Populate `path_name_members` via `add_path_name_member` (the
        // sanctioned wrapper) — never via inline concat. Then prove
        // `KeySchema::split_path_name_member` recovers the original
        // `(name, session_id)` pair for every member. Pins the
        // producer/consumer identity across the serialization boundary.
        let mut sc = sample_checkpoint();
        sc.add_path_name_member("another-stage", "sess-2");
        sc.add_path_name_member("third-stage", "sess-3");

        let bytes = sc.to_bytes().expect("encode");
        let decoded = SessionCheckpoint::from_bytes(&bytes).expect("decode");

        // Expected mapping from member back to `(name, session_id)`.
        let expected: BTreeSet<(&str, &str)> = BTreeSet::from([
            ("stage-name", "sess-1"),
            ("another-stage", "sess-2"),
            ("third-stage", "sess-3"),
        ]);

        let recovered: BTreeSet<(&str, &str)> = decoded
            .path_name_members
            .iter()
            .map(|m| KeySchema::split_path_name_member(m).expect("every member splits on NUL"))
            .collect();
        assert_eq!(recovered, expected);
    }

    #[test]
    fn session_checkpoint_nul_in_name_corrupts_split_helper_recovery() {
        // Design intent (NOT a bug): production code MUST reject NUL in
        // session names BEFORE composing via `KeySchema::path_name_member`.
        // The inbound chokepoint is `validate_session_name`
        // (`src/domain/validation.rs`). If a corrupted blob ever carries
        // a composite whose `name` half contains a NUL, the split helper
        // splits on the FIRST NUL, leaving the rest of the original
        // name glued to the session_id half. This test makes that
        // detection surface visible — the wrong-split is observable to
        // a caller that checks the recovered session_id against the
        // expected pattern, instead of silently corrupting downstream
        // key construction.
        let member = KeySchema::path_name_member("foo\u{0000}bar", "sess-x");
        let recovered = KeySchema::split_path_name_member(&member);
        assert_eq!(recovered, Some(("foo", "bar\u{0000}sess-x")));
    }

    #[test]
    fn session_checkpoint_decode_with_higher_schema_version_is_forward_tolerant() {
        // Forward-tolerant policy (chosen here, NOT forward-reject): a
        // future v2 blob read by a v1 reader decodes successfully and
        // preserves the higher version number rather than silently
        // resetting it. The alternative — emitting `Err(Permanent)` on
        // an unknown `schema_version` — is NOT what this implementation
        // provides; any change to forward-reject must be a deliberate
        // policy decision that updates this test.
        const FIXTURE: &str = r#"{"schema_version":2,"session_id":"sess-1","name":"stage-name","base_path":"stage:/projects/alpha","created_by":"alice@example.com","created_at":1700000000000000,"layer_mappings":{},"layers_index":[],"path_name_members":[],"path_index_session_id_member":"sess-1"}"#;

        let bytes = Bytes::from_static(FIXTURE.as_bytes());
        let decoded = SessionCheckpoint::from_bytes(&bytes)
            .expect("higher schema_version is forward-tolerant");
        assert_eq!(decoded.schema_version, 2);
        assert!(
            decoded.schema_version > SESSION_CHECKPOINT_SCHEMA_VERSION,
            "fixture must exercise a version higher than current"
        );
    }
}
