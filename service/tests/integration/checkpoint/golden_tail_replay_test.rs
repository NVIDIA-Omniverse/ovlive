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

//! PRD AC-6 / SR-6.2 — golden tail-lineage replay against COMMITTED artifacts.
//!
//! Subtask `t3.dfh.yj.u3.tp` (Task 6, Epic 4). This is the only evidence class
//! in the Task that survives into future refactors, because its inputs are
//! frozen bytes: Task 1's unit lanes prove engine behaviour on hand-built
//! fixtures, `t3.dfh.yj.b7.x7` proves the hook wiring on a live-built lineage,
//! and this file proves **stability of stored artifacts over time**.
//!
//! # What each test pins
//!
//! | Test | Committed artifact(s) | Claim |
//! |---|---|---|
//! | `golden_base_and_tail_compose_to_the_committed_expected_state` | `base.ompf` + `tail_a.letl` vs `expected_state.json` | THE AC-6 gate. Replaying the stored lineage against today's engine yields the stored state on the faithful column basis. |
//! | `golden_chained_two_generation_lineage_composes_to_the_same_state` | `base.ompf` + `tail_b1.letl` + `tail_b2.letl` | The same span, split into two linked generations, composes to the identical state. |
//! | `golden_tail_artifact_bytes_are_stable_under_re_encoding` | all three `.letl` tails | The *byte* half: decode → re-encode is the identity, so a codec-layout change that leaves replay semantics intact but invalidates every stored artifact in the field still fails here. |
//! | `golden_tail_a_carries_the_empty_delta_record_shape` | `tail_a.letl` | The §8.4 `EmptyDelta` record really is in the artifact, so the gate's coverage of the empty-commit path is not vacuous. |
//! | `golden_fixture_at_an_unknown_format_version_rejects_permanent_and_is_attributed` | `tail_unknown_version.letl` (+ `tail_a.letl`) | SR-6.1: an unknown `format_version` rejects `Permanent`, is index-attributed, and is a step-down — never a retry. |
//! | `every_golden_fixture_has_a_named_test` | the whole directory | No artifact can be added without a test naming it. |
//!
//! # When this file goes red
//!
//! **A red run means apply/codec semantics changed under a stored artifact.**
//! That is a decision to make explicitly, not a file to edit. Do NOT edit
//! `expected_state.json` to match the new output, and do NOT regenerate the
//! artifacts "to fix the build". Establish first whether the semantic change is
//! intended and whether artifacts already written by deployed replicas remain
//! restorable; SR-6.3's compatibility statement (Task 7, `t2.dfh.yj.p5`) is
//! where that answer is recorded. Only then regenerate, deliberately:
//!
//! ```sh
//! cargo run -p live-edit --example generate_tail_lineage_goldens --features test-support
//! ```
//!
//! `tests/fixtures/tail_lineage/README.md` (generated) carries the provenance
//! table; `REGENERATE.md` (hand-written) carries the workflow.
//!
//! # Mutation verification — this gate has been seen to fire
//!
//! A gate nobody has ever watched fail is not evidence. Performed once, locally,
//! during `t3.dfh.yj.u3.tp`, then reverted (Task-6 Success Criterion):
//!
//! **Mutation.** In `apply_set_field` (`src/domain/delta.rs`), the `Field`
//! constructor's `authoring_set_order: delta_set_order` was changed to
//! `authoring_set_order: 0` — a one-line apply-semantics change that drops the
//! writer's true per-delta authoring rank while leaving structure, values,
//! sequence numbers and per-entry versions untouched, so nothing about
//! composition, the codec, or the lineage links moves.
//!
//! **Result — RED, on the intended assertion:**
//!
//! ```text
//! panicked at tests/integration/checkpoint/golden_tail_replay_test.rs:
//!   assertion `left == right` failed: single-generation golden: the composed
//!   fields must equal the committed golden on (value, authoring_set_order,
//!   tail-written version)
//! ```
//!
//! with every `GoldenField` in `left` reporting `authoring_set_order: 0`
//! against the committed non-zero ranks. Note the mutation is invisible to the
//! *cumulative* `set_order` column — which this basis deliberately excludes —
//! so the faithful column set is what caught it.
//!
//! **Revert.** The single line was restored; `src/domain/delta.rs`'s SHA-256
//! matched its pre-mutation value byte for byte
//! (`21a26f6a…394a1de3`) and `git status --porcelain -- src/` was empty. The
//! tree carries no residue of the experiment.
//!
//! # No Valkey, no feature gate
//!
//! `compose_from_artifacts` is pure — no I/O, no async, no clock — so this
//! module needs no backing store and its tests are plain `#[test]`. It is
//! deliberately **not** gated on any cargo feature: a gated module compiles to
//! nothing and reports green (test-writing guide § Tests anchored to generated
//! artifacts). Fixtures load through `include_bytes!` / `include_str!` for the
//! same reason — a missing fixture is a compile error, never a skip.
//!
//! AC-4's composition-equivalence proptests are the sibling lane's, in
//! `tests/proptests.rs`. No `proptest!` macro belongs in this file.

use std::collections::BTreeSet;
use std::path::PathBuf;

use bytes::Bytes;
use serde::Deserialize;

use live_edit::checkpoint::{
    compose_from_artifacts, decode_tail_envelope, encode_tail_envelope, CheckpointError,
    ComposeError, ComposedLineage, TailCodecError, TailRecord, TailSite, TAIL_FORMAT_VERSION,
    TAIL_FORMAT_VERSION_OFFSET,
};
use live_edit::domain::types::{FieldValue, ObjectState};

// ── Committed fixture bytes ─────────────────────────────────────────────────
//
// `include_bytes!` / `include_str!`, never `std::fs::read`: a fixture deleted or
// renamed must be a compile error. A runtime read would turn the same mistake
// into a skip or a panic in one test while the rest of the module stayed green.

const BASE_OMPF: &[u8] = include_bytes!("../../fixtures/tail_lineage/base.ompf");
const TAIL_A: &[u8] = include_bytes!("../../fixtures/tail_lineage/tail_a.letl");
const TAIL_B1: &[u8] = include_bytes!("../../fixtures/tail_lineage/tail_b1.letl");
const TAIL_B2: &[u8] = include_bytes!("../../fixtures/tail_lineage/tail_b2.letl");
const TAIL_UNKNOWN_VERSION: &[u8] =
    include_bytes!("../../fixtures/tail_lineage/tail_unknown_version.letl");
const EXPECTED_STATE_JSON: &str = include_str!("../../fixtures/tail_lineage/expected_state.json");

/// Every non-`.md` file the fixture directory is allowed to contain.
///
/// Keep in sync with the per-fixture tests below — adding an artifact to the
/// directory without adding it here is a test failure
/// (`every_golden_fixture_has_a_named_test`), mirroring
/// `oodf-ffi/tests/wire_fixtures.rs`'s `NAMED_SYNTHETIC_FIXTURES`.
const NAMED_GOLDEN_FIXTURES: &[&str] = &[
    "base.ompf",
    "expected_state.json",
    "tail_a.letl",
    "tail_b1.letl",
    "tail_b2.letl",
    "tail_unknown_version.letl",
];

/// `format_version` the generator forged into `tail_unknown_version.letl`.
///
/// SR-6.1's scenario is a *newer* build's artifact reaching an older one, so the
/// forged value is the next version rather than a nonsense number.
const FORGED_FORMAT_VERSION: u32 = 2;

// ── Canonical golden shape ──────────────────────────────────────────────────
//
// Mirrors the `Serialize` structs in
// `examples/generate_tail_lineage_goldens.rs`. `deny_unknown_fields` is the
// anti-drift device: a column added on the generator side without being added
// here fails to deserialize rather than being silently ignored, which would
// leave the gate quietly comparing fewer columns than it claims.

#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
enum GoldenValue {
    Inline { hex: String },
    External { hash_hex: String, size: u64 },
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GoldenSection {
    section_id: u64,
    parent_id: u64,
    name: String,
    section_type: u8,
    section_order: u32,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GoldenChildren {
    parent_id: u64,
    child_ids: Vec<u64>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GoldenField {
    section_id: u64,
    key_name: String,
    value: GoldenValue,
    authoring_set_order: u64,
    version: Option<u64>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GoldenTimeSample {
    section_id: u64,
    time_bits: u64,
    value: GoldenValue,
    version: Option<u64>,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GoldenExtValue {
    hash_hex: String,
    size: u64,
}

/// The committed expected state, on **exactly** the Task-6 amendment-2 faithful
/// basis: the columns a checkpoint round trip *carries*.
///
/// Deliberately EXCLUDED, each because it is re-derived rather than carried —
/// and each identical under a plain HEAD base-only restore, so a difference in
/// one would be evidence about the base decoder, not about composition:
///
/// - **cumulative `Field::set_order` and `ObjectState::next_set_order`** — the
///   reset OMPF carries the authoring rank in OODF's single `set_order` wire
///   slot, and `apply_set_field` re-accumulates the cursor from scratch. Its
///   per-key assignment additionally depends on the base's emission order rather
///   than on live's authoring interleaving, so it need not match live even
///   though it is now stable across regeneration (b.gcb / b.vz7).
/// - **base-era entry versions** — `decode_checkpoint_ompf` applies the base's
///   reset to a fresh state, so everything the base carried comes back stamped
///   `version = 1`. `GoldenField::version` / `GoldenTimeSample::version` are
///   therefore `Some` only for entries the TAIL wrote
///   (`version > base_snap_seq`), which is the sharpest available witness that
///   records replayed *in order* rather than merely all applied.
/// - **`ext_value_index`** on an external value — an OMPF-part ordinal
///   re-derived per encoding, not a carried column. Hash + size are compared.
/// - **`ObjectState::first_version`, `Section::first_version` /
///   `last_version`** — same flattening as base-era versions; the decoder
///   additionally zeroes the object-level `first_version`.
/// - **tombstones** — a reset delta describes current state and carries no
///   tombstone list by construction.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct GoldenState {
    schema_version: u32,
    object_id: u64,
    base_snap_seq: u64,
    composed_seq: u64,
    sections: Vec<GoldenSection>,
    children: Vec<GoldenChildren>,
    fields: Vec<GoldenField>,
    time_samples: Vec<GoldenTimeSample>,
    ext_values: Vec<GoldenExtValue>,
}

/// Schema version this file understands. Pinned so a generator-side shape bump
/// is a loud red rather than a silent partial comparison.
const EXPECTED_STATE_SCHEMA_VERSION: u32 = 1;

fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

fn golden_value(value: &FieldValue) -> GoldenValue {
    match value {
        FieldValue::Inline(bytes) => GoldenValue::Inline { hex: to_hex(bytes) },
        FieldValue::External { hash, size, .. } => GoldenValue::External {
            hash_hex: to_hex(hash),
            size: *size,
        },
    }
}

/// Render a composed lineage into the committed golden shape.
///
/// Duplicated from the generator on purpose — the two sides are an oracle and
/// its subject, and a shared helper would let one bug cancel the other out. The
/// `deny_unknown_fields` derives above are what keep the duplication honest.
fn canonicalize(composed: &ComposedLineage, expected: &GoldenState) -> GoldenState {
    let base_snap_seq = expected.base_snap_seq;
    let state: &ObjectState = &composed.state;

    let mut sections: Vec<GoldenSection> = state
        .sections
        .values()
        .map(|s| GoldenSection {
            section_id: s.section_id,
            parent_id: s.parent_id,
            name: s.name.clone(),
            section_type: s.section_type,
            section_order: s.section_order,
        })
        .collect();
    sections.sort_by_key(|s| s.section_id);

    let mut children: Vec<GoldenChildren> = state
        .children
        .iter()
        .map(|(parent_id, child_ids)| GoldenChildren {
            parent_id: *parent_id,
            // NOT sorted: the list order IS the authoring order and is this
            // column's whole payload.
            child_ids: child_ids.clone(),
        })
        .collect();
    children.sort_by_key(|c| c.parent_id);

    let mut fields: Vec<GoldenField> = state
        .fields
        .values()
        .map(|f| GoldenField {
            section_id: f.section_id,
            key_name: f.key_name.clone(),
            value: golden_value(&f.value),
            authoring_set_order: f.authoring_set_order,
            version: (f.version > base_snap_seq).then_some(f.version),
        })
        .collect();
    fields.sort_by(|a, b| (a.section_id, &a.key_name).cmp(&(b.section_id, &b.key_name)));

    let mut time_samples: Vec<GoldenTimeSample> = state
        .time_samples
        .values()
        .map(|t| GoldenTimeSample {
            section_id: t.section_id,
            time_bits: t.time.into_inner().to_bits(),
            value: golden_value(&t.value),
            version: (t.version > base_snap_seq).then_some(t.version),
        })
        .collect();
    time_samples.sort_by_key(|t| (t.section_id, t.time_bits));

    let ext_values: Vec<GoldenExtValue> = composed
        .ext_values
        .iter()
        .map(|(hash, data)| GoldenExtValue {
            hash_hex: to_hex(hash),
            size: data.len() as u64,
        })
        .collect();

    GoldenState {
        schema_version: EXPECTED_STATE_SCHEMA_VERSION,
        object_id: expected.object_id,
        base_snap_seq,
        composed_seq: composed.composed_seq,
        sections,
        children,
        fields,
        time_samples,
        ext_values,
    }
}

fn committed_expected_state() -> GoldenState {
    let expected: GoldenState = serde_json::from_str(EXPECTED_STATE_JSON).expect(
        "tests/fixtures/tail_lineage/expected_state.json must deserialize into GoldenState. A \
         serde error naming an unknown field means the generator emitted a column this file does \
         not compare — add it here rather than relaxing `deny_unknown_fields`",
    );
    assert_eq!(
        expected.schema_version, EXPECTED_STATE_SCHEMA_VERSION,
        "the committed golden's schema version moved; re-read the generator's column set before \
         trusting any comparison below"
    );
    expected
}

/// Compose the committed lineage, panicking with the compose error's own
/// per-generation attribution (it already names the site and record).
fn compose_committed(expected: &GoldenState, tails: &[&'static [u8]]) -> ComposedLineage {
    let artifacts: Vec<Bytes> = tails.iter().map(|t| Bytes::from_static(t)).collect();
    compose_from_artifacts(
        expected.object_id,
        BASE_OMPF,
        expected.base_snap_seq,
        &artifacts,
    )
    .unwrap_or_else(|e| {
        panic!(
            "the COMMITTED golden lineage must still compose against today's engine: {e}. A \
             failure here (rather than a state mismatch below) means the stored artifact is no \
             longer decodable or no longer replayable at all — the sharpest possible form of the \
             AC-6 regression"
        )
    })
}

/// Assert the composed state equals the committed golden, column by column so a
/// failure names which column moved before the whole-struct comparison dumps
/// everything.
fn assert_equals_committed_golden(
    composed: &ComposedLineage,
    expected: &GoldenState,
    context: &str,
) {
    let actual = canonicalize(composed, expected);

    assert_eq!(
        actual.composed_seq, expected.composed_seq,
        "{context}: the composed head sequence must equal the committed golden's"
    );
    assert_eq!(
        actual.sections, expected.sections,
        "{context}: the composed sections must equal the committed golden on \
         (section_id, parent_id, name, section_type, section_order)"
    );
    assert_eq!(
        actual.children, expected.children,
        "{context}: the composed children lists must equal the committed golden IN ORDER — the \
         list order is the authoring order, not an implementation detail"
    );
    assert_eq!(
        actual.fields, expected.fields,
        "{context}: the composed fields must equal the committed golden on \
         (value, authoring_set_order, tail-written version)"
    );
    assert_eq!(
        actual.time_samples, expected.time_samples,
        "{context}: the composed time samples must equal the committed golden on \
         (time, value, tail-written version)"
    );
    assert_eq!(
        actual.ext_values, expected.ext_values,
        "{context}: the composed external-value refs must equal the committed golden on \
         (hash, size), in first-appearance order"
    );

    // Belt-and-braces: a column added to `GoldenState` without a named
    // assertion above would still be compared here.
    assert_eq!(
        &actual, expected,
        "{context}: composed state differs from the committed golden"
    );
}

/// The gate is only worth its runtime if the committed golden actually carries
/// the shapes it claims to compare. Every one of these would otherwise let a
/// column pass vacuously.
fn assert_golden_is_non_vacuous(expected: &GoldenState) {
    assert!(
        expected.composed_seq > expected.base_snap_seq,
        "the golden's tail must advance the sequence past the base ({} vs {}), or the whole \
         replay half of this gate is vacuous",
        expected.composed_seq,
        expected.base_snap_seq
    );
    assert!(
        !expected.sections.is_empty(),
        "the golden must carry sections — it is built from 36 real captured Kit messages"
    );
    assert!(
        !expected.children.is_empty(),
        "the golden must carry children lists; their ORDER is one of the compared columns"
    );
    assert!(!expected.fields.is_empty(), "the golden must carry fields");
    assert!(
        expected.fields.iter().any(|f| f.version.is_some()),
        "the golden must carry at least one TAIL-written field (version > base_snap_seq). \
         Without one, the per-entry version column asserts nothing and a record replayed at the \
         wrong cursor position would pass"
    );
    assert!(
        expected.fields.iter().any(|f| f.authoring_set_order != 0),
        "the golden must carry at least one non-zero authoring rank, or the \
         `authoring_set_order` column is satisfied by any engine that drops it entirely"
    );
}

// ── The AC-6 gate ───────────────────────────────────────────────────────────

#[test]
fn golden_base_and_tail_compose_to_the_committed_expected_state() {
    let expected = committed_expected_state();
    assert_golden_is_non_vacuous(&expected);

    let composed = compose_committed(&expected, &[TAIL_A]);

    assert_equals_committed_golden(&composed, &expected, "single-generation golden");
}

#[test]
fn golden_chained_two_generation_lineage_composes_to_the_same_state() {
    let expected = committed_expected_state();

    // The link the chain rests on, asserted before the replay so a fixture that
    // drifted apart fails as a fixture rather than as a lineage gap.
    let b1 = decode_tail_envelope(&Bytes::from_static(TAIL_B1)).expect("tail_b1.letl must decode");
    let b2 = decode_tail_envelope(&Bytes::from_static(TAIL_B2)).expect("tail_b2.letl must decode");
    assert_eq!(
        b1.header().from_seq,
        expected.base_snap_seq,
        "the chain's first generation must open on the lineage base"
    );
    assert_eq!(
        b2.header().from_seq,
        b1.header().to_seq,
        "a later tail links onto its predecessor through from_seq == previous.to_seq (SR-2.1)"
    );
    assert_eq!(
        b2.header().base_snap_seq,
        expected.base_snap_seq,
        "a later tail still names the lineage's BASE, not its predecessor"
    );
    assert!(
        !b1.is_empty() && !b2.is_empty(),
        "both chained generations must carry records, or the split is not exercising a chain"
    );

    let composed = compose_committed(&expected, &[TAIL_B1, TAIL_B2]);

    assert_equals_committed_golden(&composed, &expected, "chained two-generation golden");
}

#[test]
fn golden_tail_artifact_bytes_are_stable_under_re_encoding() {
    // The byte half of AC-6. A codec-layout change can leave replay semantics
    // perfectly intact — every test above would stay green — while invalidating
    // every artifact already written by a deployed replica. This is what
    // catches it.
    for (name, artifact) in [
        ("tail_a.letl", TAIL_A),
        ("tail_b1.letl", TAIL_B1),
        ("tail_b2.letl", TAIL_B2),
    ] {
        let bytes = Bytes::from_static(artifact);
        let envelope = decode_tail_envelope(&bytes)
            .unwrap_or_else(|e| panic!("committed {name} must decode against this build: {e}"));
        assert_eq!(
            envelope.format_version(),
            TAIL_FORMAT_VERSION,
            "{name} must carry this build's format version"
        );
        let header = *envelope.header();
        let reencoded = encode_tail_envelope(&header, envelope.into_records())
            .unwrap_or_else(|e| panic!("re-encoding {name}'s own records must succeed: {e}"));
        assert_eq!(
            reencoded.as_ref(),
            artifact,
            "{name}: decode → re-encode must be the identity ({} bytes in, {} bytes out). A \
             mismatch means the envelope's on-disk layout moved, so every tail artifact already \
             durable in the field is now unreadable or differently-readable — a compatibility \
             decision, not a refactor",
            artifact.len(),
            reencoded.len()
        );
    }
}

#[test]
fn golden_tail_a_carries_the_empty_delta_record_shape() {
    // PM condition C5's non-vacuity guard. All 36 captured messages carry
    // non-empty payloads, so the generator issues one real
    // `update_layer(.., None, None, ..)` inside the tail's span. Without that
    // record the golden could not detect a composition engine that dropped
    // §8.4 empty commits, and the gate would silently cover one commit shape
    // fewer than it claims.
    let envelope =
        decode_tail_envelope(&Bytes::from_static(TAIL_A)).expect("tail_a.letl must decode");
    let empties: Vec<u64> = envelope
        .records()
        .iter()
        .filter(|r| matches!(r, TailRecord::EmptyDelta { .. }))
        .map(TailRecord::sequence_num)
        .collect();
    assert_eq!(
        empties.len(),
        1,
        "tail_a.letl must carry exactly one EmptyDelta record (the generator's single §8.4 empty \
         commit); found sequences {empties:?}"
    );
    let empty_seq = empties[0];
    assert!(
        empty_seq > envelope.header().from_seq && empty_seq < envelope.header().to_seq,
        "the EmptyDelta record must sit STRICTLY inside the span ({}, {}] — at an edge it would \
         not prove that replay carries the cursor correctly across an empty commit; it is at {}",
        envelope.header().from_seq,
        envelope.header().to_seq,
        empty_seq
    );
}

// ── SR-6.1: unknown format version ──────────────────────────────────────────

#[test]
fn golden_fixture_at_an_unknown_format_version_rejects_permanent_and_is_attributed() {
    // Provenance first: the artifact must be `tail_a.letl` with exactly the
    // four `format_version` bytes patched. That is what makes it
    // generator-reproducible rather than a hand-hexed blob nobody can rebuild —
    // `encode_tail_envelope` structurally cannot emit a non-current version.
    assert_eq!(
        TAIL_UNKNOWN_VERSION.len(),
        TAIL_A.len(),
        "tail_unknown_version.letl must be tail_a.letl with four bytes patched, so the lengths \
         must match"
    );
    let version_window = TAIL_FORMAT_VERSION_OFFSET..TAIL_FORMAT_VERSION_OFFSET + 4;
    let differing: Vec<usize> = TAIL_UNKNOWN_VERSION
        .iter()
        .zip(TAIL_A.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert!(
        !differing.is_empty() && differing.iter().all(|i| version_window.contains(i)),
        "tail_unknown_version.letl must differ from tail_a.letl ONLY inside the four-byte \
         `format_version` field at TAIL_FORMAT_VERSION_OFFSET ({TAIL_FORMAT_VERSION_OFFSET}); \
         differing offsets were {differing:?}. (Fewer than four offsets differ because the two \
         versions share leading zero bytes in little-endian — the field, not the byte count, is \
         the claim.)"
    );
    let read_version = |bytes: &[u8]| -> u32 {
        u32::from_le_bytes(
            bytes[version_window.clone()]
                .try_into()
                .expect("the version window is exactly four bytes"),
        )
    };
    assert_eq!(
        read_version(TAIL_A),
        TAIL_FORMAT_VERSION,
        "tail_a.letl must carry this build's format version"
    );
    assert_eq!(
        read_version(TAIL_UNKNOWN_VERSION),
        FORGED_FORMAT_VERSION,
        "tail_unknown_version.letl must carry the forged version"
    );

    let expected = committed_expected_state();
    let err = compose_from_artifacts(
        expected.object_id,
        BASE_OMPF,
        expected.base_snap_seq,
        &[Bytes::from_static(TAIL_UNKNOWN_VERSION)],
    )
    .expect_err(
        "a tail artifact at an unsupported format_version must be REJECTED, never guessed at \
         (SR-6.1) — this build cannot know a v2 payload's layout",
    );

    // The specific variant, never `is_err()`. `UnknownFormatVersion` is
    // index-attributed by design (`t3.dfh.yj.ae.e9`): the codec validates
    // `format_version` immediately after the magic and BEFORE reading `to_seq`,
    // because a future version's header may not even have the same layout — so
    // `TailSite::to_seq` is legitimately `None` here and the artifact's INDEX is
    // the attribution the step-down caller maps back to its handle.
    assert!(
        matches!(
            &err,
            ComposeError::TailDecode {
                site: TailSite {
                    index: 0,
                    to_seq: None
                },
                source: TailCodecError::UnknownFormatVersion {
                    got: FORGED_FORMAT_VERSION,
                    supported: TAIL_FORMAT_VERSION,
                },
            }
        ),
        "expected ComposeError::TailDecode{{ site: tail[0] to_seq=unknown, source: \
         UnknownFormatVersion{{ got: {FORGED_FORMAT_VERSION}, supported: {TAIL_FORMAT_VERSION} }} \
         }}, got {err:?}"
    );

    // The step-down verdict the restore loop consumes. `Permanent` means
    // "this generation's lineage is unrestorable — step down to the next older
    // restorable generation"; it must NEVER be `Transient`, which would send
    // the loop back at the same immutable bytes forever.
    let classified = CheckpointError::from(err);
    assert!(
        classified.is_permanent(),
        "every compose failure converts to CheckpointError::Permanent — the engine performs no \
         I/O, so a retry re-reads the same bytes and reaches the same verdict. Got {classified:?}"
    );
    assert!(
        !classified.is_transient(),
        "an unknown format_version must never be classified retriable: {classified:?}"
    );
}

// ── Meta-test: no orphan fixtures ───────────────────────────────────────────

#[test]
fn every_golden_fixture_has_a_named_test() {
    // Modelled on `oodf-ffi/tests/wire_fixtures.rs`. `CARGO_MANIFEST_DIR` is
    // the live-edit crate root, which is the repo root.
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("tail_lineage");

    let mut on_disk: BTreeSet<String> = BTreeSet::new();
    for entry in std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("fixture dir {} must exist: {e}", dir.display()))
    {
        let entry = entry.expect("readable fixture dir entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        // `.md` is documentation — the generated README, the hand-written
        // REGENERATE workflow, and the per-fixture sidecars. Everything else is
        // an artifact and must be named by a test.
        if !name.ends_with(".md") {
            on_disk.insert(name);
        }
    }

    let named: BTreeSet<String> = NAMED_GOLDEN_FIXTURES
        .iter()
        .map(|s| s.to_string())
        .collect();

    let orphans: Vec<&String> = on_disk.difference(&named).collect();
    assert!(
        orphans.is_empty(),
        "golden artifact(s) present with no named test in golden_tail_replay_test.rs: \
         {orphans:?}. Add an explicit test that loads the file via `include_bytes!`, a `.md` \
         sidecar recording its provenance, and the file name to NAMED_GOLDEN_FIXTURES"
    );

    let missing: Vec<&String> = named.difference(&on_disk).collect();
    assert!(
        missing.is_empty(),
        "NAMED_GOLDEN_FIXTURES entries with no backing file on disk: {missing:?}"
    );

    // Every artifact must also carry its `.md` sidecar — the provenance
    // discipline copied from `tests/fixtures/oodf/`.
    for name in NAMED_GOLDEN_FIXTURES {
        let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(name);
        let sidecar = dir.join(format!("{stem}.md"));
        assert!(
            sidecar.exists(),
            "artifact `{name}` has no `{stem}.md` provenance sidecar. The generator writes one \
             per artifact; a missing file means the artifact was added by hand"
        );
    }
}
