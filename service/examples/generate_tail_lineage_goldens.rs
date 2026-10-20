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

//! Deterministic generator for the AC-6 / SR-6.2 golden tail-lineage fixtures
//! under `tests/fixtures/tail_lineage/` (Subtask `t3.dfh.yj.u3.tp`).
//!
//! Run by hand; its output is COMMITTED and replayed by
//! `tests/integration/checkpoint/golden_tail_replay_test.rs`:
//!
//! ```sh
//! cargo run -p live-edit --example generate_tail_lineage_goldens --features test-support
//! ```
//!
//! It needs a reachable `TEST_VALKEY_URL` (the sandbox presets
//! `redis://valkey:6379`); the replay test does NOT — `compose_from_artifacts`
//! is pure, so only fixture *production* touches a backing store.
//!
//! # Why real captured traffic
//!
//! The input is `tools/fixtures/gze_inbound_deltas.json` — 36 real captured
//! Kit inbound messages (2 × `CreateLayer`, 34 × `UpdateLayer`), each entry's
//! `b64` being the writer's raw inbound OMPF, which is exactly the payload a
//! tail `Verbatim` record carries. Real captured traffic is mandatory here:
//! synthetic single-section fixtures hid b.gze and b.zju.
//!
//! # The pipeline, and why each step is the shape it is
//!
//! 1. **Both `CreateLayer` messages replay, in order, onto the same
//!    `(session, layer)` pair** — exactly what the internal acceptance
//!    harness's replay of this history does. The second supersedes the first,
//!    so the object the lineage covers is the second one, and the
//!    superseded-overwrite path is exercised on the way there rather than
//!    papered over.
//! 2. **Every commit goes through `LayerStore::update_layer` on the DEFAULT
//!    `lua` backend, with the captured bytes passed as `relay_ompf` verbatim.**
//!    The Lua fast path does NOT call `apply_delta` — it computes
//!    seq / version / stored_set_order / next_set_order itself (Task-6
//!    amendment 1) — so a fixture built by calling `apply_delta` directly would
//!    encode the replay engine's own answer and prove nothing.
//! 3. **The base is a REAL `assemble_layer_checkpoint` product**, written into
//!    an `InMemoryCheckpointStore` and read straight back out. Not a hand-built
//!    reset OMPF.
//! 4. **One real §8.4 empty commit** (`update_layer(.., None, None, ..)`) is
//!    issued inside the tail's span. All 36 captured messages carry non-empty
//!    payloads, so without it the `EmptyDelta` record shape — and the
//!    "composition skips empty-delta records" mutation candidate — would be
//!    unrepresented in the golden (PM condition C2/C5).
//! 5. **Fixture-truthfulness precondition (PM condition F3).** Before any tail
//!    is encoded, the generator XRANGEs `le:{oid}:obj:stream` and asserts each
//!    recorded commit's bytes equal that sequence's stream entry `ompf` field —
//!    and that the empty commit's entry carries no `ompf` field at all. The
//!    b.gze lesson applied at the fixture layer: an `ompf`-less `update` entry
//!    is classified as an empty commit *by invariant*, so a fixture that lost
//!    the relay bytes would still encode cleanly while recording real commits
//!    as no-ops (the b.pw6 defect class).
//!
//! # Normalisations that make the artifacts reproducible
//!
//! - **`object_id`.** The live run's id is whatever `le:counter:object_id`
//!   hands out. Every emitted tail header carries [`GOLDEN_OBJECT_ID`] instead,
//!   and `compose_from_artifacts` takes `object_id` as a parameter, so nothing
//!   else in the golden depends on the run's real id. Sound because the header's
//!   `object_id` is a lineage-identity check (`ComposeError::ObjectIdMismatch`),
//!   not an input to replay.
//! - **Record-level `server_ts`.** A wall-clock `i64` serialized INTO the
//!   21-byte record prefix, so leaving it live would make every regeneration
//!   byte-different. Normalised to [`HAND_PUT_SERVER_TS`]. Sound because the
//!   restore path never reads a record's `server_ts` — the value restore
//!   consumes is `CheckpointMeta::server_ts`, a different field on a different
//!   struct.
//!
//! Those two are the whole normalisation list. In particular the **base OMPF
//! needs no normalising**: `build_reset_ompf_from_state` walks
//! `DiffEngine::plan_full_content`, which emits fields, time samples, and orphan
//! sections in a total order over the CONTENT of `ObjectState` — fields on
//! `(version, authoring_set_order, section_id, key_name)`, time samples on
//! `(version, section_id, time)`, the orphan tail on `section_id` — so identical
//! state encodes to identical bytes, in this process or any other (b.gcb /
//! b.vz7, Epic-5 determinism Task). `base.ompf` is therefore byte-reproducible
//! like every other artifact here, and its README SHA-256 row is a stability
//! guarantee rather than a snapshot of one arbitrary encoding.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine as _;
use bytes::Bytes;
use fred::interfaces::StreamsInterface;
use serde::Serialize;
use sha2::{Digest, Sha256};

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::assembler::{
    assemble_layer_checkpoint, AssemblyDecision, NoLeaseToKeep,
};
use live_edit::checkpoint::{
    compose_from_artifacts, encode_tail_envelope, CheckpointStore, ComposedLineage, TailHeader,
    TailRecord, TAIL_FORMAT_VERSION, TAIL_FORMAT_VERSION_OFFSET,
};
use live_edit::domain::delta::apply_delta;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::{FieldValue, ObjectState};
use live_edit::storage::external_values::ValkeyExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::notifications::RELAY_OMPF_FIELD;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

// ── Normalisation + shape constants ─────────────────────────────────────────

/// The license header every emitted Markdown file (the sidecars and the
/// README) starts with, then the blank line before its body: NVIDIA's
/// Apache-2.0 header in the Markdown form `tools/license_headers.py` renders
/// (`new_header("html", 2026)`), an HTML comment that renders nothing. The
/// year is fixed, so a regeneration reproduces the committed files byte for
/// byte; `tools/tests/test_license_headers.py` pins this text to the tool's.
const MARKDOWN_LICENSE_HEADER: &str = r#"<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

"#;

/// The `object_id` stamped into every emitted tail header, replacing the run's
/// real allocated id. See the module header's normalisation note.
const GOLDEN_OBJECT_ID: u64 = 90_100_200;

/// Record-level `server_ts` stamped into every emitted record.
///
/// Deliberately the same literal as
/// `tests/integration/checkpoint/helpers.rs::HAND_PUT_SERVER_TS`, so a reader
/// comparing a golden artifact against a hand-put one sees one timestamp, not
/// two. Duplicated rather than imported: an `examples/` target cannot reach the
/// integration-test tree.
const HAND_PUT_SERVER_TS: i64 = 1_700_000_000_000_000;

/// Number of captured `UpdateLayer` messages committed BEFORE the base is
/// assembled. The base therefore lands at `1 + BASE_SPLIT_UPDATE_COUNT`
/// (`CreateLayer` seeds the object at sequence 1).
///
/// Eight is the capture's own first natural boundary — captured updates 1..8
/// span `04:23:19.979 .. 04:23:42.766`, after which the traffic collapses into
/// the 26-message 176-byte burst. Splitting there puts real structural work on
/// BOTH sides of the base: the large 1568 B / 1160 B messages are inside the
/// base, and the burst plus the two large trailing messages are inside the tail.
const BASE_SPLIT_UPDATE_COUNT: usize = 8;

/// The empty commit is issued after this many captured `UpdateLayer` messages
/// have been committed — i.e. late in the burst, with two more captured
/// messages still to follow it, so an `EmptyDelta` record sits strictly INSIDE
/// the tail rather than at its edge.
const EMPTY_COMMIT_AFTER_UPDATE_COUNT: usize = 32;

/// Where the chained (two-generation) golden splits its tails: `tail_b1` covers
/// `(base_snap_seq, CHAIN_SPLIT_SEQ]` and `tail_b2` covers
/// `(CHAIN_SPLIT_SEQ, head]`. Chosen mid-burst so both generations carry real
/// captured records.
const CHAIN_SPLIT_SEQ: u64 = 20;

/// `format_version` forged into `tail_unknown_version.letl`.
///
/// `encode_tail_envelope` always stamps [`TAIL_FORMAT_VERSION`] and offers no
/// version parameter — structurally, this build cannot mint an artifact it
/// cannot read. The artifact is therefore produced by patching the four bytes at
/// the exported [`TAIL_FORMAT_VERSION_OFFSET`], which is rustdoc'd for exactly
/// this use. `2` rather than a nonsense value because SR-6.1's scenario is a
/// *newer* build's artifact reaching an older one.
const UNKNOWN_FORMAT_VERSION: u32 = 2;

/// Schema version of `expected_state.json`. Bump when the golden's column set
/// changes; the replay test pins it so a silent shape drift is a red test.
const EXPECTED_STATE_SCHEMA_VERSION: u32 = 1;

const SESSION_NAME: &str = "golden-tail-lineage-session";
const SESSION_PATH: &str = "/golden/tail-lineage";
const LAYER_NAME: &str = "golden-tail-lineage-layer";
const CREATED_BY: &str = "golden-tail-lineage-generator";
const WRITER_CLIENT_ID: &str = "golden-tail-lineage-writer";
const SERVICE_LABEL: &str = "golden-tail-lineage-generator";

/// Stream retention for the generator's `LayerStore`. Far above the ~37 entries
/// this run writes, so `MAXLEN ~` trimming can never reach an entry the
/// truthfulness probe needs. The store also turns the relay byte budget off
/// (`with_layer_stream_max_bytes(0)`, `LIVE_EDIT_LAYER_STREAM_MAX_BYTES=0`), so
/// its `XTRIM MINID ~` cannot reach one either, whatever the source's sizes.
const STREAM_MAXLEN: u64 = 1_000_000;

// ── Canonical golden-state shape ────────────────────────────────────────────
//
// Mirrored, field for field, by the `Deserialize` structs in
// `tests/integration/checkpoint/golden_tail_replay_test.rs`. That file's structs
// carry `#[serde(deny_unknown_fields)]`, so adding a column here without adding
// it there is a red test rather than a silent divergence.

/// A field / time-sample value on the faithful basis.
///
/// `ext_value_index` is deliberately absent: it is an OMPF-part ordinal
/// re-derived per encoding, not a carried column.
#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum GoldenValue {
    Inline { hex: String },
    External { hash_hex: String, size: u64 },
}

#[derive(Debug, Serialize, PartialEq)]
struct GoldenSection {
    section_id: u64,
    parent_id: u64,
    name: String,
    section_type: u8,
    section_order: u32,
}

#[derive(Debug, Serialize, PartialEq)]
struct GoldenChildren {
    parent_id: u64,
    child_ids: Vec<u64>,
}

#[derive(Debug, Serialize, PartialEq)]
struct GoldenField {
    section_id: u64,
    key_name: String,
    value: GoldenValue,
    authoring_set_order: u64,
    /// The commit's own sequence number, present only for entries the TAIL
    /// wrote (`version > base_snap_seq`). `null` for base-era entries, whose
    /// versions the base decoder flattens to `1` — identical to a HEAD
    /// base-only restore and therefore no evidence about composition.
    version: Option<u64>,
}

#[derive(Debug, Serialize, PartialEq)]
struct GoldenTimeSample {
    section_id: u64,
    /// `f64::to_bits` of the sample time. Exact by construction; a decimal
    /// rendering would put a float-formatting round trip inside the contract.
    time_bits: u64,
    value: GoldenValue,
    version: Option<u64>,
}

#[derive(Debug, Serialize, PartialEq)]
struct GoldenExtValue {
    hash_hex: String,
    size: u64,
}

/// The committed expected state, limited to the amendment-2 faithful basis.
///
/// EXCLUDED, each because it is re-derived rather than carried: cumulative
/// `Field::set_order`, `ObjectState::next_set_order`, base-era entry versions,
/// `FieldValue::External::ext_value_index`, `ObjectState::first_version`, and
/// `Section::first_version` / `last_version`.
#[derive(Debug, Serialize, PartialEq)]
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

fn to_hex(bytes: &[u8]) -> String {
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

/// Canonicalise a composed lineage into the committed golden shape.
///
/// Every collection is sorted on a total key so the rendering does not inherit
/// `HashMap` iteration order.
fn canonicalize(composed: &ComposedLineage, base_snap_seq: u64) -> GoldenState {
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
            // NOT sorted: the list order IS the authoring order and is the
            // payload of this column.
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
        object_id: GOLDEN_OBJECT_ID,
        base_snap_seq,
        composed_seq: composed.composed_seq,
        sections,
        children,
        fields,
        time_samples,
        ext_values,
    }
}

// ── Captured-traffic input ──────────────────────────────────────────────────

/// One captured inbound message.
struct CapturedMessage {
    index: usize,
    kind: String,
    ompf: Bytes,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn load_captured_traffic(path: &Path) -> (Vec<CapturedMessage>, String) {
    let raw = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let source_sha = to_hex(&Sha256::digest(&raw));
    let parsed: serde_json::Value =
        serde_json::from_slice(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    let entries = parsed
        .as_array()
        .unwrap_or_else(|| panic!("{} must be a JSON array", path.display()));

    let messages = entries
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let kind = entry["kind"]
                .as_str()
                .unwrap_or_else(|| panic!("entry {index} has no `kind`"))
                .to_string();
            let b64 = entry["b64"]
                .as_str()
                .unwrap_or_else(|| panic!("entry {index} has no `b64`"));
            let decoded = base64::engine::general_purpose::STANDARD
                .decode(b64)
                .unwrap_or_else(|e| panic!("entry {index} b64 decode failed: {e}"));
            let declared = entry["len"].as_u64().unwrap_or(decoded.len() as u64);
            assert_eq!(
                decoded.len() as u64,
                declared,
                "entry {index}: decoded length must match the capture's recorded `len`"
            );
            CapturedMessage {
                index,
                kind,
                ompf: Bytes::from(decoded),
            }
        })
        .collect();

    (messages, source_sha)
}

/// Parse one captured OMPF into the `OwnedDelta` the live handler would have
/// applied — the same pipeline `handle_create_layer` / `handle_update_layer`
/// run (`OmpfContainer::parse` → `oodf_ffi::parse_delta`, with the empty-payload
/// branch yielding `None`).
fn parse_captured(message: &CapturedMessage) -> Option<oodf_ffi::OwnedDelta> {
    let container = OmpfContainer::parse(&message.ompf)
        .unwrap_or_else(|e| panic!("captured entry {} is not a valid OMPF: {e}", message.index));
    if container.delta_bytes.is_empty() {
        return None;
    }
    Some(
        oodf_ffi::parse_delta(container.delta_bytes).unwrap_or_else(|e| {
            panic!(
                "captured entry {} carries an OODF payload the verifier rejects: {e}",
                message.index
            )
        }),
    )
}

// ── Commit bookkeeping ──────────────────────────────────────────────────────

/// A commit the generator landed, in the shape a tail record needs.
#[derive(Clone)]
enum Commit {
    /// A captured `UpdateLayer`, carrying the writer's raw inbound OMPF.
    Verbatim { sequence_num: u64, ompf: Bytes },
    /// The one real §8.4 empty commit.
    Empty { sequence_num: u64 },
}

impl Commit {
    fn sequence_num(&self) -> u64 {
        match self {
            Commit::Verbatim { sequence_num, .. } | Commit::Empty { sequence_num } => *sequence_num,
        }
    }

    fn to_record(&self) -> TailRecord {
        match self {
            Commit::Verbatim { sequence_num, ompf } => TailRecord::Verbatim {
                sequence_num: *sequence_num,
                server_ts: HAND_PUT_SERVER_TS,
                ompf: ompf.clone(),
            },
            Commit::Empty { sequence_num } => TailRecord::EmptyDelta {
                sequence_num: *sequence_num,
                server_ts: HAND_PUT_SERVER_TS,
            },
        }
    }
}

/// PM condition F3 — the fixture-truthfulness precondition.
///
/// Every commit the generator is about to freeze into an artifact must match
/// what the per-object stream actually recorded at that sequence: a verbatim
/// commit's `ompf` field byte-for-byte, and the empty commit's ABSENCE of that
/// field (its absence is the entire classification signal, so a fixture whose
/// empty commit accidentally carried relay bytes — or whose real commit lost
/// them — would encode cleanly while meaning something else).
async fn assert_stream_matches_commits(tv: &TestValkey, object_id: u64, commits: &[Commit]) {
    type StreamEntry = (String, HashMap<String, Bytes>);
    let entries: Vec<StreamEntry> = tv
        .client()
        .pool
        .xrange_values(tv.key_schema().layer_stream(object_id), "-", "+", None)
        .await
        .expect("XRANGE over the per-object stream must succeed");
    let by_id: HashMap<&str, &HashMap<String, Bytes>> = entries
        .iter()
        .map(|(id, fields)| (id.as_str(), fields))
        .collect();

    for commit in commits {
        let id = format!("{}-0", commit.sequence_num());
        let fields = by_id.get(id.as_str()).unwrap_or_else(|| {
            panic!(
                "no stream entry at `{id}`; the tail fixture would claim coverage the live \
                 stream never recorded. Entry ids present: {:?}",
                entries.iter().map(|(i, _)| i).collect::<Vec<_>>()
            )
        });
        match commit {
            Commit::Verbatim { ompf, .. } => {
                let stored = fields.get(RELAY_OMPF_FIELD).unwrap_or_else(|| {
                    panic!(
                        "stream entry `{id}` carries no `{RELAY_OMPF_FIELD}` field, so tail \
                         capture would classify this REAL commit as a §8.4 empty one (the \
                         b.pw6 defect class). Fields present: {:?}",
                        fields.keys().collect::<Vec<_>>()
                    )
                });
                assert_eq!(
                    stored.as_ref(),
                    ompf.as_ref(),
                    "stream entry `{id}` must carry the writer's inbound OMPF byte-for-byte \
                     (stored {} bytes, recorded {} bytes)",
                    stored.len(),
                    ompf.len()
                );
            }
            Commit::Empty { .. } => {
                assert!(
                    !fields.contains_key(RELAY_OMPF_FIELD),
                    "stream entry `{id}` is the generator's §8.4 empty commit and MUST carry no \
                     `{RELAY_OMPF_FIELD}` field — its absence is the classification signal. \
                     Fields present: {:?}",
                    fields.keys().collect::<Vec<_>>()
                );
            }
        }
    }
}

// ── Emission ────────────────────────────────────────────────────────────────

struct Artifact {
    name: &'static str,
    bytes: Vec<u8>,
    purpose: &'static str,
    sidecar: String,
}

fn write_artifacts(dir: &Path, artifacts: &[Artifact]) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("mkdir {}: {e}", dir.display()));
    for artifact in artifacts {
        let path = dir.join(artifact.name);
        std::fs::write(&path, &artifact.bytes)
            .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        let stem = artifact
            .name
            .rsplit_once('.')
            .map(|(s, _)| s)
            .unwrap_or(artifact.name);
        let sidecar_path = dir.join(format!("{stem}.md"));
        std::fs::write(
            &sidecar_path,
            format!("{MARKDOWN_LICENSE_HEADER}{}", artifact.sidecar),
        )
        .unwrap_or_else(|e| panic!("write {}: {e}", sidecar_path.display()));
    }
}

#[allow(clippy::too_many_arguments)]
fn render_readme(
    artifacts: &[Artifact],
    source_sha: &str,
    base_snap_seq: u64,
    head_seq: u64,
    tail_record_count: usize,
    empty_commit_seq: u64,
    ext_value_count: usize,
    real_object_id: u64,
) -> String {
    let mut table = String::new();
    for a in artifacts {
        let _ = writeln!(
            table,
            "| `{}` | {} | `{}` | {} |",
            a.name,
            a.bytes.len(),
            to_hex(&Sha256::digest(&a.bytes)),
            a.purpose
        );
    }

    format!(
        r#"# Golden tail-lineage fixtures

<!-- GENERATED FILE — do not hand-edit. Rewritten in full by
     `cargo run -p live-edit --example generate_tail_lineage_goldens --features test-support`.
     Prose lives in that example's `render_readme`. -->

Committed persisted-tail artifacts, built from **real captured writer traffic**,
replayed against the current engine by
`tests/integration/checkpoint/golden_tail_replay_test.rs` and asserted
byte/state-stable. A change to `apply_delta` (or codec) semantics that alters
replay of a committed fixture fails CI and forces an explicit decision, rather
than silently redefining what a stored artifact means.

See [`REGENERATE.md`](./REGENERATE.md) for the regeneration workflow, the
sidecar/meta-test registration steps, and what to do when this gate goes red.

## Provenance

| Item | Value |
|---|---|
| Source traffic | `tools/fixtures/gze_inbound_deltas.json` |
| Source SHA-256 | `{source_sha}` |
| Source shape | 36 real captured Kit inbound messages — 2 × `CreateLayer`, 34 × `UpdateLayer`, 128 B – 1.6 KB each |
| Generator | `cargo run -p live-edit --example generate_tail_lineage_goldens --features test-support` |
| Generator source | `examples/generate_tail_lineage_goldens.rs` |
| Commit path | `LayerStore::update_layer`, DEFAULT `lua` backend, captured bytes passed as `relay_ompf` verbatim |
| Base produced by | `assemble_layer_checkpoint(.., &mut NoLeaseToKeep)` into an `InMemoryCheckpointStore` |
| Split point | base assembled after {BASE_SPLIT_UPDATE_COUNT} captured `UpdateLayer` messages → `base_snap_seq = {base_snap_seq}` |
| Tail span | `({base_snap_seq}, {head_seq}]`, {tail_record_count} records |
| Empty commit | one real `update_layer(.., None, None, ..)` at sequence {empty_commit_seq} (the empty-delta shape) |
| Chained split | `tail_b1` covers `({base_snap_seq}, {CHAIN_SPLIT_SEQ}]`, `tail_b2` covers `({CHAIN_SPLIT_SEQ}, {head_seq}]` |
| External values | {ext_value_count} referenced by the composed lineage |

## Normalisations — what makes regeneration reproducible

| Normalised | To | Why it is sound |
|---|---|---|
| Tail header `object_id` (run value was `{real_object_id}`) | `{GOLDEN_OBJECT_ID}` | The live id comes from `le:counter:object_id` and differs per run. `compose_from_artifacts` takes `object_id` as a parameter, and the header field is only a lineage-identity check (`ComposeError::ObjectIdMismatch`) — never an input to replay. |
| Record-level `server_ts` | `{HAND_PUT_SERVER_TS}` | It is a wall-clock `i64` serialized INTO the 21-byte record prefix, so an unnormalised artifact differs byte-for-byte on every regeneration. The restore path never reads a record's `server_ts`; the value restore consumes is `CheckpointMeta::server_ts`, a different field on a different struct. |

### Nothing else needs normalising — including `base.ompf`

`base.ompf` used to sit here as the one documented exception: its field emission
order came off a `HashMap` walk, so regeneration produced a different valid
encoding of the same state (and, because FlatBuffers vtable sharing and offset
widths follow emission order, sometimes a different *length*). That is fixed at
the source rather than contained here. `DiffEngine::plan_full_content` now emits
fields on `(version, authoring_set_order, section_id, key_name)`, time samples on
`(version, section_id, time)`, and the orphan-section tail on `section_id` — each
a total order over the CONTENT of `ObjectState`, never over a hash table's
internal layout. Identical state now
encodes to identical bytes in any process, so this file's `base.ompf` SHA-256 row
is a byte-stability guarantee rather than a snapshot of one arbitrary encoding.

**Measured, not argued** — 2026-09-02, this regeneration: the generator was run
twice as two SEPARATE processes (each gets its own thread-local `RandomState`
seed base, which is precisely the divergence two replicas' maps carry), and all
six artifacts came back byte-identical — `base.ompf` included, at the byte count
and SHA-256 in the table below. The four `.letl` tails and `expected_state.json`
were already byte-stable before this change and remain so: `encode_tail_envelope`
is byte-deterministic with every non-deterministic input normalised in
the table above, and `canonicalize` sorts every collection on a total key.
`base.ompf` joined them with the emission-order fix.

Consequence when reviewing a regeneration diff: `base.ompf` changing is no longer
the expected, harmless case. It now means the composed state genuinely moved or
the emission order regressed — see `REGENERATE.md` § 4 and § 6.

## Artifacts

| File | Bytes | SHA-256 | Purpose |
|---|---|---|---|
{table}
Each artifact has a sibling `.md` sidecar. Every non-`.md` file in this
directory must be named by a test in `golden_tail_replay_test.rs`; the
`every_golden_fixture_has_a_named_test` meta-test fails otherwise.

## Faithful comparison basis

`expected_state.json` renders exactly the columns a checkpoint round trip
*carries*:

* sections — `section_id`, `parent_id`, `name`, `section_type`, `section_order`
* children lists, **in order** (the order is the authoring order, not an
  implementation detail)
* fields — `(section_id, key_name)` → value + `authoring_set_order`, plus the
  per-entry `version` for tail-written entries only
* time samples — `(section_id, time)` → value, plus tail-written `version`
* external-value refs by hash + size

Deliberately excluded, each because it is *re-derived* rather than carried, and
each identical under a plain HEAD base-only restore: cumulative
`Field::set_order`, `ObjectState::next_set_order`, base-era entry versions,
`FieldValue::External::ext_value_index`, `ObjectState::first_version`, and
`Section::first_version` / `last_version`.
"#
    )
}

// ── Main ────────────────────────────────────────────────────────────────────

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let root = repo_root();
    let source_path = root
        .join("tools")
        .join("fixtures")
        .join("gze_inbound_deltas.json");
    let out_dir = root.join("tests").join("fixtures").join("tail_lineage");

    let (messages, source_sha) = load_captured_traffic(&source_path);
    let creates: Vec<&CapturedMessage> = messages
        .iter()
        .filter(|m| m.kind == "CreateLayer")
        .collect();
    let updates: Vec<&CapturedMessage> = messages
        .iter()
        .filter(|m| m.kind == "UpdateLayer")
        .collect();
    assert!(
        !creates.is_empty(),
        "the captured traffic must contain at least one CreateLayer to seed the layer"
    );
    assert!(
        updates.len() > BASE_SPLIT_UPDATE_COUNT + 2,
        "the captured traffic must carry enough UpdateLayers to sit on both sides of the split"
    );
    println!(
        "source: {} ({} messages: {} CreateLayer, {} UpdateLayer)\n  sha256={source_sha}",
        source_path.display(),
        messages.len(),
        creates.len(),
        updates.len()
    );

    let tv = TestValkey::setup().await;
    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), STREAM_MAXLEN)
        .with_layer_stream_max_bytes(0);
    let ext_store = Arc::new(ValkeyExternalValueStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(SESSION_NAME, SESSION_PATH, CREATED_BY)
        .await
        .expect("session creation must succeed");

    // ── 1. Replay both CreateLayer messages onto the same (session, layer) ──
    // The second supersedes the first; the surviving object is the lineage's.
    let mut object_id = 0u64;
    for message in &creates {
        let mut state = ObjectState::default();
        apply_delta(&mut state, parse_captured(message)).unwrap_or_else(|e| {
            panic!(
                "captured CreateLayer at entry {} was rejected by apply_delta: {e}",
                message.index
            )
        });
        object_id = layer_store
            .create_layer(
                &session_id,
                LAYER_NAME,
                CREATED_BY,
                &state,
                ext_store.clone(),
            )
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "create_layer for captured entry {} must succeed: {e}",
                    message.index
                )
            });
    }
    println!("layer created: real object_id={object_id} (normalised to {GOLDEN_OBJECT_ID})");

    // ── 2. Commit captured updates, splitting at the base ───────────────────
    let mut all_commits: Vec<Commit> = Vec::new();
    let mut tail_commits: Vec<Commit> = Vec::new();
    let mut base_snap_seq: Option<u64> = None;
    let mut empty_commit_seq = 0u64;
    let mut base_bytes: Option<Bytes> = None;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());

    let commit_captured = |message: &CapturedMessage| {
        let delta = parse_captured(message);
        let ompf = message.ompf.clone();
        async {
            layer_store
                .update_layer(
                    &session_id,
                    LAYER_NAME,
                    object_id,
                    delta,
                    Some(ompf),
                    WRITER_CLIENT_ID,
                    0,
                )
                .await
        }
    };

    for (ordinal, message) in updates.iter().enumerate() {
        let (sequence_num, _server_ts) = commit_captured(message).await.unwrap_or_else(|e| {
            panic!(
                "update_layer for captured entry {} must succeed: {e}",
                message.index
            )
        });
        let commit = Commit::Verbatim {
            sequence_num,
            ompf: message.ompf.clone(),
        };
        all_commits.push(commit.clone());
        if base_snap_seq.is_some() {
            tail_commits.push(commit);
        }

        let committed = ordinal + 1;

        // The base is assembled the moment the split count is reached, before
        // any further commit lands — so the base's snapshot and the tail's
        // first record cannot overlap.
        if committed == BASE_SPLIT_UPDATE_COUNT {
            let decision = assemble_layer_checkpoint(
                tv.client(),
                tv.key_schema(),
                object_id,
                store.as_ref(),
                SERVICE_LABEL,
                &mut NoLeaseToKeep,
            )
            .await
            .expect("assemble_layer_checkpoint must succeed against the quiescent layer");
            let outcome = match decision {
                AssemblyDecision::Written(outcome) => outcome,
                AssemblyDecision::SkippedNotQuiesced => panic!(
                    "the generator commits serially and the layer is at or below the atomic-read \
                     limits, so its read proves a cut at or above the pin; got \
                     SkippedNotQuiesced"
                ),
                AssemblyDecision::LeaseLost => panic!(
                    "the generator holds no leader lease, so its keeper never reports one lost"
                ),
                AssemblyDecision::ShuttingDown => {
                    panic!("the generator's keeper never reports a shutdown; got ShuttingDown")
                }
            };
            assert_eq!(
                outcome.snap_seq, sequence_num,
                "the base must land exactly on the last committed sequence — a different \
                 snap_seq means a writer raced the generator"
            );
            let blob = store
                .get_layer_checkpoint(&outcome.handle)
                .await
                .expect("get_layer_checkpoint must succeed")
                .expect("the handle the assembler returned must address a live artifact");
            base_snap_seq = Some(outcome.snap_seq);
            base_bytes = Some(blob.bytes);
            println!("base assembled: snap_seq={}", outcome.snap_seq);
        }

        // One real §8.4 empty commit, strictly inside the tail's span.
        if committed == EMPTY_COMMIT_AFTER_UPDATE_COUNT {
            let (sequence_num, _) = layer_store
                .update_layer(
                    &session_id,
                    LAYER_NAME,
                    object_id,
                    None,
                    None,
                    WRITER_CLIENT_ID,
                    0,
                )
                .await
                .expect("the §8.4 empty commit must succeed");
            empty_commit_seq = sequence_num;
            let commit = Commit::Empty { sequence_num };
            all_commits.push(commit.clone());
            tail_commits.push(commit);
            println!("empty commit landed at seq={sequence_num}");
        }
    }

    let base_snap_seq = base_snap_seq.expect("the base split point must have been reached");
    let base_bytes = base_bytes.expect("the base blob must have been read back");
    assert!(
        empty_commit_seq > base_snap_seq,
        "the §8.4 empty commit must land inside the tail's span, not inside the base"
    );
    assert!(
        tail_commits
            .iter()
            .any(|c| matches!(c, Commit::Empty { .. })),
        "the tail must carry the empty-delta record shape"
    );
    let head_seq = tail_commits
        .last()
        .expect("the tail must carry at least one record")
        .sequence_num();
    assert!(
        matches!(tail_commits.last(), Some(Commit::Verbatim { .. })),
        "the tail's last record must be a captured commit, so the empty record sits strictly \
         inside the span rather than at its edge"
    );
    assert!(
        base_snap_seq < CHAIN_SPLIT_SEQ && CHAIN_SPLIT_SEQ < head_seq,
        "CHAIN_SPLIT_SEQ ({CHAIN_SPLIT_SEQ}) must sit strictly inside ({base_snap_seq}, \
         {head_seq}] so both chained generations carry records"
    );

    // ── 3. Fixture-truthfulness precondition (PM condition F3) ──────────────
    assert_stream_matches_commits(&tv, object_id, &all_commits).await;
    println!(
        "stream truthfulness verified for {} commits ({} in the tail)",
        all_commits.len(),
        tail_commits.len()
    );

    // ── 4. Encode the artifacts ─────────────────────────────────────────────
    let encode = |from_seq: u64, commits: &[Commit]| -> Bytes {
        let to_seq = commits
            .last()
            .expect("a tail must carry at least one record")
            .sequence_num();
        let header = TailHeader {
            object_id: GOLDEN_OBJECT_ID,
            base_snap_seq,
            from_seq,
            to_seq,
        };
        encode_tail_envelope(&header, commits.iter().map(Commit::to_record))
            .unwrap_or_else(|e| panic!("tail envelope ({from_seq}, {to_seq}] must encode: {e}"))
    };

    let tail_a = encode(base_snap_seq, &tail_commits);
    let chain_split_at = tail_commits
        .iter()
        .position(|c| c.sequence_num() > CHAIN_SPLIT_SEQ)
        .expect("CHAIN_SPLIT_SEQ must fall inside the tail's records");
    let tail_b1 = encode(base_snap_seq, &tail_commits[..chain_split_at]);
    let tail_b2 = encode(CHAIN_SPLIT_SEQ, &tail_commits[chain_split_at..]);

    // SR-6.1: `encode_tail_envelope` always stamps TAIL_FORMAT_VERSION and has
    // no version parameter, so the unknown-version artifact is `tail_a` with
    // exactly four bytes patched at the exported offset — generator-produced,
    // reproducible, and diffable against `tail_a.letl`.
    let mut tail_unknown = tail_a.to_vec();
    tail_unknown[TAIL_FORMAT_VERSION_OFFSET..TAIL_FORMAT_VERSION_OFFSET + 4]
        .copy_from_slice(&UNKNOWN_FORMAT_VERSION.to_le_bytes());
    assert_ne!(
        UNKNOWN_FORMAT_VERSION, TAIL_FORMAT_VERSION,
        "the forged version must be one this build cannot read"
    );

    // ── 5. Compose, canonicalise, self-verify ───────────────────────────────
    let composed = compose_from_artifacts(
        GOLDEN_OBJECT_ID,
        &base_bytes,
        base_snap_seq,
        std::slice::from_ref(&tail_a),
    )
    .expect("the freshly-built golden lineage must compose");
    assert_eq!(composed.composed_seq, head_seq);
    let expected = canonicalize(&composed, base_snap_seq);

    let chained = compose_from_artifacts(
        GOLDEN_OBJECT_ID,
        &base_bytes,
        base_snap_seq,
        &[tail_b1.clone(), tail_b2.clone()],
    )
    .expect("the chained golden lineage must compose");
    assert_eq!(
        canonicalize(&chained, base_snap_seq),
        expected,
        "the chained two-generation lineage must compose to the identical state — a difference \
         here means the split itself changed the replay, and the committed golden would encode \
         that bug"
    );

    let mut expected_json =
        serde_json::to_string_pretty(&expected).expect("the canonical golden state must serialize");
    expected_json.push('\n');

    // ── 6. Emit ─────────────────────────────────────────────────────────────
    let tail_span = format!("({base_snap_seq}, {head_seq}]");
    let artifacts = vec![
        Artifact {
            name: "base.ompf",
            bytes: base_bytes.to_vec(),
            purpose: "Real `assemble_layer_checkpoint` base at `snap_seq` = the split point — the lineage root every tail here layers onto.",
            sidecar: format!(
                "# `base.ompf`\n\nReal `assemble_layer_checkpoint(.., &mut NoLeaseToKeep)` \
                 product at `snap_seq = {base_snap_seq}`, read back out of an \
                 `InMemoryCheckpointStore`. An OMPF wrapping a reset OODF delta \
                 (`is_diff = true, base_version = 0`) — the shape `decode_checkpoint_ompf` \
                 requires.\n\nCovers the first {BASE_SPLIT_UPDATE_COUNT} captured `UpdateLayer` \
                 messages plus the `CreateLayer` seed.\n\nByte-reproducible: `plan_full_content` \
                 emits fields, time samples, and the orphan-section tail in a total order over the \
                 content of `ObjectState`, so identical state encodes to identical bytes in any \
                 process (b.gcb / b.vz7). A diff here means the state moved or the emission order \
                 regressed — see the README and `REGENERATE.md` § 4.\n"
            ),
        },
        Artifact {
            name: "tail_a.letl",
            bytes: tail_a.to_vec(),
            purpose: "Single-generation tail over the whole post-base span. THE replay gate's input.",
            sidecar: format!(
                "# `tail_a.letl`\n\nOne `LETL` envelope covering `{tail_span}` \
                 ({} records), header `object_id = {GOLDEN_OBJECT_ID}`, \
                 `base_snap_seq = {base_snap_seq}`.\n\nRecords are the writer's raw inbound OMPFs \
                 verbatim, in commit order, with one `EmptyDelta` record at sequence \
                 {empty_commit_seq} (the real §8.4 empty commit). Record `server_ts` is normalised \
                 to `{HAND_PUT_SERVER_TS}`.\n\nPinned by \
                 `golden_base_and_tail_compose_to_the_committed_expected_state` and \
                 `golden_tail_artifact_bytes_are_stable_under_re_encoding`.\n",
                tail_commits.len()
            ),
        },
        Artifact {
            name: "tail_b1.letl",
            bytes: tail_b1.to_vec(),
            purpose: "First generation of the chained golden — same span as `tail_a`, split in two.",
            sidecar: format!(
                "# `tail_b1.letl`\n\nFirst generation of the chained lineage: covers \
                 `({base_snap_seq}, {CHAIN_SPLIT_SEQ}]` ({chain_split_at} records) onto the same \
                 base. Pinned by \
                 `golden_chained_two_generation_lineage_composes_to_the_same_state`.\n"
            ),
        },
        Artifact {
            name: "tail_b2.letl",
            bytes: tail_b2.to_vec(),
            purpose: "Second generation of the chained golden; links onto `tail_b1` at its `to_seq`.",
            sidecar: format!(
                "# `tail_b2.letl`\n\nSecond generation of the chained lineage: covers \
                 `({CHAIN_SPLIT_SEQ}, {head_seq}]` ({} records). Its `base_snap_seq` is still \
                 `{base_snap_seq}` — a later tail names the lineage's BASE, and links onto its \
                 predecessor through `from_seq == previous.to_seq` (SR-2.1). Pinned by \
                 `golden_chained_two_generation_lineage_composes_to_the_same_state`.\n",
                tail_commits.len() - chain_split_at
            ),
        },
        Artifact {
            name: "tail_unknown_version.letl",
            bytes: tail_unknown.clone(),
            purpose: "Unknown-version refusal: `tail_a.letl` with `format_version` forged to an unsupported value, as a newer build's tail would carry.",
            sidecar: format!(
                "# `tail_unknown_version.letl`\n\n`tail_a.letl` with exactly four bytes patched at \
                 `TAIL_FORMAT_VERSION_OFFSET` ({TAIL_FORMAT_VERSION_OFFSET}) — \
                 `format_version = {UNKNOWN_FORMAT_VERSION}` instead of \
                 `{TAIL_FORMAT_VERSION}`.\n\nProduced by the generator, not by a hand-edited hex \
                 byte: `encode_tail_envelope` always stamps `TAIL_FORMAT_VERSION` and offers no \
                 version parameter, so this build structurally cannot mint an artifact it cannot \
                 read. Patching the exported offset is the documented route \
                 (`src/checkpoint/tail_codec.rs`, the const's rustdoc).\n\nSR-6.1's scenario is a \
                 NEWER build's artifact reaching an older one, hence `2` rather than a nonsense \
                 value. Pinned by \
                 `golden_fixture_at_an_unknown_format_version_rejects_permanent_and_is_attributed`, \
                 which also asserts the four-byte diff against `tail_a.letl`.\n"
            ),
        },
        Artifact {
            name: "expected_state.json",
            bytes: expected_json.into_bytes(),
            purpose: "The composed state on the faithful comparison basis below. The gate's oracle.",
            sidecar: format!(
                "# `expected_state.json`\n\nCanonical rendering (schema version \
                 {EXPECTED_STATE_SCHEMA_VERSION}) of `compose_from_artifacts(base.ompf, \
                 tail_a.letl)`, limited to the columns a checkpoint round trip CARRIES — see the \
                 README's \"Faithful comparison basis\" section for what is included and what is \
                 excluded, and why.\n\nEvery collection is sorted on a total key, so the file does \
                 not inherit `HashMap` iteration order.\n\n**Do not edit this file to make a red \
                 gate go green.** A red gate means apply/codec semantics changed under a stored \
                 artifact; decide explicitly, then regenerate deliberately.\n"
            ),
        },
    ];

    write_artifacts(&out_dir, &artifacts);
    let readme = render_readme(
        &artifacts,
        &source_sha,
        base_snap_seq,
        head_seq,
        tail_commits.len(),
        empty_commit_seq,
        composed.ext_values.len(),
        object_id,
    );
    std::fs::write(
        out_dir.join("README.md"),
        format!("{MARKDOWN_LICENSE_HEADER}{readme}"),
    )
    .expect("README.md must be writable");

    println!(
        "\nwrote {} artifacts to {}",
        artifacts.len(),
        out_dir.display()
    );
    for a in &artifacts {
        println!(
            "  {:<28} {:>8} bytes  sha256={}",
            a.name,
            a.bytes.len(),
            to_hex(&Sha256::digest(&a.bytes))
        );
    }
    println!(
        "\nsplit point: base@{base_snap_seq}, tail {tail_span} ({} records, empty at \
         {empty_commit_seq}); chain splits at {CHAIN_SPLIT_SEQ}",
        tail_commits.len()
    );
    println!("composed ext-value refs: {}", composed.ext_values.len());
}
