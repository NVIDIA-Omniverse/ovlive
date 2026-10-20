<!--
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

# Golden tail-lineage fixtures

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
| Source SHA-256 | `0c3095e2a9485121ede4449529294d12a6fb96c8bee7eb8b274ded0f08918df5` |
| Source shape | 36 real captured Kit inbound messages — 2 × `CreateLayer`, 34 × `UpdateLayer`, 128 B – 1.6 KB each |
| Generator | `cargo run -p live-edit --example generate_tail_lineage_goldens --features test-support` |
| Generator source | `examples/generate_tail_lineage_goldens.rs` |
| Commit path | `LayerStore::update_layer`, DEFAULT `lua` backend, captured bytes passed as `relay_ompf` verbatim |
| Base produced by | `assemble_layer_checkpoint(.., &mut NoLeaseToKeep)` into an `InMemoryCheckpointStore` |
| Split point | base assembled after 8 captured `UpdateLayer` messages → `base_snap_seq = 9` |
| Tail span | `(9, 36]`, 27 records |
| Empty commit | one real `update_layer(.., None, None, ..)` at sequence 34 (the empty-delta shape) |
| Chained split | `tail_b1` covers `(9, 20]`, `tail_b2` covers `(20, 36]` |
| External values | 0 referenced by the composed lineage |

## Normalisations — what makes regeneration reproducible

| Normalised | To | Why it is sound |
|---|---|---|
| Tail header `object_id` (run value was `2`) | `90100200` | The live id comes from `le:counter:object_id` and differs per run. `compose_from_artifacts` takes `object_id` as a parameter, and the header field is only a lineage-identity check (`ComposeError::ObjectIdMismatch`) — never an input to replay. |
| Record-level `server_ts` | `1700000000000000` | It is a wall-clock `i64` serialized INTO the 21-byte record prefix, so an unnormalised artifact differs byte-for-byte on every regeneration. The restore path never reads a record's `server_ts`; the value restore consumes is `CheckpointMeta::server_ts`, a different field on a different struct. |

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
| `base.ompf` | 3320 | `e9bcd1767de7139b66f9af1828756e381a1fdab15b6adae33078f80508a0ba3f` | Real `assemble_layer_checkpoint` base at `snap_seq` = the split point — the lineage root every tail here layers onto. |
| `tail_a.letl` | 6619 | `416ae4a6207c8ba595c66a0d6629049919adf414f705f05e5a9e98286a015f83` | Single-generation tail over the whole post-base span. THE replay gate's input. |
| `tail_b1.letl` | 2211 | `e3181e8a9619a7be1b40774137be34b16dd9063df8277943c5c401f0324ff213` | First generation of the chained golden — same span as `tail_a`, split in two. |
| `tail_b2.letl` | 4452 | `c54f38354cacf417011586fca45d5e2a83762c46b66584fb5e460a38faf30237` | Second generation of the chained golden; links onto `tail_b1` at its `to_seq`. |
| `tail_unknown_version.letl` | 6619 | `9a23ce29bae9018145ae0dca88dd6bd31102caf2034eab28821e55cd679c503d` | Unknown-version refusal: `tail_a.letl` with `format_version` forged to an unsupported value, as a newer build's tail would carry. |
| `expected_state.json` | 16537 | `8901188813bf88b89e171a6a28fd7fa6e2860530887de289734bcdbbcfd3efb1` | The composed state on the faithful comparison basis below. The gate's oracle. |

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
