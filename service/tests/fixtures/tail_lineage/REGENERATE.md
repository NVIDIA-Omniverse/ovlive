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

# Regenerating the golden tail-lineage fixtures

Workflow doc for `tests/fixtures/tail_lineage/`, the AC-6 / SR-6.2 committed
persisted-tail artifacts. Sibling of `tests/fixtures/oodf/CAPTURE.md`, which
serves the same role for the OODF wire fixtures.

`README.md` in this directory is **generated** — it carries the provenance table
(source SHA-256, split point, byte lengths, per-artifact SHA-256s) and is
rewritten in full by the generator. This file is hand-written and is where the
workflow, the review checklist, and the "what to do when the gate goes red"
guidance live.

## 1. Purpose

These artifacts exist so that a change to `apply_delta` (or to the tail codec's
layout) which alters replay of an already-stored artifact **fails CI and forces
an explicit decision**, rather than silently redefining what a durable
checkpoint means. Everything else in Epic 4's proof burden runs against inputs
built by the same build that consumes them; only this lane's inputs are frozen
bytes, which is what lets it survive future refactors.

The consumer is `tests/integration/checkpoint/golden_tail_replay_test.rs`. It is
Valkey-free (`compose_from_artifacts` performs no I/O), unconditionally
compiled, and loads every artifact through `include_bytes!` / `include_str!` so
a missing fixture is a compile error rather than a skip.

## 2. Authoritative source

`tools/fixtures/gze_inbound_deltas.json` — 36 **real captured Kit inbound
messages** (2 × `CreateLayer`, 34 × `UpdateLayer`), each entry
`{ts, kind, len, b64}` where `b64` is the writer's raw inbound OMPF. That is
exactly the payload a tail `Verbatim` record carries, which is why the file can
serve as a tail-fixture source at all.

Do **not** replace it with synthetic single-section fixtures: synthetic
fixtures of that shape hid both b.gze and b.zju. The same file drives the internal catch-up replay and
acceptance harnesses, whose replay of it is the reference for how the two
`CreateLayer` messages and the 34 `UpdateLayer` messages relate (both creates
land on the same `(session, layer)` pair; the second supersedes the first).

## 3. Regeneration recipe

```sh
# TEST_VALKEY_URL must be reachable. Inside the dev container it is preset to
# redis://valkey:6379 — do not override it.
cargo run -p live-edit --example generate_tail_lineage_goldens --features test-support
```

The generator is `examples/generate_tail_lineage_goldens.rs`. It:

1. loads and SHA-256s the captured traffic;
2. spins a `TestValkey` namespace (auto-cleaned on drop) and creates the session
   + layer by replaying both captured `CreateLayer` messages;
3. commits the captured `UpdateLayer` messages in order through
   `LayerStore::update_layer` on the **default `lua` backend**, passing the
   captured bytes as `relay_ompf` verbatim;
4. assembles the base with the real `assemble_layer_checkpoint(..,
   &mut NoLeaseToKeep)` into an `InMemoryCheckpointStore` at the split point,
   and reads the blob straight back;
5. issues one real §8.4 empty commit (`update_layer(.., None, None, ..)`) inside
   the tail's span;
6. **verifies fixture truthfulness** — XRANGEs `le:{oid}:obj:stream` and asserts
   every recorded commit's bytes equal that sequence's stream entry `ompf`
   field, and that the empty commit's entry carries no `ompf` field at all;
7. encodes the tails, forges the unknown-version artifact by patching the four
   bytes at `TAIL_FORMAT_VERSION_OFFSET`, composes, canonicalises, and writes
   every artifact plus `README.md` plus the per-artifact `.md` sidecars.

It prints the split point, byte lengths, and SHA-256s; those same numbers land
in the generated `README.md`.

## 4. Reviewing a regeneration diff

Expected to change on every run, harmlessly:

Nothing. Every artifact in this directory is byte-reproducible as of the Epic-5
determinism Task.

Expected **not** to change unless something real moved:

* **`base.ompf` — its bytes AND its length.** `build_reset_ompf_from_state` walks
  `DiffEngine::plan_full_content`, which emits fields on `(version,
  authoring_set_order, section_id, key_name)`, time samples on `(version,
  section_id, time)`, and the orphan-section tail on `section_id` — total orders
  over the content of `ObjectState`, so identical state encodes to identical
  bytes in any process (b.gcb / b.vz7). Measured across two separate generator
  processes at that Task: byte-identical. A `base.ompf` diff means the composed
  state genuinely moved, or the emission order regressed — never "a different
  valid encoding." Length moves the same way: FlatBuffers vtable sharing and
  offset widths follow emission order, so a length change is the loudest form of
  the same signal, not a separate harmless one. See § 6.
* **the `.letl` tails.** `encode_tail_envelope` is byte-deterministic (SR-9.2)
  and every non-deterministic input is normalised (`object_id`, record
  `server_ts` — see the README's normalisation table). A tail diff means the
  captured traffic changed, the split constants moved, or the codec layout
  moved.
* **`expected_state.json`.** A diff here is the whole point of the gate. Treat
  it as a semantic change until proven otherwise; see § 6.

## 5. Adding an artifact

Same discipline as `tests/fixtures/oodf/`:

1. emit it from the generator (never by hand — a hand-placed blob has no
   reproducible provenance);
2. give it a `.md` sidecar; the generator writes one per artifact automatically;
3. add an explicit test in `golden_tail_replay_test.rs` that loads it via
   `include_bytes!` and asserts something specific about it;
4. add the file name to that file's `NAMED_GOLDEN_FIXTURES`.

`every_golden_fixture_has_a_named_test` fails on any non-`.md` file in this
directory that is not named there, and on any named entry with no file on disk,
and on any artifact missing its sidecar.

## 6. When the gate goes red

**Do not edit `expected_state.json` to match the new output, and do not
regenerate "to fix the build."** A red run means one of:

* **Apply semantics changed.** The engine now replays a stored artifact into a
  different state. Decide explicitly whether that is intended, and whether
  artifacts already written by deployed replicas remain restorable. SR-6.3's
  compatibility statement (Task 7, `t2.dfh.yj.p5`) is where that answer is
  recorded.
* **Codec layout changed.** `golden_tail_artifact_bytes_are_stable_under_re_encoding`
  fires alone. Every tail artifact already durable in the field is now
  unreadable or differently-readable — a `TAIL_FORMAT_VERSION` bump conversation,
  not a refactor.
* **The captured traffic or the split constants moved.** Then regeneration IS
  the right answer, and the README's provenance table will show the new source
  SHA-256 / split point.
* **`base.ompf` no longer matches.** Since the Epic-5 determinism Task this is a
  real signal, not encoding noise (§ 4): either the assembled state genuinely
  changed, or `plan_full_content`'s total emission order regressed. Check the
  order first — the `plan_full_content` determinism unit tests in
  `src/domain/diff.rs` and the emission-order properties in `tests/proptests.rs`
  are the finer instrument and fire before this gate does, so a red golden with
  those green points at the state, not the order. Do not regenerate to make it
  green.

Only after that decision, regenerate deliberately and record the reason in the
commit message.

## 7. Mutation verification

The Task's Success Criterion requires that this gate has been *observed* to
fire. The performed mutation, its site, the exact failing assertion, and the
clean revert are recorded in `golden_tail_replay_test.rs`'s module header. Redo
it (and update that table) if the comparison basis is ever materially widened or
narrowed — a gate nobody has watched fail is not evidence.
