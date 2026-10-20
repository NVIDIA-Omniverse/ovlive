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

# OODF wire-format fixtures

Byte-level regression anchors for the `oodf-ffi` parse/build round-trip. Each
file is a raw FlatBuffers OODF payload (no OMPF wrapper) and is exercised by
`oodf-ffi/tests/wire_fixtures.rs`.

See [`CAPTURE.md`](./CAPTURE.md) for where the fixtures come from, naming
conventions, sidecar requirements, and meta-test registration steps.

## Fixture provenance — synthetic vs captured

Of the 11 `.oodf` files in this directory, 3 are captured from the C++
reference service (`time_sample`, `set_field_large_inline`,
`diff_after_delete`; **Real C++ capture** in the table below) and 8 are
synthesized via `cargo run -p oodf-ffi --example generate_fixtures`. Each has
a sibling `.md` sidecar recording its origin; a synthetic file's sidecar says
why no capture replaces it. The captured files are listed with their flows in
[`CAPTURE.md`](./CAPTURE.md) §3. The synthetic files exercise schema branches
that no recorded capture covers: branches a catch-up from sequence 0 (the
only kind the recorded captures ran) does not emit, which reach a subscriber
only on the live feed of an `UpdateLayer` that carries them or, for a
`DiffDeleteSection`, also in a catch-up from a nonzero sequence; or branches
that need a client encoding or a seed shape (1,100 inline fields) no
recorded capture sends (see [`CAPTURE.md`](./CAPTURE.md) §2).

The hand-captured C++ anchor in the workspace is
`oodf-ffi/tests/fixtures/postman_chunk.bin` (280 bytes). A later capture of
the same catch-up from a deployed C++ reference service was byte-identical
to it.

## Regenerating

The binaries are committed. Rerun the generator if schema, builder, or fixture
set changes:

```sh
cargo run -p oodf-ffi --example generate_fixtures
```

The generator is deterministic for a given `oodf-ffi` version — the C++
`FlatBufferBuilder` lays out fields left-to-right in a single pass so equal
input produces byte-equal output.

## Fixtures

All counts below refer to the idiomatic `OwnedDelta` structure after
`oodf_ffi::parse_delta`.

| File | Bytes | struct_commands | set_fields | time_samples | Purpose |
|---|---|---|---|---|---|
| `diff_delete_section.oodf` | 104 | 1 × `DiffDeleteSection` | 0 | 0 | Single `DiffDeleteSection` — the one struct-command variant not exercised by the Postman fixture. |
| `create_and_delete.oodf` | 176 | 1 × `CreateSection`, 1 × `DeleteSection` | 0 | 0 | Paired `CreateSection` + `DeleteSection` (non-diff path). |
| `move_section.oodf` | 144 | 1 × `MoveSection` | 0 | 0 | `MoveSection` variant with a renamed target. |
| `reorder_children.oodf` | 160 | 1 × `ReorderChildren` | 0 | 0 | `ReorderChildren` with a non-empty `children_list` (5 entries). |
| `set_field_external.oodf` | 192 | 0 | 1 external | 0 | `SetField` carrying a 32-byte hash + `ext_value_size = 1 MiB`, the external-blob branch. |
| `time_sample_external.oodf` | 168 | 0 | 0 | 1 external | `TimeSample` with a 32-byte hash + `ext_value_size > 0`. |
| `mixed_delta.oodf` | 552 | 4 (Create/Move/Reorder/DiffDelete) | 2 (inline + external) | 1 (inline) | Cross-variant delta combining struct commands, inline + external SetFields, and a TimeSample. `is_diff = true`, `base_version = 7`, `source_format = "usdc"`. |
| `large_delta_1mb.oodf` | 1 192 496 | 0 | 1100 inline (1 KiB each) | 0 | > 1 MiB payload to exercise `SubscribeReadLayer`'s multi-chunk OMPF wrapping on large deltas. |
| `time_sample.oodf` | 240 | 2 (`DiffChangeSection` ×2) | 0 | 1 inline | **Real C++ capture.** CreateLayer + SubscribeReadLayer catchup with an inline `TimeSample(sid=5, time=1.5)`. Confirms server round-trips `TimeSample` on the wire. |
| `set_field_large_inline.oodf` | 2 097 384 | 2 | 1 inline (2 MiB) | 0 | **Real C++ capture.** Server echoed a 2 MiB inline `SetField` without promoting to external — evidence that external-value promotion is a client-side encoding decision. |
| `diff_after_delete.oodf` | 176 | 2 | 0 | 0 | **Real C++ capture.** Post-state catchup after CreateLayer (3 children) + UpdateLayer (delete sid=10). Server stored the reduced state; catchup emits surviving sections only. |

## Test convention

Every fixture in this directory must be referenced by an explicit structural
assertion in `oodf-ffi/tests/wire_fixtures.rs`. A meta-test iterates the
directory and fails loudly if a fixture is present but not named in the
structural-assertion set — this prevents a new binary from being added
silently without a corresponding regression test.
