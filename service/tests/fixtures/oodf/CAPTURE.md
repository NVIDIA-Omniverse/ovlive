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

# OODF wire-format fixtures: sources and conventions

Where the byte-level regression fixtures under `tests/fixtures/oodf/` come
from, and how a new one is named, documented and registered, so that the
`oodf-ffi` parse/build round-trip remains locked to the wire format produced
by the real Omniverse live-edit C++ service.

## 1. Purpose

Every `.oodf` file in this directory is a raw FlatBuffers OODF payload (no OMPF
wrapper) that serves as a wire-format regression anchor. The fixtures exist so
that a schema drift, a builder regression, or a verifier regression in
`oodf-ffi` is caught by a failing `cargo test` rather than at runtime against a
real C++ client. See `oodf-ffi/tests/wire_fixtures.rs` for how each fixture is
exercised.

## 2. Authoritative source

A fixture is only useful if its bytes originated from a trustworthy producer of
the canonical OODF wire format. In priority order:

1. **The real C++ live-edit service** (preferred): bytes captured from a
   deployed C++ reference service, as `oodf-ffi/tests/fixtures/postman_chunk.bin`
   and the captured fixtures in §3 were.
2. **The Omniverse C++ client library** that ships with nucleus. Any blob it
   produces is by construction wire-correct.
3. **Never** `oodf_ffi::build_delta`. Our builder is one of the things the
   fixture suite is meant to validate — generating fixtures from it is
   circular.

> **Current state.** Of the 11 `.oodf` files in this directory, 3
> (`time_sample`, `set_field_large_inline`, `diff_after_delete`) are captured
> from the C++ reference service (§3). The other 8 are synthesized via
> `cargo run -p oodf-ffi --example generate_fixtures`, which calls
> `oodf_ffi::build_delta` — the source the list above rules out — and are
> kept only because no capture replaces them. Each has a sidecar `.md`
> marking it as synthetic (see §5), with scenario-specific notes on why:
> a catch-up from sequence 0, the only kind the recorded captures ran,
> emits current state, not the commands that produced it, so the branch
> reaches a subscriber only on the live feed of an `UpdateLayer` that carries
> it or, for a `DiffDeleteSection`, also in a catch-up from a nonzero
> sequence (flows no recorded capture drives for that branch); or the
> branch appears only for a client encoding no recorded capture sends
> (external values); or it needs a seed shape no recorded capture sends
> (1,100 inline fields, the many-field shape `large_delta_1mb` pins).
>
> The authoritative wire anchors in the workspace today are:
> - `oodf-ffi/tests/fixtures/postman_chunk.bin` — 280 bytes, Postman-captured
>   from the real C++ service.
> - The 3 captured fixtures above.

## 3. Captured fixtures

Each captured fixture is the C++ reference service's answer to seed deltas
this repository builds, so its content is that synthetic content alone. They
were captured from a deployed C++ reference service on 2026-04-17; each
sidecar records the flow and the delta shapes.

| Fixture | Captured flow |
|---|---|
| `time_sample.oodf` | CreateLayer of a RESET with one inline `TimeSample` on a probe section → SubscribeReadLayer catch-up. |
| `set_field_large_inline.oodf` | CreateLayer of a RESET with one 2 MiB inline `SetField` → SubscribeReadLayer catch-up. The service did **not** promote the field to an external value. |
| `diff_after_delete.oodf` | CreateLayer with 3 children, then an UpdateLayer deleting sid=10 (`is_diff=false, base_version=1`) → SubscribeReadLayer catch-up from sequence 0: the post-delete state. |

`oodf-ffi/tests/fixtures/postman_chunk.bin` is the catch-up of a layer seeded
with the Postman `CREATE_LAYER_CHUNK`, captured by hand with Postman; a later
capture of the same flow was byte-identical to it.

Each synthetic sidecar's "Real C++ verification" section records whether the
C++ reference service's `CreateLayer` accepted that fixture's bytes. A
rejection there is expected: in the recorded run the service's `CreateLayer`
rejected every non-diff delta it was sent (`is_diff=false`), with or without
structural commands, and the mid-stream diff (`is_diff=true` with
`base_version > 0`), each with `INTERNAL` `Failed to overwrite object`. The
builder output is still schema-valid, and `oodf-ffi/tests/wire_fixtures.rs`
validates each fixture's builder+parser round-trip. A rejection of any other
shape would be a wire-compatibility bug.

## 4. Naming conventions

- `<scenario>.oodf` — raw OODF FlatBuffers payload, no OMPF wrapper. Every
  fixture in this directory MUST use this form; the test suite decodes them
  directly via `oodf_ffi::parse_delta`.
- `<scenario>.ompf` — full OMPF-wrapped chunk (optional, kept only when the
  OMPF framing itself is load-bearing for the test). None of the current
  fixtures require this.
- `<scenario>` is `snake_case`, descriptive of the schema branch exercised
  (e.g. `diff_delete_section`, `set_field_external`, `large_delta_1mb`).
- Each `.oodf` gets a same-named sidecar markdown file: `<scenario>.md`.

## 5. Per-file sidecar `.md` required fields

Every fixture under `tests/fixtures/oodf/<scenario>.oodf` must ship with a
`<scenario>.md` sidecar that records:

- **Origin** — which service/tool produced the bytes (real C++ live-edit,
  client library, or — temporarily — `generate_fixtures`).
- **RPC** — which RPC the bytes came from (`CreateLayer`, `SubscribeReadLayer`,
  `UpdateLayer`, etc.), or `N/A` for synthesized fixtures.
- **Client config** — the delta shapes sent, and any non-default client flags
  used.
- **Server version** — the live-edit service version string (git SHA or image
  tag). `N/A` for synthesized fixtures.
- **Capture date** — ISO 8601 (`YYYY-MM-DD`).
- **Byte length** — `wc -c` of the `.oodf` file. Exercised by the meta-test
  as a sanity anchor.
- **Semantic summary** — counts of struct_commands / set_fields / time_samples
  and which variants are exercised. Mirror the summary table in `README.md`.
- **Regeneration** — for a synthetic fixture, the generator command that
  rewrites the file byte-for-byte. A captured fixture records its flow and
  delta shapes instead (§3).

## 6. How the test suite picks up a fixture

Two-step registration in `oodf-ffi/tests/wire_fixtures.rs`:

1. **Meta-test (`every_fixture_has_a_named_test`).** Walks the directory at
   test time and asserts every `.oodf` file is listed in the
   `NAMED_SYNTHETIC_FIXTURES` array. A new file that is NOT in the array
   fails this test — the safety net against silent additions.
2. **Named test.** Add an `include_bytes!("../../tests/fixtures/oodf/<scenario>.oodf")`
   constant and a `#[test] fn <scenario>_round_trips()` that calls
   `round_trip_semantic(...)` and asserts the schema-branch-specific
   invariants (variant counts, field values). Pattern: copy an existing
   `<scenario>_round_trips` test, rename, adjust assertions.

Both steps are required. The meta-test enforces step 2 exists.

## 7. Update cadence

Add a new fixture whenever any of the following happens:

- A new schema branch is exercised (new `CommandTag` variant, new
  `SetField` / `TimeSample` shape, new top-level `Delta` field).
- A combination of existing branches is not yet covered by an existing
  fixture (e.g. `DiffDeleteSection` + external `SetField` + inline
  `TimeSample` in one delta).
- A bug fix touches parse/build behaviour on a shape the current fixture
  set does not exercise — pin the fix with a new fixture.
- A real C++ service regression ships a delta we cannot parse — capture it
  verbatim as a fixture before filing the bug.

Re-capture existing fixtures (replace bytes, keep name + assertions)
whenever the authoritative C++ service bumps its FlatBuffers version or the
schema's field order changes.
