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

# `reorder_children.oodf` capture metadata

**Scenario**: `reorder_children` (synthetic — see the notes below)

- **Origin**: `oodf_ffi::build_delta` via `cargo run -p oodf-ffi --example generate_fixtures` (NOT captured from the real C++ service)
- **RPC**: N/A (synthesized; not captured from a live RPC)
- **Client config**:
  - Generator: `oodf-ffi/examples/generate_fixtures.rs`
  - StructCommands: 1 × ReorderChildren(section_id=100, children_list=[101..105])
  - SetFields: 0
  - TimeSamples: 0
- **Server version**: N/A (synthesized)
- **Capture date**: 2026-04-16 (generator run)
- **Byte length**: 160

## Semantic summary

See the table in `README.md` for the full per-fixture summary.

## Regenerating

Regenerate the synthetic file byte-for-byte:

```sh
cargo run -p oodf-ffi --example generate_fixtures
```

No recorded capture from the C++ reference service replaces this fixture;
the notes below say why the synthetic file is kept.

## Notes

Similar reconstruction problem — a catch-up emits the post-reorder state, not the reorder command itself; a `ReorderChildren` reaches a subscriber only on the live feed of an `UpdateLayer` that carries it, and no recorded capture drives that. Synthetic retained.

## Real C++ verification

- **Date**: 2026-04-17
- **Path**: CreateLayer on a fresh per-fixture layer, a deployed C++ reference service
- **Status**: `rejected`
- **OODF bytes**: 160 (OMPF-wrapped: 176)
- **gRPC code**: `INTERNAL`
- **gRPC detail**: `Failed to overwrite object`
- **Meaning**: the C++ reference service's `CreateLayer` rejected this delta (see [`CAPTURE.md`](./CAPTURE.md) §3). The fixture's builder+parser round-trip is still validated by `oodf-ffi/tests/wire_fixtures.rs`.
