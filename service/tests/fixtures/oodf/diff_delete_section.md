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

# `diff_delete_section.oodf` capture metadata

**Scenario**: `diff_delete_section` (synthetic — see the notes below)

- **Origin**: `oodf_ffi::build_delta` via `cargo run -p oodf-ffi --example generate_fixtures` (NOT captured from the real C++ service)
- **RPC**: N/A (synthesized; not captured from a live RPC)
- **Client config**:
  - Generator: `oodf-ffi/examples/generate_fixtures.rs`
  - StructCommands: 1 × DiffDeleteSection(section_id=42)
  - SetFields: 0
  - TimeSamples: 0
- **Server version**: N/A (synthesized)
- **Capture date**: 2026-04-16 (generator run)
- **Byte length**: 104

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

Exercises the one StructCommand variant not covered by postman_chunk.bin. A catch-up from sequence 0 emits the layer's current state, not the deletion (see `diff_after_delete.md`). A server-emitted `DiffDeleteSection` reaches a subscriber in a catch-up from a nonzero sequence below the delete, or on the live feed of the deleting `UpdateLayer`; every recorded capture subscribes from sequence 0, and none captures the live feed of a delete. Synthetic retained.

## Real C++ verification

- **Date**: 2026-04-17
- **Path**: CreateLayer on a fresh per-fixture layer, a deployed C++ reference service
- **Status**: `accepted`
- **OODF bytes**: 104 (OMPF-wrapped: 120)
- **Remote object_id**: 1408
- **Meaning**: the real C++ parser accepts bytes produced by `oodf_ffi::build_delta` for this scenario. This is empirical confirmation of Bee b.duy Acceptance Criterion 2 for the scenario's schema branches.
