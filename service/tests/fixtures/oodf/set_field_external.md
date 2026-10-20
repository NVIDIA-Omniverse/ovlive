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

# `set_field_external.oodf` capture metadata

**Scenario**: `set_field_external` (synthetic — see the notes below)

- **Origin**: `oodf_ffi::build_delta` via `cargo run -p oodf-ffi --example generate_fixtures` (NOT captured from the real C++ service)
- **RPC**: N/A (synthesized; not captured from a live RPC)
- **Client config**:
  - Generator: `oodf-ffi/examples/generate_fixtures.rs`
  - StructCommands: 0
  - SetFields: 1 × SetField(key='vertices', ext_size=1048576, ext_idx=1, 32-byte hash)
  - TimeSamples: 0
- **Server version**: N/A (synthesized)
- **Capture date**: 2026-04-16 (generator run)
- **Byte length**: 192

## Semantic summary

See the table in `README.md` for the full per-fixture summary.

## Regenerating

Regenerate the synthetic file byte-for-byte:

```sh
cargo run -p oodf-ffi --example generate_fixtures
```

The `set_field_external` capture cannot produce this fixture: the C++
reference service stored that capture's 2 MiB field inline, and the capture
is committed as `set_field_large_inline.oodf`.

## Notes

Exercises the external-value branch (ext_value_size > 0). The C++ reference service does not promote a large inline field to an external value — external-value encoding is the client's choice, and the server stores what it is given — so the `set_field_external` capture's inline seed cannot produce this branch, and no recorded capture seeds an external value. Synthetic retained for the external-encoding branch.

## Real C++ verification

- **Date**: 2026-04-17
- **Path**: CreateLayer on a fresh per-fixture layer, a deployed C++ reference service
- **Status**: `rejected`
- **OODF bytes**: 192 (OMPF-wrapped: 208)
- **gRPC code**: `INTERNAL`
- **gRPC detail**: `Failed to overwrite object`
- **Meaning**: the C++ reference service's `CreateLayer` rejected this delta (see [`CAPTURE.md`](./CAPTURE.md) §3). The fixture's builder+parser round-trip is still validated by `oodf-ffi/tests/wire_fixtures.rs`.
