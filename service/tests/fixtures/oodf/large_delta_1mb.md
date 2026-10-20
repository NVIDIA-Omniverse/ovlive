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

# `large_delta_1mb.oodf` capture metadata

**Scenario**: `large_delta_1mb` (synthetic — see the notes below)

- **Origin**: `oodf_ffi::build_delta` via `cargo run -p oodf-ffi --example generate_fixtures` (NOT captured from the real C++ service)
- **RPC**: N/A (synthesized; not captured from a live RPC)
- **Client config**:
  - Generator: `oodf-ffi/examples/generate_fixtures.rs`
  - StructCommands: 0
  - SetFields: 1100 × SetField (≈1 KB inline value each, keys bulk_key_0000..bulk_key_1099)
  - TimeSamples: 0
- **Server version**: N/A (synthesized)
- **Capture date**: 2026-04-16 (generator run)
- **Byte length**: 1192496

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

>1 MB payload exercising OMPF multi-chunk framing. Captureable in principle by sending 1100 fields to CreateLayer and re-subscribing. The recorded catch-up captures demonstrate the subscribe-and-capture path; a seed of 1,100 inline fields would need a dedicated capture, which none of them is (the `set_field_external` capture seeds a layer over 1 MB, but as one 2 MiB field). Synthetic retained.

## Real C++ verification

- **Date**: 2026-04-17
- **Path**: CreateLayer on a fresh per-fixture layer, a deployed C++ reference service
- **Status**: `rejected`
- **OODF bytes**: 1192496 (OMPF-wrapped: 1192512)
- **gRPC code**: `INTERNAL`
- **gRPC detail**: `Failed to overwrite object`
- **Meaning**: the C++ reference service's `CreateLayer` rejected this delta (see [`CAPTURE.md`](./CAPTURE.md) §3). The fixture's builder+parser round-trip is still validated by `oodf-ffi/tests/wire_fixtures.rs`.
