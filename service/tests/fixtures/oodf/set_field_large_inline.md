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

# `set_field_large_inline.oodf` capture metadata

**Scenario**: `set_field_external` capture attempt → `set_field_large_inline` (real capture)

- **Origin**: a deployed C++ reference service
- **RPC**: `CreateLayer` (seed with 2 MiB inline `SetField`) + `SubscribeReadLayer` catchup
- **Client config**:
  - Delta shape: `is_diff=true, base_version=0`, Root + sid=5, 1 inline `SetField(bigfield, 2 MiB of 0x42)`
- **Server version**: unknown (endpoint does not advertise)
- **Capture date**: 2026-04-17
- **Byte length**: 2 097 384 (2 MiB payload + vtables + Root + child)

## Semantic summary

2 `DiffChangeSection` + 1 inline `SetField("bigfield", 2 MiB of 0x42,
ext_value_size=0, ext_value_index=0)`. The 2 MiB payload came back
**inline** — the server did NOT promote to external storage despite
the size. This is empirical evidence that external-value promotion is
a **client-side encoding decision** (the client decides whether to
send inline bytes or an `(ext_value_size, 32-byte hash)` reference +
external blob in OMPF), not a server-side size-threshold operation.

## Notes

Separate from the synthetic `set_field_external.oodf` fixture, which
exercises the `ext_value_size > 0` branch of the schema. This
capture exercises the server-wire path for large inline `SetField`
values and confirms it round-trips cleanly. The `wire_fixtures.rs`
named test `real_set_field_large_inline_round_trips` asserts the
payload size and shape.
