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

# `time_sample.oodf` capture metadata

**Scenario**: `time_sample` — **REAL C++ CAPTURE**

- **Origin**: a deployed C++ reference service
- **RPC**: `CreateLayer` (seed) + `SubscribeReadLayer` catchup
- **Client config**:
  - Delta shape: `is_diff=true, base_version=0`, Root + sid=5, 1 inline `TimeSample(section_id=5, time=1.5, 18 bytes)`
  - Server re-emitted identical content on catchup
- **Server version**: unknown (endpoint does not advertise)
- **Capture date**: 2026-04-17
- **Byte length**: 240

## Semantic summary

2 `DiffChangeSection` + 1 inline `TimeSample` (section_id=5, time=1.5,
18-byte inline value). `ext_value_size=0` — server did not promote to
external storage; TimeSample external promotion appears to be a
client-side encoding choice rather than a server decision.

## Notes

First empirical confirmation that the real C++ service round-trips
`TimeSample` through the gRPC wire. The `wire_fixtures.rs` named test
`real_time_sample_round_trips` asserts the captured bytes re-parse into
the expected shape and round-trip cleanly via `oodf_ffi::build_delta →
parse_delta`.
