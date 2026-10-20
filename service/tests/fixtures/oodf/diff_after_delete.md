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

# `diff_after_delete.oodf` capture metadata

**Scenario**: `diff_after_delete` — **REAL C++ CAPTURE**

- **Origin**: a deployed C++ reference service
- **RPC**: `CreateLayer` (3 children) + `UpdateLayer` (DiffDeleteSection on sid=10) + `SubscribeReadLayer` catchup
- **Client config**:
  - Seed shape: `is_diff=true, base_version=0`, Root + 3 children (sid=10/11/12)
  - Update shape: `is_diff=false, base_version=1`, 1 `DiffDeleteSection(section_id=10)` — matches the Postman UPDATE_LAYER_CHUNK shape (`is_diff=false, base_version=1`).
  - Subscribe: `sequence_num=0` (catchup from zero — receives current state, not historical diff)
- **Server version**: unknown (endpoint does not advertise)
- **Capture date**: 2026-04-17
- **Byte length**: 176

## Semantic summary

2 `DiffChangeSection` (the 2 surviving children after sid=10 was
deleted). The server accepted our `DiffDeleteSection` update AND
applied it (sid=10 is gone from the catchup). The catchup stream
emits **current state**, not historical diffs — so we see
`DiffChangeSection` for surviving sections rather than a
`DiffDeleteSection` replaying the deletion.

Capturing a `DiffDeleteSection` emitted by the server requires an
**open subscribe stream before the delete** (so the server pushes
the delete as an incremental update on the existing stream).
This capture does not drive that flow.

## Notes

Empirical confirmation that:
1. The real C++ service accepts our `DiffDeleteSection` update bytes
   (complementing the synthetic fixtures' `CreateLayer` acceptance checks).
2. The server correctly applies the delete and stores the reduced
   state (our post-delete catchup shows 2 sections, not 3).

The `wire_fixtures.rs` named test `real_diff_after_delete_round_trips`
asserts the post-delete structure.
