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

# `tail_a.letl`

One `LETL` envelope covering `(9, 36]` (27 records), header `object_id = 90100200`, `base_snap_seq = 9`.

Records are the writer's raw inbound OMPFs verbatim, in commit order, with one `EmptyDelta` record at sequence 34 (the real §8.4 empty commit). Record `server_ts` is normalised to `1700000000000000`.

Pinned by `golden_base_and_tail_compose_to_the_committed_expected_state` and `golden_tail_artifact_bytes_are_stable_under_re_encoding`.
