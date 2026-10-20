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

# `base.ompf`

Real `assemble_layer_checkpoint(.., &mut NoLeaseToKeep)` product at `snap_seq = 9`, read back out of an `InMemoryCheckpointStore`. An OMPF wrapping a reset OODF delta (`is_diff = true, base_version = 0`) — the shape `decode_checkpoint_ompf` requires.

Covers the first 8 captured `UpdateLayer` messages plus the `CreateLayer` seed.

Byte-reproducible: `plan_full_content` emits fields, time samples, and the orphan-section tail in a total order over the content of `ObjectState`, so identical state encodes to identical bytes in any process (b.gcb / b.vz7). A diff here means the state moved or the emission order regressed — see the README and `REGENERATE.md` § 4.
