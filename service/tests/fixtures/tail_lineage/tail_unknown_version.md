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

# `tail_unknown_version.letl`

`tail_a.letl` with exactly four bytes patched at `TAIL_FORMAT_VERSION_OFFSET` (4) — `format_version = 2` instead of `1`.

Produced by the generator, not by a hand-edited hex byte: `encode_tail_envelope` always stamps `TAIL_FORMAT_VERSION` and offers no version parameter, so this build structurally cannot mint an artifact it cannot read. Patching the exported offset is the documented route (`src/checkpoint/tail_codec.rs`, the const's rustdoc).

SR-6.1's scenario is a NEWER build's artifact reaching an older one, hence `2` rather than a nonsense value. Pinned by `golden_fixture_at_an_unknown_format_version_rejects_permanent_and_is_attributed`, which also asserts the four-byte diff against `tail_a.letl`.
