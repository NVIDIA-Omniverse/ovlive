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

(live-edit-api-validation)=
# Input validation

This page lists the rules the service applies to request fields and delta content.
A violation fails `INVALID_ARGUMENT` and applies nothing; on a chained `UpdateLayer` the status also carries the `INVALID_UPDATE` reason ({ref}`Chained writes <live-edit-api-writing-layers-chained>`).
A delta over one of the two count limits carries its own reason instead, on every `CreateLayer` and `UpdateLayer` ({ref}`Deltas that are too large <live-edit-api-validation-too-large>`).

(live-edit-api-validation-fields)=
## Request fields

Request and header fields are checked before the permission check.
Lengths are in bytes, not characters; upload size limits are in {ref}`Sending an upload <live-edit-api-writing-layers-uploads>`.

| Field | Rule |
|---|---|
| `session_id` (every RPC that takes one) | Non-empty, at most 256 bytes, no control characters (`0x00`–`0x1F`). |
| Session name, `layer_name` | Not blank, at most 255 bytes, no control characters, no `..`, no leading `\\` or `//`, no Windows drive prefix such as `C:`. |
| `path` (`CreateSession`, `ListSessions`) | Non-empty, at most 4,096 bytes before and after {ref}`canonicalization <live-edit-api-validation-base-path>`, no control characters. |
| `client_id` (`UpdateLayer`, `SubscribeReadLayer`) | At most 256 bytes, no control characters, no pipe character. Empty is valid, except on an `UpdateLayer` that carries an `operation_id`. If you embed user strings, replace or reject pipes first. |
| `operation_id` (`CreateLayer`, `UpdateLayer`) | Empty means none. Otherwise at most 64 bytes, each one of `A–Z a–z 0–9 - . _ ~`, compared byte for byte. A UUIDv4 in its 36-character form fits. |
| `chained_header` (`UpdateLayer`) | Its `header` is present and valid, with a non-empty `operation_id`. `previous_operation_id` is empty or a valid `operation_id` different from the header's. |
| `object_id` (`UpdateLayer`) | Non-zero. |
| `user_application` (`JoinSession`) | Empty, or at most 256 bytes with no control characters. |

(live-edit-api-validation-delta)=
## Delta content

Delta rules are checked after authorization, once the upload is parsed, and before anything is committed.

| Delta element | Rule |
|---|---|
| Entry count | At most 99,999 entries per delta: structural commands, `SetField`s and time samples added together, not of each kind. 60,000 sections plus 60,000 fields is over it. |
| `CommandReorderChildren` list length | At most 100,000 ids in one list. |
| Section, field and move names (`section_name`, `new_name`, `key_name`) | At most 4,096 bytes, no control characters. |
| `TimeSample.time` | Finite: NaN and ±infinity are refused. `-0.0` is accepted as the time `0.0` ({ref}`below <live-edit-api-validation-time-zero>`). |
| `CreateSection`, `MoveSection`, `DiffChangeSection` | A section cannot be its own parent. `parent_id` `0` is the root and always valid. |
| `DiffChangeSection` rename of an existing section | Refused when another sibling already holds the new `(parent_id, section_name)`. |
| `CommandReorderChildren` | Within the length limit, never refused for the ids it names: ids that are not current children are ignored, and a repeated id counts once. An empty list changes nothing. |
| Section-tree depth | At most 10,000 sections from the root after the delta applies. |
| `SetField.set_order` | At most 2^53 − 1. |

Two siblings can still share a name, because a `MoveSection` rename under the same parent is never checked.
Where they do, a name conflict always resolves to the sibling with the lowest `section_id`.

Each `SetField` advances its layer's running set order by its `set_order` plus one, and each commit advances the sequence number by one.
A layer whose running set order or sequence number would pass 2^53 − 1 refuses every further update with `INTERNAL`, message `internal error`, and applies nothing; other layers are unaffected.
Keep `set_order` values small: timestamp-sized values can use up a layer within a few updates.
To recover, replace the layer with `CreateLayer`: the new object starts its set order again at zero.

(live-edit-api-validation-time-zero)=
### Time samples at `-0.0`

A `TimeSample` at time `-0.0` is accepted and is the same time as `0.0`.
The service stores it, and serves it to late joiners and in content restored from backup, as `0.0`; live subscribers receive the update as sent, `-0.0` included.
So treat `-0.0` and `0.0` as one time when you apply updates.
A section holds one sample at that time: within one delta the last time sample at either spelling is the one kept, and a later write at either spelling replaces the earlier one.
This applies to the times of `TimeSample` entries only.
Field values, including a whole `timeSamples` dictionary set by a `SetField`, are stored and served byte for byte, because the service never decodes value bytes.

(live-edit-api-validation-too-large)=
### Deltas that are too large

A delta over a count limit is refused with an `ErrorInfo` of domain `nvidia.omniverse.live.v1alpha` and no `RetryInfo`, chained or not; re-sending the same delta always fails.
Every metadata value is a decimal integer.

| Status | Reason and metadata | Applied? | What to do |
|---|---|---|---|
| `INVALID_ARGUMENT` | `DELTA_TOO_LARGE`, over 99,999 entries in total; metadata `struct_commands`, `set_fields`, `time_samples`, `max_entries` (99999) | No | Split the content into several smaller deltas. For a new layer, send the first part with `CreateLayer` and the rest with `UpdateLayer`s. |
| `INVALID_ARGUMENT` | `CHILDREN_LIST_TOO_LONG`, one `CommandReorderChildren` over 100,000 ids; metadata `limit` (100000), `actual` | No | Name at most 100,000 children in one `CommandReorderChildren`. |

The limits apply to each upload, not to a layer: a layer can grow past them through further updates.
Match the reason, not the message text.
During a rolling upgrade, a replica of an earlier version refuses the same delta `INVALID_ARGUMENT` without these reasons.

(live-edit-api-validation-external-values)=
## External values

Every `SetField` or `TimeSample` with a non-zero `extValueSize` references an external value, an OMPF part of the same upload.
For each reference:

- `extValueIndex` is 1-based: 1 names the first external-value part (part 0 is the delta, and index 0 is refused), and that part is exactly `extValueSize` bytes;
- `valueOrExtHash` is exactly 32 bytes and equals the part's content hash.

The content hash is not plain SHA-256.
It is SHA-256 over the concatenated SHA-256 digests of the value's consecutive 1,048,576-byte blocks, the last block holding the remainder:

```python
import hashlib

def external_value_hash(value: bytes) -> bytes:
    block = 1024 * 1024
    digests = b"".join(
        hashlib.sha256(value[i:i + block]).digest()
        for i in range(0, len(value), block)
    )
    return hashlib.sha256(digests).digest()
```

An empty value hashes to SHA-256 of the empty string.
A refused upload stores, applies and broadcasts nothing.

(live-edit-api-validation-base-path)=
## Base path canonicalization

`CreateSession` and `ListSessions` normalize the base path, so equivalent spellings of one address name the same sessions:

- In the path, every byte outside `A–Z a–z 0–9 - _ . ~ /` becomes an uppercase `%XX` escape, and an encoded slash becomes `/`.
  `user@example.com` and `user%40example.com` are one path, as are a space and `%20`.
- The scheme is lowercased, and so is the host under `ftp`, `file`, `http`, `https`, `ws` and `wss`.

Not normalized: the host under any other scheme (`omniverse://Server/x` and `omniverse://server/x` differ), the user-info and port, and the query and fragment, which are kept exactly as sent.

Canonicalization also refuses, with `INVALID_ARGUMENT`:

- percent-escapes that do not decode to valid UTF-8;
- escapes that decode to a NUL or control character;
- `.` or `..` path segments, encoded ones included.

Encoding can lengthen a path up to about three times, and the 4,096-byte limit applies to the result too.

(live-edit-api-validation-see-next)=
## See next

Return to the {ref}`client API overview <live-edit-api-overview>`.
