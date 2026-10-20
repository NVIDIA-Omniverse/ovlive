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

# OmniObjectDelta.fbs Provenance

This directory contains a pinned, vendored copy of the Omniverse Object Delta
FlatBuffers schema (OODF). The vendored copy is the authoritative definition
used by this repository to generate FFI bindings for cross-language
(de)serialization of OmniObjectDelta payloads. The copy differs from upstream
in its leading license notice only (see Relicensing).

## Upstream Source

- **Canonical path:** `nucleus/backend/api_server/specs/OmniObjectDelta.fbs`
  (inside the nucleus backend repository).
- **Fallback path:** `smoke-tests/backend/api_server/specs/OmniObjectDelta.fbs`
  (byte-identical mirror maintained by the smoke-tests repository; use this
  only if the nucleus checkout is unavailable).

## Pinned Version

- **Vendored file:** `OmniObjectDelta.fbs`
- **SHA256:** `41b5b1ee2ccd4404f9f66238d50e70e8af92c8deca01f6afda97fc95dd661fbe`
- **Upstream SHA256:** `fe87a05630145c72703412cc8507c371c8edc831f9bc2e4e83605ff482830434`
- **Notice lines:** `14`
- **Body SHA256:** `5e7d8c4f6de314746d8de89bb53a324d89b9101e947aa34facfa8280a06ebfa4`
- **Vendor date:** 2026-04-16

The fields:

- **SHA256** is the hash of the vendored file as committed here.
- **Upstream SHA256** is the hash of the upstream file this copy was made from,
  unmodified.
- **Notice lines** is the number of lines at the top of the vendored file that
  hold its license notice: the NVIDIA Apache-2.0 header. It is written as a
  positive decimal number without leading zeros. Every line after them is the
  body.
- **Body SHA256** is the hash of the body, `tail -n +15` of the vendored file
  (the line after the notice lines, onwards). It equals the hash of the
  upstream file after upstream's own 8-line notice, `tail -n +9`.

The GitLab CI `schema:drift-check` job and the Rust test
`oodf-ffi/tests/schema_integrity.rs` check that the vendored file hashes to
**SHA256**, that its body hashes to **Body SHA256**, and that
**Upstream SHA256** is recorded.

## Relicensing

Upstream's file begins with an 8-line notice: a 2019 NVIDIA copyright line,
followed by NVIDIA's proprietary terms, which reserve all rights and permit no
use without a separate license agreement with NVIDIA, and a closing `//`
line.

This copy replaces that notice, and only that notice, with the 14-line NVIDIA
Apache-2.0 header, copyright 2019-2026. The body is byte-identical to
upstream's. Basis: notice-only relicense by the owning team, 2026-10-15; this
copy.

## Stability Note

The upstream schema was first published in 2019 and has been stable since.
Changes to the upstream schema are rare and breaking; a change in upstream's
hash is therefore expected to be a signal of intentional schema evolution
requiring coordinated update across all Omniverse services.

## Re-Vendoring Instructions

To refresh this vendored copy against the upstream canonical path:

1. Copy the upstream file to a scratch location, not over the vendored copy:

   ```
   cp <nucleus-repo>/backend/api_server/specs/OmniObjectDelta.fbs \
      /tmp/OmniObjectDelta.fbs
   ```

2. Check it against **Upstream SHA256**:

   ```
   shasum -a 256 /tmp/OmniObjectDelta.fbs
   ```

   If the digest matches, upstream has not changed and there is nothing to
   re-vendor. If it differs, the schema has changed upstream: confirm the
   change is intended, and record the new digest as **Upstream SHA256**.

3. Replace upstream's leading notice with the header. Drop the notice (its
   copyright line through the closing `//` line; 8 lines in the upstream file
   pinned above, so adjust `+9` here and in step 4 if upstream's notice
   changed length) and let the license tool add the header, with the year
   range the vendored copy carries:

   ```
   tail -n +9 /tmp/OmniObjectDelta.fbs > oodf-ffi/schema/OmniObjectDelta.fbs
   python3 tools/license_headers.py fix --year 2019-2026 oodf-ffi/schema/OmniObjectDelta.fbs
   ```

   Change nothing else in the file.

4. Check the body is upstream's, unchanged. These two digests must be equal:

   ```
   tail -n +15 oodf-ffi/schema/OmniObjectDelta.fbs | shasum -a 256
   tail -n +9 /tmp/OmniObjectDelta.fbs | shasum -a 256
   ```

   Record it as **Body SHA256**.

5. Recompute the hash of the vendored file and update this `SOURCE.md` in
   place:

   ```
   shasum -a 256 oodf-ffi/schema/OmniObjectDelta.fbs
   ```

   - Replace the **SHA256** value above with the newly computed digest.
   - Replace the **Vendor date** value above with today's date (YYYY-MM-DD).

6. Commit the updated `OmniObjectDelta.fbs` and `SOURCE.md` together. The
   GitLab CI drift-check job fails any commit in which the recorded hashes do
   not match the file, so both must be updated atomically.
