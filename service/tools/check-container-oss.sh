#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Inspect the exact image digest that will be promoted, without needing a shell
# or Python in the distroless runtime. Run from the repository root.
set -eu
image=${1:?Usage: check-container-oss.sh IMAGE}
scratch=$(mktemp -d)
container=
cleanup() {
    if [ -n "$container" ]; then docker rm "$container" >/dev/null; fi
    rm -rf "$scratch"
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM
container=$(docker create "$image")
mkdir "$scratch/app"
docker cp "$container:/app/live-edit" "$scratch/app/live-edit"
docker cp "$container:/licenses" "$scratch/licenses"
docker cp "$container:/vendor.tar.gz" "$scratch/vendor.tar.gz"
for file in ATTRIBUTIONS.md LICENSE.txt NOTICE FLATBUFFERS-LICENSE.txt FLATBUFFERS-NOTICE.txt \
    GCC-COPYRIGHT.txt GPL-3.txt RUST-COPYRIGHT.html SOURCES.md SOURCE-MANIFEST.json SHA256SUMS; do
    test -s "$scratch/licenses/$file"
done
cmp ATTRIBUTIONS.md "$scratch/licenses/ATTRIBUTIONS.md"
(
    cd "$scratch"
    # A truncated checksum manifest must not turn verification into a partial
    # check. Compare the exact inventory before checking any hashes.
    { printf '%s\n' vendor.tar.gz; find app licenses -type f ! -path licenses/SHA256SUMS; } \
        | LC_ALL=C sort > expected-checksum-paths
    cut -c67- licenses/SHA256SUMS | LC_ALL=C sort > actual-checksum-paths
    cmp expected-checksum-paths actual-checksum-paths
    sha256sum -c licenses/SHA256SUMS
)
tar -tzf "$scratch/vendor.tar.gz" > "$scratch/members"
for member in sources/Cargo.lock sources/manifest.json sources/flatbuffers.tar \
    sources/rust/source/library/std/src/lib.rs sources/rust/source/library/Cargo.lock; do
    grep -Fx "$member" "$scratch/members" >/dev/null
done
grep -E '^sources/crates/[^/]+/Cargo.toml$' "$scratch/members" >/dev/null
# jemalloc's C sources are statically linked into the binary (global
# allocator), so the archive must carry them with their BSD-2-Clause notice.
grep -E '^sources/crates/tikv-jemalloc-sys-[^/]+/jemalloc/COPYING$' "$scratch/members" >/dev/null
grep -E '^sources/rust/dependencies/[^/]+\.crate$' "$scratch/members" >/dev/null
grep -E '^sources/gcc/[^/]+\.dsc$' "$scratch/members" >/dev/null
grep -E '^sources/gcc/[^/]+\.orig\.tar\.(gz|xz|bz2)$' "$scratch/members" >/dev/null
grep -E '^sources/gcc/[^/]+\.debian\.tar\.(gz|xz|bz2)$' "$scratch/members" >/dev/null
test -n "$(find "$scratch/licenses/rust-licenses" -type f -print)"
echo "Container OSS sources, notices and binary checksums verified."
