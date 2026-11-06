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
#
# Usage: check-container-oss.sh IMAGE [BINARY]
#   BINARY  the image's binary, /app/BINARY, that its SHA256SUMS must bind
#           (default live-edit, the service image; live-edit-conformance for
#           the conformance image). A single file name: empty or containing
#           `/` is refused.
set -eu
usage='Usage: check-container-oss.sh IMAGE [BINARY]'
image=${1:?$usage}
binary=${2-live-edit}
case "$binary" in
    '' | */*)
        echo "check-container-oss.sh: BINARY must be a file name under /app, got '$binary'" >&2
        echo "$usage" >&2
        exit 2
        ;;
    *)
        # Anything else is taken as the binary's file name under /app.
        ;;
esac
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
docker cp "$container:/app/$binary" "$scratch/app/$binary"
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
# Upstream lockfiles pin versions the binary was not built from, so the only
# lockfiles the archive may list as members, once each, are the binary's and the
# standard library's. Archives inside it (flatbuffers.tar, the .crate files, the
# GCC sources) are not opened: they ship as published, with any lockfiles of
# their own.
#
# A member may be a lockfile when lockfile_pattern, a POSIX ERE, matches its
# lower-cased file name: a name ending in .lock, .lockb or .lockfile; containing
# .lock.; that is lock, or ends in -lock, _lock or .lock, before .json, .yaml,
# .yml or .toml; pylock.toml or pylock.<name>.toml; ending in shrinkwrap.json,
# shrinkwrap.yaml or shrinkwrap.yml; or Package.resolved. It is deliberately
# broad, so a lockfile variant no list of names foresaw fails the check instead
# of shipping unnoticed; source files named for locks (rwlock.rs, lock.rs,
# spin-lock.h) and words like clock.json do not match. A matching member that
# is not a lockfile ships only when not_lockfiles lists it, with a comment saying
# what it is.
# lockfile_pattern must equal LOCKFILE_PATTERN in package_oss.py,
# kept_lockfiles its KEPT_LOCKFILES and not_lockfiles its NOT_LOCKFILES, both
# under sources/.
lockfile_pattern='\.lock(b|file)?$|\.lock\.|^lock\.(json|ya?ml|toml)$|[-_.]lock\.(json|ya?ml|toml)$|^pylock\.(.*\.)?toml$|shrinkwrap\.(json|ya?ml)$|^package\.resolved$'
kept_lockfiles='sources/Cargo.lock sources/rust/source/library/Cargo.lock'
# Members lockfile_pattern matches that are not lockfiles, separated by
# whitespace. None today.
not_lockfiles=''
# Each offending member is printed on its own line, as
# `check-container-oss.sh: <unexpected|missing|duplicated> lockfile member: PATH`.
# The pattern goes through the environment, which awk reads verbatim (`awk -v`
# would process the backslashes in it).
LOCKFILE_PATTERN=$lockfile_pattern KEPT_LOCKFILES=$kept_lockfiles NOT_LOCKFILES=$not_lockfiles awk '
    BEGIN {
        failed = 0
        kept_count = split(ENVIRON["KEPT_LOCKFILES"], kept)
        for (i = 1; i <= kept_count; i++) seen[kept[i]] = 0
        not_count = split(ENVIRON["NOT_LOCKFILES"], not_lockfiles)
        for (i = 1; i <= not_count; i++) not_lockfile[not_lockfiles[i]] = 1
    }
    {
        name = $0
        sub(/.*\//, "", name)
        if (tolower(name) !~ ENVIRON["LOCKFILE_PATTERN"] || ($0 in not_lockfile)) next
        if ($0 in seen) {
            seen[$0]++
        } else {
            print "check-container-oss.sh: unexpected lockfile member: " $0 > "/dev/stderr"
            failed = 1
        }
    }
    END {
        for (i = 1; i <= kept_count; i++) {
            if (seen[kept[i]] == 0) {
                print "check-container-oss.sh: missing lockfile member: " kept[i] > "/dev/stderr"
                failed = 1
            } else if (seen[kept[i]] > 1) {
                print "check-container-oss.sh: duplicated lockfile member: " kept[i] > "/dev/stderr"
                failed = 1
            }
        }
        exit failed
    }
' "$scratch/members"
for member in sources/Cargo.lock sources/manifest.json sources/flatbuffers.tar \
    sources/rust/source/library/std/src/lib.rs sources/rust/source/library/Cargo.lock; do
    grep -Fx "$member" "$scratch/members" >/dev/null
done
grep -E '^sources/crates/[^/]+/Cargo.toml$' "$scratch/members" >/dev/null
# jemalloc's C sources are statically linked into the service binary (global
# allocator), so the archive must carry them with their BSD-2-Clause notice.
# Every image's archive vendors the whole Cargo.lock, so this holds for any
# BINARY.
grep -E '^sources/crates/tikv-jemalloc-sys-[^/]+/jemalloc/COPYING$' "$scratch/members" >/dev/null
grep -E '^sources/rust/dependencies/[^/]+\.crate$' "$scratch/members" >/dev/null
grep -E '^sources/gcc/[^/]+\.dsc$' "$scratch/members" >/dev/null
grep -E '^sources/gcc/[^/]+\.orig\.tar\.(gz|xz|bz2)$' "$scratch/members" >/dev/null
grep -E '^sources/gcc/[^/]+\.debian\.tar\.(gz|xz|bz2)$' "$scratch/members" >/dev/null
test -n "$(find "$scratch/licenses/rust-licenses" -type f -print)"
echo "Container OSS sources, notices and binary checksums verified."
