#!/bin/bash
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
#
# Regenerate the `live-edit-capabilities-proto` crate from the Discovery
# Service v2alpha protobuf module.
#
# The protos are a canonical cross-service artifact owned by the Discovery
# team and published to the official buf registry — they are deliberately
# NOT hand-vendored into this repo (unlike `proto/`, which is compiled by
# tonic-build in build.rs). Generation runs through the buf CLI with the
# plugin pipeline declared in `capabilities-proto/buf.gen.yaml`
# (protoc-gen-prost + protoc-gen-tonic + protoc-gen-prost-crate); the
# generated sources and Cargo.toml are committed. Re-run this whenever the
# upstream module changes and commit the regenerated files.
#
# Source resolution, in order of preference:
#   1. The buf registry module `pb.api.nvidia.com/omniverse/discovery`
#      (requires `buf registry login pb.api.nvidia.com` with an NVIDIA
#      account that can read the omniverse/discovery module).
#   2. A local checkout of the module, passed explicitly as $1 or via
#      DISCOVERY_PROTOS_DIR (the directory containing the module's
#      buf.yaml, e.g. the `proto/` directory of the upstream Discovery proto
#      sources). Used automatically when the registry is unreachable
#      (sandboxed environments).
#
# Requires the buf CLI plus the pinned protoc plugins below on PATH.
# Install the plugins with:
#   cargo install --locked protoc-gen-prost@${PROTOC_GEN_PROST_VERSION} \
#       protoc-gen-tonic@${PROTOC_GEN_TONIC_VERSION} \
#       protoc-gen-prost-crate@${PROTOC_GEN_PROST_CRATE_VERSION}

# Keep in lockstep with the workspace's prost/tonic versions (root
# Cargo.toml pins prost 0.14 + tonic 0.14 + tonic-prost 0.14; these plugin
# releases emit code for exactly that combination).
PROTOC_GEN_PROST_VERSION="0.5.0"
PROTOC_GEN_TONIC_VERSION="0.5.0"
PROTOC_GEN_PROST_CRATE_VERSION="0.5.0"

REGISTRY_MODULE="pb.api.nvidia.com/omniverse/discovery"

set -euo pipefail

cd "$(dirname "$0")/../capabilities-proto"

for tool in buf protoc-gen-prost protoc-gen-tonic protoc-gen-prost-crate; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: '$tool' not found on PATH — see the header of this script" >&2
        exit 1
    fi
done

check_plugin_version() {
    local plugin="$1" want="$2" have
    have="$(cargo install --list 2>/dev/null \
        | sed -n "s/^${plugin} v\([0-9.]*\).*/\1/p")"
    if [ -n "$have" ] && [ "$have" != "$want" ]; then
        echo "warning: ${plugin} is v${have}, pinned version is v${want} —" \
             "generated output may differ from the committed sources" >&2
    fi
}
check_plugin_version protoc-gen-prost "$PROTOC_GEN_PROST_VERSION"
check_plugin_version protoc-gen-tonic "$PROTOC_GEN_TONIC_VERSION"
check_plugin_version protoc-gen-prost-crate "$PROTOC_GEN_PROST_CRATE_VERSION"

# Generate into a temp dir (`--output` redirects both plugin outputs there;
# the gen_crate template still resolves against this cwd) and swap into place
# only after `buf generate` succeeds, so a failed run — unreachable source,
# mid-generate network drop, wrong-version plugin blowing up — never leaves
# the workspace member deleted (which would break every workspace cargo
# command until `git checkout`). Replacing src/ wholesale also propagates
# upstream removals. buf.gen.yaml and Cargo.toml.template are the only
# hand-maintained files in this crate.
GEN_TMP="$(mktemp -d)"
trap 'rm -rf "$GEN_TMP"' EXIT

install_generated() {
    if [ ! -d "$GEN_TMP/src" ] || [ ! -f "$GEN_TMP/Cargo.toml" ]; then
        echo "error: buf generate did not produce src/ and Cargo.toml under" \
             "$GEN_TMP — buf.gen.yaml output wiring changed?" >&2
        exit 1
    fi
    # Generated sources carry the same NVIDIA Apache-2.0 header as the rest
    # of the repo (Cargo.toml inherits it from Cargo.toml.template).
    find "$GEN_TMP/src" -name '*.rs' -exec python3 ../tools/license_headers.py fix {} + >/dev/null
    python3 ../tools/license_headers.py fix "$GEN_TMP/Cargo.toml" >/dev/null
    rm -rf src Cargo.toml
    mv "$GEN_TMP/src" src
    mv "$GEN_TMP/Cargo.toml" Cargo.toml
}

local_module="${1:-${DISCOVERY_PROTOS_DIR:-}}"

if [ -z "$local_module" ]; then
    # Probe registry reachability (compile the module, discard the output) so
    # an unreachable registry gets the curated guidance below rather than a
    # raw `buf generate` failure.
    if ! buf build "$REGISTRY_MODULE" -o /dev/null 2>/tmp/buf-gen-err.$$; then
        echo "warning: 'buf build' against ${REGISTRY_MODULE} failed:" >&2
        sed 's/^/    /' "/tmp/buf-gen-err.$$" >&2 || true
        rm -f "/tmp/buf-gen-err.$$"
        echo "error: registry unreachable and no local module given." >&2
        echo "  Either 'buf registry login pb.api.nvidia.com' and re-run, or pass a" >&2
        echo "  local checkout of the module (the directory containing its buf.yaml):" >&2
        echo "    $0 /path/to/discovery/proto" >&2
        exit 1
    fi
    rm -f "/tmp/buf-gen-err.$$"
    buf generate --output "$GEN_TMP"
    install_generated
    echo "Regenerated capabilities-proto from registry module ${REGISTRY_MODULE}"
    exit 0
fi

if [ ! -f "$local_module/buf.yaml" ]; then
    echo "error: '$local_module' does not look like a buf module (no buf.yaml)" >&2
    exit 1
fi

if ! grep -q "name: ${REGISTRY_MODULE}" "$local_module/buf.yaml"; then
    echo "error: '$local_module/buf.yaml' does not declare module ${REGISTRY_MODULE}" >&2
    exit 1
fi

buf generate "$local_module" --template buf.gen.yaml --output "$GEN_TMP"
install_generated
echo "Regenerated capabilities-proto from local module checkout $local_module"
echo "note: verify the checkout matches the registry's main branch before committing"
