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
# Regenerate Python gRPC stubs bundled into the live-edit-service Helm
# chart for in-cluster smoke tests.
#
# The Rust service uses tonic-build (build.rs) for its own stubs; the
# Helm smoke-test Job needs a Python copy at chart-package time because
# Helm's `.Files.Get` cannot run protoc itself. Re-run this whenever
# `proto/nvidia/omniverse/live/v1alpha/live_service.proto` changes and
# commit the regenerated stubs.
#
# Requires `uv` (https://docs.astral.sh/uv/) or grpcio-tools installed
# in the current Python environment.
#
# IMPORTANT: grpcio-tools version is pinned to GRPCIO_TOOLS_VERSION below.
# The runtime smoke-test-base image ships an older grpcio than what the
# latest grpcio-tools emits code for — generated stubs raise
# `grpcio >= X.Y.Z required` at import time when the runtime is too old.
# Bump this pin in lockstep with the smoke-test-base image's grpcio.

# Keep pinned to the smoke-test-base image's grpcio version.
GRPCIO_TOOLS_VERSION="1.76.0"

set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="helm/live-edit-service/files/generated"
mkdir -p "$OUT_DIR"

if command -v uvx >/dev/null 2>&1; then
    PROTOC_CMD=(uv tool run --from "grpcio-tools==${GRPCIO_TOOLS_VERSION}" python -m grpc_tools.protoc)
elif python3 -c "import grpc_tools.protoc; import importlib.metadata as m; v=m.version('grpcio-tools'); assert v=='${GRPCIO_TOOLS_VERSION}', v" >/dev/null 2>&1; then
    PROTOC_CMD=(python3 -m grpc_tools.protoc)
else
    echo "error: need either 'uv' or grpcio-tools==${GRPCIO_TOOLS_VERSION} installed in python3" >&2
    exit 1
fi

"${PROTOC_CMD[@]}" \
    --proto_path=proto \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    proto/nvidia/omniverse/live/v1alpha/live_service.proto

python3 tools/license_headers.py fix \
    "$OUT_DIR/nvidia/omniverse/live/v1alpha/live_service_pb2.py" \
    "$OUT_DIR/nvidia/omniverse/live/v1alpha/live_service_pb2_grpc.py"

echo "Regenerated stubs in $OUT_DIR"
