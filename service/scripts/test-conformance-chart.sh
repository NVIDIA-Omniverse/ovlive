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
#
# Offline render checks: never contacts or changes a cluster.
set -eu
chart=helm/live-edit-conformance-tests
scratch="$(mktemp -d)"
helm lint "$chart" --set service.basePath=https://example.test/base --set auth.existingSecret=bearer
helm template suite "$chart" --set service.basePath=https://example.test/base --set auth.existingSecret=bearer > "$scratch/secret.yaml"
grep -q 'backoffLimit: 0' "$scratch/secret.yaml"
grep -q 'readOnlyRootFilesystem: true' "$scratch/secret.yaml"
grep -q 'secretName: bearer' "$scratch/secret.yaml"
helm template suite "$chart" --set service.basePath=https://example.test/base --set auth.sidecar.image=identity:1 --set serviceAccountName=trusted-subject > "$scratch/sidecar.yaml"
grep -q 'serviceAccountName: trusted-subject' "$scratch/sidecar.yaml"
grep -q 'restartPolicy: Always' "$scratch/sidecar.yaml"
if grep -q 'kind: ServiceAccount' "$scratch/sidecar.yaml"; then exit 1; fi
if helm template suite "$chart" --set service.basePath=https://example.test/base > "$scratch/invalid.yaml" 2>&1; then
  echo 'Chart accepted missing credentials' >&2; exit 1
fi
if helm template suite "$chart" --set service.basePath=https://example.test/base --set auth.existingSecret=bearer --set auth.sidecar.image=identity:1 > "$scratch/conflict.yaml" 2>&1; then
  echo 'Chart accepted conflicting credentials' >&2; exit 1
fi
