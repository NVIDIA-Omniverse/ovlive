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
# Exercise the exact built images. Reports are copied from the stopped runner
# before Compose tears down the isolated network and containers.
set -eu
: "${SERVICE_IMAGE:?Set SERVICE_IMAGE}"
: "${CONFORMANCE_IMAGE:?Set CONFORMANCE_IMAGE}"
CONFORMANCE_COMPOSE_PROJECT="le-conformance-${CI_JOB_ID:-local-$$}"
compose() {
  docker compose --project-name "$CONFORMANCE_COMPOSE_PROJECT" -f conformance_tests/docker-compose.yml "$@"
}
mkdir -p conformance_tests/reports
finish() {
  status=$?
  trap - EXIT
  set +e
  compose logs --no-color > conformance_tests/reports/services.log 2>&1 || true
  container="$(compose ps --all --quiet conformance)"
  if [ -n "$container" ]; then
    docker cp "$container:/reports/." conformance_tests/reports/ || status=1
  else
    status=1
  fi
  if [ ! -s conformance_tests/reports/conformance-results.xml ]; then status=1; fi
  compose down --volumes --remove-orphans || status=1
  exit "$status"
}
trap finish EXIT
compose up --abort-on-container-exit --exit-code-from conformance
