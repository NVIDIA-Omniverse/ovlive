#!/usr/bin/env bash
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

# Ensure a local Valkey is available for integration tests.
#
# Two operating modes:
#
#   * Single-node (default): one Valkey listening on `${HOST}:${PORT}`
#     (default `127.0.0.1:6379`). Reads `TEST_VALKEY_URL` (default
#     `redis://127.0.0.1:6379`). This is what every test path uses unless
#     it explicitly opts into the cluster path via `LIVE_EDIT_TEST_CLUSTER_URL`.
#
#   * Cluster (`cluster-up`): a 3-shard Valkey Cluster on ports
#     `${CLUSTER_PORT_BASE}..${CLUSTER_PORT_BASE}+2` (default 7000..7002),
#     each with `--cluster-enabled yes`, meshed via
#     `valkey-cli --cluster create`. Tests that opt in read
#     `LIVE_EDIT_TEST_CLUSTER_URL`; we print the suggested URL on success.
#     The cluster path runs IN ADDITION to single-node, never instead of it
#     — single-node integration tests are untouched.
#
#     Cluster mode requires Linux `--network host` so the per-shard
#     containers can gossip with each other on plain `${HOST}:${port}`.
#     Docker Desktop (macOS/Windows) does NOT implement host networking,
#     so `cluster-up` will fail there with a clear error. Local cluster
#     runs on a macOS machine need a proper Linux VM (lima/multipass) or
#     a Compose stack that explicitly wires `--add-host
#     host.docker.internal:host-gateway`; we keep this script's cluster
#     path Linux-focused because the CI runner is Linux.
#
# Integration tests in `tests/integration/` require a live Valkey at
# `TEST_VALKEY_URL` (default `redis://127.0.0.1:6379`). Many developers
# already run Valkey / Redis locally (Homebrew, a systemd unit, etc.).
# This script respects that:
#
#   1. If something is already listening on the target port AND responds
#      to PING, leave it alone and print "using existing Valkey".
#   2. Otherwise, spin up a disposable Docker container running upstream
#      Valkey 9 (`valkey/valkey:9.0.3-alpine`, public Docker Hub),
#      configured with no persistence and `maxmemory-policy=noeviction`.
#      Report clearly that a container was started so the caller knows to
#      tear it down later (`down` command or on host shutdown).
#
# The `down` / `cluster-down` commands only remove containers this script
# created; they will never touch a Valkey it found already running.
#
# Usage:
#   scripts/dev-valkey.sh up            # ensure a single-node Valkey is listening
#   scripts/dev-valkey.sh cluster-up    # ensure a 3-shard cluster is listening
#   scripts/dev-valkey.sh down          # remove single-node + cluster containers
#   scripts/dev-valkey.sh cluster-down  # remove only the cluster containers
#   scripts/dev-valkey.sh status        # print what's listening
#
# Fallback requires Docker. If Docker isn't installed and no Valkey is
# running, `up` fails with a clear error.

set -euo pipefail

NAME="live-edit-test-valkey"
IMAGE="${LIVE_EDIT_TEST_VALKEY_IMAGE:-valkey/valkey:9.0.3-alpine}"
HOST="${LIVE_EDIT_TEST_VALKEY_HOST:-127.0.0.1}"
PORT="${LIVE_EDIT_TEST_VALKEY_PORT:-6379}"
# `docker ps --format` template that prints one container name per line.
DOCKER_NAMES_FMT='{{.Names}}'

# Cluster mode constants. We deliberately use a distinct port range
# (default 7000..7002) so a single-node Valkey on 6379 can coexist with the
# cluster on the same host without conflict. Three shards is the minimum
# legal cluster size; we don't bother with replicas (`--cluster-replicas
# 0`) because tests just need slot-distribution coverage, not failover.
CLUSTER_NAME_PREFIX="live-edit-test-valkey-cluster"
CLUSTER_PORT_BASE="${LIVE_EDIT_TEST_VALKEY_CLUSTER_PORT_BASE:-7000}"
CLUSTER_NODE_COUNT="${LIVE_EDIT_TEST_VALKEY_CLUSTER_NODES:-3}"

# Return 0 if something on $1:$2 answers PING with PONG.
ping_existing() {
  local target_host="$1"
  local target_port="$2"
  local cli
  for cli in valkey-cli redis-cli; do
    if command -v "$cli" >/dev/null 2>&1; then
      # Suppress output; just check exit code.
      if "$cli" -h "$target_host" -p "$target_port" ping 2>/dev/null | grep -q PONG; then
        return 0
      fi
      # Found a CLI and got a response, just not PONG — stop probing.
      return 1
    fi
  done
  # No client available — fall back to a raw TCP probe + RESP PING.
  if ! command -v python3 >/dev/null 2>&1; then
    return 1
  fi
  python3 - "$target_host" "$target_port" <<'PY' 2>/dev/null
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
try:
    s = socket.create_connection((host, port), timeout=0.5)
    s.sendall(b"*1\r\n$4\r\nPING\r\n")
    buf = s.recv(32)
    sys.exit(0 if b"PONG" in buf else 1)
except Exception:
    sys.exit(1)
PY
}

# Issue a CLUSTER INFO against $1:$2 and return 0 iff `cluster_state:ok`.
# Used to wait for the post-create slot assignment to settle before
# returning to the caller — `valkey-cli --cluster create` exits as soon as
# it dispatches the slot allocation, but the cluster nodes themselves
# converge asynchronously.
#
# Probe order:
#   1. Local `valkey-cli` / `redis-cli` if installed (developer machines).
#   2. `docker exec <cluster-container-0> valkey-cli ...` (CI build containers
#      that have docker-cli but no Valkey/Redis client — the cluster state
#      is consistent across nodes, so any one node's view suffices).
cluster_info_ok() {
  local target_host="$1"
  local target_port="$2"
  local cli
  for cli in valkey-cli redis-cli; do
    if command -v "$cli" >/dev/null 2>&1; then
      if "$cli" -h "$target_host" -p "$target_port" cluster info 2>/dev/null \
        | grep -q '^cluster_state:ok'; then
        return 0
      fi
      return 1
    fi
  done
  if command -v docker >/dev/null 2>&1; then
    local fname
    fname="$(cluster_container_name 0)"
    if docker exec "$fname" valkey-cli -h "$target_host" -p "$target_port" cluster info 2>/dev/null \
      | grep -q '^cluster_state:ok'; then
      return 0
    fi
  fi
  return 1
}

cluster_container_name() {
  echo "${CLUSTER_NAME_PREFIX}-$1"
}

# Bring up the 3-shard cluster. Idempotent: if every node already responds
# to PING and `cluster info` reports `cluster_state:ok`, we leave it alone.
#
# Uses `--network host` so the per-shard containers can gossip with each
# other on plain `${HOST}:${port}` without any port-mapping translation.
# This is Linux-only. On macOS/Windows Docker Desktop, `--network host` is
# a no-op and the gossip ports never connect — `cluster_info_ok` will not
# converge and the script exits with a clear error.
cluster_up() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "error: cluster-up requires docker (not found on PATH)." >&2
    exit 1
  fi

  # Fast path: if every container is already healthy, just report status
  # and exit. Critical for CI where the same job sometimes re-invokes
  # this script across script blocks.
  local healthy=1
  local i p
  for i in $(seq 0 $((CLUSTER_NODE_COUNT - 1))); do
    p=$((CLUSTER_PORT_BASE + i))
    if ! ping_existing "$HOST" "$p"; then
      healthy=0
      break
    fi
  done
  if [ "$healthy" -eq 1 ] && cluster_info_ok "$HOST" "$CLUSTER_PORT_BASE"; then
    echo "cluster already running on ${HOST}:${CLUSTER_PORT_BASE}..${HOST}:$((CLUSTER_PORT_BASE + CLUSTER_NODE_COUNT - 1))"
    echo "suggested LIVE_EDIT_TEST_CLUSTER_URL: redis-cluster://${HOST}:${CLUSTER_PORT_BASE}"
    return 0
  fi

  # Tear down any stale per-node containers from a previous attempt before
  # we start fresh — half-up clusters cannot be re-meshed because the node
  # IDs persist in their RDB-less working state but the slot assignments
  # were never recorded.
  local name
  for i in $(seq 0 $((CLUSTER_NODE_COUNT - 1))); do
    name="$(cluster_container_name "$i")"
    docker rm -f "$name" >/dev/null 2>&1 || true
  done

  # Spin up each shard container with `--network host` so cross-shard
  # gossip works without an explicit port-bus dance.
  for i in $(seq 0 $((CLUSTER_NODE_COUNT - 1))); do
    p=$((CLUSTER_PORT_BASE + i))
    name="$(cluster_container_name "$i")"
    docker run -d --rm \
      --name "$name" \
      --network host \
      "$IMAGE" \
      valkey-server \
        --port "$p" \
        --cluster-enabled yes \
        --cluster-node-timeout 5000 \
        --appendonly no \
        --save "" \
        --maxmemory-policy noeviction \
        --tcp-backlog 4096 \
      >/dev/null
  done

  # Wait for each node to come up before issuing the cluster create.
  local up _t
  for i in $(seq 0 $((CLUSTER_NODE_COUNT - 1))); do
    p=$((CLUSTER_PORT_BASE + i))
    name="$(cluster_container_name "$i")"
    up=0
    for _t in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
      if docker exec "$name" valkey-cli -p "$p" ping 2>/dev/null | grep -q PONG; then
        up=1
        break
      fi
      sleep 0.3
    done
    if [ "$up" -ne 1 ]; then
      echo "error: cluster node $name (port $p) failed to become ready" >&2
      docker logs "$name" >&2 || true
      exit 1
    fi
  done

  # Mesh the nodes into a cluster. `valkey-cli` dispatches the slot
  # assignment and CLUSTER MEET calls in one shot. `--cluster-yes` skips
  # the interactive confirmation prompt. With `--network host`, all
  # ${HOST}:${port} addresses are reachable from every container.
  local addrs=""
  for i in $(seq 0 $((CLUSTER_NODE_COUNT - 1))); do
    p=$((CLUSTER_PORT_BASE + i))
    addrs="$addrs ${HOST}:${p}"
  done
  local first_name
  first_name="$(cluster_container_name 0)"
  if ! docker exec "$first_name" valkey-cli --cluster create $addrs --cluster-yes >/dev/null 2>&1; then
    echo "error: valkey-cli --cluster create failed" >&2
    docker logs "$first_name" >&2 || true
    exit 1
  fi

  # Wait for the cluster_state to flip to `ok`. Slot assignment is
  # synchronous inside `--cluster create` but the gossip can take a
  # second to converge on `ok`. We probe with up to ~6 s of wall clock
  # before giving up — this is the same budget the single-node path
  # uses (10 × 300 ms).
  local ready=0
  for _t in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if cluster_info_ok "$HOST" "$CLUSTER_PORT_BASE"; then
      ready=1
      break
    fi
    sleep 0.3
  done
  if [ "$ready" -ne 1 ]; then
    echo "error: cluster_state did not converge to 'ok' within timeout" >&2
    echo "       cluster_info_ok probes locally-installed valkey-cli/redis-cli" >&2
    echo "       first, then falls back to 'docker exec' against node 0. If" >&2
    echo "       neither path is available, the wait loop returns false even" >&2
    echo "       when the cluster has actually converged." >&2
    echo "       On Docker Desktop, --network host is a no-op so cluster" >&2
    echo "       gossip cannot route between containers — cluster-up is" >&2
    echo "       Linux-only in this script." >&2
    exit 1
  fi

  echo "started 3-shard Valkey cluster on ${HOST}:${CLUSTER_PORT_BASE}..${HOST}:$((CLUSTER_PORT_BASE + CLUSTER_NODE_COUNT - 1))"
  echo "suggested LIVE_EDIT_TEST_CLUSTER_URL: redis-cluster://${HOST}:${CLUSTER_PORT_BASE}"
  echo "(run '$0 cluster-down' when finished)"
}

cluster_down() {
  local i name removed=0
  for i in $(seq 0 $((CLUSTER_NODE_COUNT - 1))); do
    name="$(cluster_container_name "$i")"
    if docker ps -a --filter "name=^${name}$" --format "$DOCKER_NAMES_FMT" 2>/dev/null | grep -q "^${name}$"; then
      docker rm -f "$name" >/dev/null 2>&1
      removed=$((removed + 1))
    fi
  done
  if [ "$removed" -gt 0 ]; then
    echo "removed $removed cluster node container(s)"
  else
    echo "no cluster node containers to remove. This script never touches a cluster it didn't start."
  fi
}

cmd="${1:-up}"

case "$cmd" in
  up)
    if ping_existing "$HOST" "$PORT"; then
      echo "using existing Valkey at ${HOST}:${PORT}"
      exit 0
    fi

    if ! command -v docker >/dev/null 2>&1; then
      echo "error: no Valkey listening on ${HOST}:${PORT} and docker is not installed." >&2
      echo "       Start Valkey via Homebrew/systemd/your preferred method and re-run." >&2
      exit 1
    fi

    if docker ps --filter "name=^${NAME}$" --format "$DOCKER_NAMES_FMT" | grep -q "^${NAME}$"; then
      # Container is up but didn't answer PING yet — let it have a moment.
      :
    else
      docker rm -f "$NAME" >/dev/null 2>&1 || true
      # `--tcp-backlog 4096` raises the listen-queue depth above Valkey's
      # default 511 so the post-`b.fgj` workspace test suite's SYN burst
      # (hundreds of fixtures spinning up pools simultaneously) does not
      # overflow the accept queue and force fred clients into the
      # connect-timeout retry path (b.1kf).
      docker run -d --rm \
        --name "$NAME" \
        -p "${PORT}:6379" \
        "$IMAGE" \
        valkey-server --appendonly no --save "" --maxmemory-policy noeviction \
                      --tcp-backlog 4096 \
        >/dev/null
    fi

    # Wait briefly until Valkey responds.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if docker exec "$NAME" valkey-cli ping 2>/dev/null | grep -q PONG; then
        echo "started disposable Valkey container '${NAME}' on ${HOST}:${PORT}"
        echo "(no existing Valkey found; run '$0 down' when finished)"
        exit 0
      fi
      sleep 0.3
    done
    echo "error: ${NAME} failed to become ready" >&2
    docker logs "$NAME" >&2 || true
    exit 1
    ;;
  cluster-up)
    cluster_up
    ;;
  cluster-down)
    cluster_down
    ;;
  down)
    if docker ps -a --filter "name=^${NAME}$" --format "$DOCKER_NAMES_FMT" 2>/dev/null | grep -q "^${NAME}$"; then
      docker rm -f "$NAME" >/dev/null 2>&1
      echo "${NAME} removed"
    else
      echo "${NAME} not running (nothing to remove). This script never touches a Valkey it didn't start."
    fi
    # Also tear down any cluster containers — `down` is the "clean slate"
    # command. Use the dedicated `cluster-down` if you want to keep
    # single-node up but tear down the cluster (or vice versa).
    cluster_down
    ;;
  status)
    if ping_existing "$HOST" "$PORT"; then
      printf "Valkey listening on %s:%s — " "$HOST" "$PORT"
      if docker ps --filter "name=^${NAME}$" --format "$DOCKER_NAMES_FMT" 2>/dev/null | grep -q "^${NAME}$"; then
        echo "source: disposable Docker container '${NAME}'"
      else
        echo "source: external (not this script's container)"
      fi
    else
      echo "Valkey NOT listening on ${HOST}:${PORT}"
    fi
    # Cluster status, if any node is up.
    if ping_existing "$HOST" "$CLUSTER_PORT_BASE"; then
      if cluster_info_ok "$HOST" "$CLUSTER_PORT_BASE"; then
        echo "Cluster healthy on ${HOST}:${CLUSTER_PORT_BASE}..${HOST}:$((CLUSTER_PORT_BASE + CLUSTER_NODE_COUNT - 1))"
      else
        echo "Cluster node listening on ${HOST}:${CLUSTER_PORT_BASE} but cluster_state is not ok"
      fi
    fi
    ;;
  *)
    echo "usage: $0 {up|cluster-up|down|cluster-down|status}" >&2
    exit 2
    ;;
esac
