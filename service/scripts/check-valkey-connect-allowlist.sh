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

# Guardrail: enforce that `ValkeyClient::connect(` is only ever *called* from an
# explicit allowlist of files. Every test fixture that needs a real Valkey
# client MUST route through `connect_test_client` in
# `src/storage/test_support.rs` (which owns the setup-semaphore that bounds
# concurrent connect storms). Direct `ValkeyClient::connect(` calls outside the
# allowlist silently bypass that contract.
#
# This makes the "MUST route through connect_test_client" rule mechanically
# self-enforcing instead of convention-only. Runnable locally
# (`bash scripts/check-valkey-connect-allowlist.sh`) and in CI
# (`valkey-connect:allowlist` job in .gitlab-ci.yml).
#
# What counts as a violation:
#   A `ValkeyClient::connect(` *invocation* (note the open paren) that lives in
#   a source/test file NOT on the allowlist below. Doc-comment / `//` mentions
#   are NOT invocations and are filtered out before the allowlist check.
#
# To fix a flagged call site, route the connect through `connect_test_client`
# (see src/storage/test_support.rs). If a direct call is genuinely warranted
# (e.g. a deliberate failure-path test), add the file to ALLOWLIST below with a
# justifying comment and reviewer sign-off.

set -euo pipefail

# Run from the repo root regardless of invocation cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

HELPER="src/storage/test_support.rs"

# Allowlist by file path (line-drift independent). Any `ValkeyClient::connect(`
# invocation in a file NOT listed here is a violation.
ALLOWLIST=(
  "src/main.rs"                                          # production startup
  "src/storage/test_support.rs"                          # connect_test_client's own internal call
  "tests/integration/storage/client_test.rs"             # deliberate failure-path tests: unreachable address + rediss-against-plaintext (standalone and clustered)
  "tests/integration/storage/client_quit_timeout_test.rs" # pre-existing documented quit-timeout exceptions
)

is_allowlisted() {
  local f="$1"
  local allowed
  for allowed in "${ALLOWLIST[@]}"; do
    if [ "$f" = "$allowed" ]; then
      return 0
    fi
  done
  return 1
}

# Find every line matching the *invocation* pattern across the source tree.
# `--include` keeps us to Rust sources; `-n` gives file:line for diagnostics.
# We grep the call (with open paren, tolerating optional whitespace before it via
# `[[:space:]]*`) so prose like "the client constructed by `ValkeyClient::connect`"
# (no paren) is never a hit, while `ValkeyClient::connect (&cfg)` still is.
# NOTE: the rare cross-line-split spelling (`connect` and `(` on different lines)
# and other non-canonical forms are out of reach of this line-based grep; detection
# of those relies on the repo being `cargo fmt`-clean (CI enforces fmt), which
# normalizes them back into the canonical `connect(` form this gate matches.
MATCHES=$(grep -rnE --include='*.rs' 'ValkeyClient::connect[[:space:]]*\(' src tests 2>/dev/null || true)

VIOLATIONS=()
while IFS= read -r line; do
  [ -z "$line" ] && continue
  # line is "path:lineno:content"
  file="${line%%:*}"
  rest="${line#*:}"
  lineno="${rest%%:*}"
  content="${rest#*:}"

  # Drop comment lines: a real invocation is code, not a `//` or `///` comment.
  # Strip leading whitespace, then skip if the line starts with `//`.
  trimmed="${content#"${content%%[![:space:]]*}"}"
  case "$trimmed" in
    //*) continue ;;  # `//` line comment or `///` doc comment -> a mention, not a call
    *) ;;             # code line -> fall through to the allowlist check
  esac

  if ! is_allowlisted "$file"; then
    VIOLATIONS+=("$file:$lineno:$trimmed")
  fi
done <<< "$MATCHES"

if [ "${#VIOLATIONS[@]}" -ne 0 ]; then
  echo ""
  echo "=========================================="
  echo "ERROR: Disallowed ValkeyClient::connect( call site(s) detected."
  echo ""
  echo "These invocations bypass the test-client setup-semaphore contract:"
  for v in "${VIOLATIONS[@]}"; do
    echo "  $v"
  done
  echo ""
  echo "Every Valkey client outside the allowlist MUST be constructed via"
  echo "connect_test_client in $HELPER (it bounds concurrent connect storms"
  echo "via the setup-semaphore). Route the flagged call(s) through that helper."
  echo ""
  echo "If a direct ValkeyClient::connect( call is genuinely warranted, add the"
  echo "file to the ALLOWLIST in scripts/check-valkey-connect-allowlist.sh with"
  echo "a justifying comment and reviewer sign-off."
  echo "=========================================="
  exit 1
fi

echo "ValkeyClient::connect( call sites are all allowlisted (${#ALLOWLIST[@]} allowed files)."
