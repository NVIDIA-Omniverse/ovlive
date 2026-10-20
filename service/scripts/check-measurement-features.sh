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

# Fence on live-edit's measurement-only code: it must never reach a default or
# release build.
#
# Two such overrides exist today, each read from an environment variable only
# by a build with live-edit's test-only `test-support` feature:
#
# - the snapshot fan-out arena's decay override,
#   MEASUREMENT_FANOUT_ARENA_DIRTY_DECAY_MS, read at startup
#   (`src/allocator.rs`), so one release build with `--features test-support`
#   can measure several decay values;
# - the pod memory limit's cgroup root, MEASUREMENT_POD_MEMORY_CGROUP_ROOT
#   (`src/config/pod_memory.rs`), so a test can start the binary on a fixture
#   cgroup tree instead of the machine's own limit.
#
# The release build (`Dockerfile`: `cargo build --locked --release
# --workspace`, no features) must compile the compiled-in behaviour and
# nothing that reads either variable, so `test-support` must come only from
# the command line.
#
# The same two layers, and the same control, as the fence on fred's test-only
# `live-edit-retention-report` feature in `scripts/check-vendored-fred.sh`
# (which `test-support` also turns on, so that fence covers the resolved edge
# too, in its own CI job):
#
# - The manifest: on the comment-stripped root Cargo.toml, `test-support` is
#   defined on exactly one `test-support = [...]` line, and every other line
#   naming it is a `required-features = [...]` line, which layer 2
#   (`cargo tree`, below) backs: the check matches the line by its key, not by
#   the table it sits in, and on a target such a line only gates the target
#   on the feature and never enables it. So no `default` entry, no other
#   feature and no dependency's `features` list enables it.
# - What the release build resolves: `cargo tree -e features -i live-edit`
#   for the whole workspace with no features selected must not list the
#   feature on live-edit. Feature resolution does not depend on the profile,
#   so this is the release invocation's resolution. It catches an edge the
#   manifest check cannot see, such as a workspace member that depends on
#   live-edit with `features = ["test-support"]`. `cargo tree`'s default
#   edges include dev-dependencies, which the release `cargo build` does not
#   build, so a dev-dependency edge fails here too: deliberately stricter
#   than the image, so the feature stays reachable only through `--features`
#   on the command line.
#
#   The targets are the release image's: the two linux-gnu targets its
#   `rust:1.94-slim` builder builds natively (`RELEASE_TARGETS`), so a
#   target-specific edge that reaches the shipped binary is counted. Not
#   `--target all`: that needs every platform's packages (Windows, Android
#   and others), which no Linux build fetches, so it would need crates.io
#   beyond the build's own fetch, and an edge on such a platform never
#   reaches the image. Both targets resolve from what a Linux build of the
#   workspace has fetched, offline included.
#
#   A control run with the feature selected must list it, so the check cannot
#   pass because its pattern no longer matches `cargo tree`'s output.
#
# A third layer checks built artifacts. `--binary <path>`: the binary must not
# contain either variable's name. `--control-binary <path>`, for a build with
# `--features test-support`: it must contain both names, so the search is
# known to find each name where its override is compiled in. Both require
# each name's source file to declare it exactly once as a string literal (a
# renamed variable fails here rather than leaving the search looking for a
# name nothing carries). The Dockerfile runs the `--binary` search on the
# release binary it builds (its `measurement-feature gate`); the build context
# carries no `scripts/`, so the Dockerfile repeats the names and keeps the
# same source control. Locally, on a release build of each:
#   sh scripts/check-measurement-features.sh --binary target/release/live-edit
#   sh scripts/check-measurement-features.sh --control-binary <test-support build>
#
# POSIX sh. The CI job (`measurement-features:fence`) runs the first two layers
# on Debian (`rust:1.94-slim`, with crates.io reachable), and
# `scripts/guard-tests.sh` runs them after its build; locally:
# `sh scripts/check-measurement-features.sh`.

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PACKAGE=live-edit
FEATURE=test-support
# Each measurement-only variable and the one source file that declares it, as
# `NAME:path` pairs.
ENV_NAMES="MEASUREMENT_FANOUT_ARENA_DIRTY_DECAY_MS:src/allocator.rs MEASUREMENT_POD_MEMORY_CGROUP_ROOT:src/config/pod_memory.rs"
RELEASE_TARGETS="--target x86_64-unknown-linux-gnu --target aarch64-unknown-linux-gnu"
FAIL_WHY="the measurement-only overrides would reach a default or release build"
USAGE="usage: sh scripts/check-measurement-features.sh [--binary <path> | --control-binary <path>]"

# `$1` is the failure; `$2` names its cause, and with it the guidance: `edge`
# (the feature reaches a featureless build, or the release binary carries the
# variable), `unresolved` (`cargo tree` or the binary could not be read) or
# `control` (the fence's pattern no longer matches). Every cause fails.
fail() {
  echo ""
  echo "=========================================="
  case "$2" in
    edge)
      echo "ERROR: $PACKAGE's $FEATURE feature, which compiles in measurement-only code, could reach a default or release build." ;;
    *)
      echo "ERROR: the fence on $PACKAGE's measurement-only code ($FEATURE) could not verify the build." ;;
  esac
  echo ""
  echo "$1"
  echo ""
  case "$2" in
    edge)
      echo "The feature must be enabled only from the command line"
      echo "(\`--features $FEATURE\`, as the tests, CI's test and lint jobs and a"
      echo "measurement build pass it). Remove the edge that turns it on: a"
      echo "\`default\` entry, another feature listing it, a workspace member's"
      echo "dependency on $PACKAGE with that feature, or a dev-dependency edge with"
      echo "it. A target may name it only in \`required-features\`."
      ;;
    unresolved)
      echo "The fence did not run, so nothing is known about the feature's edges."
      echo "Check the job's network and crates.io registry access, the toolchain"
      echo "(rust-toolchain.toml), that Cargo.lock is current (\`--locked\`) and,"
      echo "for --binary or --control-binary, that the path names a built binary;"
      echo "then re-run."
      ;;
    *)
      echo "The fence's pattern no longer matches what it searches, so its check"
      echo "could not have seen the feature or the variable either. Update this"
      echo "script (and the Dockerfile's measurement-feature gate) to the current"
      echo "spelling."
      ;;
  esac
  echo "=========================================="
  exit 1
}

# The source control of the binary layer: `$2` names the variable `$1`
# exactly once, as the string literal the binary would carry.
check_env_name_source() {
  SOURCE_HITS=$(grep -c -F "\"$1\"" "$2" || true)
  [ "$SOURCE_HITS" = 1 ] \
    || fail "$2 declares \"$1\" $SOURCE_HITS time(s), not exactly once, so the binary search below would look for a name the source no longer reads." control
}

# grep's status for the name `$1` (declared in `$2`) in the binary `$3`: 0
# found, 1 not found; anything else is an error, which must not pass as
# either.
search_binary() {
  [ -s "$3" ] || fail "$3 is missing or empty." unresolved
  check_env_name_source "$1" "$2"
  set +e
  grep -q -a -F -- "$1" "$3"
  rc=$?
  set -e
  case "$rc" in
    0|1) return "$rc" ;;
    *) fail "grep could not read $3 (exit $rc)." unresolved ;;
  esac
}

case "${1:-}" in
  --binary)
    [ "$#" = 2 ] || { echo "$USAGE" >&2; exit 2; }
    for pair in $ENV_NAMES; do
      name=${pair%%:*}
      if search_binary "$name" "${pair#*:}" "$2"; then
        fail "$2 contains the name $name: it was built with $FEATURE, so $FAIL_WHY." edge
      fi
      echo "OK: $2 does not contain $name, a variable only a $FEATURE build reads."
    done
    exit 0
    ;;
  --control-binary)
    [ "$#" = 2 ] || { echo "$USAGE" >&2; exit 2; }
    for pair in $ENV_NAMES; do
      name=${pair%%:*}
      search_binary "$name" "${pair#*:}" "$2" \
        || fail "$2 does not contain the name $name, so either it was not built with --features $FEATURE or the search no longer finds the name where the override is compiled in, and the --binary check could not have seen it either." control
      echo "OK: the control $2 contains $name, so the --binary search finds the name where the override is compiled in."
    done
    exit 0
    ;;
  "") ;;
  *) echo "$USAGE" >&2; exit 2 ;;
esac
[ "$#" = 0 ] || { echo "$USAGE" >&2; exit 2; }

TOML_CODE=$(sed 's/#.*//' Cargo.toml)
DEFINITION_HITS=$(printf '%s\n' "$TOML_CODE" | grep -c -E "^[[:space:]]*$FEATURE[[:space:]]*=" || true)
OTHER_LINES=$(printf '%s\n' "$TOML_CODE" | grep -F -- "$FEATURE" | grep -v -E "^[[:space:]]*($FEATURE|required-features)[[:space:]]*=" || true)
[ "$DEFINITION_HITS" = 1 ] \
  || fail "Cargo.toml defines $FEATURE $DEFINITION_HITS time(s), not exactly once on its own \`$FEATURE = [...]\` line." control
[ -z "$OTHER_LINES" ] \
  || fail "Cargo.toml names $FEATURE outside its definition and \`required-features\` lines, so something may enable it: $FAIL_WHY. The lines:
$OTHER_LINES" edge

RESOLVED_NODE="$PACKAGE feature \"$FEATURE\""
# shellcheck disable=SC2086 # RELEASE_TARGETS is two `--target <triple>` pairs.
RELEASE_TREE=$(cargo tree --locked --workspace $RELEASE_TARGETS -e features -i "$PACKAGE") \
  || fail "cargo tree could not resolve the workspace's features (output above), so the fence could not run." unresolved
if printf '%s\n' "$RELEASE_TREE" | grep -qF "$RESOLVED_NODE"; then
  printf '%s\n' "$RELEASE_TREE"
  fail "With no features selected, the workspace resolves $PACKAGE with $FEATURE (cargo tree above; the edges under '$RESOLVED_NODE' enable it): $FAIL_WHY." edge
fi
# shellcheck disable=SC2086
CONTROL_TREE=$(cargo tree --locked --workspace $RELEASE_TARGETS -e features -i "$PACKAGE" --features "$PACKAGE/$FEATURE") \
  || fail "cargo tree could not resolve the workspace's features with $PACKAGE/$FEATURE (output above), so the fence could not run." unresolved
printf '%s\n' "$CONTROL_TREE" | grep -qF "$RESOLVED_NODE" \
  || fail "The fence's control failed: with $PACKAGE/$FEATURE, cargo tree does not list '$RESOLVED_NODE', so the release check above could not have seen it either." control

echo "OK: $PACKAGE's $FEATURE feature, which compiles in the measurement-only overrides, is enabled by no manifest edge, and the no-feature workspace build does not resolve it."
