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

# Drift check for the vendored, patched fred crate (`vendor/fred-10.1.0`).
#
# The tree must be exactly the published crates.io release plus the one patch
# this repository records, and nothing else. The check rebuilds it from upstream
# and compares:
#   1. fetch `fred-<version>.crate` from crates.io (or take a local copy from
#      `FRED_CRATE_ARCHIVE`, e.g. `$CARGO_HOME/registry/cache/.../fred-10.1.0.crate`);
#   2. verify its SHA-256 against the checksum recorded in the SOURCE.md;
#   3. extract it and apply `vendor/fred-<version>.patch` with no fuzz;
#   4. `diff -r` the result against the vendored tree. In the GitHub source
#      release only, the two upstream files git ignore rules drop from it may
#      be absent (see PUBLISH_DROPPED below); every other difference fails;
#   5. require the diff quoted in the SOURCE.md to be byte-identical to the
#      patch file, so the record cannot describe a different change;
#   6. require the root Cargo.toml's `[patch.crates-io]` table to patch fred to
#      this tree;
#   7. keep fred's test-only `live-edit-retention-report` feature (the
#      retention report, test sweep modes, sweep counters and in-sweep flag,
#      `fred::live_edit::retention::test_support`) out of the release build:
#      on the comment-stripped root Cargo.toml it appears exactly once, on the
#      `test-support = [...]` line, and the workspace's resolved features with
#      no features selected (the Dockerfile's `cargo build --locked --release
#      --workspace`) do not turn it on for fred — so an edge that later enables
#      `test-support` through feature unification fails here, not in the image.
#      A control run with `live-edit/test-support` must turn it on, so the
#      check cannot pass because its pattern stopped matching.
#   8. keep fred's `replicas` feature out of every build: with every feature
#      of every workspace member selected, the resolved features do not turn
#      it on for fred. The patch's inline-connect flush does not cover the
#      replica reconnect loops that feature compiles in, and its check before a
#      write skips a command routed to a replica (SOURCE.md change 5).
#      Step 7's control already proves `cargo tree` still prints a fred
#      feature node in the form this step greps for.
# The re-vendor and drop procedures are in `vendor/fred-10.1.0.SOURCE.md`. The
# tree's RustSec advisory check is a separate script,
# `scripts/audit-vendored-fred.sh`, which the same CI job runs after this one.
#
# POSIX sh. The CI job (`fred:vendor-drift-check`) runs it on Debian
# (`rust:1.94-slim`, whose cargo step 7 uses, with crates.io reachable) with
# curl, GNU patch and GNU diffutils installed; locally:
# `sh scripts/check-vendored-fred.sh`. `DRIFT_WORK_DIR` names a scratch
# directory to use (and keep) instead of a temporary one.

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

CRATE=fred
VERSION=10.1.0
TREE="vendor/$CRATE-$VERSION"
PATCH_FILE="$TREE.patch"
RECORD="$TREE.SOURCE.md"
URL="https://static.crates.io/crates/$CRATE/$CRATE-$VERSION.crate"

fail() {
  echo ""
  echo "=========================================="
  echo "ERROR: vendored $CRATE $VERSION drift detected."
  echo ""
  echo "$1"
  echo ""
  echo "$TREE must be the published crate plus exactly the patch in $PATCH_FILE,"
  echo "and $RECORD must record both. See the re-vendor procedure in $RECORD."
  echo "=========================================="
  exit 1
}

[ -d "$TREE" ] || fail "Vendored tree $TREE is missing."
[ -f "$PATCH_FILE" ] || fail "Patch file $PATCH_FILE is missing."
[ -f "$RECORD" ] || fail "Record $RECORD is missing."

EXPECTED_SHA=$(grep -E '\*\*Crate SHA256:\*\*' "$RECORD" | grep -oE '[0-9a-f]{64}' | head -n1 || true)
[ -n "$EXPECTED_SHA" ] || fail "Could not parse the crate checksum from $RECORD (a line containing '**Crate SHA256:**' and a 64-hex-digit digest)."

if [ -n "${DRIFT_WORK_DIR:-}" ]; then
  WORK="$DRIFT_WORK_DIR"
  mkdir -p "$WORK"
else
  WORK=$(mktemp -d)
  trap 'rm -rf "$WORK"' EXIT
fi
ARCHIVE="$WORK/$CRATE-$VERSION.crate"
UPSTREAM="$WORK/upstream"

if [ -n "${FRED_CRATE_ARCHIVE:-}" ]; then
  cp "$FRED_CRATE_ARCHIVE" "$ARCHIVE"
else
  curl -sS -L --fail --retry 3 -o "$ARCHIVE" "$URL"
fi

ACTUAL_SHA=$(sha256sum "$ARCHIVE" | awk '{print $1}')
[ "$EXPECTED_SHA" = "$ACTUAL_SHA" ] || fail "Upstream crate checksum mismatch.
  Expected: $EXPECTED_SHA (from $RECORD)
  Actual:   $ACTUAL_SHA ($URL)"

mkdir -p "$UPSTREAM"
tar -xzf "$ARCHIVE" -C "$UPSTREAM"
patch -p1 --fuzz=0 --no-backup-if-mismatch -d "$UPSTREAM/$CRATE-$VERSION" -i "$REPO_ROOT/$PATCH_FILE" \
  || fail "$PATCH_FILE does not apply cleanly to the published $CRATE $VERSION crate."

# The GitHub source release is committed with `git add -A`, which honours
# `.gitignore` rules: fred's own `.gitignore` drops `Cargo.lock`, and the
# target repository's drops `.vscode/`. Neither file is read by any build (cargo
# never reads a dependency's lockfile). So in that copy a tree that lacks one of
# them still passes; a tree that has one must match it byte for byte, like every
# other file. The copy is told apart by its missing `.gitlab-ci.yml`, which
# `.publishignore` always excludes. In this repository every file must be
# present: a re-vendor that forgets the `git add -f` of the ignored `Cargo.lock`
# fails here.
if [ -e .gitlab-ci.yml ]; then
  PUBLISH_DROPPED=""
else
  PUBLISH_DROPPED="Cargo.lock .vscode/settings.json"
fi
for rel in $PUBLISH_DROPPED; do
  if [ ! -e "$TREE/$rel" ] && [ -e "$UPSTREAM/$CRATE-$VERSION/$rel" ]; then
    rm -f "$UPSTREAM/$CRATE-$VERSION/$rel"
    dir=$(dirname "$rel")
    if [ "$dir" != "." ] && [ ! -e "$TREE/$dir" ]; then
      rmdir "$UPSTREAM/$CRATE-$VERSION/$dir" 2>/dev/null || true
    fi
    echo "note: $TREE/$rel is absent; accepted (dropped from the GitHub source release by .gitignore rules)"
  fi
done

diff -r "$UPSTREAM/$CRATE-$VERSION" "$TREE" \
  || fail "$TREE differs from the published crate with $PATCH_FILE applied (diff above)."

# The SOURCE.md quotes the patch in its only ```diff block.
awk '/^```diff$/ { inside = 1; next } /^```$/ { inside = 0 } inside' "$RECORD" > "$WORK/quoted.patch"
cmp -s "$WORK/quoted.patch" "$PATCH_FILE" \
  || fail "The diff quoted in $RECORD is not byte-identical to $PATCH_FILE."

# The root manifest must patch fred to this tree; a dropped or repointed patch
# would leave the tree checked here but unused. The entry must sit in the
# `[patch.crates-io]` table: the same line under any other table (a
# `[dependencies]` path dependency, say) does not replace the registry crate
# for the dependencies that pull fred in. Comments are stripped first. Every
# line starting with `[` (after leading whitespace) is taken as a table header
# and ends the table before it, so a quoted or spaced header such as
# `[target.'cfg(unix)'.dependencies]` cannot extend `[patch.crates-io]`; only
# `[patch.crates-io]` itself (spaces allowed around its parts) opens it. The
# quoted spelling `[patch."crates-io"]` is not recognised and fails closed.
TOML_CODE=$(sed 's/#.*//' Cargo.toml)
printf '%s\n' "$TOML_CODE" | awk -v crate="$CRATE" -v tree="$TREE" '
  BEGIN {
    gsub(/\./, "\\.", tree)
    entry = "^" crate "[[:space:]]*=[[:space:]]*\\{[[:space:]]*path[[:space:]]*=[[:space:]]*\"" tree "\""
  }
  /^[[:space:]]*\[/ {
    in_patch = ($0 ~ /^[[:space:]]*\[[[:space:]]*patch[[:space:]]*\.[[:space:]]*crates-io[[:space:]]*\][[:space:]]*$/)
    next
  }
  in_patch && $0 ~ entry { found = 1 }
  END { exit !found }
' \
  || fail "Cargo.toml does not patch $CRATE to $TREE under [patch.crates-io] ($CRATE = { path = \"$TREE\" })."

# fred's `live-edit-retention-report` feature compiles a test-only per-client
# retention report, test sweep modes and sweep counters, and a per-thread
# in-sweep flag, into fred; it must never reach the
# release build (`Dockerfile`: `cargo build --locked --release --workspace`, no
# features). Two layers:
#
# - The manifest: on the comment-stripped root Cargo.toml the feature name
#   appears exactly once, on the `test-support = [...]` line, its one intended
#   enabling edge.
# - What the release build resolves: `cargo tree -e features -i fred` for the
#   whole workspace with no features selected must not list the feature on
#   fred. Feature resolution does not depend on the profile, so this is the
#   release invocation's resolution. It catches an edge the manifest check
#   cannot see, such as a workspace member that depends on live-edit with
#   `features = ["test-support"]`. `cargo tree`'s default edges include
#   dev-dependencies, which the release `cargo build` does not build, so a
#   dev-dependency edge that turns the feature on fails here too:
#   deliberately stricter than the image, so the feature stays reachable only
#   through `--features test-support` on the command line. `--target all`
#   counts every platform's target-specific dependencies.
#
#   A control run with `live-edit/test-support` selected must list it, so the
#   check cannot pass because the pattern no longer matches `cargo tree`'s
#   output.
RETENTION_FEATURE=live-edit-retention-report
RETENTION_FAIL="the retention report would reach a non-test build"

# The fence's own failure banner, not `fail`'s: none of its failures is drift
# in the vendored tree, so the drift banner and re-vendor pointer would
# mislead. `$1` is the failure; `$2` names its cause, and with it the
# guidance: `edge` (the feature reaches a featureless build), `unresolved`
# (`cargo tree` could not run) or `control` (the fence's pattern no longer
# matches `cargo tree`'s output). Every cause fails the check.
fail_retention() {
  echo ""
  echo "=========================================="
  case "$2" in
    edge)
      echo "ERROR: $CRATE's test-only $RETENTION_FEATURE feature could reach a non-test build." ;;
    *)
      echo "ERROR: the fence on $CRATE's test-only $RETENTION_FEATURE feature could not verify the build." ;;
  esac
  echo ""
  echo "$1"
  echo ""
  case "$2" in
    edge)
      echo "The feature's only allowed enabling edge is the root Cargo.toml's"
      echo "  test-support = [\"dep:libc\", \"$CRATE/$RETENTION_FEATURE\"]"
      echo "and test-support itself must come only from the command line"
      echo "(\`--features test-support\`, as the integration tests and CI's test jobs"
      echo "pass it). An edge that turns test-support on for a featureless build is"
      echo "what this rejects: a \`default\` entry, a workspace member's dependency on"
      echo "live-edit with that feature, or a"
      echo "  [dev-dependencies] live-edit = { ..., features = [\"test-support\"] }"
      echo "edge. Remove that edge; do not move the feature elsewhere."
      ;;
    unresolved)
      echo "The fence did not run, so nothing is known about the feature's edges."
      echo "Check the job's network and crates.io registry access, the toolchain"
      echo "(rust-toolchain.toml) and that Cargo.lock is current (\`--locked\`), then"
      echo "re-run."
      ;;
    *)
      echo "The fence's pattern no longer matches cargo tree's output (a cargo"
      echo "upgrade can change its format), so its release check could not have"
      echo "seen the feature either. Update RESOLVED_NODE in this script to what this"
      echo "command prints for the feature:"
      echo "  cargo tree --workspace -e features -i $CRATE --features live-edit/test-support"
      ;;
  esac
  echo "=========================================="
  exit 1
}

MANIFEST_HITS=$(printf '%s\n' "$TOML_CODE" | grep -o "$RETENTION_FEATURE" | wc -l | tr -d ' ')
MANIFEST_LINE=$(printf '%s\n' "$TOML_CODE" | grep "$RETENTION_FEATURE" || true)
[ "$MANIFEST_HITS" = 1 ] \
  && printf '%s\n' "$MANIFEST_LINE" | grep -qE '^[[:space:]]*test-support[[:space:]]*=' \
  || fail_retention "Cargo.toml names $CRATE's $RETENTION_FEATURE feature $MANIFEST_HITS time(s), not exactly once on the test-support = [...] line: $RETENTION_FAIL." edge

RESOLVED_NODE="$CRATE feature \"$RETENTION_FEATURE\""
RELEASE_TREE=$(cargo tree --locked --workspace --target all -e features -i "$CRATE") \
  || fail_retention "cargo tree could not resolve the workspace's features (output above), so the fence could not run." unresolved
if printf '%s\n' "$RELEASE_TREE" | grep -qF "$RESOLVED_NODE"; then
  printf '%s\n' "$RELEASE_TREE"
  fail_retention "With no features selected, the workspace resolves $CRATE with $RETENTION_FEATURE (cargo tree above; the edges under '$RESOLVED_NODE' enable it): $RETENTION_FAIL." edge
fi
TEST_TREE=$(cargo tree --locked --workspace --target all -e features -i "$CRATE" --features live-edit/test-support) \
  || fail_retention "cargo tree could not resolve the workspace's features with live-edit/test-support (output above), so the fence could not run." unresolved
printf '%s\n' "$TEST_TREE" | grep -qF "$RESOLVED_NODE" \
  || fail_retention "The fence's control failed: with live-edit/test-support, cargo tree does not list '$RESOLVED_NODE', so the release check above could not have seen it either." control

# fred's `replicas` feature compiles in the replica connection loops
# (`add_replica_with_policy`, `sync_replicas_with_policy`), which the patch's
# inline-connect flush (SOURCE.md change 5) does not cover: a frame fed just
# before one of their passes could wait unsent through its connects. And the
# patch's check before a write (change 5) skips a command routed to a replica,
# so a frame fed on another connection could wait through that write. So no
# build may turn it on. `--all-features` selects every feature of every
# workspace member at once, and features only add, so a feature that is off
# here is off in every build the workspace can make. Step 7's control run has
# already shown that `cargo tree` prints a resolved fred feature as
# `fred feature "<name>"`, so this pattern still matches when the feature is on.
REPLICAS_FEATURE=replicas
REPLICAS_NODE="$CRATE feature \"$REPLICAS_FEATURE\""

# Its own failure banner, as for the retention fence: `$1` is the failure,
# `$2` its cause, `edge` (the feature resolves) or `unresolved` (`cargo tree`
# could not run).
fail_replicas() {
  echo ""
  echo "=========================================="
  case "$2" in
    edge)
      echo "ERROR: $CRATE's $REPLICAS_FEATURE feature is enabled in a workspace build." ;;
    *)
      echo "ERROR: the fence on $CRATE's $REPLICAS_FEATURE feature could not verify the build." ;;
  esac
  echo ""
  echo "$1"
  echo ""
  case "$2" in
    edge)
      echo "The vendored patch flushes fed frames before every inline connect of the"
      echo "cluster reconnect and sync loops, but not before the replica loops this"
      echo "feature compiles in, and its check before a write skips a command routed"
      echo "to a replica (vendor/$CRATE-$VERSION.SOURCE.md, change 5). Remove the edge"
      echo "that enables it (the edges under '$REPLICAS_NODE' above), or extend the"
      echo "patch to cover those loops and replica writes first."
      ;;
    *)
      echo "The fence did not run, so nothing is known about the feature's edges."
      echo "Check the job's network and crates.io registry access, the toolchain"
      echo "(rust-toolchain.toml) and that Cargo.lock is current (\`--locked\`), then"
      echo "re-run."
      ;;
  esac
  echo "=========================================="
  exit 1
}

ALL_FEATURES_TREE=$(cargo tree --locked --workspace --target all -e features -i "$CRATE" --all-features) \
  || fail_replicas "cargo tree could not resolve the workspace's features with --all-features (output above), so the fence could not run." unresolved
if printf '%s\n' "$ALL_FEATURES_TREE" | grep -qF "$REPLICAS_NODE"; then
  printf '%s\n' "$ALL_FEATURES_TREE"
  fail_replicas "With every workspace feature selected, the workspace resolves $CRATE with $REPLICAS_FEATURE (cargo tree above)." edge
fi

echo "OK: $TREE is the published $CRATE $VERSION crate ($ACTUAL_SHA) plus $PATCH_FILE, as $RECORD records."
echo "OK: $CRATE's $RETENTION_FEATURE feature is enabled only by live-edit's test-support; the no-feature workspace build does not resolve it."
echo "OK: $CRATE's $REPLICAS_FEATURE feature is off in every workspace build."
