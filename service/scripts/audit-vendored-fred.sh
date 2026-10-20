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

# RustSec advisory check for the vendored fred crate.
#
# `[patch.crates-io]` builds fred from `vendor/fred-<version>`, so its
# `Cargo.lock` entry carries no `source`, and `cargo audit` treats a package
# without a registry source as local code and never matches it against the
# RustSec database: the `supply-chain:audit` job passes even when fred has an
# advisory. This script audits it on its own instead:
#   1. find every `fred` package in `Cargo.lock` with no `source` line — the
#      vendored copy — and fail if there is none, so a step that would audit
#      nothing cannot pass;
#   2. per version, write a one-package lockfile naming the crates.io release
#      that version was vendored from (its registry `source`, and the checksum
#      `vendor/fred-<version>.SOURCE.md` records — `check-vendored-fred.sh`
#      verifies that checksum against crates.io);
#   3. run `cargo audit --deny warnings` on it, failing on any advisory against
#      that release, informational ones (unmaintained, unsound) included, and
#      on a yank. The yank check needs the crates.io index: cargo-audit
#      downgrades an index it cannot update to a warning and skips the check,
#      exactly as in `supply-chain:audit`, so only the advisory check is
#      guaranteed on every run;
#   4. tell a finding against fred apart from an advisory against cargo-audit
#      itself and from an audit that could not run (an advisory-db fetch
#      failure, say), which also fail, and say which in the failure.
#
# The one-package lockfile audits fred alone, so the `supply-chain:audit` job's
# `--ignore` list (RUSTSEC-2026-0249, smartstring) does not apply here and is
# deliberately not repeated: an ignore added for another crate cannot silence a
# fred advisory. Silencing a fred advisory means passing `--ignore <id>` to this
# script, a reviewed edit to the CI job.
#
# Every argument is passed through to `cargo audit` (e.g. `--db <dir>
# --no-fetch` to run against a local advisory-db copy). Needs `cargo-audit` on
# PATH and, without `--no-fetch`, network access to the advisory-db and the
# crates.io index. The CI job (`fred:vendor-drift-check`) runs it after the
# drift check; locally: `sh scripts/audit-vendored-fred.sh`.

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

CRATE=fred
REGISTRY_SOURCE="registry+https://github.com/rust-lang/crates.io-index"

fail() {
  echo ""
  echo "=========================================="
  echo "ERROR: vendored $CRATE advisory check failed."
  echo ""
  echo "$1"
  echo "=========================================="
  exit 1
}

cargo audit --version >/dev/null 2>&1 \
  || fail "cargo-audit is not installed (cargo binstall --no-confirm --locked cargo-audit)."

# Versions of every `$CRATE` package block in Cargo.lock that has no `source`
# line, i.e. one built from a path rather than a registry.
VERSIONS=$(awk -v crate="\"$CRATE\"" '
  function flush() {
    if (name == crate && !has_source) { gsub(/"/, "", version); print version }
  }
  /^\[\[package\]\]$/ { flush(); name = ""; version = ""; has_source = 0; next }
  /^name = /    { name = $3; next }
  /^version = / { version = $3; next }
  /^source = /  { has_source = 1; next }
  END { flush() }
' Cargo.lock)

[ -n "$VERSIONS" ] || fail "Cargo.lock has no source-less $CRATE package, so there is no vendored $CRATE to audit.
If $CRATE is no longer vendored, remove this step from the fred:vendor-drift-check job;
the supply-chain:audit job covers a registry $CRATE."

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

for VERSION in $VERSIONS; do
  RECORD="vendor/$CRATE-$VERSION.SOURCE.md"
  [ -f "$RECORD" ] || fail "Cargo.lock locks a vendored $CRATE $VERSION, but $RECORD is missing."
  CHECKSUM=$(grep -E '\*\*Crate SHA256:\*\*' "$RECORD" | grep -oE '[0-9a-f]{64}' | head -n1 || true)
  [ -n "$CHECKSUM" ] || fail "Could not parse the crate checksum from $RECORD (a line containing '**Crate SHA256:**' and a 64-hex-digit digest)."

  LOCK="$WORK/$CRATE-$VERSION/Cargo.lock"
  mkdir -p "$(dirname "$LOCK")"
  {
    echo 'version = 4'
    echo ''
    echo '[[package]]'
    echo "name = \"$CRATE\""
    echo "version = \"$VERSION\""
    echo "source = \"$REGISTRY_SOURCE\""
    echo "checksum = \"$CHECKSUM\""
  } > "$LOCK"

  echo "Auditing the vendored $CRATE $VERSION as the crates.io release it was vendored from"
  # cargo-audit (0.22.2) fails in three ways, and this script treats any
  # non-zero exit as a failure and reads the report to say which:
  #   - a finding against fred: exit 1, with an `error: <n>
  #     vulnerabilit(y|ies) found!` / `error: <n> denied warning(s) found!`
  #     summary (`presenter.rs`);
  #   - a RustSec advisory against cargo-audit's own release: under
  #     `--deny warnings` also exit 1, with `This copy of cargo-audit has known
  #     advisories!` and no fred summary;
  #   - a run that could not complete: exit 1 on an advisory-db fetch or load
  #     error (`auditor.rs`), exit 2 on a lockfile it cannot locate or load
  #     (`commands/audit.rs`).
  # A fred summary wins if both it and the self-advisory line appear. The
  # wording is that of the cargo-audit version CI pins (`CARGO_AUDIT_VERSION`
  # in .gitlab-ci.yml); a bump must re-check it.
  # `--color never` comes first because cargo-audit honours `CARGO_TERM_COLOR`
  # even when not writing to a terminal, and colour codes around `error:`
  # would hide the summary line from the grep; a later `--color` in "$@" still
  # overrides it.
  REPORT="$WORK/$CRATE-$VERSION/report.txt"
  if cargo audit --file "$LOCK" --deny warnings --color never "$@" > "$REPORT" 2>&1; then
    cat "$REPORT"
    continue
  fi
  cat "$REPORT"
  if grep -Eq '^error: [0-9]+ (vulnerabilit(y|ies)|denied warnings?) found' "$REPORT"; then
    fail "RustSec reports an advisory (or a yank) against $CRATE $VERSION, the release vendored at
vendor/$CRATE-$VERSION (report above). The supply-chain:audit job cannot see it,
because the [patch.crates-io] path entry has no registry source. Re-vendor a
release that fixes it (procedure in $RECORD), or, if the advisory does not
apply to this service, pass --ignore <id> to this script in the CI job with
the reason."
  fi
  if grep -q 'This copy of cargo-audit has known advisories!' "$REPORT"; then
    fail "cargo audit failed for $CRATE $VERSION because RustSec has an advisory against the
cargo-audit release itself (report above); no advisory summary against $CRATE
was found, but under --deny warnings the self-advisory alone fails the run.
Upgrade cargo-audit to a release without the advisory, re-check the report
wording this script greps, and re-run."
  fi
  fail "cargo audit failed for $CRATE $VERSION, but no advisory summary was found in the
report above, so the vendored $CRATE should be treated as unaudited. This is
typically a run that could not complete (the RustSec advisory-db could not be
fetched or loaded). Read the report above to confirm the cause, fix it, and
re-run."
done

echo "OK: no RustSec advisory against the vendored $CRATE ($(echo $VERSIONS))."
