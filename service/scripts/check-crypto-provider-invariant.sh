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

# Guardrail: enforce that the dependency graph carries exactly ONE rustls crypto
# provider — `ring` — and never `aws-lc-rs`. This is a cross-Epic invariant: the
# service reaches rustls' provider auto-selection on its startup path, and a
# second provider in the graph makes that selection ambiguous and fatal.
#
# Why it exists: Epic 4 (t1.tr5.we) adds an AWS SigV4 / aws-sdk dependency, which
# by default pulls `aws-lc-rs` — a SECOND rustls crypto provider. With both `ring`
# and `aws-lc-rs` compiled in, `rustls::crypto::CryptoProvider::from_crate_features()`
# returns `None` (rustls-0.23.38/src/crypto/mod.rs:265-286) and
# `ClientConfig::builder()` panics (rustls-0.23.38/src/crypto/mod.rs:248-253) with
# "Could not automatically determine the process-level CryptoProvider from Rustls
# crate features." fred reaches that builder via `TlsConnector::default_rustls()`
# (fred-10.1.0/src/protocol/tls.rs:159-179), invoked from `FredConfig::from_url`
# (src/storage/client.rs:411) — i.e. on the service's own startup path. A
# months-later regression that reintroduces aws-lc-rs must fail loudly in CI for
# whoever introduces it, not silently crash a deployed pod. This Task
# (t2.tr5.t4.oq) pins the invariant; it is already satisfied today.
#
# It is written in POSIX sh (busybox `ash`, not bash) so it runs unmodified on the
# `alpine:3.20` CI image, which ships no bash. Runnable locally with a single
# command: `sh scripts/check-crypto-provider-invariant.sh`.
#
# What counts as a violation (any one fails the check):
#   (a) Cargo.lock carries an aws-lc-rs / aws-lc-sys / aws-lc-fips-sys package.
#   (b) Cargo.lock has lost the `ring` package entirely.
#   (c) Any `rustls` package's dependency list lacks `ring` or gains `aws-lc-rs`.
#   (d) Cargo.toml's TLS feature routing drifted: `enable-rustls-ring` is no longer
#       the sole TLS feature, or it moved onto fred's inline `features = [...]`, or
#       the `enable-rustls` (aws-lc-rs-pulling) / `enable-native-tls` variants
#       appeared. Moving the feature inline is the worst case: `cfg!(feature =
#       "valkey-tls")` in live-edit then reads permanently `false`, Task 2's SR-1.4
#       scheme guard inverts, and the shipped binary rejects EVERY `rediss://` URL.
#
# How to fix a flagged case: do NOT add `enable-rustls`, `enable-native-tls`, or any
# aws-lc-rs-pulling feature. Route fred's TLS strictly through the
# `valkey-tls = ["fred/enable-rustls-ring"]` alias in Cargo.toml (see the block
# above that feature in Cargo.toml). If Epic 4's aws-sdk edge is the culprit, pin
# its crypto backend to `ring` (disable its default `aws-lc-rs` feature) so the
# single-provider invariant holds.

set -eu

# Run from the repo root regardless of invocation cwd. `$0` (not ${BASH_SOURCE},
# which busybox sh lacks) locates this script's directory.
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

LOCK="Cargo.lock"
TOML="Cargo.toml"

fail() {
  echo ""
  echo "=========================================="
  echo "ERROR: single-rustls-crypto-provider invariant violated."
  echo ""
  echo "$1"
  echo ""
  echo "The graph must carry exactly one rustls crypto provider — 'ring' — and"
  echo "never 'aws-lc-rs'. Two providers make"
  echo "rustls::crypto::CryptoProvider::from_crate_features() return None, which"
  echo "panics ClientConfig::builder() on the service's Valkey-TLS startup path"
  echo "(fred TlsConnector::default_rustls -> FredConfig::from_url). The AWS"
  echo "crates behind IAM backing-store auth ('aws-config',"
  echo "'aws-smithy-http-client') default to aws-lc-rs; Cargo.toml pins them to"
  echo "the ring backend. See the header of this script and the valkey-tls"
  echo "feature comment in Cargo.toml for the full rationale and the fix."
  echo "=========================================="
  exit 1
}

[ -f "$LOCK" ] || fail "Cargo.lock not found at $REPO_ROOT/$LOCK."
[ -f "$TOML" ] || fail "Cargo.toml not found at $REPO_ROOT/$TOML."

# ── (a) No aws-lc-* package stanza in Cargo.lock ────────────────────────────────
if grep -qE '^name = "aws-lc-(rs|sys|fips-sys)"$' "$LOCK"; then
  offending=$(grep -E '^name = "aws-lc-(rs|sys|fips-sys)"$' "$LOCK")
  fail "Cargo.lock contains a banned aws-lc crypto-provider package: $offending"
fi

# ── (b) The `ring` package stanza is still present ──────────────────────────────
# Guards against the whole ring path being dropped — which (a) alone would miss.
if ! grep -qE '^name = "ring"$' "$LOCK"; then
  fail "Cargo.lock no longer contains a 'ring' package. The ring rustls crypto provider was dropped from the graph."
fi

# ── (c) Every `rustls` package depends on ring and not aws-lc-rs ─────────────────
# Iterate ALL rustls stanzas: deny.toml sets multiple-versions = "warn", so more
# than one rustls major can legitimately coexist. Dependency entries may carry a
# version suffix ("ring 0.17.14") once duplicates exist, so match `"ring` followed
# by a space or the closing quote. This mirrors from_crate_features() directly.
awk_out=$(awk '
  function finalize() {
    if (was_rustls) {
      count++
      if (!has_ring || has_awslc) {
        bad = 1
        printf "  rustls stanza ending near line %d: has_ring=%d has_aws_lc_rs=%d\n", NR, has_ring, has_awslc
      }
    }
    was_rustls = 0; in_deps = 0; has_ring = 0; has_awslc = 0
  }
  /^\[\[package\]\]/ { finalize() }
  /^name = "rustls"$/ { was_rustls = 1 }
  /^dependencies = \[/ { if (was_rustls) in_deps = 1; next }
  in_deps && /^\]/ { in_deps = 0 }
  in_deps {
    if ($0 ~ /"ring[ "]/) has_ring = 1
    if ($0 ~ /"aws-lc-rs[ "]/) has_awslc = 1
  }
  END {
    finalize()
    if (count == 0) { print "NORUSTLS"; exit 0 }
    if (bad) { exit 3 }
    printf "OK %d\n", count
  }
' "$LOCK") || {
  fail "One or more rustls package stanzas fail the crypto-provider dependency check:
$awk_out"
}

case "$awk_out" in
  NORUSTLS)
    fail "Cargo.lock contains no 'rustls' package stanza — the TLS stack was removed or renamed unexpectedly."
    ;;
  *)
    : # "OK <n>" — the awk pass already validated every stanza.
    ;;
esac

# ── (d) Cargo.toml TLS feature routing is intact ────────────────────────────────
# Strip comments so the prose in the valkey-tls comment block (which legitimately
# mentions enable-rustls / enable-native-tls / enable-rustls-ring) does not
# false-positive. No Cargo.toml string value in this repo contains '#'.
TOML_CODE=$(sed 's/#.*//' "$TOML")

# (d1) `enable-rustls-ring` appears exactly once, and that occurrence is the
#      valkey-tls feature definition.
ring_feat_count=$(printf '%s\n' "$TOML_CODE" | grep -c 'enable-rustls-ring' || true)
if [ "$ring_feat_count" != "1" ]; then
  fail "enable-rustls-ring must appear exactly once in Cargo.toml (outside comments); found $ring_feat_count. It is the sole route by which fred receives a TLS feature."
fi
if ! printf '%s\n' "$TOML_CODE" | grep -Eq '^[[:space:]]*valkey-tls[[:space:]]*=.*enable-rustls-ring'; then
  fail "The sole enable-rustls-ring occurrence must live in the 'valkey-tls' feature definition (valkey-tls = [\"fred/enable-rustls-ring\"]), not elsewhere in Cargo.toml."
fi

# (d2) The fred dependency's inline `features = [...]` list carries NO enable-* TLS
#      feature. Moving the feature inline breaks cfg!(feature = "valkey-tls") in
#      live-edit and inverts the SR-1.4 scheme guard — the rediss://-rejects-all
#      production-outage regression.
fred_dep_line=$(printf '%s\n' "$TOML_CODE" | grep -E '^fred[[:space:]]*=' || true)
if [ -z "$fred_dep_line" ]; then
  fail "Could not locate the fred dependency line (^fred = ...) in Cargo.toml — its shape changed; retune this check."
fi
case "$fred_dep_line" in
  *enable-*)
    fail "The fred dependency's inline features list carries an 'enable-*' TLS feature:
    $fred_dep_line
Move it back to the valkey-tls feature alias. Inline, cfg!(feature = \"valkey-tls\") reads false in live-edit and the binary rejects every rediss:// URL."
    ;;
  *)
    : # no inline enable-* feature — the routing is intact.
    ;;
esac

# (d3) The aws-lc-rs-pulling `enable-rustls` variant and the stack-swapping
#      `enable-native-tls` appear nowhere. The `enable-rustls"` anchor (trailing
#      quote) survives the enable-rustls-ring superstring hazard: enable-rustls-ring
#      is followed by '-ring', never '"'. fred silently PREFERS native-tls when both
#      stacks compile in (fred-10.1.0/src/utils.rs:679-690), so enable-native-tls
#      would swap the whole TLS stack with no lockfile-visible signal — not caught
#      by (a).
if printf '%s\n' "$TOML_CODE" | grep -q 'enable-rustls"'; then
  fail "Cargo.toml uses the 'enable-rustls' fred feature, which pulls tokio-rustls/aws_lc_rs — a second crypto provider. Use enable-rustls-ring."
fi
if printf '%s\n' "$TOML_CODE" | grep -q 'enable-native-tls'; then
  fail "Cargo.toml uses the 'enable-native-tls' fred feature. fred silently prefers native-tls when both stacks are compiled in, swapping the TLS stack with no lockfile-visible signal. Remove it."
fi

echo "OK: single rustls crypto provider confirmed — 'ring' is the sole provider."
echo "  - Cargo.lock: no aws-lc-rs / aws-lc-sys / aws-lc-fips-sys; 'ring' present; all rustls stanzas depend on ring, none on aws-lc-rs."
echo "  - Cargo.toml: fred TLS routed only via valkey-tls = [\"fred/enable-rustls-ring\"]; no inline enable-* / enable-rustls / enable-native-tls."
