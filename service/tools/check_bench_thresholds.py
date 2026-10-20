#!/usr/bin/env python3
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

"""Absolute-ceiling checker for the criterion benchmarks under ``benches/``.

Input: one or more files of ``cargo bench -- --output-format bencher``
output, i.e. libtest-``bencher`` result lines
(``test <id> ... bench: <n> ns/iter (+/- <d>)``); any other line in the
stream is skipped. A later measurement of the same id wins.

Contract: an id listed in :data:`THRESHOLDS_NS` is *gated* and fails the
run if it measures above its ceiling **or is absent from the input** (no
measurement means no coverage — a broken CI script, a renamed id, or a
bench that panicked before reporting). Every other id is reported
informationally and can never fail the run.

Usage::

    python3 tools/check_bench_thresholds.py bench-enqueue.txt bench-store-values.txt

Exit codes:

* ``0`` — every gated id measured at or under its ceiling.
* ``1`` — a gated id exceeded its ceiling or was missing from the input,
  or no parsable benchmark line was found at all.
* ``2`` — usage error (no input files given, or a file could not be read).

Why the gate is an absolute ceiling rather than ``criterion --baseline``,
and why exactly one id is gated:
``docs-internal/test-writing-guide.md`` § Performance-Sensitive Tests.
"""

import re
import sys
from typing import NamedTuple

# ── Gated thresholds ─────────────────────────────────────────────────────────

#: Benchmark id → absolute ceiling in nanoseconds per iteration.
#:
#: ``ext_mirror_enqueue/mirror_drained/4096`` is the SR-3.1 ext-mirror
#: enqueue cost of a 4 KiB value — the byte charge (the oversized check, a
#: non-waiting acquire on the budget semaphore, the charge slot's fill and one
#: ``checkpoint_ext_mirror_bytes`` gauge add), one ``String`` clone of the hex
#: hash, and one non-blocking ``try_send`` onto a bounded channel with live
#: drain workers. The value's copy is built outside the timed region, as
#: production makes it whether or not the mirror is wired, so no payload copy
#: is measured. It is the single quantity in the ext-mirror subsystem that a
#: benchmark can resolve, and the arm that reflects production steady state.
#:
#: The ceiling was set with roughly 40x headroom over the median measured
#: when this bench was added (b.9ft): ~1.2 microseconds per iteration in a
#: release build on an unloaded developer machine. The "~40x" in the messages
#: below is that design floor. The current bench, on 4 pinned CPUs of the dev
#: container, measures 0.13 microseconds without the byte charge and 0.32 with
#: it (b.mhq, 2026-10-15), about 157x headroom. That is
#: intentionally enormous: the number has to
#: survive a shared, CPU-throttled CI runner co-tenanted with other jobs
#: without ever flaking, while still catching the regressions that matter,
#: all of which are one to three orders of magnitude larger than the
#: effect itself. Tighten it only alongside evidence that CI-side variance
#: is smaller than assumed.
#:
#: The sibling ``ext_mirror_store_values`` benchmark is deliberately absent
#: from this table. It measures a composed call dominated by the Valkey
#: round trip, where the ext-mirror effect is a fraction of a percent of
#: the signal; any ceiling on it would gate network latency instead of
#: this repository's code. Its ids are reported informationally.
THRESHOLDS_NS = {
    "ext_mirror_enqueue/mirror_drained/4096": 50_000,
}

# ── Parsing ──────────────────────────────────────────────────────────────────

#: A libtest-``bencher``-format result line as criterion emits it under
#: ``--output-format bencher``, e.g.::
#:
#:     test ext_mirror_enqueue/mirror_drained/4096 ... bench:       1,207 ns/iter (+/- 44)
#:
#: Both numbers are thousands-separated by criterion, hence the ``[\d,]+``.
_BENCH_LINE = re.compile(
    r"^\s*test\s+(?P<name>\S+)\s+\.\.\.\s+bench:\s+"
    r"(?P<ns>[\d,]+)\s+ns/iter\s+\(\+/-\s+(?P<deviation>[\d,]+)\)\s*$"
)


class BenchResult(NamedTuple):
    """One parsed benchmark measurement."""

    name: str
    ns_per_iter: int
    deviation_ns: int


class Violation(NamedTuple):
    """A gated id that measured over its ceiling."""

    name: str
    ns_per_iter: int
    ceiling_ns: int

    @property
    def overage_ratio(self) -> float:
        return self.ns_per_iter / self.ceiling_ns


def parse_bench_line(line):
    """Parse one bencher-format line into a :class:`BenchResult`.

    Returns ``None`` for any line that is not a result line — criterion
    interleaves progress chatter and blank lines into the same stream, and
    callers should skip those rather than treat them as errors.
    """
    match = _BENCH_LINE.match(line)
    if match is None:
        return None
    return BenchResult(
        name=match.group("name"),
        ns_per_iter=int(match.group("ns").replace(",", "")),
        deviation_ns=int(match.group("deviation").replace(",", "")),
    )


def parse_bench_output(text):
    """Parse a whole bencher-format stream into a list of results.

    Later measurements of the same id win — re-running a benchmark in the
    same stream should report the newer number.
    """
    results = {}
    for line in text.splitlines():
        parsed = parse_bench_line(line)
        if parsed is not None:
            results[parsed.name] = parsed
    return list(results.values())


# ── Checking ─────────────────────────────────────────────────────────────────


def check_thresholds(results, thresholds=None):
    """Compare parsed results against the gated ceilings.

    Returns ``(violations, missing)``:

    * ``violations`` — :class:`Violation` entries for gated ids that
      measured above their ceiling.
    * ``missing`` — gated ids that did not appear in ``results`` at all.

    Both lists are sorted by id so the output is stable across runs.
    """
    if thresholds is None:
        thresholds = THRESHOLDS_NS
    by_name = {result.name: result for result in results}

    violations = []
    missing = []
    for name in sorted(thresholds):
        ceiling = thresholds[name]
        result = by_name.get(name)
        if result is None:
            missing.append(name)
        elif result.ns_per_iter > ceiling:
            violations.append(
                Violation(name=name, ns_per_iter=result.ns_per_iter, ceiling_ns=ceiling)
            )
    return violations, missing


def format_report(results, thresholds=None):
    """Render the informational table of every parsed measurement.

    Gated ids are annotated with their ceiling and headroom; everything
    else is marked ``report-only`` so a reader never has to cross-reference
    this file to learn which numbers can fail a pipeline.
    """
    if thresholds is None:
        thresholds = THRESHOLDS_NS
    lines = ["Benchmark results ({} measurement(s)):".format(len(results))]
    for result in sorted(results, key=lambda r: r.name):
        ceiling = thresholds.get(result.name)
        if ceiling is None:
            annotation = "report-only"
        elif result.ns_per_iter > ceiling:
            annotation = "GATED ceiling={:,} ns - EXCEEDED by {:.1f}x".format(
                ceiling, result.ns_per_iter / ceiling
            )
        else:
            headroom = ceiling / result.ns_per_iter if result.ns_per_iter else float("inf")
            annotation = "GATED ceiling={:,} ns ({:.1f}x headroom)".format(ceiling, headroom)
        lines.append(
            "  {name:<48} {ns:>12,} ns/iter (+/- {dev:,})  [{annotation}]".format(
                name=result.name,
                ns=result.ns_per_iter,
                dev=result.deviation_ns,
                annotation=annotation,
            )
        )
    return "\n".join(lines)


def format_violation(violation):
    """Render the operator-facing failure text for one violation."""
    return (
        "FAIL {name}: {ns:,} ns/iter exceeds the absolute ceiling of "
        "{ceiling:,} ns/iter ({ratio:.1f}x over).\n"
        "  This ceiling carries roughly 40x headroom over the expected cost, so a\n"
        "  breach is a real regression, not runner noise. The measured effect is the\n"
        "  ext-mirror enqueue: the byte charge (the oversized check, a non-waiting\n"
        "  semaphore acquire, the charge slot's fill and one\n"
        "  checkpoint_ext_mirror_bytes gauge add), a String clone of the hex hash,\n"
        "  and a non-blocking try_send. The value's copy is built outside the timed\n"
        "  region, so no payload copy is measured. Likely causes, in order of how\n"
        "  often they turn up:\n"
        "    1. try_send became blocking (a send().await on the producer side).\n"
        "    2. New synchronous work landed on the hot path (an extra clone, a\n"
        "       serialization step, a log statement promoted to an always-on level).\n"
        "    3. A synchronous adapter call moved onto the producer side.\n"
        "    4. The item itself got more expensive to build (a payload copy moved\n"
        "       into the send, a hash re-computation, a second allocation), or the\n"
        "       enqueue itself got slower (a wider ExtMirrorItem, a waiting budget\n"
        "       acquire, metric or tracing work beyond the one gauge add on the Ok\n"
        "       branch).\n"
        "  What this is NOT: a stalled drain. The drop-on-full branch and a\n"
        "  successful enqueue cost the same order of magnitude - both take the byte\n"
        "  charge, neither copies or frees the payload in the timed region, and the\n"
        "  drop branch adds its structured WARN - so\n"
        "  no amount of stalling reaches a ceiling set ~40x above the expected cost.\n"
        "  A stall's real damage is silent rather than loud: it\n"
        "  turns mirror_drained into a second mirror_saturated and reports a\n"
        "  confident number for the wrong code path. The bench\n"
        "  forecloses that rather than leaving it to this ceiling - the drained arm\n"
        "  enqueues in chunks that fit the channel and the byte budget, waits\n"
        "  (untimed) between chunks for the workers to return every charge, and\n"
        "  reads live_edit_checkpoint_ext_mirror_dropped_total around its own\n"
        "  measurement, and panics if anything was dropped. So if you are reading\n"
        "  this text, the drain was fine and the cost really moved.\n"
        "  Reproduce locally with:\n"
        "    cargo bench -p live-edit --bench ext_mirror_enqueue"
    ).format(
        name=violation.name,
        ns=violation.ns_per_iter,
        ceiling=violation.ceiling_ns,
        ratio=violation.overage_ratio,
    )


def format_missing(name):
    """Render the operator-facing failure text for a missing gated id."""
    return (
        "FAIL {name}: gated benchmark id not present in the supplied bencher output.\n"
        "  No measurement means no coverage. Check that the benchmark still emits this\n"
        "  exact id (group/function/parameter), that the bench target actually ran, and\n"
        "  that its output was captured into the file passed to this script."
    ).format(name=name)


# ── Entry point ──────────────────────────────────────────────────────────────


def main(argv):
    """Thin CLI wrapper: read the files, check, print, pick an exit code."""
    paths = argv[1:]
    if not paths:
        print(
            "usage: check_bench_thresholds.py <bencher-output-file> [more-files...]",
            file=sys.stderr,
        )
        return 2

    chunks = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                chunks.append(handle.read())
        except OSError as exc:
            print("error: cannot read {}: {}".format(path, exc), file=sys.stderr)
            return 2

    results = parse_bench_output("\n".join(chunks))
    print(format_report(results))

    if not results:
        print(
            "\nFAIL: no benchmark result lines were parsed from {}.\n"
            "  Expected libtest 'bencher' format, which criterion emits with\n"
            "  `cargo bench -- --output-format bencher`.".format(", ".join(paths)),
            file=sys.stderr,
        )
        return 1

    violations, missing = check_thresholds(results)
    for name in missing:
        print("\n" + format_missing(name), file=sys.stderr)
    for violation in violations:
        print("\n" + format_violation(violation), file=sys.stderr)

    if violations or missing:
        return 1
    print("\nOK: every gated benchmark id is within its absolute ceiling.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
