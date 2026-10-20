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

"""Unit tests for `tools/check_bench_thresholds.py`.

Scope: the whole script. It is pure Python with no dependency beyond the
stdlib — the only I/O is `main()` reading the files named on argv, which
these tests drive through `tempfile`. There is no live service, no Valkey
and no cargo invocation anywhere in here, so no import stubbing is needed.

What the script is for (issue b.9ft): `perf:ext-mirror-bench` runs the
criterion benchmarks under `benches/` with
`cargo bench -- --output-format bencher`, captures the output, and pipes
the files to this checker. The checker fails the job when a *gated*
benchmark id exceeds a hard nanoseconds-per-iteration ceiling. Three
properties are load-bearing and each has direct coverage below:

* A gated id **missing** from the input fails, rather than passing
  vacuously. Silence means the benchmark did not run.
* An **ungated** id can never fail the job, no matter how slow — the
  `ext_mirror_store_values` ids are dominated by a Valkey round trip and
  gating them would gate network latency.
* Exit codes separate "the benchmark regressed" (1) from "the checker was
  invoked wrongly" (2), so a broken CI script is not mistaken for a
  perf regression.

One class here reaches outside the script, at text level, because the
checker's gated id is a join key spanning files with nothing in any
compiler enforcing it: `ShippedThresholdTableTests` reads
`benches/ext_mirror_enqueue.rs` and checks that the group name, the arm
name and the payload size the bench actually emits still compose the id in
`THRESHOLDS_NS`. That failure mode is otherwise silent until long after the
merge, which is the same shape of bug as a benchmark that stopped
reporting. (That CI runs this module at all is checked by an internal
CI-wiring test that reads the CI config.)

These tests are stdlib `unittest` (no pytest), matching the
`supply-chain:python-tests` CI job that runs them. Run this module from the
repo root with:

    python3 -m unittest tools.tests.test_check_bench_thresholds

or directly:

    python3 tools/tests/test_check_bench_thresholds.py
"""

from __future__ import annotations

import contextlib
import importlib
import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path


# ── Module loader ─────────────────────────────────────────────────────────


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"


def _load_checker_module():
    # Make `tools/` importable so `import check_bench_thresholds` works
    # without requiring an installed package.
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    sys.modules.pop("check_bench_thresholds", None)
    return importlib.import_module("check_bench_thresholds")


# The script has no import-time side effects, so one load is enough.
CHECKER = _load_checker_module()


# ── Shared fixtures ───────────────────────────────────────────────────────


#: The one id gated by the shipped `THRESHOLDS_NS` table. Named once here
#: so a rename in the script surfaces as a single failing assertion in
#: `ShippedThresholdTableTests` rather than as a scatter of unrelated
#: `main()` failures.
GATED_ID = "ext_mirror_enqueue/mirror_drained/4096"

#: A canonical bencher line as criterion emits it: the id, thousands
#: separators on both numbers, the `(+/- deviation)` tail.
CANONICAL_LINE = "test {} ... bench:       1,207 ns/iter (+/- 44)".format(GATED_ID)

#: Source of the gated benchmark, cross-checked against `GATED_ID` by
#: `ShippedThresholdTableTests`.
BENCH_SOURCE_PATH = REPO_ROOT / "benches" / "ext_mirror_enqueue.rs"


# ── Rust-source extraction helpers ────────────────────────────────────────


#: `const NAME: T = expr;` at the start of a line. The type class carries
#: `;` on purpose — array types are written `[usize; 4]` — so the type is
#: matched lazily up to the `=` rather than to the first semicolon.
_RUST_CONST_RE = re.compile(
    r"^\s*const\s+(?P<name>\w+)\s*:\s*[\w:<>\[\]; ]+?\s*=\s*(?P<expr>[^;]+);",
    re.MULTILINE,
)

#: Optional Rust integer suffix on a literal (`4096usize`, `1u64`).
_RUST_INT_SUFFIX_RE = re.compile(r"(?:usize|isize|u8|u16|u32|u64|u128|i8|i16|i32|i64|i128)$")


def rust_const_expr(source, name):
    """Return the right-hand side of `const <name>: T = ...;`, or `None`.

    Whitespace-trimmed, semicolon excluded. `None` means the constant is
    not declared at all — a distinct failure from "declared but does not
    match", and the callers keep them distinct.
    """
    for match in _RUST_CONST_RE.finditer(source):
        if match.group("name") == name:
            return match.group("expr").strip()
    return None


def eval_rust_int_expr(expr):
    """Evaluate a product of Rust integer literals, or return `None`.

    Accepts exactly what the bench's size constants are written as —
    `4096`, `4 * 1024`, `1024 * 1024`, with optional `_` separators and
    type suffixes. Anything outside that grammar (hex, shifts, a
    referenced constant) yields `None` so the caller fails loudly rather
    than silently guessing a size.

    Deliberately not `eval()`. The input is a checked-in source file, but
    a test that executes text pulled out of another file is a trap worth
    not setting, and the narrow grammar costs four lines.
    """
    value = 1
    for factor in expr.split("*"):
        token = _RUST_INT_SUFFIX_RE.sub("", factor.strip().replace("_", ""))
        if not token.isdigit():
            return None
        value *= int(token)
    return value


def _result(name, ns, deviation=0):
    return CHECKER.BenchResult(name=name, ns_per_iter=ns, deviation_ns=deviation)


def _violation(ns=150_000, ceiling=50_000):
    """A breach of the gated id, 3x over by default."""
    return CHECKER.Violation(name=GATED_ID, ns_per_iter=ns, ceiling_ns=ceiling)


def _run_main(*paths):
    """Invoke `main()` with a synthesised argv, capturing both streams.

    Returns `(exit_code, stdout, stderr)`. argv[0] is the script name the
    real CLI passes through, so the slicing in `main` is exercised as
    shipped.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    argv = ["check_bench_thresholds.py", *paths]
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = CHECKER.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class _TempInputMixin:
    """Writes bencher-output files into a per-test temporary directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmpdir = Path(self._tmp.name)

    def write_input(self, text, name="bench.txt"):
        path = self.tmpdir / name
        path.write_text(text, encoding="utf-8")
        return str(path)


# ── parse_bench_line ──────────────────────────────────────────────────────


class ParseBenchLineTests(unittest.TestCase):
    """The bencher-format line parser.

    This is the only place the script touches criterion's output format,
    so it carries the whole risk of a format drift going unnoticed.
    """

    def test_canonical_line_parses_all_three_fields(self):
        result = CHECKER.parse_bench_line(CANONICAL_LINE)
        self.assertEqual(
            result, CHECKER.BenchResult(name=GATED_ID, ns_per_iter=1207, deviation_ns=44)
        )

    def test_group_function_parameter_id_survives_verbatim(self):
        """The id is the checker's join key against `THRESHOLDS_NS`, so the
        `group/function/parameter` triple must come through unmodified —
        slashes and all."""
        result = CHECKER.parse_bench_line(
            "test ext_mirror_enqueue/mirror_drained/4096 ... bench: 1 ns/iter (+/- 0)"
        )
        self.assertEqual(result.name, "ext_mirror_enqueue/mirror_drained/4096")

    def test_numbers_without_thousands_separators_parse(self):
        """Criterion groups digits, but the libtest format it imitates does
        not require it. Accepting both keeps the checker usable against raw
        `cargo +nightly bench` output too."""
        result = CHECKER.parse_bench_line("test a ... bench: 1207 ns/iter (+/- 44)")
        self.assertEqual((result.ns_per_iter, result.deviation_ns), (1207, 44))

    def test_large_grouped_numbers_parse_to_full_magnitude(self):
        """A regression well past the ceiling produces a seven-digit,
        multiply-grouped number — the exact case the checker exists to
        catch, so it must not be truncated at the first separator."""
        result = CHECKER.parse_bench_line(
            "test a ... bench: 1,234,567 ns/iter (+/- 89,012)"
        )
        self.assertEqual((result.ns_per_iter, result.deviation_ns), (1234567, 89012))

    def test_leading_and_trailing_whitespace_tolerated(self):
        result = CHECKER.parse_bench_line("   " + CANONICAL_LINE + "   \n")
        self.assertIsNotNone(result)
        self.assertEqual(result.ns_per_iter, 1207)

    def test_empty_line_is_not_an_error(self):
        """Criterion interleaves blank lines into the same stream; they are
        skipped, not rejected."""
        self.assertIsNone(CHECKER.parse_bench_line(""))

    def test_criterion_progress_chatter_is_skipped(self):
        self.assertIsNone(
            CHECKER.parse_bench_line("Benchmarking ext_mirror_enqueue: Warming up for 3.0000 s")
        )

    def test_non_integer_measurement_is_rejected(self):
        """A fractional ns/iter is not the bencher format. Rejecting it
        rather than truncating keeps a format change loud."""
        self.assertIsNone(CHECKER.parse_bench_line("test a ... bench: 12.5 ns/iter (+/- 4)"))

    def test_missing_deviation_tail_is_rejected(self):
        self.assertIsNone(CHECKER.parse_bench_line("test a ... bench: 1207 ns/iter"))

    def test_trailing_garbage_after_deviation_is_rejected(self):
        """Anchored at both ends: a line that merely *contains* a
        measurement is not a measurement line."""
        self.assertIsNone(
            CHECKER.parse_bench_line("test a ... bench: 1 ns/iter (+/- 0) and then some")
        )

    def test_id_containing_whitespace_is_rejected(self):
        """Ids are `\\S+`. A whitespace-bearing id would silently truncate
        the join key, so it is refused outright."""
        self.assertIsNone(
            CHECKER.parse_bench_line("test two words ... bench: 1 ns/iter (+/- 0)")
        )


# ── parse_bench_output ────────────────────────────────────────────────────


class ParseBenchOutputTests(unittest.TestCase):
    """Whole-stream parsing: chatter filtering and duplicate resolution."""

    def test_result_lines_extracted_from_interleaved_chatter(self):
        text = "\n".join(
            [
                "Benchmarking ext_mirror_enqueue/mirror_drained/4096",
                "Warming up for 3.0000 s",
                "",
                CANONICAL_LINE,
                "Benchmarking ext_mirror_store_values/no_mirror/16",
                "test ext_mirror_store_values/no_mirror/16 ... bench: 91,000 ns/iter (+/- 12,000)",
                "",
            ]
        )
        names = sorted(r.name for r in CHECKER.parse_bench_output(text))
        self.assertEqual(
            names,
            [GATED_ID, "ext_mirror_store_values/no_mirror/16"],
        )

    def test_later_measurement_of_same_id_wins(self):
        """Re-running a benchmark inside one stream must report the newer
        number — the checker gates on the latest measurement, not the
        first one it happened to see."""
        text = "\n".join(
            [
                "test a ... bench: 100 ns/iter (+/- 1)",
                "test a ... bench: 200 ns/iter (+/- 2)",
            ]
        )
        results = CHECKER.parse_bench_output(text)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].ns_per_iter, 200)

    def test_empty_input_yields_no_results(self):
        self.assertEqual(CHECKER.parse_bench_output(""), [])

    def test_chatter_only_input_yields_no_results(self):
        """The "bench ran but reported nothing" shape. It must reach the
        caller as an empty list so `main()` can turn it into a failure
        rather than a silent pass."""
        self.assertEqual(
            CHECKER.parse_bench_output("Compiling live-edit v0.1.0\nFinished bench profile\n"),
            [],
        )


# ── check_thresholds ──────────────────────────────────────────────────────


class CheckThresholdsTests(unittest.TestCase):
    """Verdict logic: under / at / over the ceiling, and absent entirely."""

    THRESHOLDS = {"gated/a": 1000, "gated/b": 2000}

    def test_measurement_under_ceiling_is_clean(self):
        violations, missing = CHECKER.check_thresholds(
            [_result("gated/a", 999), _result("gated/b", 1)], self.THRESHOLDS
        )
        self.assertEqual((violations, missing), ([], []))

    def test_measurement_exactly_at_ceiling_passes(self):
        """The ceiling is inclusive. With ~40x headroom the boundary is
        never near a real measurement, but the comparison operator is
        worth pinning so a later `>=` typo is caught."""
        violations, missing = CHECKER.check_thresholds(
            [_result("gated/a", 1000), _result("gated/b", 2000)], self.THRESHOLDS
        )
        self.assertEqual((violations, missing), ([], []))

    def test_one_nanosecond_over_ceiling_violates(self):
        violations, missing = CHECKER.check_thresholds(
            [_result("gated/a", 1001), _result("gated/b", 1)], self.THRESHOLDS
        )
        self.assertEqual(missing, [])
        self.assertEqual(
            violations,
            [CHECKER.Violation(name="gated/a", ns_per_iter=1001, ceiling_ns=1000)],
        )

    def test_absent_gated_id_is_reported_missing_not_passing(self):
        """The core anti-vacuity property: a gated id with no measurement
        is a failure. A checker that returned clean here would turn a
        renamed benchmark or a panicking bench into invisible loss of
        coverage."""
        violations, missing = CHECKER.check_thresholds(
            [_result("gated/a", 1)], self.THRESHOLDS
        )
        self.assertEqual(violations, [])
        self.assertEqual(missing, ["gated/b"])

    def test_ungated_id_never_violates_however_slow(self):
        """`ext_mirror_store_values` is dominated by a Valkey round trip;
        gating it would gate network latency. An id absent from the table
        is informational, full stop."""
        violations, missing = CHECKER.check_thresholds(
            [_result("gated/a", 1), _result("gated/b", 1), _result("ungated/x", 10_000_000_000)],
            self.THRESHOLDS,
        )
        self.assertEqual((violations, missing), ([], []))

    def test_no_results_at_all_reports_every_gated_id_missing(self):
        violations, missing = CHECKER.check_thresholds([], self.THRESHOLDS)
        self.assertEqual(violations, [])
        self.assertEqual(missing, ["gated/a", "gated/b"])

    def test_violations_and_missing_are_sorted_by_id(self):
        """Stable output across runs: CI logs diff cleanly and the report
        does not reorder with dict iteration order."""
        thresholds = {"z/id": 10, "a/id": 10, "m/id": 10}
        violations, missing = CHECKER.check_thresholds(
            [_result("z/id", 99), _result("a/id", 99)], thresholds
        )
        self.assertEqual([v.name for v in violations], ["a/id", "z/id"])
        self.assertEqual(missing, ["m/id"])

    def test_thresholds_defaults_to_the_shipped_table(self):
        """Omitting the argument must consult `THRESHOLDS_NS` — that is the
        path `main()` takes, so a broken default would only surface in
        CI."""
        violations, missing = CHECKER.check_thresholds([])
        self.assertEqual(violations, [])
        self.assertEqual(missing, sorted(CHECKER.THRESHOLDS_NS))


class ViolationTests(unittest.TestCase):
    def test_overage_ratio_is_measured_over_ceiling(self):
        violation = CHECKER.Violation(name="a", ns_per_iter=150_000, ceiling_ns=50_000)
        self.assertAlmostEqual(violation.overage_ratio, 3.0, places=9)


class ShippedThresholdTableTests(unittest.TestCase):
    """Contract checks on `THRESHOLDS_NS` and the bench it is joined to.

    These bind CI's actual configuration, not a test fixture: the id here
    must match the criterion benchmark id emitted by
    `benches/ext_mirror_enqueue.rs`, or `perf:ext-mirror-bench` fails on
    every run with a "missing gated id".
    """

    def bench_source(self):
        """The gated benchmark's source text, or a failure naming the path."""
        self.assertTrue(
            BENCH_SOURCE_PATH.is_file(),
            "the gated benchmark's source must exist at {}".format(BENCH_SOURCE_PATH),
        )
        return BENCH_SOURCE_PATH.read_text(encoding="utf-8")

    def test_ext_mirror_enqueue_drained_arm_is_gated(self):
        self.assertIn(GATED_ID, CHECKER.THRESHOLDS_NS)

    def test_gated_id_components_still_exist_in_the_bench_source(self):
        """Cross-file guard on the two *name* dimensions of the join key.

        Criterion composes the id from three literals in
        `benches/ext_mirror_enqueue.rs`: the `benchmark_group` name, the
        `BenchmarkId::new` arm name, and the payload size. Rename any of
        them and this table stops matching — a failure that would
        otherwise only surface as a "missing gated id" in
        `perf:ext-mirror-bench`, long after the merge. The size dimension
        has its own test below; the two names are checked here.

        Text search rather than a parse: the point is a cheap tripwire on
        two string literals, and anything heavier would be a Rust parser
        living in a Python test. `assertTrue(x in source)` rather than
        `assertIn` on purpose — `assertIn` renders the whole 17 KB
        escaped Rust source into the CI log on failure, drowning the
        message that actually says what drifted.
        """
        source = self.bench_source()
        group, arm, _parameter = GATED_ID.split("/")
        self.assertTrue(
            'benchmark_group("{}")'.format(group) in source,
            "criterion group name drifted from the gated id in THRESHOLDS_NS: "
            'no benchmark_group("{}") in {}'.format(group, BENCH_SOURCE_PATH),
        )
        self.assertTrue(
            'BenchmarkId::new("{}"'.format(arm) in source,
            "criterion benchmark arm name drifted from the gated id in THRESHOLDS_NS: "
            'no BenchmarkId::new("{}", ...) in {}'.format(arm, BENCH_SOURCE_PATH),
        )

    def test_gated_payload_size_in_the_bench_matches_the_gated_id_parameter(self):
        """Cross-file guard on the *size* dimension of the join key.

        The bench pins `PAYLOAD_SIZES[1] == GATED_PAYLOAD_SIZE` at compile
        time, so the size the bench emits and the size it documents cannot
        diverge. Nothing on the Rust side links either of them to the
        `4096` in `THRESHOLDS_NS`, though: renaming *both* — dropping 4 KiB
        from the sweep and moving `GATED_PAYLOAD_SIZE` to 8 KiB — still
        compiles, and the only symptom is `perf:ext-mirror-bench` failing
        with a missing gated id after the merge. So read the constant out
        of the source and compare it to the id's third component.

        `GATED_PAYLOAD_SIZE` is written `4 * 1024`, hence the arithmetic:
        comparing the raw expression text to `"4096"` would fail on a
        no-op rewrite to `4096`, which is the wrong kind of brittle.
        """
        source = self.bench_source()
        _group, _arm, parameter = GATED_ID.split("/")

        expr = rust_const_expr(source, "GATED_PAYLOAD_SIZE")
        self.assertIsNotNone(
            expr,
            "no `const GATED_PAYLOAD_SIZE` in {} — the gated size lost its named "
            "anchor on the bench side of the boundary".format(BENCH_SOURCE_PATH),
        )
        size = eval_rust_int_expr(expr)
        self.assertIsNotNone(
            size,
            "GATED_PAYLOAD_SIZE is `{}`, which is not a product of integer "
            "literals; widen eval_rust_int_expr to cover it".format(expr),
        )
        self.assertEqual(
            size,
            int(parameter),
            "GATED_PAYLOAD_SIZE (`{}` = {}) drifted from the payload size in the "
            "gated id {!r}; criterion will emit "
            "ext_mirror_enqueue/mirror_drained/{} and the checker will report a "
            "missing gated id".format(expr, size, GATED_ID, size),
        )

    def test_the_drop_counter_named_in_the_failure_text_is_read_by_the_bench(self):
        """Cross-file guard on the one claim the failure text makes about
        another file.

        `format_violation` rules a stalled drain out as a cause and tells
        the operator why they can trust that: the drained arm reads
        `live_edit_checkpoint_ext_mirror_dropped_total` around its own
        measurement and panics if anything was dropped. Delete the
        self-check and the message becomes a false assurance pointing at
        a detector that no longer exists — worse than saying nothing, and
        invisible to every other test here.
        """
        metric = "live_edit_checkpoint_ext_mirror_dropped_total"
        self.assertIn(metric, CHECKER.format_violation(_violation()))
        self.assertTrue(
            metric in self.bench_source(),
            "the violation text tells operators the drained arm self-checks against "
            "{}, but {} never reads it — either restore the self-check or stop "
            "promising it".format(metric, BENCH_SOURCE_PATH),
        )

    def test_store_values_ids_are_deliberately_ungated(self):
        """Documented exclusion, asserted so it cannot be added by
        accident: any `ext_mirror_store_values` ceiling would gate the
        Valkey round trip that dominates that measurement."""
        gated_store_values = [
            name for name in CHECKER.THRESHOLDS_NS if name.startswith("ext_mirror_store_values")
        ]
        self.assertEqual(gated_store_values, [])

    def test_copy_ids_are_deliberately_ungated(self):
        """Documented exclusion, asserted so it cannot be added by
        accident: the `ext_mirror_copy` group in
        `benches/ext_mirror_enqueue.rs` reports the value's copy with and
        without a mirror wired and gates nothing, and the gate matches ids
        exactly, so no id of the group may be in the table. The group name
        is checked in the bench source too, so a rename leaves this test
        failing rather than vacuously passing."""
        self.assertTrue(
            'benchmark_group("ext_mirror_copy")' in self.bench_source(),
            'no benchmark_group("ext_mirror_copy") in {}'.format(BENCH_SOURCE_PATH),
        )
        gated_copy = [
            name for name in CHECKER.THRESHOLDS_NS if name.startswith("ext_mirror_copy/")
        ]
        self.assertEqual(gated_copy, [])

    def test_every_ceiling_is_a_positive_integer_nanosecond_count(self):
        for name, ceiling in CHECKER.THRESHOLDS_NS.items():
            with self.subTest(name=name):
                self.assertIsInstance(ceiling, int)
                self.assertGreater(ceiling, 0)


class RustConstExtractionTests(unittest.TestCase):
    """The two helpers behind the size tripwire.

    The tripwire is only as good as these: an extractor that quietly
    returns `None` on a valid declaration, or an evaluator that
    mis-multiplies, would put the size dimension back where it was —
    green locally, missing-gated-id in CI.
    """

    SOURCE = "\n".join(
        [
            "const PAYLOAD_SIZES: [usize; 4] = [1024, 4 * 1024, 64 * 1024, 1024 * 1024];",
            "",
            "/// Doc comment mentioning GATED_PAYLOAD_SIZE in prose.",
            "const GATED_PAYLOAD_SIZE: usize = 4 * 1024;",
            "const _: () = assert!(PAYLOAD_SIZES[1] == GATED_PAYLOAD_SIZE);",
        ]
    )

    def test_declaration_is_found_by_name_not_by_prose_mention(self):
        self.assertEqual(rust_const_expr(self.SOURCE, "GATED_PAYLOAD_SIZE"), "4 * 1024")

    def test_array_typed_declaration_does_not_swallow_the_next_const(self):
        """`[usize; 4]` puts a `;` inside the type, so a naive
        semicolon-terminated match would end the declaration early."""
        self.assertEqual(
            rust_const_expr(self.SOURCE, "PAYLOAD_SIZES"),
            "[1024, 4 * 1024, 64 * 1024, 1024 * 1024]",
        )

    def test_undeclared_name_yields_none(self):
        self.assertIsNone(rust_const_expr(self.SOURCE, "NO_SUCH_CONST"))

    def test_product_of_literals_evaluates(self):
        self.assertEqual(eval_rust_int_expr("4 * 1024"), 4096)

    def test_plain_literal_evaluates_to_itself(self):
        """A no-op rewrite of `4 * 1024` to `4096` must not fail the
        tripwire — that is exactly the brittleness the arithmetic
        avoids."""
        self.assertEqual(eval_rust_int_expr("4096"), 4096)

    def test_separators_and_type_suffixes_are_tolerated(self):
        self.assertEqual(eval_rust_int_expr("4_096usize"), 4096)

    def test_expression_outside_the_grammar_yields_none(self):
        """`None` rather than a guess: the caller turns it into a "widen
        the evaluator" failure, which is loud. A guess would be silent."""
        for expr in ["0x1000", "1 << 12", "BASE_SIZE * 4", "", "4 + 1024"]:
            with self.subTest(expr=expr):
                self.assertIsNone(eval_rust_int_expr(expr))


# ── Formatting ────────────────────────────────────────────────────────────


class FormatReportTests(unittest.TestCase):
    """The informational table. Its contract is that a reader never has to
    open the script to learn which numbers can fail a pipeline."""

    THRESHOLDS = {"gated/a": 1000}

    def test_header_states_the_measurement_count(self):
        report = CHECKER.format_report(
            [_result("gated/a", 1), _result("ungated/x", 2)], self.THRESHOLDS
        )
        self.assertEqual(report.splitlines()[0], "Benchmark results (2 measurement(s)):")

    def test_empty_results_render_a_header_only(self):
        report = CHECKER.format_report([], self.THRESHOLDS)
        self.assertEqual(report, "Benchmark results (0 measurement(s)):")

    def test_ungated_id_is_annotated_report_only(self):
        report = CHECKER.format_report([_result("ungated/x", 10_000_000)], self.THRESHOLDS)
        self.assertIn("[report-only]", report)
        self.assertNotIn("GATED", report)

    def test_passing_gated_id_shows_ceiling_and_headroom(self):
        report = CHECKER.format_report([_result("gated/a", 100, 5)], self.THRESHOLDS)
        self.assertIn("GATED ceiling=1,000 ns", report)
        self.assertIn("10.0x headroom", report)
        self.assertNotIn("EXCEEDED", report)

    def test_breaching_gated_id_is_marked_exceeded_with_the_overage(self):
        report = CHECKER.format_report([_result("gated/a", 3000, 5)], self.THRESHOLDS)
        self.assertIn("EXCEEDED by 3.0x", report)
        self.assertNotIn("headroom", report)

    def test_measurements_are_listed_in_id_order(self):
        report = CHECKER.format_report(
            [_result("z/id", 1), _result("a/id", 1), _result("m/id", 1)], self.THRESHOLDS
        )
        listed = [line.split()[0] for line in report.splitlines()[1:]]
        self.assertEqual(listed, ["a/id", "m/id", "z/id"])

    def test_measurement_and_deviation_are_both_rendered(self):
        report = CHECKER.format_report([_result("ungated/x", 1234567, 89012)], self.THRESHOLDS)
        self.assertIn("1,234,567 ns/iter", report)
        self.assertIn("(+/- 89,012)", report)

    def test_zero_nanosecond_measurement_does_not_divide_by_zero(self):
        """A degenerate-but-possible reading (an optimised-out body) must
        render, not raise — the report is printed before any verdict, so
        a crash here would hide the real failure."""
        report = CHECKER.format_report([_result("gated/a", 0)], self.THRESHOLDS)
        self.assertIn("gated/a", report)
        self.assertIn("GATED ceiling=1,000 ns", report)

    def test_thresholds_defaults_to_the_shipped_table(self):
        report = CHECKER.format_report([_result(GATED_ID, 1207, 44)])
        self.assertIn("GATED ceiling={:,} ns".format(CHECKER.THRESHOLDS_NS[GATED_ID]), report)


class FormatViolationTests(unittest.TestCase):
    """Operator-facing failure text. Its job is to make the reader's next
    action obvious without opening the script."""

    VIOLATION = _violation()

    def test_leads_with_fail_and_the_offending_id(self):
        text = CHECKER.format_violation(self.VIOLATION)
        self.assertTrue(text.startswith("FAIL ext_mirror_enqueue/mirror_drained/4096:"))

    def test_states_measured_ceiling_and_overage(self):
        text = CHECKER.format_violation(self.VIOLATION)
        self.assertIn("150,000 ns/iter", text)
        self.assertIn("50,000 ns/iter", text)
        self.assertIn("3.0x over", text)

    def test_names_the_likely_causes(self):
        """The four enumerated causes are the whole point of the message —
        without them a breach is just a number."""
        text = CHECKER.format_violation(self.VIOLATION)
        self.assertIn("try_send became blocking", text)
        self.assertIn("The item itself got more expensive to build", text)

    def test_rules_out_a_stalled_drain_as_the_cause(self):
        """The one misdiagnosis the message exists to pre-empt.

        A stalled drain sends every enqueue down a drop branch. That branch
        and a successful enqueue cost the same order of magnitude: both
        take the byte charge, neither copies or frees the payload in the
        timed region (the bench builds each value's copy outside it, and
        holds the dropped copies past it), and the drop branch adds only its
        structured WARN — so no amount of stalling moves this number far
        enough to reach a ceiling set ~40x above the expected cost. An
        operator who reaches for the drop counter on a breach is chasing the
        wrong signal, hence the explicit "what this is NOT" paragraph.

        The assertions pin the argument, not the arithmetic: the
        order-of-magnitude equivalence, why it holds at every payload size,
        and the 40x-headroom conclusion are what make the paragraph
        load-bearing; how much dearer the WARN makes the drop branch is a
        measurement that moves with the runner and the logging setup.
        """
        text = CHECKER.format_violation(self.VIOLATION)
        self.assertIn("What this is NOT: a stalled drain.", text)
        self.assertIn("cost the same order of magnitude", text)
        self.assertIn("neither copies or frees the payload in the timed region", text)
        self.assertIn("no amount of stalling reaches a ceiling set ~40x above", text)

    def test_names_the_damage_a_stall_actually_does(self):
        """Ruling the stall out is only half the message.

        A stall does not fail loudly — it quietly collapses
        `mirror_drained` onto `mirror_saturated`, so the arm keeps
        reporting a confident number for a code path nobody asked it to
        measure. Saying so is what stops a reader concluding "then a stall
        is harmless" from the paragraph above.
        """
        text = CHECKER.format_violation(self.VIOLATION)
        self.assertIn("A stall's real damage is silent rather than loud", text)
        self.assertIn("turns mirror_drained into a second mirror_saturated", text)

    def test_points_at_the_bench_self_check_for_the_stall_case(self):
        """The stall has its own detector, and the message says so —
        otherwise naming the silent damage above reads as "and nothing
        catches it"."""
        text = CHECKER.format_violation(self.VIOLATION)
        self.assertIn("live_edit_checkpoint_ext_mirror_dropped_total", text)
        self.assertIn("panics if anything was dropped", text)

    def test_gives_the_local_reproduction_command(self):
        text = CHECKER.format_violation(self.VIOLATION)
        self.assertIn("cargo bench -p live-edit --bench ext_mirror_enqueue", text)


class FormatMissingTests(unittest.TestCase):
    def test_leads_with_fail_and_the_missing_id(self):
        text = CHECKER.format_missing("gated/a")
        self.assertTrue(text.startswith("FAIL gated/a:"))

    def test_explains_that_absence_means_no_coverage(self):
        """Distinguishes this failure from a regression: the reader must
        go look at whether the bench ran, not at the enqueue hot path."""
        text = CHECKER.format_missing("gated/a")
        self.assertIn("no coverage", text.lower())
        self.assertIn("not present in the supplied bencher output", text)


# ── main() ────────────────────────────────────────────────────────────────


class MainExitCodeTests(_TempInputMixin, unittest.TestCase):
    """End-to-end CLI behaviour, driven exactly as `perf:ext-mirror-bench`
    drives it: one or more captured bencher-output files on argv.

    Exit-code contract: 0 clean, 1 perf verdict, 2 invocation error.
    """

    def test_no_arguments_is_a_usage_error(self):
        code, stdout, stderr = _run_main()
        self.assertEqual(code, 2)
        self.assertIn("usage: check_bench_thresholds.py", stderr)
        self.assertEqual(stdout, "")

    def test_unreadable_file_is_an_io_error_not_a_perf_verdict(self):
        """Exit 2, not 1: a CI script that captured its output to the wrong
        path must not look like a benchmark regression."""
        missing_path = str(self.tmpdir / "does-not-exist.txt")
        code, _stdout, stderr = _run_main(missing_path)
        self.assertEqual(code, 2)
        self.assertIn("cannot read", stderr)
        self.assertIn(missing_path, stderr)

    def test_unreadable_file_short_circuits_before_any_report(self):
        """The good file is never reported on, because the run is already
        known to be incomplete."""
        good = self.write_input(CANONICAL_LINE, name="good.txt")
        code, stdout, _stderr = _run_main(good, str(self.tmpdir / "nope.txt"))
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")

    def test_gated_id_under_ceiling_exits_clean(self):
        path = self.write_input(CANONICAL_LINE)
        code, stdout, stderr = _run_main(path)
        self.assertEqual(code, 0, msg="stderr was: " + stderr)
        self.assertIn("OK: every gated benchmark id is within its absolute ceiling.", stdout)
        self.assertEqual(stderr, "")

    def test_gated_id_over_ceiling_exits_one_with_the_violation_text(self):
        ceiling = CHECKER.THRESHOLDS_NS[GATED_ID]
        path = self.write_input(
            "test {} ... bench: {} ns/iter (+/- 10)".format(GATED_ID, ceiling + 1)
        )
        code, stdout, stderr = _run_main(path)
        self.assertEqual(code, 1)
        self.assertIn("FAIL {}".format(GATED_ID), stderr)
        self.assertNotIn("OK:", stdout)

    def test_missing_gated_id_exits_one_even_with_other_measurements(self):
        """The vacuity guard at CLI level: a stream full of ungated results
        is still a failure if the gated id is absent."""
        path = self.write_input(
            "test ext_mirror_store_values/no_mirror/16 ... bench: 91,000 ns/iter (+/- 12,000)"
        )
        code, stdout, stderr = _run_main(path)
        self.assertEqual(code, 1)
        self.assertIn("FAIL {}".format(GATED_ID), stderr)
        self.assertIn("report-only", stdout)

    def test_no_parsable_lines_exits_one_and_names_the_files(self):
        """Distinct from the missing-id path: nothing parsed at all, which
        usually means a format change or an empty capture."""
        path = self.write_input("Compiling live-edit v0.1.0\nFinished bench profile\n")
        code, _stdout, stderr = _run_main(path)
        self.assertEqual(code, 1)
        self.assertIn("no benchmark result lines were parsed", stderr)
        self.assertIn(path, stderr)
        self.assertIn("--output-format bencher", stderr)

    def test_empty_file_exits_one(self):
        path = self.write_input("")
        code, _stdout, stderr = _run_main(path)
        self.assertEqual(code, 1)
        self.assertIn("no benchmark result lines were parsed", stderr)

    def test_multiple_files_are_checked_as_one_stream(self):
        """CI passes one capture per bench target. The gated id living in
        the second file must still satisfy the check — the files are
        concatenated, not checked independently."""
        store_values = self.write_input(
            "test ext_mirror_store_values/no_mirror/16 ... bench: 91,000 ns/iter (+/- 12,000)",
            name="bench-store-values.txt",
        )
        enqueue = self.write_input(CANONICAL_LINE, name="bench-enqueue.txt")
        code, stdout, stderr = _run_main(store_values, enqueue)
        self.assertEqual(code, 0, msg="stderr was: " + stderr)
        self.assertIn("(2 measurement(s))", stdout)

    def test_report_is_printed_on_stdout_before_the_verdict_on_stderr(self):
        """Operators read the table even on a failing run, so the report
        must not be suppressed by the failure path — and it goes to stdout
        while verdicts go to stderr, keeping the two separable in CI
        logs."""
        ceiling = CHECKER.THRESHOLDS_NS[GATED_ID]
        path = self.write_input(
            "test {} ... bench: {} ns/iter (+/- 10)".format(GATED_ID, ceiling * 3)
        )
        code, stdout, stderr = _run_main(path)
        self.assertEqual(code, 1)
        self.assertIn("Benchmark results (1 measurement(s)):", stdout)
        self.assertIn("EXCEEDED by 3.0x", stdout)
        self.assertIn("FAIL", stderr)
        self.assertNotIn("FAIL", stdout)

    def test_undecodable_bytes_do_not_crash_the_reader(self):
        """`open(..., errors="replace")` is deliberate: a truncated or
        binary-contaminated capture must still be parsed for whatever
        valid lines it holds rather than raising `UnicodeDecodeError`."""
        path = self.tmpdir / "binary.txt"
        path.write_bytes(b"\xff\xfe garbage\n" + CANONICAL_LINE.encode("utf-8") + b"\n")
        code, stdout, stderr = _run_main(str(path))
        self.assertEqual(code, 0, msg="stderr was: " + stderr)
        self.assertIn(GATED_ID, stdout)

    @unittest.skipIf(os.geteuid() == 0, "root bypasses filesystem permission bits")
    def test_permission_denied_file_is_an_io_error(self):
        path = self.tmpdir / "locked.txt"
        path.write_text(CANONICAL_LINE, encoding="utf-8")
        path.chmod(0o000)
        self.addCleanup(path.chmod, 0o600)
        code, _stdout, stderr = _run_main(str(path))
        self.assertEqual(code, 2)
        self.assertIn("cannot read", stderr)


if __name__ == "__main__":
    unittest.main()
