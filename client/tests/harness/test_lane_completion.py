# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A lane passes only when every discovered test ran and passed, none skipped."""
from types import SimpleNamespace
import unittest

from tools.live import reference
from tools.live import test as live_runner


def outcome(**fields):
    values = dict(testsRun=3, skipped=[], expectedFailures=[], unexpectedSuccesses=[], failures=[], errors=[])
    values.update(fields)
    result = SimpleNamespace(**values)
    result.wasSuccessful = lambda: not (result.failures or result.errors or result.unexpectedSuccesses)
    return result


class LaneCompletionTests(unittest.TestCase):
    def test_reference_lane_requires_a_complete_nonempty_run(self):
        self.assertTrue(reference.complete_run(outcome(), 3))
        for fields, discovered in ((dict(testsRun=0), 0), (dict(testsRun=2), 3), (dict(skipped=[("t", "why")]), 3),
                                   (dict(expectedFailures=[("t", "tb")]), 3), (dict(unexpectedSuccesses=["t"]), 3),
                                   (dict(failures=[("t", "tb")]), 3)):
            with self.subTest(fields=fields):
                self.assertFalse(reference.complete_run(outcome(**fields), discovered))

    def test_runner_requires_complete_reference_evidence(self):
        good = dict(schema_version=1, lane="scene-reference", passed=True, discovered=4, tests_run=4, skipped=0,
                    failures=0, errors=0, expected_failures=0, unexpected_successes=0)
        self.assertTrue(live_runner.reference_complete(good, "scene-reference"))
        for key, value in (("passed", False), ("discovered", 0), ("tests_run", 3), ("skipped", 1),
                           ("expected_failures", 1), ("unexpected_successes", 1), ("lane", "plugin-reference"),
                           ("schema_version", 2)):
            with self.subTest(key=key):
                self.assertFalse(live_runner.reference_complete(dict(good, **{key: value}), "scene-reference"))

    def test_unittest_log_must_run_tests_without_skips(self):
        passing = "test_a (m.T) ... ok\n\n" + "-" * 70 + "\nRan 2 tests in 0.1s\n\nOK\n"
        self.assertTrue(live_runner.unittest_ran_all(passing))
        self.assertFalse(live_runner.unittest_ran_all("\n" + "-" * 70 + "\nRan 0 tests in 0.000s\n\nOK\n"))
        self.assertFalse(live_runner.unittest_ran_all("Ran 2 tests in 0.1s\n\nOK (skipped=1)\n"))
        self.assertFalse(live_runner.unittest_ran_all("Ran 2 tests in 0.1s\n\nFAILED (failures=1)\n"))
        self.assertFalse(live_runner.unittest_ran_all("OK\nRan 2 tests in 0.1s\n\nOK (skipped=1)\n"))

    def test_non_object_reference_evidence_is_incomplete(self):
        for evidence in ([], "passed", 1, None):
            with self.subTest(evidence=evidence):
                self.assertFalse(live_runner.reference_complete(evidence, "scene-reference"))
