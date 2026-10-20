# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-case JUnit, bounded run retention and identity failure text."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from tools.live import test as runner
from tools.live.test_support import ProcessResult, run_recording


class ReportingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.artifacts = self.root / "runs"

    def run_main(self, arguments, process=None):
        before = set(self.artifacts.iterdir()) if self.artifacts.exists() else set()
        with patch.object(sys, "argv", ["test.py", *arguments, "--artifact-dir", str(self.artifacts)]), \
                patch.object(runner, "fingerprint", return_value="synthetic-source"), \
                patch.object(runner, "identity", return_value="synthetic-toolchain"), \
                patch.object(runner, "run_process", side_effect=process), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = runner.main()
        return status, next(iter(set(self.artifacts.iterdir()) - before))

    def test_recording_result_keeps_each_case_outcome(self):
        def passes():
            pass

        def fails():
            raise AssertionError("injected failure")

        def skips():
            raise unittest.SkipTest("missing capability")

        tests = [unittest.FunctionTestCase(fn) for fn in (passes, fails, skips)]
        identities = [test.id() for test in tests]
        result, cases = run_recording(unittest.TestSuite(tests), stream=io.StringIO())
        self.assertEqual(result.testsRun, 3)
        self.assertEqual([case["outcome"] for case in cases], ["passed", "failed", "skipped"])
        self.assertEqual([case["id"] for case in cases], identities)
        self.assertIn("injected failure", cases[1]["message"])
        self.assertEqual(cases[2]["message"], "missing capability")

    def test_junit_carries_one_testsuite_per_suite_with_recorded_cases(self):
        cases = [dict(id="test_a.Tests.test_one", outcome="passed", message=""),
                 dict(id="test_a.Tests.test_two", outcome="passed", message="")]

        def process(command, **options):
            directory = Path(options["environment"]["LIVE_TEST_ARTIFACT_DIR"])
            summary = dict(schema_version=1, boundary="foundation", discovered=2, tests_run=2, skipped=0,
                           expected_failures=0, passed=True, test_ids=[case["id"] for case in cases], cases=cases,
                           excluded=([dict(id=name, reason=reason) for name, reason in runner.MACOS_EXCLUSIONS.items()]
                                     if sys.platform == 'darwin' else []))
            (directory / "foundation.json").write_text(json.dumps(summary))
            options["output"].write_text("")
            return ProcessResult("passed", 0, 1, 0, False, True)

        status, run = self.run_main(["--suite", "foundation,transport", "--seed", "42"], process)
        self.assertEqual(status, 2)
        junit = ET.parse(run / "junit.xml").getroot()
        self.assertEqual(junit.tag, "testsuites")
        self.assertEqual((junit.get("tests"), junit.get("failures"), junit.get("errors")), ("3", "0", "1"))
        self.assertEqual([suite.get("name") for suite in junit], ["foundation", "transport"])
        foundation = junit.find("testsuite[@name='foundation']")
        self.assertEqual([(case.get("classname"), case.get("name")) for case in foundation],
                         [("live.foundation", case["id"]) for case in cases])
        self.assertEqual(foundation.get("tests"), "2")
        transport = junit.find("testsuite[@name='transport']")
        self.assertEqual([case.get("name") for case in transport], ["transport"])
        self.assertIsNotNone(transport.find("testcase/error"))

    def test_keep_runs_prunes_the_oldest_earlier_runs_and_records_them(self):
        self.artifacts.mkdir()
        older = []
        for index in range(3):
            path = self.artifacts / f"run-older{index}"
            path.mkdir()
            (path / "manifest.json").write_text("{}")
            os.utime(path, (1000 + index, 1000 + index))
            older.append(path)
        unrelated = self.artifacts / "notes"
        unrelated.mkdir()
        status, current = self.run_main(["--suite", "transport", "--seed", "42", "--keep-runs", "2"])
        self.assertEqual(status, 2)
        self.assertEqual(sorted(self.artifacts.iterdir()), sorted([older[1], older[2], unrelated, current]))
        self.assertEqual(json.loads((current / "manifest.json").read_text())["pruned"], [older[0].name])
        before = sorted(self.artifacts.iterdir())
        with self.assertRaises(SystemExit) as error:
            self.run_main(["--suite", "transport", "--seed", "42", "--keep-runs", "1"])
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(sorted(self.artifacts.iterdir()), before)

    def test_keep_runs_never_prunes_the_replay_source_run(self):
        self.artifacts.mkdir()
        older = []
        for index in range(3):
            path = self.artifacts / f"run-older{index}"
            path.mkdir()
            os.utime(path, (1000 + index, 1000 + index))
            older.append(path)
        source = older[0] / "replay.json"
        source.write_text(json.dumps(dict(schema_version=1, suites=["transport"], master_seed=42, clients=1,
                                          timeout_ms=60000, max_output_bytes=1024,
                                          source_fingerprint="synthetic-source")))
        os.utime(older[0], (1000, 1000))
        status, current = self.run_main(["--replay", str(source), "--keep-runs", "2"])
        self.assertEqual(status, 2)
        self.assertEqual(sorted(self.artifacts.iterdir()), sorted([*older, current]))
        self.assertEqual(json.loads((current / "manifest.json").read_text())["pruned"], [])

    def test_a_passing_lane_whose_cases_differ_from_tests_run_is_an_infrastructure_error(self):
        def process(command, **options):
            directory = Path(options["environment"]["LIVE_TEST_ARTIFACT_DIR"])
            ids = ["test_a.Tests.test_one", "test_a.Tests.test_two"]
            summary = dict(schema_version=1, boundary="foundation", discovered=2, tests_run=2, skipped=0,
                           expected_failures=0, passed=True, test_ids=ids,
                           excluded=([dict(id=name, reason=reason) for name, reason in runner.MACOS_EXCLUSIONS.items()]
                                     if sys.platform == 'darwin' else []),
                           cases=[dict(id=ids[0], outcome="passed", message="")])
            (directory / "foundation.json").write_text(json.dumps(summary))
            options["output"].write_text("")
            return ProcessResult("passed", 0, 1, 0, False, True)

        status, run = self.run_main(["--suite", "foundation", "--seed", "42"], process)
        self.assertEqual(status, 2)
        result = json.loads((run / "manifest.json").read_text())["results"][0]
        self.assertEqual((result["outcome"], result["reason"]),
                         ("infrastructure_error", "case count differs from tests run"))

    def test_identity_failures_carry_their_cause(self):
        self.assertTrue(runner.identity(["definitely-missing-binary"]).startswith("unavailable: FileNotFoundError"))
        failed = runner.identity([sys.executable, "-c", "import sys; sys.exit('synthetic refusal')"])
        self.assertTrue(failed.startswith("unavailable: CalledProcessError"), failed)
        self.assertIn("synthetic refusal", failed)
        self.assertEqual(runner.identity([sys.executable, "-c", ""]), "unavailable: empty output")


if __name__ == "__main__":
    unittest.main()
