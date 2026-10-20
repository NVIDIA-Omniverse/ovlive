# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Service readiness, scratch cleanup after interrupted and crashed runners, and the unsupported-platform refusal."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from tools.live import process_guard, test_support
from tools.live.test_support import containment_unsupported, read_ready, run_process

ROOT = Path(__file__).resolve().parents[2]
READY = f"import sys; sys.path.insert(0, {str(ROOT)!r}); from tools.live.test_support import report_ready; "
# A workload that holds a directory in its scratch space and a detached grandchild, records them, and waits.
HOLDING = ("import json, os, subprocess, sys, tempfile, threading\n"
           "from pathlib import Path\n"
           "held = tempfile.mkdtemp()\n"
           "grandchild = subprocess.Popen([sys.executable, '-c', 'import threading; threading.Event().wait()'],"
           " start_new_session=True)\n"
           "record = dict(workload=os.getpid(), guard=os.getppid(), grandchild=grandchild.pid, held=held,"
           " scratch=os.environ['LIVE_TEST_SCRATCH_DIR'])\n"
           "Path('started.json.tmp').write_text(json.dumps(record)); os.replace('started.json.tmp', 'started.json')\n"
           "threading.Event().wait()\n")
# A runner process that supervises HOLDING and exits 130 when interrupted.
RUNNER = ("import sys; sys.path.insert(0, {root!r})\n"
          "from pathlib import Path\n"
          "from tools.live.test_support import run_process\n"
          "try:\n"
          "    run_process([sys.executable, '-c', {workload!r}], cwd=Path.cwd(), output=Path('workload.log'),"
          " timeout_ms=600000, scratch=True)\n"
          "except KeyboardInterrupt:\n"
          "    raise SystemExit(130)\n")


def gone(pid):
    return not process_guard.alive(pid)


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_a_record_mid_publication_is_read_again_only_while_the_workload_runs(self):
        # Windows refuses to open a file that replace() is publishing; the next poll reads the whole record.
        record = self.directory / "ready.json"
        record.write_text("{}")
        with patch.object(Path, "read_text", side_effect=PermissionError), patch.object(sys, "platform", "win32"):
            self.assertIs(read_ready(record, publishing=True), test_support.PUBLISHING)
            self.assertEqual(read_ready(record), "unreadable readiness record: PermissionError")
        with patch.object(Path, "read_text", side_effect=PermissionError), patch.object(sys, "platform", "linux"):
            self.assertEqual(read_ready(record, publishing=True), "unreadable readiness record: PermissionError")

    def test_a_refusal_that_outlasts_publication_is_the_start_failure(self):
        # A Windows record that stays unreadable is reported as unreadable, not as a missing record.
        calls = []
        def refused(path, publishing=False):
            calls.append(publishing)
            return test_support.PUBLISHING if publishing else "unreadable readiness record: PermissionError"
        with patch.object(test_support, "read_ready", side_effect=refused):
            result = self.run_service("import threading; threading.Event().wait()", timeout_ms=3000)
        self.assertEqual((result.outcome, result.start_failure),
                         ("service_start_failed", "unreadable readiness record: PermissionError"))
        self.assertEqual(calls[:test_support.PUBLICATION_POLLS + 1], [True] * test_support.PUBLICATION_POLLS + [False])

    def run_service(self, code, **options):
        return run_process([sys.executable, "-c", code], cwd=self.directory, output=self.directory / "service.log",
                           ready=self.directory / "ready.json", **dict(dict(timeout_ms=30000), **options))

    def test_a_workload_that_reports_readiness_runs_to_its_own_outcome(self):
        passed = self.run_service(READY + "report_ready(True, lane='synthetic')")
        self.assertEqual((passed.outcome, passed.ready, passed.start_failure), ("passed", True, ""))
        failed = self.run_service(READY + "report_ready(True); raise AssertionError('after readiness')")
        self.assertEqual((failed.outcome, failed.ready), ("failed", True))

    def test_the_readiness_file_reaches_only_the_reporting_process(self):
        code = READY + ("import os, subprocess; report_ready(True); "
                        "print(subprocess.run([sys.executable, '-c', 'import os; print(os.environ.get(\"LIVE_TEST_READY_FILE\"))'],"
                        " capture_output=True, text=True).stdout.strip())")
        result = self.run_service(code)
        self.assertEqual(result.outcome, "passed")
        self.assertEqual((self.directory / "service.log").read_text().strip(), "None")

    def test_an_exit_before_readiness_is_a_failed_start_whatever_its_status(self):
        for code, status in (("pass", 0), ("raise SystemExit(125)", 125), ("raise SystemExit(1)", 1)):
            with self.subTest(status=status):
                result = self.run_service(code)
                self.assertEqual((result.outcome, result.ready, result.returncode), ("service_start_failed", False, status))
                self.assertEqual(result.start_failure, f"exited with {status} before reporting readiness")

    def test_a_reported_failed_start_keeps_its_reason(self):
        result = self.run_service(READY + "report_ready(False, 'loopback bind refused'); raise SystemExit(125)")
        self.assertEqual((result.outcome, result.ready, result.start_failure),
                         ("service_start_failed", False, "loopback bind refused"))
        # Published atomically, as report_ready publishes, so the supervisor reads the whole malformed record.
        malformed = self.run_service(READY.replace("report_ready", "write_json") +
                                     "import os; from pathlib import Path; write_json(Path(os.environ['LIVE_TEST_READY_FILE']), [])")
        self.assertEqual((malformed.outcome, malformed.start_failure), ("service_start_failed", "invalid readiness record"))

    def test_a_service_that_never_answers_expires_the_startup_watchdog(self):
        # The fixture holds its start forever, so only the startup watchdog can end the wait.
        result = self.run_service("import threading; threading.Event().wait()", startup_timeout_ms=200)
        self.assertEqual((result.outcome, result.ready, result.cleanup_complete), ("service_start_failed", False, True))
        self.assertEqual(result.start_failure, "no readiness record within 200 ms")
        self.assertLess(result.elapsed_ms, 20000)

    def test_a_stale_record_from_an_earlier_run_is_removed_before_launch(self):
        (self.directory / "ready.json").write_text(json.dumps(dict(schema_version=1, ready=True, reason="")))
        self.assertEqual(self.run_service("pass").outcome, "service_start_failed")


class ScratchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_the_scratch_directory_is_the_workload_temp_directory_and_is_removed(self):
        code = ("import json, os, tempfile; held = tempfile.mkdtemp(); open(os.path.join(held, 'x'), 'w').write('x'); "
                "print(json.dumps(dict(held=held, scratch=os.environ['LIVE_TEST_SCRATCH_DIR'],"
                " temp=[os.environ[k] for k in ('TMPDIR', 'TMP', 'TEMP')])))")
        result = run_process([sys.executable, "-c", code], cwd=self.directory, output=self.directory / "child.log",
                             timeout_ms=30000, scratch=True)
        self.assertEqual((result.outcome, result.cleanup_complete), ("passed", True))
        record = json.loads((self.directory / "child.log").read_text())
        scratch = Path(record["scratch"])
        self.assertTrue(scratch.name.startswith("live-scratch-"))
        self.assertEqual(record["temp"], [str(scratch)] * 3)
        self.assertEqual(Path(record["held"]).parent, scratch)
        self.assertFalse(scratch.exists())

    def test_a_nested_run_never_removes_the_scratch_directory_of_its_suite(self):
        # The nested workload inherits LIVE_TEST_SCRATCH_DIR; only the run that created a directory removes it.
        code = (f"import os, sys; sys.path.insert(0, {str(ROOT)!r}); from pathlib import Path; "
                "from tools.live.test_support import run_process; "
                "nested = run_process([sys.executable, '-c', 'pass'], cwd=Path.cwd(), output=Path('nested.log'), timeout_ms=30000); "
                "assert nested.outcome == 'passed', nested; "
                "assert os.path.isdir(os.environ['LIVE_TEST_SCRATCH_DIR']), 'nested run removed the suite scratch'")
        result = run_process([sys.executable, "-c", code], cwd=self.directory, output=self.directory / "child.log",
                             timeout_ms=30000, scratch=True)
        self.assertEqual((result.outcome, result.cleanup_complete), ("passed", True), (self.directory / "child.log").read_text())

    def test_no_scratch_directory_is_created_unless_requested(self):
        # A run inside a suite inherits that suite's directory; it never gets one of its own.
        result = run_process([sys.executable, "-c", "import os; print(os.environ.get('LIVE_TEST_SCRATCH_DIR'))"],
                             cwd=self.directory, output=self.directory / "child.log", timeout_ms=30000)
        self.assertEqual(result.outcome, "passed")
        self.assertEqual((self.directory / "child.log").read_text().strip(), str(os.environ.get("LIVE_TEST_SCRATCH_DIR")))


class RunnerDeathTests(unittest.TestCase):
    """A runner that is interrupted or killed leaves no workload process, grandchild or scratch directory."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def start_runner(self):
        # On Windows the runner gets its own process group, so a console break reaches it alone.
        options = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if sys.platform == "win32" else {}
        runner = subprocess.Popen([sys.executable, "-c", RUNNER.format(root=str(ROOT), workload=HOLDING)],
                                  cwd=self.directory, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **options)
        self.addCleanup(lambda: runner.poll() is None and (runner.kill(), runner.wait()))
        started, deadline = self.directory / "started.json", time.monotonic() + 30  # Failure-only watchdog.
        while not started.is_file():
            self.assertIsNone(runner.poll(), "runner ended before its workload started")
            self.assertLess(time.monotonic(), deadline, "workload did not start")
            time.sleep(0.01)
        record = json.loads(started.read_text())
        self.assertTrue(Path(record["held"]).is_dir())
        self.assertEqual(Path(record["held"]).parent, Path(record["scratch"]))
        return runner, record

    def assert_nothing_left(self, record):
        self.assertTrue(gone(record["workload"]), "workload survived")
        self.assertTrue(gone(record["grandchild"]), "detached grandchild survived")
        self.assertFalse(Path(record["scratch"]).exists(), "scratch directory survived")

    def test_an_interrupted_runner_leaves_no_process_or_directory(self):
        runner, record = self.start_runner()
        if sys.platform == "win32":
            # A console break ends a Python process without raising KeyboardInterrupt; the guard sees its runner exit.
            runner.send_signal(signal.CTRL_BREAK_EVENT)
            runner.wait(timeout=30)  # Failure-only watchdog.
            self.assertTrue(process_guard.wait_exit(record["guard"], 30), "guard did not exit after the runner did")
        else:
            runner.send_signal(signal.SIGINT)
            self.assertEqual(runner.wait(timeout=30), 130)  # Failure-only watchdog.
        self.assert_nothing_left(record)
        self.assertTrue(gone(record["guard"]), "guard survived")

    def test_a_crashed_runner_leaves_no_process_or_directory(self):
        runner, record = self.start_runner()
        runner.kill()
        runner.wait(timeout=30)
        # The guard's parent-death signal, or on Windows its wait on the runner's process handle, starts its cleanup.
        self.assertTrue(process_guard.wait_exit(record["guard"], 30), "guard did not exit after the runner died")
        self.assert_nothing_left(record)


class UnsupportedPlatformTests(unittest.TestCase):
    def test_a_platform_without_a_guard_launches_nothing_and_is_an_infrastructure_error(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(sys, "platform", "unsupported-os"), \
                patch("subprocess.Popen", side_effect=AssertionError("unexpected launch")):
            self.assertIn("open item platform", containment_unsupported())
            result = run_process([sys.executable, "-c", "pass"], cwd=Path(temporary),
                                 output=Path(temporary) / "child.log", timeout_ms=3000, scratch=True)
            self.assertEqual((result.outcome, result.returncode), ("infrastructure_error", None))
            self.assertIn("unsupported-os is unsupported", (Path(temporary) / "child.log").read_text())
        for platform in ("linux", "win32", "darwin"):
            with self.subTest(platform=platform), patch.object(sys, "platform", platform):
                self.assertIsNone(containment_unsupported())


if __name__ == "__main__":
    unittest.main()
