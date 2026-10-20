# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tools.live import process_guard
from tools.live.test_support import DeterministicScheduler, ScheduleMismatch, record_seeds, run_process

WINDOWS = sys.platform == "win32"
# Reads through a null pointer in C: SIGSEGV on Linux and an access violation on Windows, where ctypes would instead
# turn the fault into a Python exception.
CRASH = "import faulthandler; faulthandler.disable(); faulthandler._read_null()"


def environment_dump():
    """Print the exact fixture environment as NUL-separated records without adding variables."""
    if WINDOWS:
        return [sys.executable, "-I", "-X", "utf8", "-c",
                "import os, sys; sys.stdout.buffer.write(''.join(f'{k}={v}\\0' for k, v in os.environ.items()).encode('utf-8'))"]
    if sys.platform == "darwin":
        return ["/usr/bin/perl", "-e", 'binmode STDOUT; print "$_=$ENV{$_}\\0" for sort keys %ENV']
    return [shutil.which("env"), "-0"]


def read_environment(path):
    return dict(entry.split("=", 1) for entry in path.read_bytes().decode("utf-8").split("\0") if entry)


class SchedulerTests(unittest.TestCase):
    def exercise(self, replay=None):
        scheduler = DeterministicScheduler(replay)
        observed = []
        scheduler.schedule(5, "server.commit", lambda: observed.append("commit"))
        scheduler.schedule(5, "response.lost", lambda: observed.append("lost"))
        cancelled = scheduler.schedule(0, "cancelled", lambda: self.fail("cancelled callback executed"))
        scheduler.cancel(cancelled)
        scheduler.schedule(1, "callback.enqueue", lambda: scheduler.schedule(
            4, "callback.start", lambda: observed.append("callback")))
        scheduler.run()
        self.assertEqual(observed, ["commit", "lost", "callback"])
        return scheduler.history

    def test_replay_controls_clock_fifo_and_cancel(self):
        history = self.exercise()
        self.assertEqual(history, self.exercise(json.loads(json.dumps(history))))

    def test_replay_rejects_changed_or_unconsumed_event(self):
        history = self.exercise()
        history[0]["boundary"] = "wrong"
        with self.assertRaises(ScheduleMismatch):
            self.exercise(history)
        with self.assertRaises(ScheduleMismatch):
            DeterministicScheduler([{"boundary": "missing"}]).run()

    def test_exact_budget_still_validates_complete_replay(self):
        first = {"time_ms": 0, "ordinal": 0, "boundary": "commit"}
        extra = {"time_ms": 1, "ordinal": 1, "boundary": "unconsumed"}
        scheduler = DeterministicScheduler([first, extra])
        scheduler.schedule(0, "commit", lambda: None)
        with self.assertRaises(ScheduleMismatch):
            scheduler.run(max_steps=1)
        complete = DeterministicScheduler([first])
        complete.schedule(0, "commit", lambda: None)
        cancelled = complete.schedule(10, "cancelled", lambda: self.fail("cancelled task"))
        complete.cancel(cancelled)
        complete.run(max_steps=1)

    def test_runaway_scheduler_is_bounded(self):
        scheduler = DeterministicScheduler()
        def again():
            scheduler.schedule(0, "retry", again)
        again()
        with self.assertRaisesRegex(RuntimeError, "step budget"):
            scheduler.run(10)

    def test_seeds_persist_before_workers_and_differ(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory)
            seeds = record_seeds(path, 42, 32)
            self.assertEqual(seeds, json.loads((path / "seeds.json").read_text()))
            self.assertEqual(len(set(seeds["writer_seeds"])), 32)
            self.assertEqual(seeds, record_seeds(path, 42, 32))


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def run_child(self, code, **kwargs):
        return run_process([sys.executable, "-c", code], cwd=self.directory,
                           output=self.directory / "child.log", **kwargs)

    def test_success_failure_crash_and_launch_failure_are_distinct(self):
        self.assertEqual(self.run_child("print('ready')", timeout_ms=3000).outcome, "passed")
        self.assertEqual(self.run_child("raise AssertionError('injected')", timeout_ms=3000).outcome, "failed")
        self.assertEqual(self.run_child(CRASH, timeout_ms=3000).outcome, "crash")
        result = run_process([str(self.directory / "missing-executable")], cwd=self.directory,
                             output=self.directory / "missing.log", timeout_ms=3000)
        self.assertEqual(result.outcome, "infrastructure_error")

    def test_workload_environment_cannot_reconfigure_supervisor_python(self):
        # A Python guard inheriting this environment cannot even initialize.
        # The non-Python workload must receive every supplied value unchanged.
        environment = {"PYTHONHOME": str(self.directory / "not-a-python-install"),
                       "PYTHONPATH": str(self.directory / "not-a-module-path"),
                       "LD_LIBRARY_PATH": str(self.directory / "workload-sdk"),
                       "SYNTHETIC_VALUE": "spaces and Unicode: café\r\nsecond line=value\n"}
        with patch.dict(os.environ, {"PARENT_ONLY_CANARY": "must-not-reach-workload"}):
            result = run_process(environment_dump(), cwd=self.directory,
                                 output=self.directory / "environment.log", timeout_ms=3000,
                                 environment=environment)
        self.assertEqual(result.outcome, "passed")
        self.assertTrue(result.cleanup_complete)
        actual = read_environment(self.directory / "environment.log")
        self.assertEqual(actual, dict(environment, LIVE_STACK_DUMP_PREFIX=str(self.directory / "environment.stacks")))

    def test_oversized_environment_is_rejected_before_launch(self):
        with patch("subprocess.Popen", side_effect=AssertionError("unexpected launch")):
            with self.assertRaisesRegex(ValueError, "environment exceeds"):
                self.run_child("raise AssertionError('must not run')", timeout_ms=3000,
                               environment={"SYNTHETIC_LARGE": "x" * (1024 * 1024)})

    def test_hung_process_is_killed(self):
        result = self.run_child("import time; time.sleep(300)", timeout_ms=100)
        self.assertEqual(result.outcome, "timeout")
        self.assertTrue(result.cleanup_complete)
        self.assertLess(result.elapsed_ms, 5000)

    def test_expiry_before_the_guard_handles_signals_is_a_contained_timeout(self):
        expire = threading.Event()
        expire.set()  # Expires as soon as the guard is launched, before it can install its handlers.
        result = self.run_child("import time; time.sleep(300)", timeout_ms=60000, expire=expire)
        self.assertEqual(result.outcome, "timeout")
        self.assertTrue(result.cleanup_complete)

    def test_outer_timeout_reaps_nested_supervised_process(self):
        root = str(Path(__file__).resolve().parents[2])
        inner = "import os,time; from pathlib import Path; Path('inner.pid').write_text(str(os.getpid())); time.sleep(300)"
        outer = (f"import sys; sys.path.insert(0, {root!r}); "
                 "from tools.live.test_support import run_process; from pathlib import Path; "
                 f"run_process([sys.executable, '-c', {inner!r}], cwd=Path.cwd(), "
                 "output=Path('inner.log'), timeout_ms=300000)")
        result = self.run_child(outer, timeout_ms=1000)
        self.assertEqual(result.outcome, "timeout")
        self.assertTrue(result.cleanup_complete)
        self.assertEqual(result.stack_dumps, ())
        if WINDOWS:
            # Windows has no stack request, so a timeout records none.
            self.assertFalse((self.directory / "child.stacks.signalled").exists())
        else:
            # Neither the inner guard nor its workload has a SIGQUIT handler, so the outer guard signals nothing.
            self.assertEqual(json.loads((self.directory / "child.stacks.signalled").read_text()), [])
        self.assertFalse(process_guard.alive(int((self.directory / "inner.pid").read_text())))

    STUCK = ("import sys, threading; sys.path.insert(0, {root!r}); from tools.live import stack_dump; "
             "{install}lock = threading.Lock(); parked = threading.Condition(lock); count = [0]\n"
             "def deadlock_on_purpose():\n"
             "    with parked:\n        count[0] += 1; parked.notify_all(); parked.wait_for(lambda: False)\n"
             "for _ in range(2): threading.Thread(target=deadlock_on_purpose, daemon=True).start()\n"
             "with parked: parked.wait_for(lambda: count[0] == 2)\n"
             "print('ready', flush=True); threading.Event().wait()\n")

    def expire_when_ready(self, code):
        """Runs code under run_process and expires its watchdog once it prints ready."""
        expire, results = threading.Event(), []
        worker = threading.Thread(target=lambda: results.append(run_process(
            [sys.executable, "-c", code], cwd=self.directory, output=self.directory / "stuck.log",
            timeout_ms=60000, expire=expire)))
        worker.start()
        log, deadline = self.directory / "stuck.log", time.monotonic() + 10  # Failure-only watchdog.
        while not (log.is_file() and "ready" in log.read_text()):
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        expire.set()
        worker.join(30)  # Failure-only watchdog.
        self.assertFalse(worker.is_alive(), "run_process did not return after its watchdog expired")
        return results[0]

    def test_watchdog_expiry_captures_python_thread_stacks(self):
        root = str(Path(__file__).resolve().parents[2])
        result = self.expire_when_ready(self.STUCK.format(root=root, install="stack_dump.install(); "))
        if not sys.platform.startswith("linux"):
            # Watchdog stack requests are Linux-only.
            self.assertEqual((result.outcome, result.cleanup_complete, result.stack_dumps), ("timeout", True, ()))
            self.assertFalse((self.directory / "stuck.stacks.signalled").exists())
            return
        self.assertEqual((result.outcome, result.cleanup_complete, len(result.stack_dumps)), ("timeout", True, 1))
        self.assertGreaterEqual(Path(result.stack_dumps[0]).read_text().count("deadlock_on_purpose"), 2)
        signalled = json.loads((self.directory / "stuck.stacks.signalled").read_text())
        self.assertEqual([(entry["file"], entry["complete"]) for entry in signalled], [(result.stack_dumps[0], True)])
        self.assertEqual(result.stack_dumps[0], f"{self.directory / 'stuck.stacks'}.{signalled[0]['pid']}")

    def test_processes_without_a_dump_handler_are_never_sent_sigquit(self):
        root = str(Path(__file__).resolve().parents[2])
        result = self.expire_when_ready(self.STUCK.format(root=root, install=""))
        self.assertEqual((result.outcome, result.cleanup_complete, result.stack_dumps), ("timeout", True, ()))
        if not sys.platform.startswith("linux"):
            self.assertFalse((self.directory / "stuck.stacks.signalled").exists())
        else:
            self.assertEqual(json.loads((self.directory / "stuck.stacks.signalled").read_text()), [])
        self.assertEqual(list(self.directory.glob("stuck.stacks.*[0-9]")), [])  # Empty dump files are removed.

    def test_a_guard_without_its_request_handler_is_never_signalled(self):
        root = str(Path(__file__).resolve().parents[2])
        if not sys.platform.startswith("linux"):
            # The runner sends stack requests only on Linux.
            with patch("tools.live.test_support.os.kill", side_effect=AssertionError("signalled the guard")):
                result = self.expire_when_ready(self.STUCK.format(root=root, install="stack_dump.install(); "))
            self.assertEqual((result.outcome, result.cleanup_complete, result.stack_dumps), ("timeout", True, ()))
            return
        sent, kill = [], os.kill
        def record(pid, signum):
            sent.append(signum)
            return kill(pid, signum)
        with patch("tools.live.test_support._handler_installed", return_value=False), \
                patch("tools.live.test_support.os.kill", side_effect=record):
            result = self.expire_when_ready(self.STUCK.format(root=root, install="stack_dump.install(); "))
        self.assertEqual((result.outcome, result.cleanup_complete, result.stack_dumps), ("timeout", True, ()))
        self.assertNotIn(signal.SIGUSR1, sent)
        self.assertFalse((self.directory / "stuck.stacks.signalled").exists())

    def test_dumps_are_listed_for_every_outcome_and_stale_ones_are_removed(self):
        prefix = self.directory / "dumping.stacks"
        stale = [Path(f"{prefix}.1"), Path(f"{prefix}.signalled")]
        for path in stale:
            path.write_text("an earlier run\n")
        # An in-process watchdog writes its own dump and crashes, as the native drivers do.
        code = ("import os; from pathlib import Path; "
                "Path(os.environ['LIVE_STACK_DUMP_PREFIX'] + '.' + str(os.getpid())).write_text('dump\\n'); " + CRASH)
        result = run_process([sys.executable, "-c", code], cwd=self.directory, output=self.directory / "dumping.log",
                             timeout_ms=30000)
        self.assertEqual(result.outcome, "crash")
        self.assertEqual(len(result.stack_dumps), 1)
        self.assertTrue(result.stack_dumps[0].startswith(f"{prefix}."))
        self.assertEqual(Path(result.stack_dumps[0]).read_text(), "dump\n")
        self.assertFalse(any(path.exists() for path in stale))

    def test_a_relative_dump_prefix_is_absolute_in_the_workload(self):
        result = run_process(environment_dump(), cwd=self.directory, output=self.directory / "env.log",
                             timeout_ms=3000, stack_dump_prefix=Path("relative.stacks"))
        self.assertEqual(result.outcome, "passed")
        actual = read_environment(self.directory / "env.log")
        self.assertEqual(actual["LIVE_STACK_DUMP_PREFIX"], str(Path.cwd() / "relative.stacks"))

    def test_successful_parent_reaps_detached_session(self):
        inner = "import os,time; from pathlib import Path; Path('inner.pid').write_text(str(os.getpid())); print('ready',flush=True); time.sleep(300)"
        # A new session on POSIX; a detached process in its own process group on Windows.
        detach = "creationflags=0x8|0x200" if WINDOWS else "start_new_session=True"
        outer = ("import subprocess,sys; "
                 f"child=subprocess.Popen([sys.executable,'-c',{inner!r}], {detach}, stdout=subprocess.PIPE); "
                 "assert child.stdout.readline().rstrip() == b'ready'")
        result = self.run_child(outer, timeout_ms=3000)
        self.assertEqual(result.outcome, "passed", (self.directory / "child.log").read_text())
        self.assertTrue(result.cleanup_complete)
        self.assertFalse(process_guard.alive(int((self.directory / "inner.pid").read_text())))

    def test_saturated_output_is_drained_but_storage_bounded(self):
        result = self.run_child("import sys; sys.stdout.write('x' * 2000000)",
                                timeout_ms=3000, max_output_bytes=4096)
        self.assertEqual(result.outcome, "passed")
        self.assertTrue(result.output_truncated)
        self.assertEqual(result.output_bytes, 2000000)
        self.assertEqual((self.directory / "child.log").stat().st_size, 4096)

    def test_successful_parent_cannot_leak_child_with_open_output(self):
        code = ("import subprocess,sys; "
                "subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)']); "
                "print('parent complete', flush=True)")
        result = self.run_child(code, timeout_ms=3000)
        self.assertEqual(result.outcome, "passed")
        self.assertTrue(result.cleanup_complete)
        self.assertLess(result.elapsed_ms, 5000)


if __name__ == "__main__":
    unittest.main()
