# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python host activity cleanup under deterministic TLS allocation failures.

These tests exercise the real Python task wrapper with a fake native task. They
do not qualify USD publication, native ownership, or interpreter shutdown.
"""
from enum import IntEnum
import importlib.util
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class Result(IntEnum):
    OK = 0
    CANCELLED = 10
    TIMEOUT = 9
    BUSY = 14


class FailingLocal:
    def __init__(self, fail_write):
        object.__setattr__(self, "writes", 0)
        object.__setattr__(self, "fail_write", fail_write)
        object.__setattr__(self, "host_task", object())
        object.__setattr__(self, "runtime", object())

    def __setattr__(self, name, value):
        count = self.writes + 1
        object.__setattr__(self, "writes", count)
        if count == self.fail_write:
            raise MemoryError("injected TLS write failure")
        object.__setattr__(self, name, value)


class HostLifecycleTests(unittest.TestCase):
    def setUp(self):
        name = "_live_host_lifecycle_test"
        self.api = ModuleType(name)
        self.api.Result = Result
        self.api.RequestError = SimpleNamespace
        self.api._local = FailingLocal(0)
        self.modules = patch.dict(sys.modules, {name: self.api})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        path = (Path(__file__).resolve().parents[2] /
                "source/bindings/python/omni/usd/live/_host.py")
        spec = importlib.util.spec_from_file_location(name + "._host", path)
        self.host = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.host)

    def context(self):
        # The constructor allocates an unstarted Thread; no worker/native state
        # is admitted in these bounded exception-path tests.
        return self.host.PublicationContext(SimpleNamespace(), None, None)

    def test_tls_setup_and_restoration_failures_retire_activity(self):
        for fail_write in range(1, 5):
            with self.subTest(fail_write=fail_write):
                self.api._local = FailingLocal(fail_write)
                context = self.context()
                calls = []
                native = SimpleNamespace(run=lambda: calls.append("run") or Result.OK)
                task = self.host._PublicationTask(context, native)
                with self.assertRaisesRegex(MemoryError, "injected TLS"):
                    task()
                self.assertEqual(calls, [] if fail_write <= 2 else ["run"])
                self.assertEqual(context._running, 0)
                self.assertTrue(context._wait_running(0), "exception stranded teardown activity")
                # Even failed host_task restoration must still attempt runtime
                # restoration, and both must still reach activity retirement.
                self.assertEqual(self.api._local.writes, 3 if fail_write == 1 else 4)
                task.cancel()

    def test_native_capture_retires_before_activity_completes(self):
        context = self.context()
        observations = []

        class Native:
            def run(self):
                observations.append(("run", context._running))
                return Result.OK

            def __del__(self):
                observations.append(("destroy", context._running))

        previous_host = self.api._local.host_task
        previous_runtime = self.api._local.runtime
        task = self.host._PublicationTask(context, Native())
        self.assertEqual(task(), Result.OK)
        self.assertEqual(observations, [("run", 1), ("destroy", 1)])
        self.assertEqual(context._running, 0)
        self.assertTrue(context._wait_running(0))
        self.assertIs(self.api._local.host_task, previous_host)
        self.assertIs(self.api._local.runtime, previous_runtime)

    def test_context_close_waits_for_reentrant_and_held_capture_cleanup(self):
        # Exercise the real close wrapper; native quiescence and consumer join
        # are already satisfied, so only Python capture retirement remains.
        self.api._local = threading.local()
        self.api._timeout = lambda milliseconds: milliseconds
        self.api._remaining = lambda deadline: max(0, int((deadline - time.monotonic()) * 1000))
        runtime = SimpleNamespace(_condition=threading.Condition(), _hosts=set())
        context = self.host.PublicationContext(runtime, None, None)
        runtime._hosts.add(context)
        context._native = SimpleNamespace(close=lambda: Result.OK, interrupt_wait=lambda: None)
        context._done.set()
        context._thread = SimpleNamespace(join=lambda: None)
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        finalizers = []
        results = []
        errors = []

        class Capture:
            def __init__(self, label, hold=False):
                self.label = label
                self.hold = hold

            def __call__(self, *args):
                raise AssertionError("close must not invoke a dispatch/error callback")

            def __del__(self):
                # Record instead of asserting in __del__, where failures could
                # become unraisable exceptions and falsely pass the fixture.
                try:
                    result = context.close(0)
                    with runtime._condition:
                        retained = context in runtime._hosts
                    finalizers.append((self.label, result, retained, context._closed))
                    if self.hold:
                        entered.set()
                        if not release.wait(5):
                            errors.append("capture release barrier timed out")
                except BaseException as error:
                    errors.append(error)
                    entered.set()

        context._dispatch = Capture("dispatch", hold=True)
        context._on_error = Capture("on_error")

        def close():
            try:
                results.append(context.close(3000))
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        closer = threading.Thread(target=close, daemon=True)
        closer.start()
        try:
            self.assertTrue(entered.wait(3), "close did not reach dispatch capture retirement")
            self.assertFalse(finished.is_set(), "close finished while capture destructor was held")
            self.assertFalse(context._closed)
            with runtime._condition:
                self.assertIn(context, runtime._hosts, "runtime released context before capture cleanup")
            # This caller overlaps the held finalizer, rather than relying on
            # sleep-based scheduling. It must not observe successful closure.
            self.assertEqual(context.close(0), Result.BUSY)
        finally:
            release.set()
            closer.join(3)
        self.assertFalse(closer.is_alive(), "capture cleanup stranded close")
        self.assertEqual(errors, [])
        self.assertEqual(results, [Result.OK])
        self.assertEqual(finalizers, [
            ("dispatch", Result.BUSY, True, False),
            ("on_error", Result.BUSY, True, False),
        ])
        self.assertTrue(context._closed)
        with runtime._condition:
            self.assertNotIn(context, runtime._hosts)
        self.assertEqual(context.close(0), Result.OK)

    def test_consumer_blocks_in_one_native_wait_until_close_interrupts_it(self):
        self.api._WAIT_FOREVER_MS = 0xffffffff
        self.api._local = threading.local()
        self.api._timeout = lambda value: value
        self.api._remaining = lambda deadline: max(0, int((deadline - time.monotonic()) * 1000))
        self.api._log_failure = lambda message: None

        class Native:
            def __init__(self):
                self.timeouts, self.entered, self.woken = [], threading.Event(), threading.Event()

            def next_task(self, timeout):
                self.timeouts.append(timeout)
                self.entered.set()
                # interrupt_wait ends the wait; the 10 s bound is a failure watchdog only.
                return (Result.CANCELLED if self.woken.wait(10) else Result.TIMEOUT), None

            def interrupt_wait(self):
                self.woken.set()

            def close(self):
                return Result.OK

        runtime = SimpleNamespace(_condition=threading.Condition(), _hosts=set(), _disarmed=False)
        context = self.host.PublicationContext(runtime, lambda task: None, None)
        native = context._native = Native()
        runtime._hosts.add(context)
        context._thread.start()
        context._admitted.set()
        self.assertTrue(native.entered.wait(10))
        self.assertEqual(context.close(10000), Result.OK)
        self.assertEqual(native.timeouts, [0xffffffff])

    def test_close_cancelling_the_wait_before_stop_is_a_clean_end(self):
        self.api._WAIT_FOREVER_MS = 0xffffffff
        self.api._local = threading.local()
        self.api._timeout = lambda value: value
        self.api._remaining = lambda deadline: max(0, int((deadline - time.monotonic()) * 1000))
        self.api._log_failure = lambda message: None
        errors = []

        class Native:
            """Native close marks the context closed and wakes the waiter with
            CANCELLED; it returns only after the consumer handled that result,
            so the consumer runs before PublicationContext.close sets _stop."""
            def __init__(self):
                self.entered, self.closed = threading.Event(), threading.Event()

            def next_task(self, timeout):
                self.entered.set()
                # close() ends the wait; the 10 s bound is a failure watchdog only.
                return (Result.CANCELLED if self.closed.wait(10) else Result.TIMEOUT), None

            def close(self):
                self.closed.set()
                assert context._done.wait(10), "consumer did not handle CANCELLED"
                return Result.OK

            def interrupt_wait(self):
                pass

            def abandon(self):
                pass

        runtime = SimpleNamespace(_condition=threading.Condition(), _hosts=set(), _disarmed=False)
        context = self.host.PublicationContext(runtime, lambda task: None, errors.append)
        native = context._native = Native()
        runtime._hosts.add(context)
        context._thread.start()
        context._admitted.set()
        self.assertTrue(native.entered.wait(10))
        self.assertEqual(context.close(10000), Result.OK)
        self.assertEqual(errors, [])
        self.assertEqual(context._failure, Result.OK)
        self.assertFalse(context._disarmed)


if __name__ == "__main__":
    unittest.main()
