# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained operational observations and isolated best-effort diagnostics."""
import asyncio
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import grpc
from omni.usd import live
from service import State, Principal, ReferenceServer
import tls_fixture


class RecordingWait:
    """Forwards to a native subscription and records every next_event timeout."""
    def __init__(self, native, kind, waits, entered):
        self._native, self._kind, self._waits, self._entered = native, kind, waits, entered

    def next_event(self, timeout_ms):
        self._waits.append((self._kind, timeout_ms))
        self._entered.set()
        return self._native.next_event(timeout_ms)

    def __getattr__(self, name):
        return getattr(self._native, name)


class PythonObserverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seed = int(os.environ["LIVE_TEST_SEED"])
        self.state = State(self.seed)
        self.state.tokens = {"synthetic-writer": Principal("writer")}
        self.server = ReferenceServer(self.state, "python-observers")
        self.address = await self.server.start()
        self.owners, self.releases = [], []
        self.addAsyncCleanup(self.cleanup)
        # One endpoint plus exactly three retained session subjects. A copied
        # Python queue must not accidentally free the native subject budget.
        self.runtime = self.owner(max_status_subjects=4)
        self.endpoint = self.runtime.register_endpoint(self.address, access_token="synthetic-writer")
        result, self.created = await live.create_session_async(self.endpoint, "observers.usda", "status")
        self.assertEqual(result, live.Result.OK)

    def owner(self, **options):
        owner = live.initialize(**options)
        self.owners.append(owner)
        return owner

    async def cleanup(self):
        for release in self.releases:
            release.set()
        for fault in self.state.faults:
            fault.release.set()
        results = [await asyncio.to_thread(owner.close, 5000) for owner in reversed(self.owners)]
        await self.server.stop()
        (Path(os.environ["LIVE_TEST_ARTIFACT_DIR"]) / (self.id().split(".")[-1] + ".python-observer.json")).write_text(
            json.dumps({"seed": self.seed, "ledger": self.state.snapshot(), "attempts": self.state.attempts,
                        "cleanup": [int(value) for value in results]}, indent=2) + "\n")
        self.state.assert_faults_fired()
        self.assertEqual(results, [live.Result.OK] * len(results))

    async def until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait(), 5)

    async def join(self, app, expected=live.Result.OK):
        events = []
        session = self.runtime.join_session(self.created.uri, app, events.append)
        await self.until(lambda: any(event.kind == live.SessionEventKind.CONNECTION and event.result == expected
                                    for event in events))
        return session

    async def test_status_closed_retention_backpressures_churn_until_python_ack(self):
        entered, release, exited = threading.Event(), threading.Event(), threading.Event()
        self.releases.append(release)
        events = []
        def held(event):
            events.append(event)
            if len(events) == 1:
                entered.set()
                release.wait()
                exited.set()
        observer = live.register_status_callback(self.endpoint, held)
        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
        self.state.plan("JoinSession", "before", code=grpc.StatusCode.PERMISSION_DENIED)
        expected = []
        for index in range(3):
            result = live.Result.PERMISSION_DENIED if index == 0 else live.Result.OK
            session = await self.join("closed-" + str(index), result)
            status = live.get_session_status(session)
            expected.append((status.subject_id, result if index == 0 else live.Result.CANCELLED, status.incident_id))
            self.assertEqual(await asyncio.to_thread(session.leave, 3000), live.Result.OK)
        attempts = self.state.attempts["JoinSession"]
        with self.assertRaises(live.LiveError) as error:
            self.runtime.join_session(self.created.uri, "must not reach service", lambda event: None)
        self.assertEqual(error.exception.result, live.Result.RESOURCE_EXHAUSTED)
        self.assertEqual(self.state.attempts["JoinSession"], attempts)
        self.assertFalse(exited.is_set())
        self.assertEqual(len(events), 1)
        release.set()
        expected_ids = {item[0] for item in expected}
        await self.until(lambda: expected_ids.issubset({event.status.subject_id for event in events
                                                      if event.status.state == live.State.CLOSED}))
        for subject, result, incident in expected:
            closed = [event.status for event in events if event.status.subject_id == subject
                      and event.status.state == live.State.CLOSED]
            self.assertEqual(len(closed), 1)
            self.assertEqual((closed[0].result, closed[0].incident_id), (result, incident))
        self.assertNotEqual(expected[0][2], 0)
        for subject in {event.status.subject_id for event in events}:
            revisions = [event.status.revision for event in events if event.status.subject_id == subject]
            self.assertEqual(revisions, sorted(set(revisions)))
        self.assertEqual(observer.delivery_status, live.Result.OK)
        # CLOSED may have entered the callback just before its ACK. Retry only
        # pre-admission capacity failures and prove ACK restores space while the
        # observer remains registered; unregister must not mask a missing ACK.
        replacement_events = []
        async def admit():
            while True:
                try:
                    return self.runtime.join_session(self.created.uri, "capacity restored", replacement_events.append)
                except live.LiveError as failure:
                    if failure.result != live.Result.RESOURCE_EXHAUSTED:
                        raise
                    await asyncio.sleep(.001)
        replacement = await asyncio.wait_for(admit(), 3)
        await self.until(lambda: any(event.kind == live.SessionEventKind.CONNECTION and event.result == live.Result.OK
                                    for event in replacement_events))
        self.assertEqual(self.state.attempts["JoinSession"], attempts + 1)
        self.assertEqual(observer.delivery_status, live.Result.OK)
        self.assertEqual(await asyncio.to_thread(replacement.leave, 3000), live.Result.OK)
        self.assertEqual(await asyncio.to_thread(observer.close, 3000), live.Result.OK)

    async def test_auth_refresh_callback_resumes_request_and_stale_answer_cannot_replace_token(self):
        # Release the only registration before recreating its authority without
        # a token; no subscription has been installed yet.
        self.assertEqual(await asyncio.to_thread(self.endpoint.close, 3000), live.Result.OK)
        self.endpoint = self.runtime.register_endpoint(self.address)
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        events, answers = [], []
        def held(event):
            events.append(event)
            entered.set()
            release.wait()
            answers.append(live.set_access_token(self.endpoint, event.rejected_generation, "must-not-replace")[0])
        observer = live.register_auth_required_callback(self.endpoint, held)
        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
        pending = live.list_sessions_async(self.endpoint, "observers.usda")
        current = live.get_auth_state(self.endpoint)
        self.assertEqual((events[0].endpoint, events[0].challenge_id, events[0].rejected_generation),
                         (self.endpoint.handle, current.challenge_id, current.generation))
        result, generation = live.set_access_token(self.endpoint, current.generation, "synthetic-writer")
        self.assertEqual(result, live.Result.OK)
        self.assertEqual((await asyncio.wait_for(pending, 3))[0], live.Result.OK)
        release.set()
        await self.until(lambda: answers)
        self.assertEqual(answers, [live.Result.STALE_TOKEN_GENERATION])
        self.assertEqual(live.get_auth_state(self.endpoint).generation, generation)
        self.assertEqual(await asyncio.to_thread(observer.close, 3000), live.Result.OK)
        # A fresh missing-token challenge can also be answered directly from an
        # operational callback, without an event-loop pump or diagnostic sink.
        self.assertEqual(await asyncio.to_thread(self.endpoint.close, 3000), live.Result.OK)
        self.endpoint = self.runtime.register_endpoint(self.address)
        refreshed = []
        observer = live.register_auth_required_callback(self.endpoint, lambda event: refreshed.append(
            live.set_access_token(self.endpoint, event.rejected_generation, "synthetic-writer")[0]))
        self.assertEqual((await live.list_sessions_async(self.endpoint, "observers.usda"))[0], live.Result.OK)
        self.assertEqual(refreshed, [live.Result.OK])
        self.assertEqual(await asyncio.to_thread(observer.close, 3000), live.Result.OK)

    async def test_server_side_expiry_raises_one_rejected_challenge_and_the_request_retries(self):
        self.state.issue("python-t2", "writer")
        events, answers = [], []

        def refresh(event):
            events.append(event)
            answers.append(live.set_access_token(self.endpoint, event.rejected_generation, "python-t2")[0])
        observer = live.register_auth_required_callback(self.endpoint, refresh)
        before = self.state.attempts.get("ListSessions", 0)
        self.state.expire("synthetic-writer")
        self.assertEqual((await live.list_sessions_async(self.endpoint, "observers.usda"))[0], live.Result.OK)
        self.assertEqual([(e.reason, e.rejected_generation, e.grpc_status_present, e.grpc_status) for e in events],
                         [(3, 1, True, 16)])
        self.assertEqual(answers, [live.Result.OK])
        self.assertEqual(self.state.attempts["ListSessions"], before + 2)
        self.assertEqual([(r["method"], r["reason"]) for r in self.state.auth_rejections], [("ListSessions", "expired")])
        self.assertEqual(await asyncio.to_thread(observer.close, 3000), live.Result.OK)

    async def ingress_rejection(self, kind, pki=None):
        from http2_proxy import Http2Proxy
        proxy = Http2Proxy(self.address, ssl_context=pki and pki.server_context())
        self.addAsyncCleanup(proxy.close)  # Also on a failed assertion; close() raises the proxy's errors.
        tls = {} if pki is None else dict(transport_security=live.TransportSecurity.TLS, tls_server_name="live.test",
                                          ca_pem=pki.ca.read_text())
        endpoint = self.runtime.register_endpoint(await proxy.start(), access_token="synthetic-writer", **tls)
        events, answers = [], []

        def refresh(event):
            events.append(event)
            answers.append(live.set_access_token(endpoint, event.rejected_generation, "synthetic-writer")[0])
        observer = live.register_auth_required_callback(endpoint, refresh)
        proxy.reject_new_streams(kind)
        before = self.state.attempts.get("ListSessions", 0)
        self.assertEqual((await live.list_sessions_async(endpoint, "observers.usda"))[0], live.Result.OK)
        self.assertEqual([(e.reason, e.rejected_generation, e.grpc_status) for e in events], [(3, 1, 16)])
        self.assertEqual(answers, [live.Result.OK])
        self.assertEqual(self.state.attempts["ListSessions"], before + 1)  # The rejected request never reached it.
        attempts = live.copy_diagnostics(endpoint)["metrics"]["rpc_attempts_by_operation"]
        self.assertEqual([m["count"] for m in attempts if m["operation"] == 10], [2])  # The rejected attempt and its retry.
        self.assertEqual(len([e for e in proxy.events if e.get("action") == "ingress-reject"]), 1)
        self.assertEqual(await asyncio.to_thread(observer.close, 3000), live.Result.OK)
        self.assertEqual(await asyncio.to_thread(endpoint.close, 3000), live.Result.OK)
        await proxy.close()

    async def test_ingress_rejection_raises_a_rejected_challenge_without_an_http_status(self):
        self.assertNotIn("http_status", {f.name for f in fields(live.AuthRequired)})  # gRPC exposes no HTTP status.
        await self.ingress_rejection("http401")

    async def test_tls_endpoint_through_the_proxy_answers_an_ingress_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            await self.ingress_rejection("envoy", await tls_fixture.generate_async(directory))

    async def test_slow_diagnostics_drop_visibly_without_stalling_operational_callbacks(self):
        entered, release, exited = threading.Event(), threading.Event(), threading.Event()
        self.releases.append(release)
        diagnostics, statuses = [], []
        def held(record):
            diagnostics.append(record)
            if len(diagnostics) == 1:
                entered.set()
                release.wait()
                exited.set()
        sink = live.register_diagnostic_callback(held, endpoint=self.endpoint, min_severity=live.Severity.DEBUG,
                                                 max_records=1, max_bytes=4096)
        observer = live.register_status_callback(None, statuses.append)
        self.assertEqual((await live.list_sessions_async(self.endpoint, "observers.usda"))[0], live.Result.OK)
        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
        for _ in range(8):
            self.assertEqual((await live.list_sessions_async(self.endpoint, "observers.usda"))[0], live.Result.OK)
        await self.until(lambda: sink.dropped_records > 0)
        # Distinct authority creates a second endpoint while retaining the same
        # local service and native status budget.
        authority = "localhost:" + self.address.rsplit(":", 1)[1]
        endpoint = self.runtime.register_endpoint(authority)
        refreshed = []
        auth = live.register_auth_required_callback(endpoint, lambda event: refreshed.append(
            live.set_access_token(endpoint, event.rejected_generation, "synthetic-writer")[0]))
        self.assertEqual((await live.list_sessions_async(endpoint, "observers.usda"))[0], live.Result.OK)
        await self.until(lambda: refreshed and any(event.endpoint == endpoint.handle for event in statuses))
        self.assertEqual(refreshed, [live.Result.OK])
        self.assertFalse(exited.is_set(), "auth/status/finite progress occurred while diagnostics stayed held")
        self.assertEqual(await asyncio.to_thread(sink.close, 10), live.Result.TIMEOUT)
        release.set()
        self.assertEqual(await asyncio.to_thread(sink.close, 3000), live.Result.OK)
        self.assertGreater(sink.dropped_records, 0)
        self.assertTrue(diagnostics)
        self.assertNotIn("synthetic-writer", json.dumps([asdict(record) for record in diagnostics]))
        self.assertEqual(await asyncio.to_thread(auth.close, 3000), live.Result.OK)
        self.assertEqual(await asyncio.to_thread(observer.close, 3000), live.Result.OK)

    async def test_status_delivery_failure_retains_native_subject_capacity_until_explicit_close(self):
        for mode in ("conversion", "callback"):
            with self.subTest(mode=mode):
                errors = []
                original = live._record
                def convert(cls, value):
                    if cls is live.StatusEvent:
                        raise MemoryError("deterministic status conversion failure")
                    return original(cls, value)
                def receive(event):
                    if mode == "callback":
                        raise RuntimeError("deterministic status handler failure")
                    self.fail("conversion failure must not call status handler")
                with patch.object(live, "_record", side_effect=convert if mode == "conversion" else original):
                    observer = live.register_status_callback(self.endpoint, receive, on_error=errors.append)
                    await self.until(lambda: errors)
                expected = live.Result.RESOURCE_EXHAUSTED if mode == "conversion" else live.Result.INTERNAL_ERROR
                self.assertEqual(errors, [expected])
                self.assertEqual(observer.delivery_status, expected)
                for index in range(3):
                    session = await self.join(mode + str(index))
                    self.assertEqual(await asyncio.to_thread(session.leave, 3000), live.Result.OK)
                with self.assertRaises(live.LiveError) as error:
                    self.runtime.join_session(self.created.uri, "must remain blocked", lambda event: None)
                self.assertEqual(error.exception.result, live.Result.RESOURCE_EXHAUSTED)
                self.assertEqual(await asyncio.to_thread(observer.close, 3000), live.Result.OK)
                replacement = await self.join("explicit close releases capacity")
                self.assertEqual(await asyncio.to_thread(replacement.leave, 3000), live.Result.OK)

    async def test_subscription_admission_and_callback_close_are_bounded(self):
        runtime = self.owner(max_status_subjects=4, python_max_subscriptions=1)
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        holder, results = [], []
        ready = threading.Event()
        def callback(event):
            ready.wait(5)
            results.append(holder[0].close(0))
            entered.set()
            release.wait()
        # Endpoint=None explicitly selects this most recent runtime owner.
        observer = live.register_status_callback(None, callback)
        holder.append(observer)
        ready.set()
        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
        self.assertEqual(results, [live.Result.BUSY])
        self.assertEqual(await asyncio.to_thread(observer.close, 10), live.Result.TIMEOUT)
        with self.assertRaises(live.LiveError) as error:
            live.register_status_callback(None, lambda event: None)
        self.assertEqual(error.exception.result, live.Result.RESOURCE_EXHAUSTED)
        release.set()
        self.assertEqual(await asyncio.to_thread(observer.close, 3000), live.Result.OK)
        observer = live.register_status_callback(None, lambda event: None)
        self.assertEqual(await asyncio.to_thread(observer.close, 3000), live.Result.OK)
        with patch.object(threading.Thread, "start", side_effect=RuntimeError("synthetic observer thread failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic observer thread failure"):
                live.register_status_callback(None, lambda event: None)
        observer = live.register_status_callback(None, lambda event: None)
        self.assertEqual(await asyncio.to_thread(observer.close, 3000), live.Result.OK)
        self.assertEqual(await asyncio.to_thread(runtime.close, 3000), live.Result.OK)

    async def test_idle_subscriptions_block_in_one_native_wait_until_close(self):
        cases = (("status", "register_status_callback", lambda cb: live.register_status_callback(self.endpoint, cb)),
                 ("auth", "register_auth_required_callback",
                  lambda cb: live.register_auth_required_callback(self.endpoint, cb)),
                 ("diagnostic", "register_diagnostic_callback",
                  lambda cb: live.register_diagnostic_callback(cb, endpoint=self.endpoint)))
        for kind, name, register in cases:
            with self.subTest(kind=kind):
                waits, entered = [], threading.Event()
                real = getattr(live._live, name)
                def recording(*args, real=real, kind=kind, waits=waits, entered=entered):
                    result, native = real(*args)
                    return result, (RecordingWait(native, kind, waits, entered) if native is not None else native)
                with patch.object(live._live, name, recording):
                    observer = register(lambda event: None)
                self.assertTrue(await asyncio.to_thread(entered.wait, 5))
                # A wait that close did not end would leave close at TIMEOUT.
                self.assertEqual(await asyncio.to_thread(observer.close, 3000), live.Result.OK)
                self.assertEqual({timeout for _, timeout in waits}, {0xffffffff})

    async def test_interpreter_exit_with_active_subscription(self):
        script = """
import threading, sys
from omni.usd import live
owner = live.initialize()
endpoint = owner.register_endpoint(sys.argv[1], access_token='synthetic-writer')
ready = threading.Event()
observed = []
def receive(event):
    observed.append(event)
    ready.set()
observer = live.register_status_callback(endpoint, receive)
assert ready.wait(5)
assert observed[0].endpoint == endpoint.handle and observed[0].status.subject_kind == 1
print('active status subscription confirmed', flush=True)
"""
        package = Path(os.environ["LIVE_TEST_PYTHON_EXTENSION"]).parents[3]
        env = dict(os.environ, PYTHONPATH=str(package) + os.pathsep + os.environ["PYTHONPATH"],
                   PYTHONDONTWRITEBYTECODE="1")
        result = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", script, self.address],
            env=env, capture_output=True, text=True, timeout=12)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("active status subscription confirmed", result.stdout)

    async def test_teardown_joins_subscription_after_bounded_close_timeout(self):
        script = """
import threading, sys
from omni.usd import live
owner = live.initialize()
endpoint = owner.register_endpoint(sys.argv[1], access_token='synthetic-writer')
entered, release, retired = threading.Event(), threading.Event(), threading.Event()
def receive(event):
    entered.set()
    release.wait()
    retired.set()
observer = live.register_status_callback(endpoint, receive)
assert entered.wait(5)
real_close = owner.close
close_results = []
def bounded_close(timeout_ms=0):
    result = real_close(0)
    close_results.append(result)
    return result
owner.close = bounded_close  # Exercise exhausted deadlines without a 60s fixture.
joining = threading.Event()
real_wait = observer._done.wait
def observe_join(timeout=None):
    if timeout is None or timeout > 0:  # The patched close waits zero; teardown's join waits.
        joining.set()
    return real_wait(timeout)
observer._done.wait = observe_join
def unblock():
    # Release only when teardown reaches its watchdog-bounded subscription join.
    # Without that join, teardown must fail the retirement assertion below.
    assert joining.wait(5)
    assert close_results[0] == live.Result.TIMEOUT, close_results
    release.set()
helper = threading.Thread(target=unblock, daemon=True)
helper.start()
live._teardown()
assert retired.is_set() and not observer._thread.is_alive(), close_results
helper.join(3)
assert not helper.is_alive()
print('subscription joined after close timeout', flush=True)
"""
        package = Path(os.environ["LIVE_TEST_PYTHON_EXTENSION"]).parents[3]
        env = dict(os.environ, PYTHONPATH=str(package) + os.pathsep + os.environ["PYTHONPATH"],
                   PYTHONDONTWRITEBYTECODE="1")
        result = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", script, self.address],
            env=env, capture_output=True, text=True, timeout=12)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("subscription joined after close timeout", result.stdout)
