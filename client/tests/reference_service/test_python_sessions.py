# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python stream ownership against independent presence and wire observations."""
import asyncio
import gc
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

import grpc
from omni.usd import live
from service import LiveService, State, Principal, ReferenceServer, pb


class RecordedService(LiveService):
    async def JoinSession(self, stream, context):
        async def record():
            async for request in stream:
                kind = request.WhichOneof("request_type")
                self.state.session_wire.append({"kind": kind,
                    "session": request.header.session_id if kind == "header" else "",
                    "app": request.header.user_application if kind == "header" else "",
                    "hex": request.message.data.hex() if kind == "message" else ""})
                yield request
        async for response in super().JoinSession(record(), context):
            yield response


class PythonSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seed = int(os.environ["LIVE_TEST_SEED"])
        self.random = random.Random(self.seed)
        self.state = State(self.seed)
        self.state.tokens = {"synthetic-writer": Principal("writer")}
        self.state.session_wire = []
        self.server = ReferenceServer(self.state, "python-stream", service_factory=RecordedService)
        self.address = await self.server.start()
        self.owners, self.release_events = [], []
        self.addAsyncCleanup(self.cleanup)
        self.runtime = self.owner()
        self.endpoint = self.runtime.register_endpoint(self.address, access_token="synthetic-writer")
        result, self.created = await live.create_session_async(self.endpoint, "scene.usda", "presence")
        self.assertEqual(result, live.Result.OK)

    def owner(self, **options):
        runtime = live.initialize(**options)
        self.owners.append(runtime)
        return runtime

    async def cleanup(self):
        for release in self.release_events:
            release.set()
        results = [await asyncio.to_thread(owner.close, 5000) for owner in reversed(self.owners)]
        await self.server.stop()
        artifact = Path(os.environ["LIVE_TEST_ARTIFACT_DIR"]) / (self.id().split(".")[-1] + ".python-session.json")
        artifact.write_text(json.dumps({"seed": self.seed, "wire": self.state.session_wire,
            "ledger": self.state.snapshot(), "attempts": self.state.attempts,
            "cleanup": [int(result) for result in results]}, indent=2) + "\n")
        self.assertEqual(results, [live.Result.OK] * len(results))

    async def until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait(), 5)

    def observer(self):
        events = []
        loop = asyncio.get_running_loop()
        def receive(event):
            loop.call_soon_threadsafe(events.append, event)
        return events, receive

    def child_env(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(os.environ["LIVE_TEST_PYTHON_EXTENSION"]).parents[3]) + os.pathsep + os.environ["PYTHONPATH"]
        return env

    async def connected(self, events, count=1):
        await self.until(lambda: len([e for e in events if e.kind == live.SessionEventKind.CONNECTION
                                      and e.result == live.Result.OK]) >= count)
        return [e for e in events if e.kind == live.SessionEventKind.CONNECTION and e.result == live.Result.OK][-1]

    async def test_binary_fifo_presence_and_independent_send_receipts(self):
        a_events, a_callback = self.observer()
        b_events, b_callback = self.observer()
        a = self.runtime.join_session(self.created.uri, "author ☃", a_callback)
        first = await self.connected(a_events)
        self.assertIsNone(first.user_name)
        self.assertIsNone(first.app_name)
        b = self.runtime.join_session(self.created.uri, "reviewer", b_callback)
        second = await self.connected(b_events)
        self.assertNotEqual(first.connection_id, second.connection_id)
        await self.until(lambda: any(e.kind == live.SessionEventKind.JOINED for e in a_events))
        joined = next(e for e in a_events if e.kind == live.SessionEventKind.JOINED)
        self.assertEqual((joined.connection_id, joined.user_name, joined.app_name),
                         (second.connection_id, "writer", "reviewer"))
        payloads = [b"", b"\x00\xffbinary\x00", self.random.randbytes(4097)]
        self.assertEqual(await a.send_message_async(payloads[0]), live.Result.OK)
        self.assertEqual(await asyncio.to_thread(a.send_message, payloads[1]), live.Result.OK)
        receipts = []
        request = a.send_message_with_callback(payloads[2], receipts.append)
        self.assertEqual(await asyncio.to_thread(request.wait, 3000), live.Result.OK)
        self.assertEqual(receipts, [live.Result.OK])
        self.assertEqual(request.outcome, live.Result.OK)
        await self.until(lambda: len([e for e in b_events if e.kind == live.SessionEventKind.MESSAGE]) == 3)
        messages = [e for e in b_events if e.kind == live.SessionEventKind.MESSAGE]
        self.assertEqual([e.bytes for e in messages], payloads)
        self.assertEqual([e.connection_id for e in messages], [first.connection_id] * 3)
        self.assertTrue(all(e.user_name is None and e.app_name is None for e in messages))
        self.assertEqual([bytes.fromhex(e["hex"]) for e in self.state.session_wire if e["kind"] == "message"], payloads)
        status = live.get_session_status(a)
        self.assertEqual(status.subject_kind, 2)
        self.assertNotEqual(status.subject_id, 0)  # Status subject IDs and session handles are distinct namespaces.
        self.assertEqual(status.state, live.State.READY)
        self.assertNotEqual(status.subject_id, live.get_session_status(b).subject_id)
        self.assertEqual(await asyncio.to_thread(a.leave, 3000), live.Result.OK)
        await self.until(lambda: any(e.kind == live.SessionEventKind.LEFT and e.connection_id == first.connection_id
                                    for e in b_events))
        self.assertEqual(await a.send_message_async(b"after leave"), live.Result.INVALID_HANDLE)
        with self.assertRaises(live.LiveError) as error:
            live.get_session_status(a)
        self.assertEqual(error.exception.result, live.Result.INVALID_HANDLE)
        self.assertEqual(messages[2].bytes, payloads[2], "events own payloads after native leave")

    async def test_reconnect_new_connection_and_continued_binary_delivery(self):
        a_events, a_callback = self.observer()
        b_events, b_callback = self.observer()
        a = self.runtime.join_session(self.created.uri, "rotating", a_callback)
        old = await self.connected(a_events)
        b = self.runtime.join_session(self.created.uri, "stable", b_callback)
        await self.connected(b_events)
        async with self.state.changed:
            peer = self.state.sessions[self.created.session_id].peers[old.connection_id]
            peer.close_code = grpc.StatusCode.UNAVAILABLE
            peer.ready.set()
        current = await self.connected(a_events, 2)
        self.assertNotEqual(current.connection_id, old.connection_id)
        self.assertEqual(self.state.attempts["JoinSession"], 3)
        self.assertEqual(a.delivery_status, live.Result.OK)
        self.assertEqual(await a.send_message_async(b"after rotation\x00"), live.Result.OK)
        await self.until(lambda: any(e.kind == live.SessionEventKind.MESSAGE for e in b_events))
        message = next(e for e in b_events if e.kind == live.SessionEventKind.MESSAGE)
        self.assertEqual((message.connection_id, message.bytes), (current.connection_id, b"after rotation\x00"))
        self.assertEqual(live.get_session_status(a).state, live.State.READY)
        self.assertEqual(live.get_session_status(b).state, live.State.READY)

    async def test_keepalive_timeout_rejoins_and_the_peer_sees_one_left(self):
        # An interval no test outlasts: only the advanced keepalive clock ends the join.
        self.state.keepalive_interval_ms = 3_600_000
        a_events, a_callback = self.observer()
        b_events, b_callback = self.observer()
        a = self.runtime.join_session(self.created.uri, "lapsing", a_callback)
        old = await self.connected(a_events)
        b = self.runtime.join_session(self.created.uri, "watching", b_callback)
        watching = await self.connected(b_events)
        self.assertEqual([j["keepalive_interval_ms"] for j in self.state.joins], [3_600_000, 3_600_000])
        await self.until(lambda: any(e.kind == live.SessionEventKind.JOINED for e in b_events))
        # The watcher sends a message at the current keepalive time, so only the lapsing join times out.
        self.assertEqual(self.state.advance_keepalive_clock(3 * 3_600_000 - 1), [])
        self.assertEqual(await b.send_message_async(b"still here"), live.Result.OK)
        await self.until(lambda: any(e.kind == live.SessionEventKind.MESSAGE for e in a_events))
        self.assertEqual(self.state.advance_keepalive_clock(1), [old.connection_id])
        current = await self.connected(a_events, 2)
        self.assertNotEqual(current.connection_id, old.connection_id)
        failures = [e for e in a_events if e.kind == live.SessionEventKind.CONNECTION and e.result != live.Result.OK]
        self.assertEqual([e.result for e in failures], [live.Result.UNAVAILABLE])
        await self.until(lambda: any(e.kind == live.SessionEventKind.JOINED and e.connection_id == current.connection_id
                                     for e in b_events))
        peers = [(e.kind, e.connection_id) for e in b_events
                 if e.kind in (live.SessionEventKind.JOINED, live.SessionEventKind.LEFT)]
        self.assertEqual(peers, [(live.SessionEventKind.JOINED, old.connection_id),
                                 (live.SessionEventKind.LEFT, old.connection_id),
                                 (live.SessionEventKind.JOINED, current.connection_id)])
        self.assertEqual(self.state.keepalive_timeouts, [old.connection_id])
        self.assertNotIn(watching.connection_id, self.state.keepalive_timeouts)
        self.assertEqual(live.get_session_status(a).state, live.State.READY)
        self.assertEqual(live.get_session_status(b).state, live.State.READY)

    async def test_held_callback_leave_timeout_other_work_and_retry(self):
        entered, release, exited = threading.Event(), threading.Event(), threading.Event()
        self.release_events.append(release)
        calls, self_close, blocking_send = [], [], []
        holder = []
        ready = threading.Event()
        def blocked(event):
            ready.wait(5)
            calls.append(event)
            blocking_send.append(holder[0].send_message(b"would depend on callback"))
            self_close.append(holder[0].leave(0))
            entered.set()
            release.wait()
            exited.set()
        session = self.runtime.join_session(self.created.uri, "held", blocked)
        holder.append(session)
        ready.set()
        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
        self.assertEqual(self_close, [live.Result.BUSY])
        self.assertEqual(blocking_send, [live.Result.BUSY])
        # Explicit leave cannot report Python callback quiescence yet.
        self.assertEqual(await asyncio.to_thread(session.leave, 10), live.Result.TIMEOUT)
        other_events, callback = self.observer()
        other = self.runtime.join_session(self.created.uri, "unaffected", callback)
        await self.connected(other_events)
        result, _ = await live.list_sessions_async(self.endpoint, "scene.usda")
        self.assertEqual(result, live.Result.OK)
        self.assertFalse(exited.is_set(), "unrelated work progressed while callback was still held")
        release.set()
        self.assertEqual(await asyncio.to_thread(session.leave, 3000), live.Result.OK)
        self.assertEqual(len(calls), 1)
        self.assertEqual(await asyncio.to_thread(other.leave, 3000), live.Result.OK)

    async def test_session_admission_refusal_precedes_native_join(self):
        runtime = self.owner(python_max_sessions=1)
        events, callback = self.observer()
        session = runtime.join_session(self.created.uri, "only", callback)
        await self.connected(events)
        attempts = self.state.attempts["JoinSession"]
        with self.assertRaises(live.LiveError) as error:
            runtime.join_session(self.created.uri, "over capacity", callback)
        self.assertEqual(error.exception.result, live.Result.RESOURCE_EXHAUSTED)
        self.assertEqual(self.state.attempts["JoinSession"], attempts)
        self.assertEqual(await asyncio.to_thread(session.leave, 3000), live.Result.OK)
        replacement_events, replacement_callback = self.observer()
        replacement = runtime.join_session(self.created.uri, "replacement", replacement_callback)
        await self.connected(replacement_events)
        self.assertEqual(await asyncio.to_thread(replacement.leave, 3000), live.Result.OK)

    async def test_queue_overflow_stops_transport_without_waiting_for_python(self):
        for limit in ("count", "bytes", "oversize"):
            with self.subTest(limit=limit):
                entered, release, exited = threading.Event(), threading.Event(), threading.Event()
                self.release_events.append(release)
                events = []
                def held(event):
                    events.append(event)
                    if len(events) == 1:
                        entered.set()
                        release.wait()
                        exited.set()
                session = self.runtime.join_session(self.created.uri, limit, held,
                    max_events=1 if limit == "count" else 8, max_bytes=1024)
                self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                identifier = events[0].connection_id
                async with self.state.changed:
                    peer = self.state.sessions[self.created.session_id].peers[identifier]
                    service = RecordedService(self.state, "inject")
                    service.emit(peer, pb.JoinSessionResponse(peer_joined=pb.PeerJoined(
                        connection_id="synthetic-peer", user_name="", user_application="")))
                    service.emit(peer, pb.JoinSessionResponse(peer_message=pb.PeerMessage(
                        connection_id="synthetic-peer", data=b"x" * (2048 if limit == "oversize" else 600))))
                    if limit == "bytes":
                        service.emit(peer, pb.JoinSessionResponse(peer_message=pb.PeerMessage(
                            connection_id="synthetic-peer", data=b"y" * 600)))
                await self.until(lambda: session.delivery_status == live.Result.RESOURCE_EXHAUSTED)
                await self.until(lambda: identifier not in self.state.sessions[self.created.session_id].peers)
                self.assertFalse(exited.is_set(), "transport stopped while callback was still held")
                release.set()
                await self.until(lambda: any(e.result == live.Result.RESOURCE_EXHAUSTED for e in events))
                failures = [e for e in events if e.result == live.Result.RESOURCE_EXHAUSTED]
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0].kind, live.SessionEventKind.CONNECTION)
                self.assertEqual(events[-1], failures[0])
                prefix = [live.SessionEventKind.CONNECTION, live.SessionEventKind.JOINED]
                if limit == "bytes":
                    prefix.append(live.SessionEventKind.MESSAGE)
                    self.assertEqual(events[2].bytes, b"x" * 600)
                self.assertEqual([e.kind for e in events], prefix + [live.SessionEventKind.CONNECTION])
                self.assertEqual((events[1].user_name, events[1].app_name), ("", ""))
                self.assertEqual(session.delivery_status, live.Result.RESOURCE_EXHAUSTED)
                self.assertNotEqual(await session.send_message_async(b"must not send"), live.Result.OK)
                self.assertEqual(await asyncio.to_thread(session.leave, 3000), live.Result.OK)

    async def test_callback_exception_and_logging_failure_do_not_kill_stream(self):
        calls, entered = [], threading.Event()
        class BrokenHandler(logging.Handler):
            def emit(self, record):
                raise RuntimeError("synthetic logging handler failure")
        handler = BrokenHandler()
        live._log.addHandler(handler)
        try:
            def receive(event):
                calls.append(event)
                if len(calls) == 1:
                    entered.set()
                    raise RuntimeError("synthetic session callback failure")
            a = self.runtime.join_session(self.created.uri, "throwing", receive)
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            other_events, callback = self.observer()
            b = self.runtime.join_session(self.created.uri, "sender", callback)
            await self.connected(other_events)
            self.assertEqual(await b.send_message_async(b"still delivered"), live.Result.OK)
            await self.until(lambda: any(e.kind == live.SessionEventKind.MESSAGE for e in calls))
            self.assertEqual(next(e.bytes for e in calls if e.kind == live.SessionEventKind.MESSAGE), b"still delivered")
            self.assertEqual(a.delivery_status, live.Result.OK)
        finally:
            live._log.removeHandler(handler)

    async def test_native_orphan_session_reaps_without_python_event_worker(self):
        result, native = live._live.join_session(self.created.uri, "native orphan")
        self.assertEqual(result, live.Result.OK)
        result, event = await asyncio.to_thread(native.next_event, 3000)
        self.assertEqual(result, live.Result.OK)
        self.assertEqual(event["kind"], live.SessionEventKind.CONNECTION)
        identifier = event["connection_id"]
        del native
        gc.collect()
        self.assertEqual(await asyncio.to_thread(self.runtime._native.cleanup, 3000), live.Result.OK)
        await self.until(lambda: identifier not in self.state.sessions[self.created.session_id].peers)

    async def test_python_event_materialization_failure_is_terminal(self):
        events, callback = self.observer()
        original = live._record
        def fail_record(cls, value):
            if cls is live.SessionEvent:
                raise MemoryError("deterministic session-event conversion failure")
            return original(cls, value)
        with patch.object(live, "_record", side_effect=fail_record):
            session = self.runtime.join_session(self.created.uri, "allocation failure", callback)
            await self.until(lambda: events)
        self.assertEqual([(e.kind, e.result) for e in events],
                         [(live.SessionEventKind.CONNECTION, live.Result.RESOURCE_EXHAUSTED)])
        self.assertEqual(session.delivery_status, live.Result.RESOURCE_EXHAUSTED)
        await self.until(lambda: not self.state.sessions[self.created.session_id].peers)
        self.assertEqual(await session.send_message_async(b"must not continue"), live.Result.RESOURCE_EXHAUSTED)
        self.assertEqual(await asyncio.to_thread(session.leave, 3000), live.Result.OK)

    async def test_thread_start_failure_rolls_back_before_join(self):
        runtime = self.owner(python_max_sessions=1)
        attempts = self.state.attempts.get("JoinSession", 0)
        with patch.object(threading.Thread, "start", side_effect=RuntimeError("synthetic thread start failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic thread start failure"):
                runtime.join_session(self.created.uri, "must not join", lambda event: None)
        self.assertEqual(self.state.attempts.get("JoinSession", 0), attempts)
        events, callback = self.observer()
        session = runtime.join_session(self.created.uri, "slot restored", callback)
        await self.connected(events)
        self.assertEqual(await asyncio.to_thread(session.leave, 3000), live.Result.OK)

    async def test_gc_stress_and_concurrent_native_consumer_gate(self):
        # CPython 3.12 defers threshold-triggered collection until evaluation:
        # GC stress does not prove same-thread reentry inside native conversion.
        # Separately observe the consumer gate under actual concurrent waiting.
        # A subprocess makes any deadlock bounded and attributable.
        script = """
import gc, sys, threading, time
from omni.usd import live
runtime = live.initialize()
endpoint = runtime.register_endpoint(sys.argv[1], access_token='synthetic-writer')
code, request = live._live.list_sessions(endpoint._native, 'scene.usda')
assert code == 0 and request.wait(3000) == 0
code, session = live._live.join_session(sys.argv[2], 'gc-reentry')
assert code == 0
active = None
session_results = []
request_queries = []
def reenter(phase, info):
    if phase != 'start':
        return
    if active == 'session':
        session_results.append(session.next_event(0)[0])
    elif active == 'request':
        request_queries.append(request.status())
old = gc.get_threshold()
gc.callbacks.append(reenter)
try:
    gc.set_threshold(1, 1, 1)
    for i in range(64):
        gc.collect()
        active = 'session'
        result = session.next_event(1)
        gc.collect()
        active = None
    assert session_results, 'GC stress must actually run the callback'
    for i in range(64):
        gc.collect()
        active = 'request'
        result = request.payload()
        gc.collect()
        active = None
        assert result[0] == 0 and len(result[1]) == 1
        if request_queries:
            break
    assert request_queries and all(value == (0, 0) for value in request_queries), request_queries
finally:
    active = None
    gc.callbacks.remove(reenter)
    gc.set_threshold(*old)
wait_results = []
def consume():
    while True:
        code, event = session.next_event(0xffffffff)
        wait_results.append(code)
        if code == 10:
            return
        if code == 14:
            time.sleep(.001)
worker = threading.Thread(target=consume, daemon=True)
worker.start()
deadline = time.monotonic() + 3
gate = None
while time.monotonic() < deadline:
    gate = session.next_event(0)[0]
    if gate == 14:
        break
    time.sleep(.001)
assert gate == 14, (gate, wait_results)
session.interrupt_wait()
worker.join(3)
assert not worker.is_alive() and wait_results[-1] == 10, wait_results
assert session.leave(3000) == 0
assert request.release() == 0
assert runtime.close(3000) == 0
print('native Python GC stress and concurrent consumer gate passed')
"""
        result = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", script, self.address, self.created.uri],
            env=self.child_env(), capture_output=True, text=True, timeout=12)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GC stress and concurrent consumer gate passed", result.stdout)

    async def test_permanent_join_failure_is_typed_and_not_retried(self):
        self.state.plan("JoinSession", "before", code=grpc.StatusCode.PERMISSION_DENIED)
        events, callback = self.observer()
        session = self.runtime.join_session(self.created.uri, "refused", callback)
        await self.until(lambda: any(e.result == live.Result.PERMISSION_DENIED for e in events))
        self.assertEqual(self.state.attempts["JoinSession"], 1)
        self.assertEqual(events[0].kind, live.SessionEventKind.CONNECTION)
        status = live.get_session_status(session)
        self.assertEqual((status.state, status.result), (live.State.FAILED, live.Result.PERMISSION_DENIED))
        self.assertEqual(await asyncio.to_thread(session.leave, 3000), live.Result.OK)
        self.state.assert_faults_fired()

    async def test_dropped_owner_and_exit_with_live_session(self):
        events, callback = self.observer()
        session = self.runtime.join_session(self.created.uri, "dropped", callback)
        connection = await self.connected(events)
        del session
        gc.collect()
        # Runtime retains callback/session ownership until it can quiesce it.
        self.assertEqual(await asyncio.to_thread(self.runtime.close, 3000), live.Result.OK)
        await self.until(lambda: connection.connection_id not in self.state.sessions[self.created.session_id].peers)
        script = """
import sys, threading
from omni.usd import live
owner = live.initialize()
endpoint = owner.register_endpoint(sys.argv[1], access_token='synthetic-writer')
ready = threading.Event()
observed = []
def receive(event):
    observed.append(event)
    if event.kind == live.SessionEventKind.CONNECTION and event.result == live.Result.OK:
        print(event.connection_id, flush=True)
        ready.set()
session = owner.join_session(sys.argv[2], 'exiting', receive)
assert ready.wait(5), observed
# Deliberately leave without explicit cleanup; interpreter exit must quiesce.
"""
        result = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", script, self.address, self.created.uri],
            env=self.child_env(), capture_output=True, text=True, timeout=12)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(self.state.namespace + "/connection-", result.stdout)
        self.assertTrue(any(item["kind"] == "header" and item["app"] == "exiting"
                            for item in self.state.session_wire))
        await self.until(lambda: not self.state.sessions[self.created.session_id].peers)
