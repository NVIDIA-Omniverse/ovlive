# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Python adapter, shared C runtime, and independent finite-service ledger."""
import asyncio
import ctypes
import gc
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

import grpc
from omni.usd import live
from service import State, Principal, ReferenceServer, pb, rpc

WRITER = (("authorization", "Bearer synthetic-writer"),)
# The plugin library, relative to the package root.
PLUGIN_LIBRARY = "bin/omni_usd_live.dll" if os.name == "nt" else "lib/libomni_usd_live.so"


class PythonFiniteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = State(int(os.environ["LIVE_TEST_SEED"]))
        self.state.tokens = {"synthetic-writer": Principal("writer")}
        self.server = ReferenceServer(self.state, "python-A")
        self.address = await self.server.start()
        self.channel = grpc.aio.insecure_channel(self.address)
        await asyncio.wait_for(self.channel.channel_ready(), 5)
        self.stub = rpc.LiveServiceStub(self.channel)
        self.owners = []
        self.addAsyncCleanup(self.cleanup)
        self.runtime = self.owner()
        self.endpoint = self.runtime.register_endpoint(self.address, access_token="synthetic-writer")

    def owner(self, **options):
        owner = live.initialize(**options)
        self.owners.append(owner)
        return owner

    async def cleanup(self):
        for fault in self.state.faults:
            fault.release.set()
        failures = []
        for owner in reversed(self.owners):
            result = await asyncio.to_thread(owner.close, 5000)
            if result != live.Result.OK:
                failures.append(result)
        await self.channel.close()
        await self.server.stop()
        (Path(os.environ["LIVE_TEST_ARTIFACT_DIR"]) / (self.id().split(".")[-1] + ".python.json")).write_text(
            json.dumps({"seed": int(os.environ["LIVE_TEST_SEED"]), "ledger": self.state.snapshot(),
                        "attempts": self.state.attempts, "faults": self.state.fault_log,
                        "cleanup_results": [int(x) for x in failures]}, indent=2) + "\n")
        self.state.assert_faults_fired()
        self.assertEqual(failures, [])

    async def create(self, name="lighting"):
        result, created = await live.create_session_async(self.endpoint, "s3://base/scene.usda", name)
        self.assertEqual(result, live.Result.OK)
        self.assertEqual(created.uri, live.make_session_uri(self.address, created.session_id))
        return created

    async def test_finite_result_shapes_and_exact_independent_metadata(self):
        created = await self.create("name ☃")
        self.assertFalse(hasattr(created, "name"))
        self.assertEqual(self.state.attempts.get("ListSessions", 0), 0, "create must not implicitly list")
        result, sessions = await live.list_sessions_async(self.endpoint, "s3://base/scene.usda")
        self.assertEqual(result, live.Result.OK)
        self.assertEqual(len(sessions), 1)
        self.assertEqual((sessions[0].session_id, sessions[0].uri, sessions[0].name, sessions[0].created_by),
                         (created.session_id, created.uri, "name ☃", "writer"))
        expected = (await self.stub.ListSessions(pb.ListSessionsRequest(path="s3://base/scene.usda"),
                                                metadata=WRITER, timeout=3)).sessions[0]
        self.assertEqual(sessions[0].created_time_ms, expected.created_at // 1000)
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=created.session_id, layer_name="root.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=b"independent opaque bytes\x00payload"))]),
            metadata=WRITER, timeout=3)
        uri = live.make_layer_uri(created.uri, "root.live")
        result, info = await asyncio.to_thread(live.stat_layer, uri)
        self.assertEqual(result, live.Result.OK)
        expected = await self.stub.StatLayer(pb.StatLayerRequest(session_id=created.session_id, layer_name="root.live"),
                                             metadata=WRITER, timeout=3)
        self.assertEqual((info.created_time_ms, info.modified_time_ms),
                         (expected.created_at // 1000, expected.last_modified_at // 1000))
        self.assertEqual(await live.delete_layer_async(uri), live.Result.OK)
        self.assertEqual(await live.stat_layer_async(uri), (live.Result.NOT_FOUND, None))
        self.assertEqual(await live.delete_session_async(created.uri), live.Result.OK)
        self.assertEqual(await live.list_sessions_async(self.endpoint, "s3://base/scene.usda"), (live.Result.OK, []))

    async def test_duplicate_endpoint_owners_and_shared_c_runtime(self):
        other = self.owner()
        duplicate = other.register_endpoint(self.address, access_token="synthetic-writer")
        handle = self.endpoint.handle
        self.assertEqual(duplicate.handle, handle)
        extension = Path(os.environ["LIVE_TEST_PYTHON_EXTENSION"])
        dll = ctypes.CDLL(str(extension.parents[4] / PLUGIN_LIBRARY))
        class Auth(ctypes.Structure):
            _fields_ = [("struct_size", ctypes.c_uint32), ("abi_version", ctypes.c_uint32),
                        ("generation", ctypes.c_uint64), ("expires_at_ms", ctypes.c_uint64),
                        ("challenge_id", ctypes.c_uint64), ("state", ctypes.c_uint32), ("reason", ctypes.c_uint32)]
        query = dll.omniUsdLiveGetAuthState
        query.argtypes = [ctypes.c_uint64, ctypes.POINTER(Auth)]
        query.restype = ctypes.c_uint32
        state = Auth(ctypes.sizeof(Auth), live.get_abi_version())
        self.assertEqual(query(handle, ctypes.byref(state)), live.Result.OK)
        self.assertEqual(state.generation, live.get_auth_state(duplicate).generation)
        self.assertEqual(self.endpoint.close(5000), live.Result.OK)
        self.assertEqual(self.endpoint.close(0), live.Result.OK, "reclosing one owner must not consume another")
        for query_owner in (live.get_auth_state, live.get_endpoint_status):
            with self.assertRaises(live.LiveError) as caught:
                query_owner(self.endpoint)
            self.assertEqual(caught.exception.result, live.Result.INVALID_HANDLE)
        self.assertEqual(await asyncio.to_thread(self.runtime.close, 5000), live.Result.OK)
        self.assertEqual(query(handle, ctypes.byref(state)), live.Result.OK)
        result, created = await live.create_session_async(duplicate, "base", "remaining owner")
        self.assertEqual(result, live.Result.OK)
        self.assertIsNotNone(created)
        self.assertEqual(await asyncio.to_thread(other.close, 5000), live.Result.OK)
        self.assertIn(query(handle, ctypes.byref(state)), (live.Result.NOT_INITIALIZED, live.Result.INVALID_HANDLE))

    async def test_immediate_submission_failure_and_cancel_held_reply(self):
        self.assertEqual(await asyncio.wait_for(live.stat_layer_async("invalid uri"), 2),
                         (live.Result.INVALID_ARGUMENT, None))
        fault = self.state.plan("ListSessions", "before_reply", pause=True)
        future = live.list_sessions_async(self.endpoint, "base")
        await asyncio.wait_for(fault.reached.wait(), 3)
        self.assertEqual(self.endpoint.close(0), live.Result.BUSY)
        future.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await future
        fault.release.set()
        self.assertEqual(await asyncio.to_thread(self.runtime.close, 5000), live.Result.OK)

    async def test_blocking_wait_timeout_retains_original_confirmed_creation(self):
        fault = self.state.plan("CreateSession", "before_reply", pause=True)
        with self.assertRaises(live.RequestTimeout) as caught:
            await asyncio.to_thread(live.create_session, self.endpoint, "timeout-base", "exactly once", timeout_ms=40)
        await asyncio.wait_for(fault.reached.wait(), 3)
        request = caught.exception.request
        self.assertEqual(caught.exception.result, live.Result.TIMEOUT)
        self.assertEqual(request.status, live.Result.BUSY)
        fault.release.set()
        self.assertEqual(await asyncio.to_thread(request.wait, 3000), live.Result.OK)
        result, created = request.outcome
        self.assertEqual(result, live.Result.OK)
        self.assertEqual(list(self.state.sessions), [created.session_id])
        self.assertEqual(self.state.attempts["CreateSession"], 1)
        self.assertEqual(request.close(), live.Result.OK)

    async def test_callback_payload_ownership_quiescence_and_self_close(self):
        entered, release, assigned = threading.Event(), threading.Event(), threading.Event()
        holder, received, self_results = {}, [], []
        def callback(result, created):
            assigned.wait(3)
            received.append((result, created, threading.get_ident()))
            self_results.append(holder["request"].close(0))
            entered.set()
            release.wait(3)
        request = live.create_session_with_callback(self.endpoint, "callback-base", "kept", callback)
        holder["request"] = request
        assigned.set()
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            self.assertEqual(request.wait(0), live.Result.TIMEOUT)
            self.assertEqual(self_results, [live.Result.BUSY])
            self.assertNotEqual(received[0][2], threading.get_ident())
        finally:
            release.set()
        self.assertEqual(await asyncio.to_thread(request.wait, 3000), live.Result.OK)
        self.assertEqual(request.status, live.Result.OK)
        self.assertEqual(request.outcome[:2], received[0][:2])
        self.assertEqual(request.close(3000), live.Result.OK)
        gc.collect()
        self.assertEqual(received[0][1].session_id, next(iter(self.state.sessions)))

    async def test_closed_loop_and_dropped_future_retire_native_requests(self):
        fault = self.state.plan("ListSessions", "before_reply", pause=True)
        loop = asyncio.new_event_loop()
        def submit():
            asyncio.set_event_loop(loop)
            async def start():
                return live.list_sessions_async(self.endpoint, "base")
            return loop.run_until_complete(start())
        future = await asyncio.to_thread(submit)
        await asyncio.wait_for(fault.reached.wait(), 3)
        loop.close()
        fault.release.set()
        del future
        gc.collect()
        abandoned = live.create_session_async(self.endpoint, "dropped", "once")
        del abandoned
        gc.collect()
        # Bounded runtime cleanup owns every admitted request even without a Future.
        self.assertEqual(await asyncio.to_thread(self.runtime.close, 5000), live.Result.OK)
        self.assertLessEqual(self.state.attempts.get("CreateSession", 0), 1)

    async def test_throwing_callback_and_logging_handler_do_not_strand_worker(self):
        owner = self.owner(python_workers=1, python_max_pending=2)
        endpoint = owner.register_endpoint(self.address, access_token="synthetic-writer")
        class BrokenHandler(logging.Handler):
            def emit(self, record):
                raise RuntimeError("test logging sink failure")
        handler = BrokenHandler()
        logger = logging.getLogger("omni.usd.live")
        def callback(*args):
            raise RuntimeError("test callback failure")
        logger.addHandler(handler)
        try:
            request = live.create_session_with_callback(endpoint, "callback-errors", "first", callback)
            self.assertEqual(await asyncio.to_thread(request.wait, 3000), live.Result.OK)
            result, created = await asyncio.wait_for(live.create_session_async(endpoint, "callback-errors", "second"), 3)
            self.assertEqual(result, live.Result.OK)
            self.assertEqual({s.name for s in self.state.sessions.values()}, {"first", "second"})
            self.assertIsNotNone(created)
        finally:
            logger.removeHandler(handler)

    async def test_bounded_python_admission_rejects_before_mutation(self):
        owner = self.owner(python_workers=1, python_max_pending=1)
        endpoint = owner.register_endpoint(self.address, access_token="synthetic-writer")
        entered, release = threading.Event(), threading.Event()
        def callback(*args):
            entered.set()
            release.wait(3)
        first = live.create_session_with_callback(endpoint, "bounded", "first", callback)
        try:
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            result, created = await live.create_session_async(endpoint, "bounded", "must not issue")
            self.assertEqual((result, created), (live.Result.RESOURCE_EXHAUSTED, None))
            self.assertEqual(self.state.attempts["CreateSession"], 1)
        finally:
            release.set()
        self.assertEqual(await asyncio.to_thread(first.wait, 3000), live.Result.OK)
        self.assertEqual(first.native_status, live.Result.OK)
        # Cleanup waits for worker reservation retirement as well as callback return.
        self.assertEqual(await asyncio.to_thread(owner.close, 3000), live.Result.OK)

    async def test_python_materialization_failure_keeps_native_commit_evidence(self):
        original = live._record
        def materialize(cls, value):
            if cls is live.CreatedSession:
                raise MemoryError("deterministic Python materialization fault")
            return original(cls, value)
        received = []
        with patch.object(live, "_record", side_effect=materialize):
            request = live.create_session_with_callback(self.endpoint, "allocation", "committed once",
                                                        lambda *value: received.append(value))
            self.assertEqual(await asyncio.to_thread(request.wait, 3000), live.Result.OK)
        self.assertEqual(received, [(live.Result.OUTCOME_UNKNOWN, None)])
        self.assertEqual(request.status, live.Result.OUTCOME_UNKNOWN)
        self.assertEqual(request.native_status, live.Result.OK)
        self.assertEqual(request.error.result, live.Result.OUTCOME_UNKNOWN)
        self.assertGreater(request.error.request_id, 0)
        self.assertGreater(request.error.attempt_id, 0)
        self.assertEqual(self.state.attempts["CreateSession"], 1)
        self.assertEqual([s.name for s in self.state.sessions.values()], ["committed once"])

    async def test_native_orphan_cleanup_and_concurrent_runtime_close(self):
        extra = self.owner()
        results = await asyncio.gather(asyncio.to_thread(extra.close, 3000), asyncio.to_thread(extra.close, 3000))
        self.assertEqual(results, [live.Result.OK, live.Result.OK])
        await self.create("remaining initialization")
        fault = self.state.plan("ListSessions", "before_reply", pause=True)
        result, native = live._live.list_sessions(self.endpoint._native, "orphan")
        self.assertEqual(result, live.Result.OK)
        handle = native.handle
        await asyncio.wait_for(fault.reached.wait(), 3)
        del native
        gc.collect()
        self.assertEqual(await asyncio.to_thread(self.runtime._native.cleanup, 3000), live.Result.OK)
        extension = Path(os.environ["LIVE_TEST_PYTHON_EXTENSION"])
        dll = ctypes.CDLL(str(extension.parents[4] / PLUGIN_LIBRARY))
        query = dll.omniUsdLiveGetRequestStatus
        query.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint32)]
        query.restype = ctypes.c_uint32
        status = ctypes.c_uint32(999)
        self.assertEqual(query(handle, ctypes.byref(status)), live.Result.INVALID_HANDLE)
        self.assertEqual(status.value, 999)
        fault.release.set()

    async def test_uncertain_mutation_survives_error_record_materialization_failure(self):
        original = live._record
        def materialize(cls, value):
            if cls is live.RequestError:
                raise MemoryError("deterministic error-record allocation fault")
            return original(cls, value)
        self.state.plan("CreateSession", "before_reply", code=grpc.StatusCode.UNAVAILABLE)
        with patch.object(live, "_record", side_effect=materialize):
            created = live.create_session_with_callback(self.endpoint, "unknown", "one commit", lambda *x: None)
            self.assertEqual(await asyncio.to_thread(created.wait, 3000), live.Result.OK)
        self.assertEqual(created.native_status, live.Result.OUTCOME_UNKNOWN)
        self.assertEqual(created.status, live.Result.OUTCOME_UNKNOWN)
        self.assertEqual(created.error.result, live.Result.OUTCOME_UNKNOWN)
        self.assertEqual(self.state.attempts["CreateSession"], 1)
        self.assertEqual(len(self.state.sessions), 1)
        uri = live.make_session_uri(self.address, next(iter(self.state.sessions)))
        self.state.plan("DeleteSession", "before_reply", code=grpc.StatusCode.UNAVAILABLE)
        with patch.object(live, "_record", side_effect=materialize):
            deleted = live.delete_session_with_callback(uri, lambda *x: None)
            self.assertEqual(await asyncio.to_thread(deleted.wait, 3000), live.Result.OK)
        self.assertEqual(deleted.native_status, live.Result.OUTCOME_UNKNOWN)
        self.assertEqual(deleted.status, live.Result.OUTCOME_UNKNOWN)
        self.assertEqual(deleted.error.result, live.Result.OUTCOME_UNKNOWN)
        self.assertEqual(self.state.attempts["DeleteSession"], 1)
        self.assertEqual(self.state.sessions, {})

    async def test_pure_helpers_and_interpreter_exit_without_explicit_close(self):
        uri = live.make_session_uri("EXAMPLE.com:443", "id/☃")
        parts = live.parse_uri(uri)
        self.assertEqual(parts.session_id, "id/☃")
        self.assertEqual(parts.authority, "example.com")
        self.assertEqual(live.get_capabilities().delegate_version, 2)
        with self.assertRaises((live.LiveError, ValueError)):
            live.make_session_uri("example.com", "truncated\x00suffix")
        path = Path(os.environ["LIVE_TEST_PYTHON_EXTENSION"])
        env = dict(os.environ, PYTHONPATH=str(path.parents[3]) + os.pathsep + os.environ["PYTHONPATH"],
                   PYTHONDONTWRITEBYTECODE="1")
        program = "from omni.usd import live; r=live.initialize(); e=r.register_endpoint('127.0.0.1:1',access_token='synthetic'); live.list_sessions_with_callback(e,'base',lambda *x: None)"
        result = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", program], env=env,
                                         capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        subinterpreter = """
import _xxsubinterpreters as interpreters
interpreter = interpreters.create()
try:
    try:
        interpreters.run_string(interpreter, 'from omni.usd import live')
    except interpreters.RunFailedError as error:
        assert 'main Python interpreter' in str(error) or 'subinterpreter' in str(error), str(error)
    else:
        raise AssertionError('unsupported subinterpreter import was accepted')
finally:
    interpreters.destroy(interpreter)
"""
        result = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", subinterpreter], env=env,
                                         capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    async def test_auth_status_and_diagnostics_queries(self):
        status = live.get_endpoint_status(self.endpoint)
        self.assertEqual(status.subject_id, self.endpoint.handle)
        self.assertEqual(status.subject_kind, 1)
        # The native mirror carries the appended field; an endpoint buffers no host edits.
        result, raw = self.endpoint._native.status()
        self.assertEqual((result, raw["buffered_edit_count"], status.buffered_edit_count), (live.Result.OK, 0, 0))
        self.assertEqual(live.Status.__dataclass_fields__["buffered_edit_count"].default, 0)
        # So do the layer progress sequences, which are zero for an endpoint.
        self.assertEqual((raw["read_sequence"], raw["applied_sequence"]), (0, 0))
        self.assertEqual((status.read_sequence, status.applied_sequence), (0, 0))
        for name in ("read_sequence", "applied_sequence"):
            self.assertEqual(live.Status.__dataclass_fields__[name].default, 0)
        # The lag threshold is a runtime option; the default equals an explicit 256.
        self.assertEqual(live.copy_diagnostics(self.endpoint, max_records=0)["effective_configuration"]
                         ["apply_lag_warning_threshold"], 256)
        # Read and join reconnects have their own backoff options, apart from the unary retry options.
        configuration = live.copy_diagnostics(self.endpoint, max_records=0)["effective_configuration"]
        self.assertEqual({name: configuration[name] for name in (
            "retry_initial_ms", "retry_maximum_ms", "retry_max_attempts",
            "reconnect_initial_ms", "reconnect_maximum_ms", "reconnect_max_attempts")},
            dict(retry_initial_ms=100, retry_maximum_ms=10000, retry_max_attempts=8,
                 reconnect_initial_ms=1000, reconnect_maximum_ms=60000, reconnect_max_attempts=10))
        state = live.get_auth_state(self.endpoint)
        result, generation = live.set_access_token(self.endpoint, state.generation, "synthetic-writer")
        self.assertEqual(result, live.Result.OK)
        self.assertGreater(generation, state.generation)
        self.assertEqual(live.set_access_token(self.endpoint, state.generation, "synthetic-writer"),
                         (live.Result.STALE_TOKEN_GENERATION, 0))
        self.assertEqual(live.configure_jitter_reduction(self.endpoint, 0, 0, 0), live.Result.OK)
        snapshot = live.copy_diagnostics(self.endpoint, max_records=8)
        self.assertEqual(snapshot["schema_version"], 1)
        self.assertEqual(snapshot["endpoint_filter"], self.endpoint.handle)
        self.assertNotIn("synthetic-writer", json.dumps(snapshot))
        # Registration recorded endpoint_registered (INFO) in history synchronously in asyncSetUp.
        for exported in (live.copy_diagnostics(self.endpoint), json.loads(live.export_diagnostics_json())):
            self.assertIn("endpoint_registered", [record["event_name"] for record in exported["recent_records"]])
        self.assertEqual(live.copy_diagnostics(self.endpoint, max_records=0)["recent_records"], [])
        # A fresh process initializes with an explicit threshold, which its configuration reports.
        path = Path(os.environ["LIVE_TEST_PYTHON_EXTENSION"])
        env = dict(os.environ, PYTHONPATH=str(path.parents[3]) + os.pathsep + os.environ["PYTHONPATH"],
                   PYTHONDONTWRITEBYTECODE="1")
        program = ("from omni.usd import live; r=live.initialize(apply_lag_warning_threshold=8, "
                   "reconnect_initial_ms=7, reconnect_maximum_ms=70, reconnect_max_attempts=3); "
                   "c=live.copy_diagnostics(max_records=0)['effective_configuration']; "
                   "print(c['apply_lag_warning_threshold'], c['reconnect_initial_ms'], c['reconnect_maximum_ms'], "
                   "c['reconnect_max_attempts']); r.close(5000)")
        result = await asyncio.to_thread(subprocess.run, [sys.executable, "-c", program], env=env,
                                         capture_output=True, text=True, timeout=10)
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "8 7 70 3"), result.stderr)
