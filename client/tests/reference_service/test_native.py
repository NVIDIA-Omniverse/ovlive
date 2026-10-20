# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native unary and layer-stream attempts; no USD or logical recovery claim."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import unittest

import grpc
from service import State, Principal, ReferenceServer, pb, rpc
import tls_fixture

WRITER = (("authorization", "Bearer synthetic-writer"),)
# Mutation outcome classification (design section 13): (gRPC code, OmniUsdLiveResult).
DEFINITIVE_MUTATION_CODES = (
    (grpc.StatusCode.INVALID_ARGUMENT, 1), (grpc.StatusCode.OUT_OF_RANGE, 1), (grpc.StatusCode.NOT_FOUND, 4),
    (grpc.StatusCode.FAILED_PRECONDITION, 16), (grpc.StatusCode.PERMISSION_DENIED, 7),
    (grpc.StatusCode.UNAUTHENTICATED, 6), (grpc.StatusCode.RESOURCE_EXHAUSTED, 13), (grpc.StatusCode.ALREADY_EXISTS, 5))
UNCERTAIN_MUTATION_CODES = tuple((code, 19) for code in (
    grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.INTERNAL,
    grpc.StatusCode.CANCELLED, grpc.StatusCode.UNKNOWN))
# Diagnostic driver modes that no correctness test runs, with the layer's expected terminal
# (OmniUsdLiveResult, gRPC code); `tools/live/test.py --driver-mode` runs one as the only case.
PROBE_MODES = {"actor-auth-timeout": (6, 0)}


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


class NativeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = State(int(os.environ["LIVE_TEST_SEED"]))
        self.state.tokens = {"synthetic-writer": Principal("writer"),
                             "synthetic-reader": Principal("reader", False)}
        self.servers, self.channels, self.children = [], [], []
        self.drainers = []
        self.proxies = []
        self.artifact = Path(os.environ["LIVE_TEST_ARTIFACT_DIR"])
        self.addAsyncCleanup(self.cleanup)
        server = ReferenceServer(self.state)
        self.servers.append(server)
        self.address = await server.start()
        channel = grpc.aio.insecure_channel(self.address)
        self.channels.append(channel)
        await asyncio.wait_for(channel.channel_ready(), 5)
        self.stub = rpc.LiveServiceStub(channel)

    async def cleanup(self):
        for process in self.children:
            if process.returncode is None:
                process.kill()
        failures = await asyncio.gather(*(p.close() for p in self.proxies), return_exceptions=True)
        failures += await asyncio.gather(*(p.wait() for p in self.children), return_exceptions=True)
        failures += await asyncio.gather(*self.drainers, return_exceptions=True)
        failures += await asyncio.gather(*(c.close() for c in self.channels), return_exceptions=True)
        failures += await asyncio.gather(*(s.stop() for s in self.servers), return_exceptions=True)
        if hasattr(self, "read_scripts"):
            (self.artifact / (self.id().split(".")[-1] + ".frames.json")).write_text(json.dumps(self.read_scripts, indent=2) + "\n")
        snapshot = self.state.snapshot()
        snapshot["rpc_attempts"] = dict(self.state.attempts)
        (self.artifact / (self.id().split(".")[-1] + ".ledger.json")).write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        self.state.assert_faults_fired()
        for failure in failures:
            if isinstance(failure, BaseException):
                raise failure

    async def driver(self, mode, result=0, code=0, *, token="synthetic-writer", extra=(), address=None, fault=None, release_on=None, steps=(), environment=None):
        # Keep draining after the bounded artifact is full, preventing log floods
        # from consuming disk or blocking a child on a full pipe.
        number = len(self.children)
        output = self.artifact / f"{self.id().split('.')[-1]}-{number}.native.log"
        process = await asyncio.create_subprocess_exec(os.environ["LIVE_TEST_UNARY_DRIVER"],
            address or self.address, mode, str(result), str(code), token, *map(str, extra),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            env=None if environment is None else dict(os.environ, **environment))
        self.children.append(process)
        self.assertFalse(steps and (fault or release_on), "step barriers cannot share fault handshakes")
        markers = []
        completed_steps = []
        async def drain():
            remaining = 65536
            marker_tail = b""
            with output.open("wb") as log:
                while data := await process.stdout.read(4096):
                    if release_on and not markers:
                        marker_tail = marker_tail + data
                        if release_on in marker_tail:
                            markers.append(release_on)
                            fault.release.set()
                        marker_tail = marker_tail[-len(release_on):]
                    kept = data[:remaining]
                    log.write(kept)
                    remaining -= len(kept)
                    if steps and len(completed_steps) < len(steps):
                        marker_tail = (marker_tail + data)[-4096:]
                        marker, observe = steps[len(completed_steps)]
                        if marker in marker_tail:
                            try:
                                await asyncio.wait_for(observe(), 3)
                            except BaseException:
                                process.stdin.close()  # Unblock the native barrier on assertion failure.
                                raise
                            completed_steps.append(marker)
                            marker_tail = b""
                            process.stdin.write(b"continue\n")
                            await process.stdin.drain()
                            if len(completed_steps) == len(steps):
                                process.stdin.close()
        drainer = asyncio.create_task(drain())
        self.drainers.append(drainer)
        if fault:
            await asyncio.wait_for(fault.reached.wait(), 3)
            self.assertIsNone(process.returncode)
            if not release_on:
                process.stdin.write(b"cancel\n")
                await process.stdin.drain()
        if not steps:
            process.stdin.close()
        await asyncio.wait_for(process.wait(), 12)
        await drainer
        # The test driver emits bounded diagnostics in normal operation; retain
        # only a bounded tail in assertion output while keeping the original log.
        with output.open("rb") as log:
            # Diagnostic cases return three bounded JSON snapshots for direct
            # schema/redaction comparisons, and the server-side auth and read
            # deadline cases one full export; preserve their complete final line.
            wide = mode.startswith(("diag-", "actor-auth-server-", "actor-auth-stale-generation", "actor-read-deadline-"))
            log.seek(max(0, output.stat().st_size - (65536 if wide else 16384)))
            tail = log.read().decode(errors="replace")
        self.assertEqual(process.returncode, 0, tail)
        if release_on:
            self.assertEqual(markers, [release_on])
        if steps:
            self.assertEqual(completed_steps, [marker for marker, _ in steps])
        data = json.loads(tail.splitlines()[-1])
        if mode.startswith(("diag-", "actor-flood-", "actor-read-pause", "actor-read-deadline-", "actor-auth-wait-", "actor-host-fault-", "actor-release-notices", "actor-buffer-emptied", "actor-issue-reservation", "actor-drop-without-close", "actor-auth-server-", "actor-auth-stale-generation")) or mode == "join-auth-wait-close":
            return data
        self.assertEqual((data["result"], data["grpc_status"]), (result, code))
        if mode.startswith("actor-"):
            self.assertIs(data["grpc_status_present"], False)
            # Typed observation is exercised without installing diagnostics.
            # Native assertions validate borrowed-data copying, callback thread,
            # canonical subject identity, and increasing per-subject revisions.
            self.assertGreaterEqual(data["status_events"], 3)
            self.assertEqual((data["closed_status"]["state"], data["closed_status"]["result"]), (9, result))
        return data

    async def probe_driver_mode(self):
        """Runs the diagnostic mode LIVE_TEST_DRIVER_MODE names against a fresh layer; discovery never selects it."""
        mode = os.environ["LIVE_TEST_DRIVER_MODE"]
        self.assertIn(mode, PROBE_MODES)
        result, code = PROBE_MODES[mode]
        session = await self.create_layer()
        await self.driver(mode, result, code, extra=[session])

    def service_rpc_count(self, method):
        return self.state.attempts.get(method, 0)

    def rejections(self):
        return [(r["method"], r["reason"], r["token_sha256"]) for r in self.state.auth_rejections]

    async def create_layer(self):
        session = (await self.stub.CreateSession(pb.CreateSessionRequest(path="base", name="fixture"),
                                               metadata=WRITER, timeout=5)).session_id
        messages = [pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=session, layer_name="root.live")),
                    pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=b"opaque-base"))]
        await self.stub.CreateLayer(iter(messages), metadata=WRITER, timeout=5)
        return session

    def check_diagnostic_snapshot(self, data, attempts, outcomes):
        snapshot = data["snapshot"]
        self.assertEqual(snapshot["schema_version"], 1)
        self.assertGreater(snapshot["runtime_instance_id"], 0)
        for name in ("provenance", "effective_configuration", "metrics", "history"):
            self.assertIsInstance(snapshot[name], dict)
            self.assertTrue(snapshot[name], name)
        self.assertGreater(snapshot["provenance"]["abi_version"], 0)
        self.assertRegex(snapshot["provenance"]["proto_sha256"], r"^[0-9a-f]{64}$")
        self.assertIsInstance(snapshot["statuses"], list)
        self.assertTrue(snapshot["statuses"])
        self.assertEqual(snapshot["closed_incidents"], [])
        metrics = snapshot["metrics"]
        self.assertEqual((metrics["rpc_attempts"], metrics["rpc_completions"], metrics["rpc_in_flight"]),
                         (attempts, attempts, 0))
        by_result = {(item["operation"], item["result"]): item["count"]
                     for item in metrics["rpc_by_operation_result"] if item["count"]}
        self.assertEqual(by_result, outcomes)
        histogram = metrics["rpc_duration_ms"]
        self.assertEqual(histogram["bounds"], [1, 5, 10, 50, 100, 500, 1000, 5000])
        self.assertEqual(len(histogram["counts"]), 9)
        self.assertEqual(sum(histogram["counts"]), attempts)
        durations = metrics['rpc_duration_ms_by_operation_result']
        self.assertEqual(durations['bounds'], histogram['bounds'])
        self.assertEqual({(r['operation'], r['result']): r['count'] for r in durations['entries']}, outcomes)
        for row in durations['entries']:
            self.assertEqual(len(row['counts']), 9)
            self.assertEqual(sum(row['counts']), row['count'])
            self.assertIs(type(row['sum_ms']), int)
            self.assertGreaterEqual(row['sum_ms'], 0)
        self.assertEqual([sum(r['counts'][i] for r in durations['entries']) for i in range(9)], histogram['counts'])
        sequences = [record["sequence"] for record in snapshot["recent_records"]]
        self.assertEqual(sequences, sorted(set(sequences)))
        self.assertTrue(sequences)
        self.assertLessEqual(snapshot["history"]["first_sequence"], sequences[0])
        self.assertGreaterEqual(snapshot["history"]["last_sequence"], sequences[-1])
        self.assertIsInstance(snapshot["history"]["truncated"], bool)
        for name in ("snapshot", "without_records", "named"):
            document = data[name]
            self.assertEqual(document["runtime_instance_id"], snapshot["runtime_instance_id"])
            self.assertEqual(document["provenance"], snapshot["provenance"])
            self.assertEqual(document["metrics"]["rpc_attempts"], attempts)
            self.assertEqual(document["metrics"]["rpc_completions"], attempts)
            subjects = document['metrics']['subjects']
            self.assertEqual([row['kind'] for row in subjects], [1, 2, 3])
            self.assertEqual(subjects[0]['current'], 1)
            self.assertEqual(subjects[0]['admitted'], 1)
            self.assertEqual(subjects[0]['detached'], 0)
            for row in subjects:
                self.assertEqual([state['state'] for state in row['states']], list(range(1, 10)))
                self.assertEqual(sum(state['current'] for state in row['states']), row['current'])
                self.assertEqual(row['current'], row['admitted'] - row['detached'])
            self.assertEqual([row['admitted'] for row in subjects[1:]], [0, 0])
            readiness = document['metrics']['readiness']
            self.assertEqual([row['kind'] for row in readiness], [2, 3])
            for row in readiness:
                self.assertEqual((row['initial_waiting'], row['recovering']), (0, 0))
                self.assertEqual(row['initial_ms_by_resolution']['entries'], [])
                self.assertEqual(row['recovery_ms_by_resolution']['entries'], [])
            capacity = document['metrics']['capacity']
            slots = capacity['request_slots']
            self.assertEqual(set(slots), {'ordinary', 'timing'})
            self.assertGreater(slots['ordinary']['admitted'], 0)
            self.assertEqual(slots['timing']['admitted'], 0)
            for slot in slots.values():
                self.assertEqual(slot['current'], 0)  # Every request handle was explicitly released.
                self.assertEqual(slot['admitted'], slot['released'])
                self.assertEqual(slot['rejected'], 0)
                self.assertLessEqual(slot['high_water'], slot['admitted'])
            charged = capacity['pending_bytes']
            self.assertEqual(charged['reservations']['current'], 0)
            self.assertEqual(charged['current_bytes'], 0)
            self.assertFalse(charged['current_bytes_overflow'])
            tags = document['metrics']['cq_tags']
            # These finite calls use one Finish tag per actual attempt. A CQ
            # ok receipt includes gRPC UNAVAILABLE responses, not only OK RPCs.
            self.assertEqual(tags['submitted'], attempts)
            self.assertEqual(tags['received_ok'], attempts)
            self.assertEqual(tags['received_not_ok'], 0)
            self.assertEqual(tags['outstanding'], 0)
            self.assertEqual(tags['abandoned'], 0)
            self.assertEqual(tags['retired'] + tags['retained'], tags['submitted'])
            self.assertLessEqual(tags['handling'], tags['retained'])
            self.assertGreaterEqual(tags['retained_high_water'], tags['retained'])
            self.assertGreater(tags['outstanding_high_water'], 0)
            executors = document['metrics']['executors']
            self.assertEqual(set(executors), {'operational', 'diagnostic'})
            for executor in executors.values():
                self.assertEqual(executor['enqueued'], executor['started'] + executor['queued'])
                self.assertEqual(executor['started'], executor['completed'] + executor['running'])
                self.assertLessEqual(executor['running'], executor['live_workers'])
                self.assertLessEqual(executor['live_workers'], executor['configured_workers'])
                self.assertGreaterEqual(executor['queue_high_water'], executor['queued'])
                for duration, count in [('queue_wait_ms', 'started'), ('dispatch_duration_ms', 'completed')]:
                    row = executor[duration]
                    self.assertEqual(row['bounds'], histogram['bounds'])
                    self.assertEqual(len(row['counts']), 9)
                    self.assertEqual(row['count'], executor[count])
                    self.assertEqual(sum(row['counts']), row['count'])
                    self.assertGreaterEqual(row['sum_ms'], 0)
                for age, gauge in [('oldest_queued_age_ms', 'queued'), ('oldest_running_age_ms', 'running')]:
                    self.assertGreaterEqual(executor[age], 0)
                    if executor[gauge] == 0:
                        self.assertEqual(executor[age], 0)
            serialized = json.dumps(document)
            self.assertNotIn("synthetic-writer", serialized)
            self.assertNotIn("Bearer", serialized)
            self.assertNotIn("opaque-base", serialized)
            if name != "named":
                self.assertNotIn(self.address, serialized)
        self.assertEqual(data["without_records"]["recent_records"], [])
        self.assertTrue(data["without_records"]["statuses"])

    async def test_diagnostics_counts_real_transport_attempts_and_exports_json(self):
        session = await self.create_layer()
        self.state.plan("ListSessions", "before_reply", code=grpc.StatusCode.UNAVAILABLE)
        self.state.plan("StatLayer", "before_reply", code=grpc.StatusCode.UNAVAILABLE)
        data = await self.driver("diag-metrics", extra=[session])
        # The service's attempt ledger is independent of native instrumentation.
        self.assertEqual(self.state.attempts,
                         {"CreateSession": 1, "CreateLayer": 1, "Ping": 1, "ListSessions": 2, "StatLayer": 2})
        self.check_diagnostic_snapshot(data, 5, {(13, 0): 1, (10, 0): 1, (10, 8): 1, (12, 0): 1, (12, 8): 1})

    async def test_blocked_diagnostic_sinks_preserve_rpc_and_status_progress(self):
        session = await self.create_layer()
        data = await self.driver("diag-blocked", extra=[session])
        self.assertEqual(self.state.attempts,
                         {"CreateSession": 1, "CreateLayer": 1, "Ping": 8, "ListSessions": 1})
        self.check_diagnostic_snapshot(data, 9, {(13, 0): 8, (10, 0): 1})
        snapshot = data["snapshot"]
        self.assertGreater(snapshot["metrics"]["sink_dropped"], 0)
        self.assertGreater(snapshot["metrics"]["history_overwritten"], 0)
        self.assertLessEqual(len(snapshot["recent_records"]), 4)
        for name in ('snapshot', 'without_records', 'named'):
            executors = data[name]['metrics']['executors']
            diagnostic = executors['diagnostic']
            # Both callbacks remain behind explicit release gates throughout
            # capture, while real RPC and status work have progressed.
            self.assertEqual((diagnostic['enqueued'], diagnostic['started'], diagnostic['completed']), (2, 2, 0))
            self.assertEqual((diagnostic['queued'], diagnostic['running'], diagnostic['live_workers']), (0, 2, 2))
            self.assertGreater(executors['operational']['completed'], 0)

    async def test_public_session_metadata_roundtrip(self):
        data = await self.driver("public-roundtrip")
        self.assertEqual(data["callbacks"], 3)
        self.assertEqual(data["created_ms"], self.state.ledger[0].timestamp_us // 1000)
        self.assertEqual(self.state.attempts, {"CreateSession": 1, "ListSessions": 1, "DeleteSession": 1})
        self.assertFalse(self.state.sessions)

    async def test_public_read_retries_are_bounded(self):
        self.state.plan("ListSessions", "before_reply", code=grpc.StatusCode.UNAVAILABLE)
        data = await self.driver("public-list")
        self.assertEqual((data["callbacks"], data["attempts"]), (1, 2))
        for attempt in (3, 4, 5):
            self.state.plan("ListSessions", "before_reply", attempt=attempt, code=grpc.StatusCode.UNAVAILABLE)
        data = await self.driver("public-list", 8, 14)
        self.assertEqual((data["callbacks"], data["attempts"]), (1, 3))
        self.assertEqual(self.state.attempts, {"ListSessions": 5})
        self.assertFalse(self.state.ledger)

    async def test_logical_auth_waits_count_requests_and_close_stream_owners(self):
        data = await self.driver("public-auth-waiters", token="")
        self.assertEqual(data['attempts'], 1)
        self.assertEqual(self.state.attempts, {'ListSessions': 1})
        waits = data['diagnostics']['metrics']['operation_auth_waits']
        self.assertEqual([row['kind'] for row in waits], [0, 1, 2])
        finite = waits[0]
        self.assertEqual((finite['current'], finite['high_water'], finite['started'], finite['completed']), (0, 2, 2, 2))
        self.assertEqual([(row['result'], row['count']) for row in finite['duration_ms_by_result']['entries']], [(0, 1), (10, 1)])
        for row in waits[1:]:
            self.assertEqual((row['current'], row['started'], row['completed']), (0, 0, 0))
        attempts = dict(self.state.attempts)
        for mode, kind, outcome in [('actor-auth-wait-close', 1, 10), ('actor-auth-wait-fail-close', 1, 8),
                                    ('join-auth-wait-close', 2, 10)]:
            data = await self.driver(mode, outcome, -1, token="", extra=['unissued-session'])
            row = data['diagnostics']['metrics']['operation_auth_waits'][kind]
            self.assertEqual((row['current'], row['high_water'], row['started'], row['completed']), (0, 1, 1, 1))
            self.assertEqual([(r['result'], r['count']) for r in row['duration_ms_by_result']['entries']], [(outcome, 1)])
            self.assertEqual(self.state.attempts, attempts)
            self.assertEqual(data['diagnostics']['metrics']['rpc_attempts'], 0)

    async def test_actor_open_deadline_while_starved_is_unauthenticated(self):
        attempts = dict(self.state.attempts)
        data = await self.driver("actor-auth-wait-open-deadline", 6, -1, token="", extra=['unissued-session'])
        self.assertEqual(data["result"], 6)
        self.assertEqual(self.state.attempts, attempts)

    async def test_public_auth_wait_refresh_failure_and_cancellation(self):
        def auth_metrics(data, reason, resolution, failure=None):
            snapshot = data['diagnostics']
            operation = snapshot['metrics']['operation_auth_waits'][0]
            self.assertEqual((operation['current'], operation['high_water'], operation['started'], operation['completed']), (0, 1, 1, 1))
            self.assertEqual([(r['result'], r['count']) for r in operation['duration_ms_by_result']['entries']],
                             [(0 if failure is None else 6, 1)])
            auth = snapshot['metrics']['auth']
            self.assertEqual(auth['blocked_endpoints'], 0)
            self.assertEqual(auth['blocked_endpoints_high_water'], 1)
            self.assertEqual(auth['stale_replacements'], 0)
            self.assertEqual(auth['challenges_by_reason'], [dict(reason=reason, count=1)])
            self.assertEqual(auth['failures_by_cause'], [] if failure is None else [dict(cause=failure, count=1)])
            histogram = auth['credential_wait_ms_by_resolution']
            self.assertEqual(histogram['bounds'], [1, 5, 10, 50, 100, 500, 1000, 5000])
            self.assertEqual(len(histogram['entries']), 1)
            row = histogram['entries'][0]
            self.assertEqual((row['resolution'], row['count']), (resolution, 1))
            self.assertEqual(len(row['counts']), 9)
            self.assertEqual(sum(row['counts']), 1)
            self.assertIs(type(row['sum_ms']), int)
            self.assertGreaterEqual(row['sum_ms'], 0)
            if failure is not None:
                failures = [r for r in snapshot['recent_records'] if r['event_name'] == 'auth_failed']
                self.assertEqual(len(failures), 1)
                self.assertEqual(failures[0]['fields']['auth_failure_cause'], failure)
            for secret in ('synthetic-writer', 'synthetic-invalid', 'Bearer', self.address):
                self.assertNotIn(secret, json.dumps(snapshot))
        data = await self.driver("public-auth-missing", token="")
        self.assertEqual(data["attempts"], 1)
        auth_metrics(data, 1, 0)
        data = await self.driver("public-auth-rejected", token="synthetic-invalid")
        self.assertEqual(data["attempts"], 2)
        auth_metrics(data, 3, 0)
        attempts = dict(self.state.attempts)
        data = await self.driver("public-auth-fail", 6, -1, token="")
        self.assertEqual(data["attempts"], 0)
        auth_metrics(data, 1, 1, 0)
        data = await self.driver("public-auth-timeout", 6, -1, token="")
        self.assertEqual(data["attempts"], 0)
        auth_metrics(data, 1, 1, 1)
        # public-deadline-starved selects a 1 ms request deadline; it
        # precedes the 30000 ms auth wait and no host refreshes: credential
        # starvation ends with the auth error.
        data = await self.driver("public-deadline-starved", 6, -1, token="")
        self.assertEqual((data["attempts"], data["callbacks"]), (0, 1))
        data = await self.driver("public-cancel-missing", 10, -1, token="")
        self.assertEqual((data["attempts"], data["callbacks"]), (0, 0))
        self.assertEqual(self.state.attempts, attempts)

    async def test_metadata_deadline_expired_before_start_while_starved_is_unauthenticated(self):
        session = await self.create_layer()
        await self.driver("metadata-starved-expired-scope", 6, -1, extra=[session], token="")
        self.assertEqual(self.state.attempts.get("StatLayer", 0), 0)

    async def test_public_mutation_uncertainty_never_retries(self):
        for boundary, commits in (("before", 0), ("after_commit", 1)):
            self.state.plan("CreateSession", boundary, code=grpc.StatusCode.UNAVAILABLE)
            data = await self.driver("public-create", 19, 14)
            self.assertEqual(data["attempts"], 1)
            self.assertEqual(len(self.state.ledger), commits)
        self.assertEqual(self.state.attempts, {"CreateSession": 2})

    async def test_public_create_session_rejection_is_definitive(self):
        # Design 5.6: a definitive rejection keeps its typed result and gRPC
        # code; every other failure may have committed and is uncertain.
        # The reference service scripts each code before execution.
        for index, (code, result) in enumerate(DEFINITIVE_MUTATION_CODES + UNCERTAIN_MUTATION_CODES, 1):
            self.state.plan("CreateSession", "before", code=code)
            data = await self.driver("public-create", result, code.value[0])
            self.assertEqual(data["attempts"], 1)
            self.assertEqual(self.state.attempts, {"CreateSession": index})
        self.assertEqual(self.state.ledger, [])

    async def test_public_create_survives_failed_listing_without_recreation(self):
        self.state.plan("ListSessions", "before_reply", code=grpc.StatusCode.PERMISSION_DENIED)
        data = await self.driver("public-create-list-failure", 7, 7)
        self.assertEqual(data["callbacks"], 2)
        self.assertEqual(self.state.attempts, {"CreateSession": 1, "ListSessions": 1})
        self.assertEqual(len(self.state.sessions), 1)

    async def test_public_issued_mutation_cancel_and_deadline(self):
        fault = self.state.plan("CreateSession", "after_commit", pause=True)
        data = await self.driver("public-cancel-create", 19, 1, fault=fault)
        self.assertEqual((data["attempts"], data["callbacks"]), (1, 0))
        self.state.plan("CreateSession", "after_commit", pause=True)
        data = await self.driver("public-deadline-create", 19, 4)
        self.assertEqual(data["attempts"], 1)
        self.assertEqual(self.state.attempts, {"CreateSession": 2})
        self.assertEqual(len(self.state.ledger), 2)

    async def test_public_read_deadline_is_not_cancellation(self):
        self.state.plan("ListSessions", "before_reply", pause=True)
        data = await self.driver("public-deadline-list", 9, 4)
        self.assertEqual((data["callbacks"], data["attempts"]), (1, 1))
        self.assertEqual(self.state.attempts, {"ListSessions": 1})

    async def test_public_malformed_created_identity_preserves_uncertainty(self):
        from unittest.mock import patch
        from service import LiveService
        original = LiveService.CreateSession
        async def malformed(service, request, context):
            response = await original(service, request, context)
            response.session_id = ""
            return response
        # The fixture commits normally but deliberately corrupts the successful
        # response at the server boundary. No production parser/oracle is shared.
        with patch.object(LiveService, "CreateSession", malformed):
            server = ReferenceServer(self.state, "malformed")
        self.servers.append(server)
        address = await server.start()
        data = await self.driver("public-create", 19, 0, address=address)
        self.assertEqual((data["callbacks"], data["attempts"]), (1, 1))
        self.assertEqual(self.state.attempts, {"CreateSession": 1})
        self.assertEqual(len(self.state.ledger), 1)

    async def test_public_layer_stat_delete(self):
        session = await self.create_layer()
        data = await self.driver("public-stat", extra=[session])
        commits = self.state.sessions[session].layers["root.live"].commits
        self.assertEqual(data["created_ms"], commits[0].timestamp_us // 1000)
        self.assertEqual(data["modified_ms"], commits[-1].timestamp_us // 1000)
        await self.driver("public-delete-layer", extra=[session])
        await self.driver("public-stat", 4, 5, extra=[session])
        self.assertEqual(self.state.attempts["DeleteLayer"], 1)

    async def test_public_layer_stat_preserves_backward_server_clock(self):
        created_us = 1700000001456789
        modified_us = 1700000000123456
        self.state.clock = lambda: created_us
        session = await self.create_layer()
        self.state.clock = lambda: modified_us
        response = await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=session, layer_name="root.live",
                object_id=1, client_id="clock-rollback-peer")),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=b"clock-rollback-edit"))
        ]), metadata=WRITER, timeout=5)
        self.assertEqual(response.sequence_num, 2)
        self.assertEqual(response.server_timestamp_microseconds_since_epoch, modified_us)
        data = await self.driver("public-stat", extra=[session])
        self.assertEqual((data["result"], data["callbacks"], data["attempts"]), (0, 1, 1))
        self.assertEqual((data["created_ms"], data["modified_ms"]),
                         (created_us // 1000, modified_us // 1000))
        self.assertLess(data["modified_ms"], data["created_ms"])
        self.assertEqual(self.state.attempts,
                         {"CreateSession": 1, "CreateLayer": 1, "UpdateLayer": 1, "StatLayer": 1})
        commits = self.state.sessions[session].layers["root.live"].commits
        self.assertEqual([(c.sequence, c.timestamp_us, c.payload) for c in commits],
                         [(1, created_us, b"opaque-base"), (2, modified_us, b"clock-rollback-edit")])

    async def test_public_malformed_metadata_is_atomic_failure(self):
        from copy import deepcopy
        reply = await self.stub.CreateSession(pb.CreateSessionRequest(path="public/base.usda", name="fixture"),
                                              metadata=WRITER, timeout=5)
        session = self.state.sessions[reply.session_id]
        session.name = "bad\x00metadata"
        data = await self.driver("public-list", 11, 0)
        self.assertEqual((data["count"], data["attempts"]), (0, 1))
        session.name = "valid"
        self.state.sessions["duplicate-identity"] = deepcopy(session)
        data = await self.driver("public-list", 11, 0)
        self.assertEqual((data["count"], data["attempts"]), (0, 1))
        self.assertEqual(self.state.attempts, {"CreateSession": 1, "ListSessions": 2})

    async def test_public_finite_refuses_locally_when_the_endpoint_budget_cannot_hold_a_response(self):
        started = self.service_rpc_count('ListSessions')
        result = await self.driver('public-finite-budget', 13, -1)
        self.assertEqual(result['result'], 13, result)  # RESOURCE_EXHAUSTED before issuance
        self.assertEqual(result['rpcs_started'], 0, result)
        self.assertEqual(self.service_rpc_count('ListSessions'), started)

    async def test_public_finite_response_above_its_allowance_is_resource_exhausted(self):
        from unittest.mock import patch
        from service import LiveService
        original = LiveService.ListSessions
        async def oversized(service, request, context):
            response = await original(service, request, context)
            entry = pb.Session(session_id="oversized", name="n" * 60000, created_by_user_name="writer")
            while response.ByteSize() <= 1 << 20:
                response.sessions.append(entry)
            return response
        # ListSessions reserves and is capped at 1 MiB under the default update limit.
        with patch.object(LiveService, "ListSessions", oversized):
            server = ReferenceServer(self.state, "oversized")
        self.servers.append(server)
        address = await server.start()
        data = await self.driver("public-list", 13, 8, address=address)
        self.assertEqual((data["result"], data["grpc_status"]), (13, 8), data)
        self.assertEqual((data["rpcs_started"], data["attempts"], data["callbacks"]), (1, 1, 1), data)
        self.assertEqual(self.service_rpc_count("ListSessions"), 1)

    @staticmethod
    def patterned(size, salt):
        # Independently construct 64 KiB bands. Adjacent 2 MiB wire chunks
        # differ, exposing a duplicated or reordered full chunk.
        bands = [bytes((index * 31 + band * 13 + salt) & 255 for index in range(256)) * 256
                 for band in range((size + 65535) // 65536)]
        return b"".join(bands)[:size]

    async def layer_session(self):
        return (await self.stub.CreateSession(pb.CreateSessionRequest(path="stream/base", name="fixture"),
                                             metadata=WRITER, timeout=5)).session_id

    async def test_layer_stream_multichunk_order_and_exact_payload(self):
        from unittest.mock import patch
        from service import LiveService
        session = await self.layer_session()
        original = LiveService.chunks
        transfers = []
        async def traced(service, incoming, context):
            frames = []
            transfers.append(frames)
            async def observed():
                async for message in incoming:
                    kind = message.WhichOneof("message_type")
                    frames.append([kind, len(message.chunk.data) if kind == "chunk" else 0])
                    yield message
            return await original(service, observed(), context)
        with patch.object(LiveService, "chunks", traced):
            data = await self.driver("layer-roundtrip", extra=[session])
        (self.artifact / "layer-frames.json").write_text(json.dumps(transfers) + "\n")
        self.assertEqual(transfers, [
            [["header", 0], ["chunk", 2097152], ["chunk", 2097152], ["chunk", 37]],
            [["header", 0], ["chunk", 2097152], ["chunk", 19]]])
        commits = self.state.sessions[session].layers["root.live"].commits
        expected = self.patterned(4194341, 7)
        self.assertNotEqual(expected[:2097152], expected[2097152:4194304])
        self.assertEqual([c.payload for c in commits], [expected, self.patterned(2097171, 23)])
        self.assertEqual([c.sequence for c in commits], [1, 2])
        self.assertEqual(commits[1].writer, data["writer"])
        self.assertEqual(commits[1].client_timestamp_us, 123456789)
        self.assertEqual((data["object_id"], data["sequence"], data["callbacks"]), (1, 2, 2))
        self.assertEqual(self.state.attempts, {"CreateSession": 1, "CreateLayer": 1, "UpdateLayer": 1})

    async def test_layer_stream_empty_and_exact_limit_payloads(self):
        session = await self.layer_session()
        await self.driver("layer-empty", extra=[session])
        self.assertEqual(self.state.ledger[-1].payload, b"")
        data = await self.driver("layer-exact-limit", extra=[session])
        self.assertEqual(data["object_id"], 2)
        self.assertEqual(self.state.ledger[-1].payload, self.patterned(8, 7))
        self.assertEqual(self.state.attempts, {"CreateSession": 1, "CreateLayer": 2})

    async def test_layer_stream_local_rejection_and_missing_auth_issue_nothing(self):
        session = await self.layer_session()
        await self.driver("layer-local-reject", 13, -1, extra=[session])
        await self.driver("layer-missing-auth", 6, -1, extra=[session], token="")
        self.assertEqual(self.state.attempts, {"CreateSession": 1})
        self.assertEqual(len(self.state.ledger), 1)

    async def test_layer_stream_create_uncertainty_without_replay(self):
        session = await self.layer_session()
        for boundary, count in (("before", 1), ("after_commit", 2)):
            self.state.plan("CreateLayer", boundary, code=grpc.StatusCode.UNAVAILABLE)
            await self.driver("layer-create", 19, 14, extra=[session])
            self.assertEqual(len(self.state.ledger), count)
        self.assertEqual(self.state.attempts, {"CreateSession": 1, "CreateLayer": 2})
        self.assertEqual(self.state.ledger[-1].payload, self.patterned(4194341, 7))

    async def test_layer_stream_update_uncertainty_without_replay(self):
        session = await self.create_layer()
        for boundary, count in (("before", 2), ("after_commit", 3)):
            self.state.plan("UpdateLayer", boundary, code=grpc.StatusCode.UNAVAILABLE)
            await self.driver("layer-update", 19, 14, extra=[session])
            self.assertEqual(len(self.state.ledger), count)
        self.assertEqual(self.state.attempts["UpdateLayer"], 2)
        self.assertEqual(self.state.ledger[-1].sequence, 2)
        self.assertEqual(self.state.ledger[-1].payload, self.patterned(4194341, 7))

    async def test_layer_stream_update_rejection_is_definitive(self):
        session = await self.create_layer()
        committed = len(self.state.ledger)
        for index, (code, result) in enumerate(DEFINITIVE_MUTATION_CODES + UNCERTAIN_MUTATION_CODES, 1):
            self.state.plan("UpdateLayer", "before", code=code)
            await self.driver("layer-update", result, code.value[0], extra=[session])
            self.assertEqual(self.state.attempts["UpdateLayer"], index)
        self.assertEqual(len(self.state.ledger), committed)

    async def test_layer_stream_auth_rejections_are_definitive(self):
        session = await self.create_layer()
        await self.driver("layer-create", 6, 16, extra=[session], token="synthetic-invalid")
        await self.driver("layer-update", 7, 7, extra=[session], token="synthetic-reader")
        self.assertEqual(len(self.state.ledger), 2)
        self.assertEqual(self.state.attempts["CreateLayer"], 2)
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)

    async def test_layer_stream_cancellation_before_and_after_commit(self):
        session = await self.create_layer()
        for method, mode in (("CreateLayer", "layer-cancel-create"), ("UpdateLayer", "layer-cancel-update")):
            for boundary in ("before", "after_commit"):
                # Create's intentional replacement changes the object. Restore
                # fixture object 1 in a fresh session for each update scenario.
                if method == "UpdateLayer":
                    session = await self.create_layer()
                before = len(self.state.ledger)
                fault = self.state.plan(method, boundary, pause=True)
                data = await self.driver(mode, 19, 1, extra=[session], fault=fault)
                self.assertEqual(data["callbacks"], 0)
                self.assertEqual(len(self.state.ledger), before + (boundary == "after_commit"))

    async def test_layer_stream_deadline_after_commit(self):
        session = await self.create_layer()
        self.state.plan("UpdateLayer", "after_commit", pause=True)
        await self.driver("layer-deadline-update", 19, 4, extra=[session])
        self.assertEqual(len(self.state.ledger), 3)
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)

    async def test_layer_stream_callback_retains_charged_buffers(self):
        session = await self.layer_session()
        data = await self.driver("layer-held-callback", extra=[session])
        self.assertEqual(data["callbacks"], 1)
        self.assertEqual(self.state.attempts, {"CreateSession": 1, "CreateLayer": 1})
        self.assertEqual(list(self.state.sessions[session].layers), ["root.live"])

    async def test_layer_stream_byte_budget_reclaims_after_cancel(self):
        session = await self.layer_session()
        fault = self.state.plan("CreateLayer", "before", pause=True)
        await self.driver("layer-capacity", 19, 1, extra=[session], fault=fault)
        self.assertEqual(self.state.attempts, {"CreateSession": 1, "CreateLayer": 2})
        self.assertEqual(list(self.state.sessions[session].layers), ["after.live"])
        self.assertEqual(self.state.ledger[-1].payload, b"after-cancel")

    async def test_layer_stream_malformed_successful_acknowledgments(self):
        from unittest.mock import patch
        from service import LiveService
        session = await self.create_layer()
        create, update = LiveService.CreateLayer, LiveService.UpdateLayer
        async def malformed_create(service, incoming, context):
            response = await create(service, incoming, context)
            response.object_id = 0
            return response
        async def malformed_update(service, incoming, context):
            response = await update(service, incoming, context)
            response.sequence_num = 0
            return response
        with patch.multiple(LiveService, CreateLayer=malformed_create, UpdateLayer=malformed_update):
            server = ReferenceServer(self.state, "malformed-write")
        self.servers.append(server)
        address = await server.start()
        await self.driver("layer-update", 19, 0, extra=[session], address=address)
        await self.driver("layer-create", 19, 0, extra=[session], address=address)
        self.assertEqual(len(self.state.ledger), 4)
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.assertEqual(self.state.attempts["CreateLayer"], 2)

    async def read_script(self, frames, *, code=None, close=False, pause=None, hold=None):
        from unittest.mock import patch
        from service import LiveService
        script = {"frames": [{"kind": frame.WhichOneof("response_type"),
                  "object": frame.header.object_id, "sequence": frame.header.sequence_num,
                  "size": frame.header.total_size, "chunk_bytes": len(frame.chunk.data)} for frame in frames],
                  "code": code.name if code else "OK" if close else "hold", "sent": 0}
        if not hasattr(self, "read_scripts"):
            self.read_scripts = []
        self.read_scripts.append(script)
        async def stream(service, request, context):
            _, attempt = await service.begin("SubscribeReadLayer", context)
            if hold:  # The stream is open and sends nothing until the test sets `hold`.
                await hold.wait()
            for frame in frames:
                script["sent"] += 1
                yield frame
            if pause:
                await service.state.boundary("SubscribeReadLayer", attempt, "before_reply", context, service.backend)
            if code:
                await context.abort(code, "synthetic read interruption")
            if not close:
                await asyncio.Future()  # Cancelled by the native client's termination.
        with patch.object(LiveService, "SubscribeReadLayer", stream):
            server = ReferenceServer(self.state, "scripted-read")
        self.servers.append(server)
        return await server.start()

    @staticmethod
    def read_header(size, sequence=1, object_id=1):
        return pb.SubscribeReadLayerResponse(header=pb.SubscribeReadLayerHeader(
            object_id=object_id, sequence_num=sequence, total_size=size))

    @staticmethod
    def read_chunk(data):
        return pb.SubscribeReadLayerResponse(chunk=pb.SubscribeReadLayerChunk(data=data))

    async def test_layer_read_native_exact_snapshot_and_delta(self):
        import hashlib
        session = await self.layer_session()
        base, delta = self.patterned(4194341, 7), self.patterned(2097171, 23)
        async def create():
            yield pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=session, layer_name="root.live"))
            for offset in range(0, len(base), 65536):
                yield pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=base[offset:offset+65536]))
        async def update():
            yield pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=session, layer_name="root.live", object_id=1, client_id="independent-writer"))
            for offset in range(0, len(delta), 65536):
                yield pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=delta[offset:offset+65536]))
        await self.stub.CreateLayer(create(), metadata=WRITER, timeout=5)
        await self.stub.UpdateLayer(update(), metadata=WRITER, timeout=5)
        data = await self.driver("read-roundtrip", 10, 1, extra=[session])
        content, caught = data["updates"][:2], data["updates"][2:]
        self.assertEqual([u["sha256"] for u in content], [hashlib.sha256(base).hexdigest(), hashlib.sha256(delta).hexdigest()])
        self.assertEqual([u["sequence"] for u in content], [1, 2])
        self.assertTrue(all(u["object"] == 1 and not u["caught"] for u in content))
        self.assertEqual([{k: u[k] for k in ("object", "sequence", "size", "caught")} for u in caught],
                         [{"object": 1, "sequence": 2, "size": 0, "caught": True}])
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        self.assertEqual(data["closed"], 0)

    async def test_layer_read_caughtup_is_separate_from_content(self):
        session = await self.create_layer()
        data = await self.driver("read-caughtup", 10, 1, extra=[session])
        self.assertEqual(len(data["updates"]), 1)
        self.assertEqual({k: data["updates"][0][k] for k in ("object", "sequence", "size", "caught")},
                         {"object": 1, "sequence": 1, "size": 0, "caught": True})
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)

    async def test_layer_read_retention_and_callback_cancellation(self):
        session = await self.create_layer()
        messages = [pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(
            session_id=session, layer_name="root.live", object_id=1, client_id="independent-writer")),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=b"distinct-retained-delta"))]
        await self.stub.UpdateLayer(iter(messages), metadata=WRITER, timeout=5)
        for mode in ("read-retain", "read-block", "read-self-cancel"):
            data = await self.driver(mode, 10, 1, extra=[session])
            self.assertEqual([u["caught"] for u in data["updates"]], [False, False, True] if mode == "read-retain" else [False])
            self.assertEqual(data["closed"], 0)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 3)
        self.assertEqual(self.state.attempts["Ping"], 1)

    async def test_layer_read_callback_failure_is_terminal(self):
        session = await self.create_layer()
        data = await self.driver("read-throw", 20, 1, extra=[session])
        self.assertEqual((len(data["updates"]), data["closed"]), (1, 1))
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)

    async def test_layer_read_malformed_frames_never_publish_partial_content(self):
        cases = [
            [self.read_chunk(b"orphan")],
            [pb.SubscribeReadLayerResponse()],
            [self.read_header(2), self.read_header(2)],
            [self.read_header(1), self.read_chunk(b"overflow")],
            [self.read_header(1, object_id=0)],
            [self.read_header(1, sequence=0)],
        ]
        for frames in cases:
            address = await self.read_script(frames)
            data = await self.driver("read-error", 11, 1, extra=["scripted"], address=address)
            self.assertEqual(data["updates"], [])
            self.assertEqual(data["closed"], 1)
        self.assertEqual(self.state.attempts, {"SubscribeReadLayer": len(cases)})
        self.assertFalse(self.state.ledger)

    async def test_layer_read_quota_and_wire_message_limit(self):
        address = await self.read_script([self.read_header(2**64 - 1)])
        data = await self.driver("read-error", 13, 1, extra=["scripted"], address=address)
        self.assertEqual(data["updates"], [])
        # Fits the general 6 MiB channel limit but exceeds this RPC's 2 MiB
        # frame allowance, proving the method-specific limit is actually active.
        address = await self.read_script([self.read_header(3 * 1024 * 1024), self.read_chunk(b"x" * (3 * 1024 * 1024))])
        # The pinned native/Python SDK pair returns CANCELLED here, not
        # RESOURCE_EXHAUSTED. Preserve the observed raw status; never infer a
        # quota cause by parsing arbitrary transport error strings.
        data = await self.driver("read-error", 10, 1, extra=["scripted"], address=address)
        self.assertEqual(data["updates"], [])
        self.assertEqual(data["closed"], 1)
        # A single frame at the supported chunk boundary still delivers exactly.
        import hashlib
        payload = self.patterned(2 * 1024 * 1024, 31)
        address = await self.read_script([self.read_header(len(payload), object_id=2), self.read_chunk(payload)])
        data = await self.driver("read-adopt", 10, 1, extra=["scripted"], address=address)
        self.assertEqual(len(data["updates"]), 1)
        self.assertEqual(data["updates"][0]["sha256"], hashlib.sha256(payload).hexdigest())
        # A read update is bounded by the endpoint's pending bytes, not the write limit: the service's catch-up
        # snapshot of a large layer exceeds any single upload.
        payload = self.patterned(6 * 1024 * 1024 + 512 * 1024, 37)
        chunks = [self.read_chunk(payload[i:i + 2 * 1024 * 1024]) for i in range(0, len(payload), 2 * 1024 * 1024)]
        address = await self.read_script([self.read_header(len(payload), object_id=2)] + chunks)
        data = await self.driver("read-adopt-large", 10, 1, extra=["scripted"], address=address)
        self.assertEqual([u["sha256"] for u in data["updates"]], [hashlib.sha256(payload).hexdigest()])
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 4)

    async def test_layer_read_truncation_and_transport_interruption_are_distinct(self):
        frames = [self.read_header(2), self.read_chunk(b"x")]
        address = await self.read_script(frames, close=True)
        data = await self.driver("read-error", 11, 0, extra=["scripted"], address=address)
        self.assertEqual(data["updates"], [])
        address = await self.read_script(frames, code=grpc.StatusCode.UNAVAILABLE)
        data = await self.driver("read-error", 8, 14, extra=["scripted"], address=address)
        self.assertEqual(data["updates"], [])
        address = await self.read_script([self.read_header(1), self.read_chunk(b"x")], close=True)
        data = await self.driver("read-error", 8, 0, extra=["scripted"], address=address)
        self.assertEqual(len(data["updates"]), 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 3)

    async def test_layer_read_duplicate_and_regressing_sequences(self):
        for suffix in ([self.read_header(1, sequence=1)],
                       [self.read_header(0, sequence=4), self.read_header(1, sequence=3)]):
            address = await self.read_script([self.read_header(1), self.read_chunk(b"x"), *suffix])
            data = await self.driver("read-error", 11, 1, extra=["scripted"], address=address)
            self.assertEqual(data["updates"][0]["sequence"], 1)
            self.assertEqual(len(data["updates"]), 1 if len(suffix) == 1 else 2)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)

    async def test_layer_read_object_replacement_policy(self):
        frames = [self.read_header(1, object_id=2), self.read_chunk(b"x")]
        address = await self.read_script(frames)
        data = await self.driver("read-expected-object", 16, 1, extra=["scripted"], address=address)
        self.assertEqual(data["updates"], [])
        data = await self.driver("read-adopt", 10, 1, extra=["scripted"], address=address)
        self.assertEqual([u["object"] for u in data["updates"]], [2])
        address = await self.read_script([self.read_header(1), self.read_chunk(b"x"), *frames])
        data = await self.driver("read-error", 16, 1, extra=["scripted"], address=address)
        self.assertEqual([u["object"] for u in data["updates"]], [1])
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 3)

    async def test_layer_read_retained_content_enforces_endpoint_budget(self):
        frames = [self.read_header(4 * 1024 * 1024)]
        frames += [self.read_chunk(b"a" * (1024 * 1024))] * 4
        frames += [self.read_header(2 * 1024 * 1024, sequence=2)]
        address = await self.read_script(frames)
        data = await self.driver("read-backpressure", 13, 1, extra=["scripted"], address=address)
        self.assertEqual([u["size"] for u in data["updates"]], [4 * 1024 * 1024])
        self.assertEqual(data["closed"], 1)

    async def actor_retained_flood(self):
        from unittest.mock import patch
        from service import LiveService
        peers = 9
        frames = asyncio.Queue()
        cancelled = asyncio.Event()
        script = {"kind": "held-local-publication-flood", "sequences": [], "cancelled": False}
        self.read_scripts = [script]

        async def stream(service, request, context):
            await service.begin("SubscribeReadLayer", context)
            self.assertEqual((request.object_id, request.sequence_num), (1, 1))
            try:
                yield self.read_header(0, sequence=1)
                while True:
                    sequence = await frames.get()
                    script["sequences"].append(sequence)
                    yield self.read_header(1024, sequence=sequence)
                    yield self.read_chunk(bytes([ord('a') + sequence % 26]) * 1024)
            finally:
                script["cancelled"] = True
                cancelled.set()

        with patch.object(LiveService, "SubscribeReadLayer", stream):
            server = ReferenceServer(self.state, "held-publication-flood")
        self.servers.append(server)
        address = await server.start()

        def release(sequence):
            async def enqueue():
                self.assertFalse(cancelled.is_set())
                self.assertNotIn("UpdateLayer", self.state.attempts)
                await frames.put(sequence)
            return enqueue

        async def exhausted():
            await cancelled.wait()
            self.assertEqual(script["sequences"], list(range(2, peers + 3)))
            self.assertNotIn("UpdateLayer", self.state.attempts)

        # Per-update coverage acknowledgements isolate the byte limit from
        # ingress pressure without delay or host publication retiring work.
        steps = [(b"actor-flood-ready", release(2))]
        steps += [(f"actor-flood-retained-{sequence}".encode(), release(sequence + 1))
                  for sequence in range(2, peers + 2)]
        steps += [(b"actor-flood-failed", exhausted)]
        data = await self.driver("actor-flood-bytes", 13,
                                 extra=["scripted", self.address], address=address, steps=steps)
        self.assertEqual((data['result'], data['local_count'], data['local_bytes']), (13, 3, 21))
        self.assertEqual((data['peers'], data['covered'], data['applied']), (peers, peers + 1, 1))
        self.assertLessEqual(data['retained_bytes'], data['peak_bytes'])
        self.assertLessEqual(data['peak_bytes'], data['budget'])
        self.assertEqual(data['byte_rejections'], 1)
        self.assertEqual((data['final_bytes'], data['final_slots']), (0, 0))
        # Two explicit Pings of the other endpoint, and at most the actor's one that learns its capabilities.
        self.assertLessEqual(self.state.attempts.pop("Ping") - 2, 1)
        self.assertEqual(self.state.attempts, {"SubscribeReadLayer": 1})

    async def test_actor_held_publication_byte_exhaustion_preserves_local_intents(self):
        await self.actor_retained_flood()

    async def actor_read_pause(self, mode):
        from unittest.mock import patch
        from service import LiveService
        frames, cancelled, resumed = asyncio.Queue(), asyncio.Event(), asyncio.Event()
        service_end = mode == "actor-read-pause-service-end"
        # The service-ended stream carries peers 2..33; the reconnect carries 34..37.
        split = 34 if service_end else 38
        cursors = []

        def frame(sequence):
            return [self.read_header(1024, sequence=sequence),
                    self.read_chunk(bytes([ord('a') + sequence % 26]) * 1024)]

        async def stream(service, request, context):
            await service.begin("SubscribeReadLayer", context)
            cursors.append((request.object_id, request.sequence_num))
            if len(cursors) > 1:
                # The reconnect: caught up at the requested cursor, then the rest.
                try:
                    yield self.read_header(0, sequence=request.sequence_num)
                    for sequence in range(request.sequence_num + 1, 38):
                        for response in frame(sequence):
                            yield response
                    await asyncio.Event().wait()  # Open until the client closes.
                finally:
                    resumed.set()
                return
            try:
                yield self.read_header(0, sequence=1)
                while True:
                    sequence = await frames.get()
                    if sequence is None:
                        await context.abort(grpc.StatusCode.UNAVAILABLE, "service ended the paused reader")
                    for response in frame(sequence):
                        yield response
            finally:
                cancelled.set()
        with patch.object(LiveService, "SubscribeReadLayer", stream):
            server = ReferenceServer(self.state, "read-pause")
        self.servers.append(server)
        address = await server.start()

        async def burst():
            for sequence in range(2, split):  # At least 32 peers, above the high-water mark of 28.
                await frames.put(sequence)

        async def paused():
            self.assertFalse(cancelled.is_set())  # Paused, not failed or reconnected.
            if service_end:
                # gRPC's 64 KB initial HTTP/2 receive window absorbs peers 2..33 (with
                # their 5-byte message prefixes) while the client is parked, so the
                # service writes them all and ends the stream without blocking.
                self.assertLess(sum(r.ByteSize() + 5 for s in range(2, split) for r in frame(s)), 65535)
                # End the stream while the client's read is parked.
                await frames.put(None)
                await asyncio.wait_for(cancelled.wait(), 3)
                self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)

        async def failed():
            await asyncio.wait_for(cancelled.wait(), 3)  # Failure alone cancels the parked read.
        steps = [(b"actor-pause-ready", burst), (b"actor-read-paused", paused)]
        if mode == "actor-read-pause-terminal":
            steps.append((b"actor-read-failed", failed))
        data = await self.driver(mode, 0, extra=["scripted"], address=address, steps=steps)
        records = data["diagnostics"]["recent_records"]
        pauses = [r for r in records if r["event_name"] == "read_paused"]
        resumes = [r for r in records if r["event_name"] == "read_resumed"]
        if mode == "actor-read-pause-terminal":
            # A host failure ends the pause interval FAILED: the parked read is
            # cancelled, retained coverage is kept, and the read never resumes.
            self.assertEqual((data["terminal"], data["result"]), (1, 20), data)
            self.assertLess(data["covered"], 37)
            self.assertEqual(data["retained"], data["covered"] - 1)
            self.assertEqual((len(pauses), len(resumes)), (1, 0), records)
            self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
            return
        if service_end:
            # One reconnect, from the coverage frontier the ended stream reached.
            await asyncio.wait_for(resumed.wait(), 3)
            self.assertEqual(cursors, [(1, 1), (1, split - 1)])
            self.assertEqual(data["generations"], 2)
            self.assertEqual((data["applied"], data["covered"]), (37, 37))
            self.assertEqual((len(pauses), len(resumes)), (1, 1), records)
            self.assertLess(records.index(pauses[0]), records.index(resumes[0]))
            self.assertEqual(len([r for r in records if r["event_name"] == "stream_disconnected"]), 1, records)
            self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)
            return
        if mode == "actor-read-pause-close":
            # Close cancels the parked read; the interval ends CLOSED, unresumed.
            await asyncio.wait_for(cancelled.wait(), 3)
            self.assertEqual(data["closed"], 1)
            self.assertLess(data["covered"], 37)
            self.assertEqual((len(pauses), len(resumes)), (1, 0), records)
            self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
            return
        self.assertEqual((data["applied"], data["covered"]), (37, 37))
        self.assertEqual((len(pauses), len(resumes)), (1, 1), records)
        self.assertGreaterEqual(pauses[0]["fields"]["queue_depth"], 28)
        self.assertLessEqual(resumes[0]["fields"]["queue_depth"], 16)
        self.assertFalse([r for r in records if r["event_name"] == "stream_disconnected"])
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)

    async def actor_read_deadline(self, mode):
        from unittest.mock import patch
        from service import LiveService
        paced = mode == "actor-read-deadline-paced"
        frames, ended = asyncio.Queue(), asyncio.Event()
        cursors, stalls = [], []

        def frame(sequence):
            return [self.read_header(1024, sequence=sequence),
                    self.read_chunk(bytes([ord('a') + sequence % 26]) * 1024)]

        async def stream(service, request, context):
            await service.begin("SubscribeReadLayer", context)
            cursors.append((request.object_id, request.sequence_num))
            if len(cursors) > 1 and paced:
                # The resubscribed attempt: caught up at the requested cursor, then the rest.
                yield self.read_header(0, sequence=request.sequence_num)
                for sequence in range(request.sequence_num + 1, 38):
                    for response in frame(sequence):
                        yield response
                await asyncio.Event().wait()  # Open until the client closes.
            if len(cursors) > 1:
                # Accepted and never answered: only the client's cancellation ends it.
                stall = {"started": asyncio.get_running_loop().time(), "ended": None}
                stalls.append(stall)
                try:
                    await asyncio.Event().wait()
                finally:
                    stall["ended"] = asyncio.get_running_loop().time()
            yield self.read_header(0, sequence=1)
            while (sequence := await frames.get()) is not None:
                for response in frame(sequence):
                    yield response
            ended.set()
            await context.abort(grpc.StatusCode.UNAVAILABLE, "service ended the first read")

        with patch.object(LiveService, "SubscribeReadLayer", stream):
            server = ReferenceServer(self.state, "read-deadline")
        self.servers.append(server)
        address = await server.start()

        async def proceed():
            if paced:
                for sequence in range(2, 34):  # 32 peers, above the high-water mark of 28.
                    await frames.put(sequence)
            else:
                await frames.put(None)
                await asyncio.wait_for(ended.wait(), 3)
        data = await self.driver(mode, 0, extra=["scripted"], address=address,
                                 steps=[(b"actor-deadline-ready", proceed)])
        records = data["diagnostics"]["recent_records"]
        disconnects = [r["fields"]["result"] for r in records if r["event_name"] == "stream_disconnected"]
        return data, cursors, stalls, disconnects

    async def test_actor_stalled_read_attempts_end_at_the_real_first_response_deadline(self):
        # No test seam: each reconnect the service accepts and never answers ends TIMEOUT once its
        # initial_open_timeout_ms (1000 ms) passes, and counts; the third fails the layer UNAVAILABLE.
        data, cursors, stalls, disconnects = await self.actor_read_deadline("actor-read-deadline-stall")
        self.assertEqual((data["failed"], data["result"]), (1, 8), data)
        self.assertEqual(len(cursors), 4)
        self.assertEqual(len(stalls), 3)
        for stall in stalls:
            # The client cancelled each attempt, and not before its deadline.
            self.assertIsNotNone(stall["ended"])
            self.assertGreaterEqual(stall["ended"] - stall["started"], 0.9, stalls)
        self.assertEqual(disconnects, [8, 9, 9, 9])
        self.assertEqual(data["held_windows"], 0)

    async def test_actor_read_held_for_pacing_keeps_its_attempt_past_the_first_response_deadline(self):
        # The resubscribed attempt waits behind the pacing hold longer than its window; the window restarts
        # instead of cancelling the attempt, which resumes once the held application is released.
        data, cursors, stalls, disconnects = await self.actor_read_deadline("actor-read-deadline-paced")
        self.assertEqual((data["failed"], data["result"]), (0, 0), data)
        self.assertEqual((data["applied"], data["covered"], data["generations"]), (37, 37, 2), data)
        self.assertGreaterEqual(data["held_windows"], 1)
        self.assertEqual(len(cursors), 2)
        self.assertEqual(stalls, [])
        self.assertEqual(disconnects, [10])  # The resubscription's own cancellation; no TIMEOUT.

    async def test_actor_read_pauses_at_high_water_while_publication_is_held(self):
        await self.actor_read_pause("actor-read-pause")

    async def test_actor_read_pauses_at_high_water_while_decode_is_held(self):
        await self.actor_read_pause("actor-read-pause-ingress")

    async def test_actor_close_finishes_a_paused_read(self):
        await self.actor_read_pause("actor-read-pause-close")

    async def test_actor_paused_read_sees_a_service_end_after_resuming_and_reconnects_from_the_frontier(self):
        await self.actor_read_pause("actor-read-pause-service-end")

    async def test_actor_host_failure_while_paused_cancels_the_read_and_keeps_retained_work(self):
        await self.actor_read_pause("actor-read-pause-terminal")

    async def test_layer_read_cancel_before_header_and_partial_payload(self):
        session = await self.create_layer()
        fault = self.state.plan("SubscribeReadLayer", "before_reply", pause=True)
        data = await self.driver("read-cancel-before", 10, 1, extra=[session], fault=fault)
        self.assertEqual(data["updates"], [])
        fault = self.state.plan("SubscribeReadLayer", "before_reply", pause=True)
        address = await self.read_script([self.read_header(100), self.read_chunk(b"partial")], pause=fault)
        data = await self.driver("read-cancel-before", 10, 1, extra=["scripted"], address=address, fault=fault)
        self.assertEqual(data["updates"], [])

    async def test_layer_read_deadline_preserves_transport_status(self):
        for frames in ([], [self.read_header(100), self.read_chunk(b"partial")]):
            address = await self.read_script(frames)
            data = await self.driver("read-deadline", 9, 4, extra=["scripted"], address=address)
            self.assertEqual(data["updates"], [])
            self.assertEqual(data["closed"], 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)

    async def test_layer_read_auth_and_permission_statuses(self):
        session = await self.create_layer()
        data = await self.driver("read-error", 6, 16, extra=[session], token="synthetic-invalid")
        self.assertEqual(data["updates"], [])
        self.state.plan("SubscribeReadLayer", "before", code=grpc.StatusCode.PERMISSION_DENIED)
        data = await self.driver("read-error", 7, 7, extra=[session])
        self.assertEqual(data["updates"], [])
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)

    async def test_actor_host_failure_blocks_admission_application_and_opening(self):
        for mode, retained in (("actor-host-fault-admission", 0), ("actor-host-fault-queued", 1),
                               ("actor-host-fault-opening", 0), ("actor-host-fault-sync", 1)):
            session = await self.create_layer()
            data = await self.driver(mode, 20, extra=[session])
            self.assertEqual((data['result'], data['retained'], data['issued']), (20, retained, 0))
            commits = [item for item in self.state.ledger if item.session == session and item.operation == 'UpdateLayer']
            self.assertEqual(commits, [])
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), 0)

    async def test_actor_refused_admission_reports_its_release_only_after_a_contended_refusal(self):
        session = await self.create_layer()
        data = await self.driver("actor-release-notices", extra=[session])
        self.assertEqual(data, {"result": 0, "quiet_releases": 0, "quiet_retries": 0, "contended_releases": 1, "contended_retries": 1})
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), 0)

    async def test_actor_publishes_a_newer_status_when_the_host_buffer_empties_without_issuing(self):
        session = await self.create_layer()
        data = await self.driver("actor-buffer-emptied", extra=[session])
        self.assertEqual(data, {"result": 0, "advanced": True, "buffered": 0, "admitted": 0})
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), 0)

    async def test_actor_released_without_close_closes_itself(self):
        from unittest.mock import patch
        from service import LiveService
        session = await self.create_layer()
        subscribe = LiveService.SubscribeReadLayer
        cancelled = asyncio.Event()

        async def stream(service, request, context):
            try:
                async for frame in subscribe(service, request, context):
                    yield frame
            finally:
                cancelled.set()

        with patch.object(LiveService, "SubscribeReadLayer", stream):
            server = ReferenceServer(self.state, "actor-drop")
        self.servers.append(server)
        address = await server.start()

        async def read_cancelled():
            # The endpoint and process stay alive: only the release cancels it.
            await cancelled.wait()

        data = await self.driver("actor-drop-without-close", extra=[session], address=address,
                                 steps=[(b"actor-dropped", read_cancelled)])
        self.assertEqual(data, {"result": 0, "closed": True})
        self.assertEqual(self.state.attempts.get("SubscribeReadLayer"), 1)

    async def test_actor_issuance_uses_the_admission_reservation(self):
        session = await self.create_layer()
        data = await self.driver("actor-issue-reservation", extra=[session])
        self.assertEqual(data, {"result": 0, "refused": True, "failed_after_refusal": False, "failure": 0,
                                "issued": True, "retired": 2, "fit_includes_read": True})
        commits = [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]
        self.assertEqual([c.payload for c in commits], [b"a" * 1000, b"b" * 1000])
        self.assertEqual(self.state.attempts["UpdateLayer"], 2)

    async def test_actor_fifo_and_bootstrap(self):
        for mode in ("actor-roundtrip", "actor-bootstrap"):
            session = await self.create_layer()
            data = await self.driver(mode, extra=[session])
            expected = [{"sequence": i + 2, "bytes": f"local-{i}"} for i in range(3)]
            if mode == "actor-bootstrap":
                expected.insert(0, {"sequence": 1, "bytes": "opaque-base"})
            self.assertEqual(data["effects"], expected)
            self.assertEqual((data["applied"], data["retained"]), (4, 0))
            self.assertEqual(data["duplicate_rejected"], mode == "actor-roundtrip")
            self.assertEqual(data["closed_status"],
                             {"state": 9, "result": 0, "incident": 0, "pending": 0, "bytes": 0, "unconfirmed": False})
            commits = [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]
            self.assertEqual([c.payload for c in commits], [f"local-{i}".encode() for i in range(3)])
            self.assertEqual([c.sequence for c in commits], [2, 3, 4])
            self.assertEqual(len({c.writer for c in commits}), 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)
        self.assertEqual(self.state.attempts["UpdateLayer"], 6)

    async def test_actor_sync_application_timing_preserves_dispatch_error(self):
        session = await self.create_layer()
        data = await self.driver("actor-sync-invalid", 20, extra=[session])
        self.assertEqual((data["applied"], data["retained"]), (1, 1))
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)

    async def test_actor_async_publication_and_capture_retirement(self):
        session = await self.create_layer()
        data = await self.driver("actor-async-release", extra=[session])
        self.assertEqual((data["applied"], data["retained"]), (4, 0))
        self.assertEqual(data["effects"], [{"sequence": i + 2, "bytes": f"local-{i}"} for i in range(3)])
        commits = [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]
        self.assertEqual([(c.sequence, c.payload) for c in commits], [(i + 2, f"local-{i}".encode()) for i in range(3)])
        self.assertEqual(self.state.attempts["UpdateLayer"], 3)
        self.assertGreaterEqual(self.state.attempts["Ping"], 1)

    async def test_actor_async_host_self_close_waits_for_owner(self):
        session = await self.create_layer()
        data = await self.driver("actor-async-close", extra=[session])
        self.assertEqual((data["applied"], data["retained"]), (2, 0))
        self.assertEqual(data["effects"], [{"sequence": 2, "bytes": "local-0"}])
        commits = [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]
        self.assertEqual([(c.sequence, c.payload) for c in commits], [(2, b"local-0")])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.assertGreaterEqual(self.state.attempts["Ping"], 1)

    async def test_actor_async_close_suppresses_queued_publication(self):
        session = await self.create_layer()
        data = await self.driver("actor-async-cancel", 10, extra=[session])
        self.assertEqual((data["applied"], data["retained"]), (1, 1))
        self.assertEqual(data["effects"], [])
        self.assertEqual(data["closed_status"]["pending"], 1)
        self.assertIs(data["closed_status"]["unconfirmed"], True)
        commits = [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]
        self.assertEqual([(c.sequence, c.payload) for c in commits], [(2, b"local-0")])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)

    async def test_actor_async_publication_failure_retains_confirmed_work(self):
        session = await self.create_layer()
        data = await self.driver("actor-async-fail", 20, extra=[session])
        self.assertEqual((data["applied"], data["retained"]), (1, 1))
        self.assertEqual(data["effects"], [])
        self.assertEqual(data["closed_status"]["pending"], 1)
        self.assertIs(data["closed_status"]["unconfirmed"], True)
        commits = [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]
        self.assertEqual([(c.sequence, c.payload) for c in commits], [(2, b"local-0")])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)

    async def test_actor_async_completed_body_then_dispatch_throw_records_rejection(self):
        session = await self.create_layer()
        data = await self.driver("actor-async-run-throw", 20, extra=[session])
        self.assertEqual((data["applied"], data["retained"]), (1, 1))
        self.assertEqual(data["closed_status"]["pending"], 1)
        commits = [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]
        self.assertEqual([(c.sequence, c.payload) for c in commits], [(2, b"local-0")])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)

    async def test_actor_async_queue_then_throw_suppresses_publication(self):
        session = await self.create_layer()
        data = await self.driver("actor-async-queue-throw", 20, extra=[session])
        self.assertEqual((data["applied"], data["retained"]), (1, 1))
        self.assertEqual(data["effects"], [])
        self.assertEqual(data["closed_status"]["pending"], 1)
        self.assertIs(data["closed_status"]["unconfirmed"], True)
        commits = [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]
        self.assertEqual([(c.sequence, c.payload) for c in commits], [(2, b"local-0")])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)

    async def test_actor_terminal_close_waits_for_application_capture_retirement(self):
        session = await self.create_layer()
        data = await self.driver("actor-async-capture", 20, extra=[session])
        self.assertEqual((data["applied"], data["retained"], data["effects"]), (1, 1, []))
        commits = [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]
        self.assertEqual([(c.sequence, c.payload) for c in commits], [(2, b"local-0")])
        self.assertEqual(len({c.writer for c in commits}), 1)
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)

    async def local_publication(self, mode, result=0):
        session = await self.create_layer()
        fault = self.state.plan("UpdateLayer", "before", pause=True) if mode == "actor-local-release" else None

        def commits():
            return [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]

        async def no_issuance():
            self.assertEqual(self.state.attempts.get("UpdateLayer", 0), 0)
            self.assertEqual(commits(), [])

        async def issued_without_commit():
            await fault.reached.wait()
            self.assertEqual(self.state.attempts["UpdateLayer"], 1)
            self.assertEqual(commits(), [])
            fault.release.set()

        steps = [(b"local-publication-held", no_issuance),
                 (b"local-publication-capture-held", no_issuance)]
        if fault:
            steps.append((b"local-publication-awaiting-ack", issued_without_commit))
        data = await self.driver(mode, result, extra=[session], steps=steps)
        if fault:
            self.assertEqual((data["applied"], data["retained"]), (2, 0))
            self.assertEqual(data["effects"], [{"sequence": 2, "bytes": "local-0"}])
            self.assertEqual([(c.sequence, c.payload) for c in commits()], [(2, b"local-0")])
            self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        else:
            self.assertEqual((data["applied"], data["retained"]), (1, 1))
            self.assertEqual(data["effects"], [])
            self.assertEqual(data["closed_status"]["pending"], 1)
            self.assertIs(data["closed_status"]["unconfirmed"], True)
            await no_issuance()

    async def test_actor_local_publication_precedes_issuance_but_not_ack(self):
        await self.local_publication("actor-local-release")

    async def test_actor_local_publication_failure_retains_unissued_intent(self):
        await self.local_publication("actor-local-fail", 20)

    async def test_actor_local_publication_cancel_retains_unissued_intent(self):
        await self.local_publication("actor-local-cancel", 10)

    async def test_actor_local_and_remote_publication_share_one_slot(self):
        session = await self.create_layer()

        def commits():
            return [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]

        async def peer_during_local_hold():
            self.assertEqual(self.state.attempts.get("UpdateLayer", 0), 0)
            messages = [pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(
                session_id=session, layer_name="root.live", object_id=1, client_id="independent-writer")),
                pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=b"peer-2"))]
            await self.stub.UpdateLayer(iter(messages), metadata=WRITER, timeout=3)
            self.assertEqual([(c.sequence, c.payload) for c in commits()], [(2, b"peer-2")])

        async def only_peer_committed():
            self.assertEqual(self.state.attempts["UpdateLayer"], 1)
            self.assertEqual([(c.sequence, c.payload) for c in commits()], [(2, b"peer-2")])

        data = await self.driver("actor-local-shared", extra=[session], steps=[
            (b"local-publication-held", peer_during_local_hold),
            (b"local-publication-capture-held", only_peer_committed)])
        self.assertEqual((data["applied"], data["retained"]), (3, 0))
        self.assertEqual(data["effects"], [{"sequence": 2, "bytes": "peer-2"},
                                          {"sequence": 3, "bytes": "local-0"}])
        self.assertEqual([(c.sequence, c.payload) for c in commits()], [(2, b"peer-2"), (3, b"local-0")])
        self.assertNotEqual(commits()[0].writer, commits()[1].writer)
        self.assertEqual(self.state.attempts["UpdateLayer"], 2)

    async def jitter_actor_stream(self, session, *, held_ping=None):
        from unittest.mock import patch
        from service import LiveService

        async def stream(service, request, context):
            principal, _ = await service.begin("SubscribeReadLayer", context)
            self.assertEqual((request.object_id, request.sequence_num), (1, 0))
            # A held timing request must already occupy its slot before the
            # native side can become ready and admit its single local write.
            if held_ping:
                await asyncio.wait_for(held_ping.reached.wait(), 3)
            async with service.state.changed:
                _, layer = await service.layer(session, "root.live", principal, context)
                self.assertEqual(len(layer.commits), 1)
                peer = service.state.commit("UpdateLayer", session, layer="root.live", object_id=1,
                    sequence=2, payload=b"peer-delayed", writer="independent-peer")
                layer.commits.append(peer)
            # Both headers deliberately omit timestamp fields. The initial
            # snapshot bypasses holdback; the subsequent delta uses the exact
            # 30-second configured fallback, regardless of background Ping.
            yield self.read_header(len(b"opaque-base"), sequence=1)
            yield self.read_chunk(b"opaque-base")
            yield self.read_header(len(peer.payload), sequence=2)
            yield self.read_chunk(peer.payload)
            await asyncio.Future() # Client close must cancel this ongoing read.

        with patch.object(LiveService, "SubscribeReadLayer", stream):
            server = ReferenceServer(self.state, "actor-jitter")
        self.servers.append(server)
        return await server.start()

    async def test_actor_jitter_coverage_fifo_and_configuration_wake(self):
        session = await self.create_layer()
        address = await self.jitter_actor_stream(session)
        data = await self.driver("actor-jitter-release", extra=[session], address=address)
        self.assertEqual((data["generation"], data["applied"], data["retained"]), (1, 3, 0))
        self.assertEqual((data["immediate_status_checks"], data["acknowledged_pending_bytes"]),
                         (1, len(b"local-after-peer")))
        self.assertEqual((data["closed_status"]["pending"], data["closed_status"]["bytes"]), (0, 0))
        self.assertEqual(data["effects"], [{"sequence": 1, "bytes": "opaque-base"},
            {"sequence": 2, "bytes": "peer-delayed"}, {"sequence": 3, "bytes": "local-after-peer"}])
        commits = [c for c in self.state.ledger if c.operation == "UpdateLayer"]
        self.assertEqual([(c.sequence, c.payload) for c in commits], [(2, b"peer-delayed"), (3, b"local-after-peer")])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        self.assertGreaterEqual(self.state.attempts["Ping"], 1)

    async def test_actor_jitter_wait_retires_before_failure_close(self):
        session = await self.create_layer()
        address = await self.jitter_actor_stream(session)
        data = await self.driver("actor-jitter-fail", 20, extra=[session], address=address)
        self.assertEqual((data["applied"], data["retained"]), (1, 2))
        self.assertEqual(data["closed_status"]["pending"], 1)
        self.assertEqual(data["effects"], [{"sequence": 1, "bytes": "opaque-base"}])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)

    async def test_actor_jitter_wait_preempted_by_local_publication_then_resumes(self):
        session = await self.create_layer()
        address = await self.jitter_actor_stream(session)
        data = await self.driver("actor-jitter-preempt", extra=[session], address=address)
        self.assertEqual((data["applied"], data["retained"]), (3, 0))
        self.assertEqual(data["effects"], [{"sequence": 1, "bytes": "opaque-base"},
            {"sequence": 2, "bytes": "peer-delayed"}, {"sequence": 3, "bytes": "local-after-peer"}])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)

    async def test_actor_drain_capture_excludes_later_held_application(self):
        session = await self.create_layer()
        address = await self.jitter_actor_stream(session)
        data = await self.driver("actor-drain-capture", extra=[session], address=address)
        self.assertEqual((data["applied"], data["retained"]), (3, 0))
        self.assertEqual(data["effects"], [{"sequence": 1, "bytes": "opaque-base"},
            {"sequence": 2, "bytes": "peer-delayed"}, {"sequence": 3, "bytes": "local-after-peer"}])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)

    async def test_actor_drain_includes_changed_callback_after_work_retirement(self):
        session = await self.create_layer()
        data = await self.driver("actor-drain-changed", extra=[session])
        self.assertEqual((data["applied"], data["retained"]), (4, 0))
        self.assertEqual(data["effects"], [{"sequence": i + 2, "bytes": f"local-{i}"} for i in range(3)])
        self.assertEqual(self.state.attempts["UpdateLayer"], 3)

    async def test_actor_clean_idle_close_does_not_poison_drain(self):
        session = await self.create_layer()
        data = await self.driver("actor-drain-clean-close", extra=[session])
        self.assertEqual((data["generation"], data["applied"], data["retained"], data["effects"]), (1, 1, 0, []))
        self.assertEqual(data["closed_status"],
                         {"state": 9, "result": 0, "incident": 0, "pending": 0, "bytes": 0, "unconfirmed": False})
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        self.assertNotIn("UpdateLayer", self.state.attempts)
        # With smoothing disabled only the Ping that learns the endpoint's capabilities is sent.
        self.assertLessEqual(self.state.attempts.get("Ping", 0), 1)

    async def test_actor_close_during_jitter_retains_peer_and_acknowledged_intent(self):
        session = await self.create_layer()
        address = await self.jitter_actor_stream(session)
        data = await self.driver("actor-jitter-close", extra=[session], address=address)
        self.assertEqual((data["generation"], data["applied"], data["retained"]), (1, 1, 2))
        self.assertEqual(data["acknowledged_pending_bytes"], len(b"local-after-peer"))
        self.assertEqual((data["closed_status"]["pending"], data["closed_status"]["bytes"],
                          data["closed_status"]["unconfirmed"]), (1, len(b"local-after-peer"), True))
        self.assertEqual(data["effects"], [{"sequence": 1, "bytes": "opaque-base"}])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        self.assertGreaterEqual(self.state.attempts["Ping"], 2)

    async def test_actor_held_timing_ping_does_not_consume_write_capacity(self):
        session = await self.create_layer()
        ping = self.state.plan("Ping", "before_reply", pause=True)
        # Native timing deadline is 60s (> whole 12s driver bound), and the
        # ACK marker additionally asserts timingPending before releasing Ping.
        address = await self.jitter_actor_stream(session, held_ping=ping)
        data = await self.driver("actor-jitter-budget", extra=[session], address=address,
                                 fault=ping, release_on=b"actor-write-settled-with-held-ping\n")
        self.assertEqual((data["applied"], data["retained"]), (3, 0))
        self.assertEqual(data["effects"], [{"sequence": 1, "bytes": "opaque-base"},
            {"sequence": 2, "bytes": "peer-delayed"}, {"sequence": 3, "bytes": "local-after-peer"}])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)

    async def test_actor_quiet_reconnect_retires_cancelled_timing_ping(self):
        from unittest.mock import patch
        from service import LiveService
        session = await self.create_layer()
        old_ping = self.state.plan("Ping", "before_reply", attempt=1, pause=True)
        new_ping = self.state.plan("Ping", "before_reply", attempt=2)
        requests = []

        async def stream(service, request, context):
            await service.begin("SubscribeReadLayer", context)
            requests.append((request.object_id, request.sequence_num))
            if len(requests) == 1:
                yield self.read_header(0, sequence=1)
                await asyncio.wait_for(old_ping.reached.wait(), 3)
                await context.abort(grpc.StatusCode.UNAVAILABLE, "rotate while generation-one Ping is held")
            else:
                # No stream event can wake the actor until its old suppressed
                # callback has retired and the new generation samples its clock.
                # The bound is shorter than the ordinary ten-second cadence;
                # native Ping deadline is 60s, so cancellation must retire it.
                await asyncio.wait_for(new_ping.reached.wait(), 3)
                yield self.read_header(0, sequence=1)
                await asyncio.Future()

        with patch.object(LiveService, "SubscribeReadLayer", stream):
            server = ReferenceServer(self.state, "actor-jitter-reconnect")
        self.servers.append(server)
        address = await server.start()
        data = await self.driver("actor-jitter-reconnect", extra=[session], address=address,
                                 fault=old_ping, release_on=b"actor-new-generation-ping\n")
        self.assertEqual(requests, [(1, 1), (1, 1)])
        self.assertEqual((data["generation"], data["applied"], data["retained"], data["effects"]), (2, 1, 0, []))
        self.assertEqual(self.state.attempts["Ping"], 2)
        self.assertNotIn("UpdateLayer", self.state.attempts)

    async def test_actor_read_retry_releases_prequeued_writes(self):
        session = await self.create_layer()
        self.state.plan("SubscribeReadLayer", "before_reply", code=grpc.StatusCode.UNAVAILABLE)
        data = await self.driver("actor-prequeue", extra=[session])
        self.assertEqual(data["generation"], 2)
        self.assertEqual(data["applied"], 4)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)
        self.assertEqual(self.state.attempts["UpdateLayer"], 3)

    async def test_actor_unknown_write_preserves_fifo_without_replay(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        session = await self.create_layer()
        fault = self.state.plan("UpdateLayer", "after_commit", code=grpc.StatusCode.UNAVAILABLE, pause=True)
        data = await self.driver("actor-uncertain", 19, extra=[session], fault=fault,
                                 release_on=b"actor-admissions-complete\n")
        self.assertEqual((data["applied"], data["retained"], data["effects"]), (1, 3, []))
        self.assertEqual(data["immediate_status_checks"], 3)
        self.assertGreater(data["closed_status"]["incident"], 0)
        self.assertEqual((data["closed_status"]["pending"], data["closed_status"]["bytes"],
                          data["closed_status"]["unconfirmed"]), (3, 21, True))
        commits = [c for c in self.state.ledger if c.operation == "UpdateLayer"]
        self.assertEqual([c.payload for c in commits], [b"local-0"])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)

    async def test_actor_terminal_close_waits_for_changed_callback_and_preserves_unknown(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        session = await self.create_layer()
        fault = self.state.plan("UpdateLayer", "after_commit", code=grpc.StatusCode.UNAVAILABLE, pause=True)
        data = await self.driver("actor-capture-changed", 19, extra=[session], fault=fault,
                                 release_on=b"actor-admissions-complete\n")
        self.assertEqual((data["applied"], data["retained"], data["effects"]), (1, 3, []))
        self.assertEqual((data["closed_status"]["pending"], data["closed_status"]["bytes"]), (3, 21))
        commits = [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]
        self.assertEqual([(c.sequence, c.payload) for c in commits], [(2, b"local-0")])
        self.assertEqual(len({c.writer for c in commits}), 1)
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)

    async def test_actor_auth_refresh_and_active_read_failure(self):
        session = await self.create_layer()
        data = await self.driver("actor-auth-refresh", token="synthetic-invalid", extra=[session])
        self.assertEqual((data["auth"], data["applied"]), (1, 4))
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)
        self.assertEqual(self.state.attempts["UpdateLayer"], 3)
        # New layers remain idle after becoming Ready: terminal auth failure
        # must still end the active read, even if credentials are replaced.
        for mode in ("actor-auth-fail", "actor-auth-expiry-idle", "actor-auth-expiry-held"):
            session = await self.create_layer()
            data = await self.driver(mode, 6, extra=[session])
            self.assertEqual((data["applied"], data["retained"], data["effects"]), (1, 0, []))
            self.assertGreaterEqual(data["auth"], 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 5)
        self.assertEqual(self.state.attempts["UpdateLayer"], 3)

    async def test_actor_update_unauthenticated_is_reissued_once_after_refresh(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        # The service authenticates before executing an update, so a rejected
        # attempt never ran: the intent waits in AUTH_WAIT and is reissued
        # exactly once with the refreshed token (design section 13).
        session = await self.create_layer()
        self.state.plan("UpdateLayer", "before", code=grpc.StatusCode.UNAUTHENTICATED)
        data = await self.driver("actor-auth-refresh", extra=[session])
        self.assertGreaterEqual(data["auth"], 1)
        self.assertEqual(self.state.attempts["UpdateLayer"], 4)  # Three prequeued writes, the first twice.
        # A second rejection of the reissue fails the layer; no third issue.
        session = await self.create_layer()
        before = self.state.attempts["UpdateLayer"]
        for attempt in (1, 2):
            self.state.plan("UpdateLayer", "before", attempt=before + attempt, code=grpc.StatusCode.UNAUTHENTICATED)
        await self.driver("actor-auth-refresh", 6, extra=[session])
        self.assertEqual(self.state.attempts["UpdateLayer"], before + 2)
        # Exhausted refresh fails the layer without reissuing.
        session = await self.create_layer()
        before = self.state.attempts["UpdateLayer"]
        self.state.plan("UpdateLayer", "before", attempt=before + 1, code=grpc.StatusCode.UNAUTHENTICATED)
        data = await self.driver("actor-auth-reissue-fail", 6, extra=[session])
        self.assertEqual(data["auth"], 1)
        self.assertEqual(self.state.attempts["UpdateLayer"], before + 1)
        self.assertFalse([c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"])

    async def server_read(self, *, replacements, expected=0):
        from http2_proxy import Http2Proxy
        session = await self.create_layer()
        proxy = Http2Proxy(self.address)
        self.proxies.append(proxy)
        self.state.issue("expiry-t1", "writer")

        async def expire_and_reset():
            self.state.expire("expiry-t1")
            self.assertEqual(len(await proxy.reset_active()), 1)  # The read is the only stream.
        data = await self.driver("actor-auth-server-read", expected, token="expiry-t1",
                                 extra=[session, *replacements], address=await proxy.start(),
                                 steps=[(b"actor-server-auth-ready", expire_and_reset)])
        self.assertFalse(proxy.errors)
        return data

    async def test_actor_read_reconnect_after_server_expiry_refreshes_and_resumes(self):
        self.state.issue("expiry-t2", "writer")
        data = await self.server_read(replacements=["expiry-t2"])
        self.assertEqual((data["result"], data["generation"], data["answers"]), (0, 2, [0]))
        self.assertEqual(data["challenges"], [[3, 1, 1, 16]])  # REJECTED, generation 1, gRPC 16.
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 3)
        self.assertNotIn("UpdateLayer", self.state.attempts)
        self.assertEqual(self.rejections(), [("SubscribeReadLayer", "expired", sha("expiry-t1"))])
        states = iter(data["states"])  # READY, AUTH_WAIT, READY in order; RECONNECTING may coalesce into AUTH_WAIT.
        self.assertTrue(all(state in states for state in (2, 4, 2)), data["states"])

    async def test_actor_rejected_generation_limit_fails_the_layer_unauthenticated(self):
        for token in ("limit-t2", "limit-t3"):
            self.state.issue(token, "writer", expires_us=0)
        data = await self.server_read(replacements=["limit-t2", "limit-t3"], expected=6)
        self.assertEqual(data["result"], 6)
        self.assertEqual([c[:2] for c in data["challenges"]], [[3, 1], [3, 2]])
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 4)  # Initial plus three rejected generations.
        self.assertEqual(data["diagnostics"]["metrics"]["auth"]["failures_by_cause"], [dict(cause=2, count=1)])

    async def test_actor_update_after_server_expiry_is_reissued_once(self):
        session = await self.create_layer()
        self.state.issue("expiry-t1", "writer")

        async def expire():
            self.state.issue("expiry-t2", "writer")
            self.state.expire("expiry-t1")
        data = await self.driver("actor-auth-server-update", token="expiry-t1", extra=[session, "expiry-t2"],
                                 steps=[(b"actor-server-auth-ready", expire)])
        self.assertEqual((data["result"], data["challenges"], data["answers"]), (0, [[3, 1, 1, 16]], [0]))
        commits = [c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]
        self.assertEqual([(c.sequence, c.payload) for c in commits], [(2, b"local-0")])
        self.assertEqual(self.state.attempts["UpdateLayer"], 2)
        self.assertEqual(self.rejections(), [("UpdateLayer", "expired", sha("expiry-t1"))])
        # A reissue rejected again fails the layer; there is no third issue.
        session = await self.create_layer()
        before = self.state.attempts["UpdateLayer"]
        self.state.issue("expiry-t3", "writer")

        async def expire_both():
            self.state.issue("expiry-t4", "writer", expires_us=0)
            self.state.expire("expiry-t3")
        data = await self.driver("actor-auth-server-update", 6, token="expiry-t3", extra=[session, "expiry-t4"],
                                 steps=[(b"actor-server-auth-ready", expire_both)])
        self.assertEqual((data["result"], data["pending"], data["answers"]), (6, 1, [0]))
        self.assertEqual(data["challenges"][0], [3, 1, 1, 16])
        self.assertEqual(self.state.attempts["UpdateLayer"], before + 2)
        self.assertFalse([c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"])

    async def ingress(self, pki=None):
        from http2_proxy import Http2Proxy
        proxy = Http2Proxy(self.address, ssl_context=pki and pki.server_context())
        self.proxies.append(proxy)
        return proxy, await proxy.start()

    tls_pki = tls_fixture.case_pki

    async def test_native_ingress_rejection_refreshes_without_replaying_create_session(self):
        for kind in ("http401", "envoy"):
            proxy, address = await self.ingress()
            proxy.reject_new_streams(kind)
            before = dict(self.state.attempts)
            data = await self.driver("auth-rotate", address=address)
            self.assertEqual(data["callbacks"], 2)
            self.assertEqual(self.state.attempts.get("CreateSession", 0), before.get("CreateSession", 0))
            self.assertEqual(self.state.attempts["Ping"], before.get("Ping", 0) + 1)
            self.assertEqual(len([e for e in proxy.events if e.get("action") == "ingress-reject"]), 1)
        self.assertFalse(self.state.ledger)

    async def test_a_half_set_test_tls_environment_aborts_the_driver(self):
        pki = await self.tls_pki()
        environment = {key: value for key, value in os.environ.items() if not key.startswith("LIVE_TEST_TLS_")}
        for half in ({"LIVE_TEST_TLS_CA": str(pki.ca)}, {"LIVE_TEST_TLS_NAME": tls_fixture.NAME}):
            process = await asyncio.create_subprocess_exec(os.environ["LIVE_TEST_UNARY_DRIVER"], self.address,
                "auth-rotate", "0", "0", "synthetic-writer", stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=dict(environment, **half))
            output, _ = await asyncio.wait_for(process.communicate(), 12)
            with self.subTest(half=sorted(half)):
                self.assertNotEqual(process.returncode, 0)
                self.assertIn(b"LIVE_TEST_TLS_CA and LIVE_TEST_TLS_NAME", output)  # No plaintext fallback.
        self.assertEqual(self.state.attempts, {})

    async def test_native_ingress_rejection_behind_tls_refreshes_once(self):
        pki = await self.tls_pki()
        proxy, address = await self.ingress(pki)
        proxy.reject_new_streams("http401")
        data = await self.driver("auth-rotate", address=address, environment=pki.environment())
        self.assertEqual(data["callbacks"], 2)
        self.assertEqual(self.state.attempts, {"Ping": 1})  # CreateSession never reached the service.
        self.assertEqual(len([e for e in proxy.events if e.get("action") == "ingress-reject"]), 1)
        self.assertFalse(self.state.ledger)

    async def test_actor_update_rejected_at_the_ingress_is_reissued_once(self):
        for rejections, expected in ((1, 0), (2, 6)):
            session = await self.create_layer()
            proxy, address = await self.ingress()
            before = self.state.attempts.get("UpdateLayer", 0)

            async def arm():
                proxy.reject_new_streams("envoy", rejections)
            data = await self.driver("actor-auth-server-update", expected, extra=[session, "synthetic-writer"],
                                     address=address, steps=[(b"actor-server-auth-ready", arm)])
            updates = sum(r["count"] for r in data["diagnostics"]["metrics"]["rpc_attempts_by_operation"]
                          if r["operation"] == 4)
            self.assertEqual(updates, 2)  # The issue and its one reissue; never a third.
            # The proxy forwards a rejected header block, so the service starts
            # and counts each rejected update before the reset cancels it.
            self.assertEqual(self.state.attempts["UpdateLayer"], before + 2)
            self.assertEqual(data["answers"], [0])
            commits = [c.payload for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"]
            self.assertEqual(commits, [b"local-0"] if expected == 0 else [])
            self.assertEqual(len([e for e in proxy.events if e.get("action") == "ingress-reject"]), rejections)
            self.assertEqual(data["challenges"][0], [3, 1, 1, 16])

    async def test_actor_read_reconnect_rejected_at_the_ingress_refreshes_and_resumes(self):
        session = await self.create_layer()
        proxy, address = await self.ingress()

        async def arm_and_reset():
            proxy.reject_new_streams("http401")
            self.assertEqual(len(await proxy.reset_active()), 1)
        data = await self.driver("actor-auth-server-read", extra=[session, "synthetic-writer"], address=address,
                                 steps=[(b"actor-server-auth-ready", arm_and_reset)])
        self.assertEqual((data["result"], data["challenges"], data["answers"]), (0, [[3, 1, 1, 16]], [0]))
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)  # The rejected request never reached it.
        self.assertFalse(self.state.auth_rejections)

    async def test_actor_rejection_of_a_replaced_generation_raises_no_challenge(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        session = await self.create_layer()
        self.state.issue("stale-t1", "writer")
        held = self.state.plan("Ping", "before_auth", attempt=1, pause=True)  # The read's first timing Ping.

        async def revoke():
            await asyncio.wait_for(held.reached.wait(), 3)
            self.state.issue("stale-t2", "writer")
            del self.state.tokens["stale-t1"]

        async def release():
            held.release.set()
        data = await self.driver("actor-auth-stale-generation", token="stale-t1", extra=[session, "stale-t2"],
                                 steps=[(b"actor-server-auth-ready", revoke), (b"actor-server-auth-committed", release)])
        self.assertEqual((data["result"], data["challenges"], data["answers"]), (0, [[3, 1, 1, 16]], [0]))
        auth = data["diagnostics"]["metrics"]["auth"]
        self.assertEqual((auth["challenges_by_reason"], auth["failures_by_cause"]), ([dict(reason=3, count=1)], []))
        self.assertEqual([(m, r) for m, r, _ in self.rejections()], [("UpdateLayer", "unknown"), ("Ping", "unknown")])
        self.assertEqual(self.state.attempts["UpdateLayer"], 2)

    async def test_actor_write_rejection_never_decorates_a_local_cause(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        session = await self.create_layer()
        fault = self.state.plan("UpdateLayer", "before", code=grpc.StatusCode.INVALID_ARGUMENT, pause=True)
        await self.driver("actor-latched-local-cause", 1, extra=[session], fault=fault,
                          release_on=b"actor-local-cause-latched\n")
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.assertFalse([c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"])

    async def test_actor_rejection_without_detached_application_closes_the_read(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        session = await self.create_layer()
        self.state.plan("UpdateLayer", "before", code=grpc.StatusCode.INVALID_ARGUMENT)
        await self.driver("actor-rejection-closes-read", 1, extra=[session])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.assertFalse([c for c in self.state.ledger if c.session == session and c.operation == "UpdateLayer"])

    async def test_actor_read_failure_and_bounded_retry(self):
        session = await self.create_layer()
        self.state.plan("SubscribeReadLayer", "before", code=grpc.StatusCode.PERMISSION_DENIED)
        await self.driver("actor-terminal", 7, extra=[session])
        # The driver bounds reads by reconnect_max_attempts (4), not by
        # retry_max_attempts (3): four failed reads fail the layer UNAVAILABLE.
        for attempt in (2, 3, 4, 5):
            self.state.plan("SubscribeReadLayer", "before", attempt=attempt, code=grpc.StatusCode.UNAVAILABLE)
        await self.driver("actor-terminal", 8, extra=[session])
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 5)
        self.assertNotIn("UpdateLayer", self.state.attempts)

    async def test_actor_reconnect_count_resets_at_recovery_readiness(self):
        """Each read that reaches recovery readiness restarts the reconnect count.

        With reconnect_max_attempts 4, the stream ends after each caught-up
        header and three reconnects then fail before one catches up again. A
        count that survived readiness would reach the bound at the fifth read
        and fail the layer; a reset count lets the ninth read catch up.
        """
        from unittest.mock import patch
        from service import LiveService
        session = await self.create_layer()
        requests = []

        async def stream(service, request, context):
            await service.begin("SubscribeReadLayer", context)
            requests.append((request.object_id, request.sequence_num))
            ordinal = len(requests)
            if ordinal % 4 != 1:
                await context.abort(grpc.StatusCode.UNAVAILABLE, "synthetic failed reconnect")
            yield self.read_header(0, sequence=1)
            if ordinal < 9:
                await context.abort(grpc.StatusCode.UNAVAILABLE, "synthetic end of a healthy stream")
            await asyncio.Future()  # Cancelled by the native client's close.

        with patch.object(LiveService, "SubscribeReadLayer", stream):
            server = ReferenceServer(self.state, "actor-reconnect-reset")
        self.servers.append(server)
        address = await server.start()
        data = await self.driver("actor-reconnect-reset", extra=[session], address=address)
        self.assertEqual(requests, [(1, 1)] * 9)
        self.assertEqual((data["result"], data["generation"], data["applied"], data["retained"]), (0, 9, 1, 0))
        self.assertNotIn("UpdateLayer", self.state.attempts)

    async def test_actor_caught_up_below_cursor_fails_the_layer_and_retains_intents(self):
        # The service's caught-up header is at sequence 2 while the client resumes
        # at 3: a server sequence regression, never an up-to-date signal.
        # The regression is held until the driver has queued its three intents,
        # so every submit precedes the failure it causes.
        session = await self.create_layer()
        regression = asyncio.Event()
        address = await self.read_script([self.read_header(0, sequence=2)], hold=regression)
        async def release():
            regression.set()
        data = await self.driver("actor-sequence-regression", 11, extra=[session], address=address,
                                 steps=[(b"actor-regression-queued", release)])
        self.assertEqual((data["applied"], data["retained"], data["effects"]), (3, 3, []))
        self.assertEqual((data["closed_status"]["pending"], data["closed_status"]["unconfirmed"]), (3, True))
        self.assertGreater(data["closed_status"]["incident"], 0)
        self.assertEqual(self.read_scripts[-1]["sent"], 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        self.assertNotIn("UpdateLayer", self.state.attempts)

    async def test_actor_held_application_close_retains_unsent_edits(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        session = await self.create_layer()
        data = await self.driver("actor-held", extra=[session])
        self.assertEqual((data["applied"], data["retained"]), (2, 2))
        self.assertEqual(data["effects"], [{"sequence": 2, "bytes": "local-0"}])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        # The native driver checks its explicit Ping completed while application
        # was held; autonomous timing may independently issue another Ping.
        self.assertGreaterEqual(self.state.attempts["Ping"], 1)

    async def test_actor_read_backlog_closes_write_gate_before_drain(self):
        from unittest.mock import patch
        from service import LiveService
        session = await self.create_layer()
        original = LiveService.SubscribeReadLayer
        requests = []
        async def stream(service, request, context):
            requests.append(request.sequence_num)
            if len(requests) == 1:
                principal, _ = await service.begin("SubscribeReadLayer", context)
                yield self.read_header(0, sequence=1)
                async with service.state.changed:
                    _, layer = await service.layer(session, "root.live", principal, context)
                    commits = []
                    for i in range(33):
                        commit = service.state.commit("UpdateLayer", session, layer="root.live", object_id=1,
                            sequence=i + 2, payload=f"peer-{i}".encode(), writer="independent-peer")
                        layer.commits.append(commit); commits.append(commit)
                for commit in commits:
                    for frame in service.frames(commit):
                        yield frame
                await context.abort(grpc.StatusCode.UNAVAILABLE, "synthetic rotation behind native backlog")
            else:
                self.assertEqual(self.state.attempts.get("UpdateLayer", 0), 0)
                async for frame in original(service, request, context):
                    yield frame
        with patch.object(LiveService, "SubscribeReadLayer", stream):
            server = ReferenceServer(self.state, "actor-backlog")
        self.servers.append(server)
        address = await server.start()
        data = await self.driver("actor-backlog", extra=[session], address=address)
        self.assertEqual(requests, [1, 34])
        self.assertEqual((data["applied"], data["retained"]), (37, 0))
        self.assertEqual(data["effects"],
            [{"sequence": i + 2, "bytes": f"peer-{i}"} for i in range(33)] +
            [{"sequence": i + 35, "bytes": f"local-{i}"} for i in range(3)])
        self.assertEqual(self.state.attempts["UpdateLayer"], 3)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)

    async def test_actor_reconnect_waits_for_issued_write_not_unsent_fifo(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        from unittest.mock import patch
        from service import LiveService
        session = await self.create_layer()
        fault = self.state.plan("UpdateLayer", "after_commit", pause=True)
        original = LiveService.SubscribeReadLayer
        requests = []
        async def stream(service, request, context):
            requests.append((request.object_id, request.sequence_num, request.client_id))
            if len(requests) == 1:
                await service.begin("SubscribeReadLayer", context)
                yield self.read_header(0, sequence=1)
                async with service.state.changed:
                    await service.state.changed.wait_for(lambda: any(c.operation == "UpdateLayer" for c in service.state.ledger))
                await context.abort(grpc.StatusCode.UNAVAILABLE, "synthetic reader rotation during issued write")
            else:
                self.assertEqual(len([c for c in self.state.ledger if c.operation == "UpdateLayer"]), 1)
                async for frame in original(service, request, context):
                    yield frame
        with patch.object(LiveService, "SubscribeReadLayer", stream):
            server = ReferenceServer(self.state, "actor-recovery")
        self.servers.append(server)
        address = await server.start()
        data = await self.driver("actor-recover-issued", extra=[session], address=address, fault=fault,
                                 release_on=b"actor-recovery-paused\n")
        self.assertEqual([(o, s) for o, s, _ in requests], [(1, 1), (1, 2)])
        self.assertEqual(len({writer for _, _, writer in requests}), 1)
        self.assertEqual((data["generation"], data["applied"], data["retained"]), (2, 4, 0))
        self.assertEqual(self.state.attempts["UpdateLayer"], 3)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)
        self.assertEqual([c.payload for c in self.state.ledger if c.operation == "UpdateLayer"],
                         [b"local-0", b"local-1", b"local-2"])

    async def test_native_session_roundtrip_and_input_ownership(self):
        await self.driver("roundtrip")
        self.assertEqual([c.operation for c in self.state.ledger], ["CreateSession", "DeleteSession"])
        self.assertEqual(self.state.attempts, {"Ping": 1, "CreateSession": 1, "ListSessions": 1, "DeleteSession": 1})
        self.assertFalse(self.state.sessions)

    async def test_timing_receipt_precedes_blocked_callback_worker(self):
        data = await self.driver("timing-receipt")
        self.assertEqual(data["callbacks"], 1)
        self.assertIs(data["receipt_before_release"], True)
        self.assertIs(data["callback_after_release"], True)
        self.assertEqual(self.state.attempts, {"Ping": 1})
        self.assertFalse(self.state.ledger)

    async def test_native_concurrent_submission_and_completion(self):
        data = await self.driver("concurrent")
        self.assertEqual(data["callbacks"], 128)
        self.assertEqual(self.state.attempts, {"Ping": 128})

    async def test_native_auth_callback_rotates_metadata_without_replaying_mutation(self):
        data = await self.driver("auth-rotate", token="synthetic-invalid")
        self.assertEqual(data["callbacks"], 2)
        self.assertEqual(self.state.attempts, {"CreateSession": 1, "Ping": 1})
        self.assertFalse(self.state.ledger)

    async def test_native_local_auth_and_capacity_rejections_issue_nothing(self):
        await self.driver("missing-auth", 6, -1, token="")
        await self.driver("expired-auth", 6, -1)
        self.assertFalse(self.state.attempts)
        await self.driver("capacity")
        self.assertEqual(self.state.attempts, {"Ping": 1})
        self.assertFalse(self.state.ledger)

    async def test_native_layer_stat_delete_and_not_found(self):
        session = await self.create_layer()
        await self.driver("stat", extra=[session])
        await self.driver("delete-layer", extra=[session])
        await self.driver("stat", 4, 5, extra=[session])
        self.assertEqual([c.operation for c in self.state.ledger], ["CreateSession", "CreateLayer", "DeleteLayer"])

    async def test_native_identical_failures_before_and_after_commit(self):
        for boundary, committed in (("before", 0), ("after_commit", 1)):
            self.state.plan("CreateSession", boundary, code=grpc.StatusCode.UNAVAILABLE)
            await self.driver("create", 19, 14)
            self.assertEqual(len(self.state.ledger), committed)
        self.assertEqual(self.state.attempts["CreateSession"], 2)
        self.assertEqual(len(self.state.sessions), 1)

    async def test_native_cancellation_before_and_after_commit(self):
        for boundary, committed in (("before", 0), ("after_commit", 1)):
            fault = self.state.plan("CreateSession", boundary, pause=True)
            data = await self.driver("cancel-create", 19, 1, fault=fault)
            self.assertEqual(data["callbacks"], 0)
            self.assertEqual(len(self.state.ledger), committed)
        self.assertEqual(self.state.attempts["CreateSession"], 2)

    async def test_native_mutation_deadline_after_commit(self):
        self.state.plan("CreateSession", "after_commit", pause=True)
        await self.driver("deadline-create", 19, 4)
        self.assertEqual(len(self.state.ledger), 1)
        self.assertEqual(self.state.attempts["CreateSession"], 1)

    async def test_native_read_cancellation_and_deadline(self):
        # Each mode performs one native warmup Ping before its target attempt.
        fault = self.state.plan("Ping", "before_reply", attempt=2, pause=True)
        await self.driver("cancel-ping", 10, 1, fault=fault)
        self.state.plan("Ping", "before_reply", attempt=4, pause=True)
        await self.driver("deadline-ping", 9, 4)
        self.assertFalse(self.state.ledger)
        self.assertEqual(self.state.attempts["Ping"], 4)

    async def test_native_read_statuses_and_mutation_auth_rejections(self):
        await self.driver("ping", 6, 16, token="synthetic-invalid")
        await self.driver("create", 6, 16, token="synthetic-invalid")
        await self.driver("create", 7, 7, token="synthetic-reader")
        for grpc_code, result in ((grpc.StatusCode.RESOURCE_EXHAUSTED, 13),
                                  (grpc.StatusCode.UNAVAILABLE, 8),
                                  (grpc.StatusCode.UNIMPLEMENTED, 12),
                                  (grpc.StatusCode.DATA_LOSS, 20)):
            self.state.plan("Ping", "before_reply", code=grpc_code)
            await self.driver("ping", result, grpc_code.value[0])
        self.assertFalse(self.state.ledger)

    async def test_native_tls_authority_and_trust(self):
        pki, other = await self.tls_pki(), await self.tls_pki("other")
        server = ReferenceServer(self.state, "TLS", credentials=grpc.ssl_server_credentials(
            [(pki.key.read_bytes(), pki.certificate.read_bytes())]))
        self.servers.append(server)
        address = await server.start()
        await self.driver("tls-ping", extra=[pki.ca, "live.test"], address=address)
        await self.driver("tls-ping", 8, 14, extra=[pki.ca, "wrong.test"], address=address)
        # A separately generated root cannot authenticate the first server.
        await self.driver("tls-ping", 8, 14, extra=[other.ca, "live.test"], address=address)
        self.assertEqual(self.state.attempts["Ping"], 1)

    async def test_native_tls_through_the_proxy_verifies_the_ca_and_name(self):
        pki, other = await self.tls_pki(), await self.tls_pki("other")
        proxy, address = await self.ingress(pki)
        await self.driver("tls-ping", extra=[pki.ca, "live.test"], address=address)
        await self.driver("tls-ping", 8, 14, extra=[pki.ca, "wrong.test"], address=address)
        await self.driver("tls-ping", 8, 14, extra=[other.ca, "live.test"], address=address)
        self.assertEqual(self.state.attempts["Ping"], 1)
