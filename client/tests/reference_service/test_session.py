# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public native session ABI against an independent loopback service.

The ledger records received wire requests; successful send receipts alone never
prove delivery. Scripted streams exercise reconnect/protocol boundaries.
"""
import asyncio
import json
import os
from pathlib import Path
import unittest

import grpc
from service import State, Principal, ReferenceServer, LiveService, pb, rpc
import tls_fixture

WRITER = (("authorization", "Bearer synthetic-writer"),)
# The marker each keepalive driver mode prints when the fixture takes over.
KEEPALIVE_CONTROLS = {"join-keepalive-queued": b"session-keepalive-expired",
                      "join-keepalive-older": b"session-keepalive-expired",
                      "join-keepalive-host-blocked": b"session-keepalive-ready",
                      "join-keepalive-timeout": b"session-keepalive-timeout-ready"}
# A keepalive interval no test outlasts, so only the driver's expiry seam or an advanced keepalive clock acts on it.
UNREACHED_INTERVAL_MS = 3_600_000


class SessionService(LiveService):
    async def JoinSession(self, stream, context):
        async def recorded():
            async for request in stream:
                kind = request.WhichOneof("request_type")
                self.state.session_wire.append({"kind": kind,
                    "session": request.header.session_id if kind == "header" else "",
                    "app": request.header.user_application if kind == "header" else "",
                    "keepalive": request.header.client_keepalive if kind == "header" else False,
                    "hex": request.message.data.hex() if kind == "message" else ""})
                if kind == "message":
                    self.state.message_recorded.set()
                yield request
        mode = self.state.session_mode
        if mode in ("join-happy", "join-held", "join-self", "join-auth-refresh", "join-discard", "join-discard-no-history", "join-callback-throw",
                    "send-status-under-prepare", "send-leave-during-prepare", "join-auth-server-expiry",
                    "join-keepalive-queued", "join-keepalive-older", "join-keepalive-host-blocked", "join-keepalive-timeout"):
            async for response in super().JoinSession(recorded(), context):
                self.state.session_responses.append(response.SerializeToString().hex())
                yield response
            return
        principal, attempt = await self.begin("JoinSession", context)
        first = await anext(recorded_stream := recorded(), None)
        if first is None or first.WhichOneof("request_type") != "header":
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "missing header")
        async with self.state.changed:
            await self.session(first.header.session_id, principal, context)
        if mode == "join-fail-queued":
            await self.state.queue_ready.wait()
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "synthetic terminal queue failure")
        if mode == "join-permission":
            await context.abort(grpc.StatusCode.PERMISSION_DENIED, "synthetic permanent refusal")
        if mode == "join-cancel-before-connected":
            await context.abort(grpc.StatusCode.CANCELLED, "synthetic cancelled attempt")
        if mode in ("join-protocol-after-unavailable", "join-revoke-after-unavailable") and attempt == 1:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "first-attempt evidence")
        if mode in ("join-protocol", "join-protocol-after-unavailable"):
            yield pb.JoinSessionResponse(peer_joined=pb.PeerJoined(
                connection_id="too-early", user_name="writer", user_application="fixture"))
            await asyncio.Event().wait()
        elif mode == "join-capacity":
            await self.state.queue_ready.wait()
            yield pb.JoinSessionResponse(connected=pb.SessionConnected(connection_id="capacity-connection"))
            yield pb.JoinSessionResponse(peer_joined=pb.PeerJoined(
                connection_id="fixture-peer", user_name="writer", user_application="fixture"))
            for _ in range(2):
                request = await anext(recorded_stream)
                if request.WhichOneof("request_type") != "message":
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "expected queued message")
            yield pb.JoinSessionResponse(peer_message=pb.PeerMessage(connection_id="fixture-peer", data=b"received-two"))
            await asyncio.Event().wait()
        elif mode in ("join-reconnect", "join-reconnect-eof", "join-reconnect-reset"):
            if attempt == 1:
                yield pb.JoinSessionResponse(connected=pb.SessionConnected(connection_id="connection-old"))
                identifiers = ("peer-keep", "peer-keep") if mode == "join-reconnect-reset" else ("peer-keep", "peer-keep", "peer-gone")
                for identifier in identifiers:
                    yield pb.JoinSessionResponse(peer_joined=pb.PeerJoined(
                        connection_id=identifier, user_name="writer", user_application="peer-app"))
                yield pb.JoinSessionResponse(peer_message=pb.PeerMessage(connection_id="peer-keep", data=b"rotate"))
                await self.state.rotate_ready.wait()
                if mode == "join-reconnect-reset":
                    await asyncio.Event().wait()  # Real proxy reset cancels this handler.
                if mode == "join-reconnect-eof":
                    # Return normally with gRPC OK while the client request
                    # stream remains open; client-side cancellation must not
                    # replace this EOF evidence with terminal CANCELLED.
                    return
                await context.abort(grpc.StatusCode.UNAVAILABLE, "synthetic load-balancer rotation")
            else:
                await self.state.queue_ready.wait()
                yield pb.JoinSessionResponse(connected=pb.SessionConnected(connection_id="connection-new"))
                for _ in range(2):
                    yield pb.JoinSessionResponse(peer_joined=pb.PeerJoined(
                        connection_id="peer-keep", user_name="writer", user_application="peer-app"))
                for _ in range(2):
                    await anext(recorded_stream)
                yield pb.JoinSessionResponse(peer_message=pb.PeerMessage(connection_id="peer-keep", data=b"resumed"))
                await asyncio.Event().wait()
        else:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "unknown fixture mode")


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = State(int(os.environ["LIVE_TEST_SEED"]))
        self.state.tokens = {"synthetic-writer": Principal("writer")}
        self.state.session_mode = "join-happy"
        self.state.session_wire, self.state.session_responses = [], []
        self.state.rotate_ready, self.state.queue_ready = asyncio.Event(), asyncio.Event()
        self.state.expiry_ready, self.state.message_recorded = asyncio.Event(), asyncio.Event()
        self.state.keepalive_ready = asyncio.Event()
        self.children, self.drainers = [], []
        self.proxies = []
        self.artifact = Path(os.environ["LIVE_TEST_ARTIFACT_DIR"])
        self.server = ReferenceServer(self.state, service_factory=SessionService)
        self.address = await self.server.start()
        self.channel = grpc.aio.insecure_channel(self.address)
        self.addAsyncCleanup(self.cleanup)
        await asyncio.wait_for(self.channel.channel_ready(), 5)
        self.stub = rpc.LiveServiceStub(self.channel)
        self.session = (await self.stub.CreateSession(pb.CreateSessionRequest(path="session-base", name="session-fixture"),
                                                     metadata=WRITER, timeout=5)).session_id

    async def cleanup(self):
        for process in self.children:
            if process.returncode is None:
                process.kill()
        results = await asyncio.gather(*(p.wait() for p in self.children), *self.drainers, return_exceptions=True)
        results += await asyncio.gather(*(p.close() for p in self.proxies), return_exceptions=True)
        await self.channel.close()
        await self.server.stop()
        data = {"seed": int(os.environ["LIVE_TEST_SEED"]), "attempts": self.state.attempts,
                "requests": self.state.session_wire, "responses": self.state.session_responses,
                "transport_proxies": [p.snapshot() for p in self.proxies]}
        (self.artifact / (self.id().split(".")[-1] + ".session-ledger.json")).write_text(json.dumps(data, indent=2) + "\n")
        for result in results:
            if isinstance(result, BaseException):
                raise result

    tls_pki = tls_fixture.case_pki

    async def driver(self, mode, expected=0, token="synthetic-writer", tls=False):
        self.state.session_mode = mode
        address, environment = self.address, None
        if mode in ("join-reconnect-reset", "join-auth-server-expiry"):
            from http2_proxy import Http2Proxy
            pki = await self.tls_pki() if tls else None
            proxy = Http2Proxy(address, ssl_context=pki and pki.server_context())
            self.proxies.append(proxy)
            address = await proxy.start()
            if tls:
                environment = dict(os.environ, **pki.environment())
        self.assertFalse(tls and environment is None, "TLS needs the fixture proxy")
        process = await asyncio.create_subprocess_exec(os.environ["LIVE_TEST_UNARY_DRIVER"], address,
            mode, str(expected), "0", token, self.session,
            stdin=asyncio.subprocess.PIPE if mode in KEEPALIVE_CONTROLS or mode == "join-auth-server-expiry" else
            asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=environment)
        self.children.append(process)
        output = bytearray()
        markers = set()
        keepalive_marker = [(KEEPALIVE_CONTROLS[mode], self.state.keepalive_ready)] if mode in KEEPALIVE_CONTROLS else []
        async def drain():
            tail = b""
            while block := await process.stdout.read(4096):
                output.extend(block[:max(0, 65536-len(output))])
                tail += block
                for marker, event in ((b"session-rotate-ready", self.state.rotate_ready),
                                      (b"session-queued", self.state.queue_ready),
                                      (b"session-expiry-ready", self.state.expiry_ready), *keepalive_marker):
                    if marker in tail:
                        markers.add(marker)
                        event.set()
                tail = tail[-64:]
        drainer = asyncio.create_task(drain())
        self.drainers.append(drainer)
        reset = None
        if mode == "join-reconnect-reset":
            async def inject_reset():
                await self.state.rotate_ready.wait()
                faults = await proxy.reset_active(code=8)
                self.assertEqual(len(faults), 1, faults)
            reset = asyncio.create_task(inject_reset())
        if mode == "join-auth-server-expiry":
            async def inject_expiry():
                try:
                    await self.state.expiry_ready.wait()
                    self.state.issue("session-t2", "writer")
                    self.state.expire("synthetic-writer")
                    self.assertEqual(len(await proxy.reset_active(code=8)), 1)
                    # A send receipt is local acceptance and Leave cancels the stream, so
                    # the driver leaves only after the service records the queued frame.
                    await self.state.message_recorded.wait()
                    process.stdin.write(b"continue\n")
                    await process.stdin.drain()
                finally:
                    # The driver reads one command; a closed pipe ends its wait when this task fails first.
                    process.stdin.close()
            reset = asyncio.create_task(inject_expiry())
        if mode in KEEPALIVE_CONTROLS:
            reset = asyncio.create_task(self.keepalive_control(mode, process))
        try:
            await asyncio.wait_for(process.wait(), 15)
            await drainer
            if reset:
                await asyncio.wait_for(reset, 5)
        finally:
            if reset:
                if not reset.done():
                    reset.cancel()
                await asyncio.gather(reset, return_exceptions=True)
        text = output.decode(errors="replace")
        (self.artifact / (self.id().split(".")[-1] + ".session-native.log")).write_text(text)
        self.assertEqual(process.returncode, 0, text)
        data = json.loads(text.splitlines()[-1])
        self.assertEqual(data["result"], expected)
        if mode in ("join-reconnect", "join-reconnect-eof", "join-reconnect-reset"):
            self.assertEqual(markers, {b"session-rotate-ready", b"session-queued"})
        if mode in ("join-capacity", "join-fail-queued"):
            self.assertEqual(markers, {b"session-queued"})
        snapshot = data['diagnostics']
        self.assertFalse(snapshot['history']['truncated'])
        self.assertEqual(snapshot['metrics']['rpc_in_flight'], 0)
        subjects = snapshot['metrics']['subjects']
        self.assertEqual([row['kind'] for row in subjects], [1, 2, 3])
        for row in subjects:
            self.assertEqual([state['state'] for state in row['states']], list(range(1, 10)))
            self.assertEqual(sum(state['current'] for state in row['states']), row['current'])
            self.assertEqual(row['current'], row['admitted'] - row['detached'])
            self.assertGreaterEqual(row['high_water'], row['current'])
        self.assertEqual(subjects[0]['current'], 1)  # Endpoint remains registered during export.
        self.assertEqual(subjects[1]['admitted'], 2 if mode == 'join-happy' else 1)
        self.assertEqual(subjects[1]['current'], 0)  # Leave detached every session.
        self.assertEqual(subjects[2]['admitted'], 0)
        readiness = snapshot['metrics']['readiness']
        self.assertEqual([row['kind'] for row in readiness], [2, 3])
        for row in readiness:
            self.assertEqual((row['initial_waiting'], row['recovering']), (0, 0))
            for name in ('initial_ms_by_resolution', 'recovery_ms_by_resolution'):
                histogram = row[name]
                self.assertEqual(histogram['bounds'], [1, 5, 10, 50, 100, 500, 1000, 5000])
                self.assertEqual([entry['resolution'] for entry in histogram['entries']],
                                 sorted({entry['resolution'] for entry in histogram['entries']}))
                for entry in histogram['entries']:
                    self.assertIn(entry['resolution'], (0, 1, 2))
                    self.assertEqual(sum(entry['counts']), entry['count'])
                    self.assertGreater(entry['count'], 0)
                    self.assertGreaterEqual(entry['sum_ms'], 0)
        initial = readiness[0]['initial_ms_by_resolution']['entries']
        self.assertEqual(sum(entry['count'] for entry in initial), subjects[1]['admitted'])
        self.assertEqual(readiness[1]['initial_ms_by_resolution']['entries'], [])
        self.assertEqual(readiness[1]['recovery_ms_by_resolution']['entries'], [])
        if mode == 'join-happy':
            self.assertEqual([(entry['resolution'], entry['count']) for entry in initial], [(0, 2)])
            self.assertEqual(readiness[0]['recovery_ms_by_resolution']['entries'], [])
        if mode in ('join-reconnect', 'join-reconnect-eof', 'join-reconnect-reset', 'join-auth-server-expiry'):
            recovered = readiness[0]['recovery_ms_by_resolution']['entries']
            self.assertEqual([(entry['resolution'], entry['count']) for entry in recovered], [(0, 1)])
        capacity = snapshot['metrics']['capacity']
        for slot in capacity['request_slots'].values():
            self.assertEqual(slot['current'], 0)
            self.assertEqual(slot['admitted'], slot['released'])
            self.assertLessEqual(slot['high_water'], slot['admitted'])
        charged = capacity['pending_bytes']
        reservations = charged['reservations']
        self.assertEqual(reservations['current'], reservations['admitted'] - reservations['released'])
        self.assertGreaterEqual(reservations['high_water'], reservations['current'])
        self.assertFalse(charged['current_bytes_overflow'])
        self.assertFalse(charged['high_water_bytes_overflow'])
        self.assertEqual(charged['current_bytes'], charged['admitted_bytes'] - charged['released_bytes'])
        self.assertGreaterEqual(charged['high_water_bytes'], charged['current_bytes'])
        if snapshot['metrics']['rpc_attempts']:
            self.assertGreater(reservations['admitted'], 0)
            self.assertGreater(charged['high_water_bytes'], 0)
        tags = snapshot['metrics']['cq_tags']
        if snapshot['metrics']['rpc_attempts']:
            self.assertGreater(tags['submitted'], snapshot['metrics']['rpc_attempts'])
            self.assertGreater(tags['outstanding_high_water'], 0)
        else:
            self.assertEqual(tags['submitted'], 0)
        self.assertEqual(tags['submitted'], tags['received_ok'] + tags['received_not_ok'])
        self.assertEqual(tags['outstanding'], 0)
        self.assertEqual(tags['handling'], 0)
        self.assertEqual(tags['abandoned'], 0)
        self.assertEqual(tags['retired'] + tags['retained'], tags['submitted'])
        for private in ('synthetic-writer', 'Bearer', address, self.session, 'native-alpha',
                        'native-beta', 'connection-old', 'connection-new', 'peer-keep', 'queued-two'):
            self.assertNotIn(private, json.dumps(snapshot))
        return data

    async def keepalive_control(self, mode, process):
        """Answers a keepalive driver mode once its marker arrives; the driver reads one command."""
        try:
            await self.state.keepalive_ready.wait()
            if mode == "join-keepalive-timeout":
                self.state.timed_out = self.state.advance_keepalive_clock(3 * UNREACHED_INTERVAL_MS)
                return
            if mode == "join-keepalive-older":
                await self.state.message_recorded.wait()
            else:
                count = 3 if mode == "join-keepalive-host-blocked" else 1
                async with self.state.changed:
                    await self.state.changed.wait_for(lambda: sum(self.state.keepalives.values()) >= count)
            process.stdin.write(b"continue\n")
            await process.stdin.drain()
        finally:
            if process.stdin:
                process.stdin.close()

    def keepalive_wire(self):
        return [(e["kind"], e["keepalive"]) if e["kind"] == "header" else (e["kind"], bytes.fromhex(e["hex"]))
                for e in self.state.session_wire]

    def check_keepalive_records(self, data):
        # A keepalive is no message: it has no request, receipt, queue charge or message record.
        records = [r for r in data['diagnostics']['recent_records'] if r['subject_id'] == data['subject']]
        self.assertFalse([r for r in records if r['event_name'].startswith('message_') and
                          r['fields']['bytes'] == 0])
        return records

    async def test_session_keepalive_queues_behind_accepted_messages_without_a_request(self):
        self.state.keepalive_interval_ms = UNREACHED_INTERVAL_MS
        data = await self.driver("join-keepalive-queued")
        self.assertEqual(self.keepalive_wire(), [("header", True), ("message", b"first"), ("message", b"second"),
                                                 ("keepalive", b"")])
        self.assertEqual([j["keepalive_interval_ms"] for j in self.state.joins], [UNREACHED_INTERVAL_MS])
        self.assertEqual(self.state.keepalive_timeouts, [])
        self.assertEqual(self.state.attempts["JoinSession"], 1)
        self.assertEqual((data["send_callbacks"], len(data["send_requests"])), (2, 2))
        self.check_keepalive_records(data)
        self.check_message_records(data, [(24, 2, 11), (25, 2, 11)])

    async def test_session_sends_no_keepalive_when_the_service_returns_no_interval(self):
        self.assertIsNone(self.state.keepalive_interval_ms)
        data = await self.driver("join-keepalive-older")
        # The join opts in on every attempt; a Connected without an interval leaves the stream as it was.
        self.assertEqual(self.keepalive_wire(), [("header", True), ("message", b"after-expiry")])
        self.assertEqual([j["keepalive_interval_ms"] for j in self.state.joins], [0])
        self.assertEqual(self.state.attempts["JoinSession"], 1)
        self.assertEqual(self.state.keepalives, {})
        self.check_keepalive_records(data)

    async def test_session_keepalives_flow_while_the_host_thread_is_blocked(self):
        # The service's interval is short and its keepalive clock never moves, so nothing times out; the fixture
        # waits on the service's count of keepalives, under a failure-only watchdog, while the driver's host thread
        # is blocked reading its standard input.
        self.state.keepalive_interval_ms = 20
        data = await self.driver("join-keepalive-host-blocked")
        wire = self.keepalive_wire()
        self.assertEqual(wire[0], ("header", True))
        self.assertGreaterEqual(len(wire), 4)
        self.assertEqual(set(wire[1:]), {("keepalive", b"")})
        self.assertEqual(self.state.attempts["JoinSession"], 1)
        self.assertEqual(data["send_callbacks"], 0)
        self.check_keepalive_records(data)

    async def test_session_keepalive_timeout_rejoins_and_peers_see_one_left(self):
        self.state.keepalive_interval_ms = UNREACHED_INTERVAL_MS
        self.state.timed_out = None
        # The fixture's own peer does not opt in, so only the driver's join times out.
        peer = self.stub.JoinSession(metadata=WRITER, timeout=60)
        await peer.write(pb.JoinSessionRequest(header=pb.JoinSessionHeader(session_id=self.session,
                                                                           user_application="fixture-peer")))
        self.assertEqual((await asyncio.wait_for(peer.read(), 5)).connected.keepalive_interval_ms, 0)
        seen = []
        async def watch():
            while (response := await peer.read()) != grpc.aio.EOF:
                kind = response.WhichOneof("response_type")
                seen.append((kind, getattr(response, kind).connection_id))
        watcher = asyncio.create_task(watch())
        try:
            data = await self.driver("join-keepalive-timeout")
            connections = [e["id"] for e in data["a"] if e["kind"] == 1 and e["result"] == 0]
            self.assertEqual(self.state.timed_out, connections[:1])
            # The driver left after its rejoin, so the peer's last event is that join's PeerLeft.
            async def departed():
                while ("peer_left", connections[1]) not in seen:
                    await asyncio.sleep(0)
            await asyncio.wait_for(departed(), 5)
        finally:
            peer.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        self.assertEqual(seen, [("peer_joined", connections[0]), ("peer_left", connections[0]),
                                ("peer_joined", connections[1]), ("peer_left", connections[1])])
        self.assertEqual([(e["kind"], e["result"]) for e in data["a"] if e["kind"] == 1], [(1, 0), (1, 8), (1, 0)])
        # The fixture peer is reannounced on the rejoin and reported Joined once.
        self.assertEqual([e["kind"] for e in data["a"] if e["kind"] != 1], [2])
        self.assertEqual(self.state.attempts["JoinSession"], 3)
        self.assertEqual([j["client_keepalive"] for j in self.state.joins], [False, True, True])
        records = self.check_keepalive_records(data)
        disconnected = [r['fields'] for r in records if r['event_name'] == 'stream_disconnected']
        self.assertEqual([(f['result'], f['reason'], f['stream_generation']) for f in disconnected], [(8, 1, 1)])
        retries = [r['fields'] for r in records if r['event_name'] == 'retry_scheduled']
        self.assertEqual([(f['result'], f['stream_generation']) for f in retries], [(8, 1)])
        failed = [r['fields'] for r in records if r['event_name'] == 'rpc_failed']
        self.assertEqual([(f['result'], f['grpc_status']) for f in failed], [(8, 14), (10, 1)])

    async def test_session_two_connections_and_copied_fifo_messages(self):
        data = await self.driver("join-happy")
        connections = [[e for e in data[side] if e["kind"] == 1 and e["result"] == 0] for side in ("a", "b")]
        self.assertEqual([len(events) for events in connections], [1, 1])
        first, second = connections[0][0]["id"], connections[1][0]["id"]
        self.assertNotEqual(first, second)
        for side, peer, app in (("a", second, "native-beta"), ("b", first, "native-alpha")):
            joined = [e for e in data[side] if e["kind"] == 2]
            self.assertEqual([(e["id"], e["user"], e["app"]) for e in joined], [(peer, "writer", app)])
            self.assertEqual(data[side][0]["kind"], 1)
        expected = [b"A\0\xff".hex(), b"second".hex(), b"third".hex()]
        self.assertEqual([e["hex"] for e in data["b"] if e["kind"] == 4], expected)
        self.assertEqual([e["id"] for e in data["b"] if e["kind"] == 4], [first]*3)
        self.assertEqual([e["hex"] for e in self.state.session_wire if e["kind"] == "message"], expected)
        self.assertEqual([e["app"] for e in self.state.session_wire if e["kind"] == "header"], ["native-alpha", "native-beta"])
        self.assertEqual(data["send_callbacks"], 2)
        self.assertEqual(self.state.attempts["JoinSession"], 2)
        # A message over the service's 4 MiB stream limit is rejected locally and recorded, never sent.
        records = self.check_message_records(data, [(24, 3, 14), (25, 3, 14), (26, 3, 14), (28, 1, 4128769)])
        incoming = [r for r in records if r['event_name'] == 'message_received']
        self.assertTrue(all(r['fields']['request_id'] == 0 for r in incoming))
        self.assertTrue(all(r['subject_id'] != data['subject'] for r in incoming))

    def check_reconnect(self, data, result=8, grpc_status=14, departed=True):
        events = data["a"]
        self.assertEqual([(e["id"], e["result"]) for e in events if e["kind"] == 1],
                         [("connection-old", 0), ("connection-old", result), ("connection-new", 0)])
        self.assertEqual([e["id"] for e in events if e["kind"] == 1 and e["result"] == 0],
                         ["connection-old", "connection-new"])
        self.assertEqual([e["id"] for e in events if e["kind"] == 2], ["peer-keep", "peer-gone"] if departed else ["peer-keep"])
        self.assertEqual([e["id"] for e in events if e["kind"] == 3], ["peer-gone"] if departed else [])
        new_connection = next(i for i, e in enumerate(events) if e["kind"] == 1 and e["id"] == "connection-new")
        resumed = next(i for i, e in enumerate(events) if e["hex"] == b"resumed".hex())
        self.assertLess(new_connection, resumed)
        self.assertEqual([e["hex"] for e in self.state.session_wire if e["kind"] == "message"],
                         [b"queued\0one".hex(), b"queued-two".hex()])
        self.assertEqual(self.state.attempts["JoinSession"], 2)
        self.assertEqual(data["send_callbacks"], 1)
        records = data['diagnostics']['recent_records']
        def selected(name):
            return [r for r in records if r['event_name'] == name and r['subject_id'] == data['subject']]
        resumed = selected('stream_resumed')
        disconnected, retries, completed = selected('stream_disconnected'), selected('retry_scheduled'), selected('rpc_failed')
        self.assertEqual((len(resumed), len(disconnected), len(retries), len(completed)), (2, 1, 1, 2))
        for items, generations in ((resumed, [1, 2]), (disconnected, [1]), (retries, [1]), (completed, [1, 2])):
            for record, generation in zip(items, generations):
                fields = record['fields']
                self.assertEqual((fields['operation'], fields['session_handle'], fields['attempt_id'],
                                  fields['stream_generation'], fields['auth_generation']),
                                 (6, data['session_handle'], generation, generation, 1))
        self.assertEqual([r['fields']['result'] for r in resumed], [0, 0])
        self.assertEqual(disconnected[0]['fields']['result'], result)
        self.assertEqual(retries[0]['fields']['result'], result)
        # The first rejoin waits below twice the driver's reconnect_initial_ms of 20 ms.
        self.assertLess(retries[0]['fields']['delay_ms'], 40)
        self.assertEqual([(r['fields']['result'], r['fields']['grpc_status'], r['fields']['request_id'],
                           r['fields']['work_id']) for r in completed], [(result, grpc_status, 0, 0), (10, 1, 0, 0)])
        self.assertEqual(data['diagnostics']['metrics']['rpc_attempts_by_operation'], [dict(operation=6, count=2)])
        outcomes = [dict(operation=6, result=10, count=2)] if result == 10 else [
            dict(operation=6, result=result, count=1), dict(operation=6, result=10, count=1)]
        self.assertEqual(data['diagnostics']['metrics']['rpc_by_operation_result'], outcomes)
        records = self.check_message_records(data, [(24, 2, 20), (25, 2, 20), (26, 2, 13)])
        written = [r for r in records if r['event_name'] == 'message_write_completed']
        self.assertEqual([r['fields']['stream_generation'] for r in written], [2, 2])

    async def test_session_reconnect_new_connection_dedup_grace_and_unsent_fifo(self):
        self.check_reconnect(await self.driver("join-reconnect"))

    async def test_session_graceful_server_eof_reconnects_with_unsent_fifo(self):
        self.check_reconnect(await self.driver("join-reconnect-eof"), grpc_status=0)

    async def test_session_http2_cancel_reset_reconnects_with_correlated_attempts(self):
        self.check_reconnect(await self.driver('join-reconnect-reset'), result=10, grpc_status=1, departed=False)
        self.assertEqual(len(self.proxies[0].connections), 1)
        self.assertFalse(self.proxies[0].errors)

    async def test_session_http2_cancel_reset_through_tls_reconnects_with_correlated_attempts(self):
        self.check_reconnect(await self.driver('join-reconnect-reset', tls=True), result=10, grpc_status=1, departed=False)
        self.assertEqual(len(self.proxies[0].connections), 1)
        self.assertFalse(self.proxies[0].errors)

    async def test_session_cancel_before_connected_has_bounded_attempts(self):
        # The driver bounds joins by reconnect_max_attempts (4), not by retry_max_attempts (3).
        # Each cancelled attempt reports CANCELLED; exhausting the attempts fails the session UNAVAILABLE.
        data = await self.driver('join-cancel-before-connected', expected=8)
        self.assertEqual(self.state.attempts['JoinSession'], 4)
        self.assertEqual([(e['kind'], e['result'], e['id']) for e in data['a']], [(1, 10, '')]*3 + [(1, 8, '')])
        records = [r for r in data['diagnostics']['recent_records'] if r['subject_id'] == data['subject']]
        self.assertEqual([r for r in records if r['event_name'] == 'stream_resumed'], [])
        for name, generations in (('rpc_failed', [1, 2, 3, 4]), ('stream_disconnected', [1, 2, 3, 4]),
                                  ('retry_scheduled', [1, 2, 3])):
            self.assertEqual([r['fields']['stream_generation'] for r in records if r['event_name'] == name], generations)
        # The initial join counts as attempt 1, so its rejoins wait in the windows of attempts 2, 3 and 4:
        # [0, 2 x min(50, 20 x 2^(n-1))) with the driver's 20 ms initial and 50 ms maximum.
        delays = [r['fields']['delay_ms'] for r in records if r['event_name'] == 'retry_scheduled']
        for delay, window in zip(delays, (80, 100, 100), strict=True):
            self.assertLess(delay, window, delays)
        completed = [r['fields'] for r in records if r['event_name'] == 'rpc_failed']
        self.assertEqual([(f['session_handle'], f['attempt_id'], f['auth_generation'], f['request_id'],
                           f['work_id'], f['result'], f['grpc_status']) for f in completed],
                         [(data['session_handle'], n, 1, 0, 0, 10, 1) for n in (1, 2, 3, 4)])
        self.assertEqual(data['diagnostics']['metrics']['rpc_attempts_by_operation'], [dict(operation=6, count=4)])
        self.assertEqual(data['diagnostics']['metrics']['rpc_by_operation_result'], [dict(operation=6, result=10, count=4)])

    async def test_session_pending_queue_limit_preserves_accepted_fifo(self):
        data = await self.driver("join-capacity")
        self.assertEqual([e["hex"] for e in self.state.session_wire if e["kind"] == "message"],
                         [b"one".hex(), b"two".hex()])
        self.assertEqual(data["send_callbacks"], 2)
        records = self.check_message_records(data, [(24, 2, 6), (25, 2, 6), (26, 1, 12), (28, 1, 5)])
        rejected = next(r for r in records if r['event_name'] == 'message_rejected')
        self.assertEqual((rejected['fields']['result'], rejected['fields']['request_id'],
                          rejected['fields']['queue_depth'], rejected['fields']['pending_bytes']), (13, 0, 2, 6))

    async def test_session_status_reads_while_send_prepares(self):
        # The driver reads session status from the send's post-preparation
        # seam on the sending thread; a held session mutex would deadlock it.
        data = await self.driver("send-status-under-prepare")
        self.assertEqual(data["send_callbacks"], 1)
        self.assertEqual(len(data["send_requests"]), 1)
        records = [r for r in data['diagnostics']['recent_records'] if r['event_name'].startswith('message_')]
        accepted = [r for r in records if r['event_name'] == 'message_accepted']
        self.assertEqual([r['fields']['request_id'] for r in accepted], data['send_requests'])
        self.assertEqual([r for r in records if r['event_name'] == 'message_rejected'], [])

    async def test_session_leave_during_send_preparation_cancels_without_callback(self):
        # Leave completes from another thread while the send is parked after
        # preparation; the commit sees the closed session and rolls back.
        data = await self.driver("send-leave-during-prepare")
        self.assertEqual(data["send_callbacks"], 0)
        self.assertEqual(data["send_requests"], [])
        self.assertEqual([r for r in self.state.session_wire if r['kind'] == 'message'], [])
        snapshot = data['diagnostics']
        self.assertEqual(snapshot['metrics']['messages_by_event'], [dict(event_code=28, count=1, bytes=9)])
        records = [r for r in snapshot['recent_records'] if r['event_name'].startswith('message_')]
        self.assertEqual([r['event_name'] for r in records], ['message_rejected'])
        fields = records[0]['fields']
        self.assertEqual((fields['result'], fields['request_id'], fields['session_handle'], fields['queue_depth'],
                          fields['pending_bytes']), (10, 0, data['session_handle'], 0, 0))

    def check_message_records(self, data, counts):
        snapshot = data['diagnostics']
        self.assertEqual(snapshot['metrics']['messages_by_event'], [
            dict(event_code=code, count=count, bytes=size) for code, count, size in counts])
        for table in ('rpc_attempts_by_operation', 'rpc_by_operation_result'):
            self.assertFalse(any(row['operation'] == 7 for row in snapshot['metrics'][table]))
        records = [r for r in snapshot['recent_records'] if r['event_name'].startswith('message_')]
        accepted = [r for r in records if r['event_name'] == 'message_accepted']
        self.assertEqual([r['fields']['request_id'] for r in accepted], data['send_requests'])
        for record in records:
            fields = record['fields']
            self.assertEqual(fields['operation'], 7)
            self.assertNotIn('grpc_status', fields)
            owner = (data['receiver_subject'], data['receiver_handle']) if record['event_name'] == 'message_received' else (data['subject'], data['session_handle'])
            self.assertEqual((record['subject_id'], fields['session_handle']), owner)
            if record['event_name'] in ('message_accepted', 'message_write_completed', 'message_received'):
                self.assertEqual((fields['result'], record['severity']), (0, 1))
            if record['event_name'] in ('message_accepted', 'message_rejected', 'message_discarded'):
                self.assertEqual((fields['stream_generation'], fields['auth_generation']), (0, 0))
            else:
                self.assertGreater(fields['stream_generation'], 0)
                self.assertEqual(fields['auth_generation'], 1)
        by_request = {r['fields']['request_id']: r for r in accepted}
        completed = [r for r in records if r['event_name'] in ('message_write_completed', 'message_discarded')]
        self.assertEqual([r['fields']['request_id'] for r in completed], data['send_requests'])
        for record in completed:
            original = by_request[record['fields']['request_id']]
            self.assertLess(original['sequence'], record['sequence'])
            self.assertEqual(original['fields']['bytes'], record['fields']['bytes'])
        return records

    async def test_session_leave_discards_accepted_messages_with_original_receipts(self):
        data = await self.driver('join-discard')
        records = self.check_message_records(data, [(24, 2, 13), (27, 2, 13)])
        self.assertEqual([r['fields']['bytes'] for r in records], [0, 13, 0, 13])
        self.assertEqual([r['fields']['queue_depth'] for r in records], [1, 2, 1, 0])
        self.assertEqual([r['fields']['pending_bytes'] for r in records], [0, 13, 13, 0])
        self.assertEqual([r['fields']['result'] for r in records], [0, 0, 10, 10])
        self.assertEqual([r['severity'] for r in records], [1, 1, 3, 3])
        self.assertEqual([r for r in self.state.session_wire if r['kind'] == 'message'], [])
        self.assertNotIn('discard', json.dumps(data['diagnostics']).replace('message_discarded', ''))

    async def test_session_terminal_failure_discards_queue_without_rewriting_receipts(self):
        data = await self.driver('join-fail-queued', expected=7)
        records = self.check_message_records(data, [(24, 2, 13), (27, 2, 13)])
        self.assertEqual([r['fields']['result'] for r in records], [0, 0, 7, 7])
        self.assertEqual(data['send_callbacks'], 1)
        self.assertEqual([r for r in self.state.session_wire if r['kind'] == 'message'], [])
        self.assertEqual(self.state.attempts['JoinSession'], 1)

    async def test_session_callback_failure_preserves_receipts_and_correlates_diagnostics(self):
        data = await self.driver('join-callback-throw', expected=20)
        message_records = self.check_message_records(data, [(24, 1, 13), (27, 1, 13)])
        self.assertEqual([r['fields']['result'] for r in message_records], [0, 20])
        self.assertEqual(data['send_callbacks'], 1)
        self.assertEqual(len(data['a']), 1)  # Throwing callback is disabled.
        self.assertFalse(data['grpc_status_present'])
        self.assertEqual([r for r in self.state.session_wire if r['kind'] == 'message'], [])
        self.assertEqual(self.state.attempts['JoinSession'], 1)
        records = [r for r in data['diagnostics']['recent_records'] if r['event_code'] == 29]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['event_name'], 'callback_failed')
        self.assertEqual(records[0]['severity'], 4)
        self.assertEqual(records[0]['subject_id'], data['subject'])
        self.assertEqual(records[0]['fields'], dict(callback_kind=3, session_handle=data['session_handle'], reason=1))
        self.assertIn(dict(event_code=29, count=1), data['diagnostics']['metrics']['events'])
        self.assertNotIn('synthetic-session-callback-secret', json.dumps(data))

    async def test_session_message_metrics_survive_disabled_history(self):
        data = await self.driver('join-discard-no-history')
        self.assertEqual(data['diagnostics']['recent_records'], [])
        self.assertEqual(data['diagnostics']['metrics']['messages_by_event'], [
            dict(event_code=24, count=2, bytes=13), dict(event_code=27, count=2, bytes=13)])
        self.assertEqual(data['send_callbacks'], 1)

    async def test_session_leave_waits_for_held_callback(self):
        data = await self.driver("join-held")
        self.assertEqual(len(data["a"]), 1)
        self.assertEqual(data["a"][0]["kind"], 1)

    async def test_session_callback_leave_is_busy_then_quiescent(self):
        data = await self.driver("join-self")
        self.assertEqual(len(data["a"]), 1)

    async def test_session_permanent_permission_failure_is_not_retried(self):
        data = await self.driver("join-permission", expected=7)
        self.assertEqual([(e["kind"], e["result"], e["id"]) for e in data["a"]], [(1, 7, "")])
        self.assertEqual(self.state.attempts["JoinSession"], 1)
        records = data['diagnostics']['recent_records']
        self.assertFalse(any(r['event_name'] in ('stream_resumed', 'retry_scheduled') for r in records))
        failure = [r for r in records if r['event_name'] == 'rpc_failed']
        self.assertEqual(len(failure), 1)
        self.assertEqual((failure[0]['subject_id'], failure[0]['fields']['session_handle'],
                          failure[0]['fields']['result'], failure[0]['fields']['grpc_status']),
                         (data['subject'], data['session_handle'], 7, 7))

    async def test_session_peer_before_connection_is_malformed(self):
        data = await self.driver("join-protocol", expected=11)
        self.assertEqual([(e["kind"], e["result"], e["id"]) for e in data["a"]], [(1, 11, "")])
        self.assertEqual(self.state.attempts["JoinSession"], 1)
        self.assertFalse(any(r['event_name'] in ('stream_resumed', 'retry_scheduled')
                             for r in data['diagnostics']['recent_records']))
        incidents = data['diagnostics']['closed_incidents']
        self.assertEqual(len(incidents), 1)
        self.assertEqual((incidents[0]['subject_id'], incidents[0]['result'], incidents[0]['grpc_status_present']),
                         (data['subject'], 11, 0))
        self.assertGreater(incidents[0]['incident_id'], 0)

    async def test_session_missing_auth_times_out_without_transport(self):
        data = await self.driver("join-auth-timeout", expected=6, token="")
        metric = data['diagnostics']['metrics']['operation_auth_waits'][2]
        self.assertEqual((metric['current'], metric['started'], metric['completed']), (0, 1, 1))
        self.assertEqual([(r['result'], r['count']) for r in metric['duration_ms_by_result']['entries']], [(6, 1)])
        self.assertEqual(self.state.attempts.get("JoinSession", 0), 0)
        self.assertEqual(self.state.session_wire, [])
        self.assertEqual(data["auth_callbacks"], 0)
        self.assertEqual(data["state"], 7)
        self.assertEqual([(e["kind"], e["result"], e["id"]) for e in data["a"]], [(1, 6, "")])

    async def test_session_open_deadline_while_starved_is_unauthenticated(self):
        # The 1 ms initial-open deadline precedes the 30000 ms auth wait; the session records starvation at construction, so the deadline ends UNAUTHENTICATED whenever its first step runs.
        data = await self.driver("join-auth-deadline", expected=6, token="")
        self.assertEqual(self.state.attempts.get("JoinSession", 0), 0)
        self.assertEqual(data["state"], 7)
        self.assertEqual([(e["kind"], e["result"], e["id"]) for e in data["a"]], [(1, 6, "")])

    async def test_session_refresh_unblocks_already_joining_session(self):
        data = await self.driver("join-auth-refresh", token="")
        metric = data['diagnostics']['metrics']['operation_auth_waits'][2]
        self.assertEqual((metric['current'], metric['started'], metric['completed']), (0, 1, 1))
        self.assertEqual([(r['result'], r['count']) for r in metric['duration_ms_by_result']['entries']], [(0, 1)])
        self.assertEqual(self.state.attempts["JoinSession"], 1)
        self.assertEqual(data["auth_callbacks"], 1)
        self.assertEqual([(e["kind"], e["result"]) for e in data["a"]], [(1, 0)])
        self.assertEqual(self.state.session_wire,
                         [{"kind": "header", "session": self.session, "app": "native-alpha", "keepalive": True, "hex": ""}])

    async def test_session_rejoin_after_server_expiry_waits_for_a_new_generation_and_keeps_queued_messages(self):
        data = await self.driver("join-auth-server-expiry")
        self.assertEqual(self.state.attempts["JoinSession"], 3)  # Joined, rejected rejoin, rejoin on generation 2.
        self.assertEqual([(r["method"], r["reason"]) for r in self.state.auth_rejections], [("JoinSession", "expired")])
        connected = [e["id"] for e in data["a"] if e["kind"] == 1 and e["result"] == 0]
        self.assertEqual(len(connected), 2)
        self.assertNotEqual(connected[0], connected[1])
        headers = [i for i, r in enumerate(self.state.session_wire) if r["kind"] == "header"]
        messages = [i for i, r in enumerate(self.state.session_wire) if r["kind"] == "message"]
        self.assertEqual(len(headers), 2)  # The service rejects a rejoin before it reads the join header.
        self.assertEqual([self.state.session_wire[i]["hex"] for i in messages], [b"queued-during-auth".hex()])
        self.assertGreater(messages[0], headers[-1])  # Sent on the accepted rejoin.
        self.assertFalse(self.proxies[0].errors)

    async def test_session_local_failure_clears_prior_attempt_grpc_evidence(self):
        data = await self.driver("join-protocol-after-unavailable", expected=11)
        self.assertEqual(self.state.attempts["JoinSession"], 2)
        self.assertEqual([(e["kind"], e["result"], e["id"]) for e in data["a"]],
                         [(1, 8, ""), (1, 11, "")])
        self.assertIs(data["grpc_status_present"], False)
        self.assertEqual(data["grpc_status"], 0)
        self.assertEqual(data["state"], 7)

    async def test_session_auth_wait_clears_prior_attempt_grpc_evidence(self):
        data = await self.driver("join-revoke-after-unavailable", expected=6)
        self.assertEqual(self.state.attempts["JoinSession"], 1)
        self.assertEqual([(e["kind"], e["result"], e["id"]) for e in data["a"]], [(1, 8, "")])
        self.assertEqual(data["state"], 4)
        self.assertIs(data["grpc_status_present"], False)
        self.assertEqual(data["grpc_status"], 0)
