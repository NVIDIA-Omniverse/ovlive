# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import hashlib
import json
import os
from pathlib import Path
import random
import unittest

import grpc
from service import State, Principal, ReferenceServer, ErrorInfo, RpcStatus, pb, rpc
import tls_fixture

WRITER = (("authorization", "Bearer synthetic-writer"),)
READER = (("authorization", "Bearer synthetic-reader"),)


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = State(int(os.environ.get("LIVE_TEST_SEED", "42")))
        self.state.tokens = {"synthetic-writer": Principal("writer"),
                             "synthetic-reader": Principal("reader", False)}
        self.servers = []
        self.channels = []
        self.calls = []
        self.addAsyncCleanup(self.cleanup)
        self.stub = await self.backend("A")

    async def backend(self, name):
        server = ReferenceServer(self.state, name)
        self.servers.append(server)
        address = await server.start()
        channel = grpc.aio.insecure_channel(address)
        self.channels.append(channel)
        await asyncio.wait_for(channel.channel_ready(), 5)
        return rpc.LiveServiceStub(channel)

    async def cleanup(self):
        for call in self.calls:
            call.cancel()
        failures = await asyncio.gather(*(channel.close() for channel in self.channels), return_exceptions=True)
        failures += await asyncio.gather(*(server.stop() for server in self.servers), return_exceptions=True)
        artifact = Path(os.environ["LIVE_TEST_ARTIFACT_DIR"])
        (artifact / (self.id().split(".")[-1] + ".ledger.json")).write_text(
            json.dumps(self.state.snapshot(), indent=2, sort_keys=True) + "\n")
        self.state.assert_faults_fired()
        for failure in failures:
            if isinstance(failure, BaseException):
                raise failure

    async def create_session(self, name="test", path="file:///base.usda?stable-identity"):
        return (await self.stub.CreateSession(pb.CreateSessionRequest(path=path, name=name),
                                              metadata=WRITER, timeout=5)).session_id

    async def create_layer(self, session, name="root.live", payload=b"base"):
        return (await self.create(session, payload, name=name)).object_id

    async def create(self, session, payload, *, name="root.live", operation=""):
        messages = [pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=session, layer_name=name,
                    operation_id=operation)), pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=payload))]
        return await self.stub.CreateLayer(iter(messages), metadata=WRITER, timeout=5)

    async def update(self, session, object_id, payload, *, writer="client", stub=None, name="root.live",
                     operation=""):
        messages = [pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=session, layer_name=name,
                    object_id=object_id, client_id=writer, timestamp_microseconds_since_epoch=987654321,
                    operation_id=operation)), pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=payload))]
        return await (stub or self.stub).UpdateLayer(iter(messages), metadata=WRITER, timeout=5)

    def subscribe(self, session, object_id=0, sequence=0, writer="", stub=None):
        call = (stub or self.stub).SubscribeReadLayer(pb.SubscribeReadLayerRequest(session_id=session,
            layer_name="root.live", object_id=object_id, sequence_num=sequence, client_id=writer),
            metadata=READER, timeout=10)
        self.calls.append(call)
        return call

    async def content(self, call):
        header = (await asyncio.wait_for(call.read(), 3)).header
        data = bytearray()
        while len(data) < header.total_size:
            message = await asyncio.wait_for(call.read(), 3)
            self.assertEqual(message.WhichOneof("response_type"), "chunk")
            data.extend(message.chunk.data)
        self.assertEqual(len(data), header.total_size)
        return header, bytes(data)

    async def error_code(self, operation, code):
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            await operation
        self.assertEqual(caught.exception.code(), code)

    async def test_session_identity_metadata_duplicates_and_permissions(self):
        first = await self.create_session("duplicate")
        second = await self.create_session("duplicate")
        self.assertNotEqual(first, second)
        result = await self.stub.ListSessions(pb.ListSessionsRequest(path="file:///base.usda?stable-identity"),
                                              metadata=READER, timeout=5)
        self.assertEqual([s.session_id for s in result.sessions], [first, second])
        self.assertEqual([s.created_by_user_name for s in result.sessions], ["writer", "writer"])
        self.assertTrue(all(s.created_at > 0 for s in result.sessions))
        await self.error_code(self.stub.DeleteSession(pb.DeleteSessionRequest(session_id=first),
                             metadata=READER, timeout=5), grpc.StatusCode.PERMISSION_DENIED)
        await self.stub.DeleteSession(pb.DeleteSessionRequest(session_id=first), metadata=WRITER, timeout=5)
        self.assertEqual(len(self.state.ledger), 3)

    async def test_mutation_committed_before_lost_response_and_rejection_before_commit(self):
        fault = self.state.plan("CreateSession", "after_commit", code=grpc.StatusCode.UNAVAILABLE)
        await self.error_code(self.stub.CreateSession(pb.CreateSessionRequest(path="base", name="lost"),
                             metadata=WRITER, timeout=5), grpc.StatusCode.UNAVAILABLE)
        self.assertTrue(fault.reached.is_set())
        self.assertEqual(len(self.state.ledger), 1)
        self.assertEqual(len(self.state.sessions), 1)
        self.state.plan("CreateSession", "before", code=grpc.StatusCode.UNAVAILABLE)
        await self.error_code(self.stub.CreateSession(pb.CreateSessionRequest(path="base", name="rejected"),
                             metadata=WRITER, timeout=5), grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(len(self.state.ledger), 1)
        sessions = await self.stub.ListSessions(pb.ListSessionsRequest(path="base"), metadata=READER, timeout=5)
        self.assertEqual([s.name for s in sessions.sessions], ["lost"])

    async def test_update_resent_with_its_operation_id_after_a_lost_reply_is_not_applied_twice(self):
        session = await self.create_session()
        object_id = await self.create_layer(session)
        plain = await self.update(session, object_id, b"plain")
        self.assertEqual((plain.sequence_num, plain.operation_outcome), (2, pb.OPERATION_OUTCOME_UNSPECIFIED))
        self.state.plan("UpdateLayer", "after_commit", code=grpc.StatusCode.UNAVAILABLE)
        await self.error_code(self.update(session, object_id, b"lost", operation="op-1"), grpc.StatusCode.UNAVAILABLE)
        committed = self.state.ledger[-1]
        self.assertEqual((committed.sequence, committed.payload), (3, b"lost"))
        peer = await self.update(session, object_id, b"peer", writer="peer", operation="op-1")
        self.assertEqual((peer.sequence_num, peer.operation_outcome), (4, pb.OPERATION_OUTCOME_APPLIED))
        # The re-send earns the original commit's reply, though the layer has moved past it.
        again = await self.update(session, object_id, b"lost", operation="op-1")
        self.assertEqual((again.sequence_num, again.server_timestamp_microseconds_since_epoch,
                          again.operation_outcome), (3, committed.timestamp_us, pb.OPERATION_OUTCOME_ALREADY_COMMITTED))
        await self.error_code(self.update(session, object_id, b"other", operation="op-1"),
                              grpc.StatusCode.ALREADY_EXISTS)
        self.assertEqual(len(self.state.ledger), 5)
        following = await self.update(session, object_id, b"next", operation="op-2")
        self.assertEqual((following.sequence_num, following.operation_outcome), (5, pb.OPERATION_OUTCOME_APPLIED))
        # A re-send matches only before the client's next write to the layer.
        late = await self.update(session, object_id, b"lost", operation="op-1")
        self.assertEqual((late.sequence_num, late.operation_outcome), (6, pb.OPERATION_OUTCOME_APPLIED))
        self.assertEqual([entry["outcome"] for entry in self.state.snapshot()["operations"]],
                         ["APPLIED", "APPLIED", "ALREADY_COMMITTED", "ALREADY_EXISTS", "APPLIED", "APPLIED"])

    async def test_operation_id_resend_before_commit_applies_and_identity_checks_precede_matching(self):
        session = await self.create_session()
        object_id = await self.create_layer(session)
        self.state.plan("UpdateLayer", "before", code=grpc.StatusCode.UNAVAILABLE)
        await self.error_code(self.update(session, object_id, b"edit", operation="op-1"), grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(len(self.state.ledger), 2)
        applied = await self.update(session, object_id, b"edit", operation="op-1")
        self.assertEqual((applied.sequence_num, applied.operation_outcome), (2, pb.OPERATION_OUTCOME_APPLIED))
        for invalid in ("x" * 65, "a/b", "spa ce", "é"):
            await self.error_code(self.update(session, object_id, b"edit", operation=invalid),
                                  grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(len(self.state.ledger), 3)
        allowed = await self.update(session, object_id, b"edit", operation="A-z.0_9~" + "x" * 56)
        self.assertEqual(allowed.operation_outcome, pb.OPERATION_OUTCOME_APPLIED)
        replaced = await self.create_layer(session, payload=b"replacement")
        self.assertNotEqual(replaced, object_id)
        await self.error_code(self.update(session, object_id, b"edit", operation="A-z.0_9~" + "x" * 56),
                              grpc.StatusCode.FAILED_PRECONDITION)
        await self.stub.DeleteLayer(pb.DeleteLayerRequest(session_id=session, layer_name="root.live"),
                                    metadata=WRITER, timeout=5)
        await self.error_code(self.update(session, replaced, b"edit", operation="A-z.0_9~" + "x" * 56),
                              grpc.StatusCode.NOT_FOUND)

    async def test_create_resent_with_its_operation_id_returns_the_original_object(self):
        session = await self.create_session()
        self.state.plan("CreateLayer", "after_commit", code=grpc.StatusCode.UNAVAILABLE)
        await self.error_code(self.create(session, b"seed", operation="create-1"), grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(len(self.state.ledger), 2)
        again = await self.create(session, b"seed", operation="create-1")
        self.assertEqual((again.object_id, again.operation_outcome), (1, pb.OPERATION_OUTCOME_ALREADY_COMMITTED))
        await self.error_code(self.create(session, b"other", operation="create-1"), grpc.StatusCode.ALREADY_EXISTS)
        self.assertEqual(len(self.state.ledger), 2)
        fresh = await self.create(session, b"seed", operation="create-2")
        self.assertEqual((fresh.object_id, fresh.operation_outcome), (2, pb.OPERATION_OUTCOME_APPLIED))
        plain = await self.create(session, b"seed")
        self.assertEqual((plain.object_id, plain.operation_outcome), (3, pb.OPERATION_OUTCOME_UNSPECIFIED))

    async def test_create_resend_matches_by_session_and_id_across_replace_and_delete(self):
        session = await self.create_session()
        created = await self.create(session, b"seed", operation="create-1")
        self.assertEqual((created.object_id, created.operation_outcome), (1, pb.OPERATION_OUTCOME_APPLIED))
        replaced = await self.create(session, b"other writer")
        self.assertEqual(replaced.object_id, 2)
        # The record survives a replace: the re-send earns the original object, which the client then finds replaced.
        again = await self.create(session, b"seed", operation="create-1")
        self.assertEqual((again.object_id, again.operation_outcome), (1, pb.OPERATION_OUTCOME_ALREADY_COMMITTED))
        await self.stub.DeleteLayer(pb.DeleteLayerRequest(session_id=session, layer_name="root.live"),
                                    metadata=WRITER, timeout=5)
        deleted = await self.create(session, b"seed", operation="create-1")
        self.assertEqual((deleted.object_id, deleted.operation_outcome), (1, pb.OPERATION_OUTCOME_ALREADY_COMMITTED))
        self.assertNotIn("root.live", self.state.sessions[session].layers)
        # The content covers the layer name: the same ID and bytes under another name is ALREADY_EXISTS.
        await self.error_code(self.create(session, b"seed", name="other.live", operation="create-1"),
                              grpc.StatusCode.ALREADY_EXISTS)
        # The key is (session, operation_id): another session does not match.
        elsewhere = await self.create_session("elsewhere")
        fresh = await self.create(elsewhere, b"seed", operation="create-1")
        self.assertEqual((fresh.object_id, fresh.operation_outcome), (1, pb.OPERATION_OUTCOME_APPLIED))
        self.assertEqual([c.operation for c in self.state.ledger],
                         ["CreateSession", "CreateLayer", "CreateLayer", "DeleteLayer", "CreateSession", "CreateLayer"])

    async def test_resend_content_is_the_concatenated_payload_whatever_its_chunking(self):
        session = await self.create_session()
        object_id = await self.create_layer(session)

        async def update(chunks, timestamp=1):
            header = pb.UpdateLayerHeader(session_id=session, layer_name="root.live", object_id=object_id,
                                          client_id="client", timestamp_microseconds_since_epoch=timestamp,
                                          operation_id="chunked")
            messages = [pb.UpdateLayerRequest(header=header)] + [
                pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=chunk)) for chunk in chunks]
            return await self.stub.UpdateLayer(iter(messages), metadata=WRITER, timeout=5)

        first = await update([b"ab", b"cd", b"ef"])
        self.assertEqual((first.sequence_num, first.operation_outcome), (2, pb.OPERATION_OUTCOME_APPLIED))
        # Other header fields, such as the timestamp estimate, do not count either.
        again = await update([b"abcdef"], timestamp=2)
        self.assertEqual((again.sequence_num, again.operation_outcome), (2, pb.OPERATION_OUTCOME_ALREADY_COMMITTED))
        await self.error_code(update([b"abcdeg"]), grpc.StatusCode.ALREADY_EXISTS)
        self.assertEqual(len(self.state.ledger), 3)

    async def test_an_unidentified_update_by_the_writer_ends_its_last_operations_match(self):
        session = await self.create_session()
        object_id = await self.create_layer(session)
        first = await self.update(session, object_id, b"edit", operation="op-A")
        self.assertEqual((first.sequence_num, first.operation_outcome), (2, pb.OPERATION_OUTCOME_APPLIED))
        plain = await self.update(session, object_id, b"plain")
        self.assertEqual((plain.sequence_num, plain.operation_outcome), (3, pb.OPERATION_OUTCOME_UNSPECIFIED))
        # The writer's next write replaced the record, so A again is a new write.
        again = await self.update(session, object_id, b"edit", operation="op-A")
        self.assertEqual((again.sequence_num, again.operation_outcome), (4, pb.OPERATION_OUTCOME_APPLIED))

    async def test_older_server_ignores_operation_ids_and_applies_a_resend_again(self):
        self.state.operation_ids = False
        session = await self.create_session()
        created = await self.create(session, b"seed", operation="create-1")
        self.assertEqual((created.object_id, created.operation_outcome), (1, pb.OPERATION_OUTCOME_UNSPECIFIED))
        self.state.plan("UpdateLayer", "after_commit", code=grpc.StatusCode.UNAVAILABLE)
        await self.error_code(self.update(session, 1, b"lost", operation="op-1"), grpc.StatusCode.UNAVAILABLE)
        again = await self.update(session, 1, b"lost", operation="op-1")
        self.assertEqual((again.sequence_num, again.operation_outcome), (3, pb.OPERATION_OUTCOME_UNSPECIFIED))
        self.assertEqual([c.payload for c in self.state.ledger[-2:]], [b"lost", b"lost"])
        unchecked = await self.update(session, 1, b"other", operation="not/valid")
        self.assertEqual(unchecked.operation_outcome, pb.OPERATION_OUTCOME_UNSPECIFIED)
        self.assertEqual(self.state.snapshot()["operations"], [])

    async def test_named_pause_and_auth_rotation(self):
        fault = self.state.plan("CreateSession", "after_commit", pause=True)
        call = self.stub.CreateSession(pb.CreateSessionRequest(path="base", name="paused"), metadata=WRITER, timeout=5)
        await asyncio.wait_for(fault.reached.wait(), 3)
        self.assertEqual(len(self.state.ledger), 1)
        self.assertFalse(call.done())
        fault.release.set()
        await call
        del self.state.tokens["synthetic-writer"]
        await self.error_code(self.stub.Ping(pb.PingRequest(), metadata=WRITER, timeout=5), grpc.StatusCode.UNAUTHENTICATED)
        self.state.tokens["synthetic-new"] = Principal("writer")
        response = await self.stub.Ping(pb.PingRequest(client_timestamp_microseconds_since_epoch=123456789),
            metadata=(("authorization", "Bearer synthetic-new"),), timeout=5)
        self.assertEqual(response.client_timestamp_microseconds_since_epoch, 123456789)
        self.assertGreater(response.server_timestamp_microseconds_since_epoch, 0)

    async def test_server_side_expiry_rejects_the_next_rpc_start_with_an_empty_message(self):
        self.state.issue("expiring", "writer")
        token = (("authorization", "Bearer expiring"),)
        session = (await self.stub.CreateSession(pb.CreateSessionRequest(path="base", name="before"),
                                                 metadata=token, timeout=5)).session_id
        first = await self.create_layer(session)
        read = self.stub.SubscribeReadLayer(pb.SubscribeReadLayerRequest(session_id=session, layer_name="root.live"),
                                            metadata=token, timeout=10)
        self.calls.append(read)
        caught_up = False
        while not caught_up:  # Established before expiry: the base frames and the caught-up header.
            message = await read.read()
            caught_up = message.WhichOneof("response_type") == "header" and message.header.total_size == 0
        self.state.expire("expiring")
        await self.update(session, first, b"after-expiry")  # Another principal commits sequence 2.
        while True:  # The open stream is never re-authenticated.
            message = await read.read()
            if message.WhichOneof("response_type") == "header" and message.header.sequence_num == 2:
                break
        with self.assertRaises(grpc.aio.AioRpcError) as rejected:
            await self.stub.CreateSession(pb.CreateSessionRequest(path="base", name="after"), metadata=token, timeout=5)
        self.assertEqual(rejected.exception.code(), grpc.StatusCode.UNAUTHENTICATED)
        self.assertFalse(rejected.exception.details())  # The real service sends no message or challenge.
        await self.stub.Ping(pb.PingRequest(), metadata=token, timeout=5)  # Ping checks no expiry.
        self.state.issue("pre-expired", "writer", expires_us=0)
        await self.error_code(self.stub.ListSessions(pb.ListSessionsRequest(path="base"),
            metadata=(("authorization", "Bearer pre-expired"),), timeout=5), grpc.StatusCode.UNAUTHENTICATED)
        await self.error_code(self.stub.Ping(pb.PingRequest(), metadata=(("authorization", "Bearer unknown"),),
                                             timeout=5), grpc.StatusCode.UNAUTHENTICATED)
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        self.assertEqual([(r["method"], r["reason"], r["token_sha256"]) for r in self.state.auth_rejections],
                         [("CreateSession", "expired", digest("expiring")), ("ListSessions", "expired", digest("pre-expired")),
                          ("Ping", "unknown", digest("unknown"))])
        self.assertEqual(self.state.snapshot()["auth_rejections"], self.state.auth_rejections)
        self.assertNotIn("expiring", json.dumps(self.state.snapshot()))  # Hashes only, never a bearer.

    async def test_explicit_expiry_ignores_the_clock_and_a_deadline_follows_it(self):
        now = [1000]
        self.state.clock = lambda: now[0]
        self.state.issue("stepped", "writer")
        self.state.expire("stepped")
        now[0] = 0  # The wall clock steps back after expire().
        await self.error_code(self.stub.ListSessions(pb.ListSessionsRequest(path="base"),
            metadata=(("authorization", "Bearer stepped"),), timeout=5), grpc.StatusCode.UNAUTHENTICATED)
        self.state.issue("deadline", "writer", expires_us=500)
        deadline = (("authorization", "Bearer deadline"),)
        await self.stub.ListSessions(pb.ListSessionsRequest(path="base"), metadata=deadline, timeout=5)
        now[0] = 500
        await self.error_code(self.stub.ListSessions(pb.ListSessionsRequest(path="base"), metadata=deadline,
                                                     timeout=5), grpc.StatusCode.UNAUTHENTICATED)
        now[0] = 499  # A deadline set by expires_us compares against the clock.
        await self.stub.ListSessions(pb.ListSessionsRequest(path="base"), metadata=deadline, timeout=5)
        self.assertEqual([(r["method"], r["reason"]) for r in self.state.auth_rejections],
                         [("ListSessions", "expired")] * 2)

    async def test_before_auth_gate_holds_a_request_ahead_of_its_credential_check(self):
        gate = self.state.plan("Ping", "before_auth", pause=True)
        call = self.stub.Ping(pb.PingRequest(), metadata=WRITER, timeout=5)
        await asyncio.wait_for(gate.reached.wait(), 3)
        del self.state.tokens["synthetic-writer"]  # Revoked while the request is held.
        gate.release.set()
        await self.error_code(call, grpc.StatusCode.UNAUTHENTICATED)
        self.assertEqual([(r["method"], r["attempt"], r["reason"]) for r in self.state.auth_rejections],
                         [("Ping", 1, "unknown")])

    async def test_ingress_rejection_reaches_grpc_as_unauthenticated_and_keeps_the_connection(self):
        from http2_proxy import Http2Proxy
        for kind in ("http401", "envoy"):
            proxy = Http2Proxy(f"127.0.0.1:{self.servers[0].port}")  # ReferenceServer.port is the bound port.
            self.addAsyncCleanup(proxy.close)  # Before the channels close: the proxy aborts its connections.
            channel = grpc.aio.insecure_channel(await proxy.start())
            self.channels.append(channel)
            stub = rpc.LiveServiceStub(channel)
            proxy.reject_new_streams(kind)
            ledger = len(self.state.ledger)
            await self.error_code(stub.CreateSession(pb.CreateSessionRequest(path="base", name=kind),
                                                     metadata=WRITER, timeout=5), grpc.StatusCode.UNAUTHENTICATED)
            self.assertEqual(len(self.state.ledger), ledger)
            await stub.Ping(pb.PingRequest(), metadata=WRITER, timeout=5)  # Same connection: HPACK and windows intact.
            self.assertEqual(len(proxy.connections), 1)

    async def test_layer_incarnations_stream_validation_and_delete(self):
        session = await self.create_session()
        first = await self.create_layer(session)
        other = await self.create_layer(session, name="other.live")
        self.assertEqual(first, other)  # Deliberate cross-resource object-ID collision.
        malformed = [pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=b"no header"))]
        await self.error_code(self.stub.UpdateLayer(iter(malformed), metadata=WRITER, timeout=5), grpc.StatusCode.INVALID_ARGUMENT)
        reply = await self.update(session, first, b"delta")
        self.assertEqual(reply.sequence_num, 2)
        stat = await self.stub.StatLayer(pb.StatLayerRequest(session_id=session, layer_name="root.live"), metadata=READER, timeout=5)
        self.assertEqual(stat.created_by_user_name, "writer")
        commits = self.state.sessions[session].layers["root.live"].commits
        self.assertEqual(stat.created_at, commits[0].timestamp_us)
        self.assertEqual(stat.last_modified_at, commits[-1].timestamp_us)
        second = await self.create_layer(session, payload=b"replacement")
        self.assertNotEqual(first, second)
        await self.error_code(self.update(session, first, b"stale"), grpc.StatusCode.FAILED_PRECONDITION)
        await self.stub.DeleteLayer(pb.DeleteLayerRequest(session_id=session, layer_name="root.live"), metadata=WRITER, timeout=5)
        await self.error_code(self.stub.StatLayer(pb.StatLayerRequest(session_id=session, layer_name="root.live"),
                             metadata=READER, timeout=5), grpc.StatusCode.NOT_FOUND)

    async def test_initial_history_resume_filtering_and_backend_rotation(self):
        session = await self.create_session()
        object_id = await self.create_layer(session, payload=b"base")
        await self.update(session, object_id, b"own", writer="self")
        await self.update(session, object_id, b"peer", writer="peer")
        fresh = self.subscribe(session, writer="self")
        content = [await self.content(fresh) for _ in range(4)]
        self.assertEqual([value for _, value in content], [b"base", b"own", b"peer", b""])
        self.assertEqual([header.sequence_num for header, _ in content], [1, 2, 3, 3])
        fresh.cancel()
        second = await self.backend("B")
        # Catch-up replays every retained update, including the subscriber's
        # own, and ends with one caught-up header at the latest sequence.
        resumed = self.subscribe(session, object_id, 1, "self", second)
        content = [await self.content(resumed) for _ in range(3)]
        self.assertEqual([(header.sequence_num, value) for header, value in content],
                         [(2, b"own"), (3, b"peer"), (3, b"")])
        self.assertEqual(content[-1][0].server_timestamp_microseconds_since_epoch, 0)
        quiet = self.subscribe(session, object_id, 3, "self", second)
        header, payload = await self.content(quiet)
        self.assertEqual((header.sequence_num, payload), (3, b""))
        await self.update(session, object_id, b"live-own", writer="self", stub=second)
        await self.update(session, object_id, b"live-peer", writer="peer", stub=second)
        header, payload = await self.content(quiet)
        self.assertEqual((header.sequence_num, payload), (5, b"live-peer"))

    async def test_cursor_ahead_of_latest_gets_caught_up_header_at_latest(self):
        session = await self.create_session()
        object_id = await self.create_layer(session, payload=b"base")
        ahead = self.subscribe(session, object_id, 10, "self")
        # Header content only: its sequence is below the cursor, which a client
        # treats as a server sequence regression (design section 12).
        header, payload = await self.content(ahead)
        self.assertEqual((header.object_id, header.sequence_num, header.total_size, payload), (object_id, 1, 0, b""))

    async def test_many_concurrent_writers_have_exact_ledger_accounting(self):
        session = await self.create_session()
        object_id = await self.create_layer(session)
        rng = random.Random(int(os.environ.get("LIVE_TEST_SEED", "42")))
        operations = [(f"writer-{i % 8}", rng.randbytes(rng.randrange(1, 100))) for i in range(128)]
        history = Path(os.environ["LIVE_TEST_ARTIFACT_DIR"]) / "reference-writers.json"
        history.write_text(json.dumps({"seed": int(os.environ.get("LIVE_TEST_SEED", "42")),
            "operations": [{"writer": writer, "payload_hex": payload.hex()} for writer, payload in operations]},
            indent=2) + "\n")
        replies = await asyncio.gather(*(self.update(session, object_id, payload, writer=writer)
                                        for writer, payload in operations))
        self.assertEqual(sorted(r.sequence_num for r in replies), list(range(2, 130)))
        commits = [c for c in self.state.ledger if c.operation == "UpdateLayer"]
        self.assertEqual(sorted((c.writer, c.payload) for c in commits), sorted(operations))
        self.assertEqual([c.sequence for c in commits], list(range(2, 130)))
        observer = self.subscribe(session)
        observed = [await self.content(observer) for _ in range(129)]
        self.assertEqual([value for _, value in observed], [b"base"] + [c.payload for c in commits])

    async def test_presence_connection_identity_binary_messages_and_leave(self):
        session = await self.create_session()
        first = self.stub.JoinSession(metadata=WRITER, timeout=10)
        self.calls.append(first)
        await first.write(pb.JoinSessionRequest(header=pb.JoinSessionHeader(session_id=session, user_application="one")))
        connected = await asyncio.wait_for(first.read(), 3)
        first_id = connected.connected.connection_id
        self.assertEqual(connected.WhichOneof("response_type"), "connected")
        second = self.stub.JoinSession(metadata=WRITER, timeout=10)
        self.calls.append(second)
        await second.write(pb.JoinSessionRequest(header=pb.JoinSessionHeader(session_id=session, user_application="two")))
        connected = await asyncio.wait_for(second.read(), 3)
        second_id = connected.connected.connection_id
        self.assertNotEqual(first_id, second_id)
        old_peer = await asyncio.wait_for(second.read(), 3)
        new_peer = await asyncio.wait_for(first.read(), 3)
        self.assertEqual(old_peer.peer_joined.connection_id, first_id)
        self.assertEqual(new_peer.peer_joined.connection_id, second_id)
        self.assertEqual(new_peer.peer_joined.user_name, "writer")
        await first.write(pb.JoinSessionRequest(message=pb.SessionMessageRequest(data=b"a\0b")))
        message = await asyncio.wait_for(second.read(), 3)
        self.assertEqual(message.peer_message.data, b"a\0b")
        self.assertEqual(message.peer_message.connection_id, first_id)
        first.cancel()
        left = await asyncio.wait_for(second.read(), 3)
        self.assertEqual(left.peer_left.connection_id, first_id)

    async def join(self, session, app, *, keepalive=False):
        call = self.stub.JoinSession(metadata=WRITER, timeout=10)
        self.calls.append(call)
        await call.write(pb.JoinSessionRequest(header=pb.JoinSessionHeader(
            session_id=session, user_application=app, client_keepalive=keepalive)))
        response = await asyncio.wait_for(call.read(), 3)
        self.assertEqual(response.WhichOneof("response_type"), "connected")
        return call, response.connected

    async def event(self, call):
        response = await asyncio.wait_for(call.read(), 3)
        kind = response.WhichOneof("response_type")
        return kind, getattr(response, kind).connection_id

    async def keepalives(self, connection, count):
        async with self.state.changed:
            await asyncio.wait_for(self.state.changed.wait_for(
                lambda: self.state.keepalives.get(connection, 0) >= count), 3)

    async def test_client_keepalive_is_opt_in_and_reports_its_interval(self):
        session = await self.create_session()
        # The default models a service without the client keepalive: no interval, and a JoinSessionKeepalive frame
        # is unknown.
        older, connected = await self.join(session, "older", keepalive=True)
        self.assertEqual(connected.keepalive_interval_ms, 0)
        await older.write(pb.JoinSessionRequest(keepalive=pb.JoinSessionKeepalive()))
        await self.error_code(older.read(), grpc.StatusCode.INVALID_ARGUMENT)
        self.state.keepalive_interval_ms = 15000
        _, plain = await self.join(session, "plain")
        _, opted = await self.join(session, "opted", keepalive=True)
        self.assertEqual((plain.keepalive_interval_ms, opted.keepalive_interval_ms), (0, 15000))
        self.assertEqual([(j["client_keepalive"], j["keepalive_interval_ms"]) for j in self.state.joins],
                         [(True, 0), (False, 0), (True, 15000)])
        # A join that did not opt in never times out.
        self.assertEqual(self.state.advance_keepalive_clock(10 ** 9), [opted.connection_id])

    async def test_keepalive_timeout_ends_a_silent_join_unavailable_with_one_peer_left(self):
        session = await self.create_session()
        self.state.keepalive_interval_ms = 1000
        observer, watching = await self.join(session, "observer")
        silent, quiet = await self.join(session, "silent", keepalive=True)
        active, busy = await self.join(session, "active", keepalive=True)
        self.assertEqual([await self.event(observer), await self.event(observer)],
                         [("peer_joined", quiet.connection_id), ("peer_joined", busy.connection_id)])
        self.assertEqual([await self.event(active), await self.event(active)],
                         [("peer_joined", watching.connection_id), ("peer_joined", quiet.connection_id)])
        self.assertEqual(self.state.advance_keepalive_clock(2999), [])
        # A keepalive restarts the timeout and reaches no peer; any client message counts too.
        await active.write(pb.JoinSessionRequest(keepalive=pb.JoinSessionKeepalive()))
        await self.keepalives(busy.connection_id, 1)
        self.assertEqual(self.state.advance_keepalive_clock(1), [quiet.connection_id])
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            while True:
                await asyncio.wait_for(silent.read(), 3)
        self.assertEqual(caught.exception.code(), grpc.StatusCode.UNAVAILABLE)
        details = [value for key, value in caught.exception.trailing_metadata() if key == "grpc-status-details-bin"]
        self.assertEqual(len(details), 1)
        status = RpcStatus.FromString(details[0])
        self.assertEqual(status.code, grpc.StatusCode.UNAVAILABLE.value[0])
        self.assertEqual([detail.type_url for detail in status.details], ["type.googleapis.com/google.rpc.ErrorInfo"])
        info = ErrorInfo.FromString(status.details[0].value)
        self.assertEqual((info.reason, info.domain), ("CLIENT_KEEPALIVE_TIMEOUT", "nvidia.omniverse.live.v1alpha"))
        # Each peer receives exactly one PeerLeft: the next event after it is a later message.
        self.assertEqual(await self.event(observer), ("peer_left", quiet.connection_id))
        self.assertEqual(await self.event(active), ("peer_left", quiet.connection_id))
        await active.write(pb.JoinSessionRequest(message=pb.SessionMessageRequest(data=b"after")))
        self.assertEqual(await self.event(observer), ("peer_message", busy.connection_id))
        self.assertEqual(self.state.advance_keepalive_clock(2999), [])
        self.assertEqual(self.state.advance_keepalive_clock(1), [busy.connection_id])
        self.assertEqual(await self.event(observer), ("peer_left", busy.connection_id))
        self.assertEqual(self.state.keepalive_timeouts, [quiet.connection_id, busy.connection_id])
        self.assertEqual(self.state.keepalives, {busy.connection_id: 1})
        self.assertEqual(list(self.state.sessions[session].peers), [watching.connection_id])

    async def test_capacity_rejection_preserves_ledger_and_unused_fault_fails(self):
        session = await self.create_session()
        self.state.max_payload = 2
        await self.error_code(self.create_layer(session, payload=b"oversized"), grpc.StatusCode.RESOURCE_EXHAUSTED)
        self.assertEqual(len(self.state.ledger), 1)
        self.state.plan("Ping", "before_reply", code=grpc.StatusCode.UNAVAILABLE)
        with self.assertRaisesRegex(AssertionError, "exactly once"):
            self.state.assert_faults_fired()
        await self.error_code(self.stub.Ping(pb.PingRequest(), metadata=READER, timeout=5), grpc.StatusCode.UNAVAILABLE)

    async def test_join_header_half_close_and_deleted_session_status(self):
        session = await self.create_session()
        closed = self.stub.JoinSession(metadata=WRITER, timeout=5)
        self.calls.append(closed)
        await closed.write(pb.JoinSessionRequest(header=pb.JoinSessionHeader(session_id=session, user_application="one")))
        await closed.done_writing()
        first = await asyncio.wait_for(closed.read(), 3)
        self.assertEqual(first.WhichOneof("response_type"), "connected")
        self.assertEqual(await asyncio.wait_for(closed.read(), 3), grpc.aio.EOF)
        joined = self.stub.JoinSession(metadata=WRITER, timeout=5)
        self.calls.append(joined)
        await joined.write(pb.JoinSessionRequest(header=pb.JoinSessionHeader(session_id=session, user_application="two")))
        self.assertEqual((await joined.read()).WhichOneof("response_type"), "connected")
        await self.stub.DeleteSession(pb.DeleteSessionRequest(session_id=session), metadata=WRITER, timeout=5)
        await self.error_code(joined.read(), grpc.StatusCode.NOT_FOUND)

    async def test_live_read_replacement_and_deletion(self):
        session = await self.create_session()
        first = await self.create_layer(session)
        subscription = self.subscribe(session)
        header, _ = await self.content(subscription)
        self.assertEqual(header.object_id, first)
        header, payload = await self.content(subscription)
        self.assertEqual((header.object_id, header.sequence_num, payload), (first, 1, b""))
        second = await self.create_layer(session, payload=b"replacement")
        # A replacement incarnation re-bootstraps the subscriber with its own caught-up header.
        header, payload = await self.content(subscription)
        self.assertEqual((header.object_id, header.sequence_num, payload), (second, 1, b"replacement"))
        header, payload = await self.content(subscription)
        self.assertEqual((header.object_id, header.sequence_num, payload), (second, 1, b""))
        await self.stub.DeleteLayer(pb.DeleteLayerRequest(session_id=session, layer_name="root.live"), metadata=WRITER, timeout=5)
        await self.error_code(subscription.read(), grpc.StatusCode.NOT_FOUND)

    async def test_paused_full_join_queue_half_close_and_deletion_race(self):
        session = await self.create_session()
        self.state.peer_queue = 1
        for deleted in (False, True):
            fault = self.state.plan("JoinSession", "before_reply", pause=True)
            call = self.stub.JoinSession(metadata=WRITER, timeout=5)
            self.calls.append(call)
            await call.write(pb.JoinSessionRequest(header=pb.JoinSessionHeader(session_id=session, user_application="app")))
            await asyncio.wait_for(fault.reached.wait(), 3)
            peer = next(iter(self.state.sessions[session].peers.values()))
            await call.done_writing()
            await asyncio.wait_for(peer.input_closed.wait(), 3)
            if deleted:
                await self.stub.DeleteSession(pb.DeleteSessionRequest(session_id=session), metadata=WRITER, timeout=5)
            fault.release.set()
            if deleted:
                await self.error_code(call.read(), grpc.StatusCode.NOT_FOUND)
            else:
                self.assertEqual((await call.read()).WhichOneof("response_type"), "connected")
                self.assertEqual(await call.read(), grpc.aio.EOF)

    tls_pki = tls_fixture.case_pki

    async def test_tls_verified_name_override_and_bad_trust(self):
        pki, other = await self.tls_pki(), await self.tls_pki("other")
        server = ReferenceServer(self.state, "TLS", credentials=grpc.ssl_server_credentials(
            [(pki.key.read_bytes(), pki.certificate.read_bytes())]))
        self.servers.append(server)
        address = await server.start()
        channel = grpc.aio.secure_channel(address, grpc.ssl_channel_credentials(pki.ca.read_bytes()),
                                           options=(("grpc.ssl_target_name_override", "live.test"),))
        self.channels.append(channel)
        await asyncio.wait_for(channel.channel_ready(), 5)
        response = await rpc.LiveServiceStub(channel).Ping(pb.PingRequest(client_timestamp_microseconds_since_epoch=19),
                                                         metadata=READER, timeout=3)
        self.assertEqual(response.client_timestamp_microseconds_since_epoch, 19)
        for roots, name in ((pki.ca.read_bytes(), "wrong.test"), (None, "live.test"), (other.ca.read_bytes(), "live.test")):
            bad = grpc.aio.secure_channel(address, grpc.ssl_channel_credentials(roots),
                                          options=(("grpc.ssl_target_name_override", name),))
            self.channels.append(bad)
            await self.error_code(rpc.LiveServiceStub(bad).Ping(pb.PingRequest(), metadata=READER, timeout=3),
                                  grpc.StatusCode.UNAVAILABLE)

    async def test_tls_proxy_verifies_the_test_ca_and_name(self):
        from http2_proxy import Http2Proxy
        pki, other = await self.tls_pki(), await self.tls_pki("other")
        proxy = Http2Proxy(f"127.0.0.1:{self.servers[0].port}", ssl_context=pki.server_context())
        self.addAsyncCleanup(proxy.close)  # Before the channels close; close() raises the proxy's errors.
        address = await proxy.start()
        cases = ((pki, "live.test", None), (pki, "wrong.test", grpc.StatusCode.UNAVAILABLE),
                 (other, "live.test", grpc.StatusCode.UNAVAILABLE))
        for trust, name, code in cases:
            channel = grpc.aio.secure_channel(address, grpc.ssl_channel_credentials(trust.ca.read_bytes()),
                                              options=(("grpc.ssl_target_name_override", name),))
            self.channels.append(channel)
            call = rpc.LiveServiceStub(channel).Ping(pb.PingRequest(client_timestamp_microseconds_since_epoch=23),
                                                     metadata=READER, timeout=3)
            if code is None:
                self.assertEqual((await call).client_timestamp_microseconds_since_epoch, 23)
            else:
                await self.error_code(call, code)
        self.assertEqual(self.state.attempts, {"Ping": 1})
        await proxy.close()  # A client that rejects the name closes before its HTTP/2 preface: ordinary teardown.
        self.assertEqual(proxy.errors, [])
