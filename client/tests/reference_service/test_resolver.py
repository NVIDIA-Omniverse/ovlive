# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Production Ar URI resolver + native metadata against synthetic loopback.

This lane supplies no file format and never subscribes to scene content.
"""
import asyncio
import json
import os
from pathlib import Path
import unittest
from urllib.parse import quote

import grpc
from service import State, Principal, ReferenceServer, LiveService, pb, rpc

WRITER = (("authorization", "Bearer synthetic-writer"),)


class MetadataService(LiveService):
    async def StatLayer(self, request, context):
        self.state.metadata_requests.append({"session": request.session_id, "layer": request.layer_name})
        response = await super().StatLayer(request, context)
        if self.state.timestamp_script:
            response.last_modified_at = self.state.timestamp_script.pop(0)
        self.state.metadata_replies.append(response.last_modified_at)
        return response


class ResolverTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seed = int(os.environ["LIVE_TEST_SEED"])
        self.state = State(self.seed, clock=lambda: 1700000000123456)
        self.state.tokens = {"synthetic-writer": Principal("writer")}
        self.state.metadata_requests, self.state.metadata_replies, self.state.timestamp_script = [], [], []
        self.children, self.drainers = [], []
        self.artifact = Path(os.environ["LIVE_TEST_ARTIFACT_DIR"])
        self.server = ReferenceServer(self.state, service_factory=MetadataService)
        self.address = await self.server.start()
        self.channel = grpc.aio.insecure_channel(self.address)
        self.addAsyncCleanup(self.cleanup)
        await asyncio.wait_for(self.channel.channel_ready(), 5)
        self.stub = rpc.LiveServiceStub(self.channel)
        self.session = (await self.stub.CreateSession(pb.CreateSessionRequest(path="resolver", name="metadata-only"),
                                                     metadata=WRITER, timeout=5)).session_id
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="root.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=b"opaque-not-parsed-by-resolver"))]),
            metadata=WRITER, timeout=5)
        self.canonical = "live://" + self.address + "/" + quote(self.session, safe="-._~") + "/root.live"

    async def cleanup(self):
        failures = []
        for child in self.children:
            if child.returncode is None:
                try:
                    child.kill()
                except ProcessLookupError:
                    pass
        for work in ((child.wait() for child in self.children), self.drainers,
                     (self.channel.close(),), (self.server.stop(),)):
            try:
                results = await asyncio.wait_for(asyncio.gather(*work, return_exceptions=True), 5)
                failures.extend(value for value in results if isinstance(value, BaseException))
            except BaseException as error:
                failures.append(error)
        data = {"seed": self.seed, "attempts": self.state.attempts,
                "metadata_requests": self.state.metadata_requests, "metadata_replies": self.state.metadata_replies,
                "ledger": self.state.snapshot()}
        (self.artifact / (self.id().split(".")[-1] + ".resolver-ledger.json")).write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n")
        self.state.assert_faults_fired()
        self.assertFalse(failures, failures)

    async def driver(self, mode, faults=()):
        ledger_before = list(self.state.ledger)
        process = await asyncio.create_subprocess_exec(os.environ["LIVE_TEST_RESOLVER_RUNTIME_DRIVER"],
            self.address, os.environ["LIVE_TEST_RESOLVER_RUNTIME_PLUG_INFO"], mode, self.session,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        self.children.append(process)
        records = asyncio.Queue(maxsize=16)
        log = self.artifact / (self.id().split(".")[-1] + ".resolver-native.log")
        async def drain():
            remaining = 65536
            with log.open("wb") as output:
                while line := await process.stdout.readline():
                    output.write(line[:remaining]); remaining -= min(len(line), remaining)
                    if line.startswith(b"{"):
                        records.put_nowait(json.loads(line))
            records.put_nowait(None)
        drainer = asyncio.create_task(drain())
        self.drainers.append(drainer)
        for number, fault in enumerate(faults):
            await asyncio.wait_for(fault.reached.wait(), 5)
            self.assertEqual(await asyncio.wait_for(records.get(), 5), {"type": "timeout", "round": number})
            self.assertIsNone(process.returncode)
            fault.release.set()
            process.stdin.write(b"continue\n")
            await process.stdin.drain()
            self.assertEqual(await asyncio.wait_for(records.get(), 5), {"type": "retry", "round": number})
        process.stdin.close()
        await asyncio.wait_for(process.wait(), 15)
        await drainer
        self.assertEqual(process.returncode, 0, log.read_text()[-16000:])
        data = await asyncio.wait_for(records.get(), 2)
        self.assertIsNotNone(data, log.read_text()[-16000:])
        self.assertEqual(data["type"], "result")
        self.assertEqual(data["canonical"], self.canonical)
        self.assertEqual(self.state.ledger, ledger_before, "resolver performed a service mutation")
        self.assertEqual(self.state.attempts.get("SubscribeReadLayer", 0), 0)
        self.assertTrue(all(item == {"session": self.session, "layer": "root.live"}
                            for item in self.state.metadata_requests))
        return data

    async def test_resolver_canonical_relative_metadata_and_generic_asset_rejection(self):
        self.state.clock = lambda: 1700000001456789
        await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name="root.live", object_id=1)),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=b"new-opaque-metadata"))]), metadata=WRITER, timeout=5)
        result = await self.driver("metadata")
        self.assertEqual(result["resolved"], self.canonical)
        self.assertEqual(result["relative"], self.canonical.removesuffix("root.live") + "nested/child.live")
        self.assertIs(result["timestamp_valid"], True)
        self.assertAlmostEqual(result["timestamp"], 1700000001.456789, places=6)
        self.assertEqual(self.state.attempts["StatLayer"], 3)
        self.assertNotIn("new.live", self.state.sessions[self.session].layers)

    async def test_resolver_ordinary_relative_assets_are_quietly_unresolved(self):
        # Design 4.3: a live URI is not the base scene's storage directory.
        before = dict(self.state.attempts)
        result = await self.driver("relative-assets")
        self.assertEqual(result["relative"], self.canonical.removesuffix("root.live") + "nested/child.live")
        # 0 is a quiet decline; 1 is INVALID_ARGUMENT and 13 RESOURCE_EXHAUSTED,
        # each with one typed resolver diagnostic.
        self.assertEqual(result["outcomes"], {
            "/abs/tex.png": 0, "file:x.png": 0, "x.LIVE": 0,
            "/abs/x.live": 1, "a:b.live": 1, "over-length.live": 13, "empty-anchor:child.live": 1})
        self.assertEqual(self.state.attempts.get("StatLayer", 0), 0)
        self.assertEqual(dict(self.state.attempts), before)

    async def test_resolver_new_identifier_neither_stats_nor_creates(self):
        result = await self.driver("new")
        self.assertEqual(result["resolved"], self.canonical.removesuffix("root.live") + "new.live")
        self.assertEqual(self.state.attempts.get("StatLayer", 0), 0)
        self.assertNotIn("new.live", self.state.sessions[self.session].layers)

    async def test_resolver_unregistered_endpoint_does_not_invent_asset(self):
        result = await self.driver("unregistered")
        self.assertEqual(result["resolved"], "")
        self.assertEqual(self.state.attempts.get("StatLayer", 0), 0)

    async def test_resolver_missing_layer_does_not_invent_asset(self):
        await self.stub.DeleteLayer(pb.DeleteLayerRequest(session_id=self.session, layer_name="root.live"),
                                    metadata=WRITER, timeout=5)
        result = await self.driver("missing")
        self.assertEqual(result["resolved"], "")
        self.assertEqual(self.state.attempts["StatLayer"], 1)

    async def test_resolver_missing_auth_is_bounded_and_issues_no_stat(self):
        result = await self.driver("auth")
        self.assertEqual(result["resolved"], "")
        self.assertEqual(self.state.attempts.get("StatLayer", 0), 0)

    async def test_resolver_metadata_deadline_while_starved_is_unauthenticated(self):
        # The 1 ms metadata deadline precedes the 30000 ms auth wait; starvation is classified before every deadline check.
        result = await self.driver("auth-deadline")
        self.assertEqual(result["resolved"], "")
        self.assertEqual(self.state.attempts.get("StatLayer", 0), 0)

    async def test_resolver_permission_failure_does_not_retry_or_invent_asset(self):
        self.state.plan("StatLayer", "before_reply", code=grpc.StatusCode.PERMISSION_DENIED)
        result = await self.driver("permission")
        self.assertEqual(result["resolved"], "")
        self.assertEqual(self.state.attempts["StatLayer"], 1)

    async def test_resolver_safe_metadata_retry_preserves_ledger(self):
        self.state.plan("StatLayer", "before_reply", code=grpc.StatusCode.UNAVAILABLE)
        result = await self.driver("retry")
        self.assertEqual(result["resolved"], self.canonical)
        self.assertEqual(self.state.attempts["StatLayer"], 2)

    async def test_resolver_held_metadata_deadlines_cleanup_and_repeated_retry(self):
        faults = [self.state.plan("StatLayer", "before_reply", attempt=attempt, pause=True)
                  for attempt in (1, 3, 5)]
        result = await self.driver("timeout-retry", faults)
        self.assertEqual(result["resolved"], self.canonical)
        self.assertEqual(self.state.attempts["StatLayer"], 6)

    async def test_resolver_callback_worker_refuses_blocking_metadata(self):
        result = await self.driver("worker")
        self.assertEqual(result["resolved"], "")
        self.assertEqual(self.state.attempts.get("StatLayer", 0), 0)

    async def test_resolver_timestamp_preserves_submillisecond_changes_and_epoch(self):
        expected = [1700000001456789, 1700000001456999, 0]
        self.state.timestamp_script = list(expected)
        result = await self.driver("timestamps")
        self.assertEqual(self.state.metadata_replies, expected)
        self.assertEqual(self.state.attempts["StatLayer"], 3)
        self.assertEqual(len(result["timestamps"]), 3)
        for actual, micros in zip(result["timestamps"], expected):
            self.assertAlmostEqual(actual, micros / 1_000_000, places=6)
        self.assertGreater(result["timestamps"][1], result["timestamps"][0])
        self.assertEqual(result["timestamps"][2], 0)
