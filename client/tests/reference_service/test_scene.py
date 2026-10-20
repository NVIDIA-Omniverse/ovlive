# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real actor/scene integration against an independent bounded root-field model.

This compact-response profile is separate from service.py's opaque replay mode.
It proves root int32 edit accounting, not general section/conflict semantics.
"""
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

import grpc
from service import State, Principal, ReferenceServer, LiveService, pb, rpc, status_details
from scene_wire import Model, WireError, decode_update, empty_snapshot, update, snapshot, aggregate
from scene_graph_wire import Command, Field, Delta, Specifier, Token, encode

WRITER = (("authorization", "Bearer synthetic-writer"),)
# Root int fields per direct edit in the transient flush test. With a
# 4096-byte update limit the read subscription reserves 4160 endpoint bytes;
# an admitted update reserves its bytes plus 256 for the retained intent and
# its bytes plus about 220 for its issued write's buffers, until both retire.
# The endpoint limit is the read, its catch-up frames (a 230-byte snapshot
# frame and a 68-byte caught-up header) and one 4096-byte StatLayer response
# allowance, which the resolver's queries reserve beside them while the layer
# opens: 8554 bytes. Under it one held update of 1150..1750 bytes fits, and a
# second cannot be admitted until the first retires.
TRANSIENT_FIELDS = 17
TRANSIENT_LIMITS = "limits=4096,8554"
# The peer backlog test: root fields in its layer and peer updates queued behind one withheld publication.
BACKLOG_FIELDS = 2000
BACKLOG_UPDATES = 24
# Positions the applied sequence may trail the read before apply_lag_entered,
# and the peer updates the lag test queues behind a held publication.
LAG_WARNING = 8
LAG_UPDATES = 16


def find_key(value, key):
    """Return the first value stored under `key` in a nested JSON report."""
    if isinstance(value, dict):
        if key in value:
            return value[key]
        value = list(value.values())
    if isinstance(value, list):
        for item in value:
            found = find_key(item, key)
            if found is not None:
                return found
    return None


class CompactSceneService(LiveService):
    def opaque(self, request):
        """Whether the layer's base payload is outside the root-field oracle,
        as a USD creation seed from Export or CreateNew is."""
        # The probe only selects the read route; the base path still authenticates.
        session = self.state.sessions.get(request.session_id)
        layer = session.layers.get(request.layer_name) if session else None
        if layer is None:
            return False
        try:
            Model().apply(layer.commits[0].payload)
        except WireError:
            return True
        return False

    async def SubscribeReadLayer(self, request, context):
        if self.opaque(request):
            # The base service's per-commit catch-up serves opaque history.
            async for frame in super().SubscribeReadLayer(request, context):
                yield frame
            return
        principal, attempt = await self.begin("SubscribeReadLayer", context)
        self.state.scene_reads.append(dict(backend=self.backend, attempt=attempt, layer=request.layer_name,
                                          object=request.object_id, cursor=request.sequence_num,
                                          writer=request.client_id))
        cursor = request.sequence_num
        catch_up = []
        async with self.state.changed:
            _, layer = await self.layer(request.session_id, request.layer_name, principal, context)
            if request.object_id not in (0, layer.object_id):
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "incarnation changed")
            latest = layer.commits[-1]
            if not cursor:
                model = Model()
                for commit in layer.commits:
                    model.apply(commit.payload, commit.sequence)
                first = replace(latest, payload=snapshot(model.state))
            elif cursor < latest.sequence and getattr(self.state, 'per_commit_catch_up', False):
                # The Live Edit service's per-commit mode: every update after the
                # cursor as it committed, including the subscriber's own.
                catch_up = [commit for commit in layer.commits if commit.sequence > cursor]
                first = None
            elif cursor < latest.sequence:
                # Include complete range effects, even self writes; the client retires a retained ACK the aggregate covers.
                fields = {}
                for commit in layer.commits:
                    if commit.sequence > cursor:
                        fields.update(decode_update(commit.payload))
                first = replace(latest, payload=aggregate(fields, cursor))
            else:
                first = None  # A cursor at or ahead of the service's latest sequence is up to date.
            cursor = latest.sequence
        await self.end("SubscribeReadLayer", attempt, context)
        if first:
            frames = list(self.frames(first))
            if attempt in self.state.partial_scene_reads:
                self.state.partial_scene_reads.remove(attempt)
                self.state.partial_scene_fired.append(attempt)
                yield frames[0]
                yield pb.SubscribeReadLayerResponse(chunk=pb.SubscribeReadLayerChunk(data=first.payload[:7]))
                await context.abort(grpc.StatusCode.UNAVAILABLE, "partial first scene")
            for frame in frames:
                yield frame
        for commit in catch_up:
            for frame in self.frames(commit):
                yield frame
        # Every catch-up ends with the service's caught-up header.
        yield self.caught_up(latest.object_id, cursor)
        while True:
            async with self.state.changed:
                _, layer = await self.layer(request.session_id, request.layer_name, principal, context)
                entries = [commit for commit in layer.commits if commit.sequence > cursor]
                # A coalescing service's live aggregate: the commits through `through`, every writer's, as one
                # aggregate based at the read's position, delivered once the layer reaches `through`.
                through = getattr(self.state, 'live_scene_aggregate', None)
                if not entries or (through and entries[0].sequence <= through and entries[-1].sequence < through):
                    await self.state.changed.wait()
                    continue
                base, cursor = cursor, entries[-1].sequence
            if through and entries[0].sequence <= through:
                self.state.live_scene_aggregate = None
                coalesced = [commit for commit in entries if commit.sequence <= through]
                entries = entries[len(coalesced):]
                fields = {}
                for commit in coalesced:
                    fields.update(decode_update(commit.payload))
                for frame in self.frames(replace(coalesced[-1], payload=aggregate(fields, base))):
                    yield frame
            for commit in entries:
                if commit.writer != request.client_id:
                    hold = self.state.live_scene_hold
                    if hold and self.backend == "A" and commit.sequence == hold[0]:
                        hold[1].set()
                        await hold[2].wait()
                    for frame in self.frames(commit):
                        yield frame


class RotatingProxy:
    """Test-only byte proxy: a stable endpoint, shared-state backend rotation.

    Given a server context, it terminates TLS in front of plaintext backends.
    """
    def __init__(self, target, *, ssl_context=None):
        self.target, self.ssl_context = target, ssl_context
        self.gate = asyncio.Event()
        self.gate.set()
        self.writers, self.tasks = set(), set()

    async def start(self):
        self.server = await asyncio.start_server(self.accept, "127.0.0.1", 0, ssl=self.ssl_context)
        self.address = "127.0.0.1:" + str(self.server.sockets[0].getsockname()[1])
        return self.address

    def accept(self, reader, writer):
        # Own the socket before scheduling: cancellation may retire a forward
        # before its coroutine has entered the try/finally in forward().
        self.writers.add(writer)
        task = asyncio.create_task(self.forward(reader, writer))
        self.tasks.add(task)
        def retired(task):
            self.tasks.discard(task)
            self.writers.discard(writer)
            writer.transport.abort()
        task.add_done_callback(retired)

    async def forward(self, reader, writer):
        peers, pumps = [writer], []
        self.writers.add(writer)
        try:
            await self.gate.wait()
            host, port = self.target.rsplit(":", 1)
            upstream, outbound = await asyncio.open_connection(host, int(port))
            peers.append(outbound)
            self.writers.add(outbound)
            async def pump(source, destination):
                while data := await source.read(65536):
                    destination.write(data)
                    await destination.drain()
            pumps = [asyncio.create_task(pump(reader, outbound)), asyncio.create_task(pump(upstream, writer))]
            await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
        except (ConnectionError, OSError):
            pass
        finally:
            for task in pumps:
                task.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            for peer in peers:
                self.writers.discard(peer)
                peer.transport.abort()

    def rotate(self, target, pause=False):
        if pause:
            self.gate.clear()
        self.target = target
        for writer in tuple(self.writers):
            writer.transport.abort()

    async def rotate_and_wait(self, target, pause=False):
        previous = tuple(self.tasks)
        self.rotate(target, pause=pause)
        # Abort also forwards suspended in open_connection, which could otherwise
        # establish an old-backend connection after the cut. New accepts use the
        # new target and are outside this fixed retirement set.
        for task in previous:
            task.cancel()
        await asyncio.gather(*previous, return_exceptions=True)
        return len(previous)

    async def close(self):
        self.server.close()
        await self.server.wait_closed()
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)


class Driver:
    def __init__(self, process, log):
        self.process, self.log = process, log
        self.queue = asyncio.Queue(maxsize=8)
        self.failure = None
        self.drainer = asyncio.create_task(self.drain())
        self.lock = asyncio.Lock()

    async def drain(self):
        # The log keeps the last 64 KiB of output, where a late failure's
        # cause is, and is written once the output ends.
        tail = bytearray()
        try:
            while line := await self.process.stdout.readline():
                tail += line
                del tail[:-65536]
                if line.startswith(b"{") and self.failure is None:
                    try:
                        self.queue.put_nowait(json.loads(line))
                    except (ValueError, asyncio.QueueFull) as error:
                        self.failure = error  # Keep draining after bounded protocol storage fails.
        finally:
            self.log.write_bytes(bytes(tail))
            if not self.queue.full():
                self.queue.put_nowait(None)

    async def receive(self, *, timeout=15):
        if self.failure is not None:
            raise AssertionError("native driver protocol failure") from self.failure
        result = await asyncio.wait_for(self.queue.get(), timeout)
        if self.failure is not None:
            raise AssertionError("native driver protocol failure") from self.failure
        if result is None:
            raise AssertionError("native driver exited: " + self.log.read_text()[-8192:])
        return result

    async def command(self, text, *, timeout=15):
        async with self.lock:
            self.process.stdin.write((text + "\n").encode())
            await self.process.stdin.drain()
            return await self.receive(timeout=timeout)


class SceneTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seed = int(os.environ["LIVE_TEST_SEED"])
        self.state = State(self.seed)
        self.state.tokens = {"synthetic-writer": Principal("writer")}
        self.state.scene_reads = []
        self.state.partial_scene_reads = set()
        self.state.partial_scene_fired = []
        self.state.live_scene_hold = None
        self.state.live_scene_aggregate = None
        self.state.per_commit_catch_up = False
        self.servers, self.channels, self.drivers, self.proxies = [], [], [], []
        self.artifact = Path(os.environ["LIVE_TEST_ARTIFACT_DIR"])
        self.trace = {"seed": self.seed, "admissions": [], "states": []}
        self.addAsyncCleanup(self.cleanup)
        self.address, self.stub = await self.backend("A")
        self.session = (await self.stub.CreateSession(pb.CreateSessionRequest(path="scene", name="fixture"),
                                                     metadata=WRITER, timeout=5)).session_id
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="root.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=empty_snapshot()))]), metadata=WRITER, timeout=5)

    async def backend(self, name):
        server = ReferenceServer(self.state, name, service_factory=CompactSceneService)
        self.servers.append(server)
        address = await server.start()
        channel = grpc.aio.insecure_channel(address)
        self.channels.append(channel)
        await asyncio.wait_for(channel.channel_ready(), 5)
        return address, rpc.LiveServiceStub(channel)

    async def cleanup(self):
        failures = []
        for driver in self.drivers:
            if driver.process.returncode is None:
                try:
                    driver.process.kill()
                except ProcessLookupError:
                    pass  # Already exited after the returncode observation.
                except Exception as error:
                    failures.append(error)
        try:
            for operations in ((d.process.wait() for d in self.drivers),
                               (d.drainer for d in self.drivers),
                               (p.close() for p in self.proxies),
                               (c.close() for c in self.channels),
                               (s.stop() for s in self.servers)):
                try:
                    results = await asyncio.wait_for(asyncio.gather(*operations, return_exceptions=True), 5)
                    failures.extend(result for result in results if isinstance(result, BaseException))
                except Exception as error:
                    failures.append(error)
            try:
                self.state.assert_faults_fired()
                self.assertFalse(self.state.partial_scene_reads, "partial-read fault did not fire")
            except Exception as error:
                failures.append(error)
        finally:
            self.trace.update(ledger=self.state.snapshot(), reads=self.state.scene_reads,
                              attempts=self.state.attempts, partial_fired=self.state.partial_scene_fired,
                              cleanup_errors=[repr(error) for error in failures])
            (self.artifact / (self.id().split(".")[-1] + ".scene.json")).write_text(
                json.dumps(self.trace, indent=2, sort_keys=True) + "\n")
        self.assertFalse(failures, failures)

    async def start(self, address=None, mode=None, extra=()):
        process = await asyncio.create_subprocess_exec(os.environ["LIVE_TEST_SCENE_DRIVER"],
            address or self.address, self.session, str(self.seed), *([mode] if mode else []), *extra,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        driver = Driver(process, self.artifact / f"{self.id().split('.')[-1]}-{len(self.drivers)}.scene-native.log")
        self.drivers.append(driver)
        ready = await driver.receive()
        self.assertEqual(ready["type"], "ready", ready)
        return driver

    async def close(self, driver):
        result = await driver.command("close")
        await asyncio.wait_for(driver.process.wait(), 10)
        await driver.drainer
        self.assertIsNone(driver.failure, "native driver exceeded bounded protocol storage")
        self.assertEqual(driver.process.returncode, 0, driver.log.read_text()[-8192:])
        return result

    def commits(self):
        return self.state.sessions[self.session].layers["root.live"].commits

    def oracle(self):
        model = Model()
        for commit in self.commits():
            model.apply(commit.payload, commit.sequence)
        return model

    async def peer(self, fields=None, *, payload=None, stub=None):
        payload = update(fields) if payload is None else payload
        result = await (stub or self.stub).UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name="root.live",
                                                             object_id=1, client_id="independent-peer")),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=payload))]), metadata=WRITER, timeout=5)
        return result.sequence_num

    async def settled(self, driver, sequence):
        result = await driver.command(f"wait {sequence}")
        self.trace["states"].append(result)
        self.assertEqual(result["result"], 0, result)
        self.assertEqual(result["applied"], sequence, result)
        self.assertEqual(result["authoritative"], self.oracle().state)
        self.assertEqual(result["visible"], self.oracle().state)
        self.assertEqual(result["pending_ids"], [])
        self.assertEqual(result["retained"], 0)
        return result

    def export_directory(self):
        directory = self.artifact / (self.id().split(".")[-1] + "-export")
        directory.mkdir()
        return directory

    async def test_compact_bootstrap_contains_all_existing_history(self):
        await self.peer({"a": 1, "gone": 4})
        await self.peer({"a": 2, "gone": None, "b": -9})
        driver = await self.start()
        await self.settled(driver, 3)
        self.assertEqual(self.state.scene_reads[0]["cursor"], 0)
        await self.close(driver)

    async def test_host_publication_waits_for_usd_notice_and_capture_retirement(self):
        driver = await self.start(mode="--publication")
        await driver.command("arm")
        sequence = await self.peer({"remote": 31})
        held = await driver.command("held")
        self.assertEqual(held["applied"], sequence - 1)
        self.assertEqual(held["journal_applied"], sequence)
        self.assertEqual(held["published"], sequence - 1)
        self.assertGreater(held["retained"], 0)
        retired = await driver.command("release-notice")
        self.assertEqual(retired["published"], sequence)
        self.assertEqual(retired["fields"], self.oracle().state)
        self.assertEqual(retired["applied"], sequence - 1)
        self.assertGreater(retired["retained"], 0)
        await driver.command("release-job")
        settled = await driver.command(f"wait {sequence}")
        self.assertEqual(settled["applied"], sequence)
        self.assertEqual(settled["retained"], 0)
        await driver.command("set local 77")
        local = await driver.command(f"wait {sequence + 1}")
        self.assertEqual(local["fields"], self.oracle().state)
        self.assertEqual(local["fields"], {"remote": 31, "local": 77})
        self.assertGreater(local["revision"], settled["revision"])
        self.assertEqual(local["notices"], 3)  # Peer, local publication, then confirmed application.
        self.trace["states"] += [held, retired, settled, local]
        await self.close(driver)

    async def test_host_publication_failure_retains_candidate_and_original_actor_work(self):
        driver = await self.start(mode="--publication")
        await driver.command("fail-next")
        sequence = await self.peer({"preserved": 19})
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 20)  # INTERNAL_ERROR, never claimed as rollback.
        self.assertEqual(failed["applied"], sequence - 1)
        self.assertEqual(failed["journal_applied"], sequence)
        self.assertEqual(failed["published"], sequence)
        self.assertEqual(failed["fields"], self.oracle().state)
        self.assertGreater(failed["retained"], 0)
        self.trace["states"].append(failed)
        closed = await self.close(driver)
        self.assertEqual(closed["fields"], failed["fields"])
        self.assertEqual(closed["retained"], failed["retained"])

    async def test_publication_callback_failure_before_install_leaves_the_host_unchanged(self):
        # Both callbacks fail the layer with the same result; only the host
        # contents show whether the failure came before or after installation.
        for command, expected_host in (("fail-before-next", {}), ("fail-next", {"preserved": 73})):
            driver = await self.start(mode="--publication")
            await driver.command(command)
            await driver.command("set preserved 73")
            failed = await driver.command("failed")
            self.assertEqual((failed["result"], failed["applied"], failed["journal_applied"]), (20, 1, 1), failed)
            self.assertEqual(failed["fields"], expected_host)
            self.assertEqual(self.state.attempts.get("UpdateLayer", 0), 0)
            self.assertEqual(len(self.commits()), 1)
            self.assertEqual(self.oracle().state, {})
            closed = await self.close(driver)
            self.assertEqual((closed["applied"], closed["retained"], closed["fields"]), (1, 1, expected_host))
            self.trace["states"] += [failed, closed]

    async def test_production_sdf_export_preserves_the_host_scene_after_publication_rejection(self):
        initial_sequence = await self.peer({"installed": 41})
        driver = await self.start(mode="--file-format")
        initial = await driver.command("state")
        self.assertEqual(initial["fields"], {"installed": 41})
        await driver.command("arm")
        await driver.command("reject-next")
        sequence = await self.peer({"remote": 12})
        held = await driver.command("held")
        self.assertEqual((held["applied"], held["fields"]), (initial_sequence, {"installed": 41}))
        await driver.command("release-held")
        failed = await driver.command("failed")
        self.assertEqual((failed["result"], failed["applied"], failed["journal_applied"]), (14, initial_sequence, sequence))
        captured = await driver.command("capture-host")
        self.assertEqual((captured["result"], captured["publication_result"], captured["applied"], captured["journal_applied"]),
                         (14, 14, initial_sequence, sequence))
        self.assertEqual((captured["last_attempted_revision"], captured["last_completed_revision"]), (2, 1))
        self.assertEqual(captured["host_fields"], {"installed": 41})
        self.assertEqual(captured["journal_fields"], {"installed": 41, "remote": 12})
        self.assertEqual(captured["journal_fields"], self.oracle().state)
        self.assertNotEqual(captured["host_fields"], captured["journal_fields"])
        self.assertEqual((self.state.attempts["CreateLayer"], self.state.attempts["SubscribeReadLayer"],
                          self.state.attempts["UpdateLayer"]), (1, 1, 2))
        self.assertTrue(all(commit.writer == "independent-peer" for commit in self.commits()[1:]))
        attempts = dict(self.state.attempts)
        exported = await driver.command(f"export-usd {self.export_directory() / 'peer-host-divergence.usdc'}")
        self.assertEqual(exported["fields"], {"installed": 41})  # The host scene, not the journal.
        self.assertEqual(self.state.attempts, attempts, "export issued service RPCs")
        self.assertEqual((await driver.command("state"))["fields"], captured["host_fields"])
        closed = await self.close(driver)
        self.assertEqual((closed["fields"], closed["retained"]), ({"installed": 41}, 1))
        self.trace["states"] += [initial, held, failed, captured, exported, closed]

    async def test_production_sdf_export_succeeds_when_the_deadline_passes_after_activation(self):
        driver = await self.start(mode="--file-format")
        attempts = dict(self.state.attempts)
        exported = await driver.command("export-live-deadline exported 17")
        # The fake creation clock crosses the deadline at the destination's
        # identifier notice, inside activation. Nothing after activation
        # checks the deadline, so the confirmed creation stays published.
        self.assertEqual((exported["succeeded"], exported["crossed"], exported["diagnosed"]), (True, True, False), exported)
        self.assertEqual((exported["alive"], exported["editable"], exported["fields"]), (True, True, {"counter": 17}), exported)
        ready = await driver.command("await-layer exported 2 0")  # READY, OK
        self.assertEqual((ready["state"], ready["status_result"]), (2, 0), ready)
        self.assertEqual(self.state.attempts.get("CreateLayer", 0), attempts.get("CreateLayer", 0) + 1)
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), attempts.get("UpdateLayer", 0))
        created = self.state.sessions[self.session].layers["exported.live"].commits
        self.assertEqual([commit.operation for commit in created], ["CreateLayer"])
        # The activated destination belongs to its layer: no workflow is retained.
        retired = await driver.command("retire-creations")
        self.assertEqual(retired["retired"], 0, retired)
        self.trace["states"] += [exported, ready, retired]
        await self.close(driver)

    async def test_production_sdf_unreferenced_export_frees_its_slot_before_the_destination_closes(self):
        driver = await self.start(mode="--file-format")
        exported = await driver.command("export-live-unreferenced unreferenced 23")
        # At the adoption seam the slot is still listed, and the destination
        # layer, whose last release starts its close, is still alive.
        self.assertEqual((exported["succeeded"], exported["adopted"]), (True, True), exported)
        self.assertEqual((exported["listed_at_adoption"], exported["alive_at_adoption"]), (True, True), exported)
        self.assertFalse(exported["retained_after"], exported)
        created = self.state.sessions[self.session].layers["unreferenced.live"].commits
        self.assertEqual([commit.operation for commit in created], ["CreateLayer"])
        self.trace["states"] += [exported]
        await self.close(driver)

    async def test_production_sdf_create_new_leaves_the_opening_deadline_to_its_finalizer(self):
        driver = await self.start(mode="--file-format")
        attempts = dict(self.state.attempts)
        created = await driver.command("create-new-deadline created")
        # The fake clock crosses the deadline when the host accepts the
        # finalizer, the only job the hook dispatches from the host. The hook
        # checks no deadline after that dispatch, so CreateNew returns its
        # target; the finalizer finalizes it, then reports the missed opening
        # deadline without activating it.
        self.assertEqual((created["succeeded"], created["crossed"], created["diagnosed"], created["host_dispatches"]),
                         (True, True, False, 1), created)
        self.assertEqual((created["alive"], created["editable"], created["fields"]), (True, False, {}), created)
        failed = await driver.command("await-layer created 7 9")  # FAILED, TIMEOUT
        self.assertEqual((failed["state"], failed["status_result"]), (7, 9), failed)
        self.assertEqual(self.state.attempts.get("CreateLayer", 0), attempts.get("CreateLayer", 0) + 1)
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), attempts.get("UpdateLayer", 0))
        retired = await driver.command("retire-creations")
        self.assertEqual(retired["retired"], 1, retired)
        self.trace["states"] += [created, failed, retired]
        await self.close(driver)

    async def test_production_sdf_file_format_open_alias_reload_and_edits(self):
        await self.peer({"initial": 41, "erased": -2})
        driver = await self.start(mode="--file-format")
        initial = await driver.command("state")
        self.assertEqual(initial["fields"], self.oracle().state)
        # The native opening path first rejects detached/argument-varied opens
        # and metadata-only/wrong-context Reads. None can add a subscription.
        self.assertEqual(self.state.attempts.get("SubscribeReadLayer", 0), 1)
        self.assertEqual(len(self.state.scene_reads), 1)
        reloaded = await driver.command("reload")
        self.assertEqual(reloaded["fields"], initial["fields"])
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        sequence = await self.peer({"remote": 12, "erased": None})
        peer = await driver.command(f"wait {sequence}")
        self.assertEqual(peer["fields"], self.oracle().state)
        accepted = await driver.command("set local 23")
        self.assertEqual((accepted["type"], accepted["result"]), ("accepted", 0))
        current = await driver.command(f"wait {sequence + 1}")
        self.assertEqual(current["fields"], self.oracle().state)
        self.assertEqual(current["fields"], {"initial": 41, "remote": 12, "local": 23})
        self.assertEqual(current["pending_ids"], [])
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        self.assertEqual([decode_update(c.payload) for c in self.commits()[1:]],
                         [{"initial": 41, "erased": -2}, {"remote": 12, "erased": None}, {"local": 23}])
        self.trace["states"] += [initial, reloaded, peer, current]
        await self.close(driver)

    async def test_production_sdf_unopened_live_uris_composed_off_the_host_say_to_open_them_on_the_host(self):
        await self.peer({"initial": 41})
        for name in ("first.live", "second.live"):
            await self.stub.CreateLayer(iter([
                pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name=name)),
                pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=snapshot({name: 1})))]), metadata=WRITER, timeout=5)
        driver = await self.start(mode="--file-format")
        attempts = dict(self.state.attempts)
        composed = await driver.command("compose-unopened first.live second.live")
        # The layers exist, so each URI resolves; a direct open and both
        # sublayer opens then fail in Read with BUSY (14) before they
        # subscribe. USD's sublayer warning carries the guidance, and no live
        # layer loads or stays registered.
        self.assertEqual((composed["direct_opened"], composed["direct_result"], composed["direct_errors"],
                          composed["direct_guided"]), (False, 14, 1, True), composed)
        self.assertEqual((composed["sublayer_failures"], composed["sublayer_guided"], composed["live_loaded"],
                          composed["registered"]), (2, 2, 0, False), composed)
        for method in ("SubscribeReadLayer", "CreateLayer", "UpdateLayer"):
            self.assertEqual(self.state.attempts.get(method, 0), attempts.get(method, 0), method)
        self.trace["states"] += [composed]
        await self.close(driver)

    async def test_production_sdf_direct_authoring_flushes_one_update_per_host_job(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        initial = await driver.command("state")
        base = len(self.commits())
        # One change block of plain Sdf calls is compiled into one UpdateLayer
        # by the flush job; the driver checks no intent is admitted in the block.
        direct = await driver.command("direct-set alpha 11 beta 22")
        self.assertEqual((direct["type"], direct["result"]), ("direct", 0), direct)
        authored = await driver.command(f"wait {base + 1}")
        self.assertEqual(authored["fields"], {"initial": 41, "alpha": 11, "beta": 22})
        self.assertEqual(authored["fields"], self.oracle().state)
        self.assertEqual(len(self.commits()), base + 1)
        self.assertEqual(decode_update(self.commits()[-1].payload), {"alpha": 11, "beta": 22})
        self.assertEqual(self.state.attempts["UpdateLayer"], 2)
        # A peer publication queued ahead of the flush job flushes the pending
        # host edit before installing, so the recorded edit is not discarded.
        await driver.command("arm")
        sequence = await self.peer({"peer": 99})
        await driver.command("held")
        flushed = await driver.command("direct-set-before-held gamma 33")
        self.assertEqual(flushed["result"], 0, flushed)
        # Read in the host job right after the peer install, before the local
        # publication job for the flushed edit can re-install it.
        self.assertEqual(flushed["installed"], {"initial": 41, "alpha": 11, "beta": 22, "peer": 99, "gamma": 33})
        settled = await driver.command(f"wait {sequence + 1}")
        self.assertEqual(settled["fields"], {"initial": 41, "alpha": 11, "beta": 22, "peer": 99, "gamma": 33})
        self.assertEqual(settled["fields"], self.oracle().state)
        self.assertEqual(settled["pending_ids"], [])
        self.assertEqual([decode_update(c.payload) for c in self.commits()[base:]],
                         [{"alpha": 11, "beta": 22}, {"peer": 99}, {"gamma": 33}])
        self.assertEqual(self.state.attempts["UpdateLayer"], 4)
        self.trace["states"] += [initial, direct, authored, flushed, settled]
        await self.close(driver)

    async def test_production_sdf_direct_authoring_rebaselines_to_host_content_between_flushes(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        await driver.command("arm")
        sequence = await self.peer({"peer": 99})
        await driver.command("held")
        # Host queue: flush(first) -> author(second) -> peer publication. The
        # publication flushes the second edit against the rebaselined base.
        direct = await driver.command("direct-set-between-flush-and-held first 1 second 2")
        self.assertEqual(direct["result"], 0, direct)
        settled = await driver.command(f"wait {sequence + 2}")
        self.assertEqual(settled["result"], 0, settled)
        self.assertEqual(settled["fields"], {"initial": 41, "peer": 99, "first": 1, "second": 2})
        self.assertEqual(settled["fields"], self.oracle().state)
        self.assertEqual(settled["pending_ids"], [])
        self.assertEqual([decode_update(c.payload) for c in self.commits()[-3:]],
                         [{"peer": 99}, {"first": 1}, {"second": 2}])
        self.trace["states"] += [direct, settled]
        await self.close(driver)

    async def test_production_sdf_stale_local_publication_keeps_a_later_admission(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        base = len(self.commits())
        # No acknowledgement arrives, so the only publications are local ones.
        hold = self.state.plan("UpdateLayer", "before", pause=True)
        self.addCleanup(hold.release.set)
        # The flush admits a; the local publication of that admission, with the
        # journal snapshot taken at admission, is withheld.
        await driver.command("direct-set-hold-local-publication a 1")
        # Host queue: flush(b) admits b -> withheld publication(a) -> read.
        # The publication's snapshot predates b's admission; installing it
        # alone would drop b from the host content.
        direct = await driver.command("direct-set-flush-before-held b 2")
        self.assertEqual(direct["result"], 0, direct)
        self.assertEqual(direct["installed"], {"initial": 41, "a": 1, "b": 2}, direct)
        hold.release.set()
        settled = await driver.command(f"wait {base + 2}")
        self.assertEqual(settled["result"], 0, settled)
        self.assertEqual(settled["fields"], {"initial": 41, "a": 1, "b": 2})
        self.assertEqual(settled["fields"], self.oracle().state)
        self.assertEqual(settled["pending_ids"], [])
        self.assertEqual([decode_update(c.payload) for c in self.commits()[base:]], [{"a": 1}, {"b": 2}])
        self.trace["states"] += [direct, settled]
        await self.close(driver)

    async def test_production_sdf_local_publication_skips_identical_install(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        before = await driver.command("replacements")
        base = len(self.commits())
        await driver.command("direct-set a 1")
        settled = await driver.command(f"wait {base + 1}")
        self.assertEqual(settled["fields"], {"initial": 41, "a": 1})
        self.assertEqual(settled["fields"], self.oracle().state)
        # The publication of the host's own update equals host content: no
        # install and no notice beyond the host's own edit.
        after = await driver.command("replacements")
        self.assertEqual((after["count"], after["changes"]), (before["count"], before["changes"] + 1), after)
        sequence = await self.peer({"peer": 2})
        peered = await driver.command(f"wait {sequence}")
        self.assertEqual(peered["fields"], {"initial": 41, "a": 1, "peer": 2})
        # A peer update is published in place with one change notice.
        published = await driver.command("replacements")
        self.assertEqual((published["count"], published["changes"]), (before["count"], after["changes"] + 1), published)
        self.trace["states"] += [settled, peered]
        await self.close(driver)

    async def test_production_sdf_peer_publication_sends_per_path_stage_notices(self):
        driver = await self.start(mode="--file-format")
        self.assertEqual((await driver.command("stage-open"))["layers"], 2)

        async def publish(commands=(), fields=()):
            sequence = await self.peer(payload=encode(Delta(commands=tuple(commands), fields=tuple(fields))))
            await driver.command(f"wait {sequence}")
            return await driver.command("stage-notices")

        def prim(section, parent, name):
            return [Command(1, section, parent=parent, name=name, spec_type=6)], [Field(section, "specifier", Specifier(0))]

        def attribute(section, parent, name, value):
            return ([Command(1, section, parent=parent, name=name, spec_type=1)],
                    [Field(section, "typeName", Token("double")), Field(section, "default", value)])

        commands, fields = [], []
        for created in (prim(1000, 1, "World"), attribute(1001, 1000, "a", 1.0), attribute(1002, 1000, "b", 0.0),
                        prim(1003, 1000, "Child"), prim(1004, 1, "Other")):
            commands += created[0]
            fields += created[1]
        seeded = await publish(commands, fields)
        self.assertEqual(seeded["resynced"], ["/Other", "/World"], seeded)
        installs = await driver.command("replacements")
        # A one-attribute peer edit is an info-only change of that attribute.
        edited = await publish(fields=[Field(1001, "default", 2.0, set_order=1)])
        self.assertEqual((edited["notices"], edited["resynced"], edited["info"]), (1, [], ["/World.a"]), edited)
        # A create resyncs the new prim only.
        created = await publish(*prim(1005, 1000, "Added"))
        self.assertEqual((created["notices"], created["resynced"], created["info"]), (1, ["/World/Added"], []), created)
        self.assertTrue((await driver.command("stage-prim /World/Added"))["exists"])
        # A rename resyncs its old and new path, and a delete its path.
        renamed = await publish([Command(3, 1004, old_parent=1, parent=1, name="Renamed")])
        self.assertEqual((renamed["resynced"], renamed["info"]), (["/Other", "/Renamed"], []), renamed)
        deleted = await publish([Command(2, 1003, parent=1000)])
        self.assertEqual((deleted["resynced"], deleted["info"]), (["/World/Child"], []), deleted)
        self.assertFalse((await driver.command("stage-prim /World/Child"))["exists"])
        # None of these publications replaced the layer's content.
        published = await driver.command("replacements")
        self.assertEqual((published["count"], published["changes"]), (installs["count"], installs["changes"] + 4))
        self.trace["states"] += [seeded, edited, created, renamed, deleted]
        await self.close(driver)

    async def test_production_sdf_applies_a_peer_backlog_in_one_publication(self):
        # A layer of BACKLOG_FIELDS root fields, so each publication does whole-layer work.
        await self.peer({f"seed{i}": i for i in range(BACKLOG_FIELDS)})
        # Without holdback, every covered update is ready when the held one is released.
        driver = await self.start(mode="--file-format", extra=("--no-holdback",))
        installs = await driver.command("replacements")
        # The first peer publication is withheld; the backlog queues behind it.
        await driver.command("arm")
        first = await self.peer({"backlog": 0})
        await driver.command("held")
        sequences = [await self.peer({"backlog": i, f"peer{i}": i}) for i in range(1, BACKLOG_UPDATES)]
        covered = await driver.command(f"covered {sequences[-1]}")
        self.assertEqual((covered["covered"], covered["applied"]), (sequences[-1], first - 1), covered)
        started = time.perf_counter()
        await driver.command("release-held")
        settled = await driver.command(f"wait {sequences[-1]}")
        elapsed = time.perf_counter() - started
        counted = await driver.command("replacements")
        publications = counted["changes"] - installs["changes"]
        # Informational wall clock, never a pass condition.
        self.trace["backlog"] = {"updates": BACKLOG_UPDATES, "fields": BACKLOG_FIELDS, "publications": publications,
                                 "release_to_applied_s": elapsed}
        self.assertEqual(settled["fields"], self.oracle().state)
        self.assertEqual(settled["pending_ids"], [])
        # The held update installs alone; every update queued behind it installs in one publication.
        # Each sends one change notice and none replaces the layer's content.
        self.assertEqual((publications, counted["count"]), (2, installs["count"]))
        self.trace["states"] += [covered]
        await self.close(driver)

    async def test_production_sdf_status_reports_read_versus_applied_lag(self):
        # A held publication queues a peer backlog. The layer stays READY while
        # its status reports the read ahead of the applied sequence; the
        # apply-lag warning enters once past the threshold and exits once the
        # backlog applies, and the receive-to-apply aggregate counts every update.
        await self.peer({"seed": 0})
        driver = await self.start(mode="--file-format", extra=(f"--lag-warning={LAG_WARNING}",))
        before = (await driver.command("diagnostics 0"))["diagnostics"]["metrics"]["apply_lag"]
        self.assertEqual((before["current"], before["high_water"]), (0, 0), before)
        await driver.command("arm")
        first = await self.peer({"backlog": 0})
        await driver.command("held")
        sequences = [await self.peer({"backlog": i}) for i in range(1, LAG_UPDATES)]
        await driver.command(f"covered {sequences[-1]}")
        lagging = await driver.command(f"await-sequences {sequences[-1]} {first - 1}")
        self.assertEqual((lagging["state"], lagging["status_result"]), (2, 0), lagging)  # READY
        self.assertGreater(sequences[-1] - (first - 1), LAG_WARNING)
        held = await driver.command("diagnostics 0")
        entered = self.lag_records(held, "apply_lag_entered")
        self.assertEqual(len(entered), 1, entered)
        self.assertEqual(entered[0]["fields"]["applied_sequence"], first - 1, entered)
        self.assertGreater(entered[0]["fields"]["read_sequence"] - (first - 1), LAG_WARNING, entered)
        self.assertEqual(self.lag_records(held, "apply_lag_exited"), [])
        self.assertEqual(held["diagnostics"]["metrics"]["apply_lag"]["current"], 1)
        # Live empty commits arrive as header-only frames behind the held
        # backlog. They are not a catch-up target, so the lag and its
        # interval stand until the backlog applies.
        expected = self.oracle().state
        sequences += [await self.peer(payload=b"") for _ in range(2)]
        empty = await driver.command(f"await-sequences {sequences[-1]} {first - 1}")
        self.assertEqual((empty["state"], empty["status_result"]), (2, 0), empty)  # READY
        still = await driver.command("diagnostics 0")
        self.assertEqual(len(self.lag_records(still, "apply_lag_entered")), 1)
        self.assertEqual(self.lag_records(still, "apply_lag_exited"), [])
        self.assertEqual(still["diagnostics"]["metrics"]["apply_lag"]["current"], 1)
        await driver.command("release-held")
        settled = await driver.command(f"wait {sequences[-1]}")
        self.assertEqual(settled["fields"], expected)
        caught_up = await driver.command(f"await-sequences {sequences[-1]} {sequences[-1]}")
        self.assertEqual((caught_up["state"], caught_up["status_result"]), (2, 0), caught_up)  # READY
        after = await driver.command("diagnostics 0")
        self.assertEqual(len(self.lag_records(after, "apply_lag_entered")), 1)
        exited = self.lag_records(after, "apply_lag_exited")
        self.assertEqual(len(exited), 1, exited)
        # The backlog can apply in more than one publication; the exit follows the
        # first that brings the lag within half the threshold.
        self.assertEqual(exited[0]["fields"]["read_sequence"], sequences[-1], exited)
        self.assertLessEqual(sequences[-1] - exited[0]["fields"]["applied_sequence"], LAG_WARNING // 2, exited)
        metric = after["diagnostics"]["metrics"]["apply_lag"]
        self.assertEqual((metric["current"], metric["high_water"]), (0, 1), metric)
        # Every peer update adds one sample: the held one alone and the backlog's batch.
        self.assertEqual(metric["receive_to_apply_ms"]["count"] - before["receive_to_apply_ms"]["count"],
                         LAG_UPDATES, metric)
        self.assertEqual(sum(metric["receive_to_apply_ms"]["counts"]), metric["receive_to_apply_ms"]["count"])
        layer = [s for s in after["diagnostics"]["statuses"] if s["subject_id"] == after["subject"]]
        self.assertEqual([(s["read_sequence"], s["applied_sequence"]) for s in layer],
                         [(sequences[-1], sequences[-1])], layer)
        # An acknowledged own write applies at a position the read never
        # delivers, so the applied sequence passes the read: the lag is zero,
        # not negative, and no warning fires.
        direct = await driver.command("direct-set own 7")
        self.assertEqual(direct["result"], 0, direct)
        own = sequences[-1] + 1
        await driver.command(f"wait {own}")
        ahead = await driver.command(f"await-sequences {sequences[-1]} {own}")
        self.assertEqual((ahead["state"], ahead["status_result"]), (2, 0), ahead)  # READY
        self.assertEqual(self.commits()[-1].sequence, own)
        final = await driver.command("diagnostics 0")
        self.assertEqual(len(self.lag_records(final, "apply_lag_entered")), 1)
        self.assertEqual(len(self.lag_records(final, "apply_lag_exited")), 1)
        self.assertEqual(final["diagnostics"]["metrics"]["apply_lag"]["current"], 0)
        self.trace["states"] += [lagging, caught_up, ahead]
        await self.close(driver)

    async def assert_no_apply_lag(self, driver):
        report = await driver.command("diagnostics 0")
        self.assertEqual(self.lag_records(report, "apply_lag_entered"), [])
        self.assertEqual(self.lag_records(report, "apply_lag_exited"), [])
        metric = report["diagnostics"]["metrics"]["apply_lag"]
        self.assertEqual((metric["current"], metric["high_water"]), (0, 0), metric)

    async def test_production_sdf_open_catch_up_beyond_the_lag_threshold_is_not_lag(self):
        # The bootstrap snapshot covers more positions than the threshold and
        # applies as one aggregate: the read is ahead of the applied sequence
        # until it retires, which is catch-up, not a host falling behind.
        last = 0
        for i in range(LAG_UPDATES):
            last = await self.peer({"seed": i, f"peer{i}": i})
        self.assertGreater(last, LAG_WARNING)
        driver = await self.start(mode="--file-format", extra=(f"--lag-warning={LAG_WARNING}",))
        opened = await driver.command(f"await-sequences {last} {last}")
        self.assertEqual((opened["state"], opened["status_result"]), (2, 0), opened)  # READY
        await self.assert_no_apply_lag(driver)
        self.trace["states"] += [opened]
        await self.close(driver)

    async def test_production_sdf_reconnect_catch_up_beyond_the_lag_threshold_is_not_lag(self):
        # The service commits more peer updates than the threshold while the
        # read is cut; the reconnect's catch-up to its target is not lag.
        other, _ = await self.backend("B")
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        first = await self.peer({"seed": 0})
        driver = await self.start(await proxy.start(), mode="--file-format", extra=(f"--lag-warning={LAG_WARNING}",))
        await driver.command(f"await-sequences {first} {first}")
        reads = self.state.attempts["SubscribeReadLayer"]
        await proxy.rotate_and_wait(other, pause=True)
        last = first
        for i in range(1, LAG_UPDATES):
            last = await self.peer({"seed": i, f"peer{i}": i})
        self.assertGreater(last - first, LAG_WARNING)
        proxy.gate.set()
        settled = await driver.command(f"wait {last}")
        self.assertEqual(settled["fields"], self.oracle().state)
        resumed = await driver.command(f"await-sequences {last} {last}")
        self.assertEqual((resumed["state"], resumed["status_result"]), (2, 0), resumed)  # READY
        self.assertGreater(self.state.attempts["SubscribeReadLayer"], reads)  # The read reconnected.
        await self.assert_no_apply_lag(driver)
        self.trace["states"] += [resumed]
        await self.close(driver)

    @staticmethod
    def lag_records(diagnostics, name):
        return [r for r in diagnostics["diagnostics"]["recent_records"]
                if r["event_name"] == name and r["subject_id"] == diagnostics["subject"]]

    async def test_production_sdf_peer_backlog_publication_keeps_a_pending_own_edit(self):
        await self.peer({f"seed{i}": i for i in range(BACKLOG_FIELDS)})
        # Without holdback, every covered update is ready when the held one is released.
        driver = await self.start(mode="--file-format", extra=("--no-holdback",))
        installs = await driver.command("replacements")
        await driver.command("arm")
        first = await self.peer({"backlog": 0, "own": -1})
        await driver.command("held")
        # The own edit's publication and write wait for the held application's slot.
        direct = await driver.command("direct-set own 7")
        self.assertEqual(direct["result"], 0, direct)
        sequences = [await self.peer({"backlog": i, "own": -1 - i}) for i in range(1, BACKLOG_UPDATES)]
        await driver.command(f"covered {sequences[-1]}")
        await driver.command("release-held")
        # The own edit commits after the backlog and wins.
        settled = await driver.command(f"wait {sequences[-1] + 1}")
        self.assertEqual(settled["fields"], self.oracle().state)
        self.assertEqual((settled["fields"]["own"], settled["fields"]["backlog"]), (7, BACKLOG_UPDATES - 1))
        self.assertEqual(settled["pending_ids"], [])
        self.assertEqual(decode_update(self.commits()[-1].payload), {"own": 7})
        self.assertEqual(self.commits()[-1].sequence, sequences[-1] + 1)
        self.assertEqual(first, sequences[0] - 1)
        # The backlog's own-edit replay keeps the host's edit; the held update and the batch install,
        # each with one change notice beside the host's own edit's notice.
        counted = await driver.command("replacements")
        self.assertEqual((counted["count"], counted["changes"] - installs["changes"]), (installs["count"], 3), counted)
        self.trace["states"] += [direct, settled]
        await self.close(driver)

    async def test_production_sdf_authoring_inside_a_publication_notice_fails_the_layer(self):
        driver = await self.start(mode="--file-format")
        updates = self.state.attempts.get("UpdateLayer", 0)
        installs = (await driver.command("replacements"))["count"]
        await driver.command("notice-author reentrant 5")
        await self.peer({"peer": 1})
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 12, failed)  # UNSUPPORTED
        self.assertEqual(failed["fields"], {"peer": 1, "reentrant": 5})  # Host content keeps the edit.
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), updates + 1)  # Only the peer's update.
        # The peer's publication keeps the backing in place, and so does the
        # reentrant edit, which installs nothing.
        self.assertEqual((await driver.command("replacements"))["count"], installs)
        status = await driver.command("status")
        self.assertEqual((status["status_result"], status["buffered"]), (12, 1), status)
        # A FAILED layer freezes: later host edits land in its current backing
        # and its buffered count only grows.
        await driver.command("direct-set later 6")
        grown = await driver.command("await-buffered 2")
        self.assertEqual((grown["status_result"], grown["buffered"]), (12, 2), grown)
        await driver.command("direct-set last 7")
        final = await driver.command("await-buffered 3")
        self.assertEqual((final["status_result"], final["buffered"]), (12, 3), final)
        self.assertEqual(final["fields"], {"peer": 1, "reentrant": 5, "later": 6, "last": 7})
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), updates + 1)
        # The later edits landed in the same backing; nothing was installed.
        self.assertEqual((await driver.command("replacements"))["count"], installs)
        exported = await driver.command(f"export-usd {self.export_directory() / 'notice.usdc'}")
        self.assertEqual(exported["fields"], final["fields"])  # SdfLayer::Export keeps every edit.
        self.trace["states"] += [failed, status, grown, final]
        await self.close(driver)

    async def test_production_sdf_authoring_off_the_host_context_fails_the_layer(self):
        driver = await self.start(mode="--file-format")
        updates = self.state.attempts.get("UpdateLayer", 0)
        await driver.command("direct-set-offhost stray 3")
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 12, failed)  # UNSUPPORTED
        self.assertEqual(failed["fields"], {"stray": 3})  # Host content keeps the edit.
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), updates)
        status = await driver.command("status")
        self.assertEqual((status["status_result"], status["buffered"]), (12, 1), status)
        exported = await driver.command(f"export-usd {self.export_directory() / 'offhost.usdc'}")
        self.assertEqual(exported["fields"], {"stray": 3})  # SdfLayer::Export keeps the edit.
        self.trace["states"] += [failed, status]
        await self.close(driver)

    async def test_production_sdf_failed_layer_counts_edits_without_retaining_records(self):
        driver = await self.start(mode="--file-format")
        updates = self.state.attempts.get("UpdateLayer", 0)
        await driver.command("direct-set-offhost stray 3")
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 12, failed)  # UNSUPPORTED
        expected = {"stray": 3}
        for value in range(64):
            await driver.command(f"direct-set edit{value} {value}")
            expected[f"edit{value}"] = value
        grown = await driver.command("await-buffered 65")
        self.assertEqual((grown["state"], grown["status_result"], grown["buffered"]), (7, 12, 65), grown)
        self.assertEqual(grown["fields"], expected)
        # The count keeps growing; the FAILED layer retains no record.
        retained = await driver.command("retained-records")
        self.assertEqual(retained["records"], 0, retained)
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), updates)
        exported = await driver.command(f"export-usd {self.export_directory() / 'counted.usdc'}")
        self.assertEqual(exported["fields"], expected)  # SdfLayer::Export keeps every edit.
        self.trace["states"] += [failed, grown, retained]
        await self.close(driver)

    async def test_production_sdf_rejects_whole_content_replacement_before_mutation(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        ready = await driver.command("status")
        updates = self.state.attempts.get("UpdateLayer", 0)
        directory = self.export_directory()
        rejected = await driver.command(f"replace-content {directory / 'import-source.usda'}")
        # TransferContent throws LayerOperationRejected(UNSUPPORTED) before any
        # mutation; the host's std::exception handler catches it and continues.
        self.assertEqual((rejected["transfer"], rejected["typed"]), (12, True), rejected)
        # USD 25.11 routes SdfLayer::Import through the layer's own format Read
        # hook, which rejects another asset with a typed UNSUPPORTED diagnostic.
        self.assertEqual((rejected["import"], rejected["import_result"], rejected["import_string"]), (False, 12, False), rejected)
        self.assertEqual((rejected["unchanged"], rejected["replacements"], rejected["result"]), (True, 0, 0), rejected)
        status = await driver.command("status")
        self.assertEqual((status["state"], status["status_result"], status["buffered"]), (ready["state"], 0, 0), status)
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), updates)
        # The layer keeps accepting host edits.
        base = len(self.commits())
        await driver.command("direct-set after 7")
        settled = await driver.command(f"wait {base + 1}")
        self.assertEqual((settled["result"], settled["fields"]), (0, {"initial": 41, "after": 7}), settled)
        self.assertEqual(settled["fields"], self.oracle().state)
        # SdfLayer::Clear swaps the backing before any plugin hook runs, so it
        # cannot be rejected; the plugin detects the swap and fails the layer.
        # The same host job authors x into the swapped-in backing.
        await driver.command("clear x 1")
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 12, failed)  # UNSUPPORTED
        cleared = await driver.command("status")
        self.assertEqual((cleared["state"], cleared["status_result"]), (7, 12), cleared)  # FAILED
        # Host edits after Clear are recorded by whichever backing holds them;
        # the FAILED layer's buffered count keeps growing and nothing is issued.
        await driver.command("direct-set y 2")
        grown = await driver.command("await-buffered 2")
        self.assertEqual((grown["status_result"], grown["buffered"]), (12, 2), grown)
        self.assertEqual(grown["fields"], {"x": 1, "y": 2})
        retained = await driver.command("retained-records")
        self.assertEqual(retained["records"], 0, retained)  # The swapped-in backing retains none either.
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), updates + 1)  # Only the edit before Clear.
        exported = await driver.command(f"export-usd {directory / 'cleared.usdc'}")
        self.assertEqual(exported["fields"], {"x": 1, "y": 2})  # SdfLayer::Export keeps the later edits.
        self.trace["states"] += [ready, rejected, status, settled, failed, cleared, grown]
        await self.close(driver)

    async def test_production_sdf_direct_authoring_flush_after_last_layer_release_is_inert(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        initial = await driver.command("state")
        updates = self.state.attempts["UpdateLayer"]
        # The host job authors a field and releases the last layer reference;
        # the flush job it enqueued then runs with no layer to pin.
        dropped = await driver.command("direct-set-drop orphan 5")
        self.assertIs(dropped["dropped"], True)
        self.assertEqual(dropped["result"], 0, dropped)
        self.assertEqual(dropped["fields"], initial["fields"])
        self.assertEqual(dropped["pending_ids"], [])
        self.assert_host_edits_discarded(dropped, 1, 0)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        self.assertEqual(self.oracle().state, {"initial": 41})
        self.trace["states"] += [initial, dropped]
        await self.close(driver)

    async def test_production_sdf_failed_layer_release_reports_discarded_host_edits(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        updates = self.state.attempts["UpdateLayer"]
        unencodable = await driver.command("direct-set-unencodable probe")
        self.assertEqual(unencodable["result"], 12, unencodable)  # UNSUPPORTED
        # The FAILED layer keeps the unencodable record and one more; the
        # release discards both and reports the layer's cause.
        dropped = await driver.command("direct-set-drop orphan 5")
        self.assertEqual(dropped["result"], 12, dropped)
        self.assert_host_edits_discarded(dropped, 2, 12)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        self.assertEqual(self.oracle().state, {"initial": 41})
        self.trace["states"] += [unencodable, dropped]
        await self.close(driver)

    def assert_host_edits_discarded(self, dropped, count, result):
        # The CLOSED status keeps the discarded count, and one numeric
        # host_edits_discarded record correlates with the released layer.
        closed = dropped["closed"]
        self.assertEqual((closed["state"], closed["buffered"], closed["unconfirmed"]), (9, count, 1), dropped)
        self.assertEqual(closed["result"], result, dropped)
        records = [r for r in dropped["diagnostics"]["recent_records"] if r["event_name"] == "host_edits_discarded"]
        self.assertEqual(len(records), 1, records)
        record = records[0]
        self.assertEqual((record["event_code"], record["subject_id"]), (31, dropped["subject"]), record)
        self.assertEqual(record["fields"], {"count": count, "result": result}, record)
        self.assertEqual(record["incident_id"], closed["incident"], record)
        self.assertNotIn("canonical_resource", record)

    async def test_production_sdf_off_host_save_never_drains_around_an_unadmitted_host_edit(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        updates = self.state.attempts["UpdateLayer"]
        # The probe value's copies run inside the flush job while it reads the
        # edit buffer; each performs an off-host save. The codec then rejects
        # the value, so the flush fails the layer with a typed cause.
        probe = await driver.command("direct-set-probe")
        *during, after = probe["saves"]
        self.assertTrue(during, probe)
        self.assertNotIn(0, during, probe)  # Never a clean drain that omits the edit.
        self.assertEqual(after, 12, probe)  # UNSUPPORTED: the codec rejects the probe value and the flush fails the layer.
        self.assertEqual(probe["admitted"], 0, probe)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        self.trace["states"].append(probe)
        await self.close(driver)

    async def test_production_sdf_oversized_flush_fails_the_layer_and_keeps_host_edits(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=("limits=4096,268435456",))
        updates = self.state.attempts["UpdateLayer"]
        # No wait can shrink this update below the runtime's per-update limit.
        fields = {f"oversized{i}": i for i in range(400)}
        direct = await driver.command("direct-set " + " ".join(f"{k} {v}" for k, v in fields.items()))
        self.assertEqual(direct["result"], 13, direct)  # FAILED with RESOURCE_EXHAUSTED, not deferred.
        self.assertEqual(direct["fields"], {"initial": 41, **fields})  # Host content keeps the edits for export.
        saved = await driver.command("save 13")
        self.assertTrue(saved["dirty"], saved)
        self.assertEqual(saved["pending_ids"], [])
        self.assertEqual(saved["fields"], direct["fields"])
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        self.assertEqual(self.oracle().state, {"initial": 41})
        self.trace["states"] += [direct, saved]
        await self.close(driver)

    async def test_production_sdf_flush_that_cannot_fit_beside_its_own_read_fails_the_layer(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=(TRANSIENT_LIMITS,))
        updates = self.state.attempts["UpdateLayer"]
        # Within the update limit, but its admission (twice its bytes plus
        # about 480) never fits beside the layer's own 4160-byte read.
        fields = {f"wide{i}": i for i in range(2 * TRANSIENT_FIELDS)}
        direct = await driver.command("direct-set " + " ".join(f"{k} {v}" for k, v in fields.items()))
        self.assertEqual(direct["result"], 13, direct)  # FAILED with RESOURCE_EXHAUSTED, not BLOCKED.
        self.assertEqual(direct["fields"], {"initial": 41, **fields})
        status = await driver.command("status")
        self.assertEqual((status["state"], status["status_result"], status["pending"]), (7, 13, 0), status)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        self.assertEqual(self.oracle().state, {"initial": 41})
        self.trace["states"] += [direct, status]
        await self.close(driver)

    async def test_production_sdf_transient_flush_failure_reschedules_on_the_next_edit(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=(TRANSIENT_LIMITS,))
        held = self.state.plan("UpdateLayer", "before", pause=True)
        # Each update fits alone; two cannot be pending on the endpoint at once.
        first = await driver.command("direct-set " + " ".join(f"first{i} {i}" for i in range(TRANSIENT_FIELDS)))
        self.assertEqual(first["result"], 0, first)
        await asyncio.wait_for(held.reached.wait(), 5)
        state = await driver.command("state")
        admitted = state["pending_ids"]
        self.assertEqual(len(admitted), 1)
        size = len(state["pending_hex"][0]) // 2
        self.assertTrue(1150 < size < 1750, size)
        rejected = lambda report: find_key(report, "capacity")["pending_bytes"]["reservations"]["rejected"]
        before = rejected(await driver.command("diagnostics 0"))
        # The limits mode sends no timing Ping, so each rejection below has
        # exactly one trigger: an endpoint release would reschedule the flush.
        pings = self.state.attempts.get("Ping", 0)
        second = await driver.command("direct-set " + " ".join(f"second{i} {i}" for i in range(TRANSIENT_FIELDS)))
        self.assertEqual(second["result"], 0, second)  # Transient: the layer stays healthy.
        self.assertEqual((await driver.command("state"))["pending_ids"], admitted)
        # BLOCKED with the transient cause (RESOURCE_EXHAUSTED) while the buffer waits.
        await driver.command("await-state 5 13")
        blocked = await driver.command("status")
        self.assertEqual((blocked["state"], blocked["status_result"], blocked["pending"]), (5, 13, 1), blocked)
        self.assertEqual(blocked["grpc_present"], 0, blocked)  # A local deferral carries no transport status.
        after_second = rejected(await driver.command("diagnostics 13"))
        self.assertEqual(after_second, before + 1)
        # An off-host SdfLayer::Save returns BUSY and schedules another flush.
        saved = await driver.command("save-offhost")
        self.assertEqual((await driver.command("state"))["pending_ids"], admitted)
        after_save = rejected(await driver.command("diagnostics 13"))
        self.assertEqual(after_save, after_second + 1)
        # The next record schedules another flush of the still-pending buffer.
        third = await driver.command("direct-set third 3")
        self.assertEqual(third["result"], 0, third)
        self.assertEqual((await driver.command("state"))["pending_ids"], admitted)
        self.assertEqual(rejected(await driver.command("diagnostics 13")), after_save + 1)
        self.assertEqual(self.state.attempts.get("Ping", 0), pings)
        held.release.set()
        # The acknowledgement retires the first intent; its publication job admits the buffer.
        ready = await driver.command("await-buffered 0")
        self.assertEqual((ready["state"], ready["status_result"]), (2, 0), ready)
        self.trace["states"] += [first, second, blocked, saved, third, ready]
        await self.close(driver)

    async def test_production_sdf_rejected_update_fails_with_its_result(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        driver = await self.start(mode="--file-format")
        self.state.plan("UpdateLayer", "before", code=grpc.StatusCode.INVALID_ARGUMENT)
        await driver.command("set rejected 5")
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 1, failed)  # INVALID_ARGUMENT, definitive
        self.assertEqual(failed["fields"], {"rejected": 5})  # Rejected host content stays until reopen.
        status = await driver.command("status")
        self.assertEqual((status["state"], status["status_result"], status["grpc_present"], status["grpc"],
                          status["operation"], status["pending"]),
                         (7, 1, 1, grpc.StatusCode.INVALID_ARGUMENT.value[0], 4, 1), status)
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        await self.close(driver)

    async def test_production_sdf_rejected_layer_journal_keeps_receiving_without_install(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        reads, updates = self.state.attempts["SubscribeReadLayer"], self.state.attempts["UpdateLayer"]
        self.state.plan("UpdateLayer", "before", code=grpc.StatusCode.INVALID_ARGUMENT)
        await driver.command("set rejected 5")
        failed = await driver.command("failed")
        self.assertEqual((failed["result"], failed["read_started"], failed["reading_after_failure"]), (1, True, True), failed)
        self.assertEqual(failed["fields"], {"initial": 41, "rejected": 5})
        sequence = await self.peer({"after": 6})
        received = await driver.command(f"journal {sequence}")
        self.assertEqual((received["journal_applied"], received["result"]), (sequence, 1), received)
        self.assertEqual(received["fields"], {"initial": 41, "rejected": 5})  # Installation into the layer stops at FAILED.
        status = await driver.command("status")
        self.assertEqual((status["state"], status["status_result"], status["grpc_present"], status["pending"]),
                         (7, 1, 1, 1), status)
        saved = await driver.command("save 1")  # A later save returns the cause.
        self.assertEqual(saved["fields"], {"initial": 41, "rejected": 5})
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], reads)  # The same read kept receiving.
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 2)  # The rejected issue and the peer.
        kept = await driver.command("state")
        self.assertEqual((kept["result"], kept["journal_applied"], kept["read_started"], kept["reading_after_failure"]),
                         (1, sequence, True, True), kept)
        self.trace["states"] += [failed, received, status, saved, kept]
        await self.close(driver)

    async def test_production_sdf_host_failed_layer_journal_keeps_receiving_without_install(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        reads = self.state.attempts["SubscribeReadLayer"]
        await driver.command("direct-set-unencodable probe")  # An unsupported flush: a host failure.
        failed = await driver.command("failed")
        self.assertEqual((failed["result"], failed["reading_after_failure"]), (12, True), failed)
        sequence = await self.peer({"after": 5})
        received = await driver.command(f"journal {sequence}")
        self.assertEqual((received["journal_applied"], received["result"]), (sequence, 12), received)
        self.assertNotIn("after", received["fields"])  # Installation into the layer stops at FAILED.
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], reads)
        self.trace["states"] += [failed, received]
        await self.close(driver)

    async def test_production_sdf_unknown_outcome_closes_the_read_and_freezes_the_journal(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        before = await driver.command("state")
        self.state.plan("UpdateLayer", "after_commit", code=grpc.StatusCode.UNAVAILABLE)
        await driver.command("set uncertain 5")
        failed = await driver.command("failed")
        self.assertEqual((failed["result"], failed["read_started"], failed["reading_after_failure"]),
                         (19, False, False), failed)
        self.assertEqual(failed["journal_applied"], before["journal_applied"])
        # The read closed with the failure, so no later peer work is received.
        sequence = await self.peer({"after": 6})
        frozen = await driver.command("state")
        self.assertEqual((frozen["journal_applied"], frozen["result"]), (before["journal_applied"], 19), frozen)
        self.assertLess(frozen["journal_applied"], sequence)
        self.assertEqual(frozen["fields"], {"initial": 41, "uncertain": 5})
        self.trace["states"] += [before, failed, frozen]
        await self.close(driver)

    def own_updates(self):
        return [decode_update(c.payload) for c in self.commits()[1:] if c.writer != "independent-peer"]

    async def proven(self, driver):
        """A confirmed first write, applied once."""
        self.assertEqual((await driver.command("set proven 1"))["result"], 0)
        self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
        self.assertEqual([entry["outcome"] for entry in self.state.operation_log], ["APPLIED"])
        return self.state.attempts["UpdateLayer"]

    async def test_production_sdf_lost_reply_after_commit_is_resent_and_applied_once(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        updates = await self.proven(driver)
        fault = self.state.plan("UpdateLayer", "after_commit", pause=True, code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await driver.command("set recovered 5"))["result"], 0)
        await asyncio.wait_for(fault.reached.wait(), 5)
        # A peer commits above the lost write before the re-send; the read skips the write's own position.
        after = await self.peer({"after": 6})
        fault.release.set()
        # The re-send with the write's operation ID earns the original commit's reply: the layer stays READY.
        self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 3)  # The write, the peer and the re-send.
        self.assertEqual(self.own_updates(), [{"proven": 1}, {"recovered": 5}])
        self.assertEqual([entry["outcome"] for entry in self.state.operation_log],
                         ["APPLIED", "APPLIED", "ALREADY_COMMITTED"])
        later = await self.peer({"later": 7})
        state = await driver.command(f"wait {later}")
        self.assertEqual((state["result"], state["applied"]), (0, later), state)
        self.assertEqual(state["fields"], {"initial": 41, "proven": 1, "recovered": 5, "after": 6, "later": 7})
        self.assertEqual(state["pending_ids"], [])
        self.assertLess(after, later)
        self.trace["states"].append(state)
        await self.close(driver)

    async def test_production_sdf_a_first_write_with_a_lost_reply_is_resent_once(self):
        # The Ping reports operation-ID support before any write, so even a writer's first write is re-sent with
        # its operation ID after a lost reply, and earns the original commit's reply.
        driver = await self.start(mode="--file-format")
        self.state.plan("UpdateLayer", "after_commit", code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await driver.command("set first 1"))["result"], 0)
        await asyncio.wait_for(self.state.observed(lambda: len(self.state.operation_log) == 2), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual(self.own_updates(), [{"first": 1}])
        self.assertEqual([entry["outcome"] for entry in self.state.operation_log], ["APPLIED", "ALREADY_COMMITTED"])
        state = await driver.command("state")
        self.assertEqual((state["result"], state["pending_ids"]), (0, []), state)
        await self.close(driver)

    async def test_production_sdf_a_pre_chaining_replica_mid_chain_drops_to_unchained_writes(self):
        # During a rolling upgrade or rollback, a replica that does not know the chained header refuses a chained
        # write INVALID_ARGUMENT before applying anything. The endpoint's writes stay unchained: the window drops to
        # one, the writes already in flight settle, and the refused write is re-sent unchanged and unchained.
        driver = await self.start(mode="--file-format")
        self.assertNotEqual((await driver.command("set a 1"))["id"], 0)
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 1), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        hold = self.state.plan_write("hold", position=2)
        for index in range(3):
            # Each admitted write is issued before the next turn, which would otherwise coalesce behind it.
            self.assertNotEqual((await driver.command(f"set w{index} {index}"))["id"], 0)
            await driver.command("issued")
        await asyncio.wait_for(hold.reached.wait(), 5)
        await asyncio.wait_for(self.state.observed(lambda: len(self.state.waiting) == 2), 5)
        self.state.pre_chaining = True
        self.assertNotEqual((await driver.command("set w3 3"))["id"], 0)
        await asyncio.wait_for(self.state.observed(
            lambda: [event["event"] for event in self.state.chain_log].count("unsupported") == 1), 5)
        hold.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 5), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual(self.own_updates(), [{"a": 1}] + [{f"w{i}": i} for i in range(4)])
        # One write in flight, unchained, and later turns coalesce behind it.
        held = self.state.plan("UpdateLayer", "before", attempt=self.state.attempts["UpdateLayer"] + 1, pause=True)
        self.assertNotEqual((await driver.command("set w4 4"))["id"], 0)
        await asyncio.wait_for(held.reached.wait(), 5)
        self.assertEqual((await driver.command("set w5 5"))["id"], 0)
        held.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 7), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual(self.own_updates()[5:], [{"w4": 4}, {"w5": 5}])
        commits = [event for event in self.state.chain_log if event["event"] == "commit"]
        self.assertEqual(len(commits), 4)  # a and w0-w2; w3 and every later write are unchained.
        # The endpoint stays unchained: w4 and w5 each went out once, unchained, and no later chained header reached
        # the replica that refuses it.
        self.assertEqual(self.state.attempts["UpdateLayer"], held.attempt + 1)
        self.assertEqual([entry["chained"] for entry in self.arrivals(self.commits()[-1].writer)][-2:], [False, False])
        self.assertEqual([event["event"] for event in self.state.chain_log].count("unsupported"), 1)
        state = await driver.command("state")
        self.assertEqual((state["result"], state["pending_ids"], state["fields"]), (0, [], self.oracle().state))
        await self.close(driver)

    async def test_production_sdf_reply_lost_before_commit_is_applied_by_its_resend(self):
        driver = await self.start(mode="--file-format")
        updates = await self.proven(driver)
        self.state.plan("UpdateLayer", "before", code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await driver.command("set applied 7"))["result"], 0)
        self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 2)
        self.assertEqual(self.own_updates(), [{"proven": 1}, {"applied": 7}])
        self.assertEqual([entry["outcome"] for entry in self.state.operation_log], ["APPLIED", "APPLIED"])
        later = await self.peer({"later": 8})
        state = await driver.command(f"wait {later}")
        self.assertEqual((state["result"], state["fields"]), (0, {"proven": 1, "applied": 7, "later": 8}), state)
        await self.close(driver)

    async def test_production_sdf_resend_fits_a_full_request_registry(self):
        # The registry holds two ordinary requests, the read and the write: the completed attempt's slot is released
        # before its re-send is submitted.
        driver = await self.start(mode="--file-format", extra=["--max-requests=2"])
        updates = await self.proven(driver)
        self.state.plan("UpdateLayer", "after_commit", code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await driver.command("set crowded 3"))["result"], 0)
        self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 2)
        self.assertEqual(self.own_updates(), [{"proven": 1}, {"crowded": 3}])
        self.assertEqual(self.state.operation_log[-1]["outcome"], "ALREADY_COMMITTED")
        await self.close(driver)

    async def lost_reply_with_revoked_token(self, driver):
        """Loses the reply to the next write while its token is revoked, and waits for the re-send's auth wait."""
        updates = await self.proven(driver)
        fault = self.state.plan("UpdateLayer", "after_commit", pause=True, code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await driver.command("set waited 4"))["result"], 0)
        await asyncio.wait_for(fault.reached.wait(), 5)
        self.assertEqual((await driver.command("token-revoke"))["result"], 0)
        fault.release.set()
        # The re-send cannot be submitted without a usable token: the layer waits for one, AUTH_WAIT/UNAUTHENTICATED.
        waiting = await driver.command("await-state 4 6")
        self.assertEqual(waiting["fields"], {"proven": 1, "waited": 4})
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)
        return updates

    async def test_production_sdf_resend_waits_for_a_usable_token(self):
        driver = await self.start(mode="--file-format")
        updates = await self.lost_reply_with_revoked_token(driver)
        self.assertEqual((await driver.command("token-set"))["result"], 0)
        self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 2)  # The write and its single re-send.
        self.assertEqual(self.own_updates(), [{"proven": 1}, {"waited": 4}])
        self.assertEqual(self.state.operation_log[-1]["outcome"], "ALREADY_COMMITTED")
        await self.close(driver)

    async def test_production_sdf_resend_awaiting_a_token_fails_unknown_when_refresh_fails(self):
        driver = await self.start(mode="--file-format")
        updates = await self.lost_reply_with_revoked_token(driver)
        failed_refresh = await driver.command("token-fail")
        self.assertEqual(failed_refresh["result"], 0, failed_refresh)
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 19, failed)
        self.assertEqual(failed["fields"], {"proven": 1, "waited": 4})
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)  # The re-send was never issued.
        await self.close(driver)

    async def test_production_sdf_host_failure_during_a_resend_auth_wait_never_resends(self):
        # A host failure latched on a host thread while the re-send waits for a token: the worker abandons the re-send
        # itself, the write settles OUTCOME_UNKNOWN, and a token installed afterwards issues nothing.
        driver = await self.start(mode="--file-format")
        updates = await self.lost_reply_with_revoked_token(driver)
        await driver.command("direct-set-unencodable probe")
        # The flush job coalesces while the re-send holds the window; an on-host save compiles the buffer, an
        # unsupported flush and so a host failure.
        await driver.command("save 19")
        failed = await driver.command("await-state 7 19")
        self.assertEqual(failed["result"], 19, failed)
        self.assertEqual((await driver.command("token-set"))["result"], 0)
        state = await driver.command("state")
        self.assertEqual(state["result"], 19, state)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)  # The re-send was never submitted.
        self.assertEqual(self.own_updates(), [{"proven": 1}, {"waited": 4}])
        self.trace["states"] += [failed, state]
        await self.close(driver)

    async def test_production_sdf_definitive_rejection_of_a_resend_fails_the_layer_with_its_result(self):
        # ALREADY_EXISTS answers a re-send only when its operation ID was committed with other content, a client
        # defect; like every definitive rejection of a re-send it is final, with its typed result. The first attempt
        # committed at a position the layer never learns, so the read closes, as for an unknown outcome, and no
        # later peer work is received over that position.
        driver = await self.start(mode="--file-format")
        updates = await self.proven(driver)
        before = await driver.command("state")
        lost = self.state.plan("UpdateLayer", "after_commit", code=grpc.StatusCode.UNAVAILABLE)
        self.state.plan("UpdateLayer", "before", attempt=lost.attempt + 1, code=grpc.StatusCode.ALREADY_EXISTS)
        await driver.command("set rejected 9")
        failed = await driver.command("failed")
        self.assertEqual((failed["result"], failed["read_started"], failed["reading_after_failure"]),
                         (5, False, False), failed)
        self.assertEqual(failed["fields"], {"proven": 1, "rejected": 9})
        self.assertEqual(failed["journal_applied"], before["journal_applied"])
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 2)
        self.assertEqual(self.own_updates(), [{"proven": 1}, {"rejected": 9}])
        await self.peer({"after": 6})
        frozen = await driver.command("state")
        self.assertEqual((frozen["journal_applied"], frozen["result"]), (before["journal_applied"], 5), frozen)
        self.trace["states"] += [failed, frozen]
        await self.close(driver)

    async def test_production_sdf_unproven_endpoint_never_resends_a_lost_reply(self):
        # The server ignores operation IDs, so no reply proves the endpoint honours them: the lost reply is not
        # re-sent, and the layer fails with its edit kept and the service holding it once.
        self.state.operation_ids = False
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        self.assertEqual((await driver.command("set first 1"))["result"], 0)
        self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
        updates = self.state.attempts["UpdateLayer"]
        self.state.plan("UpdateLayer", "after_commit", code=grpc.StatusCode.UNAVAILABLE)
        await driver.command("set uncertain 5")
        failed = await driver.command("failed")
        self.assertEqual((failed["result"], failed["read_started"], failed["reading_after_failure"]),
                         (19, False, False), failed)
        self.assertEqual(failed["fields"], {"initial": 41, "first": 1, "uncertain": 5})
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)
        self.assertEqual(self.own_updates(), [{"first": 1}, {"uncertain": 5}])
        self.assertEqual(self.state.operation_log, [])
        self.trace["states"].append(failed)
        await self.close(driver)

    async def test_production_sdf_a_backend_losing_chaining_never_applies_an_unsettled_write_twice(self):
        # The endpoint proved operation IDs, then the write's reply is lost and its backend loses operation-ID and
        # chaining support. The write's chained call may still commit, so it is never sent unchained, which such a
        # backend would apply a second time: its chained re-sends are refused, and after reconnect_max_attempts (10)
        # refusals the layer fails OUTCOME_UNKNOWN with its edit kept.
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        updates = await self.proven(driver)
        fault = self.state.plan("UpdateLayer", "after_commit", pause=True, code=grpc.StatusCode.UNAVAILABLE)
        await driver.command("set uncertain 5")
        await asyncio.wait_for(fault.reached.wait(), 5)
        self.state.operation_ids = False
        fault.release.set()
        failed = await driver.command("failed")
        self.assertEqual((failed["result"], failed["read_started"], failed["reading_after_failure"]),
                         (19, False, False), failed)
        self.assertEqual(failed["fields"], {"initial": 41, "proven": 1, "uncertain": 5})
        # The first attempt, the no_wait probe that finds the backend without chaining, and ten chained re-sends.
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 12)
        self.assertEqual(self.own_updates(), [{"proven": 1}, {"uncertain": 5}])
        self.trace["states"].append(failed)
        await self.close(driver)

    async def unchained_writer(self, *, fallback=False):
        """A driver whose endpoint re-sends with operation IDs but does not chain writes, with one confirmed write.
        With `fallback`, a chaining service's replica refuses the chained header, so the endpoint stops chaining.
        Returns the driver, the writer's applied writes and the UpdateLayer attempts so far."""
        self.state.chained_writes = fallback
        driver = await self.start(mode="--file-format")
        writer, _, _ = await self.proven_writer(driver)
        if fallback:
            self.state.pre_chaining = True
            self.assertNotEqual((await driver.command("set fell 0"))["id"], 0)
            await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 2), 5)
            await driver.command(f"wait {self.commits()[-1].sequence}")
            self.assertEqual(self.chain_events("unsupported", writer)[-1:], [self.arrivals(writer)[-1]["operation_id"]])
            self.assertFalse(self.arrivals(writer)[-1]["chained"])
        applied = sum(1 for entry in self.state.ledger if entry.writer == writer)
        return driver, applied, self.state.attempts["UpdateLayer"]

    def outcomes(self, outcome):
        return [entry["outcome"] for entry in self.state.operation_log].count(outcome)

    async def resent_with_a_pending_first_attempt(self, driver, fault):
        """Issues write a, whose first attempt fails while `fault` retains its service work, and waits until a's
        re-send has committed and the layer holds later writes with nothing in flight."""
        applied = self.outcomes("APPLIED")
        self.assertNotEqual((await driver.command("set a 1"))["id"], 0)
        await asyncio.wait_for(fault.reached.wait(), 5)
        await asyncio.wait_for(self.state.observed(lambda: self.outcomes("APPLIED") == applied + 1), 5)
        return await driver.command("await-resend-hold")

    async def held_then_issued(self, driver, updates):
        """Admits write b while the re-send hold is in force and checks it is not issued; then expires the hold and
        waits for b to commit."""
        self.assertNotEqual((await driver.command("set b 2"))["id"], 0)
        self.assertEqual((await driver.command("await-resend-hold"))["queued"], 1)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 2)  # a's first attempt and its re-send.
        self.assertEqual([entry["attempts"] for entry in (await driver.command("operations"))["writes"]][-1:], [0])
        records = (await driver.command("diagnostics 0"))["diagnostics"]["recent_records"]
        holds = [record["fields"] for record in records
                 if record["event_name"] == "retry_scheduled" and record["fields"].get("reason") == 4]
        self.assertEqual(len(holds), 1, holds)
        self.assertEqual(holds[0]["result"], 19)  # OUTCOME_UNKNOWN
        # request_timeout_ms (35 s) plus 5 s from a's submission, and the drop is well inside the watchdog.
        self.assertTrue(30000 < holds[0]["delay_ms"] <= 40000, holds[0])
        applied = self.outcomes("APPLIED")
        await driver.command("expire-resend-hold")
        await asyncio.wait_for(self.state.observed(lambda: self.outcomes("APPLIED") == applied + 1), 5)

    async def settled_once(self, driver, base):
        """a and b are each in the ledger once, and the client's fields equal the service's after a peer update."""
        self.assertEqual(self.own_updates()[base:], [{"a": 1}, {"b": 2}])
        await self.matches_service(driver)
        await self.close(driver)

    async def matches_service(self, driver):
        """After a peer update, the layer is READY with every write settled and its fields equal the ledger's."""
        sequence = await self.peer({"after": 3})
        state = await driver.command(f"wait {sequence}")
        self.assertEqual((state["result"], state["applied"], state["pending_ids"]), (0, sequence, []), state)
        self.assertEqual(state["fields"], self.oracle().state)
        self.trace["states"].append(state)

    async def test_production_sdf_a_late_first_attempt_after_the_next_write_never_applies_twice(self):
        # Issue 174: a's first call ends with an unknown outcome while its commit work remains held at the service.
        # a's re-send commits; b, held until the service has dropped or finished a's
        # first attempt, commits after it; the released first attempt read its start record before b and is overtaken.
        driver, applied, updates = await self.unchained_writer()
        first = self.state.plan_write("stale", position=applied + 1)
        await self.resent_with_a_pending_first_attempt(driver, first)
        await self.held_then_issued(driver, updates)
        first.release.set()
        await asyncio.wait_for(self.state.observed(lambda: self.outcomes("OVERTAKEN") == 1), 5)
        await self.settled_once(driver, len(self.own_updates()) - 2)

    async def test_production_sdf_a_first_attempt_reading_its_start_record_late_is_dropped_before_the_next_write(self):
        # The documented gap: a's first attempt reads its start record only after its upload admission, so a commit of
        # b before that read would let it apply again. b is held until the first attempt's deadline has passed, by
        # which the service drops a handler still waiting, which the delay's code models.
        driver, applied, updates = await self.unchained_writer()
        delay = self.state.plan_write("stale_start", position=applied + 1)
        await self.resent_with_a_pending_first_attempt(driver, delay)
        self.assertNotEqual((await driver.command("set b 2"))["id"], 0)
        self.assertEqual((await driver.command("await-resend-hold"))["queued"], 1)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 2)
        delay.code = grpc.StatusCode.DEADLINE_EXCEEDED
        delay.release.set()
        await asyncio.wait_for(delay.done.wait(), 5)
        applied = self.outcomes("APPLIED")
        await driver.command("expire-resend-hold")
        await asyncio.wait_for(self.state.observed(lambda: self.outcomes("APPLIED") == applied + 1), 5)
        await self.settled_once(driver, len(self.own_updates()) - 2)

    async def test_production_sdf_a_resend_whose_first_attempt_commits_meanwhile_settles_once(self):
        # a's first attempt commits while its re-send is held at the service: the re-send finds the record holding a's
        # own operation, which precedes the start-record guard, and is answered ALREADY_COMMITTED.
        driver, applied, _ = await self.unchained_writer()
        first = self.state.plan_write("stale", position=applied + 1)
        resend = self.state.plan_write("hold", position=applied + 1)
        self.assertNotEqual((await driver.command("set a 1"))["id"], 0)
        await asyncio.wait_for(first.reached.wait(), 5)
        await asyncio.wait_for(resend.reached.wait(), 5)
        first.release.set()
        await asyncio.wait_for(self.state.observed(lambda: self.outcomes("APPLIED") == 2), 5)
        resend.release.set()
        await asyncio.wait_for(self.state.observed(lambda: self.outcomes("ALREADY_COMMITTED") == 1), 5)
        self.assertEqual(self.own_updates()[-1:], [{"a": 1}])
        self.assertEqual(self.own_updates().count({"a": 1}), 1)
        await self.matches_service(driver)
        await self.close(driver)

    async def test_production_sdf_an_overtaken_resend_leaves_the_outcome_unknown(self):
        # a's re-send is held at the service while restore_checkpoint moves its writer's record: the service refuses
        # it UNAVAILABLE without ErrorInfo, applying nothing but unable to tell whether the first attempt committed.
        # The layer fails OUTCOME_UNKNOWN with the edit kept.
        driver, applied, _ = await self.unchained_writer()
        first = self.state.plan_write("stale", position=applied + 1)
        resend = self.state.plan_write("hold", position=applied + 1)
        self.assertNotEqual((await driver.command("set a 1"))["id"], 0)
        await asyncio.wait_for(first.reached.wait(), 5)
        await asyncio.wait_for(resend.reached.wait(), 5)
        self.assertEqual(self.state.restore_checkpoint(), 1)
        resend.release.set()
        failed = await driver.command("await-state 7 19")  # FAILED, OUTCOME_UNKNOWN
        self.assertEqual(failed["fields"].get("a"), 1)
        self.assertEqual(len(failed["pending_ids"]), 1)
        self.assertEqual(self.outcomes("OVERTAKEN"), 1)
        first.release.set()
        await asyncio.wait_for(self.state.observed(lambda: self.outcomes("OVERTAKEN") == 2), 5)
        self.assertNotIn({"a": 1}, self.own_updates())
        self.trace["states"].append(failed)
        await self.close(driver)

    async def test_production_sdf_a_reopened_layer_keeps_the_resend_hold(self):
        # The service keys a's operation record by the endpoint's writer ID, which a reopened layer shares: after a's
        # re-send the layer is released and opened again, and its first write b still waits for the hold, while a's
        # first attempt has not read its start record.
        driver, applied, updates = await self.unchained_writer()
        delay = self.state.plan_write("stale_start", position=applied + 1)
        await self.resent_with_a_pending_first_attempt(driver, delay)
        reopened = await driver.command("reopen")
        self.assertEqual(reopened["fields"].get("a"), 1, reopened)
        self.assertNotEqual((await driver.command("set b 2"))["id"], 0)
        self.assertEqual((await driver.command("await-resend-hold"))["queued"], 1)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 2)
        delay.code = grpc.StatusCode.DEADLINE_EXCEEDED
        delay.release.set()
        await asyncio.wait_for(delay.done.wait(), 5)
        applied = self.outcomes("APPLIED")
        await driver.command("expire-resend-hold")
        await asyncio.wait_for(self.state.observed(lambda: self.outcomes("APPLIED") == applied + 1), 5)
        await self.settled_once(driver, len(self.own_updates()) - 2)

    async def test_production_sdf_a_first_attempt_ended_at_its_deadline_holds_only_the_margin(self):
        # The pause gate holds a's first attempt, so its 2 s client deadline ends it; by then its request timeout has
        # passed, and the hold on later writes is the 5 s margin alone.
        self.state.chained_writes = False
        driver = await self.start(mode="--file-format", extra=["--request-timeout=2000"])
        await self.proven_writer(driver)
        applied = sum(1 for entry in self.state.ledger if entry.writer == self.commits()[-1].writer)
        first = self.state.plan_write("hold", position=applied + 1)
        self.assertNotEqual((await driver.command("set a 1"))["id"], 0)
        await asyncio.wait_for(first.reached.wait(), 5)
        await asyncio.wait_for(self.state.observed(lambda: self.outcomes("APPLIED") == 2), 10)
        await driver.command("await-resend-hold")
        records = (await driver.command("diagnostics 0"))["diagnostics"]["recent_records"]
        holds = [record["fields"] for record in records
                 if record["event_name"] == "retry_scheduled" and record["fields"].get("reason") == 4]
        self.assertEqual(len(holds), 1, holds)
        self.assertTrue(0 < holds[0]["delay_ms"] <= 5000, holds[0])
        first.release.set()
        await driver.command("expire-resend-hold")
        self.assertEqual(self.own_updates().count({"a": 1}), 1)
        await self.matches_service(driver)
        await self.close(driver)

    async def test_production_sdf_a_fallback_endpoint_never_applies_a_late_first_attempt_twice(self):
        # The same race after the endpoint stopped chaining because a replica refused the chained header.
        driver, applied, updates = await self.unchained_writer(fallback=True)
        first = self.state.plan_write("stale", position=applied + 1)
        await self.resent_with_a_pending_first_attempt(driver, first)
        await self.held_then_issued(driver, updates)
        first.release.set()
        await asyncio.wait_for(self.state.observed(lambda: self.outcomes("OVERTAKEN") == 1), 5)
        await self.settled_once(driver, len(self.own_updates()) - 2)

    async def test_production_sdf_rejected_layer_read_reconnects_after_a_transport_reset(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        await self.peer({"initial": 41})
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        driver = await self.start(await proxy.start(), mode="--file-format")
        self.state.plan("UpdateLayer", "before", code=grpc.StatusCode.INVALID_ARGUMENT)
        await driver.command("set rejected 5")
        failed = await driver.command("failed")
        self.assertEqual((failed["result"], failed["reading_after_failure"]), (1, True), failed)
        reads, updates = self.state.attempts["SubscribeReadLayer"], self.state.attempts["UpdateLayer"]
        await proxy.rotate_and_wait(self.address)  # Resets the read's transport.
        sequence = await self.peer({"after": 6})
        received = await driver.command(f"journal {sequence}")
        self.assertEqual((received["journal_applied"], received["result"], received["reading_after_failure"]),
                         (sequence, 1, True), received)
        self.assertEqual(received["fields"], {"initial": 41, "rejected": 5})
        self.assertGreater(self.state.attempts["SubscribeReadLayer"], reads)  # The read resumed.
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)  # Only the peer: no reissue of the rejected intent.
        status = await driver.command("status")
        self.assertEqual((status["state"], status["status_result"], status["grpc_present"]), (7, 1, 1), status)
        self.trace["states"] += [failed, received, status]
        await self.close(driver)

    async def test_production_sdf_status_counts_buffered_host_edits(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        prefix = await driver.command("status-prefix")
        self.assertEqual((prefix["result"], prefix["untouched"]), (0, True))
        base, updates = len(self.commits()), self.state.attempts["UpdateLayer"]
        # The held job is the flush the first record schedules, so both
        # records stay in the host edit buffer until it is released.
        await driver.command("arm")
        direct = await driver.command("direct-set a 1 b 2")
        self.assertEqual(direct["result"], 0, direct)
        buffered = await driver.command("await-buffered 2")
        self.assertEqual((buffered["state"], buffered["status_result"], buffered["pending"],
                          buffered["buffered"], buffered["unconfirmed"]), (2, 0, 0, 2, 1))
        self.assertEqual(buffered["fields"], {"initial": 41, "a": 1, "b": 2})
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        await driver.command("release-held")
        admitted = await driver.command("await-buffered 0")
        self.assertEqual((admitted["state"], admitted["status_result"], admitted["buffered"]), (2, 0, 0))
        settled = await driver.command(f"wait {base + 1}")
        self.assertEqual(settled["fields"], {"initial": 41, "a": 1, "b": 2})
        self.assertEqual(settled["fields"], self.oracle().state)
        self.assertEqual(decode_update(self.commits()[-1].payload), {"a": 1, "b": 2})
        self.trace["states"] += [buffered, admitted, settled]
        await self.close(driver)

    async def test_production_sdf_transient_flush_keeps_buffer_coalesces_and_defers_install(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=("--max-pending=1",))
        base, updates = len(self.commits()), self.state.attempts["UpdateLayer"]
        hold = self.state.plan("UpdateLayer", "before", pause=True)
        self.addCleanup(hold.release.set)
        await driver.command("direct-set a 1")
        await asyncio.wait_for(hold.reached.wait(), 5)
        # One intent is pending; the next flushes hit maxPending and keep their records.
        await driver.command("direct-set b 2")
        await driver.command("direct-set c 3")
        blocked = await driver.command("await-buffered 2")
        # BLOCKED with the transient cause (RESOURCE_EXHAUSTED) while the buffer waits.
        self.assertEqual((blocked["state"], blocked["status_result"], blocked["pending"], blocked["unconfirmed"]),
                         (5, 13, 1, 1), blocked)
        # A peer publication cannot install over unflushed host edits.
        sequence = await self.peer({"peer": 9})
        held = await driver.command(f"applied {sequence}")
        self.assertEqual(held["fields"], {"initial": 41, "a": 1, "b": 2, "c": 3})
        self.assertEqual(held["result"], 0)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 2)  # a, then the peer
        still = await driver.command("status")
        self.assertEqual((still["state"], still["status_result"], still["buffered"]), (5, 13, 2), still)
        hold.release.set()
        settled = await driver.command(f"wait {sequence + 2}")
        self.assertEqual(settled["fields"], {"initial": 41, "a": 1, "b": 2, "c": 3, "peer": 9})
        self.assertEqual(settled["fields"], self.oracle().state)
        ready = await driver.command("await-buffered 0")
        self.assertEqual((ready["state"], ready["status_result"], ready["pending"]), (2, 0, 0), ready)
        # The issued intent is never merged; the unissued records coalesce into one update.
        self.assertEqual([decode_update(c.payload) for c in self.commits()[base:]], [{"peer": 9}, {"a": 1}, {"b": 2, "c": 3}])
        self.trace["states"] += [blocked, held, still, settled, ready]
        await self.close(driver)

    async def test_production_sdf_buffer_quota_refuses_a_new_turn_until_a_flush_admits(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=("--max-pending=1", "--max-buffered=2"))
        base, updates = len(self.commits()), self.state.attempts["UpdateLayer"]
        hold = self.state.plan("UpdateLayer", "before", pause=True)
        self.addCleanup(hold.release.set)
        await driver.command("direct-set a 1")
        await asyncio.wait_for(hold.reached.wait(), 5)
        # a is pending; b and c wait behind the pending-intent limit and fill the quota.
        await driver.command("direct-set b 2")
        await driver.command("direct-set c 3")
        full = await driver.command("await-buffered 2")
        self.assertEqual((full["state"], full["status_result"], full["buffered"]), (5, 13, 2), full)
        # A new host turn at the quota is refused before the layer content
        # changes, with a typed exception the host catches. Stock SdfLayer
        # sends the change notice before its data edit, so the refused
        # SetField still yields one notice; the notice count and the
        # unchanged content are the refusal's evidence. (The layer reports
        # dirty only because its buffer is non-empty, so dirty is not pinned.)
        refused = await driver.command("direct-set-try d 4")
        self.assertEqual((refused["refused"], refused["notices"]), (13, 1), refused)  # RESOURCE_EXHAUSTED
        self.assertEqual(refused["fields"], {"initial": 41, "a": 1, "b": 2, "c": 3})
        # The deferral already reports BLOCKED/RESOURCE_EXHAUSTED; the quota
        # evidence is the exception above and the buffered count at the cap.
        blocked = await driver.command("status")
        self.assertEqual((blocked["state"], blocked["status_result"], blocked["pending"], blocked["buffered"]),
                         (5, 13, 1, 2), blocked)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)
        hold.release.set()
        # The acknowledgement retires a; its publication job admits b and c.
        ready = await driver.command("await-buffered 0")
        self.assertEqual((ready["state"], ready["status_result"]), (2, 0), ready)
        accepted = await driver.command("direct-set-try d 4")
        self.assertEqual((accepted["refused"], accepted["notices"]), (0, 1), accepted)
        settled = await driver.command(f"wait {base + 3}")
        self.assertEqual(settled["fields"], {"initial": 41, "a": 1, "b": 2, "c": 3, "d": 4})
        self.assertEqual(settled["fields"], self.oracle().state)
        # Every applied edit reaches the service exactly once, in order.
        self.assertEqual([decode_update(c.payload) for c in self.commits()[base:]], [{"a": 1}, {"b": 2, "c": 3}, {"d": 4}])
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 3)
        self.trace["states"] += [full, refused, blocked, ready, accepted, settled]
        await self.close(driver)

    async def test_production_sdf_buffer_quota_refuses_an_off_host_turn_as_unsupported(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=("--max-pending=1", "--max-buffered=2"))
        updates = self.state.attempts["UpdateLayer"]
        hold = self.state.plan("UpdateLayer", "before", pause=True)
        self.addCleanup(hold.release.set)
        await driver.command("direct-set a 1")
        await asyncio.wait_for(hold.reached.wait(), 5)
        await driver.command("direct-set b 2")
        await driver.command("direct-set c 3")
        full = await driver.command("await-buffered 2")
        self.assertEqual((full["state"], full["status_result"], full["buffered"]), (5, 13, 2), full)
        # A mutation off the host context at the quota is unsupported: it is
        # refused before the layer content changes, the layer fails
        # UNSUPPORTED and the exception carries that cause.
        refused = await driver.command("direct-set-offhost-try d 4")
        self.assertEqual(refused["refused"], 12, refused)  # UNSUPPORTED
        self.assertEqual(refused["fields"], {"initial": 41, "a": 1, "b": 2, "c": 3})
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 12, failed)
        status = await driver.command("status")
        self.assertEqual((status["state"], status["status_result"], status["buffered"]), (7, 12, 2), status)
        # An off-host SdfLayer::Save of the FAILED layer returns its cause.
        saved = await driver.command("save-offhost 12")
        self.assertEqual(saved["fields"], {"initial": 41, "a": 1, "b": 2, "c": 3}, saved)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)
        self.trace["states"] += [full, refused, failed, status, saved]
        hold.release.set()
        await self.close(driver)

    async def test_production_sdf_buffer_quota_never_refuses_a_failed_layer(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=("--max-pending=1", "--max-buffered=2"))
        updates = self.state.attempts["UpdateLayer"]
        fault = self.state.plan("UpdateLayer", "after_commit", pause=True, code=grpc.StatusCode.UNAVAILABLE)
        self.addCleanup(fault.release.set)
        await driver.command("direct-set a 1")
        await asyncio.wait_for(fault.reached.wait(), 5)
        await driver.command("direct-set b 2")
        await driver.command("direct-set c 3")
        # The window of one write is full, so both turns coalesce in the buffer, at its quota.
        full = await driver.command("await-buffered 2")
        self.assertEqual((full["state"], full["status_result"], full["buffered"]), (2, 0, 2), full)
        # The ambiguous outcome fails the layer while its buffer is at the
        # quota. The host worker is parked before the release, and the next
        # host turn starts in that job once the actor has failed, so no flush
        # latches the failure on the host side first: the layer's quota sink
        # decides from the actor's cause.
        await driver.command("park-try-on-failure d 4")
        fault.release.set()
        recorded = await driver.command("parked-try")
        # A FAILED layer's buffer only counts: the turn applies, the count
        # grows past the quota and nothing is issued.
        self.assertEqual(recorded["refused"], 0, recorded)
        self.assertEqual(recorded["fields"], {"initial": 41, "a": 1, "b": 2, "c": 3, "d": 4})
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 19, failed)  # OUTCOME_UNKNOWN
        grown = await driver.command("await-buffered 3")
        self.assertEqual((grown["state"], grown["status_result"], grown["buffered"]), (7, 19, 3), grown)
        again = await driver.command("direct-set-try e 5")
        self.assertEqual(again["refused"], 0, again)
        more = await driver.command("await-buffered 4")
        self.assertEqual((more["state"], more["status_result"], more["buffered"]), (7, 19, 4), more)
        retained = await driver.command("retained-records")
        self.assertEqual(retained["records"], 0, retained)  # Counted, not retained.
        # An off-host SdfLayer::Save of the layer the actor failed returns its cause.
        saved = await driver.command("save-offhost 19")
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)
        self.trace["states"] += [full, recorded, failed, grown, again, more, saved]
        await self.close(driver)

    async def test_production_sdf_back_to_back_turns_without_a_flush_block_at_the_quota(self):
        # Twenty change blocks of ten records inside one host job, so no flush job runs between them. Each change
        # notice ends a host turn, so the quota applies to every block: the seventh starts at 60 records, under
        # the 64-record quota, and records whole; every later block is refused before it changes anything.
        driver = await self.start(mode="--file-format", extra=("--max-buffered=64",))
        base = len(self.commits())
        turns = await driver.command("direct-turns 20 10 f")
        self.assertEqual((turns["refused"], turns["buffered"]), (13, 70), turns)
        # The flush job then runs and admits every recorded turn as one update; the status count follows.
        ready = await driver.command("await-buffered 0")
        self.assertEqual((ready["state"], ready["status_result"]), (2, 0), ready)
        await asyncio.wait_for(self.state.observed(lambda: len(self.commits()) == base + 1), 5)
        self.assertEqual(decode_update(self.commits()[-1].payload), {f"f{i}": 6 for i in range(10)})
        settled = await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual(settled["fields"], self.oracle().state)
        await self.close(driver)

    async def test_production_sdf_a_frame_of_a_thousand_blocks_collapses_into_one_update(self):
        # One host job authors 1,000 change blocks of the same 150 root fields. Every block records, and
        # compilation collapses all 150,000 records into one update with the final value of each field.
        driver = await self.start(mode="--file-format", extra=("--max-delta-entries=3000",))
        base = len(self.commits())
        blocks, fields = 1000, 150
        turns = await driver.command(f"direct-turns {blocks} {fields} f", timeout=60)
        self.assertEqual((turns["refused"], turns["buffered"]), (0, blocks * fields), turns)
        await asyncio.wait_for(self.state.observed(lambda: len(self.commits()) == base + 1), 120)
        settled = await driver.command(f"wait {self.commits()[-1].sequence}", timeout=60)
        updates = [decode_update(commit.payload) for commit in self.commits()[base:]]
        # The update retains every field's final value.
        self.assertEqual(updates, [{f"f{i}": blocks - 1 for i in range(fields)}])
        self.assertEqual(settled["fields"], self.oracle().state)
        self.assertEqual(settled["fields"], {f"f{i}": blocks - 1 for i in range(fields)})
        ready = await driver.command("await-buffered 0")
        self.assertEqual((ready["state"], ready["status_result"], ready["pending"]), (2, 0, 0), ready)
        metrics = (await driver.command("diagnostics 0"))["diagnostics"]["metrics"]["layer_writes"]
        self.assertEqual(metrics["max_turns_per_update"], blocks)
        await self.close(driver)

    async def test_production_sdf_split_prefix_counts_collapsed_writes_above_raw_operation_limit(self):
        # Five hundred repeated turns collapse into 150 fields before twenty distinct turns reach the entry
        # limit. The first fitting prefix holds 75,750 raw records, above the compiler's emitted-operation limit.
        driver = await self.start(mode="--file-format", extra=("--max-delta-entries=1000",))
        base = len(self.commits())
        turns = await driver.command("direct-turns 520 150 g mixed 500", timeout=60)
        self.assertEqual((turns["refused"], turns["buffered"]), (0, 78000), turns)
        await asyncio.wait_for(self.state.observed(lambda: len(self.commits()) == base + 4), 120)
        updates = [decode_update(commit.payload) for commit in self.commits()[base:]]
        first = {f"g{f}": 499 for f in range(150)}
        first.update({f"g{t}_{f}": t for t in range(500, 505) for f in range(150)})
        expected = [first] + [{f"g{t}_{f}": t for t in range(start, min(start + 6, 520))
                              for f in range(150)} for start in (505, 511, 517)]
        self.assertEqual(updates, expected)
        settled = await driver.command(f"wait {self.commits()[-1].sequence}", timeout=60)
        self.assertEqual(settled["fields"], self.oracle().state)
        metrics = (await driver.command("diagnostics 0"))["diagnostics"]["metrics"]["layer_writes"]
        self.assertEqual(metrics["max_turns_per_update"], 505)
        await self.close(driver)

    async def test_production_sdf_split_prefix_stays_within_the_raw_log_operation_limit(self):
        # One host job authors 1,025 change blocks of the same 1,024 root fields, 1,049,600 records, past the
        # compiler's 1,048,576-operation log limit, which counts every write. The prefix of 1,024 blocks holds exactly
        # that many records and collapses to 1,024 fields, far above the 65,536 emitted-operation limit in raw records,
        # so it is admitted whole; the prefix of every block is over the log limit, so the last block follows alone.
        driver = await self.start(mode="--file-format")
        base = len(self.commits())
        blocks, fields = 1025, 1024
        turns = await driver.command(f"direct-turns {blocks} {fields} r", timeout=300)
        self.assertEqual((turns["refused"], turns["buffered"]), (0, blocks * fields), turns)
        await asyncio.wait_for(self.state.observed(lambda: len(self.commits()) == base + 2), 300)
        updates = [decode_update(commit.payload) for commit in self.commits()[base:]]
        self.assertEqual(updates, [{f"r{f}": turn for f in range(fields)} for turn in (blocks - 2, blocks - 1)])
        settled = await driver.command(f"wait {self.commits()[-1].sequence}", timeout=60)
        self.assertEqual(settled["fields"], self.oracle().state)
        ready = await driver.command("await-buffered 0", timeout=60)
        self.assertEqual((ready["state"], ready["status_result"], ready["pending"]), (2, 0, 0), ready)
        metrics = (await driver.command("diagnostics 0"))["diagnostics"]["metrics"]["layer_writes"]
        self.assertEqual(metrics["max_turns_per_update"], blocks - 1)
        await self.close(driver)

    async def test_production_sdf_split_search_uses_the_journal_raw_log_operation_limit(self):
        # Under a journal raw log limit of 8 operations, four change blocks of the same three root fields (12 records)
        # exceed it together. The split search compiles the whole buffer once, then probes one and two blocks and
        # never a prefix over the configured limit; it admits two blocks, and the other two follow in one update.
        driver = await self.start(mode="--file-format", extra=("--max-log-operations=8",))
        base = len(self.commits())
        compiles = (await driver.command("compiles"))["count"]
        turns = await driver.command("direct-turns 4 3 q")
        self.assertEqual((turns["refused"], turns["buffered"]), (0, 12), turns)
        await asyncio.wait_for(self.state.observed(lambda: len(self.commits()) == base + 2), 30)
        settled = await driver.command(f"wait {self.commits()[-1].sequence}")
        updates = [decode_update(commit.payload) for commit in self.commits()[base:]]
        self.assertEqual(updates, [{f"q{f}": turn for f in range(3)} for turn in (1, 3)])
        self.assertEqual(settled["fields"], self.oracle().state)
        # The whole buffer, the two probes and the admitted prefix, then the rest.
        self.assertEqual((await driver.command("compiles"))["count"], compiles + 5)
        await self.close(driver)

    async def test_production_sdf_a_frame_past_the_entry_limit_splits_into_updates_at_turn_boundaries(self):
        # Twenty change blocks of 150 distinct root fields in one host job, under a 1,000-entry service limit: each
        # update holds the six whole blocks (900 entries) that fit, and the last the remaining two.
        driver = await self.start(mode="--file-format", extra=("--max-delta-entries=1000",))
        base = len(self.commits())
        turns = await driver.command("direct-turns 20 150 g distinct")
        self.assertEqual((turns["refused"], turns["buffered"]), (0, 3000), turns)
        await asyncio.wait_for(self.state.observed(lambda: len(self.commits()) == base + 4), 30)
        settled = await driver.command(f"wait {self.commits()[-1].sequence}")
        updates = [decode_update(commit.payload) for commit in self.commits()[base:]]
        self.assertEqual([len(update) for update in updates], [900, 900, 900, 300])
        blocks = [sorted({int(key[1:].split("_")[0]) for key in update}) for update in updates]
        self.assertEqual(blocks, [list(range(0, 6)), list(range(6, 12)), list(range(12, 18)), [18, 19]])
        expected = {f"g{t}_{f}": t for t in range(20) for f in range(150)}
        self.assertEqual({key: value for update in updates for key, value in update.items()}, expected)
        self.assertEqual(settled["fields"], self.oracle().state)
        await self.close(driver)

    async def test_production_sdf_an_oversized_turn_in_a_frame_still_fails_the_layer(self):
        # Two change blocks of 17 distinct root fields in one host job under a 16-entry limit: the first block alone
        # exceeds the limit, so the frame fails the layer as a single oversized turn does, issuing nothing.
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=("--max-delta-entries=16",))
        updates = self.state.attempts["UpdateLayer"]
        turns = await driver.command("direct-turns 2 17 h distinct")
        self.assertEqual((turns["refused"], turns["buffered"]), (0, 34), turns)
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 13, failed)  # RESOURCE_EXHAUSTED
        expected = {"initial": 41, **{f"h{t}_{f}": t for t in range(2) for f in range(17)}}
        self.assertEqual(failed["fields"], expected)  # Host content keeps every edit for export.
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        self.assertEqual(self.oracle().state, {"initial": 41})
        await self.close(driver)

    async def test_production_sdf_the_endpoint_buffer_budget_refuses_new_turns_once_exhausted(self):
        # A 1 MiB endpoint budget bounds every host edit buffer on the endpoint together. One host job authors 200
        # change blocks of 100 root fields with no flush between them: blocks record while the buffer holds less than
        # the budget, and every later block is refused before it changes anything, with a USD runtime error naming
        # the endpoint's budget. The flush then admits the recorded blocks, which releases the budget.
        budget = 1 << 20
        driver = await self.start(mode="--file-format", extra=(f"limits=65536,{budget}",))
        base = len(self.commits())
        turns = await driver.command("direct-turns 200 100 f")
        record = turns["buffered_bytes"] // turns["buffered"]
        accepted = -(-budget // (100 * record))  # The first block that starts at or past the budget is refused.
        self.assertEqual((turns["first_refused"], turns["refused"], turns["buffered"]),
                         (accepted, 200 - accepted, 100 * accepted), turns)
        self.assertGreaterEqual(turns["buffered_bytes"], budget)
        self.assertTrue(turns["endpoint_bound"], turns)
        ready = await driver.command("await-buffered 0")
        self.assertEqual((ready["state"], ready["status_result"]), (2, 0), ready)
        await asyncio.wait_for(self.state.observed(lambda: len(self.commits()) > base), 30)
        settled = await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual(settled["fields"], {f"f{i}": accepted - 1 for i in range(100)})
        self.assertEqual(settled["fields"], self.oracle().state)
        # The released budget admits the next frame.
        again = await driver.command("direct-turns 1 100 f")
        self.assertEqual((again["refused"], again["buffered"]), (0, 100), again)
        await self.close(driver)

    async def test_production_sdf_an_empty_buffer_is_refused_while_another_layer_exhausts_the_endpoint_budget(self):
        # Two live layers share a 1 MiB endpoint budget. In one host job the first layer's change blocks fill it, and
        # the second layer's first turn, with nothing of its own buffered, is refused before it changes anything,
        # with a USD runtime error naming the endpoint's budget, and its status reports BLOCKED with RESOURCE_EXHAUSTED.
        # The first layer's flush releases the budget, which retries the second layer's flush; that clears its cause,
        # the layer reports READY with OK, and its next turn records and reaches the service.
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="other.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=snapshot({"other": 1})))]), metadata=WRITER, timeout=5)
        budget = 1 << 20
        driver = await self.start(mode="--file-format", extra=(f"limits=65536,{budget}",))
        await driver.command("open-second other.live")
        shared = await driver.command("shared-budget-turns 200 100 f 100 s")
        first, second = shared["first"], shared["second"]
        self.assertGreater(first["refused"], 0, shared)
        self.assertGreaterEqual(first["buffered_bytes"], budget, shared)
        self.assertEqual((second["refused"], second["buffered"], second["buffered_bytes"]), (1, 0, 0), shared)
        self.assertTrue(second["endpoint_bound"], shared)
        self.assertEqual(second["blocked"], 13, shared)  # RESOURCE_EXHAUSTED
        self.assertFalse(second["changed"], shared)
        # While the host job holds the first layer's buffer, the empty second layer is publicly BLOCKED.
        self.assertEqual((second["state"], second["status_result"]), (5, 13), shared)  # BLOCKED, RESOURCE_EXHAUSTED
        ready = await driver.command("await-second-ready")
        self.assertEqual((ready["state"], ready["status_result"], ready["buffered"]), (2, 0, 0), ready)
        again = await driver.command("second-turns 1 100 s")
        self.assertEqual((again["refused"], again["buffered"]), (0, 100), again)
        other = self.state.sessions[self.session].layers["other.live"]
        expected = {f"s{i}": 0 for i in range(100)}
        await asyncio.wait_for(self.state.observed(
            lambda: any(decode_update(commit.payload) == expected for commit in other.commits[1:])), 30)
        await driver.command("drop-second")
        await self.close(driver)

    async def test_production_sdf_change_blocks_of_one_host_job_split_at_an_entry_limit(self):
        # Two change blocks of two distinct root fields inside one host job, so no flush job runs between them. Each
        # change notice ends a host turn: each block fits the 3-entry limit alone and both together do not, so the
        # flush admits the first block and then the second, two updates, instead of failing the layer.
        driver = await self.start(mode="--file-format", extra=("--max-delta-entries=3",))
        base = len(self.commits())
        turns = await driver.command("direct-distinct-turns 2 2 d")
        self.assertEqual((turns["refused"], turns["buffered"]), (0, 4), turns)
        await asyncio.wait_for(self.state.observed(lambda: len(self.commits()) == base + 2), 5)
        self.assertEqual([decode_update(commit.payload) for commit in self.commits()[base:]],
                         [{"d0": 0, "d1": 0}, {"d2": 1, "d3": 1}])
        settled = await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual((settled["result"], settled["pending_ids"], settled["fields"]), (0, [], self.oracle().state))
        self.trace["states"] += [turns, settled]
        await self.close(driver)

    async def test_production_sdf_buffer_quota_refusal_retries_an_allocation_deferral(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=("--max-buffered=1",))
        base, updates = len(self.commits()), self.state.attempts["UpdateLayer"]
        # The flush job fails its first allocation once: the transient
        # RESOURCE_EXHAUSTED keeps a in the buffer, which then holds the quota.
        faulted = await driver.command("direct-set-alloc-fault a 1")
        if not faulted["injection"]:
            # The thread-sanitizer driver has no replacement allocator to fail
            # an allocation, so it pins only the uninjected flush.
            self.assertEqual((faulted["result"], faulted["faulted"]), (0, False), faulted)
            settled = await driver.command(f"wait {base + 1}")
            self.assertEqual(settled["fields"], {"initial": 41, "a": 1})
            self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)
            self.trace["states"] += [faulted, settled]
            await self.close(driver)
            return
        self.assertEqual((faulted["result"], faulted["faulted"]), (0, True), faulted)
        await driver.command("await-state 5 13")
        blocked = await driver.command("status")
        self.assertEqual((blocked["state"], blocked["status_result"], blocked["pending"], blocked["buffered"]),
                         (5, 13, 0, 1), blocked)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        # The fault has cleared, but no record, publication, pending-byte
        # release or save follows. The refused turn is the only event, and it
        # retries the flush.
        refused = await driver.command("direct-set-try b 2")
        self.assertEqual(refused["refused"], 13, refused)  # RESOURCE_EXHAUSTED
        self.assertEqual(refused["fields"], {"initial": 41, "a": 1})
        ready = await driver.command("await-buffered 0")
        self.assertEqual((ready["state"], ready["status_result"]), (2, 0), ready)
        settled = await driver.command(f"wait {base + 1}")
        self.assertEqual(settled["fields"], {"initial": 41, "a": 1})
        self.assertEqual(settled["fields"], self.oracle().state)
        self.assertEqual([decode_update(c.payload) for c in self.commits()[base:]], [{"a": 1}])
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)
        self.trace["states"] += [faulted, blocked, refused, ready, settled]
        await self.close(driver)

    async def test_production_sdf_refused_flush_dispatch_keeps_the_buffer_quota(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=("--max-buffered=2",))
        base, updates = len(self.commits()), self.state.attempts["UpdateLayer"]
        # The dispatcher refuses every flush job for capacity: each record
        # stays buffered, the layer is BLOCKED, and each call ends its host
        # turn, so the quota bounds the buffer.
        await driver.command("refuse-dispatch 1")
        first = await driver.command("direct-set-try a 1")
        self.assertEqual(first["refused"], 0, first)
        second = await driver.command("direct-set-try b 2")
        self.assertEqual(second["refused"], 0, second)
        full = await driver.command("await-buffered 2")
        self.assertEqual((full["state"], full["status_result"], full["buffered"]), (5, 13, 2), full)
        refused = await driver.command("direct-set-try c 3")
        self.assertEqual((refused["refused"], refused["fields"]), (13, {"initial": 41, "a": 1, "b": 2}), refused)
        blocked = await driver.command("status")
        self.assertEqual((blocked["state"], blocked["status_result"], blocked["pending"], blocked["buffered"]),
                         (5, 13, 0, 2), blocked)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        # With capacity back and no other event, the quota refusal requests
        # the flush, which admits both records as one update. An endpoint
        # pending-byte release between "refuse-dispatch 0" and the retried
        # edit would also request the flush. The window is theoretical: the
        # test issues nothing that releases pending bytes there, and the
        # retried edit's quota refusal pins that no flush ran first.
        await driver.command("refuse-dispatch 0")
        retried = await driver.command("direct-set-try c 3")
        self.assertEqual(retried["refused"], 13, retried)
        ready = await driver.command("await-buffered 0")
        self.assertEqual((ready["state"], ready["status_result"]), (2, 0), ready)
        settled = await driver.command(f"wait {base + 1}")
        self.assertEqual(settled["fields"], {"initial": 41, "a": 1, "b": 2})
        self.assertEqual(settled["fields"], self.oracle().state)
        self.assertEqual([decode_update(c.payload) for c in self.commits()[base:]], [{"a": 1, "b": 2}])
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)
        self.trace["states"] += [first, second, full, refused, blocked, retried, ready, settled]
        await self.close(driver)

    async def chained_window(self, faults, *, writes=8, release=None, base=0):
        """Starts a writer, confirms one write (which marks the endpoint as chaining writes), plans `faults(state)`
        and admits `writes` more, each its own update, which issue together in the window. `release`, when given,
        runs once every write is admitted and released. Values start at `base`, so that a later writer of the
        layer changes every field. `faults` receives the writer's applied writes before them, which positions count after.
        Returns the driver, its writer ID and the planned faults."""
        driver = await self.start(mode="--file-format")
        self.assertEqual((await driver.command(f"set proven {base + 1}"))["result"], 0)
        self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
        writer = self.commits()[-1].writer
        applied = sum(1 for entry in self.state.ledger if entry.writer == writer)
        # The service holds the first write until every write is admitted as its own update.
        window = self.state.plan_write("hold", position=applied + 1)
        planned = faults(self.state, applied)
        for index in range(writes):
            # Each admitted write is issued before the next turn, which would otherwise coalesce behind it.
            self.assertNotEqual((await driver.command(f"set w{index} {base + index}"))["id"], 0)
            await driver.command("issued")
        await asyncio.wait_for(window.reached.wait(), 5)
        window.release.set()
        if release:
            await release(planned)
        return driver, writer, planned

    async def settled_chain(self, driver, writer, writes=8, base=0):
        """Waits until every write is confirmed and checks that each turn committed once, in order: an update holds
        one turn or, when the window was full, several consecutive whole turns."""
        # A drain covers admitted writes; turns that coalesced while the window was full are admitted by the flush
        # the layer requests once a write settles. Failure-only watchdog.
        deadline = asyncio.get_running_loop().time() + 10
        while True:
            self.assertEqual((await driver.command("save-wait 10000"))["result"], 0)
            if not (await driver.command("state"))["dirty"]:
                break
            self.assertLess(asyncio.get_running_loop().time(), deadline, "coalesced turns were never admitted")
        updates = [decode_update(commit.payload) for commit in self.commits() if commit.writer == writer]
        self.assertEqual(updates[0], {"proven": base + 1})
        turns = [sorted(int(key[1:]) for key in update) for update in updates[1:]]
        self.assertEqual([index for group in turns for index in group], list(range(writes)), updates)
        self.assertTrue(all(group == list(range(group[0], group[0] + len(group))) for group in turns), updates)
        self.assertTrue(all(update[key] == base + int(key[1:]) for update in updates[1:] for key in update), updates)
        state = await driver.command("state")
        self.assertEqual((state["result"], state["pending_ids"], state["fields"]), (0, [], self.oracle().state))
        return state

    async def test_production_sdf_a_lost_reply_at_each_window_position_commits_once(self):
        # A write whose reply is lost has an unknown outcome; the chain resolves it. When a later write of the
        # chain committed, it committed too, at the position the resubscribed read's catch-up repeats; otherwise
        # its re-send is answered. Every write commits once, in order, and no outcome stays unknown. A re-send, which
        # the last write always needs, locates the chain at once: it carries no_wait.
        for position in range(2, 10):
            with self.subTest(position=position):
                driver, writer, _ = await self.chained_window(
                    lambda state, applied: state.plan_write("drop_reply", position=applied + position - 1),
                    base=100 * position)
                await self.settled_chain(driver, writer, base=100 * position)
                lost = [entry["operation_id"] for entry in self.state.fault_log
                        if entry["boundary"].startswith("write:drop_reply")][-1]
                attempts = [entry["no_wait"] for entry in self.arrivals(writer, lost)]
                self.assertIn(attempts, ([False, True],) if position == 9 else ([False], [False, True]))
                await self.close(driver)

    async def test_production_sdf_a_refusal_at_each_window_position_is_reissued_in_order(self):
        # A refusal that applied nothing keeps its write; the successors waiting on it are refused or cancelled,
        # and the chain is re-sent from the refused write, which goes alone until it is acknowledged.
        for position in range(2, 10):
            with self.subTest(position=position):
                driver, writer, _ = await self.chained_window(
                    lambda state, applied: state.plan_write("refuse", position=applied + position - 1),
                    base=100 * position)
                await self.settled_chain(driver, writer, base=100 * position)
                await self.close(driver)

    async def test_production_sdf_a_stale_attempt_committing_late_is_answered_already_committed(self):
        # The service answers the last write's attempt UNAVAILABLE at once but commits it later; the re-send, held
        # at the service until then, is answered ALREADY_COMMITTED with that commit's position.
        async def release(planned):
            stale, hold = planned
            await asyncio.wait_for(hold.reached.wait(), 5)
            stale.release.set()
            await asyncio.wait_for(stale.done.wait(), 5)
            hold.release.set()
        driver, writer, (stale, _) = await self.chained_window(
            lambda state, applied: (state.plan_write("stale", position=applied + 8),
                                    state.plan_write("hold", position=applied + 8)),
            release=release)
        await self.settled_chain(driver, writer)
        self.assertEqual(stale.outcome, "APPLIED")
        self.assertIn("ALREADY_COMMITTED", [entry["outcome"] for entry in self.state.operation_log])
        await self.close(driver)

    async def test_production_sdf_capacity_refusals_halve_the_window_and_back_off(self):
        # Three refusals of one write with a retry delay: each halves the issuance window, from 8 to 4, 2 and 1, and
        # the write goes out again after the delay; then the chain commits in order. The service holds each later
        # attempt of the write before it answers, while nothing can change the window.
        def faults(state, applied):
            holds = []
            for _ in range(3):
                state.plan_write("refuse", position=applied + 1, retry_delay_ms=5)
                holds.append(state.plan_write("hold", position=applied + 1))
            return holds
        driver, writer, holds = await self.chained_window(faults)
        windows = []
        for hold in holds:
            await asyncio.wait_for(hold.reached.wait(), 5)
            windows.append((await driver.command("window"))["window"])
            hold.release.set()
        self.assertEqual(windows, [4, 2, 1])
        await self.settled_chain(driver, writer)
        metrics = (await driver.command("diagnostics 0"))["diagnostics"]["metrics"]["layer_writes"]
        self.assertEqual(metrics["window_halved"], 3)
        # Each refusal's backoff takes the service's RetryInfo delay (reason 1); no backoff falls back to the
        # reconnect schedule (reason 2), and no other retry is scheduled.
        records = (await driver.command("diagnostics 0"))["diagnostics"]["recent_records"]
        self.assertEqual([r["fields"]["reason"] for r in records if r["event_name"] == "retry_scheduled"], [1, 1, 1])
        await self.close(driver)

    async def test_production_sdf_an_unauthenticated_write_mid_chain_is_reissued_after_refresh(self):
        # The service rejects the third write of the window UNAUTHENTICATED: it never executed, so it waits for a
        # new token generation and is issued again; its successors cannot commit before it and are re-sent.
        attempts = {}

        def faults(state, applied):
            attempts['base'] = state.attempts["UpdateLayer"]
            # Held until every write is admitted: the rejection breaks the chain, and turns after it coalesce.
            return state.plan("UpdateLayer", "before", attempt=attempts['base'] + 3,
                              code=grpc.StatusCode.UNAUTHENTICATED, pause=True)

        async def release(planned):
            await asyncio.wait_for(planned.reached.wait(), 5)
            planned.release.set()
        driver, writer, _ = await self.chained_window(faults, release=release)
        waiting = await driver.command("await-state 4 6")  # AUTH_WAIT, UNAUTHENTICATED
        self.assertEqual(waiting["result"], 0)
        self.assertEqual((await driver.command("token-set"))["result"], 0)
        await self.settled_chain(driver, writer)
        await self.close(driver)

    async def test_production_sdf_turns_admit_no_faster_than_writes_issue(self):
        # At most one admitted write waits to be issued, even while the window has room: twenty frames authored in
        # one host job, which holds the host so the first admitted write cannot be published locally and issued,
        # admit that write alone, and the other frames coalesce into one update once it is issued.
        driver = await self.start(mode="--file-format")
        self.assertEqual((await driver.command("set proven 1"))["result"], 0)
        self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
        frames = 20
        self.assertEqual((await driver.command(f"busy-frames {frames} c"))["admitted"], 1)
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 3), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual(self.own_updates(), [{"proven": 1}, {"c0": 0}, {f"c{i}": i for i in range(1, frames)}])
        state = await driver.command("state")
        self.assertEqual((state["result"], state["pending_ids"], state["fields"]), (0, [], self.oracle().state))
        await self.close(driver)

    async def test_production_sdf_a_service_record_loss_leaves_the_outcome_unknown(self):
        # The write's reply is lost; before its re-send is evaluated the service restores saved state and loses
        # its operation records. The re-send is refused naming no operation, which the proven chain contradicts:
        # the layer fails OUTCOME_UNKNOWN with the edit kept.
        driver = await self.start(mode="--file-format")
        self.assertEqual((await driver.command("set proven 1"))["result"], 0)
        self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
        updates = self.state.attempts["UpdateLayer"]
        self.state.plan_write("drop_reply", position=2)
        resend = self.state.plan("UpdateLayer", "before", attempt=updates + 2, pause=True)
        self.assertNotEqual((await driver.command("set lost 7"))["id"], 0)
        await asyncio.wait_for(resend.reached.wait(), 5)
        self.assertEqual(self.state.restore_checkpoint(), 1)
        resend.release.set()
        failed = await driver.command("await-state 7 19")  # FAILED, OUTCOME_UNKNOWN
        self.assertEqual(failed["fields"], {**self.oracle().state, "lost": 7})
        self.assertEqual(len(failed["pending_ids"]), 1)
        await self.close(driver)

    def arrivals(self, writer, operation=None):
        """The reference service's log of `writer`'s identified UpdateLayer headers, in arrival order, optionally of
        one operation."""
        return [entry for entry in self.state.arrival_log
                if entry["client_id"] == writer and operation in (None, entry["operation_id"])]

    def chain_events(self, event, writer):
        """The operation IDs of `writer`'s chain events of kind `event`, in order."""
        return [entry["operation_id"] for entry in self.state.chain_log
                if entry["event"] == event and entry["client_id"] == writer]

    async def proven_writer(self, driver):
        """Confirms one write, which marks the endpoint as chaining writes; returns the writer ID, its applied
        writes and the UpdateLayer attempts so far."""
        self.assertEqual((await driver.command("set proven 1"))["result"], 0)
        self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
        writer = self.commits()[-1].writer
        applied = sum(1 for entry in self.state.ledger if entry.writer == writer)
        return writer, applied, self.state.attempts["UpdateLayer"]

    async def issue(self, driver, writes):
        for index in range(writes):
            # Each admitted write is issued before the next turn, which would otherwise coalesce behind it.
            self.assertNotEqual((await driver.command(f"set w{index} {index}"))["id"], 0)
            await driver.command("issued")

    async def test_production_sdf_successors_of_an_unknown_write_are_cancelled_and_it_is_resent_at_once(self):
        # The service answers the second write of a full window with a plain UNAVAILABLE, so its outcome is unknown,
        # and keeps its commit step pending; the six writes after it wait for it at the service. The writer cancels
        # them, which drops each with nothing applied, re-sends the write at once with no_wait, and then the others
        # in order with the same IDs. Waiting for the successors instead would wait out their deadlines.
        driver = await self.start(mode="--file-format")
        writer, applied, _ = await self.proven_writer(driver)
        hold = self.state.plan_write("hold", position=applied + 1)
        stale = self.state.plan_write("stale", position=applied + 2)
        await self.issue(driver, 8)
        await asyncio.wait_for(hold.reached.wait(), 5)
        await asyncio.wait_for(self.state.observed(lambda: len(self.state.waiting) == 7), 5)
        hold.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.chain_events("dropped", writer)) == 6), 5)
        await self.settled_chain(driver, writer)
        stale.release.set()
        await asyncio.wait_for(stale.done.wait(), 5)
        self.assertEqual(stale.outcome, "ABORTED")  # The re-send committed first, so the late step applies nothing.
        unknown = [entry["operation_id"] for entry in self.state.fault_log if entry["boundary"].startswith("write:stale")]
        self.assertEqual([entry["no_wait"] for entry in self.arrivals(writer, unknown[0])], [False, True])
        # A cancelled attempt that ends after the break resolved breaks the chain again, so the first write after
        # the unknown one may be re-sent with no_wait too; no other write is.
        third = self.arrivals(writer)[3]["operation_id"]
        self.assertLessEqual({entry["operation_id"] for entry in self.arrivals(writer) if entry["no_wait"]},
                             {unknown[0], third})
        self.assertEqual(self.chain_events("expired", writer), [])
        await self.close(driver)

    async def test_production_sdf_a_refusal_naming_a_write_without_an_operation_id_leaves_the_outcome_unknown(self):
        # The writer's first write commits and its reply is lost. Before its no_wait re-send is evaluated, a write
        # without an operation ID under the same writer ID commits, so the service's latest write for the writer has
        # no operation ID and refuses the re-send with last_record no_operation_id. The writer never issued such a
        # write, so the chain cannot be located: the layer fails OUTCOME_UNKNOWN with its edit kept, instead of
        # reading the empty last_operation_id as an absent record and re-sending the write as uncommitted.
        driver = await self.start(mode="--file-format")
        self.state.plan_write("drop_reply", position=1)
        resend = self.state.plan("UpdateLayer", "before", attempt=self.state.attempts.get("UpdateLayer", 0) + 2,
                                 pause=True)
        self.assertNotEqual((await driver.command("set first 1"))["id"], 0)
        await asyncio.wait_for(resend.reached.wait(), 5)
        writer = self.commits()[-1].writer
        await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name="root.live",
                                                             object_id=1, client_id=writer)),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=update({"foreign": 2})))]), metadata=WRITER, timeout=5)
        resend.release.set()
        failed = await driver.command("await-state 7 19")  # FAILED, OUTCOME_UNKNOWN
        self.assertEqual(failed["fields"].get("first"), 1)
        self.assertEqual(len(failed["pending_ids"]), 1)
        refusals = [entry for entry in self.state.chain_log if entry["event"] == "refused"]
        self.assertEqual([(entry["last_operation_id"], entry["last_record"]) for entry in refusals],
                         [("", "no_operation_id")])
        self.assertEqual([entry["no_wait"] for entry in self.arrivals(writer)], [False, True])
        self.assertEqual(len([commit for commit in self.commits() if commit.writer == writer]), 2)
        await self.close(driver)

    async def test_production_sdf_invalid_update_on_a_chained_write_is_definitive(self):
        # A chaining replica marks every INVALID_ARGUMENT it gives a chained call with an ErrorInfo of its domain,
        # here INVALID_UPDATE: the rejection is definitive, so the layer fails INVALID_ARGUMENT with the edit kept,
        # and nothing is re-sent unchained.
        driver = await self.start(mode="--file-format")
        writer, _, updates = await self.proven_writer(driver)
        self.state.plan("UpdateLayer", "before", attempt=updates + 1, code=grpc.StatusCode.INVALID_ARGUMENT,
                        details=status_details(grpc.StatusCode.INVALID_ARGUMENT, "invalid", "INVALID_UPDATE", {}))
        self.assertNotEqual((await driver.command("set rejected 2"))["id"], 0)
        failed = await driver.command("await-state 7 1")  # FAILED, INVALID_ARGUMENT
        self.assertEqual(len(failed["pending_ids"]), 1)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)
        self.assertEqual(self.own_updates(), [{"proven": 1}])
        await self.close(driver)

    async def writes_settled(self, driver):
        """Waits until the layer has no write attempt in flight, cancelled ones included, and returns its state.
        Failure-only watchdog."""
        deadline = asyncio.get_running_loop().time() + 10
        while True:
            result = (await driver.command("state"))["result"]
            metrics = (await driver.command(f"diagnostics {result}"))["diagnostics"]["metrics"]["layer_writes"]
            if not metrics["in_flight"]:
                return await driver.command("state")
            self.assertLess(asyncio.get_running_loop().time(), deadline, "write attempts never settled")

    async def test_production_sdf_an_invalid_update_ahead_of_a_full_window_stays_the_cause(self):
        # The service rejects the first write of a full window INVALID_UPDATE while the seven writes after it wait
        # for it at the service. They commit only after it, so none committed: their cancelled calls settle with
        # the rejection as the cause. The layer fails INVALID_ARGUMENT, not OUTCOME_UNKNOWN, with every edit kept,
        # and its read keeps receiving.
        driver = await self.start(mode="--file-format")
        writer, _, updates = await self.proven_writer(driver)
        rejected = self.state.plan(
            "UpdateLayer", "before", attempt=updates + 1, pause=True, code=grpc.StatusCode.INVALID_ARGUMENT,
            details=status_details(grpc.StatusCode.INVALID_ARGUMENT, "invalid", "INVALID_UPDATE", {}))
        await self.issue(driver, 8)
        await asyncio.wait_for(rejected.reached.wait(), 5)
        await asyncio.wait_for(self.state.observed(lambda: len(self.state.waiting) == 7), 5)
        rejected.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.chain_events("dropped", writer)) == 7), 5)
        failed = await self.writes_settled(driver)
        self.assertEqual((failed["result"], failed["reading_after_failure"]), (1, True), failed)
        self.assertEqual(failed["fields"], {"proven": 1, **{f"w{index}": index for index in range(8)}})
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 8)
        self.assertEqual(self.own_updates(), [{"proven": 1}])
        sequence = await self.peer({"after": 6})
        received = await driver.command(f"journal {sequence}")
        self.assertEqual((received["journal_applied"], received["result"], received["reading_after_failure"]),
                         (sequence, 1, True), received)
        self.trace["states"] += [failed, received]
        await self.close(driver)

    async def test_production_sdf_refused_resends_of_a_write_of_unknown_outcome_leave_it_unknown(self):
        # The write's first chained attempt is answered with a plain UNAVAILABLE, so its outcome is unknown, while
        # the service keeps its commit step pending. Each no_wait re-send is refused UNAVAILABLE with nothing applied
        # until reconnect_max_attempts (10) refusals. The refusals say nothing about the first attempt, which may
        # still commit: the layer fails OUTCOME_UNKNOWN and closes its read with the edit kept, never UNAVAILABLE
        # with the read open. The pending commit step then applies the write.
        driver = await self.start(mode="--file-format")
        writer, applied, updates = await self.proven_writer(driver)
        stale = self.state.plan_write("stale", position=applied + 1)
        for _ in range(10):
            self.state.plan_write("refuse", position=applied + 1)
        self.assertNotEqual((await driver.command("set uncertain 5"))["id"], 0)
        failed = await driver.command("failed")
        self.assertEqual((failed["result"], failed["read_started"], failed["reading_after_failure"]),
                         (19, False, False), failed)
        self.assertEqual(failed["fields"], {"proven": 1, "uncertain": 5})
        self.assertEqual(len(failed["pending_ids"]), 1)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 11)
        self.assertEqual(self.own_updates(), [{"proven": 1}])
        stale.release.set()
        await asyncio.wait_for(stale.done.wait(), 5)
        self.assertEqual(stale.outcome, "APPLIED")
        self.assertEqual(self.own_updates(), [{"proven": 1}, {"uncertain": 5}])
        self.trace["states"].append(failed)
        await self.close(driver)

    async def test_production_sdf_invalid_argument_without_the_service_error_info_drops_to_unchained_writes(self):
        # An INVALID_ARGUMENT on a chained write whose only ErrorInfo is of another domain carries no ErrorInfo of
        # the service's domain, so it came from a replica that does not know the chained header and applied
        # nothing: the write is re-sent unchained, and the endpoint's later writes are unchained.
        driver = await self.start(mode="--file-format")
        writer, _, updates = await self.proven_writer(driver)
        self.state.plan("UpdateLayer", "before", attempt=updates + 1, code=grpc.StatusCode.INVALID_ARGUMENT,
                        details=status_details(grpc.StatusCode.INVALID_ARGUMENT, "invalid", "INVALID_UPDATE", {},
                                               domain="other.domain"))
        self.assertNotEqual((await driver.command("set fell 2"))["id"], 0)
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 2), 5)
        # The write is acknowledged before the next turn, which would otherwise coalesce behind it.
        await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertNotEqual((await driver.command("set after 3"))["id"], 0)
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 3), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual(self.own_updates(), [{"proven": 1}, {"fell": 2}, {"after": 3}])
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 3)
        self.assertEqual([entry["chained"] for entry in self.arrivals(writer)], [True, False, False])
        state = await driver.command("state")
        self.assertEqual((state["result"], state["pending_ids"], state["fields"]), (0, [], self.oracle().state))
        await self.close(driver)

    async def test_production_sdf_a_chained_header_refusal_keeps_later_chained_writes_in_flight(self):
        # A replica without chaining refuses the second write of a window INVALID_ARGUMENT without an ErrorInfo
        # while the first is held at its commit step and the two after it wait for it at the service. The endpoint
        # stops chaining, but the later chained calls stay in flight: the refused write is sent unchained only once
        # the chained call of the earlier write has returned, and the later chained calls then commit after it.
        driver = await self.start(mode="--file-format")
        writer, applied, updates = await self.proven_writer(driver)
        hold = self.state.plan_write("hold", position=applied + 1)
        refused = self.state.plan("UpdateLayer", "before", attempt=updates + 2, pause=True,
                                  code=grpc.StatusCode.INVALID_ARGUMENT)
        await self.issue(driver, 4)
        await asyncio.wait_for(hold.reached.wait(), 5)
        await asyncio.wait_for(refused.reached.wait(), 5)
        await asyncio.wait_for(self.state.observed(lambda: len(self.state.waiting) == 2), 5)
        refused.release.set()
        # Failure-only watchdog: the writer has received the refusal.
        deadline = asyncio.get_running_loop().time() + 5
        while not [record for record in (await driver.command("diagnostics 0"))["diagnostics"]["recent_records"]
                   if record["event_name"] == "rpc_failed" and record["fields"].get("result") == 1]:
            self.assertLess(asyncio.get_running_loop().time(), deadline, "the refusal never reached the writer")
        await driver.command("state")
        self.assertEqual([entry["chained"] for entry in self.arrivals(writer)].count(False), 0)
        hold.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 5), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual(self.own_updates(), [{"proven": 1}] + [{f"w{i}": i} for i in range(4)])
        self.assertEqual(self.chain_events("dropped", writer), [])
        self.assertEqual(len(self.chain_events("commit", writer)), 4)  # proven, w0, w2 and w3; w1 unchained.
        self.assertEqual([entry["chained"] for entry in self.arrivals(writer)].count(False), 1)
        state = await driver.command("state")
        self.assertEqual((state["result"], state["pending_ids"], state["fields"]), (0, [], self.oracle().state))
        await self.close(driver)

    async def test_production_sdf_a_fallback_settles_an_uncertain_cancelled_call_through_the_chain(self):
        # The second write of a window gets a plain UNAVAILABLE, so the writer cancels the two chained calls behind
        # it. A cancelled call may still commit later, after its deadline too, so the third and fourth writes are
        # unsettled. The second write's no_wait re-send commits; then the third write's chained re-send meets a
        # replica without chaining while the fourth's chained re-send waits for it at the service. The endpoint
        # stops chaining, but no write is sent unchained before the third is settled through the chain: re-sent
        # chained with no_wait, it commits, and the fourth's chained call, left in flight, commits after it. Later
        # writes are unchained.
        driver = await self.start(mode="--file-format")
        writer, applied, _ = await self.proven_writer(driver)
        # The first write commits with its reply held, so the break resolves only after the cancelled calls end.
        reply = self.state.plan_write("hold_reply", position=applied + 1)
        hold = self.state.plan_write("hold", position=applied + 2)
        stale = self.state.plan_write("stale", position=applied + 2)
        await self.issue(driver, 4)
        await asyncio.wait_for(reply.reached.wait(), 5)
        await asyncio.wait_for(hold.reached.wait(), 5)
        await asyncio.wait_for(self.state.observed(lambda: len(self.state.waiting) == 2), 5)
        second = [entry["operation_id"] for entry in self.state.fault_log if entry["boundary"].startswith("write:hold:")][-1]
        successor = {entry["previous_operation_id"]: entry["operation_id"] for entry in self.state.chain_log
                     if entry["event"] == "wait" and entry["client_id"] == writer}
        third = successor[second]
        fourth = successor[third]
        self.state.pre_chaining_operations.add(third)
        hold.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.chain_events("dropped", writer)) == 2), 5)
        # Failure-only watchdog: both cancelled attempts have ended at the writer.
        deadline = asyncio.get_running_loop().time() + 5
        while len([record for record in (await driver.command("diagnostics 0"))["diagnostics"]["recent_records"]
                   if record["event_name"] == "rpc_failed" and record["fields"].get("grpc_status") == 1]) < 2:
            self.assertLess(asyncio.get_running_loop().time(), deadline, "the cancelled attempts never ended")
        reply.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 5), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual(self.own_updates(), [{"proven": 1}] + [{f"w{i}": i} for i in range(4)])
        self.assertEqual(self.chain_events("unsupported", writer), [third])
        self.assertEqual([(entry["chained"], entry["no_wait"]) for entry in self.arrivals(writer, third)],
                         [(True, False), (True, True)])
        self.assertEqual([(entry["chained"], entry["no_wait"]) for entry in self.arrivals(writer, fourth)],
                         [(True, False), (True, False)])
        self.assertEqual(self.chain_events("dropped", writer), [third, fourth] if
                         self.chain_events("dropped", writer)[0] == third else [fourth, third])
        self.assertEqual(self.chain_events("commit", writer)[-2:], [third, fourth])
        self.assertTrue(all(entry["chained"] for entry in self.arrivals(writer)))
        self.assertNotEqual((await driver.command("set after 9"))["id"], 0)
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 6), 5)
        self.assertFalse(self.arrivals(writer)[-1]["chained"])
        stale.release.set()
        await asyncio.wait_for(stale.done.wait(), 5)
        self.assertEqual(stale.outcome, "ABORTED")  # The re-send committed first, so the late step applies nothing.
        await driver.command(f"wait {self.commits()[-1].sequence}")
        state = await driver.command("state")
        self.assertEqual((state["result"], state["pending_ids"], state["fields"]), (0, [], self.oracle().state))
        await self.close(driver)

    async def test_production_sdf_a_cancelled_call_ended_by_a_plain_status_is_settled_through_the_chain(self):
        # The first write of a window gets a plain UNAVAILABLE, so its outcome is unknown, and the cancelled chained
        # call of the second ends with a plain RESOURCE_EXHAUSTED. Only OK, an error with the service's ErrorInfo or
        # an INVALID_ARGUMENT without one settles a chained call; any other may still commit later, so the second
        # write is unsettled too. The first write's no_wait re-send then meets a replica without chaining, and the
        # endpoint stops chaining, but neither write is sent unchained: each is finished through the chain, re-sent
        # chained with no_wait, and commits. Later writes are unchained.
        driver = await self.start(mode="--file-format")
        writer, applied, updates = await self.proven_writer(driver)
        hold = self.state.plan_write("hold", position=applied + 1)
        stale = self.state.plan_write("stale", position=applied + 1)
        exhausted = self.state.plan("UpdateLayer", "before", attempt=updates + 2, pause=True,
                                    code=grpc.StatusCode.RESOURCE_EXHAUSTED)
        # One worker pass settles both attempts, the first write's before the second's, which the first's unknown
        # outcome cancels; the second's attempt has already ended with the service's status.
        await driver.command("settle-together 2")
        await self.issue(driver, 2)
        await asyncio.wait_for(hold.reached.wait(), 5)
        await asyncio.wait_for(exhausted.reached.wait(), 5)
        first = [entry["operation_id"] for entry in self.state.fault_log if entry["boundary"].startswith("write:hold:")][-1]
        self.state.pre_chaining_operations.add(first)
        exhausted.release.set()
        hold.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 3), 10)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual(self.own_updates(), [{"proven": 1}, {"w0": 0}, {"w1": 1}])
        self.assertEqual(self.chain_events("unsupported", writer), [first])
        self.assertEqual([(entry["chained"], entry["no_wait"]) for entry in self.arrivals(writer, first)],
                         [(True, False), (True, True)])
        second = self.chain_events("commit", writer)[-1]
        self.assertEqual([(entry["chained"], entry["no_wait"]) for entry in self.arrivals(writer, second)],
                         [(True, True)])
        self.assertTrue(all(entry["chained"] for entry in self.arrivals(writer)))
        self.assertNotEqual((await driver.command("set after 9"))["id"], 0)
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 4), 5)
        self.assertFalse(self.arrivals(writer)[-1]["chained"])
        stale.release.set()
        await asyncio.wait_for(stale.done.wait(), 5)
        self.assertEqual(stale.outcome, "ABORTED")  # The re-send committed first, so the late step applies nothing.
        await driver.command(f"wait {self.commits()[-1].sequence}")
        state = await driver.command("state")
        self.assertEqual((state["result"], state["pending_ids"], state["fields"]), (0, [], self.oracle().state))
        await self.close(driver)

    async def test_production_sdf_a_refusal_naming_an_unissued_write_leaves_the_outcome_unknown(self):
        # A chain refusal must name a write this writer issued. The first write's chained attempt is held while the
        # next edit is admitted and waits, never issued, for its local publication. The attempt is then refused
        # naming that admitted write: the writer never issued it, so the chain cannot be located, and the layer fails
        # OUTCOME_UNKNOWN with both edits kept instead of taking the refused write as committed.
        driver = await self.start(mode="--file-format")
        writer, _, updates = await self.proven_writer(driver)
        refused = self.state.plan("UpdateLayer", "before", attempt=updates + 1, pause=True,
                                  code=grpc.StatusCode.ABORTED)
        self.assertNotEqual((await driver.command("set first 1"))["id"], 0)
        await asyncio.wait_for(refused.reached.wait(), 5)
        await driver.command("direct-set-hold-local-publication second 2")
        writes = (await driver.command("operations"))["writes"]
        self.assertEqual([entry["attempts"] for entry in writes], [1, 0])
        refused.details = status_details(grpc.StatusCode.ABORTED, "refused", "PREDECESSOR_NOT_COMMITTED",
                                         dict(last_operation_id=writes[1]["operation_id"], last_record="operation"))
        refused.release.set()
        failed = await driver.command("await-state 7 19")  # FAILED, OUTCOME_UNKNOWN
        self.assertEqual(failed["fields"], {"proven": 1, "first": 1, "second": 2})
        self.assertEqual(len(failed["pending_ids"]), 2)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates + 1)
        self.assertEqual(self.own_updates(), [{"proven": 1}])
        await driver.command("release-held")
        self.trace["states"].append(failed)
        await self.close(driver)

    async def test_production_sdf_a_service_without_operation_ids_issues_one_unchained_write(self):
        # Without operation-ID support the window is one write, no update carries a predecessor, and turns authored
        # while it is in flight coalesce into the next update.
        self.state.operation_ids = False
        driver = await self.start(mode="--file-format")
        held = self.state.plan("UpdateLayer", "before", attempt=self.state.attempts.get("UpdateLayer", 0) + 1,
                               pause=True)
        self.assertNotEqual((await driver.command("set w0 0"))["id"], 0)
        await asyncio.wait_for(held.reached.wait(), 5)
        for index in range(1, 4):
            self.assertEqual((await driver.command(f"set w{index} {index}"))["id"], 0)  # Deferred, coalescing.
        held.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 2), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual(self.own_updates(), [{"w0": 0}, {f"w{i}": i for i in range(1, 4)}])
        self.assertEqual(self.state.chain_log, [])
        metrics = (await driver.command("diagnostics 0"))["diagnostics"]["metrics"]["layer_writes"]
        self.assertEqual((metrics["in_flight_high_water"], metrics["window_high_water"]), (1, 1))
        await self.close(driver)

    async def test_production_sdf_cheap_turns_authored_faster_than_writes_settle_coalesce_beyond_the_window(self):
        # Admitted intents never exceed the window of 8: turns authored faster than the writes settle, here while
        # the first is held, coalesce into one update once a slot frees.
        driver = await self.start(mode="--file-format")
        hold = self.state.plan_write("hold", position=1)
        self.assertNotEqual((await driver.command("set c0 0"))["id"], 0)
        await driver.command("issued")
        await asyncio.wait_for(hold.reached.wait(), 5)
        for index in range(1, 8):
            # Each admitted write is issued before the next turn, which would otherwise coalesce behind it.
            self.assertNotEqual((await driver.command(f"set c{index} {index}"))["id"], 0)
            await driver.command("issued")
        turns = 50
        for index in range(8, turns):
            self.assertEqual((await driver.command(f"set c{index} {index}"))["id"], 0)  # Deferred, coalescing.
            self.assertEqual(len((await driver.command("state"))["pending_ids"]), 8)
        hold.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 9), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual(self.own_updates(), [{f"c{i}": i} for i in range(8)] + [{f"c{i}": i for i in range(8, turns)}])
        state = await driver.command("state")
        self.assertEqual((state["result"], state["pending_ids"], state["fields"]), (0, [], self.oracle().state))
        await self.close(driver)

    async def fill_issuance_window(self, driver):
        """Confirm one write, which marks the endpoint as chaining, then admit eight more whose first the service
        holds at its commit step: the window of eight is full. Returns the hold and the base commit count."""
        base = len(self.commits())
        self.assertNotEqual((await driver.command("set a 1"))["id"], 0)
        await asyncio.wait_for(self.state.observed(lambda: len(self.commits()) == base + 1), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")  # Its acknowledgment has settled.
        hold = self.state.plan_write("hold", position=2)
        for index in range(8):
            # Each admitted write is issued before the next turn, which would otherwise coalesce behind it.
            self.assertNotEqual((await driver.command(f"set w{index} {index}"))["id"], 0)
            await driver.command("issued")
        await asyncio.wait_for(hold.reached.wait(), 5)
        return hold, base

    async def test_production_sdf_turns_coalesce_while_the_issuance_window_is_full(self):
        # While admitted writes fill the issuance window, a host turn's flush defers and its records coalesce
        # with later turns into one update once a write settles (design section 8); each turn stays whole.
        driver = await self.start(mode="--file-format")
        hold, base = await self.fill_issuance_window(driver)
        # A deferred turn costs no compilation: the buffer compiles once, when a write settles.
        compiles = (await driver.command("compiles"))["count"]
        turns = 200
        for index in range(turns):
            self.assertEqual((await driver.command(f"set c{index % 3} {index}"))["id"], 0)  # Deferred, coalescing.
        self.assertEqual((await driver.command("compiles"))["count"], compiles)
        self.assertEqual(len(self.commits()), base + 1)
        hold.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.commits()) == base + 10), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        self.assertEqual((await driver.command("compiles"))["count"], compiles + 1)
        updates = [decode_update(commit.payload) for commit in self.commits()[base:]]
        self.assertEqual(updates, [{"a": 1}] + [{f"w{i}": i} for i in range(8)] +
                         [{"c0": turns - 2, "c1": turns - 1, "c2": turns - 3}])
        state = await driver.command("state")
        self.assertEqual(state["fields"], self.oracle().state)
        metrics = (await driver.command("diagnostics 0"))["diagnostics"]["metrics"]["layer_writes"]
        self.trace["layer_writes"] = metrics
        # Eight acknowledgments without a refusal widen the window to nine.
        self.assertEqual((metrics["in_flight"], metrics["window_high_water"], metrics["window_grown"],
                          metrics["max_turns_per_update"]), (0, 9, 1, turns))
        self.assertEqual((metrics["admitted_updates"], metrics["coalesced_turns"]), (10, 9 + turns))
        await self.close(driver)

    async def test_production_sdf_frames_coalesce_through_publication_jobs_while_the_window_is_full(self):
        # Each frame authors one change block, then a peer update arrives and the host runs the publication job
        # that installs it, which flushes first. While the window is full that flush defers too, so admitted
        # intents stay at the window and the frames' turns coalesce into one update when the window frees.
        driver = await self.start(mode="--file-format")
        hold, base = await self.fill_issuance_window(driver)
        frames = 6
        for frame in range(frames):
            self.assertEqual((await driver.command(f"set c{frame} {frame}"))["id"], 0)
            sequence = await self.peer({f"p{frame}": frame})
            applied = await driver.command(f"applied {sequence}")  # Its publication job, and so its flush, ran.
            self.assertEqual(len(applied["pending_ids"]), 8, applied)
        hold.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.own_updates()) == 10), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        updates = [update for update in self.own_updates() if any(key.startswith("c") for key in update)]
        self.assertEqual(updates, [{f"c{i}": i for i in range(frames)}])
        state = await driver.command("state")
        self.assertEqual((state["result"], state["pending_ids"], state["fields"]), (0, [], self.oracle().state))
        await self.close(driver)

    async def test_production_sdf_coalescing_stops_before_an_entry_limit(self):
        # Each turn alone fits the 3-entry limit and four coalesced turns do not: the turns that fit are admitted
        # as one update and the rest follow in another, so merging never fails the layer.
        driver = await self.start(mode="--file-format", extra=("--max-delta-entries=3",))
        hold, base = await self.fill_issuance_window(driver)
        for index in range(4):
            await driver.command(f"set c{index} {index}")
        hold.release.set()
        await asyncio.wait_for(self.state.observed(lambda: len(self.commits()) == base + 11), 5)
        await driver.command(f"wait {self.commits()[-1].sequence}")
        state = await driver.command("state")
        self.assertEqual((state["result"], state["fields"]), (0, self.oracle().state))
        coalesced = [decode_update(commit.payload) for commit in self.commits()[base + 9:]]
        self.assertEqual(len(coalesced), 2)
        self.assertGreater(len(coalesced[0]), 1)
        self.assertEqual([key for update in coalesced for key in update], ["c0", "c1", "c2", "c3"])
        await self.close(driver)

    async def test_production_sdf_flush_over_the_entry_limit_fails_the_layer_and_keeps_host_edits(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=("--max-delta-entries=16",))
        updates = self.state.attempts["UpdateLayer"]
        # One host turn of 17 root fields, one past the 16-entry limit. Like
        # any compile limit, it fails the layer and issues nothing.
        fields = {f"entry{i}": i for i in range(17)}
        direct = await driver.command("direct-set " + " ".join(f"{k} {v}" for k, v in fields.items()))
        self.assertEqual(direct["result"], 13, direct)  # FAILED with RESOURCE_EXHAUSTED, not deferred.
        self.assertEqual(direct["fields"], {"initial": 41, **fields})  # Host content keeps the edits.
        status = await driver.command("await-state 7 13")  # FAILED, RESOURCE_EXHAUSTED
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        exported = await driver.command(f"export-usd {self.export_directory() / 'entry-limit.usdc'}")
        self.assertEqual(exported["fields"], direct["fields"])  # SdfLayer::Export keeps every edit.
        self.assertEqual(self.state.attempts["UpdateLayer"], updates, "export issued service RPCs")
        self.trace["states"] += [direct, status, exported]
        await self.close(driver)

    async def test_production_sdf_whole_map_samples_over_the_entry_limit_fail_the_layer(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=("--max-delta-entries=16",))
        updates = self.state.attempts["UpdateLayer"]
        # One host turn: a prim and an attribute whose whole timeSamples map of
        # 20 keys alone exceeds the 16-entry limit. A runtime error names the
        # counts in the service's terms, and the host content keeps the prim.
        failed = await driver.command("direct-animate Anim 20")
        self.assertEqual((failed["result"], failed["exists"], failed["pending"]), (13, True, 0), failed)
        self.assertEqual(len(failed["errors"]), 1, failed)
        self.assertIn("entry limits (DELTA_TOO_LARGE struct_commands=2 set_fields=2 time_samples=20 max_entries=16); "
                      "nothing was sent and the layer fails", failed["errors"][0])
        status = await driver.command("await-state 7 13")  # FAILED, RESOURCE_EXHAUSTED
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        self.trace["states"] += [failed, status]
        await self.close(driver)

    async def test_production_sdf_reorder_over_the_children_list_limit_fails_the_layer(self):
        driver = await self.start(mode="--file-format", extra=("--max-children-list-entries=3",))
        base = len(self.commits())
        # A parent, then four children appended one turn each: Creates and no
        # Reorder, so the turns are admitted, alone or coalesced.
        created = await driver.command("direct-parent Parent 4")
        self.assertEqual((created["result"], created["errors"]), (0, []), created)
        # Turns that coalesced behind a write in flight are admitted by the flush a settled write requests, after
        # the drain a save covers: save until no host edit is buffered. Failure-only watchdog.
        deadline = asyncio.get_running_loop().time() + 10
        while True:
            self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
            if not (await driver.command("state"))["dirty"]:
                break
            self.assertLess(asyncio.get_running_loop().time(), deadline, "coalesced turns were never admitted")
        self.assertGreater(len(self.commits()), base)
        updates = self.state.attempts["UpdateLayer"]
        # Reversing them is one Reorder naming all four, over the 3-entry list limit.
        failed = await driver.command("direct-reverse Parent")
        self.assertEqual(failed["result"], 13, failed)  # RESOURCE_EXHAUSTED
        self.assertEqual(len(failed["errors"]), 1, failed)
        self.assertIn("entry limits (CHILDREN_LIST_TOO_LONG limit=3 actual=4); nothing was sent and the layer fails",
                      failed["errors"][0])
        status = await driver.command("await-state 7 13")  # FAILED, RESOURCE_EXHAUSTED
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        self.trace["states"] += [created, failed, status]
        await self.close(driver)

    async def test_production_sdf_export_over_the_entry_limit_issues_no_create(self):
        driver = await self.start(mode="--file-format", extra=("--max-delta-entries=4",))
        creates = self.state.attempts.get("CreateLayer", 0)
        # The pseudoroot's section and four fields: five entries, one past the limit.
        refused = await driver.command("export-live-fields over 4")
        self.assertFalse(refused["succeeded"], refused)
        self.assertTrue(any("entry limits (DELTA_TOO_LARGE struct_commands=1 set_fields=4 time_samples=0 "
                            "max_entries=4); nothing was sent" in error for error in refused["errors"]), refused)
        self.assertEqual(self.state.attempts.get("CreateLayer", 0), creates)
        self.assertNotIn("over.live", self.state.sessions[self.session].layers)
        # Exactly at the limit, the seed is created.
        exact = await driver.command("export-live-fields exact 3")
        self.assertTrue(exact["succeeded"], exact)
        self.assertEqual(self.state.attempts.get("CreateLayer", 0), creates + 1)
        created = self.state.sessions[self.session].layers["exact.live"].commits
        self.assertEqual([commit.operation for commit in created], ["CreateLayer"])
        self.trace["states"] += [refused, exact]
        await self.close(driver)

    async def test_production_sdf_endpoint_release_retries_a_flush_another_layer_blocked(self):
        await self.peer({"initial": 41})
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="other.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=snapshot({"other": 1})))]), metadata=WRITER, timeout=5)
        # Two standing read reservations (4160 bytes each) and 4394 more
        # bytes: enough for the second layer's StatLayer response allowance
        # (4096) beside its catch-up frames (a 230-byte snapshot frame and a
        # 68-byte caught-up header) while it opens, not for the wide edit's
        # admission (twice its bytes plus about 480, over 4394). Once one read
        # releases, it fits.
        driver = await self.start(mode="--file-format", extra=("limits=4096,12714",))
        base, updates = len(self.commits()), self.state.attempts["UpdateLayer"]
        await driver.command("await-standing-reads 1")
        await driver.command("open-second other.live")
        # The endpoint's pending bytes are exactly the two standing read
        # reservations. A caught-up header may still arrive; the headroom
        # holds it, and the wide edit's admission is refused either way.
        await driver.command("await-standing-reads 2")
        fields = {f"wide{i}": i for i in range(2 * TRANSIENT_FIELDS)}
        await driver.command("direct-set " + " ".join(f"{k} {v}" for k, v in fields.items()))
        await driver.command("direct-set y 6")
        await driver.command("await-state 5 13")
        blocked = await driver.command("status")
        self.assertEqual((blocked["state"], blocked["status_result"], blocked["pending"], blocked["buffered"]),
                         (5, 13, 0, len(fields) + 1), blocked)
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        # Nothing of this layer changes; releasing the other layer's read
        # reservation is the only event, and it retries the blocked flush.
        await driver.command("drop-second")
        ready = await driver.command("await-buffered 0")
        self.assertEqual((ready["state"], ready["status_result"]), (2, 0), ready)
        settled = await driver.command(f"wait {base + 1}")
        self.assertEqual(settled["fields"], {"initial": 41, **fields, "y": 6})
        self.assertEqual(settled["fields"], self.oracle().state)
        self.assertEqual(decode_update(self.commits()[-1].payload), {**fields, "y": 6})
        self.trace["states"] += [blocked, ready, settled]
        await self.close(driver)

    async def test_production_open_allocation_failures_before_the_first_rpc_leave_nothing_behind(self):
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="other.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=snapshot({"other": 1})))]), metadata=WRITER, timeout=5)
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="control.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=snapshot({"control": 1})))]), metadata=WRITER, timeout=5)
        driver = await self.start(mode="--file-format")
        before = dict(self.state.attempts)
        sweep = await driver.command("alloc-sweep-open other.live control.live 512", timeout=120)
        self.assertEqual((sweep["complete"], sweep["control"]), (True, 2), sweep)
        if sweep["injection"]:
            new, read = sweep["new"], sweep["read"]
            self.assertGreaterEqual(new["sites"], 2, sweep)
            self.assertGreaterEqual(read["sites"], 8, sweep)
            # Setup failures (S) are allocations in a constructed layer's adoption; how many the sweep meets depends
            # on where the platform's standard library allocates, and each reports the allocation cause below.
            self.assertEqual(new["sites"], new["threw"] + new["refused"] + new["setup"] + new["absorbed"] + new["unbacked"])
            self.assertEqual(read["sites"], read["failed"] + read["absorbed"])
            # Every failure before the first RPC reports one allocation cause.
            reported = new["refused"] + new["setup"]
            self.assertEqual(new["codes"], {"13": reported} if reported else {})
            self.assertEqual(read["codes"], {"13": read["failed"]} if read["failed"] else {})
            # A New that absorbed its allocation failure reports the resolver's
            # allocation cause or nothing.
            self.assertLessEqual(set(new["absorbed_codes"]), {"13", "-1"}, sweep)
            self.assertEqual(sum(new["absorbed_codes"].values()), new["absorbed"] + new["unbacked"], sweep)
            # An absorbed New under a fresh identifier opens a layer the service lacks.
            attached = new["absorbed"] + new["unbacked"] + read["absorbed"] + 2
        else:
            attached = 2
        # One subscription per attached or unbacked open; no mutating RPC.
        self.assertEqual(self.state.attempts["SubscribeReadLayer"] - before.get("SubscribeReadLayer", 0), attached)
        changed = {kind for kind, count in self.state.attempts.items() if count != before.get(kind, 0)}
        self.assertLessEqual(changed, {"Ping", "StatLayer", "SubscribeReadLayer"}, self.state.attempts)
        await self.close(driver)

    async def test_production_sdf_flush_that_issues_nothing_installs_a_deferred_publication(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format", extra=("--max-pending=1",))
        hold = self.state.plan("UpdateLayer", "before", pause=True)
        self.addCleanup(hold.release.set)
        await driver.command("direct-set a 1")
        await asyncio.wait_for(hold.reached.wait(), 5)
        # A temporary prim's Create waits behind the pending-intent limit.
        await driver.command("direct-create-prim Temp")
        await driver.command("await-state 5 13")
        sequence = await self.peer({"peer": 9})
        deferred = await driver.command(f"applied {sequence}")
        self.assertEqual(deferred["fields"], {"initial": 41, "a": 1})
        # Removing it again compiles the buffer to nothing. The save inside
        # that host job installs nothing itself (a caller's change block may
        # enclose it); the queued flush job, whose flush finds the buffer
        # empty and issues nothing, installs the deferred peer content.
        saved = await driver.command("direct-remove-save Temp")
        self.assertIs(saved["saved"], False, saved)  # a is still pending: BUSY.
        self.assertEqual(saved["fields"], {"initial": 41, "a": 1})
        installed = await driver.command("state")  # Its host read runs after the queued flush job.
        self.assertEqual(installed["fields"], {"initial": 41, "a": 1, "peer": 9})
        ready = await driver.command("await-buffered 0")
        self.assertEqual((ready["state"], ready["status_result"]), (2, 0), ready)
        hold.release.set()
        settled = await driver.command(f"wait {sequence + 1}")
        self.assertEqual(settled["fields"], {"initial": 41, "a": 1, "peer": 9})
        self.assertEqual(settled["fields"], self.oracle().state)
        self.trace["states"] += [deferred, saved, installed, ready, settled]
        await self.close(driver)

    async def test_production_sdf_unsupported_flush_fails_layer_and_keeps_recording(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        updates = self.state.attempts["UpdateLayer"]
        unencodable = await driver.command("direct-set-unencodable probe")
        self.assert_unsupported_flush(unencodable, "FlushProbe", "value type is not representable in version zero")
        await self.assert_flush_unsupported_record(driver, 1)
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 12, failed)  # UNSUPPORTED
        later = await driver.command("direct-set after 7")
        self.assertEqual(later["fields"]["after"], 7)
        # FAILED records keep growing the buffer; nothing is issued.
        status = await driver.command("await-buffered 2")
        self.assertEqual((status["state"], status["status_result"], status["pending"], status["unconfirmed"]), (7, 12, 0, 1), status)
        saved = await driver.command("save 12")
        self.assertTrue(saved["dirty"])
        self.assertEqual(saved["fields"], {"initial": 41, "after": 7})
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        self.trace["states"] += [failed, later, status, saved]
        await self.close(driver)

    def assert_unsupported_flush(self, report, type_name, reason):
        # One USD runtime error names the layer, path, field, value type and
        # reason; `reason` is the exact UnsupportedOperation::describe text.
        self.assertEqual(report["result"], 12, report)  # UNSUPPORTED
        errors = [e for e in report["errors"] if "unsupported edit" in e]
        self.assertEqual(len(errors), 1, report)
        for part in (report["identifier"], "at / field probe, value type ", f"{type_name}: {reason} (result=12)"):
            self.assertIn(part, errors[0])
        self.assertNotIn("synthetic-writer", errors[0])

    async def assert_flush_unsupported_record(self, driver, reason):
        # One numeric flush_unsupported record correlates with the failed layer.
        report = await driver.command("diagnostics 12")
        records = [r for r in report["diagnostics"]["recent_records"] if r["event_name"] == "flush_unsupported"]
        self.assertEqual(len(records), 1, records)
        record = records[0]
        self.assertEqual((record["event_code"], record["subject_id"]), (30, report["subject"]), record)
        self.assertNotEqual(record["incident_id"], 0, record)
        self.assertEqual(record["fields"], {"operation": 4, "result": 12, "reason": reason}, record)
        self.assertNotIn("canonical_resource", record)

    async def test_production_sdf_payload_offset_flush_names_its_operation(self):
        await self.peer({"initial": 41})
        driver = await self.start(mode="--file-format")
        updates = self.state.attempts["UpdateLayer"]
        report = await driver.command("direct-set-payload-offset probe")
        self.assert_unsupported_flush(report, "SdfPayload", "payload offset is not representable in version zero")
        await self.assert_flush_unsupported_record(driver, 2)
        self.assertEqual(report["fields"], {"initial": 41})
        self.assertEqual(self.state.attempts["UpdateLayer"], updates)
        self.trace["states"] += [report]
        await self.close(driver)

    async def test_production_sdf_last_layer_release_suppresses_queued_publication(self):
        driver = await self.start(mode="--file-format")
        initial = await driver.command("state")
        await driver.command("arm")
        sequence = await self.peer({"late": 57})
        held = await driver.command("held")
        self.assertEqual(held["covered"], sequence)
        self.assertEqual(held["applied"], sequence - 1)
        self.assertEqual(held["fields"], initial["fields"])
        dropped = await driver.command("drop-queued")
        self.assertIs(dropped["dropped"], True)
        self.assertEqual(dropped["applied"], initial["applied"])
        self.assertEqual(dropped["journal_applied"], initial["journal_applied"])
        self.assertEqual(dropped["fields"], initial["fields"])
        self.assertGreater(dropped["retained"], 0)
        self.assertEqual(self.oracle().state, {"late": 57})
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.trace["states"] += [initial, held, dropped]
        await self.close(driver)

    async def test_production_sdf_save_alternating_clients_preserves_existing_object(self):
        drivers = await asyncio.gather(*(self.start(mode="--file-format") for _ in range(2)))
        expected = {}
        for index in range(4):
            driver = drivers[index % 2]
            key, value = f"client_{index % 2}", 31 + index
            accepted = await driver.command(f"set {key} {value}")
            self.assertEqual(accepted["result"], 0)
            self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
            expected[key] = value
            for observer in drivers:
                state = await observer.command(f"wait {index + 2}")
                self.assertEqual(state["fields"], expected)
                self.assertEqual(state["object"], 1)
                self.assertEqual(state["pending_ids"], [])
                saved = await observer.command("save 0")  # Default Save(false), not forced Save.
                self.assertFalse(saved["dirty"])
                self.trace["states"].append(saved)
            self.assertEqual(self.oracle().state, expected)
        self.assertEqual(self.state.attempts["CreateLayer"], 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)
        self.assertEqual(self.state.attempts["UpdateLayer"], 4)
        self.assertEqual([decode_update(c.payload) for c in self.commits()[1:]],
                         [{f"client_{i % 2}": 31 + i} for i in range(4)])
        await asyncio.gather(*(self.close(driver) for driver in drivers))

    async def test_production_sdf_save_waits_for_local_publication_without_issuing(self):
        driver = await self.start(mode="--file-format")
        await driver.command("direct-set-hold-local-publication held_local 71")
        held = await driver.command("held")
        self.assertEqual(len(held["pending_ids"]), 1)
        self.assertEqual([decode_update(bytes.fromhex(v)) for v in held["pending_hex"]], [{"held_local": 71}])
        self.assertEqual(held["fields"], {"held_local": 71})  # Host content carries the edit; its publication is held.
        self.assertTrue(held["dirty"])
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), 0)
        busy = await driver.command("save 14")
        self.assertEqual((await driver.command("save-wait 25"))["result"], 9)
        still_held = await driver.command("state")
        for key in ("pending_ids", "pending_hex", "applied", "fields"):
            self.assertEqual(still_held[key], held[key])
        self.assertTrue(still_held["dirty"])
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), 0)
        self.assertEqual(len(self.commits()), 1)
        await driver.command("release-held")
        self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
        saved = await driver.command("save 0")
        self.assertEqual(saved["fields"], {"held_local": 71})
        self.assertEqual(saved["fields"], self.oracle().state)
        self.assertEqual(saved["pending_ids"], [])
        self.assertFalse(saved["dirty"])
        self.assertEqual((saved["object"], self.state.attempts["CreateLayer"],
                          self.state.attempts["SubscribeReadLayer"], self.state.attempts["UpdateLayer"]), (1, 1, 1, 1))
        self.trace["states"] += [held, busy, still_held, saved]
        await self.close(driver)

    async def test_production_sdf_save_preserves_unknown_commit_and_dirty_failure(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        driver = await self.start(mode="--file-format")
        fault = self.state.plan("UpdateLayer", "after_commit", pause=True, code=grpc.StatusCode.UNAVAILABLE)
        accepted = await driver.command("set uncertain 83")
        await asyncio.wait_for(fault.reached.wait(), 5)
        before = await driver.command("state")
        self.assertEqual(before["pending_ids"], [accepted["id"]])
        self.assertEqual([decode_update(bytes.fromhex(v)) for v in before["pending_hex"]], [{"uncertain": 83}])
        fault.release.set()
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 19)  # Original OUTCOME_UNKNOWN cause remains inspectable.
        self.assertEqual((await driver.command("save-wait 1000"))["result"], 19)
        saved = await driver.command("save 19")
        self.assertTrue(saved["dirty"])
        self.assertEqual(saved["result"], 19)
        for key in ("pending_ids", "pending_hex", "applied", "fields", "object"):
            self.assertEqual(saved[key], before[key])
        self.assertEqual(saved["fields"], {"uncertain": 83})
        self.assertEqual(saved["fields"], self.oracle().state)
        self.assertEqual((self.state.attempts["CreateLayer"], self.state.attempts["SubscribeReadLayer"],
                          self.state.attempts["UpdateLayer"]), (1, 1, 1))
        self.assertEqual([decode_update(c.payload) for c in self.commits()[1:]], [{"uncertain": 83}])
        captured = await driver.command("capture-host")
        self.assertEqual((captured["result"], captured["publication_result"], captured["applied"], captured["journal_applied"]),
                         (19, 19, 1, 1))
        self.assertEqual((captured["last_attempted_revision"], captured["last_completed_revision"]), (2, 2))
        self.assertEqual(captured["host_fields"], {"uncertain": 83})
        self.assertEqual(captured["journal_fields"], captured["host_fields"])
        attempts = dict(self.state.attempts)
        exported = await driver.command(f"export-usd {self.export_directory() / 'unknown-original.usdc'}")
        self.assertEqual(exported["fields"], {"uncertain": 83})
        self.assertEqual(self.state.attempts, attempts, "export retried the unknown write or issued discovery")
        after_export = await driver.command("state")
        for key in ("pending_ids", "pending_hex", "applied", "fields", "object", "result"):
            self.assertEqual(after_export[key], saved[key])
        self.trace["states"] += [before, failed, saved, captured, exported]
        closed = await self.close(driver)
        self.assertEqual(closed["pending_ids"], before["pending_ids"])
        self.assertEqual(closed["pending_hex"], before["pending_hex"])

    async def test_save_diagnostics_exclude_dirty_queries_and_keep_the_failed_layer(self):
        # A service without operation IDs cannot answer a re-send, so the lost
        # reply below leaves its write's outcome unknown.
        self.state.operation_ids = False
        driver = await self.start(mode='--file-format')
        async def capture(expected=0):
            report = await driver.command(f'diagnostics {expected}')
            snapshot = report['diagnostics']
            self.assertFalse(snapshot['history']['truncated'])
            records = [r for r in snapshot['recent_records'] if r['event_name'] == 'save_completed']
            self.assertTrue(all(r['subject_id'] == report['subject'] for r in records))
            self.assertFalse(any(r['operation'] == 5 for r in snapshot['metrics']['rpc_attempts_by_operation']))
            self.assertFalse(any(r['operation'] == 5 for r in snapshot['metrics']['rpc_by_operation_result']))
            durations = snapshot['metrics']['save_duration_ms_by_result']
            self.assertEqual(durations['bounds'], [1, 5, 10, 50, 100, 500, 1000, 5000])
            expected_rows = {}
            for record in records:
                result, duration = record['fields']['result'], record['fields']['duration_ms']
                row = expected_rows.setdefault(result, dict(result=result, count=0, sum_ms=0, counts=[0] * 9))
                row['count'] += 1
                row['sum_ms'] += duration
                row['counts'][sum(duration > bound for bound in durations['bounds'])] += 1
            self.assertEqual(durations['entries'], [expected_rows[r] for r in sorted(expected_rows)])
            self.trace['states'].append(report)
            return snapshot, records
        await driver.command('state')
        self.assertEqual((await capture())[1], [])
        await driver.command('direct-set-hold-local-publication held_local 71')
        held = await driver.command('held')
        self.assertEqual(len(held['pending_ids']), 1)
        self.assertTrue(held['dirty'])
        self.assertEqual((await capture())[1], [], 'dirty queries emitted save events')
        await driver.command('save 14')
        _, records = await capture()
        self.assertEqual(len(records), 1)
        busy = records[0]
        self.assertEqual((busy['severity'], busy['incident_id']), (1, 0))
        self.assertEqual(tuple(busy['fields'][k] for k in ('operation', 'result', 'reason',
                          'target_local_ordinal', 'local_retired', 'pending_writes')), (5, 14, 0, 1, 0, 1))
        self.assertGreater(busy['fields']['pending_bytes'], 0)
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), 0)
        await driver.command('release-held')
        self.assertEqual((await driver.command('save-wait 5000'))['result'], 0)
        saved = await driver.command('save 0')  # Clean default skip, then actual forced hook.
        self.assertFalse(saved['dirty'])
        await driver.command('state')
        _, records = await capture()
        self.assertEqual([r['fields']['result'] for r in records], [14, 0, 0])
        for record in records[1:]:
            self.assertEqual(tuple(record['fields'][k] for k in ('target_local_ordinal', 'local_retired',
                              'pending_writes', 'pending_bytes')), (1, 1, 0, 0))
        fault = self.state.plan('UpdateLayer', 'after_commit', pause=True, code=grpc.StatusCode.UNAVAILABLE)
        self.addCleanup(fault.release.set)
        uncertain = await driver.command('set uncertain 83')
        await asyncio.wait_for(fault.reached.wait(), 5)
        fault.release.set()
        # Initial status delivery or its terminal callback fences the incident;
        # an actor-progress poll alone would not prove status publication.
        snapshot, records = await capture(19)
        self.assertEqual(len(records), 3)
        status = next(s for s in snapshot['statuses'] if s['subject_kind'] == 3)
        self.assertGreater(status['incident_id'], 0)
        retained = await driver.command('state')
        self.assertEqual(retained['pending_ids'], [uncertain['id']])
        self.assertEqual(retained['pending_hex'], [self.commits()[-1].payload.hex()])
        self.assertEqual((await driver.command('save-wait 5000'))["result"], 19)
        after_wait = await driver.command('state')
        failed = await driver.command('save 19')
        self.assertTrue(failed['dirty'])
        self.assertEqual(failed['pending_ids'], [uncertain['id']])
        self.assertEqual(failed['fields'], self.oracle().state)
        for report in (after_wait, failed):
            for key in ('pending_ids', 'pending_hex'):
                self.assertEqual(report[key], retained[key])
        snapshot, records = await capture(19)
        self.assertEqual([r['fields']['result'] for r in records], [14, 0, 0, 19, 19])
        for record in records[3:]:
            self.assertEqual((record['severity'], record['incident_id']), (4, status['incident_id']))
            self.assertEqual(tuple(record['fields'][k] for k in ('reason', 'target_local_ordinal',
                              'local_retired', 'pending_writes')), (19, 2, 1, 1))
        self.assertEqual([e for e in snapshot['metrics']['events'] if e['event_code'] == 17],
                         [dict(event_code=17, count=5)])
        self.assertEqual(self.state.attempts['UpdateLayer'], 2)  # The confirmed write and the uncertain one.
        for private in ('synthetic-writer', 'Bearer', self.address, self.session, 'root.live', 'held_local', 'uncertain'):
            self.assertNotIn(private, json.dumps(snapshot))
        closed = await self.close(driver)
        for key in ('pending_ids', 'pending_hex'):
            self.assertEqual(closed[key], retained[key])

    async def test_production_sdf_save_accepts_service_clock_rollback(self):
        driver = await self.start(mode="--file-format")
        created = self.commits()[0].timestamp_us
        self.state.clock = lambda: created - 1000000
        await driver.command("set clock_rollback 31")
        self.assertEqual((await driver.command("save-wait 5000"))["result"], 0)
        saved = await driver.command("save 0")  # Both default and forced Sdf Save.
        self.assertFalse(saved["dirty"])
        self.assertEqual(saved["fields"], self.oracle().state)
        self.assertEqual(saved["fields"], {"clock_rollback": 31})
        self.assertEqual(self.commits()[-1].timestamp_us, created - 1000000)
        self.assertEqual(self.state.attempts["CreateLayer"], 1)
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        self.trace["states"].append(saved)
        await self.close(driver)

    async def test_production_sdf_save_after_off_host_save_while_retirement_is_held(self):
        driver = await self.start(mode="--file-format")
        held = await driver.command("save-held-retirement held_retirement 72")
        self.assertEqual(held["offhost"], 0)
        self.assertTrue(held["onhost"])  # The actor worker is held after completing the edit's drain tickets.
        self.assertEqual(held["fields"], {"held_retirement": 72})
        saved = await driver.command("save 0")
        self.assertFalse(saved["dirty"])
        self.assertEqual(saved["fields"], self.oracle().state)
        self.assertEqual(saved["pending_ids"], [])
        self.trace["states"] += [held, saved]
        await self.close(driver)

    async def test_production_sdf_clean_save_ignores_held_peer_publication(self):
        driver = await self.start(mode="--file-format")
        await driver.command("arm")
        sequence = await self.peer({"peer_only": 97})
        held = await driver.command("held")
        self.assertEqual((held["covered"], held["applied"]), (sequence, sequence - 1))
        self.assertEqual(held["pending_ids"], [])
        self.assertEqual((await driver.command("save-wait 0"))["result"], 0)
        saved = await driver.command("save 0")
        self.assertFalse(saved["dirty"])
        self.assertEqual(saved["fields"], {})
        self.assertEqual(saved["applied"], held["applied"])
        self.assertGreater(saved["retained"], 0)
        self.assertEqual(self.state.attempts["CreateLayer"], 1)
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)  # The independent peer only.
        await driver.command("release-held")
        current = await driver.command(f"wait {sequence}")
        self.assertEqual(current["fields"], self.oracle().state)
        self.trace["states"] += [held, saved, current]
        await self.close(driver)

    async def test_prepared_bootstrap_activates_on_opening_context_before_queued_peers(self):
        driver = await self.start(mode="--prepared-publication")
        prepared = await driver.command("state")
        self.assertEqual((prepared["applied"], prepared["journal_applied"], prepared["published"]), (0, 1, 0))
        self.assertEqual(prepared["dispatches"], 0)
        self.assertEqual(prepared["fields"], {})
        sequence = await self.peer({"peer": 12})
        retained = await driver.command(f"retained {sequence} {update({'peer': 12}).hex()}")
        self.assertEqual((retained["covered"], retained["applied"], retained["retained"]), (sequence, 0, 2))
        self.assertEqual((retained["published"], retained["dispatches"]), (0, 0))
        activated = await driver.command("activate")
        self.assertEqual(activated["result"], 0)
        self.assertEqual((await driver.command("activate"))["result"], 0)  # Idempotent activation.
        current = await driver.command(f"wait {sequence}")
        self.assertEqual(current["fields"], self.oracle().state)
        self.assertEqual(current["dispatches"], 1)  # Bootstrap never posted to the blocked caller.
        await driver.command("set local 23")
        current = await driver.command(f"wait {sequence + 1}")
        self.assertEqual(current["fields"], {"peer": 12, "local": 23})
        self.assertEqual(current["fields"], self.oracle().state)
        self.trace["states"] += [prepared, retained, current]
        await self.close(driver)

    async def test_prepared_bootstrap_close_retires_handoff_without_publication(self):
        driver = await self.start(mode="--prepared-publication")
        closed = await self.close(driver)
        self.assertEqual((closed["applied"], closed["journal_applied"], closed["published"]), (0, 1, 0))
        self.assertEqual(closed["dispatches"], 0)
        self.assertEqual(closed["retained"], 1)
        self.trace["states"].append(closed)

    async def test_prepared_bootstrap_deadline_prevents_late_activation(self):
        driver = await self.start(mode="--prepared-timeout")
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 9)
        self.assertEqual((failed["applied"], failed["published"], failed["dispatches"]), (0, 0, 0))
        self.assertEqual((await driver.command("activate"))["result"], 9)
        self.trace["states"].append(failed)
        await self.close(driver)

    async def test_running_bootstrap_reports_concurrent_close_and_timeout(self):
        for mode, command, expected in (("--prepared-publication", "activate-close", 10),
                                        ("--prepared-timeout", "activate-timeout", 9)):
            driver = await self.start(mode=mode)
            reply = await driver.command(command)
            self.assertEqual(reply["result"], expected)
            closed = await self.close(driver)
            self.assertEqual(closed["published"], 1)  # Completed host mutation is not claimed rolled back.
            self.assertEqual(closed["dispatches"], 0)
            if expected == 9:
                self.assertEqual(reply["publication"], dict(
                    configured=True, has_attempt=True, has_completed=True,
                    last_attempted_revision=1, last_completed_revision=1, result=9))
                self.assertEqual(reply["actual_host_fields"], self.oracle().state)
                self.assertEqual(closed["result"], 9)
                self.assertEqual((closed["applied"], closed["retained"], closed["journal_applied"]), (0, 1, 1))
            self.trace["states"] += [reply, closed]

    async def test_seeded_many_writers_preserve_every_admission_and_exact_scene(self):
        rng = random.Random(self.seed)
        drivers = await asyncio.gather(*(self.start() for _ in range(4)))
        schedules = []
        for writer in range(4):
            operations = [(f"writer_{writer}", writer)]
            for _ in range(31):
                key = f"shared_{rng.randrange(7)}"
                operations.append((key, None if rng.randrange(5) == 0 else rng.randrange(-10000, 10001)))
            schedules.append(operations)
        async def author(writer):
            ids = []
            for key, value in schedules[writer]:
                reply = await drivers[writer].command(f"erase {key}" if value is None else f"set {key} {value}")
                self.assertEqual(reply["type"], "accepted", reply)
                self.assertEqual(reply["result"], 0)
                ids.append(reply["id"])
                self.trace["admissions"].append(dict(writer=writer, id=reply["id"], key=key, value=value))
            self.assertEqual(len(set(ids)), len(ids))
        await asyncio.gather(*(author(writer) for writer in range(4)))
        await asyncio.gather(*(self.settled(driver, 129) for driver in drivers))
        groups = {}
        for commit in self.commits()[1:]:
            groups.setdefault(commit.writer, []).append(decode_update(commit.payload))
        self.assertEqual(len(groups), 4)
        self.assertEqual(len(self.commits()), 129)
        seen_writers = set()
        for records in groups.values():
            first = next(iter(records[0]))
            self.assertTrue(first.startswith("writer_"))
            writer = int(first.removeprefix("writer_"))
            self.assertIn(writer, range(4))
            self.assertNotIn(writer, seen_writers)
            seen_writers.add(writer)
            self.assertEqual(records, [{key: value} for key, value in schedules[writer]])
        self.assertEqual(seen_writers, set(range(4)))
        self.assertEqual(self.oracle().edits, 128)
        await asyncio.gather(*(self.close(driver) for driver in drivers))

    async def test_uncertain_commit_preserves_visible_and_all_pending_intents(self):
        self.state.advertise_capabilities = False  # One unchained write at a time, never re-sent.
        driver = await self.start()
        fault = self.state.plan("UpdateLayer", "after_commit", pause=True, code=grpc.StatusCode.UNAVAILABLE)
        replies = [await driver.command("set first 1")]
        await asyncio.wait_for(fault.reached.wait(), 5)
        replies += [await driver.command("set second 2"), await driver.command("set third 3")]
        self.assertTrue(all(reply["type"] == "accepted" for reply in replies))
        fault.release.set()
        result = await driver.command("failed")
        self.assertEqual(result["result"], 19, result)
        self.assertEqual(result["applied"], 1)
        self.assertEqual(result["authoritative"], {})
        self.assertEqual(result["visible"], {"first": 1, "second": 2, "third": 3})
        self.assertEqual(result["pending_ids"], [reply["id"] for reply in replies])
        self.assertEqual(result["retained"], 3)
        rejection = await driver.command("set fourth 4")
        self.assertEqual((rejection["type"], rejection["result"]), ("rejected", 19))
        closed = await self.close(driver)
        self.assertEqual(closed["pending_ids"], result["pending_ids"])
        self.assertEqual(closed["visible"], result["visible"])
        self.assertEqual(self.state.attempts["UpdateLayer"], 1)
        self.assertEqual([decode_update(c.payload) for c in self.commits()[1:]], [{"first": 1}])
        self.trace["states"].append(result)

    async def proxy_driver(self):
        other, stub = await self.backend("B")
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        address = await proxy.start()
        return proxy, other, stub, await self.start(address)

    async def test_caught_up_sequence_below_the_client_cursor_fails_the_layer(self):
        other, _ = await self.backend("B")
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        driver = await self.start(await proxy.start(), mode="--file-format")
        await self.peer({"a": 1})
        sequence = await self.peer({"b": 2})
        held = await driver.command(f"wait {sequence}")
        self.assertEqual(held["fields"], {"a": 1, "b": 2})
        reads, writes = self.state.attempts["SubscribeReadLayer"], self.state.attempts.get("UpdateLayer", 0)
        # The service loses its newest commits: its caught-up header is below the client cursor.
        del self.commits()[1:]
        recovery = self.state.plan("SubscribeReadLayer", "before_reply", pause=True)
        proxy.rotate(other)
        await asyncio.wait_for(recovery.reached.wait(), 5)
        # The resume is held at the service, so this edit queues behind the recovery gate.
        admitted = await driver.command("set c 3")
        self.assertEqual(admitted["type"], "accepted", admitted)
        recovery.release.set()
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 11, failed)  # MALFORMED_STREAM: server sequence regression
        self.assertEqual((failed["applied"], failed["fields"]), (sequence, {"a": 1, "b": 2, "c": 3}))
        self.assertEqual(failed["pending_ids"], [admitted["id"]])
        status = await driver.command("status")
        self.assertEqual((status["state"], status["status_result"], status["operation"]), (7, 11, 3), status)
        report = await driver.command("diagnostics 11")
        records = [r for r in report["diagnostics"]["recent_records"] if r["event_name"] == "stream_disconnected"
                   and r["subject_id"] == report["subject"] and r["fields"]["result"] == 11]
        self.assertEqual(len(records), 1, records)
        self.assertEqual((records[0]["fields"]["read_sequence"], records[0]["fields"]["resume_sequence"]),
                         (sequence, sequence), records)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], reads + 1)
        self.assertEqual(self.state.attempts.get("UpdateLayer", 0), writes)
        closed = await self.close(driver)
        self.assertEqual(closed["pending_ids"], [admitted["id"]])
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], reads + 1)
        self.trace["states"] += [held, failed, status, closed]

    async def test_backend_rotation_recovers_complete_aggregate_before_new_writes(self):
        proxy, other, stub, driver = await self.proxy_driver()
        recovery = self.state.plan("SubscribeReadLayer", "before_reply", pause=True)
        proxy.rotate(other, pause=True)
        await self.peer({"a": 1, "b": 2}, stub=stub)
        await self.peer({"a": 7}, stub=stub)
        await self.peer({"b": None, "c": 9}, stub=stub)
        write = self.state.plan("UpdateLayer", "before", pause=True)
        proxy.gate.set()
        await asyncio.wait_for(recovery.reached.wait(), 5)
        result = await driver.command("state")
        self.assertEqual(result["gate"], 0, result)  # Recovering; aggregate is held at service.
        self.assertEqual(result["applied"], 1)
        admitted = await driver.command("set after 8")
        self.assertEqual(admitted["type"], "accepted")
        queued = await driver.command("state")
        self.assertEqual(queued["pending_ids"], [admitted["id"]])
        self.assertEqual(queued["authoritative"], {})
        self.assertEqual(queued["visible"], {"after": 8})
        # The actor processed the admission's wake and went idle while the
        # recovery read is held, so it issued no UpdateLayer.
        idle = await driver.command("idle")
        self.assertEqual((idle["gate"], idle["issued"], idle["queued"]), (0, 0, 1), idle)
        self.assertFalse(write.reached.is_set())
        self.assertEqual(self.state.attempts["UpdateLayer"], 3)
        recovery.release.set()
        await asyncio.wait_for(write.reached.wait(), 5)
        rebased = await driver.command("state")
        self.assertEqual(rebased["applied"], 4)
        self.assertEqual(rebased["authoritative"], {"a": 7, "c": 9})
        self.assertEqual(rebased["visible"], {"a": 7, "c": 9, "after": 8})
        self.assertEqual(rebased["pending_ids"], [admitted["id"]])
        self.assertGreaterEqual(rebased["generation"], 2)
        self.assertTrue(any(r["backend"] == "B" and r["cursor"] == 1 for r in self.state.scene_reads))
        write.release.set()
        await self.settled(driver, 5)
        self.assertEqual(self.state.attempts["UpdateLayer"], 4)
        self.assertEqual([decode_update(c.payload) for c in self.commits()[4:]], [{"after": 8}])
        self.trace["states"].extend([queued, rebased])
        await self.close(driver)

    async def test_quiet_backend_rotation_accepts_caught_up_without_reapplying(self):
        proxy, other, _, driver = await self.proxy_driver()
        proxy.rotate(other)
        result = await driver.command("generation 2")
        self.assertGreaterEqual(result["generation"], 2)
        self.assertEqual(result["applied"], 1)
        self.assertEqual(result["authoritative"], {})
        self.assertEqual((await driver.command("set resumed 5"))["type"], "accepted")
        await self.settled(driver, 2)
        await self.close(driver)

    async def test_stalled_reconnect_first_responses_spend_the_reconnect_budget(self):
        proxy, other, _, driver = await self.proxy_driver()
        await self.peer({"a": 1})
        await self.settled(driver, 2)
        reads, writes = self.state.attempts["SubscribeReadLayer"], self.state.attempts["UpdateLayer"]
        # The service accepts every reconnect and never answers it; the driver allows three attempts.
        stalls = [self.state.plan("SubscribeReadLayer", "before_reply", attempt=reads + n, pause=True)
                  for n in (1, 2, 3)]
        for stall in stalls:
            self.addCleanup(stall.release.set)
        proxy.rotate(other)
        await asyncio.wait_for(stalls[0].reached.wait(), 5)
        admitted = await driver.command("set local 3")
        self.assertEqual(admitted["type"], "accepted", admitted)
        idle = await driver.command("idle")
        self.assertEqual((idle["gate"], idle["issued"], idle["queued"]), (0, 0, 1), idle)
        for stall in stalls:
            await asyncio.wait_for(stall.reached.wait(), 5)
            self.assertEqual(await driver.command("expire-read"), {"type": "read-expired"})
        failed = await driver.command("failed")
        self.assertEqual(failed["result"], 8, failed)  # UNAVAILABLE after reconnect_max_attempts attempts
        self.assertEqual((failed["applied"], failed["authoritative"]), (2, {"a": 1}), failed)
        self.assertEqual(failed["visible"], {"a": 1, "local": 3}, failed)
        self.assertEqual(failed["pending_ids"], [admitted["id"]], failed)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], reads + 3)
        self.assertEqual(self.state.attempts["UpdateLayer"], writes)
        for stall in stalls:
            stall.release.set()
        closed = await self.close(driver)
        self.assertEqual(closed["pending_ids"], [admitted["id"]])
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], reads + 3)
        self.trace["states"] += [failed, closed]

    async def test_stalled_reconnect_first_response_retries_and_a_caught_up_read_keeps_no_deadline(self):
        proxy, other, _, driver = await self.proxy_driver()
        reads = self.state.attempts["SubscribeReadLayer"]
        # A read past its first complete response has no deadline, however long it stays open.
        self.assertEqual(await driver.command("expire-read"), {"type": "read-expired"})
        self.assertEqual((await driver.command("idle"))["gate"], 1)
        await self.peer({"a": 1})
        await self.settled(driver, 2)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], reads)
        stall = self.state.plan("SubscribeReadLayer", "before_reply", attempt=reads + 1, pause=True)
        self.addCleanup(stall.release.set)
        proxy.rotate(other)
        await asyncio.wait_for(stall.reached.wait(), 5)
        admitted = await driver.command("set local 3")
        self.assertEqual(admitted["type"], "accepted", admitted)
        self.assertEqual(await driver.command("expire-read"), {"type": "read-expired"})
        # The next attempt is answered: the layer resumes and issues its queued edit.
        await self.settled(driver, 3)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], reads + 2)
        self.assertEqual(decode_update(self.commits()[-1].payload), {"local": 3})
        report = await driver.command("state")
        self.assertEqual(report["gate"], 1, report)
        # The resumed read is caught up and keeps no deadline either.
        self.assertEqual(await driver.command("expire-read"), {"type": "read-expired"})
        self.assertEqual((await driver.command("idle"))["gate"], 1)
        await self.peer({"b": 2})
        await self.settled(driver, 4)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], reads + 2)
        stall.release.set()
        await self.close(driver)

    async def test_reconnect_aggregate_retires_the_local_ack_it_covers(self):
        proxy, other, _, driver = await self.proxy_driver()
        reached, release = asyncio.Event(), asyncio.Event()
        self.state.live_scene_hold = (2, reached, release)
        await self.peer({"missing_peer": 11})
        await asyncio.wait_for(reached.wait(), 5)
        accepted = await driver.command("set local 12")
        self.assertEqual(accepted["type"], "accepted")
        before = await driver.command(f"ack {accepted['id']}")
        self.assertEqual(before["result"], 0)
        self.assertEqual((before["applied"], before["covered"]), (1, 1))
        self.assertEqual(before["pending_ids"], [accepted["id"]])
        proxy.rotate(other)
        release.set()
        result = await driver.command("wait 3")
        # Aggregate [2,3] is the service's state at 3, which contains the local
        # write acknowledged at 3: applying it retires the intent.
        self.assertEqual((result["result"], result["gate"]), (0, 1), result)
        self.assertEqual((result["applied"], result["covered"]), (3, 3))
        self.assertEqual(result["authoritative"], {"missing_peer": 11, "local": 12})
        self.assertEqual(result["visible"], {"missing_peer": 11, "local": 12})
        self.assertEqual(result["pending_ids"], [])
        self.assertEqual(result["retained"], 0, result)  # No coverage stays stranded.
        self.assertTrue(any(r["backend"] == "B" and r["cursor"] == 1 for r in self.state.scene_reads))
        closed = await self.close(driver)
        self.assertEqual(closed["pending_ids"], [])
        self.assertEqual(self.state.attempts["UpdateLayer"], 2)
        self.assertEqual([decode_update(c.payload) for c in self.commits()[1:]],
                         [{"missing_peer": 11}, {"local": 12}])
        self.trace["states"].extend([before, result])

    async def test_live_aggregate_before_the_reply_absorbs_the_local_write(self):
        # A coalescing service delivers peer 2 and the layer's own write at 3 as one live aggregate before the
        # write's reply. The aggregate waits for the reply, which lands inside it: the write is absorbed and
        # retires when the aggregate applies.
        driver = await self.start()
        self.state.live_scene_aggregate = 3
        await self.peer({"missing_peer": 11})
        reply = self.state.plan_write("hold_reply", position=1)  # The layer's first write; the peer wrote already.
        accepted = await driver.command("set local 12")
        self.assertEqual(accepted["type"], "accepted")
        await asyncio.wait_for(reply.reached.wait(), 5)
        # Failure-only watchdog: the driver accepts the aggregate while the reply is held.
        deadline = asyncio.get_running_loop().time() + 5
        while (held := await driver.command("state"))["covered"] != 3:
            self.assertLess(asyncio.get_running_loop().time(), deadline, held)
        self.assertIsNone(self.state.live_scene_aggregate)
        self.assertEqual((held["result"], held["gate"], held["applied"]), (0, 1, 1), held)
        self.assertEqual(held["pending_ids"], [accepted["id"]])
        self.assertEqual(held["visible"], {"local": 12})
        reply.release.set()
        result = await self.settled(driver, 3)
        self.assertEqual(result["authoritative"], {"missing_peer": 11, "local": 12})
        self.assertEqual(self.state.attempts["UpdateLayer"], 2)
        self.assertEqual([decode_update(c.payload) for c in self.commits()[1:]],
                         [{"missing_peer": 11}, {"local": 12}])
        self.trace["states"].extend([held, result])
        await self.close(driver)

    async def test_per_commit_catch_up_replays_the_resent_local_write(self):
        proxy, other, _, driver = await self.proxy_driver()
        reached, release = asyncio.Event(), asyncio.Event()
        self.state.live_scene_hold = (2, reached, release)
        await self.peer({"missing_peer": 11})
        await asyncio.wait_for(reached.wait(), 5)
        accepted = await driver.command("set local 12")
        self.assertEqual(accepted["type"], "accepted")
        before = await driver.command(f"ack {accepted['id']}")
        self.assertEqual(before["result"], 0)
        self.assertEqual((before["applied"], before["covered"]), (1, 1))
        self.assertEqual(before["pending_ids"], [accepted["id"]])
        self.state.per_commit_catch_up = True
        proxy.rotate(other)
        release.set()
        result = await driver.command("wait 3")
        # The per-commit catch-up re-sends peer 2 and the local write at 3 with
        # the issued bytes: the re-sent write is a replay, and the intent
        # retires through its own confirmed application.
        self.assertEqual((result["result"], result["gate"]), (0, 1), result)
        self.assertEqual((result["applied"], result["covered"]), (3, 3))
        self.assertEqual(result["authoritative"], {"missing_peer": 11, "local": 12})
        self.assertEqual(result["visible"], {"missing_peer": 11, "local": 12})
        self.assertEqual(result["pending_ids"], [])
        self.assertEqual(result["retained"], 0, result)  # No coverage stays stranded.
        self.assertTrue(any(r["backend"] == "B" and r["cursor"] == 1 for r in self.state.scene_reads))
        reads = [r for r in self.state.scene_reads if r["backend"] == "B"]
        self.assertEqual([r["cursor"] for r in reads], [1])
        closed = await self.close(driver)
        self.assertEqual(closed["pending_ids"], [])
        self.assertEqual(self.state.attempts["UpdateLayer"], 2)
        self.assertEqual([decode_update(c.payload) for c in self.commits()[1:]],
                         [{"missing_peer": 11}, {"local": 12}])
        self.trace["states"].extend([before, result])

    async def test_partial_first_response_is_discarded_and_full_retry_applies_once(self):
        await self.peer({"from_history": 23})
        self.state.partial_scene_reads.add(1)
        driver = await self.start()
        result = await self.settled(driver, 2)
        self.assertGreaterEqual(result["generation"], 2)
        self.assertEqual([r["cursor"] for r in self.state.scene_reads], [0, 0])
        self.assertEqual(self.state.partial_scene_fired, [1])
        await self.close(driver)

    async def test_malformed_peer_preserves_published_scene(self):
        driver = await self.start()
        await self.peer({"retained": 11})
        await self.settled(driver, 2)
        await self.peer(payload=b"invalid OMPF")
        result = await driver.command("failed")
        self.assertEqual(result["result"], 11)
        self.assertEqual((result["applied"], result["authoritative"], result["visible"]),
                         (2, {"retained": 11}, {"retained": 11}))
        await self.close(driver)

    async def test_reset_after_bootstrap_requires_recovery_without_replacing_scene(self):
        driver = await self.start()
        await self.peer({"retained": 11})
        await self.settled(driver, 2)
        # The local write waits before its commit, so the reset takes position 3 while the write is pending.
        hold = self.state.plan("UpdateLayer", "before", pause=True)
        self.addCleanup(hold.release.set)
        admitted = await driver.command("set local 17")
        self.assertEqual(admitted["type"], "accepted", admitted)
        await asyncio.wait_for(hold.reached.wait(), 5)
        self.assertEqual(await self.peer(payload=snapshot({"replacement": 99})), 3)
        result = await driver.command("failed")
        self.assertEqual(result["result"], 11, result)  # MALFORMED_STREAM: a reset snapshot is bootstrap-only.
        self.assertEqual((result["applied"], result["authoritative"]), (2, {"retained": 11}), result)
        self.assertEqual(result["visible"], {"retained": 11, "local": 17}, result)
        self.assertEqual(result["pending_ids"], [admitted["id"]], result)
        hold.release.set()
        await self.close(driver)


    async def start_discovery(self, *, auth=True, flag=None):
        process = await asyncio.create_subprocess_exec(os.environ["LIVE_TEST_SCENE_DRIVER"],
            self.address, self.session, str(self.seed), flag or ("--discovery" if auth else "--discovery-no-auth"),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        driver = Driver(process, self.artifact / f"{self.id().split('.')[-1]}-{len(self.drivers)}.scene-native.log")
        self.drivers.append(driver)
        self.assertEqual(await driver.receive(), {"type": "discovery-ready"})
        return driver

    async def exit_discovery(self, driver):
        self.assertEqual(await driver.command("exit"), {"type": "exited"})
        await asyncio.wait_for(driver.process.wait(), 10)
        await driver.drainer
        self.assertIsNone(driver.failure)
        self.assertEqual(driver.process.returncode, 0, driver.log.read_text()[-8192:])

    def check_discovered(self, value, object_id):
        self.assertEqual(value["type"], "opened", value)
        self.assertEqual(value["result"], 0, value)
        self.assertEqual(value["object"], object_id, value)
        self.assertGreater(value["applied"], 0, value)
        self.assertEqual(value["applied"], value["covered"], value)
        self.assertEqual(value["authoritative"], self.oracle().state, value)
        self.assertEqual(value["visible"], self.oracle().state, value)

    async def test_scene_discovery_canonical_claim_and_closed_owner_lifetime(self):
        # A non-default incarnation prevents a hard-coded object1 open from
        # accidentally satisfying this discovery test.
        await self.stub.DeleteLayer(pb.DeleteLayerRequest(session_id=self.session, layer_name="root.live"),
                                    metadata=WRITER, timeout=5)
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="root.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=snapshot({"existing": 41})))]),
            metadata=WRITER, timeout=5)
        object_id = self.state.sessions[self.session].layers["root.live"].object_id
        self.assertNotEqual(object_id, 1)
        pause = self.state.plan("SubscribeReadLayer", "before_reply", pause=True)
        driver = await self.start_discovery()
        self.assertEqual(await driver.command("begin-open"), {"type": "opening"})
        await asyncio.wait_for(pause.reached.wait(), 5)
        self.assertEqual(await driver.command("duplicate"), {"type": "rejected", "result": 5})
        self.assertEqual(self.state.attempts.get("StatLayer", 0), 0)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        pause.release.set()
        self.check_discovered(await driver.command("finish-open"), object_id)
        self.assertEqual(self.state.scene_reads[0]["object"], 0)
        self.assertEqual(self.state.scene_reads[0]["cursor"], 0)
        admitted = await driver.command("set local 17")
        self.assertEqual(admitted["type"], "accepted")
        settled = await driver.command("wait 2")
        self.assertEqual(settled["result"], 0)
        self.assertEqual(settled["authoritative"], {"existing": 41, "local": 17})
        self.assertEqual(self.commits()[-1].object_id, object_id)
        self.assertEqual(decode_update(self.commits()[-1].payload), {"local": 17})
        self.assertEqual((await driver.command("close-held"))["type"], "closed-held")
        self.assertEqual(await driver.command("duplicate"), {"type": "rejected", "result": 5})
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        self.assertEqual(await driver.command("release"), {"type": "released"})
        self.check_discovered(await driver.command("open"), object_id)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)
        self.assertEqual(self.state.scene_reads[-1]["object"], 0)
        self.assertEqual(self.state.attempts.get("StatLayer", 0), 0)
        await self.exit_discovery(driver)

    async def test_scene_failed_discovery_releases_claim_after_read_cleanup(self):
        failure = self.state.plan("SubscribeReadLayer", "before_reply", pause=True,
                                  code=grpc.StatusCode.PERMISSION_DENIED)
        driver = await self.start_discovery()
        self.assertEqual(await driver.command("begin-open"), {"type": "opening"})
        await asyncio.wait_for(failure.reached.wait(), 5)
        self.assertEqual(await driver.command("duplicate"), {"type": "rejected", "result": 5})
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        failure.release.set()
        self.assertEqual(await driver.command("finish-open"), {"type": "rejected", "result": 7})
        self.check_discovered(await driver.command("open"), 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)
        await self.exit_discovery(driver)

    async def test_scene_failed_discovery_immediate_retry_waits_for_admission_retirement(self):
        # The retrying open waits on the failed backing's held retirement gate.
        # A retiring admission pinned to its host refuses at once.
        failure = self.state.plan("SubscribeReadLayer", "before_reply", pause=True,
                                  code=grpc.StatusCode.PERMISSION_DENIED)
        driver = await self.start_discovery()
        self.assertEqual(await driver.command("hold-failed-retirement"), {"type": "retirement-held"})
        self.assertEqual(await driver.command("begin-open"), {"type": "opening"})
        await asyncio.wait_for(failure.reached.wait(), 5)
        failure.release.set()
        self.assertEqual(await driver.command("finish-open-immediate"), {"type": "rejected", "result": 7})
        self.check_discovered(await driver.command("retry-after-failure"), 1)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)
        self.assertEqual(await driver.command("release"), {"type": "released"})
        self.assertEqual(await driver.command("pinned-refusal"), {"type": "pinned-refusal", "refused": True})
        await self.exit_discovery(driver)

    async def test_scene_missing_discovery_never_publishes_empty_asset(self):
        await self.stub.DeleteLayer(pb.DeleteLayerRequest(session_id=self.session, layer_name="root.live"),
                                    metadata=WRITER, timeout=5)
        driver = await self.start_discovery()
        self.assertEqual(await driver.command("open"), {"type": "rejected", "result": 4})
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 1)
        self.assertEqual(self.state.attempts.get("StatLayer", 0), 0)
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="root.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=snapshot({"restored": 9})))]),
            metadata=WRITER, timeout=5)
        self.check_discovered(await driver.command("open"), 2)
        self.assertEqual(self.state.attempts["SubscribeReadLayer"], 2)
        await self.exit_discovery(driver)

    async def test_scene_missing_auth_discovery_never_publishes_empty_asset(self):
        driver = await self.start_discovery(auth=False)
        self.assertEqual(await driver.command("open"), {"type": "rejected", "result": 6})
        self.assertEqual(self.state.attempts.get("SubscribeReadLayer", 0), 0)
        self.assertEqual(self.state.attempts.get("StatLayer", 0), 0)
        await self.exit_discovery(driver)

    async def test_scene_open_deadline_while_starved_is_unauthenticated(self):
        # The 1 ms initial-open deadline precedes the 30000 ms auth wait; the
        # layer actor records credential starvation when it is constructed.
        driver = await self.start_discovery(flag="--discovery-auth-deadline")
        self.assertEqual(await driver.command("open"), {"type": "rejected", "result": 6})
        # The deferred open used by the USD file format classifies the same way.
        self.assertEqual(await driver.command("prepare"), {"type": "rejected", "result": 6})
        self.assertEqual(self.state.attempts.get("SubscribeReadLayer", 0), 0)
        self.assertEqual(self.state.attempts.get("StatLayer", 0), 0)
        await self.exit_discovery(driver)


class SceneInfrastructureTests(unittest.IsolatedAsyncioTestCase):
    async def test_proxy_cut_retires_an_accepted_socket_before_forward_starts(self):
        class Writer:
            def __init__(self):
                self.transport, self.aborted = self, False
            def abort(self):
                self.aborted = True
        proxy, writer = RotatingProxy('127.0.0.1:1'), Writer()
        with patch('asyncio.open_connection', side_effect=AssertionError('cancelled forward started')) as connect:
            async def cut_before_yield():
                proxy.accept(None, writer)
                return await proxy.rotate_and_wait('127.0.0.1:2', pause=True)
            self.assertEqual(await asyncio.wait_for(cut_before_yield(), 5), 1)
            connect.assert_not_called()
        self.assertTrue(writer.aborted)
        self.assertFalse(proxy.tasks)
        self.assertFalse(proxy.writers)

    async def test_proxy_cut_retires_pending_old_connect_without_cancelling_new_accepts(self):
        class Writer:
            def __init__(self):
                self.transport, self.aborted = self, False
            def abort(self):
                self.aborted = True
        entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        async def connecting(*args):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()
                await release.wait()
        proxy, old, new = RotatingProxy('127.0.0.1:1'), Writer(), Writer()
        with patch('asyncio.open_connection', side_effect=connecting):
            proxy.accept(None, old)
            await asyncio.wait_for(entered.wait(), 5)
            cut = asyncio.create_task(proxy.rotate_and_wait('127.0.0.1:2', pause=True))
            try:
                await asyncio.wait_for(cancelled.wait(), 5)
                self.assertFalse(cut.done(), 'cut returned before old connect retired')
                proxy.accept(None, new)
                release.set()
                self.assertEqual(await asyncio.wait_for(cut, 5), 1)
                self.assertTrue(old.aborted)
                self.assertFalse(new.aborted)
                self.assertEqual(len(proxy.tasks), 1)
                self.assertEqual(proxy.writers, {new})
            finally:
                release.set()
                await asyncio.gather(cut, return_exceptions=True)
                await asyncio.wait_for(proxy.rotate_and_wait('127.0.0.1:2', pause=True), 5)
        self.assertTrue(new.aborted)
        self.assertFalse(proxy.tasks)
        self.assertFalse(proxy.writers)

    async def test_cleanup_attempts_all_resources_and_exports_after_close_failure(self):
        calls = []
        class Process:
            returncode = None
            def kill(self):
                calls.append("kill")
                raise ProcessLookupError("already exited")
            async def wait(self):
                calls.append("wait")
        class Proxy:
            async def close(self):
                calls.append("proxy")
                raise RuntimeError("injected proxy cleanup failure")
        class Channel:
            async def close(self):
                calls.append("channel")
        class Server:
            async def stop(self):
                calls.append("server")
        class FixtureState:
            scene_reads, attempts, partial_scene_fired, partial_scene_reads = [], {}, [], set()
            def snapshot(self):
                calls.append("ledger")
                return {"retained": True}
            def assert_faults_fired(self):
                calls.append("faults")
        class Child:
            process = Process()
        child = Child()
        child.drainer = asyncio.get_running_loop().create_future()
        child.drainer.set_result(None)
        case = SceneTests("test_compact_bootstrap_contains_all_existing_history")
        case.drivers, case.proxies, case.channels, case.servers = [child], [Proxy()], [Channel()], [Server()]
        case.state, case.trace = FixtureState(), {"seed": 42}
        with tempfile.TemporaryDirectory(dir=os.environ["LIVE_TEST_ARTIFACT_DIR"]) as directory:
            case.artifact = Path(directory)
            with self.assertRaisesRegex(AssertionError, "injected proxy cleanup failure"):
                await case.cleanup()
            self.assertEqual(calls, ["kill", "wait", "proxy", "channel", "server", "faults", "ledger"])
            artifact = json.loads(next(case.artifact.glob("*.scene.json")).read_text())
            self.assertEqual(artifact["ledger"], {"retained": True})
            self.assertIn("injected proxy cleanup failure", artifact["cleanup_errors"][0])
