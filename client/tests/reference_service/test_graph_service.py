# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire tests of the explicit reference graph policy, not deployed semantics."""
import asyncio
import hashlib
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import grpc

from graph_service import ErrorInfo, GraphState, GraphServer, RetryInfo, RpcStatus
from service import Principal, pb, rpc
from scene_graph_model import Model, ranked
from scene_graph_wire import Command, Field, Delta, Token, Paths, PathOp, Sample, TimeSamples, typed_value, encode, decode

AUTH = (("authorization", "Bearer graph-test-token"),)


def prim(identity, name, parent=1):
    return Command(1, identity, parent=parent, name=name, spec_type=6)


def ordinary(*commands, fields=()):
    return encode(Delta(commands=tuple(commands), fields=tuple(fields)))


class GraphServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.state = GraphState(int(os.environ.get("LIVE_TEST_SEED", "42")))
        self.state.tokens["graph-test-token"] = Principal("writer")
        self.servers, self.channels, self.calls = [], [], []
        self.addAsyncCleanup(self.cleanup)
        self.stub = await self.backend("A")
        self.session = (await self.stub.CreateSession(pb.CreateSessionRequest(path="/graph-test", name="graph"),
                                                     metadata=AUTH, timeout=5)).session_id

    async def backend(self, name):
        server = GraphServer(self.state, name)
        self.servers.append(server)
        address = await server.start()
        channel = grpc.aio.insecure_channel(address)
        self.channels.append(channel)
        await asyncio.wait_for(channel.channel_ready(), 5)
        return rpc.LiveServiceStub(channel)

    async def cleanup(self):
        for fault in self.state.faults:
            fault.release.set()
        for call in self.calls:
            call.cancel()
        failures = []
        for group in ([channel.close() for channel in self.channels], [server.stop() for server in self.servers]):
            try:
                failures.extend(await asyncio.wait_for(asyncio.gather(*group, return_exceptions=True), 5))
            except BaseException as error:
                failures.append(error)
        artifact = os.environ.get("LIVE_TEST_ARTIFACT_DIR")
        if artifact:
            Path(artifact, self.id().split(".")[-1] + ".graph-ledger.json").write_text(
                json.dumps(self.state.snapshot(), indent=2, sort_keys=True) + "\n")
        self.state.assert_faults_fired()
        for failure in failures:
            if isinstance(failure, BaseException):
                raise failure

    async def create(self, payload=None, *, operation=""):
        return (await self.create_reply(payload, operation=operation)).object_id

    async def create_reply(self, payload=None, *, operation=""):
        payload = ordinary() if payload is None else payload
        frames = [pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="root.live",
                  operation_id=operation)), pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=payload))]
        return await self.stub.CreateLayer(iter(frames), metadata=AUTH, timeout=5)

    async def update(self, object_id, payload, *, writer="writer", stub=None, operation=""):
        frames = [pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name="root.live",
                  object_id=object_id, client_id=writer, timestamp_microseconds_since_epoch=123,
                  operation_id=operation)), pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=payload))]
        return await (stub or self.stub).UpdateLayer(iter(frames), metadata=AUTH, timeout=5)

    def subscribe(self, object_id=0, cursor=0, *, writer="", stub=None):
        call = (stub or self.stub).SubscribeReadLayer(pb.SubscribeReadLayerRequest(session_id=self.session,
            layer_name="root.live", object_id=object_id, sequence_num=cursor, client_id=writer),
            metadata=AUTH, timeout=10)
        self.calls.append(call)
        return call

    async def content(self, call):
        frame = await asyncio.wait_for(call.read(), 3)
        self.assertEqual(frame.WhichOneof("response_type"), "header")
        header, payload = frame.header, bytearray()
        while len(payload) < header.total_size:
            frame = await asyncio.wait_for(call.read(), 3)
            self.assertEqual(frame.WhichOneof("response_type"), "chunk")
            payload.extend(frame.chunk.data)
        self.assertEqual(len(payload), header.total_size)
        return header, bytes(payload)

    async def caught_up(self, call, object_id, sequence):
        frame = await asyncio.wait_for(call.read(), 3)
        self.assertEqual(frame.WhichOneof("response_type"), "header")
        self.assertEqual((frame.header.object_id, frame.header.sequence_num, frame.header.total_size),
                         (object_id, sequence, 0))

    def layer(self):
        return self.state.sessions[self.session].layers["root.live"]

    def ledger_replay(self, object_id, through):
        """Replay the untrimmed test ledger; the layer keeps only its retained window."""
        model = Model()
        for entry in self.state.ledger:
            if entry.layer == "root.live" and entry.object_id == object_id and entry.sequence <= through:
                model.trim_history(0)
                model.apply(entry.payload, entry.sequence)
        return model

    def checkpoint(self):
        return (self.layer(), tuple(self.state.ledger), self.state.retained_bytes,
                dict(self.state.incarnations), self.layer().model.state, self.layer().model.history,
                self.layer().model.source_format)

    def unchanged(self, before):
        self.assertIs(self.layer(), before[0])
        self.assertEqual(self.checkpoint()[1:], before[1:])

    async def rejects(self, awaitable, code):
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            await awaitable
        self.assertEqual(caught.exception.code(), code)

    def replay(self, through=None):
        model = Model()
        for entry in self.layer().commits:
            if through is None or entry.sequence <= through:
                model.apply(entry.payload, entry.sequence)
        return model

    async def test_operation_id_resend_returns_the_original_commit_without_applying_twice(self):
        self.state.plan("CreateLayer", "after_commit", code=grpc.StatusCode.UNAVAILABLE)
        with self.assertRaises(grpc.aio.AioRpcError):
            await self.create(ordinary(prim(2, "A")), operation="create-1")
        created = await self.create_reply(ordinary(prim(2, "A")), operation="create-1")
        self.assertEqual((created.object_id, created.operation_outcome), (1, pb.OPERATION_OUTCOME_ALREADY_COMMITTED))
        self.state.plan("UpdateLayer", "after_commit", code=grpc.StatusCode.UNAVAILABLE)
        edit = ordinary(prim(3, "B"))
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            await self.update(1, edit, operation="op-1")
        self.assertEqual(caught.exception.code(), grpc.StatusCode.UNAVAILABLE)
        # Applying the Create again would be a duplicate identity; the re-send matches the commit instead.
        again = await self.update(1, edit, operation="op-1")
        self.assertEqual((again.sequence_num, again.operation_outcome), (2, pb.OPERATION_OUTCOME_ALREADY_COMMITTED))
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            await self.update(1, ordinary(prim(4, "C")), operation="op-1")
        self.assertEqual(caught.exception.code(), grpc.StatusCode.ALREADY_EXISTS)
        self.assertEqual([c.sequence for c in self.layer().commits], [1, 2])
        self.assertEqual(sorted(node["name"] for node in self.layer().model.state if node["id"] != 1), ["A", "B"])

    def entry_limit_info(self, error):
        """The google.rpc.ErrorInfo a rejection's grpc-status-details-bin carries."""
        details = [value for key, value in error.trailing_metadata() if key == "grpc-status-details-bin"]
        self.assertEqual(len(details), 1, error.trailing_metadata())
        status = RpcStatus.FromString(details[0])
        self.assertEqual(status.code, grpc.StatusCode.INVALID_ARGUMENT.value[0])
        self.assertEqual([detail.type_url for detail in status.details], ["type.googleapis.com/google.rpc.ErrorInfo"])
        info = ErrorInfo.FromString(status.details[0].value)
        self.assertEqual(info.domain, "nvidia.omniverse.live.v1alpha")
        return info.reason, dict(info.metadata)

    async def test_delta_over_the_entry_limit_is_invalid_argument_with_error_info(self):
        self.state.delta_entry_limit = 3
        # Struct commands, set fields and time samples count together: three
        # entries create the layer, and four are refused before anything commits.
        identity = await self.create(ordinary(prim(2, "A"), Command(1, 3, parent=2, name="v", spec_type=1),
                                              fields=(Field(2, "counter", 1),)))
        before = self.checkpoint()
        sample = Delta(commands=(prim(4, "B"),), fields=(Field(4, "counter", 1),), samples=(Sample(3, 1.0, 5),))
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            await self.update(identity, encode(replace(sample, fields=sample.fields + (Field(1, "y", 3),))))
        self.assertEqual(caught.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(self.entry_limit_info(caught.exception), ("DELTA_TOO_LARGE", dict(
            struct_commands="1", set_fields="2", time_samples="1", max_entries="3")))
        self.unchanged(before)
        self.assertEqual(self.state.rejections[-1]["kind"], "delta_too_large")
        # Exactly at the limit, the update commits.
        self.assertEqual((await self.update(identity, encode(sample))).sequence_num, 2)
        # CreateLayer is held to the same limit and creates nothing.
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            await self.create(ordinary(prim(2, "A"), prim(3, "B"), prim(4, "C"), prim(5, "D")))
        self.assertEqual(self.entry_limit_info(caught.exception), ("DELTA_TOO_LARGE", dict(
            struct_commands="4", set_fields="0", time_samples="0", max_entries="3")))
        self.assertEqual([c.sequence for c in self.layer().commits], [1, 2])

    async def test_children_list_over_its_limit_is_invalid_argument_with_error_info(self):
        self.state.children_list_limit = 2
        identity = await self.create(ordinary(prim(2, "P"), prim(3, "A", 2), prim(4, "B", 2), prim(5, "C", 2)))
        before = self.checkpoint()
        # The list's three entries are outside the delta entry count, which is one Reorder.
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            await self.update(identity, ordinary(Command(4, 2, children=(5, 4, 3))))
        self.assertEqual(caught.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(self.entry_limit_info(caught.exception),
                         ("CHILDREN_LIST_TOO_LONG", dict(limit="2", actual="3")))
        self.unchanged(before)
        self.assertEqual(self.state.rejections[-1]["kind"], "children_list_too_long")
        self.state.children_list_limit = 3
        self.assertEqual((await self.update(identity, ordinary(Command(4, 2, children=(5, 4, 3))))).sequence_num, 2)

    def unavailable_details(self, error):
        """The ErrorInfo and RetryInfo of an UNAVAILABLE refusal's grpc-status-details-bin, None when absent."""
        details = [value for key, value in error.trailing_metadata() if key == "grpc-status-details-bin"]
        if not details:
            return None
        self.assertEqual(len(details), 1, error.trailing_metadata())
        status = RpcStatus.FromString(details[0])
        self.assertEqual(status.code, grpc.StatusCode.UNAVAILABLE.value[0])
        found = {}
        for detail in status.details:
            if detail.type_url == "type.googleapis.com/google.rpc.ErrorInfo":
                info = ErrorInfo.FromString(detail.value)
                found["error_info"] = (info.reason, info.domain)
            else:
                self.assertEqual(detail.type_url, "type.googleapis.com/google.rpc.RetryInfo")
                delay = RetryInfo.FromString(detail.value).retry_delay
                found["retry_delay"] = (delay.seconds, delay.nanos)
        return found

    async def test_planned_unavailable_refusals_apply_nothing_with_and_without_details(self):
        identity = await self.create(ordinary(prim(2, "A")))
        before = self.checkpoint()
        self.state.refuse_updates(2, retry_delay_ms=1500)
        self.state.refuse_updates(1, error_info=False)
        self.state.refuse_updates(1, error_info=False, retry_delay_ms=25)
        self.state.refuse_updates(1, domain="other.domain")
        payload = ordinary(prim(3, "B"))
        info = ("UPLOAD_NOT_ADMITTED", "nvidia.omniverse.live.v1alpha")
        expected = [dict(error_info=info, retry_delay=(1, 500_000_000))] * 2 + [
            None, dict(retry_delay=(0, 25_000_000)), dict(error_info=("UPLOAD_NOT_ADMITTED", "other.domain"))]
        for details in expected:
            with self.assertRaises(grpc.aio.AioRpcError) as caught:
                await self.update(identity, payload, operation="op-1")
            self.assertEqual(caught.exception.code(), grpc.StatusCode.UNAVAILABLE)
            self.assertEqual(self.unavailable_details(caught.exception), details)
            self.unchanged(before)
        # Each refusal read the whole upload and records its identity and bytes.
        self.assertEqual({(r["operation_id"], r["client_id"], r["sha256"]) for r in self.state.refusals},
                         {("op-1", "writer", hashlib.sha256(payload).hexdigest())})
        self.assertEqual(len(self.state.refusals), 5)
        # With the plan used, the same upload applies once.
        self.assertEqual((await self.update(identity, payload, operation="op-1")).sequence_num, 2)
        self.assertEqual([c.sequence for c in self.layer().commits], [1, 2])

    async def test_live_identity_create_is_invalid_argument_with_service_message(self):
        identity = await self.create(ordinary(prim(2, "A")))
        before = self.checkpoint()
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            await self.update(identity, ordinary(prim(3, "B"), prim(2, "Other"), fields=(Field(1, "counter", 9),)))
        self.assertEqual((caught.exception.code(), caught.exception.details()),
                         (grpc.StatusCode.INVALID_ARGUMENT, "delta error: duplicate section_id 2 in CreateSection"))
        self.unchanged(before)
        self.assertEqual(self.state.rejections[-1]["code"], grpc.StatusCode.INVALID_ARGUMENT.value[0])

    async def test_reorder_naming_moved_or_unknown_member_is_invalid_argument_with_service_message(self):
        identity = await self.create(ordinary(prim(2, "P"), prim(3, "A", 2), prim(4, "B", 2), prim(5, "Q")))
        await self.update(identity, ordinary(Command(3, 4, parent=5, name="B", old_parent=2)), writer="peer")
        for children, message in (
                ((4, 3), "delta error: ReorderChildren(2) listed section_id 4 whose actual parent_id is 5"),
                ((3, 99), "delta error: ReorderChildren contained unknown section_id 99 for parent 2"),
                ((3, 3), "delta error: ReorderChildren(2) lists section_id 3 more than once in children_list")):
            before = self.checkpoint()
            with self.assertRaises(grpc.aio.AioRpcError) as caught:
                await self.update(identity, ordinary(Command(4, 2, children=children), fields=(Field(1, "counter", 9),)))
            self.assertEqual((caught.exception.code(), caught.exception.details()),
                             (grpc.StatusCode.INVALID_ARGUMENT, message))
            self.unchanged(before)
        self.assertEqual([item["kind"] for item in self.state.rejections], ["malformed", "malformed", "malformed"])

    async def test_per_list_reorder_ignores_non_children_and_repeats_and_orders_each_child_list(self):
        self.state.reorder_rule = "per_list"
        identity = await self.create(ordinary(prim(2, "P"), prim(3, "A", 2), Command(1, 4, parent=2, name="x", spec_type=1),
                                              prim(5, "B", 2), Command(1, 6, parent=2, name="y", spec_type=8),
                                              prim(7, "C", 2), prim(8, "Q")))
        await self.update(identity, ordinary(Command(3, 7, parent=8, name="C", old_parent=2)), writer="peer")
        # C moved away, 99 is unknown, P is the parent itself and the later B and A are repeats: all are ignored,
        # as USD's list ordering ignores them, and the prims B and A are ordered within the positions prims hold,
        # leaving the properties where they are.
        await self.update(identity, ordinary(Command(4, 2, children=(7, 5, 99, 2, 3, 3, 5)), fields=(Field(1, "counter", 9),)))
        self.assertFalse(self.state.rejections)
        state = {node["id"]: node for node in self.layer().model.state}
        self.assertEqual((state[2]["fields"]["primChildren"], state[2]["fields"]["properties"]), (("B", "A"), ("x", "y")))
        self.assertEqual([n["name"] for n in sorted((n for n in state.values() if n["parent"] == 2), key=lambda n: n["order"])],
                         ["B", "x", "A", "y"])
        # A partial property order leaves no child unlisted, so a fresh snapshot holds every child.
        await self.update(identity, ordinary(Command(4, 2, children=(6,), child_list=1)))
        _, full = await self.content(self.subscribe())
        self.assertEqual(Model().apply(full), self.layer().model.state)
        self.assertEqual({node["id"] for node in self.layer().model.state}, {1, 2, 3, 4, 5, 6, 7, 8})

    async def test_reorder_list_is_stored_verbatim_and_unlisted_sections_leave_catch_up(self):
        identity = await self.create(ordinary(prim(2, "A"), prim(3, "B"), prim(4, "C"), prim(5, "Child", 4),
                                    fields=(Field(5, "counter", 7),)))
        live = self.subscribe(identity, 1, writer="observer")
        await self.caught_up(live, identity, 1)
        omitting = ordinary(Command(4, 1, children=(3, 2)))
        await self.update(identity, omitting, writer="peer")
        header, payload = await self.content(live)
        self.assertEqual((header.sequence_num, payload), (2, omitting))
        self.assertEqual(set(self.layer().model.detached), {4, 5})
        # A fresh snapshot walks stored children, so the unlisted subtree is absent.
        _, full = await self.content(self.subscribe())
        observer = Model()
        state = {node["id"]: node for node in observer.apply(full)}
        self.assertEqual((set(state), state[1]["fields"]["primChildren"]), ({1, 2, 3}, ("B", "A")))
        # Reference policy: a resumer that held the unlisted subtree has it retracted.
        _, diff = await self.content(self.subscribe(identity, 1))
        self.assertEqual({command.section for command in decode(diff).commands if command.tag == 6}, {4, 5})
        self.assertEqual(self.ledger_replay(identity, 1).apply(diff), self.layer().model.state)
        # Listing the child again restores it, its subtree and its field for every reader.
        await self.update(identity, ordinary(Command(4, 1, children=(4, 3, 2))), writer="peer")
        _, full = await self.content(self.subscribe())
        self.assertEqual({node["id"]: node["path"] for node in Model().apply(full)},
                         {1: "/", 2: "/A", 3: "/B", 4: "/C", 5: "/C/Child"})
        _, diff = await self.content(self.subscribe(identity, 2))
        self.assertEqual(observer.apply(diff), self.layer().model.state)
        self.assertEqual(next(n for n in observer.state if n["id"] == 5)["fields"], {"counter": 7})
        self.assertFalse(self.state.rejections)

    async def test_same_parent_rename_onto_occupied_name_commits_duplicate_siblings(self):
        identity = await self.create(ordinary(prim(2, "A"), prim(3, "B"), fields=(Field(3, "counter", 3),)))
        response = await self.update(identity, ordinary(Command(3, 3, parent=1, name="A", old_parent=1)))
        self.assertEqual(response.sequence_num, 2)
        state = {node["id"]: node for node in self.layer().model.state}
        self.assertEqual((state[2]["path"], state[3]["path"], state[1]["fields"]["primChildren"]), ("/A", "/A", ("A", "A")))
        _, full = await self.content(self.subscribe())
        self.assertEqual(Model().apply(full), self.replay().state)
        self.assertFalse(self.state.rejections)

    async def test_concurrent_creates_at_one_path_both_commit_and_later_replaces_earlier(self):
        identity = await self.create()
        pause = self.state.plan("UpdateLayer", "before", pause=True)
        later = asyncio.create_task(self.update(identity, ordinary(prim(2, "A")), writer="later"))
        await asyncio.wait_for(pause.reached.wait(), 3)
        other = await self.backend("B")
        earlier = await self.update(identity, ordinary(prim(3, "A"), prim(4, "Child", 3),
                                                       fields=(Field(4, "counter", 5),)), writer="earlier", stub=other)
        pause.release.set()
        self.assertEqual((earlier.sequence_num, (await asyncio.wait_for(later, 5)).sequence_num), (2, 3))
        self.assertEqual({node["id"]: node["path"] for node in self.replay().state}, {1: "/", 2: "/A"})
        self.assertEqual(self.replay().state, self.layer().model.state)
        self.assertEqual([c.writer for c in self.layer().commits[1:]], ["earlier", "later"])
        self.assertFalse(self.state.rejections)

    async def test_field_on_deleted_section_commits_as_unprojected_orphan(self):
        identity = await self.create(ordinary(prim(2, "A"), fields=(Field(2, "counter", 1),)))
        await self.update(identity, ordinary(Command(2, 2, parent=1)), writer="deleter")
        response = await self.update(identity, ordinary(fields=(Field(2, "counter", 2),)), writer="editor")
        self.assertEqual(response.sequence_num, 3)
        self.assertEqual(self.layer().commits[-1].writer, "editor")
        self.assertEqual({node["id"] for node in self.layer().model.state}, {1})
        self.assertEqual(self.layer().model.orphans, {2: {"counter": 2}})
        _, fresh = await self.content(self.subscribe())
        self.assertEqual(Model().apply(fresh), self.replay().state)
        self.assertFalse(self.state.rejections)

    async def test_retired_identity_create_is_accepted(self):
        identity = await self.create(ordinary(prim(2, "A"), fields=(Field(2, "counter", 1),)))
        await self.update(identity, ordinary(Command(2, 2, parent=1)))
        response = await self.update(identity, ordinary(prim(2, "Again")))
        self.assertEqual(response.sequence_num, 3)
        again = next(node for node in self.layer().model.state if node["id"] == 2)
        self.assertEqual((again["path"], again["fields"]), ("/Again", {}))

    async def test_late_batch_conflict_is_atomic(self):
        identity = await self.create(ordinary(prim(2, "A"), fields=(Field(1, "counter", 1),)))
        before = self.checkpoint()
        # The first Create, the replacing Create and the field would be visible if a
        # service applied incrementally before discovering the final duplicate identity.
        await self.rejects(self.update(identity, ordinary(prim(3, "B"), prim(4, "A"), prim(3, "Late"),
                          fields=(Field(1, "counter", 99),))), grpc.StatusCode.INVALID_ARGUMENT)
        self.unchanged(before)
        self.assertEqual(self.state.rejections[-1]["kind"], "malformed")
        response = await self.update(identity, ordinary(prim(3, "B")))
        self.assertEqual(response.sequence_num, 2)
        self.assertEqual(len([c for c in self.state.ledger if c.operation == "UpdateLayer"]), 1)

    async def test_missing_stale_occupied_and_retired_targets_commit(self):
        identity = await self.create(ordinary(prim(2, "A"), prim(3, "B"), prim(4, "Child", 2)))
        before = self.layer().model.state
        # Missing parent, stale Move parents that select a rename, and a
        # missing Delete target each commit without changing the graph.
        for payload in (ordinary(prim(5, "X", 999)), ordinary(Command(3, 4, parent=3, name="Child", old_parent=3)),
                        ordinary(Command(2, 99, parent=1))):
            await self.update(identity, payload)
            self.assertEqual(self.layer().model.state, before)
        await self.update(identity, ordinary(prim(6, "A")))
        self.assertEqual({node["id"]: node["path"] for node in self.replay().state}, {1: "/", 3: "/B", 6: "/A"})
        for retired in (2, 4):
            await self.update(identity, ordinary(prim(retired, f"Again{retired}")))
        self.assertEqual({node["id"] for node in self.replay().state}, {1, 2, 3, 4, 6})
        self.assertEqual(len(self.layer().commits), 7)
        self.assertFalse(self.state.rejections)

    async def test_invalid_replacement_preserves_old_incarnation(self):
        identity = await self.create(ordinary(prim(2, "A")))
        for payload, code in ((b"not ompf", grpc.StatusCode.INVALID_ARGUMENT),
                              (ordinary(prim(3, "B"), prim(3, "C")), grpc.StatusCode.INVALID_ARGUMENT)):
            before = self.checkpoint()
            await self.rejects(self.create(payload), code)
            self.unchanged(before)
        header, payload = await self.content(self.subscribe())
        self.assertEqual(header.object_id, identity)
        self.assertEqual(Model().apply(payload), self.replay().state)

    async def test_identity_capacity_preserves_tombstones_and_resets_only_on_replacement(self):
        # Root plus three created IDs fills the bound even with no live prims.
        # Lower the real bound for this test; no wall-clock expiration is involved.
        with patch('graph_service.MAX_USED_IDS', 4):
            identity = await self.create(ordinary(prim(2, 'A')))
            await self.update(identity, ordinary(Command(2, 2, parent=1), prim(3, 'B'),
                                                prim(4, 'Temporary'), Command(2, 4, parent=1)))
            self.assertEqual(self.layer().used_ids, frozenset((1, 2, 3, 4)))
            before = self.checkpoint()
            await self.rejects(self.update(identity, ordinary(prim(5, 'TooMany'),
                              fields=(Field(1, 'counter', 99),))), grpc.StatusCode.RESOURCE_EXHAUSTED)
            self.unchanged(before)
            await self.update(identity, ordinary(Command(2, 3, parent=1)))
            before = self.checkpoint()
            await self.rejects(self.update(identity, ordinary(prim(5, 'StillFull'))),
                               grpc.StatusCode.RESOURCE_EXHAUSTED)
            self.unchanged(before)
            # Ordinary edits and retired-identity creates work at the bound; nothing is evicted.
            await self.update(identity, ordinary(prim(2, 'Retired'), fields=(Field(1, 'counter', 7),)))
            self.assertEqual(self.layer().used_ids, frozenset((1, 2, 3, 4)))
            replacement = await self.create(ordinary(prim(2, 'Fresh')))
            self.assertNotEqual(replacement, identity)
            self.assertEqual(self.layer().used_ids, frozenset((1, 2)))

    async def test_scalar_token_capacity_reserves_resume_union_atomically(self):
        with patch('scene_graph_wire.MAX_STRING_BYTES', 64):
            identity = await self.create()
            await self.update(identity, ordinary(fields=(Field(1, 'a', Token('x' * 31)),)))
            before = self.checkpoint()
            # Both individual Updates fit; their retained union exceeds the
            # half-budget reserve needed for a before/after resume diff.
            await self.rejects(self.update(identity, ordinary(fields=(Field(1, 'b', Token('y')),))),
                               grpc.StatusCode.RESOURCE_EXHAUSTED)
            self.unchanged(before)
            await self.update(identity, ordinary(fields=(Field(1, 'a', Token('x' * 30)), Field(1, 'b', Token('')))))
            self.assertEqual(self.layer().model.state[0]['fields'], {'a': Token('x' * 30), 'b': Token('')})

    async def test_path_vector_and_list_bucket_capacity_reserves_resume_union(self):
        for wrap in (lambda p: Paths((p,)), lambda p: PathOp(added=(p,)),
                     lambda p: PathOp(prepended=(p[:15],), deleted=(p[15:],))):
            with self.subTest(wrap=wrap), patch('scene_graph_wire.MAX_STRING_BYTES', 64):
                identity = await self.create()
                # Two canonical paths concatenate to 31 bytes; every path in
                # every list bucket must be charged, including deleted entries.
                full = '/abcdefghijklmn/opqrstuvwxyzabc'
                self.assertEqual(len(full), 31)
                value = wrap(full)
                await self.update(identity, ordinary(fields=(Field(1, 'a', value),)))
                before = self.checkpoint()
                await self.rejects(self.update(identity, ordinary(fields=(Field(1, 'b', Paths(('/B',))),))),
                                   grpc.StatusCode.RESOURCE_EXHAUSTED)
                self.unchanged(before)

    async def test_explicit_replacement_resets_identity_epoch_and_closes_old_read(self):
        old_id = await self.create(ordinary(prim(2, "A")))
        await self.update(old_id, ordinary(Command(2, 2, parent=1)))
        stream = self.subscribe(old_id, 2)
        _, empty = await self.content(stream)
        self.assertEqual(empty, b"")
        new_id = await self.create(ordinary(prim(2, "Replacement")))
        self.assertEqual(new_id, old_id + 1)
        self.assertEqual(self.layer().commits[-1].sequence, 1)
        self.assertEqual(len(self.layer().model.history), 1)
        await self.rejects(stream.read(), grpc.StatusCode.FAILED_PRECONDITION)
        before = self.checkpoint()
        await self.rejects(self.update(old_id, ordinary(fields=(Field(1, "counter", 9),))),
                           grpc.StatusCode.FAILED_PRECONDITION)
        self.unchanged(before)
        header, payload = await self.content(self.subscribe(old_id, 2))
        self.assertEqual((header.object_id, header.sequence_num), (new_id, 1))
        self.assertEqual(Model().apply(payload), self.replay().state)

    async def test_concurrent_distinct_creates_commit_exactly_once_across_backends(self):
        identity = await self.create()
        other = await self.backend("B")
        responses = await asyncio.gather(self.update(identity, ordinary(prim(2, "A")), writer="a"),
                                        self.update(identity, ordinary(prim(3, "B")), writer="b", stub=other))
        self.assertEqual({reply.sequence_num for reply in responses}, {2, 3})
        self.assertEqual({node["path"] for node in self.replay().state}, {"/", "/A", "/B"})
        self.assertEqual(self.replay().state, self.layer().model.state)
        self.assertEqual([c.sequence for c in self.layer().commits], [1, 2, 3])
        self.assertEqual({c.writer for c in self.layer().commits[1:]}, {"a", "b"})
        self.assertEqual(self.state.retained_bytes, sum(len(c.payload) for c in self.state.ledger))

    async def test_resume_diff_fresh_snapshot_and_quiet_caught_up(self):
        identity = await self.create(ordinary(prim(2, "A"), prim(3, "Child", 2), prim(4, "B"),
                                    fields=(Field(3, "counter", 7), Field(1, "gone", 9))))
        base = self.replay()
        await self.update(identity, ordinary(Command(3, 3, parent=4, name="Moved", old_parent=2),
                                            Command(2, 2, parent=1), prim(5, "New"),
                                            fields=(Field(1, "gone", None), Field(3, "counter", -0.0))), writer="own")
        await self.update(identity, ordinary(fields=(Field(5, "counter", 88),)), writer="peer")
        other = await self.backend("B")
        resumed = self.subscribe(identity, 1, writer="own", stub=other)
        header, payload = await self.content(resumed)
        delta = decode(payload)
        self.assertTrue(delta.is_diff)
        self.assertEqual(delta.base_version, 1)
        self.assertEqual((header.object_id, header.sequence_num), (identity, 3))
        self.assertEqual(header.client_timestamp_microseconds_since_epoch, 123)
        await self.caught_up(resumed, identity, 3)
        self.assertEqual(ranked(base.apply(payload, 3)), ranked(self.replay().state))
        moved = next(node for node in base.state if node["id"] == 3)
        self.assertEqual(moved["path"], "/B/Moved")
        self.assertEqual(math.copysign(1, moved["fields"]["counter"]), -1)
        fresh_call = self.subscribe(writer="own")
        fresh_header, fresh = await self.content(fresh_call)
        self.assertEqual(fresh_header.sequence_num, 3)
        await self.caught_up(fresh_call, identity, 3)
        self.assertEqual(ranked(Model().apply(fresh)), ranked(self.replay().state))
        quiet_header, quiet = await self.content(self.subscribe(identity, 3))
        self.assertEqual((quiet_header.sequence_num, quiet), (3, b""))
        self.assertEqual(self.state.reads[0], dict(backend="B", attempt=1, object=identity, cursor=1))

    async def test_sample_map_and_key_edits_survive_backend_resume(self):
        initial = TimeSamples(((-0.0, 1), (2.0, Token('old'))))
        identity = await self.create(ordinary(fields=(Field(1, 'timeSamples', initial),)))
        base = self.replay()
        await self.update(identity, encode(Delta(samples=(Sample(1, 0.0, True), Sample(1, 3.0, 9)))), writer='peer')
        observed = self.layer().model.state[0]['fields']['timeSamples']
        self.assertEqual(typed_value(observed), typed_value(TimeSamples(((-0.0, True), (2.0, Token('old')), (3.0, 9)))))
        replacement = TimeSamples(((0.0, 3), (2.0, Token('new'))))
        await self.update(identity, ordinary(fields=(Field(1, 'timeSamples', replacement),)), writer='own')
        await self.update(identity, encode(Delta(samples=(Sample(1, 2.0, None), Sample(1, 4.0, False)))), writer='peer')
        wanted = typed_value(TimeSamples(((0.0, 3), (4.0, False))))
        other = await self.backend('B')
        header, payload = await self.content(self.subscribe(identity, 1, writer='own', stub=other))
        self.assertEqual((header.sequence_num, decode(payload).base_version), (4, 1))
        base.apply(payload, 4)
        self.assertEqual(typed_value(base.state[0]['fields']['timeSamples']), wanted)
        fresh_header, fresh = await self.content(self.subscribe(stub=other))
        self.assertEqual(fresh_header.sequence_num, 4)
        self.assertEqual(typed_value(Model().apply(fresh)[0]['fields']['timeSamples']), wanted)
        self.assertEqual(len(self.layer().commits), 4)

    async def test_sample_elements_and_strings_reserve_resume_capacity(self):
        with patch('scene_graph_wire.MAX_RECORDS', 12):
            identity = await self.create()
            await self.update(identity, encode(Delta(samples=tuple(Sample(1, float(i), i) for i in range(4)))))
            before = self.checkpoint()
            await self.rejects(self.update(identity, encode(Delta(samples=(Sample(1, 4.0, 4),)))),
                               grpc.StatusCode.RESOURCE_EXHAUSTED)
            self.unchanged(before)
            await self.update(identity, encode(Delta(samples=(Sample(1, 0.0, None), Sample(1, 4.0, 4)))))
        with patch('scene_graph_wire.MAX_STRING_BYTES', 64):
            identity = await self.create()
            # timeSamples consumes 11 bytes; nested token consumes the rest of
            # the 32-byte retained reserve. A second token must reject atomically.
            await self.update(identity, encode(Delta(samples=(Sample(1, 0.0, Token('x'*21)),))))
            before = self.checkpoint()
            await self.rejects(self.update(identity, encode(Delta(samples=(Sample(1, 1.0, Token('y')),)))),
                               grpc.StatusCode.RESOURCE_EXHAUSTED)
            self.unchanged(before)

    async def test_creation_source_format_survives_edits_and_backend_resume(self):
        seed = encode(Delta(True, commands=(Command(5, 1, spec_type=7),
                            Command(5, 2, parent=1, name="P", spec_type=6)),
                            fields=(Field(2, "counter", 17),), source_format="usda"))
        identity = await self.create(seed)
        base = self.replay()
        self.assertEqual(base.source_format, "usda")
        await self.update(identity, ordinary(fields=(Field(2, "counter", 29),)))
        other = await self.backend("B")
        header, payload = await self.content(self.subscribe(identity, 1, stub=other))
        self.assertEqual((header.object_id, header.sequence_num), (identity, 2))
        self.assertEqual(decode(payload).source_format, "")  # Incremental preservation.
        base.apply(payload, 2)
        self.assertEqual((base.state, base.source_format), (self.replay().state, "usda"))
        for stub in (self.stub, other):
            header, full = await self.content(self.subscribe(stub=stub))
            fresh = Model()
            fresh.apply(full, header.sequence_num)
            self.assertEqual(decode(full).source_format, "usda")
            self.assertEqual((fresh.state, fresh.source_format), (base.state, "usda"))
        # Failed replacement creation preserves the prior incarnation, format,
        # graph and ledger. Snapshot retention reserves format string capacity.
        for payload, code in ((encode(Delta(True, 1, fields=(Field(2, "value", 1),),
                                          source_format="rejected")), grpc.StatusCode.INVALID_ARGUMENT),
                              (encode(Delta(True, source_format="s" * (128 * 1024 + 1))),
                               grpc.StatusCode.RESOURCE_EXHAUSTED)):
            before = self.checkpoint()
            await self.rejects(self.create(payload), code)
            self.unchanged(before)
        replaced = await self.create(encode(Delta(True)))
        self.assertGreater(replaced, identity)
        self.assertEqual(self.layer().model.source_format, "")
        header, full = await self.content(self.subscribe())
        self.assertEqual(header.object_id, replaced)
        self.assertEqual(decode(full).source_format, "")

    async def test_live_delivery_filters_self_only_after_initial_response(self):
        identity = await self.create(ordinary(prim(2, "A")))
        stream = self.subscribe(identity, 1, writer="own")
        await self.content(stream)
        own = ordinary(fields=(Field(2, "counter", 1),))
        peer = ordinary(fields=(Field(2, "counter", 2),))
        await self.update(identity, own, writer="own")
        await self.update(identity, peer, writer="peer")
        header, payload = await self.content(stream)
        self.assertEqual(header.sequence_num, 3)
        self.assertEqual(payload, peer)
        self.assertFalse(decode(payload).is_diff)
        self.assertEqual(header.client_timestamp_microseconds_since_epoch, 123)

    async def test_unverified_catch_up_delivers_own_commits_made_before_the_read_opened(self):
        # Without a proven stable cut a fresh snapshot is acked below commits it holds and delivers them again. The
        # reader's own commits among them were made before the read opened, so they are delivered like a peer's;
        # only the reader's commits made after the read opened are filtered.
        identity = await self.create(ordinary(prim(2, "A")))
        peer = ordinary(fields=(Field(2, "counter", 1),))
        own = ordinary(fields=(Field(2, "counter", 2),))
        await self.update(identity, peer, writer="peer")
        await self.update(identity, own, writer="own")
        self.state.unverified_cut = 2
        stream = self.subscribe(writer="own")
        header, snapshot = await self.content(stream)
        self.assertEqual(header.sequence_num, 1)
        self.assertEqual(ranked(Model().apply(snapshot)), ranked(self.replay().state))
        await self.caught_up(stream, identity, 1)
        delivered = [await self.content(stream) for _ in range(2)]
        self.assertEqual([(header.sequence_num, payload) for header, payload in delivered], [(2, peer), (3, own)])
        await self.update(identity, ordinary(fields=(Field(2, "counter", 3),)), writer="own")
        later = ordinary(fields=(Field(2, "counter", 4),))
        await self.update(identity, later, writer="peer")
        header, payload = await self.content(stream)
        self.assertEqual((header.sequence_num, payload), (5, later))

    async def test_lost_reply_keeps_commit_and_late_create_observes_current_graph(self):
        identity = await self.create()
        pause = self.state.plan("UpdateLayer", "before", pause=True)
        delayed = asyncio.create_task(self.update(identity, ordinary(prim(2, "A"), fields=(Field(3, "counter", 1),))))
        await asyncio.wait_for(pause.reached.wait(), 3)
        await self.update(identity, ordinary(prim(3, "A")))
        pause.release.set()
        self.assertEqual((await asyncio.wait_for(delayed, 5)).sequence_num, 3)
        # The late Create replaces the committed occupant; its field on that occupant is orphaned.
        self.assertEqual({node["id"]: node["path"] for node in self.layer().model.state}, {1: "/", 2: "/A"})
        self.assertEqual(self.layer().model.orphans, {3: {"counter": 1}})
        for boundary in ("after_commit", "before_reply"):
            fault = self.state.plan("UpdateLayer", boundary, code=grpc.StatusCode.UNAVAILABLE)
            old_count = len(self.layer().commits)
            await self.rejects(self.update(identity, ordinary(fields=(Field(1, "counter", old_count),))),
                               grpc.StatusCode.UNAVAILABLE)
            self.assertEqual(fault.fired, 1)
            self.assertEqual(len(self.layer().commits), old_count + 1)
            self.assertEqual(self.layer().model.state[0]["fields"]["counter"], old_count)
        self.assertEqual(self.replay().state, self.layer().model.state)

    async def test_malformed_diff_and_snapshot_capacity_are_atomic(self):
        identity = await self.create()
        cases = [(b"bad", grpc.StatusCode.INVALID_ARGUMENT),
                 (encode(Delta(True, 1, fields=(Field(1, "counter", 5),))), grpc.StatusCode.INVALID_ARGUMENT),
                 (ordinary(fields=(Field(1, "large", tuple("T" + str(i) for i in range(2048))),)),
                  grpc.StatusCode.RESOURCE_EXHAUSTED)]
        for payload, code in cases:
            before = self.checkpoint()
            await self.rejects(self.update(identity, payload), code)
            self.unchanged(before)
        self.assertEqual([item["kind"] for item in self.state.rejections], ["malformed", "unsupported", "limit"])
        self.assertNotIn("graph-test-token", json.dumps(self.state.snapshot()))

    async def test_cursor_ahead_of_latest_gets_caught_up_header_at_latest(self):
        identity = await self.create(ordinary(prim(2, "A")))
        ahead = self.subscribe(identity, 10, writer="own")
        # The only initial response is the caught-up header at the service's
        # latest sequence, below the cursor. This asserts the header content
        # only; a client treats it as a server sequence regression (design section 12).
        await self.caught_up(ahead, identity, 1)
        self.assertEqual(self.state.reads[-1]["cursor"], 10)
        self.assertFalse(self.state.rejections)

    async def test_resume_behind_retained_history_gets_one_coalesced_diff(self):
        # Retention equal to the model's history bound, as with the default 4096.
        patcher = patch('scene_graph_model.MAX_COMMITS', 2)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.state.retained_updates = 2
        identity = await self.create(ordinary(prim(2, "A"), prim(3, "Child", 2), prim(4, "B"),
                                    fields=(Field(3, "counter", 7), Field(4, "gone", 1))))
        payloads = [ordinary(prim(5, "Ghost"), fields=(Field(4, "gone", None),)),
                    ordinary(Command(2, 5, parent=1), Command(3, 3, parent=4, name="Moved", old_parent=2)),
                    ordinary(Command(2, 2, parent=1)),
                    ordinary(prim(2, "Again", 4), fields=(Field(2, "fresh", 1),)),
                    ordinary(fields=(Field(3, "counter", 8),))]
        for payload in payloads:
            await self.update(identity, payload, writer="peer")
        latest = len(payloads) + 1
        # The service accepts every update and keeps only its newest retained entries.
        self.assertEqual([entry.sequence for entry in self.layer().commits], [latest - 1, latest])
        self.assertLessEqual(len(self.layer().model.history), 2)
        self.assertEqual(self.layer().model.state, self.ledger_replay(identity, latest).state)
        for cursor in range(1, latest):
            with self.subTest(cursor=cursor):
                call = self.subscribe(identity, cursor, writer="peer")
                header, payload = await self.content(call)
                delta = decode(payload)
                self.assertEqual((header.object_id, header.sequence_num), (identity, latest))
                self.assertEqual((delta.is_diff, delta.base_version), (True, cursor))
                await self.caught_up(call, identity, latest)
                before = self.ledger_replay(identity, cursor)
                self.assertEqual(before.apply(payload, latest), self.layer().model.state)
                call.cancel()
        # A section created and deleted after the cursor is dropped, never
        # retracted; a section the subscriber held at its cursor is retracted.
        retracted = {}
        for cursor in (1, 2):
            _, payload = await self.content(self.subscribe(identity, cursor))
            retracted[cursor] = {command.section for command in decode(payload).commands if command.tag == 6}
        self.assertEqual(retracted, {1: set(), 2: {5}})

    async def test_live_subscriber_behind_retained_history_is_unavailable(self):
        self.state.retained_updates = 2
        identity = await self.create()
        stream = self.subscribe(identity, 1)
        await self.caught_up(stream, identity, 1)
        gate = self.state.plan_read_frame(1, 2)
        await self.update(identity, ordinary(fields=(Field(1, "counter", 2),)))
        await asyncio.wait_for(gate.reached.wait(), 3)
        for value in range(3, 7):
            await self.update(identity, ordinary(fields=(Field(1, "counter", value),)))
        gate.release.set()
        header, _ = await self.content(stream)
        self.assertEqual(header.sequence_num, 2)
        await self.rejects(stream.read(), grpc.StatusCode.UNAVAILABLE)

    async def test_delayed_move_and_delete_apply_to_latest_entire_subtree(self):
        for move in (True, False):
            with self.subTest(move=move):
                identity = await self.create(ordinary(prim(2, "A"), prim(3, "Original", 2), prim(4, "B")))
                base = self.replay()
                operation = Command(3, 2, parent=4, name="Moved", old_parent=1) if move else Command(2, 2, parent=1)
                pause = self.state.plan("UpdateLayer", "before", pause=True)
                delayed = asyncio.create_task(self.update(identity, ordinary(operation), writer="structural"))
                try:
                    await asyncio.wait_for(pause.reached.wait(), 3)
                    child = ordinary(prim(5, "PeerChild", 2), prim(6, "Grandchild", 5),
                                     Command(3, 3, parent=4, name="Escaped", old_parent=2),
                                     fields=(Field(5, "counter", 77), Field(6, "counter", 88)))
                    peer = await self.update(identity, child, writer="peer")
                    self.assertEqual(peer.sequence_num, 2)
                finally:
                    pause.release.set()
                    response = await asyncio.wait_for(delayed, 5)
                self.assertEqual(response.sequence_num, 3)
                state = {node["id"]: node for node in self.replay().state}
                self.assertEqual(state[3]["path"], "/B/Escaped")
                if move:
                    self.assertEqual(set(state), {1, 2, 3, 4, 5, 6})
                    self.assertEqual(state[5]["path"], "/B/Moved/PeerChild")
                    self.assertEqual(state[6]["path"], "/B/Moved/PeerChild/Grandchild")
                    self.assertEqual(state[5]["fields"], {"counter": 77, "primChildren": ("Grandchild",)})
                    self.assertEqual(state[6]["fields"], {"counter": 88})
                else:
                    self.assertEqual(set(state), {1, 3, 4})
                self.assertEqual([entry.writer for entry in self.layer().commits[1:]], ["peer", "structural"])
                self.assertEqual(self.layer().commits[1].payload, child)
                _, fresh = await self.content(self.subscribe())
                self.assertEqual(ranked(Model().apply(fresh)), ranked(self.replay().state))
                _, resumed = await self.content(self.subscribe(identity, 1))
                self.assertEqual(ranked(base.apply(resumed, 3)), ranked(self.replay().state))

    async def test_partial_read_gate_releases_exact_remainder(self):
        object_id = await self.create(ordinary(prim(2, 'P'), fields=(Field(2, 'marker', 17),)))
        before = self.checkpoint()
        expected = self.layer().snapshot_bytes
        for attempt, sequence, prefix in ((0, 1, 7), (1, 0, 7), (1, 1, 0), (1, 1, 65536), (1, 1, True)):
            with self.assertRaises(ValueError):
                self.state.plan_partial_read(attempt, sequence, prefix)
        gate = self.state.plan_partial_read(1, 1, 7)
        with self.assertRaises(ValueError):
            self.state.plan_partial_read(1, 1, 7)
        call = self.subscribe()
        header = await asyncio.wait_for(call.read(), 3)
        self.assertEqual(header.WhichOneof('response_type'), 'header')
        self.assertEqual((header.header.object_id, header.header.sequence_num, header.header.total_size),
                         (object_id, 1, len(expected)))
        partial = await asyncio.wait_for(call.read(), 3)
        self.assertEqual(partial.WhichOneof('response_type'), 'chunk')
        self.assertEqual(partial.chunk.data, expected[:7])
        await asyncio.wait_for(gate.reached.wait(), 3)
        self.assertEqual(gate.fired, 1)
        gate.release.set()
        payload = bytearray(partial.chunk.data)
        while len(payload) < len(expected):
            frame = await asyncio.wait_for(call.read(), 3)
            self.assertEqual(frame.WhichOneof('response_type'), 'chunk')
            payload.extend(frame.chunk.data)
        self.assertEqual(bytes(payload), expected)
        self.unchanged(before)
        self.assertEqual(self.state.snapshot()['graph_partial_reads'], [dict(attempt=1, sequence=1, prefix_bytes=7)])


if __name__ == "__main__":
    unittest.main()
