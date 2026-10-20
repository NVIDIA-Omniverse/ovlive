# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire tests of the proposed chained UpdateLayer semantics in both reference profiles.

A chained call sends a ChainedUpdateLayerHeader, whose previous_operation_id
names its predecessor, and commits only when its writer's latest committed
operation on the object is that predecessor. After its content is read it
waits for that predecessor until it commits, its deadline or a cancellation;
with no_wait it is refused at once instead, naming the chain's latest write.
Every case synchronizes on reference-service events; timeouts are failure-only
watchdogs, apart from the deadline case, whose predecessor a pause gate holds.
"""
import asyncio
import json
import os
from pathlib import Path
import unittest

import grpc

from graph_service import GraphServer, GraphState
from service import ErrorInfo, Principal, ReferenceServer, RetryInfo, RpcStatus, State, pb, rpc
from scene_graph_wire import Delta, Field, encode

AUTH = (("authorization", "Bearer chain-test-token"),)
DOMAIN = "nvidia.omniverse.live.v1alpha"
WATCHDOG = 10


class ChainedWriteCases:
    """Cases shared by the opaque and graph profiles; subclasses name the profile."""

    state_type = server_type = None

    def payload(self, value):
        raise NotImplementedError

    async def asyncSetUp(self):
        self.state = self.state_type(int(os.environ.get("LIVE_TEST_SEED", "42")))
        self.state.tokens["chain-test-token"] = Principal("writer")
        self.servers, self.channels, self.tasks = [], [], []
        self.addAsyncCleanup(self.cleanup)
        server = self.server_type(self.state, "A")
        self.servers.append(server)
        channel = grpc.aio.insecure_channel(await server.start())
        self.channels.append(channel)
        await asyncio.wait_for(channel.channel_ready(), WATCHDOG)
        self.stub = rpc.LiveServiceStub(channel)
        self.session = (await self.stub.CreateSession(pb.CreateSessionRequest(path="/chain-test", name="chain"),
                                                     metadata=AUTH, timeout=WATCHDOG)).session_id

    async def cleanup(self):
        for fault in self.state.faults:
            fault.release.set()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await asyncio.wait_for(self.state.settle(), WATCHDOG)
        failures = []
        for group in ([channel.close() for channel in self.channels], [server.stop() for server in self.servers]):
            failures.extend(await asyncio.wait_for(asyncio.gather(*group, return_exceptions=True), WATCHDOG))
        artifact = os.environ.get("LIVE_TEST_ARTIFACT_DIR")
        if artifact:
            Path(artifact, self.id().split(".")[-1] + ".chain-ledger.json").write_text(
                json.dumps(self.state.snapshot(), indent=2, sort_keys=True) + "\n")
        self.state.assert_faults_fired()
        for failure in failures:
            if isinstance(failure, BaseException):
                raise failure

    async def create(self, operation=""):
        frames = [pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="root.live",
                  operation_id=operation)), pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=self.payload(0)))]
        return (await self.stub.CreateLayer(iter(frames), metadata=AUTH, timeout=WATCHDOG)).object_id

    def header(self, object_id, operation, previous=None, *, writer="w1", no_wait=False):
        """The first request of an UpdateLayer call: a chained header unless `previous` is None."""
        header = pb.UpdateLayerHeader(session_id=self.session, layer_name="root.live", object_id=object_id,
                                      client_id=writer, operation_id=operation,
                                      timestamp_microseconds_since_epoch=123)
        if previous is None:
            return pb.UpdateLayerRequest(header=header)
        return pb.UpdateLayerRequest(chained_header=pb.ChainedUpdateLayerHeader(
            header=header, previous_operation_id=previous, no_wait=no_wait))

    async def write(self, object_id, operation, previous=None, *, writer="w1", value=0, timeout=WATCHDOG,
                    no_wait=False):
        """One UpdateLayer call; `previous` None sends an unchained call."""
        payload = self.payload(value)
        frames = [self.header(object_id, operation, previous, writer=writer, no_wait=no_wait),
                  pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=payload))]
        return await self.stub.UpdateLayer(iter(frames), metadata=AUTH, timeout=timeout)

    def start(self, *args, **kwargs):
        task = asyncio.create_task(self.write(*args, **kwargs))
        self.tasks.append(task)
        return task

    async def until(self, predicate):
        await asyncio.wait_for(self.state.observed(predicate), WATCHDOG)

    async def waiting(self, operation, writer="w1"):
        await self.until(lambda: (writer, operation) in self.state.waiting)

    def events(self, event, writer="w1"):
        """The operation IDs of `writer`'s chain events of kind `event`, in order."""
        return [entry["operation_id"] for entry in self.state.chain_log
                if entry["event"] == event and entry["client_id"] == writer]

    async def cancelled(self, *calls):
        """Cancels waiting calls, as a client's RST_STREAM does, and waits until the service has dropped each."""
        dropped = len(self.events("dropped"))
        for call in calls:
            call.cancel()
        await asyncio.gather(*calls, return_exceptions=True)
        await self.until(lambda: len(self.events("dropped")) == dropped + len(calls))

    async def refused(self, call, code):
        with self.assertRaises(grpc.aio.AioRpcError) as caught:
            await asyncio.wait_for(call, WATCHDOG)
        self.assertEqual(caught.exception.code(), code, caught.exception.details())
        return caught.exception

    def details(self, error):
        """The status details a refusal's grpc-status-details-bin carries, by type name."""
        raw = [value for key, value in error.trailing_metadata() if key == "grpc-status-details-bin"]
        if not raw:
            return {}
        self.assertEqual(len(raw), 1)
        status = RpcStatus.FromString(raw[0])
        self.assertEqual(status.code, error.code().value[0])
        types = {"type.googleapis.com/google.rpc.ErrorInfo": ErrorInfo,
                 "type.googleapis.com/google.rpc.RetryInfo": RetryInfo}
        return {detail.type_url.rsplit(".", 1)[-1]: types[detail.type_url].FromString(detail.value)
                for detail in status.details}

    async def chain_refused(self, call, last, record=None):
        """Asserts a PREDECESSOR_NOT_COMMITTED refusal naming `last` as the writer's latest operation. Its
        last_record is `operation` when it names one and otherwise `record`, by default `absent`."""
        error = await self.refused(call, grpc.StatusCode.ABORTED)
        info = self.details(error)["ErrorInfo"]
        record = "operation" if last else record or "absent"
        self.assertEqual((info.reason, info.domain, dict(info.metadata)),
                         ("PREDECESSOR_NOT_COMMITTED", DOMAIN, {"last_operation_id": last, "last_record": record}))

    async def invalid_update(self, call):
        """Asserts the INVALID_ARGUMENT a chaining replica gives a chained call, with its INVALID_UPDATE ErrorInfo."""
        error = await self.refused(call, grpc.StatusCode.INVALID_ARGUMENT)
        info = self.details(error)["ErrorInfo"]
        self.assertEqual((info.reason, info.domain), ("INVALID_UPDATE", DOMAIN))

    def writes(self, writer="w1"):
        """(sequence, payload) of each committed update by `writer`, in commit order."""
        return [(entry.sequence, entry.payload) for entry in self.state.ledger
                if entry.operation == "UpdateLayer" and entry.writer == writer]

    def record(self, object_id, writer="w1"):
        record = self.state.operations.get(("UpdateLayer", self.session, "root.live", object_id, writer))
        return record.operation_id if record else ""

    async def test_chained_replies_report_operation_outcomes_and_reject_other_content(self):
        object_id = await self.create()
        first = await self.write(object_id, "op-1", "", value=1)
        self.assertEqual(first.operation_outcome, pb.OPERATION_OUTCOME_APPLIED)
        again = await self.write(object_id, "op-1", "", value=1)
        self.assertEqual((again.sequence_num, again.operation_outcome),
                         (first.sequence_num, pb.OPERATION_OUTCOME_ALREADY_COMMITTED))
        await self.refused(self.write(object_id, "op-1", "", value=2), grpc.StatusCode.ALREADY_EXISTS)
        # The own-ID check precedes the chain check: a re-send names any predecessor and still matches.
        again = await self.write(object_id, "op-1", "elsewhere", value=1)
        self.assertEqual(again.operation_outcome, pb.OPERATION_OUTCOME_ALREADY_COMMITTED)
        self.assertEqual(len(self.writes()), 1)

    async def test_chained_call_without_operation_or_client_id_is_refused_before_its_chunks(self):
        object_id = await self.create()
        for header in (self.header(object_id, "", ""), self.header(object_id, "op-1", "", writer=""),
                       self.header(object_id, "op-1", "bad id!")):
            with self.subTest(header=str(header)):
                call = self.stub.UpdateLayer(metadata=AUTH, timeout=WATCHDOG)
                await call.write(header)
                # The stream stays open: the refusal cannot have waited for the chunks.
                await self.invalid_update(call)
        self.assertEqual(self.writes(), [])

    async def test_every_invalid_argument_of_a_chained_call_carries_invalid_update(self):
        # A chaining replica marks each INVALID_ARGUMENT it gives a chained call, so that a client tells it from a
        # replica that does not know the chained header; an unchained call's refusal carries no ErrorInfo.
        object_id = await self.create()
        await self.invalid_update(self.write(0, "op-1", ""))
        error = await self.refused(self.write(0, "op-1"), grpc.StatusCode.INVALID_ARGUMENT)
        self.assertEqual(self.details(error), {})
        call = self.stub.UpdateLayer(metadata=AUTH, timeout=WATCHDOG)
        await call.write(self.header(object_id, "op-1", ""))
        await call.write(self.header(object_id, "op-1", ""))
        await call.done_writing()
        await self.invalid_update(call)
        self.assertEqual(self.writes(), [])

    async def test_calls_in_flight_commit_in_chain_order(self):
        object_id = await self.create()
        operations = [f"op-{index}" for index in range(1, 9)]
        calls = [self.start(object_id, operation, previous, value=index)
                 for index, (previous, operation) in enumerate(zip([""] + operations, operations), 1)]
        replies = await asyncio.wait_for(asyncio.gather(*calls), WATCHDOG)
        self.assertEqual([reply.sequence_num for reply in replies], list(range(2, 10)))
        self.assertEqual([reply.operation_outcome for reply in replies], [pb.OPERATION_OUTCOME_APPLIED] * 8)
        self.assertEqual(self.writes(), [(index + 1, self.payload(index)) for index in range(1, 9)])
        self.assertEqual(self.record(object_id), "op-8")

    async def test_successors_arriving_first_wait_for_their_predecessors(self):
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        later = []
        for index in range(5, 1, -1):  # op-5 first, op-2 last.
            if index == 2:
                self.assertEqual(len(self.writes()), 1)
            later.append(self.start(object_id, f"op-{index}", f"op-{index - 1}", value=index))
            if index > 2:
                await self.waiting(f"op-{index}")
        replies = await asyncio.wait_for(asyncio.gather(*later), WATCHDOG)
        self.assertEqual([reply.sequence_num for reply in replies], [6, 5, 4, 3])
        self.assertEqual(self.writes(), [(index + 1, self.payload(index)) for index in range(1, 6)])
        self.assertEqual(self.state.waiting, set())

    async def test_a_waiting_call_does_not_block_other_writers_or_its_own_predecessor(self):
        object_id = await self.create()
        hold = self.state.plan_write("hold", position=1, writer="w1")
        first = self.start(object_id, "op-1", "", value=1)
        await asyncio.wait_for(hold.reached.wait(), WATCHDOG)
        second = self.start(object_id, "op-2", "op-1", value=2)
        await self.waiting("op-2")
        # Another writer commits while w1's first call is held and its second waits.
        other = await self.write(object_id, "x-1", "", writer="w2", value=7)
        self.assertEqual(other.sequence_num, 2)
        hold.release.set()
        replies = await asyncio.wait_for(asyncio.gather(first, second), WATCHDOG)
        self.assertEqual([reply.sequence_num for reply in replies], [3, 4])

    async def test_successors_of_a_refused_predecessor_wait_until_cancelled(self):
        object_id = await self.create()
        refusal = self.state.plan_write("refuse", position=2, retry_delay_ms=2500)
        hold = self.state.plan_write("hold", position=1)
        calls = [self.start(object_id, f"op-{index}", "" if index == 1 else f"op-{index - 1}", value=index)
                 for index in range(1, 5)]
        await asyncio.wait_for(hold.reached.wait(), WATCHDOG)
        for index in range(2, 5):
            await self.waiting(f"op-{index}")
        hold.release.set()
        self.assertEqual((await asyncio.wait_for(calls[0], WATCHDOG)).sequence_num, 2)
        error = await self.refused(calls[1], grpc.StatusCode.UNAVAILABLE)
        details = self.details(error)
        self.assertEqual((details["ErrorInfo"].reason, details["ErrorInfo"].domain,
                          dict(details["ErrorInfo"].metadata)),
                         ("UPLOAD_ADMISSION_SATURATED", DOMAIN, {"nothing_applied": "true"}))
        self.assertEqual((details["RetryInfo"].retry_delay.seconds, details["RetryInfo"].retry_delay.nanos),
                         (2, 500_000_000))
        self.assertEqual(refusal.fired, 1)
        # The successors keep waiting for op-2, which may still be re-sent; the service refuses none of them.
        self.assertEqual(self.state.waiting, {("w1", "op-3"), ("w1", "op-4")})
        self.assertEqual(self.events("refused"), [])
        await self.cancelled(*calls[2:])
        self.assertEqual(self.state.waiting, set())
        self.assertEqual(self.writes(), [(2, self.payload(1))])
        # The writer re-sends from after op-1 with the same IDs, the first with no_wait, and each commits once,
        # in order, whichever arrives first.
        resent = [self.start(object_id, f"op-{index}", f"op-{index - 1}", value=index, no_wait=index == 2)
                  for index in range(4, 1, -1)]
        replies = await asyncio.wait_for(asyncio.gather(*resent), WATCHDOG)
        self.assertEqual([reply.sequence_num for reply in replies], [5, 4, 3])
        self.assertEqual(self.writes(), [(index + 1, self.payload(index)) for index in range(1, 5)])

    async def test_no_wait_is_refused_at_once_naming_the_latest_write(self):
        object_id = await self.create()
        # No write yet: the record is absent and only the empty predecessor matches.
        await self.chain_refused(self.write(object_id, "op-2", "op-1", value=2, no_wait=True), "")
        hold = self.state.plan_write("hold", position=1)
        first = self.start(object_id, "op-1", "", value=1)
        await asyncio.wait_for(hold.reached.wait(), WATCHDOG)
        # The predecessor is in flight: without no_wait the call waits, with it the call is refused at once.
        waiting = self.start(object_id, "op-2", "op-1", value=2)
        await self.waiting("op-2")
        await self.chain_refused(self.write(object_id, "op-2", "op-1", value=2, no_wait=True), "")
        hold.release.set()
        self.assertEqual((await asyncio.wait_for(first, WATCHDOG)).sequence_num, 2)
        self.assertEqual((await asyncio.wait_for(waiting, WATCHDOG)).sequence_num, 3)
        # A no_wait call commits when its predecessor is the latest write and is answered ALREADY_COMMITTED as
        # the latest write; a stale predecessor is refused naming the latest.
        reply = await self.write(object_id, "op-3", "op-2", value=3, no_wait=True)
        self.assertEqual((reply.sequence_num, reply.operation_outcome), (4, pb.OPERATION_OUTCOME_APPLIED))
        again = await self.write(object_id, "op-3", "op-2", value=3, no_wait=True)
        self.assertEqual((again.sequence_num, again.operation_outcome), (4, pb.OPERATION_OUTCOME_ALREADY_COMMITTED))
        await self.chain_refused(self.write(object_id, "op-2", "op-1", value=2, no_wait=True), "op-3")
        self.assertEqual(len(self.writes()), 3)
        self.assertEqual([entry["no_wait"] for entry in self.state.arrival_log if entry["operation_id"] == "op-3"],
                         [True, True])

    async def test_an_unidentified_write_leaves_a_record_without_an_operation_id(self):
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        await self.write(object_id, "", value=2)
        # The writer's latest write has no operation ID: neither the empty predecessor nor op-1 matches it.
        await self.chain_refused(self.write(object_id, "op-2", "", value=3, no_wait=True), "", "no_operation_id")
        await self.chain_refused(self.write(object_id, "op-2", "op-1", value=3, no_wait=True), "", "no_operation_id")
        self.assertEqual(len(self.writes()), 2)
        # An identified write replaces it.
        await self.write(object_id, "op-3", value=4)
        await self.chain_refused(self.write(object_id, "op-4", "", value=5, no_wait=True), "op-3")

    async def test_predecessor_lost_before_the_service_commits_after_its_resend(self):
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        successors = []
        for index in (3, 4):
            successors.append(self.start(object_id, f"op-{index}", f"op-{index - 1}", value=index))
            await self.waiting(f"op-{index}")
        # op-2's first attempt never reached the service; its re-send does.
        resent = await self.write(object_id, "op-2", "op-1", value=2)
        self.assertEqual(resent.sequence_num, 3)
        replies = await asyncio.wait_for(asyncio.gather(*successors), WATCHDOG)
        self.assertEqual([reply.sequence_num for reply in replies], [4, 5])

    async def test_resend_of_a_lost_reply_after_its_successors_committed_names_the_chain_head(self):
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        self.state.plan_write("drop_reply", position=2)
        await self.refused(self.write(object_id, "op-2", "op-1", value=2), grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await self.write(object_id, "op-3", "op-2", value=3)).sequence_num, 4)
        # op-2 committed and op-3 followed it, so the re-send can never commit: it waits until cancelled, and
        # with no_wait it is refused at once naming op-3.
        waiting = self.start(object_id, "op-2", "op-1", value=2)
        await self.waiting("op-2")
        await self.cancelled(waiting)
        await self.chain_refused(self.write(object_id, "op-2", "op-1", value=2, no_wait=True), "op-3")
        self.assertEqual(len(self.writes()), 3)

    async def test_resend_before_successors_is_answered_already_committed(self):
        object_id = await self.create()
        self.state.plan_write("drop_reply", position=1)
        await self.refused(self.write(object_id, "op-1", "", value=1), grpc.StatusCode.UNAVAILABLE)
        again = await self.write(object_id, "op-1", "", value=1)
        self.assertEqual((again.sequence_num, again.operation_outcome), (2, pb.OPERATION_OUTCOME_ALREADY_COMMITTED))

    async def test_stale_call_committing_after_its_resend_answers_the_resend_already_committed(self):
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        stale = self.state.plan_write("stale", position=2)
        held = self.state.plan_write("hold", position=2)
        # The first attempt's reply is lost, but its commit is still pending in the service.
        await self.refused(self.write(object_id, "op-2", "op-1", value=2), grpc.StatusCode.UNAVAILABLE)
        resend = self.start(object_id, "op-2", "op-1", value=2)
        await asyncio.wait_for(held.reached.wait(), WATCHDOG)
        stale.release.set()
        await asyncio.wait_for(stale.done.wait(), WATCHDOG)
        self.assertEqual(stale.outcome, "APPLIED")
        held.release.set()
        reply = await asyncio.wait_for(resend, WATCHDOG)
        self.assertEqual((reply.sequence_num, reply.operation_outcome), (3, pb.OPERATION_OUTCOME_ALREADY_COMMITTED))
        self.assertEqual((await self.write(object_id, "op-3", "op-2", value=3)).sequence_num, 4)
        self.assertEqual(len(self.writes()), 3)

    async def test_stale_call_committing_after_its_resend_committed_applies_nothing(self):
        object_id = await self.create()
        stale = self.state.plan_write("stale", position=1)
        await self.refused(self.write(object_id, "op-1", "", value=1), grpc.StatusCode.UNAVAILABLE)
        successor = self.start(object_id, "op-2", "op-1", value=2)
        await self.waiting("op-2")  # The stale op-1 is in flight, so its successor waits.
        self.assertEqual((await self.write(object_id, "op-1", "", value=1)).sequence_num, 2)
        self.assertEqual((await asyncio.wait_for(successor, WATCHDOG)).sequence_num, 3)
        stale.release.set()
        await asyncio.wait_for(stale.done.wait(), WATCHDOG)
        self.assertEqual(stale.outcome, "ABORTED")  # The record is op-2: the stale op-1 cannot commit.
        self.assertEqual(len(self.writes()), 2)

    async def test_many_writers_with_full_windows_commit_without_deadlock(self):
        object_id = await self.create()
        writers, window = [f"w{index}" for index in range(1, 9)], 16
        calls = {}
        for writer in writers:
            for index in range(window, 1, -1):
                calls[writer, index] = self.start(object_id, f"op-{index}", f"op-{index - 1}", writer=writer,
                                                  value=index)
        await self.until(lambda: len(self.state.waiting) == len(writers) * (window - 1))
        for writer in writers:
            calls[writer, 1] = self.start(object_id, "op-1", "", writer=writer, value=1)
        await asyncio.wait_for(asyncio.gather(*calls.values()), WATCHDOG)
        for writer in writers:
            sequences = [calls[writer, index].result().sequence_num for index in range(1, window + 1)]
            self.assertEqual(sequences, sorted(sequences))
            self.assertEqual([payload for _, payload in self.writes(writer)],
                             [self.payload(index) for index in range(1, window + 1)])

    async def test_replacing_the_object_mid_chain_fails_waiting_and_later_calls(self):
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        waiting = self.start(object_id, "op-3", "op-2", value=3)
        await self.waiting("op-3")
        replacement = await self.create()
        self.assertNotEqual(replacement, object_id)
        await self.refused(waiting, grpc.StatusCode.FAILED_PRECONDITION)
        await self.refused(self.write(object_id, "op-2", "op-1", value=2), grpc.StatusCode.FAILED_PRECONDITION)
        # The new object has no record and has never seen op-1, so a call naming it waits.
        unplaced = self.start(replacement, "op-9", "op-1", value=9)
        await self.waiting("op-9")
        unplaced.cancel()
        await asyncio.gather(unplaced, return_exceptions=True)
        await self.until(lambda: ("w1", "op-9") not in self.state.waiting)
        # Its first chained write carries the empty predecessor.
        self.assertEqual((await self.write(replacement, "op-10", "", value=10)).sequence_num, 2)

    async def test_a_waiting_call_fails_aborted_at_its_deadline(self):
        object_id = await self.create()
        hold = self.state.plan_write("hold", position=1)
        first = self.start(object_id, "op-1", "", value=1)
        await asyncio.wait_for(hold.reached.wait(), WATCHDOG)
        # The pause gate holds op-1, so op-2 cannot commit before its deadline.
        await self.chain_refused(self.write(object_id, "op-2", "op-1", value=2, timeout=4), "")
        hold.release.set()
        self.assertEqual((await asyncio.wait_for(first, WATCHDOG)).sequence_num, 2)
        self.assertEqual(len(self.writes()), 1)

    async def test_a_second_call_on_a_committed_predecessor_names_the_first(self):
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        await self.write(object_id, "op-2", "op-1", value=2)
        await self.chain_refused(self.write(object_id, "fork", "op-1", value=3, no_wait=True), "op-2")
        await self.chain_refused(self.write(object_id, "fresh", "", value=4, no_wait=True), "op-2")

    async def test_checkpoint_restore_loses_records(self):
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        await self.write(object_id, "op-2", "op-1", value=2)
        waiting = self.start(object_id, "op-4", "op-3", value=4)
        await self.waiting("op-4")
        self.assertEqual(self.state.restore_checkpoint(), 1)
        # The record is absent, so neither op-3 nor the waiting op-4 can commit.
        await self.chain_refused(self.write(object_id, "op-3", "op-2", value=3, no_wait=True), "")
        self.assertIn(("w1", "op-4"), self.state.waiting)
        await self.cancelled(waiting)
        # Without its record, a re-send of op-2 is not recognized as committed.
        await self.chain_refused(self.write(object_id, "op-2", "op-1", value=2, no_wait=True), "")
        self.assertEqual(len(self.writes()), 2)

    async def test_held_reply_lets_a_later_reply_arrive_first(self):
        object_id = await self.create()
        delayed = self.state.plan_write("hold_reply", position=1)
        first = self.start(object_id, "op-1", "", value=1)
        await asyncio.wait_for(delayed.reached.wait(), WATCHDOG)
        self.assertEqual((await self.write(object_id, "op-2", "op-1", value=2)).sequence_num, 3)
        self.assertFalse(first.done())
        delayed.release.set()
        self.assertEqual((await asyncio.wait_for(first, WATCHDOG)).sequence_num, 2)

    async def test_write_faults_select_by_writer_and_operation(self):
        object_id = await self.create()
        by_writer = self.state.plan_write("refuse", position=1, writer="w2", reason="STORE_WRITE_REFUSED")
        by_operation = self.state.plan_write("refuse", operation_id="op-2")
        self.assertEqual((await self.write(object_id, "op-1", "", value=1)).sequence_num, 2)
        error = await self.refused(self.write(object_id, "x-1", "", writer="w2", value=5), grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(self.details(error)["ErrorInfo"].reason, "STORE_WRITE_REFUSED")
        self.assertNotIn("RetryInfo", self.details(error))
        await self.refused(self.write(object_id, "op-2", "op-1", value=2), grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await self.write(object_id, "op-2", "op-1", value=2)).sequence_num, 3)
        self.assertEqual((by_writer.fired, by_operation.fired), (1, 1))
        for bad in (dict(action="explode", position=1), dict(action="hold"), dict(action="hold", position=0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.state.plan_write(bad.pop("action"), **bad)

    async def test_unchained_calls_commit_in_arrival_order_without_waiting(self):
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        hold = self.state.plan_write("hold", position=1, writer="w2")
        held = self.start(object_id, "u-1", writer="w2", value=2)
        await asyncio.wait_for(hold.reached.wait(), WATCHDOG)
        # An unchained call neither waits for nor is refused by the writer's chain, nor by another writer's call.
        self.assertEqual((await self.write(object_id, "u-2", value=3)).sequence_num, 3)
        self.assertEqual((await self.write(object_id, "", value=4)).sequence_num, 4)
        hold.release.set()
        reply = await asyncio.wait_for(held, WATCHDOG)
        self.assertEqual((reply.sequence_num, reply.operation_outcome), (5, pb.OPERATION_OUTCOME_APPLIED))
        self.assertEqual(self.state.waiting, set())

    async def overtaken(self, call):
        """Asserts the start-record guard's refusal: UNAVAILABLE, the overtaken message and no ErrorInfo."""
        error = await self.refused(call, grpc.StatusCode.UNAVAILABLE)
        self.assertTrue(error.details().startswith(
            "this attempt applied nothing, but an earlier attempt of this call may have committed"), error.details())
        self.assertEqual(self.details(error), {})
        self.assertEqual(self.state.operation_log[-1], dict(method="UpdateLayer", outcome="OVERTAKEN"))

    async def test_an_unchained_call_applies_nothing_after_its_writers_next_commit(self):
        # The call read its start record, op-1, when it was received; once the writer's record moves, no attempt of it
        # applies, and the service cannot tell whether an earlier attempt of it committed.
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        hold = self.state.plan_write("hold", position=2)
        held = self.start(object_id, "op-2", value=2)
        await asyncio.wait_for(hold.reached.wait(), WATCHDOG)
        self.assertEqual((await self.write(object_id, "op-3", value=3)).sequence_num, 3)
        hold.release.set()
        await self.overtaken(held)
        self.assertEqual([payload for _, payload in self.writes()], [self.payload(1), self.payload(3)])

    async def test_a_stale_commit_step_after_its_writers_next_commit_applies_nothing(self):
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        stale = self.state.plan_write("stale", position=2)
        await self.refused(self.write(object_id, "op-2", value=2), grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await self.write(object_id, "op-2", value=2)).sequence_num, 3)  # The re-send applies.
        self.assertEqual((await self.write(object_id, "op-3", value=3)).sequence_num, 4)
        stale.release.set()
        await asyncio.wait_for(stale.done.wait(), WATCHDOG)
        self.assertEqual(stale.outcome, "UNAVAILABLE")
        self.assertEqual(len(self.writes()), 3)

    async def test_a_resend_whose_first_attempt_commits_meanwhile_is_answered_already_committed(self):
        # The own-key match precedes the start-record guard: the re-send's start record is op-1, and by its commit step
        # the record holds the re-sent operation itself, so it is answered from that commit.
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        first = self.state.plan_write("hold", position=2)
        attempt = self.start(object_id, "op-2", value=2)
        await asyncio.wait_for(first.reached.wait(), WATCHDOG)
        second = self.state.plan_write("hold", position=2)
        resend = self.start(object_id, "op-2", value=2)
        await asyncio.wait_for(second.reached.wait(), WATCHDOG)
        first.release.set()
        self.assertEqual((await asyncio.wait_for(attempt, WATCHDOG)).operation_outcome, pb.OPERATION_OUTCOME_APPLIED)
        second.release.set()
        reply = await asyncio.wait_for(resend, WATCHDOG)
        self.assertEqual((reply.sequence_num, reply.operation_outcome), (3, pb.OPERATION_OUTCOME_ALREADY_COMMITTED))
        self.assertEqual(len(self.writes()), 2)

    async def test_a_checkpoint_restore_overtakes_an_unchained_call_in_flight(self):
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        hold = self.state.plan_write("hold", position=2)
        held = self.start(object_id, "op-2", value=2)
        await asyncio.wait_for(hold.reached.wait(), WATCHDOG)
        self.assertEqual(self.state.restore_checkpoint(), 1)
        hold.release.set()
        await self.overtaken(held)
        self.assertEqual(len(self.writes()), 1)

    async def test_a_start_record_read_after_the_next_commit_is_not_guarded(self):
        # The documented gap: a call whose start-record read runs only after its writer's next commit reads that
        # commit as its start, so it applies once more. The service drops a call still waiting at its deadline
        # before that read, which the release with a code models.
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        for dropped in (False, True):
            with self.subTest(dropped=dropped):
                base = len(self.writes())
                delay = self.state.plan_write("delay_start", position=base + 1)
                late = self.start(object_id, f"late-{dropped}", value=2)
                await asyncio.wait_for(delay.reached.wait(), WATCHDOG)
                self.assertEqual((await self.write(object_id, f"late-{dropped}", value=2)).operation_outcome,
                                 pb.OPERATION_OUTCOME_APPLIED)
                await self.write(object_id, f"next-{dropped}", value=3)
                if dropped:
                    delay.code = grpc.StatusCode.DEADLINE_EXCEEDED
                    delay.release.set()
                    await self.refused(late, grpc.StatusCode.DEADLINE_EXCEEDED)
                    self.assertEqual(len(self.writes()), base + 2)
                else:
                    delay.release.set()
                    reply = await asyncio.wait_for(late, WATCHDOG)
                    self.assertEqual(reply.operation_outcome, pb.OPERATION_OUTCOME_APPLIED)
                    self.assertEqual([payload for _, payload in self.writes()][-3:],
                                     [self.payload(2), self.payload(3), self.payload(2)])

    async def test_a_detached_start_record_read_obeys_its_release_and_deadline(self):
        object_id = await self.create()
        await self.write(object_id, "op-1", "", value=1)
        for dropped in (False, True):
            with self.subTest(dropped=dropped):
                base = len(self.writes())
                delay = self.state.plan_write("stale_start", position=base + 1)
                operation = f"detached-{dropped}"
                await self.refused(self.write(object_id, operation, value=2), grpc.StatusCode.UNAVAILABLE)
                self.assertTrue(delay.reached.is_set())
                self.assertFalse(delay.done.is_set())
                self.assertEqual(len(self.writes()), base)
                self.assertEqual((await self.write(object_id, operation, value=2)).operation_outcome,
                                 pb.OPERATION_OUTCOME_APPLIED)
                await self.write(object_id, f"next-detached-{dropped}", value=3)
                if dropped:
                    delay.code = grpc.StatusCode.DEADLINE_EXCEEDED
                delay.release.set()
                await asyncio.wait_for(delay.done.wait(), WATCHDOG)
                self.assertEqual(delay.outcome, "DEADLINE_EXCEEDED" if dropped else "APPLIED")
                self.assertEqual(len(self.writes()), base + (2 if dropped else 3))
                if not dropped:
                    self.assertEqual([payload for _, payload in self.writes()][-3:],
                                     [self.payload(2), self.payload(3), self.payload(2)])

    async def test_ping_reports_the_capabilities(self):
        for operation_ids, chained_writes in ((True, True), (True, False), (False, True)):
            with self.subTest(operation_ids=operation_ids, chained_writes=chained_writes):
                self.state.operation_ids, self.state.chained_writes = operation_ids, chained_writes
                reply = await self.stub.Ping(pb.PingRequest(client_timestamp_microseconds_since_epoch=7),
                                             metadata=AUTH, timeout=WATCHDOG)
                self.assertEqual((reply.operation_ids, reply.chained_writes),
                                 (operation_ids, operation_ids and chained_writes))

    async def test_server_without_chaining_refuses_a_chained_header_before_its_chunks(self):
        # A server or replica that does not know the chained header sees a call without a header: it refuses it
        # INVALID_ARGUMENT, with no ErrorInfo, and applies nothing. A pre-chaining replica of a chaining service
        # does so although the service's Ping reports chained writes.
        object_id = await self.create()
        for mode in ("operation_ids", "chained_writes", "pre_chaining"):
            with self.subTest(mode=mode):
                self.state.operation_ids = mode != "operation_ids"
                self.state.chained_writes = mode != "chained_writes"
                self.state.pre_chaining = mode == "pre_chaining"
                call = self.stub.UpdateLayer(metadata=AUTH, timeout=WATCHDOG)
                await call.write(self.header(object_id, "op-1", ""))
                # The stream stays open: the refusal cannot have waited for the chunks.
                error = await self.refused(call, grpc.StatusCode.INVALID_ARGUMENT)
                self.assertEqual(self.details(error), {})
        self.assertEqual(self.writes(), [])
        self.state.pre_chaining = False
        reply = await self.write(object_id, "op-1", "", value=1)
        self.assertEqual((reply.sequence_num, reply.operation_outcome), (2, pb.OPERATION_OUTCOME_APPLIED))


class OpaqueChainedWriteTests(ChainedWriteCases, unittest.IsolatedAsyncioTestCase):
    state_type, server_type = State, ReferenceServer

    def payload(self, value):
        return b"value-%d" % value


class GraphChainedWriteTests(ChainedWriteCases, unittest.IsolatedAsyncioTestCase):
    state_type, server_type = GraphState, GraphServer

    def payload(self, value):
        return encode(Delta(fields=(Field(1, "counter", value),)))


if __name__ == "__main__":
    unittest.main()
