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
from dataclasses import replace

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
        self.owners, self.release_events, self.extra_servers = [], [], []
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
        for server in self.extra_servers:
            await server.stop()
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

    def owner_events(self, events):
        return [(e.merge_owner_outcome, e.connection_id, e.merge_owner_epoch, e.merge_owner_held)
                for e in events if e.kind == live.SessionEventKind.MERGE_OWNER]

    async def owner_join(self, app, runtime=None, *, uri=None, merge_owner=True):
        events, callback = self.observer()
        session = (runtime or self.runtime).join_session(uri or self.created.uri, app, callback,
                                                         merge_owner=merge_owner)
        return session, events, await self.connected(events)

    async def test_merge_owner_grants_one_of_two_same_principal_joins_and_hands_off_when_the_holder_leaves(self):
        self.state.merge_owner = True
        a, a_events, a_connected = await self.owner_join("first")
        b, b_events, b_connected = await self.owner_join("second")
        self.assertTrue(a_connected.merge_owner_supported and b_connected.merge_owner_supported)
        self.assertEqual(a.merge_owner, live.MergeOwner(True, False, False, 0))
        answers = await asyncio.gather(a.request_merge_owner_async(live.MergeOwnerAction.CLAIM),
                                       b.request_merge_owner_async(live.MergeOwnerAction.CLAIM))
        self.assertEqual([result for result, _ in answers], [live.Result.OK, live.Result.OK])
        outcomes = sorted(state.outcome for _, state in answers)
        self.assertEqual(outcomes, [live.MergeOwnerOutcome.GRANTED, live.MergeOwnerOutcome.HELD_BY_OTHER])
        (holder, holder_events, held), (waiter, waiter_events, waiting) = sorted(
            [(a, a_events, answers[0][1]), (b, b_events, answers[1][1])],
            key=lambda entry: entry[2].outcome != live.MergeOwnerOutcome.GRANTED)
        holder_id = held.owner_connection_id
        self.assertEqual((held.epoch, held.held, held.owner_user_name), (1, True, "writer"))
        self.assertEqual((waiting.owner_connection_id, waiting.epoch, waiting.held), (holder_id, 1, False))
        self.assertEqual(holder.merge_owner, live.MergeOwner(True, True, True, 1))
        self.assertEqual(waiter.merge_owner, live.MergeOwner(True, False, True, 1))
        self.assertEqual(await holder.request_merge_owner_async(live.MergeOwnerAction.VERIFY),
                         (live.Result.OK, live.MergeOwnerState(live.MergeOwnerOutcome.GRANTED, holder_id, "writer", 1,
                                                                True, True)))
        # The holder leaves: its join's end releases the lease, and the waiter, whose claim still stands, takes it
        # at once with a higher epoch.
        self.assertEqual(await asyncio.to_thread(holder.leave, 3000), live.Result.OK)
        waiter_id = next(e.connection_id for e in waiter_events if e.kind == live.SessionEventKind.CONNECTION)
        await self.until(lambda: (live.MergeOwnerOutcome.GRANTED, waiter_id, 2, True) in self.owner_events(waiter_events))
        owner_events = self.owner_events(waiter_events)
        released = owner_events.index((live.MergeOwnerOutcome.RELEASED, None, 1, False))
        self.assertEqual(owner_events[released + 1], (live.MergeOwnerOutcome.GRANTED, waiter_id, 2, True))
        self.assertEqual(waiter.merge_owner, live.MergeOwner(True, True, True, 2))
        result, state = await waiter.request_merge_owner_async(live.MergeOwnerAction.CLAIM)
        self.assertEqual((result, state.outcome, state.epoch, state.held), (live.Result.OK, live.MergeOwnerOutcome.GRANTED,
                                                                            2, True))
        grants = [e for e in self.state.merge_owner_log if e["outcome"] == "MERGE_OWNER_OUTCOME_GRANTED"]
        self.assertEqual([(e["owner"], e["epoch"]) for e in grants if e["request_id"]],
                         [(holder_id, 1), (holder_id, 1), (waiter_id, 2), (waiter_id, 2)])

    async def test_merge_owner_keepalive_expiry_releases_and_a_reconnect_reclaims_after_a_handoff(self):
        self.state.merge_owner = True
        self.state.keepalive_interval_ms = 3_600_000
        a, a_events, old = await self.owner_join("lapsing")
        b, b_events, _ = await self.owner_join("watching")
        result, state = await a.request_merge_owner_async(live.MergeOwnerAction.CLAIM)
        self.assertEqual((result, state.outcome, state.epoch), (live.Result.OK, live.MergeOwnerOutcome.GRANTED, 1))
        await self.until(lambda: self.owner_events(b_events) == [(live.MergeOwnerOutcome.HELD_BY_OTHER,
                                                                   old.connection_id, 1, False)])
        # The lapsing join's rejoin waits at a pause gate, so the handoff happens before its re-claim.
        rejoin = self.state.plan("JoinSession", "before", pause=True)
        self.assertEqual(self.state.advance_keepalive_clock(3 * 3_600_000 - 1), [])
        self.assertEqual(await b.send_message_async(b"still here"), live.Result.OK)
        await self.until(lambda: any(e.kind == live.SessionEventKind.MESSAGE for e in a_events))
        self.assertEqual(self.state.advance_keepalive_clock(1), [old.connection_id])
        await self.until(lambda: (live.MergeOwnerOutcome.RELEASED, None, 1, False) in self.owner_events(b_events))
        # Reconnecting drops ownership at once.
        await self.until(lambda: any(e.kind == live.SessionEventKind.CONNECTION and e.result == live.Result.UNAVAILABLE
                                     for e in a_events))
        lost = next(e for e in a_events if e.kind == live.SessionEventKind.CONNECTION and e.result != live.Result.OK)
        self.assertEqual((lost.merge_owner_supported, lost.merge_owner_held), (True, False))
        self.assertFalse(a.merge_owner.held)
        self.assertEqual((await a.request_merge_owner_async(live.MergeOwnerAction.VERIFY))[0], live.Result.UNAVAILABLE)
        await rejoin.reached.wait()
        result, state = await b.request_merge_owner_async(live.MergeOwnerAction.CLAIM)
        self.assertEqual((result, state.outcome, state.epoch), (live.Result.OK, live.MergeOwnerOutcome.GRANTED, 2))
        rejoin.release.set()
        current = await self.connected(a_events, 2)
        self.assertNotEqual(current.connection_id, old.connection_id)
        # The rejoined connection claims again on its own, and learns that the watcher holds the lease.
        await self.until(lambda: (live.MergeOwnerOutcome.HELD_BY_OTHER, state.owner_connection_id, 2, False)
                         in self.owner_events(a_events))
        self.assertEqual([(e["connection"], e["outcome"]) for e in self.state.merge_owner_log
                          if e["connection"] == current.connection_id and e["request_id"]],
                         [(current.connection_id, "MERGE_OWNER_OUTCOME_HELD_BY_OTHER")])
        # VERIFY after the handoff answers the old holder NOT_HELD.
        result, verified = await a.request_merge_owner_async(live.MergeOwnerAction.VERIFY)
        self.assertEqual((result, verified.outcome, verified.held), (live.Result.OK, live.MergeOwnerOutcome.NOT_HELD,
                                                                      False))
        self.assertEqual(a.merge_owner, live.MergeOwner(True, False, True, 2))
        self.state.assert_faults_fired()

    async def test_merge_owner_reconnect_drops_ownership_and_reclaims_on_the_new_connection(self):
        self.state.merge_owner = True
        a, a_events, old = await self.owner_join("rotating")
        result, state = await a.request_merge_owner_async(live.MergeOwnerAction.CLAIM)
        self.assertEqual((result, state.epoch, state.held), (live.Result.OK, 1, True))
        async with self.state.changed:
            peer = self.state.sessions[self.created.session_id].peers[old.connection_id]
            peer.close_code = grpc.StatusCode.UNAVAILABLE
            peer.ready.set()
        current = await self.connected(a_events, 2)
        self.assertNotEqual(current.connection_id, old.connection_id)
        await self.until(lambda: (live.MergeOwnerOutcome.GRANTED, current.connection_id, 2, True)
                         in self.owner_events(a_events))
        self.assertEqual(a.merge_owner, live.MergeOwner(True, True, True, 2))
        # A RELEASE ends the intent: the next connection does not claim again.
        result, released = await a.request_merge_owner_async(live.MergeOwnerAction.RELEASE)
        self.assertEqual((result, released.outcome), (live.Result.OK, live.MergeOwnerOutcome.RELEASED))
        async with self.state.changed:
            peer = self.state.sessions[self.created.session_id].peers[current.connection_id]
            peer.close_code = grpc.StatusCode.UNAVAILABLE
            peer.ready.set()
        third = await self.connected(a_events, 3)
        self.assertEqual(await a.request_merge_owner_async(live.MergeOwnerAction.VERIFY),
                         (live.Result.OK, live.MergeOwnerState(live.MergeOwnerOutcome.NOT_HELD, None, None, 2, True,
                                                                False)))
        self.assertEqual([e["request_id"] != 0 for e in self.state.merge_owner_log
                          if e["connection"] == third.connection_id], [True])

    async def test_merge_owner_refuses_a_non_creator_and_an_expired_credential_without_revoking_the_lease(self):
        self.state.merge_owner = True
        self.state.tokens["synthetic-other"] = Principal("other")
        self.state.sessions[self.created.session_id].writers.add("other")
        holder, _, _ = await self.owner_join("holder")
        # A second authority on the same service state carries the other principal's bearer.
        other = ReferenceServer(self.state, "python-other", service_factory=RecordedService)
        self.extra_servers.append(other)
        address = await other.start()
        self.runtime.register_endpoint(address, access_token="synthetic-other")
        stranger, _, _ = await self.owner_join("stranger", uri=live.make_session_uri(address, self.created.session_id))
        result, state = await stranger.request_merge_owner_async(live.MergeOwnerAction.CLAIM)
        self.assertEqual((result, state.outcome, state.owner_connection_id), (live.Result.OK,
                                                                             live.MergeOwnerOutcome.NOT_ELIGIBLE, None))
        result, state = await holder.request_merge_owner_async(live.MergeOwnerAction.CLAIM)
        self.assertEqual((result, state.outcome), (live.Result.OK, live.MergeOwnerOutcome.GRANTED))
        # Another join of the holder's principal, on a third authority with its own bearer, whose credential then
        # expires: its claim is refused UNAUTHENTICATED and the lease holds.
        self.state.issue("synthetic-late", "writer")
        late_server = ReferenceServer(self.state, "python-late", service_factory=RecordedService)
        self.extra_servers.append(late_server)
        late_address = await late_server.start()
        self.runtime.register_endpoint(late_address, access_token="synthetic-late")
        late, _, _ = await self.owner_join("late", uri=live.make_session_uri(late_address, self.created.session_id))
        self.state.tokens["synthetic-late"] = replace(self.state.tokens["synthetic-late"], expires_us=1)
        result, refused = await late.request_merge_owner_async(live.MergeOwnerAction.CLAIM)
        self.assertEqual((result, refused.outcome, refused.owner_connection_id),
                         (live.Result.OK, live.MergeOwnerOutcome.UNAUTHENTICATED, state.owner_connection_id))
        self.assertEqual((await holder.request_merge_owner_async(live.MergeOwnerAction.VERIFY))[1].outcome,
                         live.MergeOwnerOutcome.GRANTED)
        self.assertTrue(holder.merge_owner.held)
        self.assertEqual(await asyncio.to_thread(late.leave, 3000), live.Result.OK)

    async def test_merge_owner_claims_again_when_its_earlier_connection_leaves_presence_after_a_reconnect(self):
        # A half-open reconnect: the service still lists the join's earlier connection, so the new connection's
        # claim finds it holding; once the service drops it, the release that follows is claimed at once.
        self.state.merge_owner = True
        a, a_events, old = await self.owner_join("half-open")
        result, state = await a.request_merge_owner_async(live.MergeOwnerAction.CLAIM)
        self.assertEqual((result, state.outcome, state.epoch), (live.Result.OK, live.MergeOwnerOutcome.GRANTED, 1))
        async with self.state.changed:
            self.state.merge_owner_linger.add(old.connection_id)
            peer = self.state.sessions[self.created.session_id].peers[old.connection_id]
            peer.close_code = grpc.StatusCode.UNAVAILABLE
            peer.ready.set()
        current = await self.connected(a_events, 2)
        await self.until(lambda: (live.MergeOwnerOutcome.HELD_BY_OTHER, old.connection_id, 1, False)
                         in self.owner_events(a_events))
        self.assertIn(old.connection_id, self.state.sessions[self.created.session_id].peers)
        self.assertEqual(a.merge_owner, live.MergeOwner(True, False, True, 1))
        async with self.state.changed:
            self.assertTrue(self.state.evict(old.connection_id))
        await self.until(lambda: (live.MergeOwnerOutcome.GRANTED, current.connection_id, 2, True)
                         in self.owner_events(a_events))
        self.assertEqual(a.merge_owner, live.MergeOwner(True, True, True, 2))
        self.assertEqual([(e["request_id"], e["action"]) for e in self.state.merge_owner_requests
                          if e["connection"] == current.connection_id],
                         [(2, "MERGE_OWNER_ACTION_CLAIM"), (3, "MERGE_OWNER_ACTION_CLAIM")])

    async def test_merge_owner_request_in_flight_settles_unavailable_on_its_connections_end_and_is_not_resent(self):
        self.state.merge_owner = True
        a, a_events, old = await self.owner_join("pending")
        self.state.merge_owner_hold = True
        pending = a.request_merge_owner_async(live.MergeOwnerAction.VERIFY)
        await self.until(lambda: len(self.state.merge_owner_requests) == 1)
        self.assertFalse(pending.done())
        async with self.state.changed:
            peer = self.state.sessions[self.created.session_id].peers[old.connection_id]
            peer.close_code = grpc.StatusCode.UNAVAILABLE
            peer.ready.set()
        self.assertEqual(await pending, (live.Result.UNAVAILABLE, None))
        current = await self.connected(a_events, 2)
        self.state.merge_owner_hold = False
        result, verified = await a.request_merge_owner_async(live.MergeOwnerAction.VERIFY)
        self.assertEqual((result, verified.outcome), (live.Result.OK, live.MergeOwnerOutcome.NOT_HELD))
        # The new connection carried only the later request: the pending one was not sent again.
        self.assertEqual([(e["connection"], e["request_id"]) for e in self.state.merge_owner_requests],
                         [(old.connection_id, 1), (current.connection_id, 2)])

    async def test_merge_owner_request_in_flight_settles_cancelled_on_leave(self):
        self.state.merge_owner = True
        a, _, _ = await self.owner_join("leaving")
        self.state.merge_owner_hold = True
        pending = a.request_merge_owner_async(live.MergeOwnerAction.VERIFY)
        await self.until(lambda: len(self.state.merge_owner_requests) == 1)
        self.assertFalse(pending.done())
        self.assertEqual(await asyncio.to_thread(a.leave, 3000), live.Result.OK)
        self.assertEqual(await pending, (live.Result.CANCELLED, None))

    async def test_merge_owner_view_survives_into_the_terminal_delivery_failure_event(self):
        self.state.merge_owner = True
        entered, release = threading.Event(), threading.Event()
        self.release_events.append(release)
        events = []
        def held(event):
            events.append(event)
            if len(events) == 1:
                entered.set()
                release.wait()
        session = self.runtime.join_session(self.created.uri, "overflowing", held, max_events=1, max_bytes=4096,
                                            merge_owner=True)
        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
        identifier = events[0].connection_id
        async with self.state.changed:
            peer = self.state.sessions[self.created.session_id].peers[identifier]
            LiveService.emit_to(peer, pb.JoinSessionResponse(merge_owner=pb.MergeOwnerState(
                outcome=pb.MERGE_OWNER_OUTCOME_HELD_BY_OTHER, owner_connection_id="synthetic-holder",
                owner_user_name="writer", epoch=3)))
            LiveService.emit_to(peer, pb.JoinSessionResponse(peer_joined=pb.PeerJoined(
                connection_id="synthetic-peer", user_name="", user_application="")))
        await self.until(lambda: session.delivery_status == live.Result.RESOURCE_EXHAUSTED)
        release.set()
        await self.until(lambda: any(e.result == live.Result.RESOURCE_EXHAUSTED for e in events))
        self.assertEqual([e.kind for e in events], [live.SessionEventKind.CONNECTION,
                                                   live.SessionEventKind.MERGE_OWNER,
                                                   live.SessionEventKind.CONNECTION])
        terminal = events[-1]
        self.assertEqual((terminal.merge_owner_supported, terminal.merge_owner_held, terminal.merge_owner_epoch),
                         (True, False, 3))
        self.assertEqual(await asyncio.to_thread(session.leave, 3000), live.Result.OK)

    async def test_merge_owner_fails_open_on_a_service_without_it_and_sends_nothing(self):
        self.assertFalse(self.state.merge_owner)
        a, a_events, connected = await self.owner_join("asking")
        self.assertFalse(connected.merge_owner_supported)
        self.assertEqual(a.merge_owner, live.MergeOwner(False, False, False, 0))
        self.assertEqual(await a.request_merge_owner_async(live.MergeOwnerAction.CLAIM), (live.Result.UNSUPPORTED, None))
        self.assertEqual(await asyncio.to_thread(a.request_merge_owner, live.MergeOwnerAction.VERIFY),
                         (live.Result.UNSUPPORTED, None))
        self.assertEqual([(j["merge_owner_requested"], j["merge_owner"]) for j in self.state.joins], [(True, False)])
        self.assertEqual(self.state.merge_owner_log, [])
        self.assertEqual([e["kind"] for e in self.state.session_wire], ["header"])
        self.assertEqual(live.get_session_status(a).state, live.State.READY)

    async def test_merge_owner_frames_reach_only_joins_that_opt_in(self):
        self.state.merge_owner = True
        plain, plain_events, plain_connected = await self.owner_join("plain", merge_owner=False)
        self.assertFalse(plain_connected.merge_owner_supported)
        self.assertEqual(await plain.request_merge_owner_async(live.MergeOwnerAction.CLAIM), (live.Result.UNSUPPORTED, None))
        opted, opted_events, _ = await self.owner_join("opted")
        result, state = await opted.request_merge_owner_async(live.MergeOwnerAction.CLAIM)
        self.assertEqual((result, state.outcome), (live.Result.OK, live.MergeOwnerOutcome.GRANTED))
        self.assertEqual(await opted.send_message_async(b"after the grant"), live.Result.OK)
        await self.until(lambda: any(e.kind == live.SessionEventKind.MESSAGE for e in plain_events))
        self.assertEqual(self.owner_events(plain_events), [])
        self.assertEqual([(j["merge_owner_requested"], j["merge_owner"]) for j in self.state.joins],
                         [(False, False), (True, True)])
        # An arm 5 frame the join never negotiated is a protocol violation that fails it MALFORMED_STREAM, while
        # an opted-in join accepts the same frame.
        frame = pb.JoinSessionResponse(merge_owner=pb.MergeOwnerState(
            outcome=pb.MERGE_OWNER_OUTCOME_HELD_BY_OTHER, owner_connection_id=state.owner_connection_id,
            owner_user_name="writer", epoch=1))
        async with self.state.changed:
            for peer in self.state.sessions[self.created.session_id].peers.values():
                LiveService.emit_to(peer, frame)
        await self.until(lambda: any(e.kind == live.SessionEventKind.CONNECTION and
                                     e.result == live.Result.MALFORMED_STREAM for e in plain_events))
        self.assertEqual(live.get_session_status(plain).state, live.State.FAILED)
        await self.until(lambda: len(self.owner_events(opted_events)) == 2)
        self.assertEqual(live.get_session_status(opted).state, live.State.READY)
        self.assertEqual(await asyncio.to_thread(plain.leave, 3000), live.Result.OK)

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
        # The executor thread that ran next_event can still hold the session until it drops its work item, so the
        # abandonment can follow gc.collect(). The service sees the peer leave only after the orphan's stream ends,
        # which follows its abandonment; the drain after that waits for the orphan's reap to finish, so the runtime
        # closes with no session left.
        await self.until(lambda: identifier not in self.state.sessions[self.created.session_id].peers)
        self.assertEqual(await asyncio.to_thread(self.runtime._native.cleanup, 3000), live.Result.OK)

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
