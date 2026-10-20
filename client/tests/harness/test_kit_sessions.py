# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session UI state driven by explicit completions and a manual polling clock."""
import asyncio
from enum import IntEnum
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


class Result(IntEnum):
    OK = 0
    NOT_INITIALIZED = 3
    PERMISSION_DENIED = 5
    TIMEOUT = 9
    RESOURCE_EXHAUSTED = 13
    BUSY = 14
    OUTCOME_UNKNOWN = 18
    INTERNAL_ERROR = 20


class LiveError(RuntimeError):
    def __init__(self, result):
        self.result = result


class ManualClock:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.timers = []
        self.reject_delivery = False

    def create_future(self):
        return self.loop.create_future()

    def call_soon_threadsafe(self, callback, *args):
        if self.reject_delivery:
            raise RuntimeError("controlled loop teardown")
        return self.loop.call_soon_threadsafe(callback, *args)

    def call_later(self, delay, callback):
        handle = SimpleNamespace(delay=delay, callback=callback, cancelled=False)
        handle.cancel = lambda: setattr(handle, "cancelled", True)
        self.timers.append(handle)
        return handle

    def fire(self):
        handle = self.timers.pop(0)
        if not handle.cancelled:
            handle.callback()


class Request:
    def __init__(self, callback):
        self.callback = callback
        self.done = False
        self.outcome = None
        self.close = Mock(side_effect=lambda timeout: Result.OK if self.done else Result.TIMEOUT)

    def complete(self, *outcome):
        self.outcome = outcome if len(outcome) > 1 else outcome[0]
        self.done = True
        self.callback(*outcome)

    def wait(self, timeout):
        assert timeout == 0
        return Result.OK if self.done else Result.TIMEOUT


class SessionCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.calls = []
        self.api = ModuleType("kit_catalog_fixture")
        self.api.Result, self.api.LiveError = Result, LiveError
        self.api.make_layer_uri = lambda uri, name: uri + "/layer/" + name
        def start(kind, *args):
            request = Request(args[-1])
            self.calls.append((kind, args[:-1], request))
            return request
        self.api.list_sessions_with_callback = lambda *args: start("list", *args)
        self.api.create_session_with_callback = lambda *args: start("create", *args)
        self.owner = SimpleNamespace(_submit=lambda kind, args, callback, internal: start(kind, *args, callback))
        self.endpoint = SimpleNamespace(_runtime=self.owner)
        name = self.api.__name__ + "._kit_sessions"
        spec = importlib.util.spec_from_file_location(name, ROOT / "source/bindings/python/omni/usd/live/_kit_sessions.py")
        self.module = importlib.util.module_from_spec(spec)
        modules = patch.dict(sys.modules, {self.api.__name__: self.api, name: self.module})
        modules.start(); self.addCleanup(modules.stop)
        spec.loader.exec_module(self.module)
        self.clock = ManualClock()
        self.catalog = self.new_catalog()

    def new_catalog(self, **options):
        return self.module.SessionCatalog(self.endpoint, "s3://bucket/asset.usd", current_user="alice",
                                         loop=self.clock, **options)

    @staticmethod
    def created(name="opaque/id"):
        return SimpleNamespace(session_id=name, uri="live://host/session/opaque%2Fid")

    def listed(self, creator="alice"):
        return SimpleNamespace(**vars(self.created()), name="authoritative name", created_by=creator, created_time_ms=123)

    async def finish(self, operation, *result):
        operation.request.complete(*result)
        return await operation.wait()

    async def test_confirmed_create_survives_failed_or_absent_metadata_without_recreate(self):
        operation = self.catalog.create("local label")
        self.assertEqual(self.calls[0][1], (self.endpoint, "s3://bucket/asset.usd", "local label"))
        self.assertEqual(await self.finish(operation, Result.OK, self.created()), (Result.OK, self.created()))
        self.assertFalse(self.catalog.can_merge("opaque/id"))
        for result, rows in ((Result.PERMISSION_DENIED, []), (Result.OK, [])):
            await self.finish(self.catalog.refresh(), result, rows)
            self.assertEqual(self.catalog.rows[0].session_id, "opaque/id")
            self.assertIsNone(self.catalog.rows[0].created_by)
            self.assertFalse(self.catalog.can_merge("opaque/id"))
        await self.finish(self.catalog.refresh(), Result.OK, [self.listed()])
        self.assertTrue(self.catalog.can_merge("opaque/id"))
        self.assertEqual(self.catalog.rows[0].name, "authoritative name")
        self.assertEqual(self.catalog.rows[0].root_uri, self.created().uri + "/layer/root.live")
        await self.finish(self.catalog.refresh(), Result.OK, [self.listed("bob")])
        self.assertFalse(self.catalog.can_merge("opaque/id"))
        self.assertEqual([c[0] for c in self.calls].count("create"), 1)
        self.assertEqual(self.catalog.close(), Result.BUSY)
        self.assertEqual(self.catalog.acknowledge(operation), Result.OK)
        self.assertEqual(self.catalog.close(), Result.OK)

    async def test_cancelled_waiter_does_not_cancel_or_forget_mutation(self):
        operation = self.catalog.create("one request")
        entered = asyncio.Event()
        async def wait():
            entered.set()
            return await operation.wait()
        waiter = asyncio.create_task(wait())
        await entered.wait()
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertEqual(self.catalog.close(), Result.BUSY)
        operation.request.close.assert_not_called()
        self.assertEqual(await self.finish(operation, Result.OK, self.created()), (Result.OK, self.created()))
        self.assertEqual(self.catalog.operations, (operation,))
        self.assertEqual(self.catalog.acknowledge(operation), Result.OK)
        self.assertEqual(self.catalog.close(), Result.OK)

    async def test_unknown_outcome_retained_through_poll_and_close(self):
        operation = self.catalog.create("uncertain")
        await self.finish(operation, Result.OUTCOME_UNKNOWN, None)
        await self.finish(self.catalog.refresh(), Result.OK, [])
        self.assertEqual(self.catalog.acknowledge(operation), Result.BUSY)
        self.assertEqual(self.catalog.close(), Result.BUSY)
        self.assertEqual(operation.outcome, (Result.OUTCOME_UNKNOWN, None))
        self.assertEqual([c[0] for c in self.calls], ["create", "list"])

    async def test_polling_is_active_only_and_issued_requests_are_coalesced(self):
        self.catalog.set_active(True)
        first = self.catalog.refresh()
        self.assertIs(first, self.catalog.refresh())
        self.assertFalse(self.clock.timers)
        await self.finish(first, Result.OK, [])
        self.assertEqual(len(self.clock.timers), 1)
        self.assertEqual(self.clock.timers[0].delay, 2.0)
        self.clock.fire()
        second = self.catalog.refresh()
        self.catalog.set_active(False)
        await self.finish(second, Result.OK, [])
        self.assertFalse(self.clock.timers)
        self.assertEqual([c[0] for c in self.calls], ["list", "list"])
        self.catalog.set_active(True)
        await self.finish(self.catalog.refresh(), Result.OK, [])
        self.catalog.set_active(False)
        self.clock.fire()
        self.assertEqual(len(self.calls), 3)

    async def test_delete_stays_with_endpoint_owner_and_retains_row_until_confirmed(self):
        await self.finish(self.catalog.refresh(), Result.OK, [self.listed()])
        operation = self.catalog.delete("opaque/id")
        self.assertEqual(self.calls[-1][:2], ("delete_session", (self.created().uri,)))
        self.assertEqual(len(self.catalog.rows), 1)
        await self.finish(operation, Result.PERMISSION_DENIED)
        self.assertEqual(len(self.catalog.rows), 1)
        self.assertEqual(self.catalog.acknowledge(operation), Result.OK)
        operation = self.catalog.delete("opaque/id")
        await self.finish(operation, Result.OK)
        self.assertEqual(self.catalog.rows, ())

    async def test_bounds_reject_before_mutation_and_failed_snapshot_is_atomic(self):
        catalog = self.new_catalog(max_sessions=1, max_operations=1)
        operation = catalog.create("first")
        with self.assertRaises(LiveError) as busy:
            catalog.refresh()
        self.assertEqual(busy.exception.result, Result.BUSY)
        await self.finish(operation, Result.OK, self.created())
        with self.assertRaises(LiveError) as full:
            catalog.create("second")
        self.assertEqual(full.exception.result, Result.RESOURCE_EXHAUSTED)
        with self.assertRaises(LiveError):
            catalog.delete("opaque/id")
        before = catalog.rows
        other = self.listed(); other.session_id = "another"
        listing = catalog.refresh()
        await self.finish(listing, Result.OK, [self.listed(), other])
        self.assertEqual(listing.projection_result, Result.RESOURCE_EXHAUSTED)
        self.assertEqual(catalog.metadata_result, Result.RESOURCE_EXHAUSTED)
        self.assertEqual(catalog.rows, before)
        self.assertEqual(len(self.calls), 2)

    async def test_loop_delivery_failure_still_allows_direct_result_recovery(self):
        operation = self.catalog.create("recover receipt")
        self.clock.reject_delivery = True
        operation.request.complete(Result.OK, self.created())
        self.assertEqual(self.catalog.close(), Result.BUSY)
        self.assertEqual(operation.outcome, (Result.OK, self.created()))
        self.assertEqual(self.catalog.acknowledge(operation), Result.OK)
        self.assertEqual(self.catalog.close(), Result.OK)

    async def test_invalid_host_keys_and_off_thread_calls_issue_nothing(self):
        for key in ("", "s3://bucket/scene?signature=secret", "scene#part", "scene\x00"):
            with self.assertRaises(ValueError):
                self.module.SessionCatalog(self.endpoint, key, current_user="alice", loop=self.clock)
        observed = []
        def wrong_thread():
            try:
                self.catalog.create("bad thread")
            except LiveError as error:
                observed.append(error.result)
        thread = threading.Thread(target=wrong_thread)
        thread.start(); thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(observed, [Result.BUSY])
        self.assertFalse(self.calls)

    async def test_admitted_create_with_interrupted_request_return_keeps_late_receipt(self):
        original = self.api.create_session_with_callback
        def interrupted(*args):
            original(*args)
            raise RuntimeError("interrupted after admission, before Request return")
        self.api.create_session_with_callback = interrupted
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.catalog.create("exactly once")
        operation, = self.catalog.operations
        self.assertIsNone(operation.request)
        self.assertEqual(operation.outcome, (Result.OUTCOME_UNKNOWN, None))
        self.assertEqual(self.catalog.close(), Result.BUSY)
        self.calls[0][2].complete(Result.OK, self.created())
        self.catalog.reconcile()
        self.assertEqual(operation.outcome, (Result.OK, self.created()))
        self.assertEqual((await operation.wait())[0], Result.OK)
        self.assertEqual(self.catalog.rows[0].session_id, "opaque/id")
        self.assertEqual(self.catalog.acknowledge(operation), Result.OK)
        self.assertEqual(self.catalog.close(), Result.OK)
        self.assertEqual([c[0] for c in self.calls], ["create"])

    async def test_local_projection_failure_preserves_confirmed_service_receipt(self):
        operation = self.catalog.create("confirmed but cannot construct row")
        self.api.make_layer_uri = Mock(side_effect=MemoryError("controlled projection failure"))
        self.assertEqual(await self.finish(operation, Result.OK, self.created()), (Result.OK, self.created()))
        self.assertEqual(operation.projection_result, Result.INTERNAL_ERROR)
        self.assertEqual(self.catalog.metadata_result, Result.INTERNAL_ERROR)
        self.assertEqual(self.catalog.rows, ())
        self.assertEqual(self.catalog.close(), Result.BUSY)
        self.assertEqual(self.catalog.acknowledge(operation), Result.OK)
        self.assertEqual(self.catalog.close(), Result.OK)
        self.assertEqual(len(self.calls), 1)

    async def test_receipt_reservation_failure_issues_nothing_and_does_not_block_close(self):
        class Exhausted(list):
            def append(self, operation):
                raise MemoryError("controlled receipt reservation failure")
        self.catalog._operations = Exhausted()
        with self.assertRaises(MemoryError):
            self.catalog.create("never issued")
        self.assertFalse(self.calls)
        self.assertEqual(self.catalog.operations, ())
        self.assertEqual(self.catalog.close(), Result.OK)

    async def test_view_observer_receives_host_state_and_pending_delete_before_admission(self):
        observed = []
        def project(catalog):
            observed.append((threading.get_ident(), catalog.can_merge("opaque/id"), len(self.calls)))
            return Result.OK
        self.catalog = self.new_catalog(on_change=project)
        await self.finish(self.catalog.refresh(), Result.OK, [self.listed()])
        self.assertEqual(observed[-1][:2], (threading.get_ident(), True))
        operation = self.catalog.delete("opaque/id")
        self.assertEqual(observed[-1][1:], (False, 1))
        self.assertEqual(len(self.calls), 2)
        await self.finish(operation, Result.OK)

    async def test_view_failure_preserves_confirmed_receipt_and_reprojects_without_rpc(self):
        observer = Mock(return_value=Result.OK)
        self.catalog = self.new_catalog(on_change=observer)
        operation = self.catalog.create("confirmed")
        observer.side_effect = MemoryError("native row projection")
        self.assertEqual(await self.finish(operation, Result.OK, self.created()), (Result.OK, self.created()))
        self.assertEqual(self.catalog.view_result, Result.RESOURCE_EXHAUSTED)
        self.assertEqual(operation.outcome, (Result.OK, self.created()))
        observer.side_effect = None
        self.assertEqual(self.catalog.refresh_projection(), Result.OK)
        self.assertEqual([call[0] for call in self.calls], ["create"])

    async def test_failed_view_prevents_mutation_admission_and_can_be_acknowledged(self):
        self.catalog = self.new_catalog(on_change=lambda catalog: Result.INTERNAL_ERROR)
        operation = self.catalog.create("not issued")
        self.assertEqual(await operation.wait(), (Result.INTERNAL_ERROR, None))
        self.assertIsNone(operation.request)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.catalog.acknowledge(operation), Result.OK)
        self.assertEqual(self.catalog.close(), Result.OK)

    async def test_view_observer_cannot_reenter_mutations_or_close(self):
        def project(catalog):
            for action in (lambda: catalog.create("recursive"), catalog.close,
                           lambda: catalog.set_active(False), catalog.refresh_projection,
                           catalog.reconcile, catalog.refresh,
                           lambda: catalog.delete("opaque/id"),
                           lambda: catalog.acknowledge(catalog.operations[0])):
                with self.assertRaises(LiveError) as error:
                    action()
                self.assertEqual(error.exception.result, Result.BUSY)
            return Result.OK
        self.catalog = self.new_catalog(on_change=project)
        operation = self.catalog.create("one")
        await self.finish(operation, Result.OK, self.created())
        self.assertEqual(len(self.calls), 1)

    async def test_view_cannot_reconcile_delivered_receipt_during_interrupted_admission(self):
        observations = []
        def project(catalog):
            if catalog.operations and catalog.operations[0]._provisional:
                operation = catalog.operations[0]
                self.assertIsNotNone(operation._delivery)
                with self.assertRaises(LiveError) as error:
                    catalog.reconcile()
                self.assertEqual(error.exception.result, Result.BUSY)
                observations.append(operation.outcome)
            return Result.OK
        self.catalog = self.new_catalog(on_change=project)
        original = self.api.create_session_with_callback
        def interrupted(*args):
            request = original(*args)
            request.complete(Result.OK, self.created())
            raise RuntimeError("interrupted after delivery")
        self.api.create_session_with_callback = interrupted
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.catalog.create("one confirmed create")
        self.assertEqual(observations, [(Result.OUTCOME_UNKNOWN, None)])
        operation, = self.catalog.operations
        self.assertEqual(operation.outcome, (Result.OK, self.created()))
        self.assertEqual(len(self.calls), 1)
