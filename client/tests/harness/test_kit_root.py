# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root workflow ownership with real Python host wrappers and controlled native calls."""
import asyncio
from enum import IntEnum
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


class Result(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    NOT_INITIALIZED = 3
    NOT_FOUND = 4
    ALREADY_EXISTS = 5
    PERMISSION_DENIED = 7
    TIMEOUT = 9
    CANCELLED = 10
    MALFORMED_STREAM = 11
    RESOURCE_EXHAUSTED = 13
    BUSY = 14
    OUTCOME_UNKNOWN = 19
    INTERNAL_ERROR = 20
    IO_ERROR = 21


class LiveError(RuntimeError):
    def __init__(self, result):
        self.result = result


class RootTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.api = ModuleType('kit_root_fixture')
        self.api.__path__ = []
        self.api.Result, self.api.LiveError = Result, LiveError
        self.api.RequestError = SimpleNamespace
        self.api._local = SimpleNamespace()
        self.api._timeout = lambda value: value
        self.api._remaining = lambda deadline: 0
        self.api.make_layer_uri = lambda uri, name: uri + '/' + name
        def query(value):
            if value[0] != Result.OK:
                raise LiveError(value[0])
            return value[1]
        self.api._query = query
        self.api._record = lambda cls, value: cls(**value)
        modules = patch.dict(sys.modules, {self.api.__name__: self.api})
        modules.start(); self.addCleanup(modules.stop)
        self.loaded = {}
        for part in ('_host', '_kit_sessions', '_kit_root', '_kit_host'):
            name = self.api.__name__ + '.' + part
            spec = importlib.util.spec_from_file_location(name,
                ROOT / 'source/bindings/python/omni/usd/live' / (part + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            self.loaded[part] = module
        self.api.CreationPhase = self.loaded['_host'].CreationPhase
        self.host = self.loaded['_kit_host'].KitHost(lambda coroutine: None, max_session_roots=1)
        self.host.runtime = SimpleNamespace(close=Mock(return_value=Result.OK))
        self.context = self.loaded['_host'].PublicationContext(self.host.runtime, None, None)
        self.host.publication_context = self.context
        self.native_owner = SimpleNamespace(start=Mock(return_value=Result.OK), wait=Mock(return_value=Result.OK),
            close=Mock(return_value=Result.OK), prepare_layer=Mock(return_value=Result.OK),
            activate=Mock(return_value=Result.OK), layer=Mock(return_value=(Result.OK, object())))
        self.outcome = dict(phase=2, result=0, error={}, subject_id=12, started=True, closing=False, quiescent=True)
        self.native_owner.outcome = Mock(side_effect=lambda: (Result.OK, self.outcome))
        self.context._native = SimpleNamespace(allocate_creation=Mock(return_value=(Result.OK, self.native_owner)),
            prepare_creation=Mock(return_value=Result.OK))
        self.catalog = self.loaded['_kit_sessions'].SessionCatalog(object(), 'stable-key', current_user='alice')
        self.host._catalogs.append(self.catalog)
        self.operation = self.receipt('one')

    def receipt(self, name, result=Result.OK):
        operation = self.loaded['_kit_sessions'].SessionOperation(self.catalog, 'create', name)
        operation._outcome = result, SimpleNamespace(session_id=name, uri='live://host/' + name)
        self.catalog._operations.append(operation)
        return operation

    def prepare(self, operation=None, seed=None):
        return self.host.prepare_session_root(self.catalog, operation or self.operation, seed or object())

    def test_confirmed_receipt_binds_one_seed_and_one_attempt(self):
        seed = object()
        root = self.prepare(seed=seed)
        self.assertEqual(root.root_uri, 'live://host/one/root.live')
        self.assertIs(self.prepare(seed=object()), root)
        self.context._native.prepare_creation.assert_called_once_with(self.native_owner, seed, root.root_uri)
        self.assertEqual(root.start(), Result.OK)
        self.assertEqual(root.start(), Result.ALREADY_EXISTS)
        self.native_owner.start.assert_called_once()
        self.assertEqual(self.catalog.operations, (self.operation,))
        self.assertEqual(root.activate(), Result.OK)
        self.assertIs(root.layer, self.native_owner.layer.return_value[1])
        self.assertEqual(root.close(), Result.OK)
        self.assertIs(self.prepare(), root)
        self.assertEqual(root.start(), Result.NOT_INITIALIZED)
        self.assertEqual(self.catalog.operations, (self.operation,))

    def test_native_preparation_interruption_preserves_context_and_root_association(self):
        def interrupt(owner, seed, uri):
            root, = self.host.session_roots
            self.assertIs(root.creation, self.context.creations[0])
            self.assertTrue(root.creation._admitting)
            self.assertEqual(root.creation.close(), Result.BUSY)
            with self.assertRaises(LiveError) as error:
                root.start()
            self.assertEqual(error.exception.result, Result.BUSY)
            raise KeyboardInterrupt('after native preparation boundary')
        self.context._native.prepare_creation.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.prepare()
        root, = self.host.session_roots
        self.assertIs(self.prepare(seed=object()), root)
        self.assertEqual(root.preparation_result, Result.INTERNAL_ERROR)
        self.assertEqual(root.start(), Result.INTERNAL_ERROR)
        self.assertFalse(root.creation._admitting)
        self.assertIs(root.creation, self.context.creations[0])
        self.assertEqual(root.close(), Result.OK)
        self.native_owner.start.assert_not_called()

    def test_preparation_allocation_failure_has_discoverable_nonissuing_root(self):
        self.context._native.allocate_creation.side_effect = MemoryError('native owner wrapper')
        with self.assertRaises(MemoryError):
            self.prepare()
        root, = self.host.session_roots
        self.assertIsNone(root.creation)
        self.assertEqual(root.start(), Result.RESOURCE_EXHAUSTED)
        self.assertIs(self.prepare(), root)
        self.assertEqual(root.close(), Result.OK)
        self.context._native.allocate_creation.assert_called_once()

    def test_start_delivery_interruption_cannot_repeat_native_mutation(self):
        root = self.prepare()
        self.native_owner.start.side_effect = KeyboardInterrupt('issued before return')
        with self.assertRaises(KeyboardInterrupt):
            root.start()
        self.assertEqual(root.start(), Result.ALREADY_EXISTS)
        self.native_owner.start.assert_called_once()
        self.assertEqual(root.creation.outcome.subject_id, 12)
        self.assertEqual(root.activate(), Result.OK)
        self.assertEqual(root.close(), Result.OK)

    async def test_cancelled_waiters_share_retained_native_wait_and_keep_close_busy(self):
        root = self.prepare()
        self.assertEqual(root.start(), Result.OK)
        entered, release = asyncio.Event(), asyncio.Event()
        async def worker(action, timeout):
            entered.set()
            await release.wait()
            return action(timeout)
        with patch.object(self.loaded['_kit_root'].asyncio, 'to_thread', side_effect=worker) as off_host:
            waiter = asyncio.create_task(root.wait(5000))
            await entered.wait()
            original = root._wait_task
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            self.assertFalse(original.cancelled())
            self.assertEqual(root.close(), Result.BUSY)
            self.assertEqual(root.activate(), Result.BUSY)
            second = asyncio.create_task(root.wait(9999))
            release.set()
            self.assertEqual(await second, Result.OK)
            self.assertIs(root._wait_task, original)
            off_host.assert_called_once()
            self.native_owner.wait.assert_called_once_with(5000)
        self.assertEqual(root.close(), Result.OK)

    def test_unknown_root_retains_both_receipts_and_cannot_activate(self):
        root = self.prepare()
        root.start()
        self.outcome.update(phase=5, result=19)
        self.assertEqual(root.activate(), Result.OUTCOME_UNKNOWN)
        self.assertEqual(root.result, Result.OUTCOME_UNKNOWN)
        self.native_owner.prepare_layer.assert_not_called()
        self.native_owner.activate.assert_not_called()
        self.assertEqual(root.start(), Result.ALREADY_EXISTS)
        self.assertEqual(self.catalog.operations, (self.operation,))
        self.assertEqual(root.creation.outcome.result, Result.OUTCOME_UNKNOWN)

    def test_invalid_or_foreign_receipts_and_capacity_reject_before_preparation(self):
        unknown = self.receipt('unknown', Result.OUTCOME_UNKNOWN)
        with self.assertRaises(LiveError) as error:
            self.prepare(unknown)
        self.assertEqual(error.exception.result, Result.OUTCOME_UNKNOWN)
        foreign = self.loaded['_kit_sessions'].SessionOperation(self.catalog, 'create', 'foreign')
        foreign._outcome = self.operation.outcome
        with self.assertRaises(ValueError):
            self.prepare(foreign)
        self.context._native.prepare_creation.assert_not_called()
        root = self.prepare()
        with self.assertRaises(LiveError) as error:
            self.prepare(self.receipt('second'))
        self.assertEqual(error.exception.result, Result.RESOURCE_EXHAUSTED)
        self.context._native.prepare_creation.assert_called_once()
        root.close()

    def test_host_close_preserves_root_even_after_explicit_session_acknowledgement(self):
        root = self.prepare()
        self.assertEqual(self.catalog.acknowledge(self.operation), Result.OK)
        with patch.object(self.context, 'close', return_value=Result.OK):
            self.assertEqual(self.host.close(), Result.BUSY)
            self.host.runtime.close.assert_not_called()
            self.native_owner.close.assert_not_called()
            self.assertEqual(root.close(), Result.OK)
            self.assertEqual(self.host.close(), Result.OK)

    def test_observer_cannot_prepare_a_root_reentrantly(self):
        def observer(catalog):
            with self.assertRaises(LiveError) as error:
                self.prepare()
            self.assertEqual(error.exception.result, Result.BUSY)
            return Result.OK
        self.catalog._on_change = observer
        self.assertEqual(self.catalog.refresh_projection(), Result.OK)
        self.context._native.prepare_creation.assert_not_called()

    def test_interruption_after_retention_cannot_allocate_another_workflow(self):
        class InterruptedList(list):
            def append(self, item):
                super().append(item)
                raise KeyboardInterrupt("retained before receipt association")
        self.host._session_roots = InterruptedList()
        with self.assertRaises(KeyboardInterrupt):
            self.prepare()
        root, = self.host.session_roots
        self.assertIsNone(self.operation._root)
        self.assertIs(self.prepare(), root)
        self.assertIs(self.operation._root, root)
        self.context._native.prepare_creation.assert_not_called()
        self.assertEqual(root.start(), Result.NOT_INITIALIZED)
        self.assertEqual(root.close(), Result.OK)

    async def test_wait_failure_is_retained_after_its_ui_waiter_is_cancelled(self):
        root = self.prepare()
        entered, release = asyncio.Event(), asyncio.Event()
        async def worker(action, timeout):
            entered.set()
            await release.wait()
            raise MemoryError("off-host wait handoff")
        with patch.object(self.loaded['_kit_root'].asyncio, 'to_thread', side_effect=worker):
            waiter = asyncio.create_task(root.wait(5000))
            await entered.wait()
            waiter.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await waiter
            release.set()
            self.assertEqual(await root._wait_task, Result.RESOURCE_EXHAUSTED)
            self.assertEqual(root.wait_result, Result.RESOURCE_EXHAUSTED)
            self.assertIs(root.creation, self.context.creations[0])
        self.assertEqual(root.close(), Result.OK)
