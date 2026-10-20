# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kit catalogue contracts against the actual binding and independent service."""
import asyncio
import json
import os
from pathlib import Path
import unittest

import grpc
from omni.usd import live
from omni.usd.live._kit_host import KitHost
from service import State, Principal, ReferenceServer


class FixtureCleanupError(RuntimeError):
    def __init__(self, failures):
        self.failures = tuple(failures)
        super().__init__("; ".join(f"{stage}: {type(error).__name__}: {error}" for stage, error in failures))


class PythonKitCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seed = int(os.environ["LIVE_TEST_SEED"])
        self.state = State(self.seed)
        self.state.tokens = {"synthetic-writer": Principal("writer")}
        self.server = ReferenceServer(self.state, "kit-catalog")
        self.host = self.catalog = None
        self.addAsyncCleanup(self.cleanup)
        self.address = await self.server.start()
        self.host = KitHost(asyncio.create_task)
        self.host.start()
        self.registration = self.host.register_endpoint(self.address, access_token="synthetic-writer",
            on_auth_required=lambda event: None, on_status=lambda event: None)
        self.catalog = self.host.session_catalog(self.registration, "s3://bucket/stable.usda", current_user="writer")

    async def cleanup(self):
        failures, receipts = [], []
        def failed(stage, error):
            failures.append((stage, error))
        try:
            for fault in self.state.faults:
                fault.release.set()
            # Preserve uncertain evidence before fixture teardown. Direct
            # runtime cleanup here is not a product reconciliation workflow.
            for operation in self.catalog.operations if self.catalog is not None else ():
                try:
                    if operation.request is not None:
                        self.assertEqual(await asyncio.to_thread(operation.request.wait, 5000), live.Result.OK)
                    self.catalog.reconcile()
                    result, payload = operation.outcome
                    receipts.append(dict(kind=operation.kind, argument=operation.argument, result=int(result),
                        session_id=getattr(payload, "session_id", None)))
                    if result != live.Result.OUTCOME_UNKNOWN:
                        self.assertEqual(self.catalog.acknowledge(operation), live.Result.OK)
                except BaseException as error:
                    failed("receipt retirement", error)
            artifact = Path(os.environ["LIVE_TEST_ARTIFACT_DIR"]) / (self.id().split(".")[-1] + ".kit-catalog.json")
            artifact.write_text(json.dumps(dict(seed=self.seed, ledger=self.state.snapshot(),
                attempts=self.state.attempts, faults=self.state.fault_log, receipts=receipts), indent=2) + "\n")
        except BaseException as error:
            failed("receipt evidence capture", error)
        finally:
            # Each retirement attempt runs even when an earlier assertion,
            # export or close fails. All failures remain visible to unittest.
            try:
                if self.catalog is not None:
                    self.catalog.set_active(False)
            except BaseException as error:
                failed("stop catalogue polling", error)
            try:
                if self.host is not None and self.host.publication_context is not None:
                    self.assertEqual(self.host.publication_context.close(5000), live.Result.OK)
            except BaseException as error:
                failed("publication consumer retirement", error)
            try:
                if self.host is not None and self.host.runtime is not None:
                    self.assertEqual(await asyncio.to_thread(self.host.runtime.close, 5000), live.Result.OK)
            except BaseException as error:
                failed("runtime retirement", error)
            try:
                await self.server.stop()
            except BaseException as error:
                failed("reference server retirement", error)
        try:
            self.state.assert_faults_fired()
        except BaseException as error:
            failed("reference fault coverage", error)
        if failures:
            raise FixtureCleanupError(failures) from failures[0][1]

    async def wait(self, operation):
        outcome = await asyncio.wait_for(operation.wait(), 5)  # Failure-only watchdog.
        self.assertEqual(await asyncio.to_thread(operation.request.wait, 5000), live.Result.OK)
        return outcome

    async def test_confirmed_creation_failed_listing_and_authoritative_owner(self):
        operation = self.catalog.create("lighting")
        result, created = await self.wait(operation)
        self.assertEqual(result, live.Result.OK)
        self.assertEqual(self.state.attempts, {"CreateSession": 1})
        self.assertFalse(self.catalog.can_merge(created.session_id))
        self.state.plan("ListSessions", "before", code=grpc.StatusCode.PERMISSION_DENIED)
        result, _ = await self.wait(self.catalog.refresh())
        self.assertEqual(result, live.Result.PERMISSION_DENIED)
        self.assertEqual(operation.outcome, (live.Result.OK, created))
        self.assertEqual(self.catalog.rows[0].session_id, created.session_id)
        self.assertFalse(self.catalog.can_merge(created.session_id))
        self.assertEqual((await self.wait(self.catalog.refresh()))[0], live.Result.OK)
        row, = self.catalog.rows
        self.assertEqual((row.name, row.created_by), ("lighting", "writer"))
        self.assertEqual(row.root_uri, live.make_layer_uri(created.uri, "root.live"))
        self.assertTrue(self.catalog.can_merge(created.session_id))
        self.assertEqual(self.state.attempts["CreateSession"], 1)
        self.assertEqual(self.catalog.acknowledge(operation), live.Result.OK)
        # A later runtime must not steal a URI-only mutation from this owner.
        unrelated = live.initialize()
        try:
            deletion = self.catalog.delete(created.session_id)
            self.assertIs(deletion.request._runtime, self.host.runtime)
            self.assertEqual((await self.wait(deletion))[0], live.Result.OK)
            self.assertEqual(self.catalog.rows, ())
            self.assertEqual(self.catalog.acknowledge(deletion), live.Result.OK)
        finally:
            self.assertEqual(await asyncio.to_thread(unrelated.close, 5000), live.Result.OK)

    async def test_cancelled_ui_waiter_preserves_original_accepted_create(self):
        fault = self.state.plan("CreateSession", "before_reply", pause=True)
        operation = self.catalog.create("one accepted create")
        entered = asyncio.Event()
        async def ui_waiter():
            entered.set()
            return await operation.wait()
        waiter = asyncio.create_task(ui_waiter())
        await entered.wait()
        await asyncio.wait_for(fault.reached.wait(), 5)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertEqual(self.catalog.close(), live.Result.BUSY)
        self.assertEqual(self.host.close(5000), live.Result.BUSY)
        self.assertEqual(self.registration.auth.delivery_status, live.Result.OK)
        fault.release.set()
        result, created = await self.wait(operation)
        self.assertEqual(result, live.Result.OK)
        self.assertEqual(list(self.state.sessions), [created.session_id])
        self.assertEqual(self.state.attempts["CreateSession"], 1)
        self.assertFalse(fault.cancelled.is_set())
        self.assertEqual(self.catalog.acknowledge(operation), live.Result.OK)
        self.assertEqual(self.host.close(5000), live.Result.OK)

    async def test_lost_creation_reply_is_retained_without_automatic_retry(self):
        self.state.plan("CreateSession", "after_commit", code=grpc.StatusCode.UNAVAILABLE)
        operation = self.catalog.create("uncertain creation")
        self.assertEqual(await self.wait(operation), (live.Result.OUTCOME_UNKNOWN, None))
        self.assertEqual(len(self.state.sessions), 1)
        self.assertEqual(self.state.attempts["CreateSession"], 1)
        self.assertEqual(self.catalog.acknowledge(operation), live.Result.BUSY)
        self.assertEqual(self.catalog.close(), live.Result.BUSY)
        self.assertEqual(self.host.close(5000), live.Result.BUSY)
        self.assertEqual(self.catalog.operations, (operation,))
        self.assertEqual(self.registration.status.delivery_status, live.Result.OK)

    async def test_native_view_failure_preserves_confirmed_creation_without_resubmission(self):
        fail_projection = False
        def project(catalog):
            return live.Result.INTERNAL_ERROR if fail_projection else live.Result.OK
        self.catalog = self.host.session_catalog(self.registration, "s3://bucket/projected.usda",
                                                current_user="writer", on_change=project)
        fault = self.state.plan("CreateSession", "before_reply", pause=True)
        operation = self.catalog.create("confirmed despite view failure")
        await asyncio.wait_for(fault.reached.wait(), 5)
        fail_projection = True
        fault.release.set()
        result, created = await self.wait(operation)
        self.assertEqual(result, live.Result.OK)
        self.assertEqual(operation.outcome, (live.Result.OK, created))
        self.assertEqual(self.catalog.view_result, live.Result.INTERNAL_ERROR)
        self.assertEqual(self.catalog.rows[0].session_id, created.session_id)
        fail_projection = False
        self.assertEqual(self.catalog.refresh_projection(), live.Result.OK)
        self.assertEqual(self.state.attempts, {"CreateSession": 1})
        self.assertEqual(list(self.state.sessions), [created.session_id])
