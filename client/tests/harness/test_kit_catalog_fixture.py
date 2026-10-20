# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed catalogue assertion/export must still retire the reference fixture."""
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from tests.harness.test_kit_sessions import Result

ROOT = Path(__file__).resolve().parents[2]


class KitCatalogFixtureTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_failures_do_not_skip_later_retirement(self):
        spec = importlib.util.spec_from_file_location("kit_catalog_cleanup_fixture",
            ROOT / "tests/reference_service/test_python_kit_catalog.py")
        module = importlib.util.module_from_spec(spec)
        live = SimpleNamespace(Result=Result)
        with patch.dict(sys.modules, {"grpc": SimpleNamespace(), "omni": SimpleNamespace(),
                "omni.usd": SimpleNamespace(live=live), "omni.usd.live._kit_host": SimpleNamespace(KitHost=object),
                "service": SimpleNamespace(State=object, Principal=object, ReferenceServer=object)}):
            spec.loader.exec_module(module)
        async def immediate_worker(function, *args):
            return function(*args)
        # These owners are synchronous fakes. This test checks failure routing;
        # real worker retirement is covered by the reference-service fixtures.
        worker = patch.object(module.asyncio, "to_thread", immediate_worker)
        worker.start(); self.addCleanup(worker.stop)
        for failure in ("receipt", "artifact", "context", "runtime", "partial_startup"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(os.environ, LIVE_TEST_ARTIFACT_DIR=directory):
                fixture = module.PythonKitCatalogTests("test_lost_creation_reply_is_retained_without_automatic_retry")
                fixture.seed = 42
                fixture.state = SimpleNamespace(faults=[], snapshot=lambda: {}, attempts={}, fault_log=[],
                                                assert_faults_fired=Mock())
                fixture.server = SimpleNamespace(stop=AsyncMock())
                context = SimpleNamespace(close=Mock(return_value=Result.OK))
                runtime = SimpleNamespace(close=Mock(return_value=Result.OK))
                fixture.host = SimpleNamespace(publication_context=context, runtime=runtime)
                fixture.catalog = SimpleNamespace(operations=(), set_active=Mock())
                if failure == "receipt":
                    fixture.catalog.operations = (SimpleNamespace(request=SimpleNamespace(wait=Mock(return_value=Result.TIMEOUT))),)
                if failure == "context":
                    context.close.side_effect = RuntimeError("controlled context close failure")
                if failure == "runtime":
                    runtime.close.side_effect = RuntimeError("controlled runtime close failure")
                if failure == "partial_startup":
                    fixture.host = fixture.catalog = None
                failing_write = patch.object(Path, "write_text", side_effect=OSError("controlled evidence failure"))
                with failing_write if failure == "artifact" else nullcontext():
                    if failure == "partial_startup":
                        await fixture.cleanup()
                    else:
                        with self.assertRaises(module.FixtureCleanupError):
                            await fixture.cleanup()
                fixture.server.stop.assert_awaited_once_with()
                fixture.state.assert_faults_fired.assert_called_once_with()
                if failure != "partial_startup":
                    context.close.assert_called_once_with(5000)
                    runtime.close.assert_called_once_with(5000)
