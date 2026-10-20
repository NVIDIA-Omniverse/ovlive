# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control transports retire before Python 3.12 server closure can complete."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_driver():
    spec = importlib.util.spec_from_file_location("kit_process_cleanup_fixture",
        ROOT / "tests/reference_service/test_kit_process.py")
    module = importlib.util.module_from_spec(spec)
    support = SimpleNamespace(PluginTests=SimpleNamespace(cleanup=None, commits=None))
    with patch.dict(sys.modules, {
        "service": SimpleNamespace(pb=None), "test_plugin": support,
        "test_scene": SimpleNamespace(RotatingProxy=None),
        "graph_service": SimpleNamespace(GraphServer=None),
        "scene_graph_wire": SimpleNamespace(**{key: None for key in
            ("Command", "Field", "Delta", "Token", "Specifier", "encode", "decode")}),
    }):
        spec.loader.exec_module(module)
    return module.KitProcess


class KitProcessCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepted_connection_closes_before_listener_wait(self):
        for handshake_completed in (False, True):
            with self.subTest(handshake_completed=handshake_completed):
                driver = load_driver()(Path("unused"))
                writer = SimpleNamespace(close=Mock(), wait_closed=AsyncMock())
                async def listener_closed():
                    # Immediate ownership witness, never a deliberate timeout.
                    writer.close.assert_called_once_with()
                    writer.wait_closed.assert_awaited_once_with()
                driver.server = SimpleNamespace(close=Mock(), wait_closed=listener_closed)
                driver.accepted = asyncio.get_running_loop().create_future()
                driver.accepted.set_result((None, writer))
                if handshake_completed:
                    driver.writer = writer
                await driver.cleanup()
                driver.server.close.assert_called_once_with()

    async def test_partial_startup_has_no_transport_to_wait_for(self):
        await load_driver()(Path("unused")).cleanup()
