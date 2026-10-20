# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded Python teardown at interpreter exit with the actual extension (design 8)."""
import asyncio
import json
import os
from pathlib import Path
import sys
import unittest


class PythonTeardownTests(unittest.IsolatedAsyncioTestCase):
    async def test_interpreter_exit_abandons_a_never_returning_callback_and_logs(self):
        extension = Path(os.environ["LIVE_TEST_PYTHON_EXTENSION"])
        environment = dict(os.environ, PYTHONPATH=str(extension.parents[3]) + os.pathsep + os.environ["PYTHONPATH"],
                           PYTHONDONTWRITEBYTECODE="1")
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(Path(__file__).parents[1] / "native/python/teardown.py"), "127.0.0.1:1", "0",
            env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        self.addCleanup(lambda: process.returncode is None and process.kill())
        # Failure-only watchdog: an unbounded teardown never exits.
        output, _ = await asyncio.wait_for(process.communicate(), 30)
        text = output.decode(errors="replace")
        (Path(os.environ["LIVE_TEST_ARTIFACT_DIR"]) / "python-teardown.log").write_text(text)
        self.assertEqual(process.returncode, 0, text[-8192:])
        lines = text.splitlines()
        self.assertIn(json.dumps(dict(type="blocked")), lines)
        errors = [line for line in lines if line.startswith("omni.usd.live ERROR Live Services teardown abandoned")]
        # Teardown is idempotent: one teardown pass logs exactly one ERROR line.
        self.assertEqual([line for line in lines if line.startswith("omni.usd.live ERROR")], errors, text[-8192:])
        self.assertEqual(len(errors), 1, text[-8192:])
        self.assertTrue(any("subscription" in line for line in errors), text[-8192:])
        self.assertFalse(any("native" in line for line in errors), text[-8192:])
        native = [json.loads(line) for line in lines if line.startswith('{"type": "native"')]
        self.assertEqual(len(native), 1, text[-8192:])
        # INVALID_HANDLE: teardown released both native owners despite the stuck callback.
        self.assertEqual(native[0]["runtime"], 2, text[-8192:])
        self.assertEqual(native[0]["endpoint"], 2, text[-8192:])
