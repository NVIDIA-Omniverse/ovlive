# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained Kit root workflows against the real plugin and independent graph service."""
import asyncio
import os
from pathlib import Path
import sys
import unittest

import grpc
from scene_graph_wire import decode
import test_plugin as plugin_support
from test_scene import Driver


class PythonKitRootTests(unittest.IsolatedAsyncioTestCase):
    graph_creation = True
    asyncSetUp = plugin_support.PluginTests.asyncSetUp
    cleanup = plugin_support.PluginTests.cleanup

    async def start_driver(self):
        extension = Path(os.environ['LIVE_TEST_PYTHON_EXTENSION'])
        environment = dict(os.environ, PYTHONPATH=str(extension.parents[3]) + os.pathsep + os.environ['PYTHONPATH'],
                           PYTHONDONTWRITEBYTECODE='1')
        process = await asyncio.create_subprocess_exec(sys.executable,
            str(Path(__file__).parents[1] / 'native/python/kit_root.py'), self.address, str(extension.parents[4]),
            env=environment, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT)
        driver = Driver(process, self.artifact / (self.id().split('.')[-1] + '.kit-root.log'))
        self.drivers.append(driver)
        self.assertEqual(await driver.receive(), dict(type='ready'))
        return driver

    async def prepare(self, driver):
        attempts = dict(self.state.attempts)
        prepared = await driver.command('prepare')
        self.assertEqual(prepared['outcome']['phase'], 0)
        self.assertFalse(prepared['outcome']['started'])
        self.assertEqual(self.state.attempts['CreateSession'], attempts['CreateSession'] + 1)
        self.assertEqual(self.state.attempts['CreateLayer'], attempts['CreateLayer'])
        self.assertNotIn('root.live', self.state.sessions[prepared['session']].layers)
        return prepared

    def assert_seed(self, session):
        commits = self.state.sessions[session].layers['root.live'].commits
        self.assertEqual(len(commits), 1)
        decoded = decode(commits[0].payload)
        self.assertEqual((decoded.is_diff, decoded.base_version, decoded.source_format), (True, 0, 'usda'))
        self.assertEqual([(field.section, field.key, field.value) for field in decoded.fields],
                         [(1, 'startTimeCode', 17.0)])
        self.assertIs(type(decoded.fields[0].value), float)
        return commits[0].payload

    async def close(self, driver):
        attempts = dict(self.state.attempts)
        self.assertEqual(await driver.command('close'), dict(type='closed'))
        self.assertEqual(await asyncio.wait_for(driver.process.wait(), 5), 0)
        self.assertEqual(self.state.attempts.get('CreateLayer'), attempts.get('CreateLayer'))
        self.assertEqual(self.state.attempts.get('CreateSession'), attempts.get('CreateSession'))

    async def test_kit_root_freezes_seed_and_activates_one_original_creation(self):
        driver = await self.start_driver()
        prepared = await self.prepare(driver)
        before = self.state.attempts['CreateLayer']
        self.assertEqual((await driver.command('start'))['result'], 0)
        finished = await driver.command('wait')
        self.assertEqual((finished['result'], finished['outcome']['phase']), (0, 2))
        self.assertEqual(finished['outcome']['subject_id'], prepared['outcome']['subject_id'])
        self.assert_seed(prepared['session'])
        self.assertEqual(await driver.command('activate'),
                         dict(type='activated', result=0, fields=dict(startTimeCode=17.0)))
        self.assertEqual((await driver.command('start'))['result'], 5)
        self.assertEqual((await driver.command('activate'))['result'], 0)
        self.assertEqual(self.state.attempts['CreateLayer'], before + 1)
        await self.close(driver)

    async def test_kit_root_cancelled_waiter_retains_accepted_create(self):
        driver = await self.start_driver()
        prepared = await self.prepare(driver)
        before = self.state.attempts['CreateLayer']
        fault = self.state.plan('CreateLayer', 'before_reply', pause=True)
        self.assertEqual((await driver.command('start'))['result'], 0)
        await asyncio.wait_for(fault.reached.wait(), 5)
        cancelled = await driver.command('wait-cancel')
        self.assertEqual((cancelled['close_result'], cancelled['receipts']), (14, 1))
        self.assertFalse(fault.cancelled.is_set())
        fault.release.set()
        finished = await driver.command('wait')
        self.assertEqual((finished['result'], finished['outcome']['phase']), (0, 2))
        self.assertEqual(self.state.attempts['CreateLayer'], before + 1)
        self.assert_seed(prepared['session'])
        await self.close(driver)

    async def test_kit_root_unknown_commit_retains_receipt_and_blocks_host_close(self):
        driver = await self.start_driver()
        prepared = await self.prepare(driver)
        before = self.state.attempts['CreateLayer']
        self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual((await driver.command('start'))['result'], 0)
        finished = await driver.command('wait')
        self.assertEqual((finished['result'], finished['outcome']['phase'], finished['outcome']['result']), (0, 5, 19))
        self.assert_seed(prepared['session'])
        self.assertEqual((await driver.command('activate'))['result'], 19)
        self.assertEqual((await driver.command('start'))['result'], 5)
        self.assertEqual(await driver.command('close-busy'),
                         dict(type='closing', result=14, receipts=1, roots=1, root_closed=False))
        self.assertEqual(self.state.attempts['CreateLayer'], before + 1)
        await self.close(driver)
        outcome = finished['outcome']
        self.assertGreater(outcome['error']['request_id'], 0)
        self.assertGreater(outcome['error']['attempt_id'], 0)
