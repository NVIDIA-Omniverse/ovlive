# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the shipped headless workflow using real public APIs and service faults."""
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest

import grpc
from service import Principal
import test_plugin as plugin_support
import test_scene as scene_support
import tls_fixture
from test_scene import Driver

# The operator's retry event for a retained sample: SIGUSR1 on Linux, and on Windows CTRL_BREAK_EVENT, which reaches
# only a process group of its own.
RETRY_SIGNAL = signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGUSR1
RETRY_SIGNAL_NAME = 'CTRL_BREAK_EVENT' if os.name == 'nt' else 'SIGUSR1'
PROCESS_GROUP = dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else {}


class PythonSampleTests(unittest.IsolatedAsyncioTestCase):
    graph_creation = True
    asyncSetUp = plugin_support.PluginTests.asyncSetUp
    cleanup = plugin_support.PluginTests.cleanup

    async def start(self, *, address=None, ca_bundle=None, tls_server_name=None, transport_security=None):
        extension = Path(os.environ['LIVE_TEST_PYTHON_EXTENSION'])
        self.output = self.artifact / (self.id().split('.')[-1] + '.sample')
        environment = dict(os.environ, PYTHONPATH=str(extension.parents[3]) + os.pathsep + os.environ['PYTHONPATH'],
                           PYTHONDONTWRITEBYTECODE='1', LIVE_SAMPLE_ACCESS_TOKEN='synthetic-writer')
        options = []
        if ca_bundle is not None:
            options += ['--ca-bundle', str(ca_bundle)]
        if tls_server_name is not None:
            options += ['--tls-server-name', tls_server_name]
        if transport_security is not None:
            options += ['--transport-security', transport_security]
        process = await asyncio.create_subprocess_exec(sys.executable, '-B', '-m', 'omni.usd.live.sample',
            '--authority', address or self.address, '--output', str(self.output), *options, env=environment,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            **PROCESS_GROUP)
        driver = Driver(process, self.artifact / (self.id().split('.')[-1] + '.sample.log'))
        self.drivers.append(driver)
        self.assertEqual(await driver.receive(), {'type': 'ready'})
        return driver

    async def test_sample_connects_through_tls_proxy_with_private_ca(self):
        pki = await tls_fixture.case_pki(self)
        proxy = scene_support.RotatingProxy(self.address, ssl_context=pki.server_context())
        self.proxies.append(proxy)
        address = await proxy.start()
        driver = await self.start(address=address, ca_bundle=pki.ca,
                                  tls_server_name=tls_fixture.NAME, transport_security='tls')
        await self.create(driver)
        await self.close(driver)

    async def command(self, driver, operation, **fields):
        return await driver.command(json.dumps(dict(operation=operation, **fields)))

    async def create(self, driver):
        value = await self.command(driver, 'create', base_layer_key='sample-base', name='Sample')
        self.assertEqual(value['result'], 0, value)
        self.assertEqual(set(value['session']), {'session_id', 'uri'})
        self.assertEqual(value['metadata'][0]['created_by'], 'writer')
        return value['session']['session_id']

    async def close(self, driver):
        closed = await self.command(driver, 'close')
        self.assertEqual((closed['result'], closed.get('closed')), (0, True), closed)
        self.assertEqual((closed['logging']['dropped'], closed['logging']['error']), (0, None))
        self.assertEqual(await asyncio.wait_for(driver.process.wait(), 5), 0)
        await driver.drainer
        self.assertIsNone(driver.failure)

    async def test_sample_create_edit_refresh_save_diagnostics_and_leave(self):
        driver = await self.start()
        session = await self.create(driver)
        await self.command(driver, 'edit', value=42)
        self.assertEqual((await self.command(driver, 'save'))['result'], 0)
        self.assertEqual((await self.command(driver, 'inspect'))['value'], 42)
        token = 'sample-refresh-secret-canary'
        self.state.tokens[token] = Principal('writer')
        token_file = self.artifact / 'sample-token'
        token_file.write_text(token)
        refreshed = await self.command(driver, 'refresh', token_file=str(token_file))
        self.assertEqual(refreshed['result'], 0)
        await self.command(driver, 'edit', value=99)
        self.assertEqual((await self.command(driver, 'save'))['result'], 0)
        self.assertEqual((await self.command(driver, 'inspect'))['value'], 99)
        self.assertEqual((await self.command(driver, 'snapshot'))['result'], 0)
        commits = self.state.sessions[session].layers['sample.live'].commits
        self.assertEqual([c.operation for c in commits], ['CreateLayer', 'UpdateLayer', 'UpdateLayer'])
        await self.close(driver)
        self.assertFalse(self.state.sessions[session].peers)
        events = [json.loads(line) for line in (self.output / 'events.jsonl').read_text().splitlines()]
        self.assertTrue(any(e['type'] == 'status' for e in events))
        self.assertTrue(any(e['type'] == 'diagnostic' for e in events))
        for path in self.output.iterdir():
            self.assertNotIn(token, path.read_text())
            self.assertNotIn('synthetic-writer', path.read_text())

    def exported_example(self, path, value):
        # The export is a plain USDA layer holding the local /Example.value.
        text = path.read_text()
        self.assertTrue(text.startswith('#usda 1.0'), text)
        self.assertRegex(text, r'def Xform "Example"')
        self.assertRegex(text, rf'int value = {value}\b')

    async def test_sample_uncertain_edit_retains_until_successful_export(self):
        # A service without operation IDs cannot answer a re-send, so a lost reply leaves the write's outcome
        # unknown; a service that chains writes resolves it instead.
        self.state.operation_ids = False
        driver = await self.start()
        session = await self.create(driver)
        self.state.plan('UpdateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        await self.command(driver, 'edit', value=73)
        failed = await self.command(driver, 'save')
        self.assertEqual(failed['result'], 19)  # Original OUTCOME_UNKNOWN reaches the blocked save.
        self.assertEqual((await self.command(driver, 'close'))["result"], 19)
        occupied = self.output / 'occupied'
        occupied.write_bytes(b'keep existing data')
        self.assertEqual((await self.command(driver, 'export', path=str(occupied)))['result'], 5)
        self.assertEqual(occupied.read_bytes(), b'keep existing data')
        self.assertEqual((await self.command(driver, 'close'))["result"], 19)
        path = self.output / 'failed-layer.usda'
        self.assertEqual((await self.command(driver, 'export', path=str(path)))['result'], 0)
        await self.close(driver)
        self.exported_example(path, 73)
        self.assertEqual(len(self.state.sessions[session].layers['sample.live'].commits), 2)
        self.assertEqual(self.state.attempts['UpdateLayer'], 1, 'uncertain edit was sent again')

    async def test_sample_uncertain_creation_keeps_seed_until_export(self):
        driver = await self.start()
        self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        created = await self.command(driver, 'create', base_layer_key='sample-base')
        self.assertTrue(created['export_required'])
        self.assertEqual((await self.command(driver, 'close'))["result"], 19)
        path = self.output / 'creation.usda'
        self.assertEqual((await self.command(driver, 'export', path=str(path)))['result'], 0)
        await self.close(driver)
        self.exported_example(path, 0)  # The caller's seed; the creation never activated.

    async def test_sample_failed_export_retains_until_a_later_export(self):
        driver = await self.start()
        self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        created = await self.command(driver, 'create', base_layer_key='sample-base')
        self.assertTrue(created['export_required'])
        # SdfLayer::Export creates missing directories, so a parent that is a
        # regular file makes the write fail.
        blocker = self.output / 'not-a-directory'
        blocker.write_bytes(b'blocker')
        self.assertEqual((await self.command(driver, 'export', path=str(blocker / 'x.usda')))['result'], 21)
        self.assertEqual(blocker.read_bytes(), b'blocker')
        self.assertEqual((await self.command(driver, 'close'))["result"], 19)
        path = self.output / 'after-failure.usda'
        self.assertEqual((await self.command(driver, 'export', path=str(path)))['result'], 0)
        await self.close(driver)
        self.exported_example(path, 0)

    async def test_sample_failed_bootstrap_preserves_confirmed_creation(self):
        driver = await self.start()
        self.state.plan('SubscribeReadLayer', 'before', code=grpc.StatusCode.PERMISSION_DENIED)
        created = await self.command(driver, 'create', base_layer_key='sample-base')
        self.assertNotEqual(created['result'], 0)
        self.assertEqual((await self.command(driver, 'close'))["result"], 7)
        driver.process.stdin.close()
        closed = await driver.receive()
        self.assertEqual((closed['result'], closed.get('closed')), (0, True), closed)
        self.assertEqual(await asyncio.wait_for(driver.process.wait(), 5), 0)
        self.exported_example(self.output / 'eof-export.usda', 0)

    async def test_sample_eof_failed_export_retains_until_explicit_retry_signal(self):
        driver = await self.start()
        self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        created = await self.command(driver, 'create', base_layer_key='sample-base')
        self.assertTrue(created['export_required'])
        path = self.output / 'eof-export.usda'
        path.write_bytes(b'preserve existing file')
        driver.process.stdin.close()
        retained = await driver.receive()
        self.assertEqual((retained['type'], retained['retry_signal']), ('retained', RETRY_SIGNAL_NAME))
        self.assertEqual(path.read_bytes(), b'preserve existing file')  # The sample refuses an existing destination.
        self.assertIsNone(driver.process.returncode)
        path.rename(self.output / 'original-file')
        driver.process.send_signal(RETRY_SIGNAL)
        closed = await driver.receive()
        self.assertEqual((closed['result'], closed.get('closed')), (0, True), closed)
        self.assertEqual(await asyncio.wait_for(driver.process.wait(), 5), 0)
        self.exported_example(path, 0)
