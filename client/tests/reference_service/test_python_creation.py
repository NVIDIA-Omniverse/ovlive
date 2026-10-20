# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public Python creation owners against the independent section-graph service."""
import asyncio
import os
from pathlib import Path
import sys
import unittest

import grpc
from scene_graph_wire import decode
from scene_wire import update
from service import pb
import test_plugin as plugin_support
from test_scene import Driver


class PythonCreationTests(unittest.IsolatedAsyncioTestCase):
    graph_creation = True
    asyncSetUp = plugin_support.PluginTests.asyncSetUp
    cleanup = plugin_support.PluginTests.cleanup

    async def start(self):
        extension = Path(os.environ['LIVE_TEST_PYTHON_EXTENSION'])
        environment = dict(os.environ, PYTHONPATH=str(extension.parents[3]) + os.pathsep + os.environ['PYTHONPATH'],
                           PYTHONDONTWRITEBYTECODE='1')
        process = await asyncio.create_subprocess_exec(sys.executable,
            str(Path(__file__).parents[1] / 'native/python/creation.py'), self.address, self.session,
            str(extension.parents[4]), env=environment, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        driver = Driver(process, self.artifact / f'{self.id().split(".")[-1]}-{len(self.drivers)}.python-creation.log')
        self.drivers.append(driver)
        self.assertEqual(await driver.receive(), {'type': 'ready'})
        return driver

    async def prepared(self, driver):
        before = dict(self.state.attempts)
        value = await driver.command('prepare')
        self.assertEqual((value['result'], value['owners'], value['outcome']['phase'], value['outcome']['started']),
                         (0, 1, 0, False))
        self.assertGreater(value['outcome']['subject_id'], 0)
        self.assertEqual(self.state.attempts, before)
        return value['outcome']['subject_id']

    async def close(self, driver):
        self.assertEqual(await driver.command('close'), {'type': 'closed'})
        self.assertEqual(await asyncio.wait_for(driver.process.wait(), 5), 0)

    def assert_seed(self):
        commits = [entry for entry in self.state.ledger if entry.layer == 'created.live']
        self.assertEqual(len(commits), 1)
        seed = decode(commits[0].payload)
        self.assertEqual((seed.is_diff, seed.base_version, seed.source_format), (True, 0, 'usda'))
        self.assertEqual([(field.section, field.key, field.value) for field in seed.fields], [(1, 'startTimeCode', 17.0)])
        return commits[0].payload

    async def test_python_creation_frozen_seed_retention_capacity_and_activation(self):
        driver = await self.start()
        self.assertEqual(await driver.command('contracts'), {'type': 'contracts'})
        self.assertEqual(await driver.command('replace-context'), {'type': 'replaced'})
        subject = await self.prepared(driver)
        self.assertEqual((await driver.command('retain'))['outcome']['subject_id'], subject)
        self.assertEqual((await driver.command('wait 0'))['result'], 14)
        self.assertEqual((await driver.command('start'))['result'], 0)
        done = await driver.command('wait 5000')
        self.assertEqual((done['result'], done['outcome']['phase'], done['outcome']['result']), (0, 2, 0))
        self.assert_seed()
        attempts = dict(self.state.attempts)
        self.assertEqual((await driver.command('start'))['result'], 5)
        self.assertEqual((await driver.command('prepare-layer'))['result'], 0)
        active = await driver.command('activate')
        self.assertEqual((active['result'], active['fields']), (0, {'startTimeCode': 17.0}))
        subscriptions = self.state.attempts['SubscribeReadLayer']
        self.assertGreater(subscriptions, 0)
        repeated = await driver.command('activate')
        self.assertEqual((repeated['result'], repeated['fields']), (0, active['fields']))
        self.assertEqual(self.state.attempts['SubscribeReadLayer'], subscriptions)
        # Bootstrap read is allowed; repeat start/activate must not issue CreateLayer.
        self.assertEqual(self.state.attempts['CreateLayer'], attempts['CreateLayer'])
        self.assertEqual(await driver.command('close-owner'), {'type': 'owner-closed'})
        await self.prepared(driver)  # Explicit retirement frees the bounded slot.
        await self.close(driver)
        self.assert_seed()

    async def test_python_creation_uncertain_write_never_retries(self):
        driver = await self.start()
        await self.prepared(driver)
        self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        await driver.command('start')
        done = await driver.command('wait 5000')
        outcome = done['outcome']
        self.assertEqual((done['result'], outcome['phase'], outcome['result']), (0, 5, 19))
        self.assert_seed()
        attempts = dict(self.state.attempts)
        self.assertEqual((await driver.command('retain'))['outcome'], outcome)
        self.assertEqual((await driver.command('start'))['result'], 5)
        self.assertEqual(self.state.attempts, attempts)
        await self.close(driver)
        self.assertEqual(self.state.attempts['CreateLayer'], attempts['CreateLayer'])
        for name in ('request_id', 'attempt_id'):
            self.assertGreater(outcome['error'][name], 0)

    async def test_python_creation_wait_pins_owner_and_context_close_keeps_progress(self):
        driver = await self.start()
        await self.prepared(driver)
        fault = self.state.plan('CreateLayer', 'before', pause=True)
        attempts = self.state.attempts['CreateLayer']
        await driver.command('start')
        await asyncio.wait_for(fault.reached.wait(), 5)
        self.assertEqual((await driver.command('wait 10'))['result'], 9)
        self.assertEqual(await driver.command('wait-background'), {'type': 'waiting'})
        fault.release.set()
        self.assertEqual((await driver.command('join-wait'))['outcome']['phase'], 2)
        self.assertEqual(self.state.attempts['CreateLayer'], attempts + 1)
        self.assert_seed()
        self.assertEqual((await driver.command('prepare-layer'))['result'], 0)
        active = await driver.command('activate')
        self.assertEqual((active['result'], active['fields']), (0, {'startTimeCode': 17.0}))
        await self.close(driver)

    async def test_python_creation_dispatch_failure_keeps_the_host_layer_exportable(self):
        driver = await self.start()
        await self.prepared(driver)
        await driver.command('start')
        await driver.command('wait 5000')
        await driver.command('prepare-layer')
        self.assertEqual((await driver.command('activate'))['result'], 0)
        self.assert_seed()
        self.assertEqual(await driver.command('throw-dispatch'), {'type': 'throwing'})
        layer = self.state.sessions[self.session].layers['created.live']
        peer_wire = update({'startTimeCode': 72.0})
        await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name='created.live',
                                                             object_id=layer.object_id, client_id='peer')),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=peer_wire))]),
            metadata=plugin_support.WRITER, timeout=5)
        committed = self.state.sessions[self.session].layers['created.live'].commits[-1]
        self.assertEqual((committed.operation, committed.sequence, committed.payload), ('UpdateLayer', 2, peer_wire))
        failed = await driver.command('failed')
        self.assertNotEqual(failed['delivery'], 0)
        path = self.artifact / (self.id().split('.')[-1] + '.usda')
        attempts = dict(self.state.attempts)
        exported = await driver.command(f'export {path}')
        # The host layer holds the fields the failed report shows; the peer edit never installed.
        self.assertEqual(exported['fields'], {'startTimeCode': 17.0})
        self.assertTrue(path.read_bytes().startswith(b'#usda 1.0'))
        self.assertEqual(self.state.attempts, attempts)
        await self.close(driver)

    async def test_python_unstarted_creation_close_and_interpreter_exit_never_issue(self):
        for command in ('close', 'exit-active'):
            driver = await self.start()
            await self.prepared(driver)
            attempts = dict(self.state.attempts)
            result = await driver.command(command)
            self.assertEqual(result['type'], 'closed' if command == 'close' else 'exiting')
            self.assertEqual(await asyncio.wait_for(driver.process.wait(), 8), 0)
            self.assertEqual(self.state.attempts, attempts)

    async def test_python_interrupt_after_native_prepare_keeps_accessible_owner(self):
        driver = await self.start()
        attempts = dict(self.state.attempts)
        retained = await driver.command('interrupt-prepare')
        self.assertEqual((retained['owners'], retained['outcome']['phase'], retained['outcome']['started']),
                         (1, 0, False))
        self.assertGreater(retained['outcome']['subject_id'], 0)
        self.assertEqual((await driver.command('retain'))['outcome'], retained['outcome'])
        self.assertEqual(await driver.command('close-owner'), {'type': 'owner-closed'})
        await self.prepared(driver)
        await self.close(driver)
        self.assertEqual(self.state.attempts, attempts)

    async def retained_prepared(self, driver):
        attempts = dict(self.state.attempts)
        prepared = await driver.command('retained-prepare')
        self.assertEqual(prepared['type'], 'retained-prepared')
        self.assertGreater(prepared['subject'], 0)
        self.assertTrue(prepared['uri'].endswith('/created.live'))
        self.assertEqual(self.state.attempts, attempts)
        return prepared['subject']

    async def test_python_retained_creation_original_context_capacity_and_frozen_seed(self):
        driver = await self.start()
        subject = await self.retained_prepared(driver)
        attempts = dict(self.state.attempts)
        self.assertEqual(await driver.command('retained-contracts'),
                         dict(type='retained-contracts', subject=subject))
        self.assertEqual(await driver.command('retained-replace-context'),
                         dict(type='retained-context-replaced', subject=subject))
        acquired = await driver.command('retained-acquire')
        self.assertEqual((acquired['owners'], acquired['outcome']['subject_id'],
                          acquired['outcome']['phase'], acquired['outcome']['started']), (1, subject, 0, False))
        self.assertEqual(self.state.attempts, attempts)
        self.assertEqual((await driver.command('start'))['result'], 0)
        self.assertEqual((await driver.command('wait 5000'))['outcome']['phase'], 2)
        self.assert_seed()
        self.assertEqual((await driver.command('prepare-layer'))['result'], 0)
        active = await driver.command('activate')
        self.assertEqual((active['result'], active['fields']), (0, {'startTimeCode': 17.0}))
        await self.close(driver)
        self.assertEqual(self.state.attempts['CreateLayer'], attempts['CreateLayer'] + 1)

    async def test_python_retained_creation_interrupted_transfer_preserves_context_owner(self):
        driver = await self.start()
        subject = await self.retained_prepared(driver)
        attempts = dict(self.state.attempts)
        acquired = await driver.command('retained-interrupt-acquire')
        self.assertEqual((acquired['owners'], acquired['outcome']['subject_id'],
                          acquired['outcome']['phase']), (1, subject, 0))
        self.assertEqual(self.state.attempts, attempts)
        self.assertEqual((await driver.command('retain-existing'))['outcome'], acquired['outcome'])
        await driver.command('start')
        self.assertEqual((await driver.command('wait 5000'))['outcome']['phase'], 2)
        self.assert_seed()
        await self.close(driver)
        self.assertEqual(self.state.attempts['CreateLayer'], attempts['CreateLayer'] + 1)

    async def test_python_retained_creation_explicit_release_never_issues(self):
        driver = await self.start()
        await self.retained_prepared(driver)
        attempts = dict(self.state.attempts)
        released = await driver.command('retained-release')
        self.assertEqual(released['type'], 'retained-released')
        self.assertIn(released['first_result'], (0, 14))
        await self.close(driver)
        self.assertEqual(self.state.attempts, attempts)

    async def test_python_actual_usd_export_hands_off_and_frees_its_slot(self):
        driver = await self.start()
        attempts = dict(self.state.attempts)
        exported = await driver.command('usd-export')
        # The driver waited for the destination subject to close: the adoption
        # job dropped the only reference to the Export destination.
        self.assertEqual((exported['ok'], exported['result'], exported['retained']), (True, 0, False))
        self.assertIn(exported['state'], (1, 2))
        wire = self.assert_seed()
        self.assertTrue(wire)
        opened = await driver.command('open-created-layer')
        self.assertEqual((opened['error'], opened['fields']), (None, {'startTimeCode': 17.0}))
        await self.close(driver)
        self.assertEqual(self.state.attempts['CreateLayer'], attempts['CreateLayer'] + 1)

    async def test_python_actual_usd_create_new_finalizes_after_outer_call(self):
        driver = await self.start()
        attempts = dict(self.state.attempts)
        created = await driver.command('usd-create-new')
        self.assertEqual((created['ok'], created['result'], created['retained']), (True, 0, False))
        self.assertIn(created['state'], (1, 2))
        commits = [entry for entry in self.state.ledger if entry.layer == 'created.live']
        self.assertEqual(len(commits), 1)
        seed = decode(commits[0].payload)
        self.assertEqual((seed.is_diff, seed.base_version, seed.source_format, seed.fields), (True, 0, '', ()))
        self.assertEqual(len(seed.commands), 1)
        self.assertEqual((seed.commands[0].section, seed.commands[0].parent, seed.commands[0].name,
                          seed.commands[0].spec_type), (1, 0, '', 7))
        self.assertEqual((await driver.command('verify-created-layer'))['fields'], {})
        await self.close(driver)
        self.assertEqual(self.state.attempts['CreateLayer'], attempts['CreateLayer'] + 1)

    async def test_python_successful_create_new_frees_its_retention_slot(self):
        driver = await self.start()
        created = await driver.command('usd-create-new')
        self.assertEqual((created['ok'], created['result'], created['retained']), (True, 0, False))
        dropped = await driver.command('drop-created-layer')
        self.assertEqual((dropped['subject'], dropped['retained']), (created['subject'], False))
        # No retained workflow holds the publication context.
        self.assertEqual(await driver.command('close-context'), dict(type='context-closed', result=0))
        await self.close(driver)

    async def test_python_actual_usd_hooks_uncertain_retained_recovery_without_retry(self):
        await self.uncertain_hooks(abandon=False)

    async def test_python_retained_unknown_hooks_survive_original_context_destruction(self):
        await self.uncertain_hooks(abandon=True)

    async def uncertain_hooks(self, *, abandon):
        for command in ('usd-export', 'usd-create-new'):
            with self.subTest(command=command):
                driver = await self.start()
                before = dict(self.state.attempts)
                count = len([entry for entry in self.state.ledger if entry.layer == 'created.live'])
                self.state.plan('CreateLayer', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
                failed = await driver.command(command)
                self.assertEqual((failed['ok'], failed['state'], failed['result'], failed['retained']),
                                 (False, 7, 19, True))
                commits = [entry for entry in self.state.ledger if entry.layer == 'created.live']
                self.assertEqual(len(commits), count + 1)
                wire = commits[-1].payload
                decoded = decode(wire)
                self.assertEqual([(field.key, field.value) for field in decoded.fields],
                                 [('startTimeCode', 17.0)] if command == 'usd-export' else [])
                if abandon:
                    attempts = dict(self.state.attempts)
                    destroyed = await driver.command('retained-abandon-context')
                    self.assertEqual(destroyed, dict(type='retained-context-abandoned',
                                                    subject=failed['subject'], original_destroyed=True))
                    self.assertEqual(self.state.attempts, attempts)
                acquired = await driver.command('retained-acquire')
                self.assertEqual((acquired['owners'], acquired['outcome']['subject_id'],
                                  acquired['outcome']['phase'], acquired['outcome']['result']),
                                 (1, failed['subject'], 5, 19))
                attempts = dict(self.state.attempts)
                self.assertEqual((await driver.command('start'))['result'], 5)
                self.assertEqual(self.state.attempts, attempts)
                await self.close(driver)
                for name in ('request_id', 'attempt_id'):
                    self.assertGreater(acquired['outcome']['error'][name], 0)
                self.assertEqual(self.state.attempts['CreateLayer'], before['CreateLayer'] + 1)


    async def test_python_creation_wait_closed_from_dispatch_callback_never_self_waits(self):
        driver = await self.start()
        await self.prepared(driver)
        self.assertEqual((await driver.command('start'))['result'], 0)
        self.assertEqual((await driver.command('wait 5000'))['outcome']['phase'], 2)
        self.assertEqual((await driver.command('prepare-layer'))['result'], 0)
        self.assertEqual((await driver.command('activate'))['fields'], {'startTimeCode': 17.0})
        self.assert_seed()
        self.assertEqual(await driver.command('hold-close-callback'), dict(type='callback-armed'))
        layer = self.state.sessions[self.session].layers['created.live']
        wire = update({'startTimeCode': 33.0})
        await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name='created.live',
                                                             object_id=layer.object_id, client_id='peer')),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=wire))]),
            metadata=plugin_support.WRITER, timeout=5)
        committed = self.state.sessions[self.session].layers['created.live'].commits[-1]
        self.assertEqual((committed.operation, committed.sequence, committed.payload), ('UpdateLayer', 2, wire))
        self.assertEqual(await driver.command('close-with-callback-held'),
                         dict(type='callback-held-close', result=14))
        attempts = dict(self.state.attempts)
        self.assertEqual(await driver.command('finish-close-callback'),
                         dict(type='callback-close-finished', callback_result=14, owners=0))
        await self.close(driver)
        self.assertEqual(self.state.attempts, attempts)
