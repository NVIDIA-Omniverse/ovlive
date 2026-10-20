# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual Python USD host publication and authoring against an independent oracle."""
import asyncio
import os
from pathlib import Path
import sys
import unittest

import grpc

import test_plugin as plugin_support
import test_scene as scene_support
from scene_wire import decode_update, empty_snapshot, update


class PythonHostTests(unittest.IsolatedAsyncioTestCase):
    # Share the independent service and process-supervision fixture, not its C++
    # consumer tests or its production implementation.
    asyncSetUp = plugin_support.PluginTests.asyncSetUp
    cleanup = plugin_support.PluginTests.cleanup
    commits = plugin_support.PluginTests.commits
    oracle = plugin_support.PluginTests.oracle
    peer = plugin_support.PluginTests.peer
    wait_fields = plugin_support.PluginTests.wait_fields
    close = plugin_support.PluginTests.close

    async def start(self, *, no_history=False, max_pending=None):
        extension = Path(os.environ['LIVE_TEST_PYTHON_EXTENSION'])
        environment = dict(os.environ, PYTHONPATH=str(extension.parents[3]) + os.pathsep + os.environ['PYTHONPATH'],
                           PYTHONDONTWRITEBYTECODE='1')
        process = await asyncio.create_subprocess_exec(sys.executable,
            str(Path(__file__).parents[1] / 'native/python/host.py'), self.address, self.session,
            str(extension.parents[4]), *(['no-history'] if no_history else []),
            *([f'max-pending={max_pending}'] if max_pending else []), env=environment, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        driver = scene_support.Driver(process, self.artifact / f'{self.id().split(".")[-1]}-{len(self.drivers)}.python-host.log')
        self.drivers.append(driver)
        ready = await driver.receive()
        self.assertEqual(ready['type'], 'ready', ready)
        self.assertEqual((ready['fields'], ready['delivery']), (self.oracle(), 0))
        return driver

    async def until_held(self, driver):
        async def wait():
            while True:
                result = await driver.command('held')
                if result['count']:
                    return result
                await asyncio.sleep(.001)
        return await asyncio.wait_for(wait(), 5)

    async def test_python_plain_usd_edits_converge_between_two_hosts(self):
        await self.peer({'framesPerSecond': 24.0})
        # The off-host save can request a second flush while the first flush job holds a slot.
        first, second = await self.start(max_pending=2), await self.start(max_pending=2)
        self.assertEqual((await first.command('contracts'))['fields'], self.oracle())
        changes = [{'framesPerSecond': 24.0}, {'startTimeCode': -1.5}, {'endTimeCode': 120.25}, {'framesPerSecond': 48.0}]
        expected = dict(changes[0])
        for index, change in enumerate(changes[1:]):
            key, value = next(iter(change.items()))
            driver = (first, second)[index % 2]
            await driver.command(f'set {key} {value}')
            self.assertEqual((await driver.command('save 5000'))['result'], 0)
            expected.update(change)
            for observer in (first, second):
                await self.wait_fields(observer, expected)
        self.assertEqual(self.oracle(), expected)
        self.assertEqual([decode_update(commit.payload) for commit in self.commits()[1:]], changes)
        for driver in (first, second):
            await self.close(driver)

    async def test_python_single_slot_host_edits_publish_and_converge(self):
        # A layer's own publication job is not charged to the host quota. Each
        # edit's flush job is held in the host's dispatcher, keeping the one
        # charged slot, while an on-host save admits the edit; the runtime then
        # dispatches the local publication job into that full context, and the
        # Update it enables reaches the service. The Update is paused there
        # until the flush job runs, because the ordered application of the
        # confirmed edit is a charged job.
        await self.peer({'framesPerSecond': 24.0})
        driver = await self.start(max_pending=1)
        expected = {'framesPerSecond': 24.0}
        for key, value in (('startTimeCode', -1.5), ('endTimeCode', 120.25), ('framesPerSecond', 48.0)):
            fault = self.state.plan('UpdateLayer', 'before', pause=True)
            await driver.command('hold-one')
            await driver.command(f'set {key} {value}')
            expected[key] = value
            self.assertEqual((await self.until_held(driver))['count'], 1)
            self.assertIn((await driver.command('save-host'))['result'], (0, 14))
            async def issued():
                # A refused publication fails the layer and never issues the Update.
                while True:
                    state = await driver.command('state')
                    if fault.reached.is_set() or state['delivery'] or state['result']:
                        return state
                    await asyncio.sleep(.001)
            state = await asyncio.wait_for(issued(), 5)
            self.assertEqual((state['delivery'], state['result']), (0, 0), state)
            self.assertTrue(fault.reached.is_set())
            self.assertEqual((await driver.command('held'))['count'], 1)
            await driver.command('resume')
            self.assertEqual((await driver.command('state'))['delivery'], 0)
            fault.release.set()
            async def confirmed():
                while True:
                    state = await driver.command('state')
                    if (self.oracle() == expected and state['pending'] == 0) or state['delivery'] or state['result']:
                        return state
                    await asyncio.sleep(.001)
            state = await asyncio.wait_for(confirmed(), 5)
            self.assertEqual((state['fields'], state['delivery'], state['result'], state['pending']),
                             (expected, 0, 0, 0), state)
        self.assertEqual(self.oracle(), expected)
        await self.close(driver)

    async def test_python_unknown_write_exports_every_local_edit_without_resend(self):
        self.state.advertise_capabilities = False  # The unknown outcome is kept, never re-sent.
        # The off-host save can request a second flush while the first flush job holds a slot.
        driver = await self.start(no_history=True, max_pending=2)
        directory = self.artifact / (self.id().split('.')[-1] + '-exports')
        directory.mkdir()
        fault = self.state.plan('UpdateLayer', 'after_commit', pause=True, code=grpc.StatusCode.UNAVAILABLE)
        self.assertEqual(await driver.command('set framesPerSecond 48'), {'type': 'authored', 'result': 0})
        await asyncio.wait_for(fault.reached.wait(), 5)
        # The first intent is issued and held after commit; the window of one write is full, so the second
        # edit stays buffered, coalescing.
        self.assertEqual(await driver.command('set startTimeCode -3.5'), {'type': 'authored', 'result': 0})
        both = {'framesPerSecond': 48.0, 'startTimeCode': -3.5}
        queued = await driver.command('state')
        self.assertEqual((queued['fields'], queued['pending'], queued['status']['buffered_edit_count']), (both, 1, 1))
        self.assertEqual(self.state.attempts['UpdateLayer'], 1)
        fault.release.set()
        failed = await driver.command('failed')
        self.assertEqual((failed['state'], failed['result'], failed['pending'], failed['fields']), (7, 19, 1, both))
        self.assertGreater(failed['subject'], 0)
        self.assertGreater(failed['incident'], 0)
        self.assertTrue(failed['status']['has_unconfirmed_edits'])
        observed = await driver.command('observed-failure')
        for field in ('subject_id', 'state', 'result', 'incident_id', 'pending_write_count', 'has_unconfirmed_edits'):
            self.assertEqual(observed['status'][field], failed['status'][field], field)
        # A FAILED layer stays locally editable; the edit only buffers.
        self.assertEqual(await driver.command('set endTimeCode 7'), {'type': 'authored', 'result': 0})
        every = dict(both, endTimeCode=7.0)
        edited = await driver.command('state')
        self.assertEqual((edited['fields'], edited['status']['buffered_edit_count']), (every, 2))
        self.assertEqual(await driver.command('diagnostics'), {'type': 'diagnostics', 'records': []})
        self.assertEqual((await driver.command('save 1000'))['result'], 19)
        attempts = dict(self.state.attempts)
        path = directory / 'failed-layer.usda'
        exported = await driver.command(f'export {path}')
        self.assertEqual(exported, {'type': 'exported', 'fields': every, 'status_unchanged': True})
        self.assertEqual((await driver.command('state'))['fields'], every)
        encoded = path.read_bytes()
        self.assertTrue(encoded.startswith(b'#usda 1.0'))
        self.assertEqual(self.state.attempts, attempts, 'SdfLayer::Export issued service operations')
        self.assertEqual([decode_update(commit.payload) for commit in self.commits()[1:]], [{'framesPerSecond': 48.0}])
        await self.close(driver)
        self.assertEqual(self.state.attempts['UpdateLayer'], 1)
        self.assertEqual(path.read_bytes(), encoded, 'the export changed after its producer closed')
        # Reopen: a fresh layer holds only the authoritative, committed first edit.
        reopened = await self.start()
        self.assertEqual(self.oracle(), {'framesPerSecond': 48.0})
        await self.close(reopened)

    async def test_retained_host_task_limits_publication_of_the_other_layer(self):
        pb, writer = plugin_support.pb, plugin_support.WRITER
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="other.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=empty_snapshot()))]), metadata=writer, timeout=5)
        driver = await self.start()
        self.assertEqual(await driver.command('open-other'), {'type': 'other-opened', 'fields': {}})
        self.assertEqual(self.state.attempts['SubscribeReadLayer'], 2)
        await driver.command('hold-one')
        await self.peer({'framesPerSecond': 84.0})
        held = await self.until_held(driver)
        self.assertEqual((held['count'], held['delivery']), (1, 0))
        blocked = await driver.command('state')
        self.assertEqual(blocked['fields'], {})
        attempts = dict(self.state.attempts)
        layers = self.state.sessions[self.session].layers
        original = {name: [entry.payload for entry in layer.commits] for name, layer in layers.items()}
        self.assertEqual((await driver.command('held'))['count'], 1)
        self.assertEqual(self.state.attempts, attempts)
        self.assertEqual({name: [entry.payload for entry in layer.commits] for name, layer in layers.items()}, original)
        blocked = await driver.command('state')
        self.assertEqual((blocked['fields'], blocked['delivery']), ({}, 0))
        other_value = {'startTimeCode': -8.5}
        await self.stub.UpdateLayer(iter([
            pb.UpdateLayerRequest(header=pb.UpdateLayerHeader(session_id=self.session, layer_name="other.live",
                object_id=layers['other.live'].object_id, client_id="independent-capacity-peer")),
            pb.UpdateLayerRequest(chunk=pb.UpdateLayerChunk(data=update(other_value)))]), metadata=writer, timeout=5)
        async def rejected():
            while True:
                value = await driver.command('other-state')
                if value['delivery'] == 13 and value['result'] != 0:
                    return value
                await asyncio.sleep(.001)
        failure = await asyncio.wait_for(rejected(), 5)
        self.assertEqual(failure['fields'], {})
        self.assertIn(failure['state'], (6, 7))
        self.assertEqual((await driver.command('held'))['count'], 1)
        self.assertEqual([decode_update(entry.payload) for entry in layers['other.live'].commits[1:]], [other_value])
        self.assertEqual(self.state.attempts['UpdateLayer'], attempts['UpdateLayer'] + 1)
        await driver.command('resume')
        await self.wait_fields(driver, self.oracle())
        recovered_first = await driver.command('state')
        self.assertEqual((recovered_first['fields'], recovered_first['result'], recovered_first['delivery']),
                         ({'framesPerSecond': 84.0}, 0, 13))
        self.assertEqual((await driver.command('held'))['count'], 0)
        self.assertEqual([decode_update(entry.payload) for entry in self.commits()[1:]], [{'framesPerSecond': 84.0}])
        self.assertEqual(self.state.attempts['CreateLayer'], 2)
        self.assertEqual(self.state.attempts['SubscribeReadLayer'], 2)
        await self.close(driver)

    async def test_refused_flush_job_blocks_the_other_layer_until_its_next_save(self):
        pb, writer = plugin_support.pb, plugin_support.WRITER
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name="other.live")),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=empty_snapshot()))]), metadata=writer, timeout=5)
        driver = await self.start()  # One charged slot.
        self.assertEqual(await driver.command('open-other'), {'type': 'other-opened', 'fields': {}})
        await driver.command('hold-one')
        await self.peer({'framesPerSecond': 84.0})
        held = await self.until_held(driver)  # The root layer's peer publication holds the slot.
        self.assertEqual((held['count'], held['delivery']), (1, 0))
        updates = self.state.attempts.get('UpdateLayer', 0)
        await driver.command('other-set startTimeCode -8.5')
        blocked = await driver.command('other-wait-state 5')
        self.assertEqual((blocked['state'], blocked['result'], blocked['buffered'], blocked['delivery']),
                         (5, 13, 1, 13), blocked)
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), updates)
        # A second record is refused its flush job too and stays buffered.
        await driver.command('other-set endTimeCode 3')
        grown = await driver.command('other-wait-buffered 2')
        self.assertEqual((grown['state'], grown['result'], grown['buffered']), (5, 13, 2), grown)
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0), updates)
        await driver.command('resume')
        await self.wait_fields(driver, self.oracle())  # The held job ran and released its slot.
        self.assertEqual((await driver.command('other-save 5000'))['result'], 0)
        converged = await driver.command('other-state')
        both = {'startTimeCode': -8.5, 'endTimeCode': 3.0}
        self.assertEqual((converged['state'], converged['result'], converged['buffered'], converged['fields']),
                         (2, 0, 0, both), converged)
        layers = self.state.sessions[self.session].layers
        self.assertEqual([decode_update(e.payload) for e in layers['other.live'].commits[1:]], [both])
        self.assertEqual(self.state.attempts['UpdateLayer'], updates + 1)
        await self.close(driver)

    async def test_busy_context_close_keeps_existing_publication_progressing(self):
        driver = await self.start()
        await driver.command('hold')
        await self.peer({'framesPerSecond': 72.0})
        self.assertEqual((await self.until_held(driver))['count'], 1)
        self.assertEqual((await driver.command('close-context'))['result'], 14)
        await driver.command('resume')
        await self.wait_fields(driver, self.oracle())
        await self.peer({'startTimeCode': 2.5})
        await self.wait_fields(driver, self.oracle())
        await self.close(driver)

    async def test_queued_then_throw_disarms_task_and_reports_delivery_failure(self):
        driver = await self.start()
        await driver.command('hold throw')
        await self.peer({'framesPerSecond': 96.0})
        await self.until_held(driver)
        async def failed():
            while True:
                result = await driver.command('state')
                if result['delivery'] == 20:
                    return result
                await asyncio.sleep(.001)
        result = await asyncio.wait_for(failed(), 5)
        self.assertEqual(result['fields'], {})
        await driver.command('resume')
        # Host queue barrier is the following state query; disarmed task cannot
        # apply its late publication or pretend that the bridge recovered.
        result = await driver.command('state')
        self.assertEqual((result['fields'], result['delivery']), ({}, 20))
        await self.close(driver)

    async def test_interpreter_exit_disarms_an_active_usd_context(self):
        driver = await self.start()
        self.assertEqual((await driver.command('exit-active'))['type'], 'exit-active')
        await asyncio.wait_for(driver.process.wait(), 8)
        await driver.drainer
        self.assertEqual(driver.process.returncode, 0, driver.log.read_text()[-8192:])

    async def test_teardown_waits_for_running_publication_and_python_usd_notice(self):
        driver = await self.start()
        await driver.command('hold-notice')
        await self.peer({'framesPerSecond': 120.0})
        async def entered():
            while not (await driver.command('notice-entered'))['entered']:
                await asyncio.sleep(.001)
        await asyncio.wait_for(entered(), 5)
        self.assertEqual(await driver.command('teardown-running'), {'type': 'running-publication-retired'})
        await asyncio.wait_for(driver.process.wait(), 8)
        await driver.drainer
        self.assertEqual(driver.process.returncode, 0, driver.log.read_text()[-8192:])
