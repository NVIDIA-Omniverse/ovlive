# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two real headless Kit processes and an independent ordered service ledger."""
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import secrets
import shlex
import sys
import time
import unittest

from service import pb
import test_plugin as support
from test_scene import RotatingProxy
from graph_service import GraphServer
from scene_graph_wire import Command, Field, Delta, Token, Specifier, encode, decode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live.test_support import run_process


class KitProcess:
    def __init__(self, output):
        # Kit runs inside this directory, so app/probe/portable arguments must
        # not reinterpret a caller's relative artifact path from the child cwd.
        self.output = output.absolute()
        self.writer = None
        self.task = None
        self.server = self.accepted = None
        self.identity = 0
        self.lock = asyncio.Lock()
        self.commands = []

    def record_command(self, phase, identity, kind):
        assert len(self.commands) < 256, 'Kit command evidence overflow'
        self.commands.append(dict(phase=phase, id=identity, type=kind, time_ns=time.monotonic_ns()))
        (self.output / 'control.json').write_text(json.dumps(self.commands, indent=2) + '\n')

    async def start(self, authority, session, *, stage_root='composed'):
        self.output.mkdir()
        loop = asyncio.get_running_loop()
        accepted = loop.create_future()

        def connect(reader, writer):
            if accepted.done():
                writer.close()
            else:
                accepted.set_result((reader, writer))

        self.server = await asyncio.start_server(connect, '127.0.0.1', 0, limit=65536)
        self.accepted = accepted
        nonce = secrets.token_hex(16)
        app = self.output / 'reference.kit'
        app.write_text('''[package]
version = "1.0.0"
[settings]
app.extensions.registryEnabled = false
app.extensions.fsWatcherEnabled = false
app.fastShutdown = false
app.useFabricSceneDelegate = false
exts."omni.usd".locking.earlyRelease = true
exts."omni.usd".locking.onClose = true
app.uploadDumpsOnStartup = false
crashreporter.enabled = false
telemetry.mode = "test"
''')
        package = Path(os.environ['LIVE_TEST_KIT_PACKAGE'])
        client = Path(os.environ['LIVE_TEST_KIT_CLIENT_EXTENSION'])
        resolver = Path(os.environ['LIVE_TEST_KIT_RESOLVER_EXTENSION'])
        core = os.environ.get('LIVE_TEST_KIT_CORE_EXTENSION')
        argv = [str(ROOT / 'tests/native/python/kit_process.py'), '--authority', authority,
                '--session', session, '--nonce', nonce, '--control-port', str(self.server.sockets[0].getsockname()[1]),
                '--package', str(package), '--client-extension', str(client), '--resolver-extension', str(resolver),
                '--output', str(self.output / 'probe.json'), '--stage-root', stage_root]
        if core:
            argv += ['--core-extension', core]
        command = [os.environ['LIVE_TEST_KIT_EXECUTABLE'], str(app), '--no-window', '--portable',
                   '--portable-root', str(self.output / 'portable'), '--ext-path', str(client),
                   '--ext-path', str(resolver), '--ext-path', str(package), '--enable', 'omni.kit.loop',
                   '--enable', 'omni.usd', '--enable', 'omni_usd_live', '--exec', shlex.join(argv),
                   '--/log/file=', '--/log/outputStreamLevel=Info',
                   '--/plugins/carb.scripting-python.plugin/pythonFlags/Py_IsolatedFlag=0',
                   '--/plugins/carb.scripting-python.plugin/pythonFlags/Py_IgnoreEnvironmentFlag=0']
        if core:
            command += ['--ext-path', core]
        (self.output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
        env = {key: value for key, value in os.environ.items()
               if key in {'PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'TEMP', 'TMP'}}
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        if os.environ.get('LIVE_TEST_KIT_LIBRARY_DIR'):
            env['LD_LIBRARY_PATH'] = os.environ['LIVE_TEST_KIT_LIBRARY_DIR']
        self.task = asyncio.create_task(asyncio.to_thread(run_process, command, cwd=self.output,
            output=self.output / 'stdout.log', timeout_ms=45000, max_output_bytes=4*1024*1024, environment=env))
        ready, _ = await asyncio.wait((accepted, self.task), return_when=asyncio.FIRST_COMPLETED)
        if accepted not in ready:
            raise AssertionError('Kit exited before opening control channel: ' + str(self.output))
        self.reader, self.writer = accepted.result()
        response = await self.receive()
        assert response.pop('nonce') == nonce and response['type'] == 'ready', response
        return response

    async def receive(self):
        raw = await asyncio.wait_for(self.reader.readline(), 15)
        if not raw:
            raise AssertionError('Kit control channel closed: ' + str(self.output))
        return json.loads(raw)

    async def command(self, kind, **values):
        async with self.lock:
            self.identity += 1
            payload = json.dumps(dict(id=self.identity, type=kind, **values)).encode() + b'\n'
            assert len(payload) < 65536
            self.record_command('sent', self.identity, kind)
            self.writer.write(payload)
            await self.writer.drain()
            response = await self.receive()
            assert response['id'] == self.identity, response
            self.record_command('received', self.identity, kind)
            return response

    async def cleanup(self):
        if self.server is not None:
            self.server.close()
        if self.writer is None and self.accepted is not None and self.accepted.done() and not self.accepted.cancelled():
            _, self.writer = self.accepted.result()
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except (ConnectionError, OSError):
                pass
        # Python 3.12 waits for accepted transports as well as the listener.
        # Retire our accepted writer before waiting for server closure.
        if self.server is not None:
            await self.server.wait_closed()
        if self.task is not None:
            result = asdict(await asyncio.shield(self.task))
            (self.output / 'process.json').write_text(json.dumps(result, indent=2) + '\n')
            assert result['cleanup_complete'] and not result['output_truncated'], result
            assert result['outcome'] == 'passed' and result['returncode'] == 0, result
            assert '[Error]' not in (self.output / 'stdout.log').read_text(errors='replace')
            evidence = json.loads((self.output / 'probe.json').read_text())
            assert evidence['passed'] is True, evidence


class KitProcessTests(unittest.IsolatedAsyncioTestCase):
    graph_creation = True
    cleanup = support.PluginTests.cleanup
    commits = support.PluginTests.commits

    async def asyncSetUp(self):
        await support.PluginTests.asyncSetUp(self)
        self.kit = []
        self.addAsyncCleanup(self.close_kit)
        commands = [Command(1, 2, parent=1, name='World', spec_type=6)]
        fields = [Field(2, 'specifier', Specifier(0)), Field(2, 'typeName', Token('Scope'))]
        for identity, key in enumerate(('a', 'b', 'c'), 3):
            commands.append(Command(1, identity, parent=2, name=key, spec_type=1))
            fields.extend((Field(identity, 'typeName', Token('double')), Field(identity, 'default', 0.0)))
        payload = encode(Delta(commands=tuple(commands), fields=tuple(fields)))
        await self.stub.CreateLayer(iter([
            pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name='root.live')),
            pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=payload))]), metadata=support.WRITER, timeout=5)

    async def close_kit(self):
        results = await asyncio.gather(*(driver.cleanup() for driver in self.kit), return_exceptions=True)
        self.assertFalse([result for result in results if isinstance(result, BaseException)], results)

    async def start(self, authority=None, *, stage_root='composed'):
        output = self.artifact / (self.id().split('.')[-1] + '-kit-' + str(len(self.kit)))
        driver = KitProcess(output)
        self.kit.append(driver)
        ready = await driver.start(authority or self.address, self.session, stage_root=stage_root)
        initial = dict(a=0.0, b=0.0, c=0.0)
        self.assertEqual((ready['fields'], ready['stage'], ready['delivery']), (initial, initial, 0))
        return driver

    def oracle(self, layer='root.live'):
        nodes = self.state.sessions[self.session].layers[layer].model.state
        return {node['name']: node['fields']['default'] for node in nodes if node['type'] == 1}

    async def test_two_kit_stages_converge_after_seeded_overlapping_edits(self):
        first, second = await self.start(), await self.start()
        rng = random.Random(self.seed)
        fields = ['a', 'b', 'c']
        edits = []
        for round_id in range(8):
            changes = [{rng.choice(fields): float(round_id * 20 + index * 10 + rng.randrange(1, 10))} for index in range(2)]
            await asyncio.gather(*(driver.command('edit', fields=change)
                                   for driver, change in zip((first, second), changes)))
            await asyncio.gather(first.command('save'), second.command('save'))
            expected = self.oracle()
            observed = await asyncio.gather(first.command('wait', fields=expected), second.command('wait', fields=expected))
            self.assertTrue(all(item['fields'] == expected and item['pending'] == 0 for item in observed))
            edits.extend(changes)
        actual = []
        for commit in self.commits()[1:]:
            delta = decode(commit.payload)
            self.assertFalse(delta.commands)
            self.assertTrue(all(field.key == 'default' and 3 <= field.section <= 5 for field in delta.fields))
            actual.append({('a', 'b', 'c')[field.section - 3]: field.value for field in delta.fields})
        self.assertCountEqual([json.dumps(x, sort_keys=True) for x in actual],
                              [json.dumps(x, sort_keys=True) for x in edits])
        self.assertEqual(self.state.attempts['UpdateLayer'], 16)
        self.trace.update(edits=edits, oracle=self.oracle(), committed=actual)
        for driver in (first, second):
            self.assertEqual((await driver.command('close'))['type'], 'closed')

    async def test_kit_stage_resumes_after_load_balancer_backend_rotation(self):
        alternate = GraphServer(self.state, 'B')
        self.extra_servers.append(alternate)
        target = await alternate.start()
        proxy = RotatingProxy(self.address)
        self.proxies.append(proxy)
        authority = await proxy.start()
        driver = await self.start(authority)
        await driver.command('edit', fields={'a': 30.0})
        await driver.command('save')
        await driver.command('wait', fields=self.oracle())
        # Observe the exact replacement subscription through the service fault
        # gate, never through retry backoff expiry or a fixed-duration sleep.
        fault = self.state.plan('SubscribeReadLayer', 'before', pause=True)
        await proxy.rotate_and_wait(target)
        await asyncio.wait_for(fault.reached.wait(), 15)
        fault.release.set()
        await driver.command('edit', fields={'c': 72.0})
        await driver.command('save')
        observed = await driver.command('wait', fields=self.oracle())
        self.assertEqual(observed['fields'], {'a': 30.0, 'b': 0.0, 'c': 72.0})
        self.assertEqual(self.state.attempts['UpdateLayer'], 2)
        self.assertTrue(any(row['backend'] == 'B' and row['cursor'] > 0 for row in self.state.reads))
        self.trace.update(oracle=self.oracle())
        self.assertEqual((await driver.command('close'))['type'], 'closed')

    async def test_kit_preserves_unknown_and_queued_edits_after_lost_response(self):
        await self.preserve_unknown()

    async def test_kit_direct_live_root_issues_host_edits(self):
        driver = await self.start(stage_root='live')
        edited = await driver.command('root-host-edit')
        initial = dict(a=0.0, b=0.0, c=0.0)
        self.assertEqual((edited['start_time_code'], edited['end_time_code']), (37.0, 100.0))
        self.assertEqual((edited['status']['state'], edited['status']['result'], edited['pending'],
                          edited['status']['incident_id']), (2, 0, 0, 0))
        self.assertEqual((edited['fields'], edited['stage']), (initial,) * 2)
        self.assertEqual(self.state.attempts['UpdateLayer'], 1)
        commits = self.commits()
        self.assertEqual(len(commits), 2)
        delta = decode(commits[1].payload)
        self.assertFalse(delta.commands)
        root = {field.key: field.value for field in delta.fields if field.section == 1}
        self.assertEqual((root.get('startTimeCode'), root.get('endTimeCode')), (37.0, 100.0))
        # Every root metadata field the service received matches the host's observation.
        self.assertEqual(root, {key: item[1] for key, item in edited['root_fields'].items() if key in root})
        self.assertEqual(self.oracle(), initial)
        self.trace.update(oracle=initial, edited=edited, committed_root=root)
        self.assertEqual((await driver.command('close'))['type'], 'closed')

    async def test_kit_healthy_layer_progresses_during_recovery_with_stalled_diagnostics(self):
        await self.preserve_unknown(isolation=True)

    async def preserve_unknown(self, *, isolation=False):
        import grpc
        driver = await self.start()
        if isolation:
            # Separate namespace, same session/endpoint/runtime/publication owner
            # and actual composed Kit stage as the layer that will fail.
            seed = decode(self.commits()[0].payload)
            from dataclasses import replace
            seed = replace(seed, commands=tuple(replace(command, name='Healthy') if command.section == 2
                                                else command for command in seed.commands))
            await self.stub.CreateLayer(iter([
                pb.CreateLayerRequest(header=pb.CreateLayerHeader(session_id=self.session, layer_name='healthy.live')),
                pb.CreateLayerRequest(chunk=pb.CreateLayerChunk(data=encode(seed)))]),
                metadata=support.WRITER, timeout=5)
            ready = await driver.command('open-healthy')
            self.assertEqual((ready['fields'], ready['stage']), (dict(a=0.0, b=0.0, c=0.0),) * 2)
            self.assertTrue((await driver.command('hold-diagnostics'))['held'])
        fault = self.state.plan('UpdateLayer', 'after_commit', pause=True, code=grpc.StatusCode.UNAVAILABLE)
        self.addCleanup(fault.release.set)
        await driver.command('edit', fields={'a': 48.0})
        await asyncio.wait_for(fault.reached.wait(), 15)
        await driver.command('edit', fields={'b': -3.5})
        all_fields = dict(a=48.0, b=-3.5, c=0.0)
        # Publication is allowed to finish before fault release; its USD notice
        # and the service commit gate establish the schedule without a sleep.
        await driver.command('wait', fields=all_fields)
        self.assertEqual(self.state.attempts['UpdateLayer'], 1)
        self.assertEqual(self.oracle(), dict(a=48.0, b=0.0, c=0.0))
        fault.release.set()
        failed = await driver.command('failed')
        self.assertEqual((failed['status']['state'], failed['status']['result'], failed['pending']), (7, 19, 2))
        for key in ('subject_id', 'incident_id', 'state', 'result', 'pending_write_count', 'has_unconfirmed_edits'):
            self.assertEqual(failed['observed_status'][key], failed['status'][key])
        self.assertEqual((failed['fields'], failed['stage']), (all_fields,) * 2)
        if isolation:
            self.assertTrue((await driver.command('diagnostics-state'))['held'])
            await driver.command('edit', target='healthy', fields={'c': 91.25})
            healthy_saved = await driver.command('save', target='healthy')
            healthy_expected = dict(a=0.0, b=0.0, c=91.25)
            healthy_observed = await driver.command('wait', target='healthy', fields=healthy_expected)
            self.assertEqual((healthy_observed['fields'], healthy_observed['stage']), (healthy_expected,) * 2)
            self.assertEqual((healthy_saved['status']['state'], healthy_saved['pending']), (2, 0))
            self.assertNotEqual(healthy_saved['subject'], failed['subject'])
            self.assertEqual((await driver.command('state'))['status'], failed['status'])
            healthy_commits = self.state.sessions[self.session].layers['healthy.live'].commits
            self.assertEqual(self.oracle('healthy.live'), healthy_expected)
            self.assertEqual(len(healthy_commits), 2)
            self.assertEqual(decode(healthy_commits[1].payload), Delta(fields=(Field(5, 'default', 91.25, set_order=1),)))
            self.assertEqual(self.state.attempts['UpdateLayer'], 2)
            self.trace['healthy'] = healthy_observed
        attempts = dict(self.state.attempts)
        exported = await driver.command('export')
        self.assertEqual(exported['exported'], all_fields)
        self.assertEqual(exported['status'], failed['status'])
        if isolation:
            self.assertTrue((await driver.command('diagnostics-state'))['held'])
            self.assertTrue((await driver.command('release-diagnostics'))['retired'])
        self.assertEqual((await driver.command('close', exported=True))['type'], 'closed')
        # Wait for actual process retirement before inspecting the exported
        # layer; no original host, layer or runtime remains to supply data.
        await driver.cleanup()
        self.kit.remove(driver)
        from tools.live.kit_reference import digest
        self.assertEqual(digest(driver.output / 'retained.usda'), exported['sha256'])
        self.assertTrue((driver.output / 'retained.usda').read_bytes().startswith(b'#usda 1.0'))
        self.assertEqual(self.state.attempts, attempts, 'failure handling/export/retirement issued service work')
        self.assertEqual(len(self.commits()), 2)
        self.trace.update(oracle=self.oracle(), retained=all_fields,
                          export_sha256=exported['sha256'], failure=failed['status'])
