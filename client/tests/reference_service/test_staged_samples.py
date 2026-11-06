# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The samples of a staged package beside seeded writers: the writer-fuzz `staged-samples` case.

The samples run from the stage that tools/live/build.py produced, with the environment the build guide gives a user,
and only the writer cases run this module: no lane discovers it, since only the package campaign has a stage. A
semantic or accounting mismatch of the case reduces its seeded plan through fresh trials (staged_trial), and a replay
of the reduced plan runs here too, with the stage the reduction recorded.
"""
import asyncio
import dataclasses
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from graph_service import GraphService
import plan_replay
import scene_graph_model as graph_model
from scene_graph_wire import decode as graph_decode, encode as graph_encode
import staged_sample_history as history
from staged_sample_history import SAMPLES, listed_sessions
import test_plugin as plugin_support
import test_python_sample as sample_support
import test_scene as scene_support
from tools.live import build, platforms
from tools.live import writer_campaign as campaign
from writer_history import WriterMismatch, read_json
from writer_reduction import publish as publish_writer, reduce_history as reduce_writers, staged_domain

# The layer each participant sample creates, and the base layer key both samples create their sessions under.
LAYER = 'sample.live'
BASE = 'staged-samples'
# The campaign case's test, whose evaluator (staged_trial) every reduction trial and replay runs.
CASE_TEST = 'test_staged_samples_converge_beside_seeded_writers'
# Trials of one staged-samples reduction, the final confirmation included.
REDUCTION_TRIALS = 24
# Environment variables a user's shell carries that the samples may read; nothing else reaches them.
USER_ENVIRONMENT = {'PATH', 'HOME', 'TMPDIR', 'TEMP', 'TMP', 'LANG', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT',
                    'USERPROFILE'}
# The value field a set or define authors on each prim's `value` attribute.
DEFAULT = 'default'
# Run by the stage's SDK Python on a sample's exported layer: its prim paths and each prim's int `value` default, as
# JSON in the form StagedSampleTests.scene returns.
READ_SCENE = '''
import json, sys
from pxr import Sdf
layer = Sdf.Layer.FindOrOpen(sys.argv[1])
paths, values = [], {}
def visit(path):
    if path.IsPrimPath():
        paths.append(str(path))
        attribute = layer.GetAttributeAtPath(path.AppendProperty('value'))
        if attribute and type(attribute.default) is int:
            values[str(path)] = attribute.default
layer.Traverse(Sdf.Path.absoluteRootPath, visit)
print(json.dumps(dict(paths=sorted(paths), values=values)))
'''


def user_environment(stage, support):
    """The build guide's environment for running the package's samples, over the user-shell variables alone."""
    usd, python = Path(support['sdks']['usd_root']), Path(support['sdks']['python_root'])
    environment = {key: value for key, value in os.environ.items() if key.upper() in USER_ENVIRONMENT}
    environment.update(PYTHONDONTWRITEBYTECODE='1',
                       PYTHONPATH=os.pathsep.join([str(stage / 'python'), str(usd / 'lib/python')]))
    if os.name == 'nt':
        environment['PATH'] = os.pathsep.join([str(usd / 'lib'), str(usd / 'bin'), str(python),
                                               environment.get('PATH', '')])
    else:
        environment['LD_LIBRARY_PATH'] = os.pathsep.join([str(usd / 'lib'), str(python / 'lib')])
    return environment


class CorruptStagedSampleService(GraphService):
    """The named test-only defect corrupt-w0:2: the service commits -1 in place of `state.staged_corrupt`, the value of
    writer 0's round-2 edit when the plan retains it, while the writer's call succeeds. Values are distinct, so the
    value names that edit alone. This exercises staged-samples failure reduction; no production fault switch exists.
    """
    async def apply_update(self, method, attempt, session, old, header, payload, context):
        value = getattr(self.state, 'staged_corrupt', None)
        if value is not None and payload:
            delta = graph_decode(payload)
            if any(field.key == DEFAULT and field.value == value for field in delta.fields):
                payload = graph_encode(dataclasses.replace(delta, fields=tuple(
                    dataclasses.replace(field, value=-1) if field.key == DEFAULT else field for field in delta.fields)))
        return await super().apply_update(method, attempt, session, old, header, payload, context)


class StagedSampleTests(unittest.IsolatedAsyncioTestCase):
    graph_creation = True
    asyncSetUp = plugin_support.PluginTests.asyncSetUp
    cleanup = plugin_support.PluginTests.cleanup
    close = plugin_support.PluginTests.close

    def stage(self):
        if not os.environ.get('LIVE_TEST_STAGED_PACKAGE'):
            self.fail('the staged-samples case runs only in the package writer campaign, which names a stage')
        stage = Path(os.environ['LIVE_TEST_STAGED_PACKAGE'])
        record = build.validate_stage(stage, expected_platform=platforms.host().name)
        return stage, record['support_inputs']

    def native_files(self):
        """The directory the staged samples and the SDK's Python write and read their files in, and run in: the
        test's artifacts, or on Windows a short directory under the suite's scratch directory. USD's file functions
        fail there for a path of MAX_PATH (260) characters or more, which a reducer's standalone replay nests the
        artifacts deeper than; cleanup copies that directory into the artifacts as `native-files`, as far as it can."""
        if os.name != 'nt':
            return self.artifact
        if getattr(self, 'native_directory', None) is None:
            self.native_directory = Path(tempfile.mkdtemp())

            async def keep():
                try:
                    shutil.copytree(self.native_directory, self.artifact / 'native-files', dirs_exist_ok=True)
                except (OSError, shutil.Error):
                    pass  # Evidence only: a file a process still holds open may not copy.
            self.addAsyncCleanup(keep)
        return self.native_directory

    async def sample(self, name, command):
        """Starts the staged participant sample `name` from `command`, as the samples guide runs it."""
        campaign.admit_client(self)
        output = self.native_files() / f'staged-{name}.sample'
        process = await asyncio.create_subprocess_exec(
            *command, '--authority', self.address, '--output', str(output), cwd=str(self.native_files()),
            env=dict(self.environment, LIVE_SAMPLE_ACCESS_TOKEN='synthetic-writer'),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            **sample_support.PROCESS_GROUP)
        driver = scene_support.Driver(process, self.artifact / f'staged-{name}.sample.log')
        self.drivers.append(driver)
        self.sample_drivers.append(driver)
        self.assertEqual(await driver.receive(), {'type': 'ready'})
        return driver

    @staticmethod
    async def ask(driver, operation, **fields):
        reply = await driver.command(json.dumps(dict(operation=operation, **fields)))
        if reply.get('type') != 'reply':
            raise AssertionError(f'sample {operation} replied {reply}')
        return reply

    async def writer(self, session):
        """A seeded plugin writer attached to the sample layer of `session`."""
        campaign.admit_client(self)
        process = await asyncio.create_subprocess_exec(
            os.environ['LIVE_TEST_PLUGIN_DRIVER'], self.address, session, str(self.seed),
            os.environ['LIVE_TEST_PLUGIN_PLUG_INFO'], LAYER, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, limit=1024 * 1024)
        driver = scene_support.Driver(process, self.artifact / f'staged-writer-{len(self.drivers)}.plugin.log')
        self.drivers.append(driver)
        ready = await driver.receive()
        self.assertEqual((ready['type'], ready['result'], ready['state'], ready['pending'], ready['fields']),
                         ('ready', 0, 2, 0, {}), ready)
        return driver

    def layer(self, session):
        return self.state.sessions[session].layers[LAYER]

    def scene(self, session):
        """The service's prims of the sample layer of `session` and their `value` defaults."""
        nodes = self.layer(session).model.state
        paths = {node['id']: node['path'] for node in nodes}
        return dict(paths=sorted(node['path'] for node in nodes if node['type'] == graph_model.PRIM),
                    values={paths[node['parent']]: node['fields'][DEFAULT] for node in nodes
                            if node['type'] == graph_model.ATTRIBUTE and node['name'] == 'value'
                            and type(node['fields'].get(DEFAULT)) is int})

    def account(self, plan, name, session, boundaries):
        """The layer's commit accounting (staged_sample_history.account) after the rounds `boundaries` counts, which
        holds the layer's commit count after each round run; raises WriterMismatch. Returns the scene the commits
        make."""
        commits = self.layer(session).commits
        nodes = {node['id']: node for node in self.layer(session).model.state}
        attributes = {nodes[node['parent']]['path']: node['id'] for node in nodes.values()
                      if node['type'] == graph_model.ATTRIBUTE and node['name'] == 'value'}
        return history.account(plan, name, [(commit.sequence, commit.writer, commit.payload) for commit in commits[1:]],
                               attributes, boundaries[:-1], len(boundaries))

    async def exported_scene(self, name, driver, number):
        """The sample's own live layer, as its `export` command writes it and the stage's SDK Python reads it: every
        prim path and each prim's `value` default."""
        path = self.native_files() / f'staged-{name}-{number}.usda'
        exported = await self.ask(driver, 'export', path=str(path))
        self.assertEqual(exported['result'], 0, exported)
        reader = await asyncio.create_subprocess_exec(
            self.python, '-B', '-c', READ_SCENE, str(path), cwd=str(self.native_files()), env=self.environment,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(reader.communicate(), 30)
        finally:
            if reader.returncode is None:
                reader.kill()
                await reader.wait()
        self.assertEqual(reader.returncode, 0, stderr.decode(errors='replace')[-4096:])
        return json.loads(stdout)

    async def converged_sample(self, name, driver, final, expected):
        """Polls the sample's status, its live-layer value and an export of its whole live layer, an observable
        predicate under a failure-only watchdog, until it has applied the layer through `final` and its own scene is
        `expected`, every prim and value; returns the last status."""
        async def settled():
            number = 0
            while True:
                status = (await self.ask(driver, 'status'))['status']
                shown = await self.ask(driver, 'inspect')
                if status['applied_sequence'] >= final and shown['value'] == expected['values']['/Example']:
                    scene = await self.exported_scene(name, driver, number)
                    number += 1
                    if scene == expected:
                        return status, shown
        status, shown = await asyncio.wait_for(settled(), 30)
        self.assertEqual((status['state'], status['result'], status['pending_write_count'],
                          status['buffered_edit_count'], status['applied_sequence'], shown['result']),
                         (2, 0, 0, 0, final, 0), status)
        return status

    async def close_sample(self, driver):
        """Close a participant sample through its checked path: its `close` command, a clean exit and no drain
        failure."""
        closed = await self.ask(driver, 'close')
        self.assertEqual((closed['result'], closed.get('closed'), closed['logging']['dropped'],
                          closed['logging']['error']), (0, True, 0, None), closed)
        try:
            self.assertEqual(await asyncio.wait_for(driver.process.wait(), 10), 0)
        except TimeoutError as error:
            evidence = self.artifact / (driver.log.stem + '-exit-timeout')
            evidence.mkdir(exist_ok=False)
            state = dict(pid=driver.process.pid, returncode=driver.process.returncode,
                         drainer_done=driver.drainer.done(), sample=driver.log.stem)
            try:
                if os.name == 'nt' and driver.process.returncode is None:
                    helper = Path(__file__).resolve().parents[2] / 'tools/live/windows_dump.py'
                    collector = await asyncio.create_subprocess_exec(
                        sys.executable, '-B', str(helper), str(driver.process.pid), str(evidence / 'child.dmp'),
                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                    try:
                        output, _ = await asyncio.wait_for(collector.communicate(), 20)
                        (evidence / 'collector.log').write_bytes(output)
                        state['collector_returncode'] = collector.returncode
                    except TimeoutError:
                        state['collector_timeout'] = True
                    finally:
                        if collector.returncode is None:
                            collector.kill()
                            await collector.wait()
                    stage, _ = self.stage()
                    build_tree = Path(os.environ['LIVE_TEST_PLUGIN_DRIVER']).parent
                    names = ('live_participant_sample.exe', 'omni_usd_live.dll') if 'native' in driver.log.stem else ('omni_usd_live.dll',)
                    for name in names:
                        shutil.copy2(stage / 'bin' / name, evidence / name)
                        for pdb in build_tree.rglob(Path(name).stem + '.pdb'):
                            if pdb.stat().st_size <= 512 * 1024 * 1024:
                                shutil.copy2(pdb, evidence / pdb.name)
            except Exception as capture_error:
                state['capture_error'] = repr(capture_error)
            (evidence / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
            raise AssertionError('participant exit timed out: ' + json.dumps(state)) from error
        await driver.drainer
        self.assertIsNone(driver.failure)

    async def retire_writers(self):
        """Close every participant still running through its checked path, samples with close_sample and writers with
        PluginTests.close, and check that every one that already exited did so cleanly; raises when any failed, so no
        reduction accepts a mismatch beside a crashed participant."""
        for driver in self.drivers:
            if driver.process.returncode is None:
                await (self.close_sample(driver) if driver in getattr(self, 'sample_drivers', ()) else
                       self.close(driver))
            else:
                await driver.drainer
                self.assertIsNone(driver.failure)
                self.assertEqual(driver.process.returncode, 0, driver.log.read_text()[-8192:])

    async def test_staged_samples_converge_beside_seeded_writers(self):
        await self.staged_with_reduction(history.generate(self.seed, getattr(self, 'writer_clients', 4),
                                                          getattr(self, 'writer_steps', 4)))

    async def staged_with_reduction(self, plan):
        """Run a staged-samples plan; a WriterMismatch is reduced through fresh trials and still fails."""
        try:
            await self.exercise(plan)
        except WriterMismatch as error:
            publish_writer(self.artifact / 'staged-samples-mismatch.json', dict(signature=error.signature,
                                                                               evidence=error.evidence))
            # The mismatch counts only once every participant closed cleanly; a crashed one raises a plain error here,
            # which is not reduced. The mismatch is published first, so it stays on record either way.
            await self.retire_writers()
            # Retire the original fixture before launching any reduction trial.
            await self.cleanup()
            self._cleanups.clear()
            summary = await reduce_writers(plan, error.signature, self.artifact / 'staged-samples-reduction',
                                           self.staged_trial, max_trials=REDUCTION_TRIALS, domain=staged_domain(),
                                           context=plan_replay.context('staged-samples'))
            print('staged-samples reduction: ' + json.dumps(summary, sort_keys=True), flush=True)
            raise  # Reduction never turns the original failure into a pass.

    async def staged_trial(self, plan, directory, *, automatic_reduction=False):
        """One fresh fixture, service, set of staged samples and native writers executing `plan`; the reducer's
        evaluator. A campaign case's client budget applies to the trial, and its peak counts toward the case's."""
        history.validate(plan)
        directory.mkdir(parents=True, exist_ok=True)
        fixture = StagedSampleTests(CASE_TEST)
        fixture.artifact_override = directory
        fixture.seed_override = plan['master_seed']
        if plan.get('test_service_fault') == history.CORRUPT:
            fixture.graph_service_factory = CorruptStagedSampleService
        if hasattr(self, 'writer_process_limit'):
            fixture.writer_process_limit = self.writer_process_limit
        try:
            return await plugin_support.fresh_trial(
                fixture, fixture.staged_with_reduction(plan) if automatic_reduction else fixture.exercise(plan))
        finally:
            if hasattr(self, 'writer_process_limit'):
                self.writer_peak_clients = max(getattr(self, 'writer_peak_clients', 0),
                                               getattr(fixture, 'writer_peak_clients', 0))

    async def exercise(self, plan):
        """Run a staged-samples plan (staged_sample_history), a generated one or a selection of one: its samples create
        their layers, its writers join them, and every participant authors its edit of each round."""
        history.validate(plan)
        stage, support = self.stage()
        self.environment = user_environment(stage, support)
        self.python = support['tools']['python']
        self.sample_drivers = []
        names = [name for name in SAMPLES if name in plan['samples']]
        rounds = plan['rounds']
        # The test-only defect corrupt-w0:2 corrupts writer 0's round-2 edit, when the plan holds it.
        self.state.staged_corrupt = next((op['value'] for edits in plan['writers'] for op in edits
                                          if op['edit'] == 'w0:2'), None) if plan.get('test_service_fault') else None
        # Seeds before workers: the plan is on disk before any participant starts.
        (self.artifact / 'staged-samples.plan.json').write_text(json.dumps(plan, indent=2, sort_keys=True) + '\n')
        campaign.observe_workload(self, plan)
        diagnostics_sample, participant_sample = (stage / relative for relative in build.samples(platforms.host().name))
        commands = dict(native=[str(participant_sample), '--plug-info', str(stage / 'resources/plugInfo.json')],
                        python=[support['tools']['python'], '-B', '-m', 'omni.usd.live.sample'])
        attempts = dict(self.state.attempts)
        samples, sessions = {}, {}
        for index, name in enumerate(names):
            samples[name] = await self.sample(name, commands[name])
            created = await self.ask(samples[name], 'create', base_layer_key=BASE, name=f'staged-{name}')
            self.assertEqual(created['result'], 0, created)
            self.assertEqual(len(created['metadata']), index + 1, created)
            sessions[name] = created['session']['session_id']
            self.assertEqual(self.scene(sessions[name]), dict(paths=['/Example'], values={'/Example': 0}))
        writers = [await self.writer(sessions[layer]) for layer in plan['layers']]
        for name in names:
            self.assertEqual(len(self.layer(sessions[name]).commits), 1)  # The creation.
        boundaries = {name: [] for name in names}

        async def sample_edit(name, edit):
            self.assertEqual((await self.ask(samples[name], 'edit', value=edit['value']))['result'], 0)
            saved = await self.ask(samples[name], 'save')
            self.assertEqual(saved['result'], 0, saved)

        async def writer_edit(index, edit):
            authored = await writers[index].command(f"usd-prim {edit['kind']} {edit['path'][1:]} {edit['value']}")
            self.assertEqual(authored['result'], 0, authored)
            saved = await writers[index].command('save-admitted 10000')
            self.assertEqual(saved['result'], 0, saved)

        for number in range(rounds):
            await asyncio.gather(*(sample_edit(name, plan['samples'][name][number]) for name in names),
                                 *(writer_edit(index, edits[number]) for index, edits in enumerate(plan['writers'])))
            # Each participant saved its edit of the round, so each layer holds exactly its participants' edits, which
            # the accounting checks round by round, so a mismatch is raised in the round that commits it.
            for name in names:
                commits = len(self.layer(sessions[name]).commits) - 1
                boundaries[name].append(commits)
                self.account(plan, name, sessions[name], boundaries[name])
                self.assertEqual(commits, (number + 1) * (1 + plan['layers'].count(name)))
        edits = rounds * (len(names) + len(plan['writers']))
        self.assertEqual({method: self.state.attempts.get(method, 0) - attempts.get(method, 0)
                          for method in ('CreateSession', 'CreateLayer', 'UpdateLayer')},
                         dict(CreateSession=len(names), CreateLayer=len(names), UpdateLayer=edits))
        self.assertFalse(self.state.rejections)
        for name in names:
            expected = self.account(plan, name, sessions[name], boundaries[name])
            self.assertEqual(self.scene(sessions[name]), expected)
            final = self.layer(sessions[name]).commits[-1].sequence
            command = 'wait-prims ' + str(len(expected['paths'])) + ''.join(
                f" {path} {expected['values'][path]}" for path in expected['paths'])
            for index, layer in enumerate(plan['layers']):
                if layer == name:
                    shown = await writers[index].command(command)
                    self.assertEqual((shown['result'], shown['paths'], shown['values']),
                                     (0, expected['paths'], expected['values']), shown)
            await self.converged_sample(name, samples[name], final, expected)
            observer = await self.writer(sessions[name])
            shown = await observer.command(command)
            self.assertEqual((shown['result'], shown['paths'], shown['values']),
                             (0, expected['paths'], expected['values']), shown)
            await self.close(observer)
        # The read-only diagnostics sample lists exactly the sample sessions.
        campaign.admit_client(self)
        listing = await asyncio.create_subprocess_exec(
            str(diagnostics_sample), '--authority', self.address, '--base-layer-key', BASE, '--timeout-ms', '5000',
            cwd=str(self.native_files()), env=dict(self.environment, LIVE_SERVICES_TOKEN='synthetic-writer'),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(listing.communicate(), 30)
        finally:
            if listing.returncode is None:
                listing.kill()
                await listing.wait()
        (self.artifact / 'staged-diagnostics.sample.stdout').write_bytes(stdout)
        (self.artifact / 'staged-diagnostics.sample.stderr').write_bytes(stderr)
        self.assertEqual(listing.returncode, 0, stderr.decode(errors='replace')[-4096:])
        self.assertEqual(listed_sessions(stderr), [(0, len(names))])
        for name in names:
            await self.close_sample(samples[name])
        for driver in writers:
            await self.close(driver)
        self.assertEqual(self.state.attempts.get('UpdateLayer', 0) - attempts.get('UpdateLayer', 0), edits)

    async def test_staged_samples_corrupt_reduction_reexecutes_staged_samples_and_service(self):
        """The defect commits -1 for writer 0's round-2 edit; the failure keeps that writer, the native sample whose
        layer it joins and round 2, with the round that defines writer 0's prim when that edit sets it. The original
        plan passes without the defect, the minimal plan replays to the same signature with it and passes without it,
        and the recorded plans replay through the standalone command."""
        plan = dict(history.generate(self.seed, getattr(self, 'writer_clients', 2), getattr(self, 'writer_steps', 4)),
                    test_service_fault=history.CORRUPT)
        campaign.observe_workload(self, plan)
        edit = plan['writers'][0][2]
        signature = dict(reason='commit carries no single planned edit', layer=plan['layers'][0], path=edit['path'])
        # A set of writer 0's own prim needs the round whose edit defines it.
        rounds = sorted({2, next(op['ordinal'] for op in plan['writers'][0] if op['kind'] == 'define')}) \
            if edit['path'] != '/Example' else [2]
        directory = self.artifact / 'staged-samples-reducer-selftest'

        def compared(result):
            return {key: value for key, value in result.items() if key != 'evidence'}
        first = await self.staged_trial(plan, directory, automatic_reduction=True)
        self.assertEqual(compared(first), dict(outcome='mismatch', signature=signature, cleanup_complete=True))
        root = directory / 'staged-samples-reduction'
        summary = read_json(root / 'reduction.json')
        self.assertEqual(summary['status'], 'complete', summary)
        self.assertTrue(summary['verified'] and summary['deletion_minimal'], summary)
        self.assertEqual((summary['retained_units'], summary['retained_edits']),
                         ([f'sample:{plan["layers"][0]}', 'writer:0', *(f'round:{r}' for r in rounds)],
                          [name for r in rounds for name in (f'{plan["layers"][0]}:{r}', f'w0:{r}')]))
        self.assertEqual(summary['context']['kind'], 'staged-samples')
        self.assertEqual(history.validate(read_json(root / 'original.plan.json')), plan)
        best = history.validate(read_json(root / 'best.plan.json'))
        replay = await self.staged_trial(best, directory / 'replay')
        self.assertEqual(compared(replay), compared(first))
        for name, selected in (('correct-service', best), ('original-correct-service', plan)):
            repaired = {key: value for key, value in selected.items() if key != 'test_service_fault'}
            self.assertEqual(await self.staged_trial(repaired, directory / name),
                             dict(outcome='passed', cleanup_complete=True))
        await plan_replay.check_standalone(self, root)
