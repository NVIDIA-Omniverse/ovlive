# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The native participant satisfies the same public workflow scenarios as Python."""
import asyncio
import os
import test_python_sample as contract
from test_scene import Driver


class NativeSampleTests(contract.PythonSampleTests):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        # Shared method names must not collide when both language suites run in
        # one common-runner artifact directory, including their cleanup ledgers.
        self.artifact = self.artifact / 'native-participant'
        self.artifact.mkdir(exist_ok=True)

    async def start(self, *, address=None, ca_bundle=None, tls_server_name=None, transport_security=None):
        self.output = self.artifact / (self.id().split('.')[-1] + '.sample')
        environment = dict(os.environ, LIVE_SAMPLE_ACCESS_TOKEN='synthetic-writer')
        options = []
        if ca_bundle is not None:
            options += ['--ca-bundle', str(ca_bundle)]
        if tls_server_name is not None:
            options += ['--tls-server-name', tls_server_name]
        if transport_security is not None:
            options += ['--transport-security', transport_security]
        process = await asyncio.create_subprocess_exec(os.environ['LIVE_TEST_PARTICIPANT_SAMPLE'],
            '--authority', address or self.address, '--output', str(self.output), *options,
            '--plug-info', os.environ['LIVE_TEST_PLUGIN_PLUG_INFO'], env=environment,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            **contract.PROCESS_GROUP)
        driver = Driver(process, self.artifact / (self.id().split('.')[-1] + '.sample.log'))
        self.drivers.append(driver)
        self.assertEqual(await driver.receive(), {'type': 'ready'})
        return driver

    async def test_native_sample_logger_progress_and_failure_witnesses(self):
        process = await asyncio.create_subprocess_exec(os.environ['LIVE_TEST_PARTICIPANT_TESTS'],
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            output, _ = await asyncio.wait_for(process.communicate(), 10)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        (self.artifact / 'native-participant-witnesses.log').write_bytes(output)
        self.assertEqual(process.returncode, 0, output.decode(errors='replace'))
        self.assertIn(b'native status progresses with blocked/full logger', output)

    async def test_native_sample_rejects_unobserved_status(self):
        driver = await self.start()
        for state in (0, 2, 10):
            reply = await self.command(driver, 'wait-status', state=state)
            self.assertNotEqual(reply['result'], 0)
        await self.create(driver)
        for state in (0, 10, 2.5):
            self.assertNotEqual((await self.command(driver, 'wait-status', state=state))['result'], 0)
        status = await self.command(driver, 'wait-status', state=2)
        self.assertEqual(status['result'], 0)
        self.assertGreater(status['status']['subject_id'], 0)
        await self.close(driver)

    async def test_native_sample_uncertain_session_never_reissues_mutation(self):
        import grpc
        import json
        driver = await self.start()
        before = self.state.attempts['CreateSession']
        self.state.plan('CreateSession', 'after_commit', code=grpc.StatusCode.UNAVAILABLE)
        created = await self.command(driver, 'create', base_layer_key='sample-base')
        self.assertEqual(created['result'], 19)
        self.assertNotEqual((await self.command(driver, 'create', base_layer_key='sample-base'))['result'], 0)
        outcome = await self.command(driver, 'session-outcome')
        self.assertEqual(outcome['result'], 19)
        self.assertEqual(outcome['error']['grpc_status'], 14)
        self.assertGreater(outcome['error']['attempt_id'], 0)
        self.assertEqual(self.state.attempts['CreateSession'], before + 1)
        await self.close(driver)
        events = [json.loads(line) for line in (self.output / 'events.jsonl').read_text().splitlines()]
        recorded = [e for e in events if e['type'] == 'session_outcome']
        self.assertTrue(recorded)
        self.assertEqual(recorded[-1]['error'], outcome['error'])

    async def test_native_sample_startup_failure_remains_failure_after_cleanup(self):
        import json
        process = await asyncio.create_subprocess_exec(os.environ['LIVE_TEST_PARTICIPANT_SAMPLE'],
            '--authority', 'invalid authority', '--output', str(self.artifact / 'invalid-startup'),
            '--plug-info', os.environ['LIVE_TEST_PLUGIN_PLUG_INFO'],
            env=dict(os.environ, LIVE_SAMPLE_ACCESS_TOKEN='synthetic-writer'),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        (self.artifact / 'invalid-startup.log').write_bytes(stdout + stderr)
        self.assertEqual(process.returncode, 2)
        replies = [json.loads(line) for line in stdout.splitlines()]
        self.assertNotEqual(replies[0]['result'], 0)
        self.assertTrue(replies[-1]['closed'])
        self.assertNotIn(b'synthetic-writer', stdout + stderr)
