# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Suite selection cannot widen the supervised macOS qualification boundary."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.live import test as runner
from tools.live.test_support import ProcessResult


class PlatformAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.artifacts = Path(temporary.name) / 'runs'

    def run_main(self, arguments, host='darwin', workload=None):
        before = set(self.artifacts.iterdir()) if self.artifacts.exists() else set()
        with patch.object(sys, 'platform', host), \
                patch.object(sys, 'argv', ['test.py', *arguments, '--seed', '42',
                                          '--artifact-dir', str(self.artifacts)]), \
                patch.object(runner, 'fingerprint', return_value='synthetic-source'), \
                patch.object(runner, 'identity', return_value='synthetic-toolchain'), \
                patch.object(runner.platform, 'platform', return_value='synthetic-platform'), \
                patch.object(runner, 'native_identity', return_value={'sanitizers': False}) as native, \
                patch.object(runner.subprocess, 'run', side_effect=AssertionError('launched a preflight process')) as preflight, \
                patch.object(runner, 'run_process', side_effect=workload or AssertionError('launched an unsupported workload')) as process, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = runner.main()
        directory = next(iter(set(self.artifacts.iterdir()) - before))
        return status, directory, json.loads((directory / 'manifest.json').read_text()), native, preflight, process

    def test_macos_rejects_every_unqualified_suite_before_preflight_or_workload(self):
        denied = sorted(set(runner.SUITES) - {'foundation', 'uri', 'wire-envelope', 'lock-noalloc'})
        status, _, manifest, native, preflight, process = self.run_main([
            '--suite', ','.join(denied), '--reference-python', sys.executable,
            '--reference-openssl', '/synthetic/openssl', '--runtime-driver', '/synthetic/runtime',
            '--plugin-driver', '/synthetic/plugin', '--unary-driver', '/synthetic/unary'])
        self.assertEqual(status, 2)
        self.assertEqual([result['suite'] for result in manifest['results']], denied)
        for result in manifest['results']:
            self.assertEqual(result['outcome'], 'infrastructure_error')
            self.assertIn(result['suite'] + ' is unsupported on macOS', result['reason'])
        native.assert_not_called()
        preflight.assert_not_called()
        process.assert_not_called()

    def test_tags_packages_scenarios_and_replay_cannot_admit_reference_workloads(self):
        selections = [['--tag', 'reference'], ['--package', 'reference'],
                      ['--suite', 'reference-service', '--scenario', 'test_auth'],
                      ['--suite', 'reference-service', '--shard', '0/2']]
        for arguments in selections:
            with self.subTest(arguments=arguments):
                status, directory, manifest, _, preflight, process = self.run_main(arguments)
                self.assertEqual(status, 2)
                self.assertTrue(manifest['results'])
                self.assertTrue(all('unsupported on macOS' in result['reason'] for result in manifest['results']))
                preflight.assert_not_called()
                process.assert_not_called()
        status, _, manifest, _, preflight, process = self.run_main(['--replay', str(directory / 'replay.json')])
        self.assertEqual(status, 2)
        self.assertTrue(all('unsupported on macOS' in result['reason'] for result in manifest['results']))
        preflight.assert_not_called()
        process.assert_not_called()

    def test_macos_admits_foundation_and_sdk_free_components(self):
        def failure(command, **options):
            options['output'].write_text('synthetic test failure\n')
            return ProcessResult('failed', 1, 1, 23, False, True)
        status, _, manifest, native, _, process = self.run_main([
            '--tag', 'pr', '--uri-library', '/synthetic/uri', '--wire-driver', '/synthetic/wire',
            '--lock-driver', '/synthetic/lock'], workload=failure)
        self.assertEqual(status, 1)
        self.assertEqual([result['suite'] for result in manifest['results']],
                         ['foundation', 'uri', 'wire-envelope', 'lock-noalloc'])
        self.assertTrue(all(result['outcome'] == 'failed' for result in manifest['results']))
        self.assertEqual(native.call_count, 3)
        self.assertEqual(process.call_count, 4)

    def test_linux_and_windows_keep_every_implemented_suite_admitted(self):
        for host in ('linux', 'win32'):
            with self.subTest(host=host), patch.object(sys, 'platform', host):
                self.assertTrue(all(runner.suite_platform_error(suite) is None for suite in runner.SUITES))
