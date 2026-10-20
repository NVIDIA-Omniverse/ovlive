# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mac provisioning publishes the pinned, complete native build configuration."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.live import provision_macos as provision


class MacProvisionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.output = self.root / 'worker'
        self.python = self.root / 'Python SDK'
        self.cmake = str(self.root / 'tools/cmake')
        self.argv = ['--python-root', str(self.python), '--output', str(self.output),
                     '--cmake', self.cmake, '--parallel', '3']
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        # The client root of a synthetic repository whose root `.cache` is under the temporary directory.
        self.stack.enter_context(patch.object(provision, 'ROOT', self.root / 'client'))
        self.host = self.stack.enter_context(patch.object(provision.platforms, 'host',
            return_value=provision.platforms.get('macos-arm64')))
        self.usd = self.stack.enter_context(patch.object(provision.provision_macos_usd, 'main'))
        self.transport = self.stack.enter_context(patch.object(provision.provision_macos_transport, 'provision'))
        self.checkout = self.stack.enter_context(patch.object(provision.provision_macos_usd, 'checkout'))
        self.stack.enter_context(patch.object(provision.shutil, 'which',
            side_effect=lambda name: str(self.root / 'tools' / name)))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def test_complete_config_uses_pinned_sdks_and_selected_tools(self):
        provision.main(self.argv)
        config = json.loads((self.output / 'worker.json').read_text())
        self.assertEqual(config['schema_version'], 1)
        self.assertEqual(config['purpose'], 'macos-source-build')
        self.assertEqual(config['usd_profile'], 'stock-25.11')
        self.assertEqual(config['sdks'], dict(
            usd_root=str(self.root / '.cache/mac-sdk/install'), python_root=str(self.python),
            dependency_root=str(self.root / '.cache/mac-transport'),
            pybind11_root=str(self.root / '.cache/mac-pybind11')))
        self.assertEqual(config['tools'], dict(cmake=self.cmake,
            make=str(self.root / 'tools/make'), cc=str(self.root / 'tools/cc'), cxx=str(self.root / 'tools/c++'),
            protoc=str(self.root / '.cache/mac-transport/protobuf-release/bin/protoc'),
            grpc_cpp_plugin=str(self.root / '.cache/mac-transport/grpc-release/bin/grpc_cpp_plugin')))
        self.usd.assert_called_once_with(['--root', str(self.root / '.cache/mac-sdk'),
            '--python-root', str(self.python), '--cmake', self.cmake, '--parallel', '3'])
        self.transport.assert_called_once_with(self.root / '.cache/mac-transport', self.cmake, 3, 'arm64')
        self.checkout.assert_called_once_with(self.root / '.cache/mac-pybind11',
            'https://github.com/pybind/pybind11.git', '8a099e44b3d5f85b20f05828d919d2332a8de841')

    def test_bare_cmake_name_records_resolved_executable(self):
        argv = list(self.argv)
        argv[argv.index('--cmake') + 1] = 'cmake'
        provision.main(argv)
        config = json.loads((self.output / 'worker.json').read_text())
        self.assertEqual(config['tools']['cmake'], self.cmake)
        self.assertNotEqual(config['tools']['cmake'], str(Path('cmake').absolute()))
        usd_args = self.usd.call_args.args[0]
        self.assertEqual(usd_args[usd_args.index('--cmake') + 1], self.cmake)
        self.assertEqual(self.transport.call_args.args[1], self.cmake)

    def test_reuse_only_validates_both_sdk_closures_before_publishing_config(self):
        (self.root / '.cache/mac-pybind11/.git').mkdir(parents=True)
        provision.main([*self.argv, '--reuse-only'])
        self.assertIn('--reuse-only', self.usd.call_args.args[0])
        self.assertEqual(self.transport.call_args.kwargs, dict(reuse_only=True))
        self.assertEqual(self.checkout.call_args.kwargs, dict(reuse_only=True))
        self.assertTrue((self.output / 'worker.json').is_file())

    def test_other_host_rejects_before_provisioning_or_output(self):
        self.host.return_value = provision.platforms.get('linux-aarch64')
        with self.assertRaises(SystemExit):
            provision.main(self.argv)
        self.usd.assert_not_called()
        self.transport.assert_not_called()
        self.checkout.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_each_provisioning_failure_invalidates_existing_config(self):
        self.output.mkdir()
        existing = self.output / 'worker.json'
        for step in (self.usd, self.transport, self.checkout):
            with self.subTest(step=step):
                existing.write_text('previous configuration\n')
                def fail(*args):
                    self.assertFalse(existing.exists())
                    raise subprocess.CalledProcessError(1, 'native build')
                step.side_effect = fail
                with self.assertRaises(subprocess.CalledProcessError):
                    provision.main(self.argv)
                self.assertFalse(existing.exists())
                step.side_effect = None

    def test_config_is_absent_during_provisioning_and_replaced_atomically(self):
        self.output.mkdir()
        existing = self.output / 'worker.json'
        existing.write_text('previous configuration\n')
        for step in (self.usd, self.transport, self.checkout):
            step.side_effect = lambda *args: self.assertFalse(existing.exists())
        replace = Path.replace
        publications = []
        def publish(pending, destination):
            self.assertEqual(destination, existing)
            self.assertFalse(existing.exists())
            self.assertEqual(json.loads(pending.read_text())['purpose'], 'macos-source-build')
            publications.append(pending)
            return replace(pending, destination)
        with patch.object(Path, 'replace', publish):
            provision.main(self.argv)
        self.assertEqual(len(publications), 1)
        self.assertEqual(list(self.output.iterdir()), [existing])

    def test_publication_failure_keeps_config_unavailable_and_cleans_pending_file(self):
        with patch.object(Path, 'replace', side_effect=OSError('publication failed')):
            with self.assertRaisesRegex(OSError, 'publication failed'):
                provision.main(self.argv)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_invalid_parallelism_rejects_before_invalidating_config(self):
        self.output.mkdir()
        existing = self.output / 'worker.json'
        existing.write_text('previous configuration\n')
        for jobs in ('0', '65'):
            with self.subTest(jobs=jobs), self.assertRaises(SystemExit):
                provision.main([*self.argv[:-1], jobs])
        self.usd.assert_not_called()
        self.assertEqual(existing.read_text(), 'previous configuration\n')

    def test_pybind_pin_failure_does_not_publish_config(self):
        self.checkout.side_effect = ValueError('source differs from pin')
        with self.assertRaisesRegex(ValueError, 'differs from pin'):
            provision.main(self.argv)
        self.assertFalse((self.output / 'worker.json').exists())


if __name__ == '__main__':
    unittest.main()
