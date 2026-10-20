# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The source provisioner preserves pins and never marks a failed install complete."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.live import provision_macos_transport as transport


class MacTransportTests(unittest.TestCase):
    def install_fixture(self, root, name):
        for relative in transport.REQUIRED_ARTIFACTS[name]:
            path = root / f'{name}-release' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'installed artifact')

    def test_artifact_inventory_rejects_missing_empty_and_corrupted_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.install_fixture(root, 'abseil')
            install = root / 'abseil-release'
            complete = transport.installed_artifacts(install, 'abseil')
            library = install / 'lib/libabsl_base.a'
            library.write_bytes(str(root.resolve()).encode())
            with self.assertRaisesRegex(ValueError, 'build-machine path'):
                transport.installed_artifacts(install, 'abseil')
            library.write_bytes(b'corrupted')
            self.assertNotEqual(transport.installed_artifacts(install, 'abseil'), complete)
            library.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'Missing or empty'):
                transport.installed_artifacts(install, 'abseil')
            library.unlink()
            with self.assertRaisesRegex(ValueError, 'Missing or empty'):
                transport.installed_artifacts(install, 'abseil')

    def test_failure_leaves_no_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(transport.subprocess, 'check_output', return_value='tool version'), \
                    patch.object(transport, 'run', side_effect=subprocess.CalledProcessError(1, 'git')):
                with self.assertRaises(subprocess.CalledProcessError):
                    transport.provision(root, 'cmake', 4, 'arm64')
            self.assertFalse(list(root.glob('*-release/source-pin.json')))

    def test_source_pins_install_layout_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def fake_run(command, cwd=None):
                if command[:2] == ['git', 'init']:
                    (Path(command[2]) / '.git').mkdir(parents=True)
                elif command[0] == 'make' and command[1] == 'install_sw':
                    staged = Path(command[2].removeprefix('DESTDIR=')) / 'usr/local'
                    for relative in transport.REQUIRED_ARTIFACTS['openssl']:
                        path = staged / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b'installed artifact')
                elif command[0] == 'cmake' and command[1] == '--install':
                    self.install_fixture(root, Path(command[2]).name)
            with patch.object(transport.subprocess, 'check_output',
                              side_effect=lambda command, **kwargs: '' if 'status' in command else 'tool version'), \
                    patch.object(transport, 'run', side_effect=fake_run) as run:
                self.assertEqual(transport.provision(root, 'cmake', 4, 'arm64'), root)
                commands = [call.args[0] for call in run.call_args_list]
                fetches = [c for c in commands if c[0] == 'git' and 'fetch' in c]
                self.assertEqual([c[-1] for c in fetches], [pin for _, pin in transport.PINS.values()])
                grpc = next(c for c in commands if '-DgRPC_INSTALL=ON' in c)
                for dependency in ('ABSL', 'CARES', 'PROTOBUF', 'RE2', 'SSL', 'ZLIB'):
                    self.assertIn(f'-DgRPC_{dependency}_PROVIDER=package', grpc)
                path_map = f'-ffile-prefix-map={root.resolve()}=transport'
                for command in commands:
                    if command[:2] == ['cmake', '-S']:
                        self.assertIn(f'-DCMAKE_C_FLAGS={path_map}', command)
                        self.assertIn(f'-DCMAKE_CXX_FLAGS={path_map}', command)
                openssl = next(c for c in commands if Path(c[0]).name == 'Configure')
                self.assertIn('--prefix=/usr/local', openssl)
                self.assertIn('--openssldir=/etc/ssl', openssl)
                self.assertIn('-ffile-prefix-map=../../source/openssl=transport/openssl', openssl)
                self.assertEqual(len(list(root.glob('*-release/source-pin.json'))), 7)
                run.reset_mock()
                transport.provision(root, 'cmake', 4, 'arm64', reuse_only=True)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(run.call_args.args[0], [root / 'protobuf-release/bin/protoc', '--version'])
                marker = root / 'abseil-release/source-pin.json'
                completed = marker.read_text()
                mismatched = json.loads(completed)
                mismatched['identity']['toolchain']['sdk'] = 'different system SDK'
                marker.write_text(json.dumps(mismatched))
                run.reset_mock()
                with self.assertRaisesRegex(ValueError, 'No matching completed transport SDK'):
                    transport.provision(root, 'cmake', 4, 'arm64', reuse_only=True)
                run.assert_not_called()
                marker.write_text(completed)
                library = root / 'abseil-release/lib/libabsl_base.a'
                library.write_bytes(b'corrupted')
                with self.assertRaisesRegex(ValueError, 'differ from their completion marker'):
                    transport.provision(root, 'cmake', 4, 'arm64')
                library.unlink()
                with self.assertRaisesRegex(ValueError, 'Missing or empty'):
                    transport.provision(root, 'cmake', 4, 'arm64')
                library.write_bytes(b'installed artifact')
                (root / 'abseil-release/source-pin.json').write_text('{}')
                (root / 'abseil-release/stale.dylib').write_text('old artifact')
                transport.provision(root, 'cmake', 4, 'arm64')
                self.assertFalse((root / 'abseil-release/stale.dylib').exists())
                retired = list((root / 'retired').glob('abseil-*/stale.dylib'))
                self.assertEqual(len(retired), 1)
                self.assertEqual(retired[0].read_text(), 'old artifact')

    def test_dirty_source_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'source/abseil/.git').mkdir(parents=True)
            def output(command, **kwargs):
                return ' M CMakeLists.txt' if 'status' in command else 'tool version'
            with patch.object(transport.subprocess, 'check_output', side_effect=output), \
                    patch.object(transport, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'local changes'):
                    transport.provision(root, 'cmake', 4, 'arm64')
                run.assert_not_called()

    def test_reuse_only_refuses_missing_sdk_without_fetching_or_building(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(transport.subprocess, 'check_output', return_value='tool version'), \
                patch.object(transport, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Cached transport source checkout is missing'):
                transport.provision(Path(directory), 'cmake', 4, 'arm64', reuse_only=True)
            run.assert_not_called()

    def test_rejects_unqualified_architecture_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'transport'
            with self.assertRaisesRegex(ValueError, 'requires arm64'):
                transport.provision(root, 'cmake', 4, 'x86_64')
            self.assertFalse(root.exists())
