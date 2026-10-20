# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The license bundle and bill of materials derive only from pinned license files."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'tools/release/third_party_notices.py'
PINS = {'grpc': '1.65.4-1', 'protobuf': '5.26.0-2', 'abseil': '20240116.1-1', 're2': '20230301-1',
        'c-ares': '1.34.8-0', 'openssl': '3.5.9-0', 'zlib': '1.3.2-0'}


def load():
    spec = importlib.util.spec_from_file_location('third_party_notices', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ThirdPartyNoticesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.module = load()
        self.deps = self.root / 'deps'
        for name, (license_name, _, _) in self.module.COMPONENTS.items():
            real = self.deps / f'{PINS[name]}-synthetic_abi-static-release'
            (real / 'PACKAGE-LICENSES').mkdir(parents=True)
            (real / 'PACKAGE-LICENSES' / license_name).write_text(f'synthetic {name} license text\n')
            # Reading package metadata fails loudly: it is a directory.
            (real / 'PACKAGE-INFO.yaml').mkdir()
            (self.deps / f'{name}-release').symlink_to(real.name)
        decoy = self.deps / f'{PINS["openssl"]}-synthetic_abi-static-release/PACKAGE-LICENSES/openssl-LICENSE'
        decoy.write_text('Text::Template decoy, never bundled\n')
        self.pybind11 = self.root / 'pybind11/2.11.1-0'
        (self.pybind11 / 'PACKAGE-LICENSES').mkdir(parents=True)
        (self.pybind11 / 'PACKAGE-LICENSES/pybind11-LICENSE.txt').write_text('synthetic pybind11 license text\n')
        (self.pybind11 / 'PACKAGE-INFO.yaml').mkdir()
        self.output = self.root / 'output'
        self.output.mkdir()

    def run_tool(self, *arguments, pybind11=True):
        argv = ['--dependency-root', str(self.deps), '--config', 'release', '--output-dir', str(self.output),
                *arguments]
        if pybind11:
            argv += ['--pybind11-root', str(self.pybind11)]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return self.module.main(argv)

    def test_write_records_pinned_static_components_and_named_license_files(self):
        self.assertEqual(self.run_tool('--write'), 0)
        bom = json.loads((self.output / 'BOM.json').read_text())
        self.assertEqual(bom['schema_version'], 1)
        self.assertEqual(bom['scope'], json.loads((ROOT / 'BOM.json').read_text())['scope'])
        self.assertIn('client', bom['scope'])
        static = {c['name']: c for c in bom['components'] if c['linkage'] == 'static'}
        self.assertEqual({name: c['version'] for name, c in static.items()}, PINS)
        for component in bom['components']:
            fields = {'name', 'version', 'license', 'upstream', 'linkage', 'license_sha256'}
            if component['linkage'] == 'source':
                # Vendored source records where it lives; it may have no public upstream.
                self.assertEqual(set(component), fields | {'provenance'})
                if component['upstream'] is None:
                    continue
            else:
                self.assertEqual(set(component), fields)
            self.assertTrue(component['upstream'].startswith('https://github.com/'))
        linkage = {c['name']: c['linkage'] for c in bom['components']}
        self.assertEqual(linkage['flatbuffers'], 'header')
        self.assertEqual(linkage['pybind11'], 'header')
        for name in ('OpenUSD', 'TBB', 'Python'):
            self.assertEqual(linkage[name], 'host-shared')
        licenses = (self.output / 'THIRD_PARTY_LICENSES.txt').read_text()
        self.assertIn('==== openssl 3.5.9-0 (Apache-2.0) ====\n\nsynthetic openssl license text\n', licenses)
        self.assertNotIn('Text::Template', licenses)
        self.assertIn('synthetic pybind11 license text', licenses)
        text = json.dumps(bom) + licenses
        self.assertNotIn(str(self.root), text)

    def test_bom_lists_the_vendored_sha256_source_with_both_notices(self):
        self.assertEqual(self.run_tool('--write'), 0)
        for bom in (json.loads((self.output / 'BOM.json').read_text()), json.loads((ROOT / 'BOM.json').read_text())):
            entries = [c for c in bom['components'] if c['name'] == 'sha256']
            self.assertEqual(len(entries), 1)
            notices = (ROOT / 'THIRD_PARTY_NOTICES.md').read_text(encoding='utf-8')
            text = notices.split('## SHA-256 implementation (MIT and BSD-2-Clause)', 1)[1].split('```text\n', 1)[1].split('```', 1)[0]
            self.assertTrue(text.startswith('The MIT License\n'))
            self.assertEqual(entries[0], dict(
                name='sha256', version=None, license='MIT AND BSD-2-Clause', upstream=None, linkage='source',
                license_sha256=hashlib.sha256(text.encode('utf-8') + (ROOT / 'LICENSES/SHA256-BSD-2-Clause.txt').read_bytes()).hexdigest(),
                provenance="source/live/support/SHA256.cpp, derived from Zilong Tan's SHA-256 implementation, "
                           "itself derived from Allan Saddi's code; MIT and BSD-2-Clause texts in THIRD_PARTY_NOTICES.md"))
        licenses = (self.output / 'THIRD_PARTY_LICENSES.txt').read_text()
        self.assertNotIn('==== sha256', licenses)

    def test_check_accepts_unchanged_output_and_rejects_one_changed_byte(self):
        self.assertEqual(self.run_tool('--write'), 0)
        self.assertEqual(self.run_tool('--check'), 0)
        # Without a pybind11 root the checked-in pybind11 record is reused and must still match.
        self.assertEqual(self.run_tool('--check', pybind11=False), 0)
        for name in ('THIRD_PARTY_LICENSES.txt', 'BOM.json'):
            path = self.output / name
            original = path.read_bytes()
            path.write_bytes(original.replace(b'synthetic grpc', b'synthetic grpC', 1)
                             if name.endswith('.txt') else original.replace(b'1.65.4-1', b'1.65.4-2', 1))
            self.assertEqual(self.run_tool('--check'), 1)
            path.write_bytes(original)
        self.assertEqual(self.run_tool('--check'), 0)

    def test_check_without_pybind11_root_rejects_a_tampered_pybind11_section(self):
        self.assertEqual(self.run_tool('--write'), 0)
        path = self.output / 'THIRD_PARTY_LICENSES.txt'
        path.write_bytes(path.read_bytes().replace(b'synthetic pybind11', b'synthetic pybind12', 1))
        self.assertEqual(self.run_tool('--check', pybind11=False), 2)

    def test_resolved_directory_must_carry_the_pinned_version(self):
        link = self.deps / 'zlib-release'
        link.unlink()
        other = self.deps / '1.3.1-0-synthetic_abi-static-release'
        (other / 'PACKAGE-LICENSES').mkdir(parents=True)
        (other / 'PACKAGE-LICENSES/zlib-LICENSE.txt').write_text('synthetic zlib license text\n')
        link.symlink_to(other.name)
        self.assertEqual(self.run_tool('--write'), 2)
        self.assertFalse((self.output / 'BOM.json').exists())

    def test_generator_reads_only_the_named_license_files(self):
        read = []
        original_text, original_bytes = Path.read_text, Path.read_bytes

        def read_text(path, *arguments, **options):
            read.append(Path(path))
            return original_text(path, *arguments, **options)

        def read_bytes(path):
            read.append(Path(path))
            return original_bytes(path)

        with patch.object(Path, 'read_text', read_text), patch.object(Path, 'read_bytes', read_bytes):
            self.assertEqual(self.run_tool('--write'), 0)
        self.assertFalse([path for path in read if 'PACKAGE-INFO' in path.name or 'PACKAGE-DEPS' in path.name])
        outside = sorted(str(path) for path in read if self.root in path.parents and self.output not in path.parents)
        expected = sorted([str(self.deps / f'{name}-release/PACKAGE-LICENSES' / license_name)
                           for name, (license_name, _, _) in self.module.COMPONENTS.items()]
                          + [str(self.pybind11 / 'PACKAGE-LICENSES/pybind11-LICENSE.txt')])
        self.assertEqual(outside, expected)

    def test_components_equal_the_linked_archives_and_packman_pins(self):
        cmake = (ROOT / 'cmake/LiveGrpcDependencies.cmake').read_text()
        archives = set(re.findall(r'^live_archive\((\S+)\s', cmake, re.M))
        self.assertEqual(archives, set(self.module.COMPONENTS))
        self.assertEqual(self.module.pins(), PINS)
        bom = json.loads((ROOT / 'BOM.json').read_text())
        static = {c['name']: c['version'] for c in bom['components'] if c['linkage'] == 'static'}
        self.assertEqual(static, PINS)
        licenses = (ROOT / 'THIRD_PARTY_LICENSES.txt').read_text()
        for component in bom['components']:
            # Vendored source carries its license text in THIRD_PARTY_NOTICES.md, not the bundle.
            if component['license_sha256'] is not None and component['linkage'] != 'source':
                self.assertIn(f"==== {component['name']} {component['version']} ({component['license']}) ====\n",
                              licenses)


if __name__ == '__main__':
    unittest.main()
