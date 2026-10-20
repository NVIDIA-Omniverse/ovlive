# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kit lane qualifies a released packman package from its own manifest, and a build tree from its sidecars."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile

from tools.live import kit_reference as kit, platforms
from tools.release import package

KIT_PROFILE = '{"schema_version":1,"usd_profile":"kit-25.11"}'


def pack_zip(command, check):
    """A stand-in for `packman pack -c zip` that writes `<parent>@<folder>.zip` of the folder's files."""
    tree, output = Path(command[-1]), Path(command[command.index('-o') + 1])
    with zipfile.ZipFile(output / f'{tree.parent.name}@{tree.name}.zip', 'w') as writer:
        for path in sorted(tree.rglob('*')):
            if path.is_file():
                writer.write(path, path.relative_to(tree).as_posix())


class KitReleaseLaneTests(unittest.TestCase):
    def setUp(self):
        # The archive fixture stages synthetic package bytes; the repository's archive and package code make the
        # Kit-profile release package from them as the release candidate does.
        from test_release_archive import ReleaseArchiveTests
        fixture = ReleaseArchiveTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.record['support_identity']['usd_profile'] = kit.KIT_USD_PROFILE
        (fixture.package / 'config/live-sdk.json').write_text(KIT_PROFILE)
        self.root = fixture.root
        archive_path = package.archive.create(fixture.package, self.root / 'qualification')
        self.zip = package.create(archive_path, self.root / 'assets', run=pack_zip)
        self.package = self.root / 'kit/omni_usd_live'
        with zipfile.ZipFile(self.zip) as reader:
            reader.extractall(self.package)
            self.members = {name: reader.read(name) for name in reader.namelist()}
        self.sha256 = hashlib.sha256(self.zip.read_bytes()).hexdigest()
        self.adapter = 'python/omni/usd/live/_live.cpython-312-x86_64-linux-gnu.so'

    def verify(self):
        return kit.release_candidate(self.zip, self.package)

    def rewrite(self, members):
        """Replace the zip with `members` and a matching checksum file, so only the content checks can fail."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with zipfile.ZipFile(self.zip, 'w') as writer:
                for name, data in members.items():
                    writer.writestr(name, data)
        self.zip.with_name(self.zip.name + '.sha256').write_text(
            hashlib.sha256(self.zip.read_bytes()).hexdigest() + '  ' + self.zip.name + '\n')

    def inputs(self, **overrides):
        """Synthetic Kit executable and companion extensions beside the extracted package."""
        args = SimpleNamespace(**{name: None for name in kit.INPUTS})
        args.kit_executable = self.root / 'kit/kit'
        args.kit_executable.write_bytes(b'synthetic executable')
        args.kit_executable.chmod(0o700)
        for name in ('kit_client_extension', 'kit_resolver_extension'):
            path = self.root / 'kit' / name
            (path / 'config').mkdir(parents=True, exist_ok=True)
            (path / 'config/extension.toml').write_text('synthetic config')
            setattr(args, name, path)
        for name, relative in (('kit_client_extension', 'bin/libomni.client.lifecycle.plugin.so'),
                               ('kit_resolver_extension', 'lib/libcarb.usdresolver.plugin.so'),
                               ('kit_resolver_extension', 'lib/libomni_usd_resolver.so'),
                               ('kit_resolver_extension', 'bin/libomni.ext-usd_resolver.plugin.so')):
            path = getattr(args, name) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'synthetic library')
        args.kit_package, args.kit_release_package = self.package, self.zip
        for name, value in overrides.items():
            setattr(args, name, value)
        return args

    @staticmethod
    def runtime(identity, **changes):
        """The startup identity every Kit participant reports for `identity`, with `changes` in each probe."""
        probe = dict(python_version='synthetic-python', python_optimize=0, dont_write_bytecode=True,
                     settings=kit.SETTINGS, extension_configs=identity['extension_configs'],
                     executable_sha256=identity['executable']['sha256'],
                     driver_sha256=kit.digest(kit.ROOT / 'tests/native/python/kit_process.py'),
                     native_libraries=identity['required_libraries'],
                     python_abi=dict(soabi='cpython-312-x86_64-linux-gnu', ext_suffix=identity['adapter']['ext_suffix'],
                                     cache_tag='cpython-312'),
                     live_module=dict(path=identity['adapter']['path'], sha256=identity['adapter']['sha256'],
                                      abi_version=0x00010000))
        probe.update(changes)
        names = [test.rsplit('.', 1)[-1] + '-kit-' + str(index) for test, count in kit.TESTS.items()
                 for index in range(count)]
        return {name: dict(probe, stage_root='live' if name == 'test_kit_direct_live_root_issues_host_edits-kit-0'
                           else 'composed') for name in names}

    def test_release_package_verifies_from_its_manifest(self):
        candidate = self.verify()
        self.assertEqual(candidate['archive']['sha256'], self.sha256)
        self.assertEqual(candidate['abi_version'], 0x00010000)
        self.assertEqual(candidate['python_extension'], self.adapter)
        self.assertEqual(candidate['manifest'], package.check(self.zip))
        # Package metadata beside the extracted files is never read and never part of the identity.
        (self.package / 'PACKAGE-INFO.yaml').write_text('unread')
        self.assertEqual(self.verify(), candidate)

    def test_snapshot_records_the_lane_and_archive_and_checks_the_loaded_module(self):
        if not sys.platform.startswith('linux'):
            with self.assertRaisesRegex(ValueError, 'requires Linux'):
                kit.snapshot(self.inputs())
            return
        with patch('tools.live.python_identity.validate', side_effect=AssertionError('build-tree validation')):
            identity = kit.snapshot(self.inputs())
        self.assertEqual(identity['lane'], kit.RELEASE_PACKAGE)
        self.assertEqual(kit.release_sha256(identity), self.sha256)
        self.assertEqual(identity['adapter']['path'], str((self.package / self.adapter).resolve()))
        self.assertEqual(identity['extension_configs']['omni_usd_live'],
                         identity['candidate']['manifest']['files']['omni_usd_live/config/extension.toml']['sha256'])
        self.assertIn('--kit-release-package', kit.arguments(identity))
        environment = kit.reference_environment({'LD_LIBRARY_PATH': '/sdk/lib', 'PATH': '/bin'}, identity)
        self.assertNotIn('LD_LIBRARY_PATH', environment)
        self.assertEqual(environment['PATH'], '/bin')
        kit.validate_runtime(self.runtime(identity), identity)
        loaded = dict(path=identity['adapter']['path'], sha256=identity['adapter']['sha256'], abi_version=0x00010000)
        for module, message in ((dict(loaded, abi_version=0x00020000), 'ABI version than the release manifest'),
                                (dict(loaded, path='/elsewhere/_live.so'), 'another Python adapter'),
                                (dict(loaded, sha256='0' * 64), 'another Python adapter')):
            with self.subTest(module=module), self.assertRaisesRegex(ValueError, message):
                kit.validate_runtime(self.runtime(identity, live_module=module), identity)
        with self.assertRaisesRegex(ValueError, 'folder named omni_usd_live'):
            renamed = self.package.rename(self.package.with_name('pkg-kit-x86_64'))
            kit.snapshot(self.inputs(kit_package=renamed))

    def test_tampered_binary_or_configuration_fails(self):
        for name in ('lib/libomni_usd_live.so', self.adapter, 'config/extension.toml', 'config/live-sdk.json'):
            path = self.package / name
            original = path.read_bytes()
            with self.subTest(member=name):
                path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
                with self.assertRaisesRegex(ValueError, 'differs from its manifest: ' + name):
                    self.verify()
                path.write_bytes(original)
        self.verify()
        # A package zip whose member differs from its manifest fails even with a matching checksum file.
        changed = dict(self.members, **{'lib/libomni_usd_live.so': self.members['lib/libomni_usd_live.so'] + b'x'})
        self.rewrite(changed)
        with self.assertRaisesRegex(ValueError, 'differs from its manifest'):
            self.verify()
        self.rewrite(self.members)
        self.zip.with_name(self.zip.name + '.sha256').write_text('0' * 64 + '  ' + self.zip.name + '\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.verify()

    def test_missing_or_malformed_manifest_fails(self):
        manifest = self.package / kit.RELEASE_MANIFEST
        original = manifest.read_text()
        manifest.unlink()
        with self.assertRaisesRegex(ValueError, r'missing=\[.release-manifest.json.\]'):
            self.verify()
        for text, message in (('{"schema_version": 1', 'Expecting'),
                              (json.dumps(dict(json.loads(original), git_head='0' * 40)), "differs from the release")):
            with self.subTest(message=message):
                manifest.write_text(text)
                with self.assertRaisesRegex(ValueError, message):
                    self.verify()
        manifest.write_text(original)
        self.verify()
        cases = (({n: d for n, d in self.members.items() if n != package.MANIFEST}, 'no release-manifest.json'),
                 (dict(self.members, **{package.MANIFEST: b'{"schema_version": 1'}), 'Expecting'),
                 (dict(self.members, **{package.MANIFEST: json.dumps(dict(json.loads(original), git_head='1' * 40)).encode()}),
                  'differs from the release'))
        for members, message in cases:
            with self.subTest(message=message):
                self.rewrite(members)
                with self.assertRaisesRegex(ValueError, message):
                    self.verify()
        self.zip.write_bytes(b'not a zip')
        self.zip.with_name(self.zip.name + '.sha256').write_text(
            hashlib.sha256(b'not a zip').hexdigest() + '  ' + self.zip.name + '\n')
        with self.assertRaisesRegex(ValueError, 'not a readable zip'):
            self.verify()
        self.zip.unlink()
        with self.assertRaisesRegex(ValueError, 'zip is missing'):
            self.verify()

    def test_interpreter_or_platform_mismatch_fails(self):
        with patch.object(platforms, 'host', return_value=platforms.PLATFORMS['linux-aarch64']):
            with self.assertRaisesRegex(ValueError, 'built for linux-x86_64, not this linux-aarch64 host'):
                self.verify()
        if not sys.platform.startswith('linux'):
            return
        identity = kit.snapshot(self.inputs())
        for abi in (dict(soabi='cpython-311-x86_64-linux-gnu', ext_suffix='.cpython-311-x86_64-linux-gnu.so',
                         cache_tag='cpython-311'),
                    dict(soabi='cpython-312-aarch64-linux-gnu', ext_suffix='.cpython-312-aarch64-linux-gnu.so',
                         cache_tag='cpython-312')):
            with self.subTest(abi=abi), self.assertRaisesRegex(ValueError, 'interpreter of another ABI'):
                kit.validate_runtime(self.runtime(identity, python_abi=abi), identity)

    def test_unexpected_files_and_build_tree_sidecars_fail_the_release_lane(self):
        for name in (self.adapter + '.manifest.json', 'python/omni/usd/live/__pycache__/__init__.cpython-312.pyc'):
            path = self.package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
            with self.subTest(extra=name), self.assertRaisesRegex(ValueError, 'extra=.*' + Path(name).name):
                self.verify()
            path.unlink()

    def test_build_tree_input_still_requires_its_sidecars(self):
        if not sys.platform.startswith('linux'):
            return
        # The extracted release package is selected only explicitly, never by a missing sidecar.
        with self.assertRaisesRegex(ValueError, '--kit-release-package'):
            kit.snapshot(self.inputs(kit_release_package=None))
        (self.package / kit.RELEASE_MANIFEST).unlink()
        with self.assertRaisesRegex(FileNotFoundError, r'_live\.cpython-312-x86_64-linux-gnu\.so\.manifest\.json'):
            kit.snapshot(self.inputs(kit_release_package=None))


if __name__ == '__main__':
    unittest.main()
