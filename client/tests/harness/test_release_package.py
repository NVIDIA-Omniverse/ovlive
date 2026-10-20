# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The packman release package re-packs a checked release archive under Kit's package naming and verifies alone."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

from tools.release import package

HEAD = '0123456789abcdef' * 2 + '01234567'
NAME = 'omni_usd_live_ov_openusd_0.25.11_py_3.12'
VERSION = '0.1.0.manylinux_2_35_x86_64.release'


def record(**overrides):
    value = dict(schema_version=1, product='ovlive', version='0.1.0', usd_profile='kit-25.11',
                 platform='linux-x86_64', configuration='release', git_head=HEAD, git_tag='v0.1.0', dirty=False,
                 archive_root='ovlive-0.1.0-kit-25.11-linux-x86_64', extension_folder='omni_usd_live', files={})
    value.update(overrides)
    return value


class PackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / 'ovlive-0.1.0-kit-25.11-linux-x86_64.tar.gz'
        self.archive.write_bytes(b'archive')
        self.output = self.root / 'out'
        self.commands = []

    def check(self, value):
        """A stand-in for archive.check that extracts a one-file package tree."""
        def fake(archive_path, extract_to, forbidden):
            tree = Path(extract_to) / value['archive_root'] / 'omni_usd_live'
            (tree / 'lib').mkdir(parents=True)
            (tree / 'lib/libomni_usd_live.so').write_bytes(b'\x7fELF')
            return value
        return fake

    def run_packman(self, command, check):
        """A stand-in for `packman pack` that writes `<parent>@<folder>.7z` as a zip of the folder."""
        self.commands.append(command)
        tree, output = Path(command[-1]), Path(command[command.index('-o') + 1])
        with zipfile.ZipFile(output / f'{tree.parent.name}@{tree.name}.zip', 'w') as writer:
            for path in sorted(tree.rglob('*')):
                if path.is_file():
                    writer.write(path, path.relative_to(tree).as_posix())

    def create(self, value, **kwargs):
        return package.create(self.archive, self.output, check=self.check(value), run=self.run_packman, **kwargs)

    def test_tagged_identity(self):
        self.assertEqual(package.identity(record()), (NAME, VERSION))

    def test_identity_is_the_extension_version_with_or_without_a_tag(self):
        # A candidate qualifies main before its tag exists, and finalize promotes the same bytes.
        self.assertEqual(package.identity(record(git_tag=None)), (NAME, VERSION))

    def test_stock_identity(self):
        self.assertEqual(package.identity(record(usd_profile='stock-25.11'))[0],
                         'omni_usd_live_openusd_0.25.11_py_3.12')

    def test_aarch64_identity_names_its_dependency_abi(self):
        self.assertEqual(package.identity(record(platform='linux-aarch64')),
                         (NAME, '0.1.0.manylinux_2_35_aarch64.release'))

    def test_unknown_platform_rejected(self):
        with self.assertRaisesRegex(ValueError, 'platform'):
            package.identity(record(platform='solaris-sparc'))

    def test_package_layout_and_checksum(self):
        path = self.create(record())
        # GitHub renames release assets with special characters, so the asset separates name and version with
        # `--`; package.packman_filename restores the `<name>@<version>.zip` packman expects.
        self.assertEqual(path.name, f'{NAME}--{VERSION}.zip')
        self.assertEqual(package.packman_filename(path.name), f'{NAME}@{VERSION}.zip')
        self.assertRegex(path.name, r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
        self.assertEqual(self.commands[0][1:4], ['pack', '-c', 'zip'])
        with zipfile.ZipFile(path) as reader:
            self.assertEqual(sorted(reader.namelist()), ['lib/libomni_usd_live.so', 'release-manifest.json'])
            inner = json.loads(reader.read('release-manifest.json'))
        self.assertEqual(inner['package'], dict(name=NAME, version=VERSION))
        self.assertEqual(inner['git_head'], HEAD)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual((self.output / (path.name + '.sha256')).read_text(), f'{digest}  {path.name}\n')
        # Both profiles' packages share one candidate directory, so nothing else is written beside them.
        self.assertEqual(sorted(p.name for p in self.output.iterdir()), [path.name, path.name + '.sha256'])

    def test_dirty_archive_rejected(self):
        with self.assertRaisesRegex(ValueError, 'modified checkout'):
            self.create(record(dirty=True))
        self.assertEqual(self.commands, [])

    def test_existing_output_rejected(self):
        self.output.mkdir()
        (self.output / f'{NAME}--{VERSION}.zip.sha256').write_text('')
        with self.assertRaisesRegex(ValueError, 'exists'):
            self.create(record())
        self.assertEqual(self.commands, [])


def pack_zip(command, check):
    """A stand-in for `packman pack -c zip` that writes `<parent>@<folder>.zip` of the folder's files."""
    tree, output = Path(command[-1]), Path(command[command.index('-o') + 1])
    with zipfile.ZipFile(output / f'{tree.parent.name}@{tree.name}.zip', 'w') as writer:
        for path in sorted(tree.rglob('*')):
            if path.is_file():
                writer.write(path, path.relative_to(tree).as_posix())


class PackageCheckTests(unittest.TestCase):
    """A package verifies on its own against its checksum file and the release manifest it carries."""

    def setUp(self):
        # The archive fixture supplies a real release archive of synthetic package bytes.
        from test_release_archive import ReleaseArchiveTests
        fixture = ReleaseArchiveTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        archive_path = package.archive.create(fixture.package, fixture.root / 'qualification')
        self.path = package.create(archive_path, fixture.root / 'assets', run=pack_zip)
        with zipfile.ZipFile(self.path) as reader:
            self.members = {name: reader.read(name) for name in reader.namelist()}
        self.manifest = json.loads(self.members[package.MANIFEST])

    def rewrite(self, members, path=None):
        """Replace the package with `members` and a matching checksum file, so only the content checks can fail."""
        path = path or self.path
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with zipfile.ZipFile(path, 'w') as writer:
                for name, data in members:
                    writer.writestr(name, data)
        path.with_name(path.name + '.sha256').write_text(
            hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.name + '\n')
        return path

    def with_manifest(self, text):
        return [*((n, d) for n, d in self.members.items() if n != package.MANIFEST), (package.MANIFEST, text.encode())]

    def test_package_checks_against_its_manifest(self):
        record = package.check(self.path)
        self.assertEqual(record['package'], dict(name='omni_usd_live_openusd_0.25.11_py_3.12',
                                                 version='0.1.0.manylinux_2_35_x86_64.release'))
        self.assertEqual(record['archive_root'], 'ovlive-0.1.0-stock-25.11-linux-x86_64')
        self.assertEqual(set(record['files']), {'omni_usd_live/' + name for name in self.members
                                                if name != package.MANIFEST})

    def test_checksum_file_is_required_and_exact(self):
        sidecar = self.path.with_name(self.path.name + '.sha256')
        sidecar.write_text('0' * 64 + '  ' + self.path.name + '\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            package.check(self.path)
        sidecar.unlink()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            package.check(self.path)

    def test_member_content_and_set_must_match_the_manifest(self):
        changed = dict(self.members, **{'resources/plugInfo.json': self.members['resources/plugInfo.json'] + b' '})
        cases = [('differs from its manifest', list(changed.items())),
                 ('missing package members', [*self.members.items(), ('extra.txt', b'x')]),
                 ('missing package members', [(n, d) for n, d in self.members.items() if n != 'resources/plugInfo.json']),
                 ('Unsafe', [*self.members.items(), ('../outside.txt', b'x')]),
                 ('Unsafe', [*self.members.items(), ('/outside.txt', b'x')]),
                 ('Duplicate', [*self.members.items(), ('resources/plugInfo.json', b'x')]),
                 ('no release-manifest.json', [(n, d) for n, d in self.members.items() if n != package.MANIFEST])]
        for message, members in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                package.check(self.rewrite(members))

    def test_manifest_must_be_path_free_and_name_this_package(self):
        bom = dict(self.manifest['components'][0], name='other')
        cases = [('absolute path', dict(git_tag='/build/ovlive')),
                 ('package name', dict(package=dict(name='other', version=self.manifest['package']['version']))),
                 ('Unrecognized release manifest', dict(configuration='debug')),
                 ('Unrecognized release manifest', dict(archive_root='ovlive-0.1.0-kit-25.11-linux-x86_64')),
                 ('bill of materials', dict(components=[bom, *self.manifest['components'][1:]]))]
        for message, changes in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                package.check(self.rewrite(self.with_manifest(json.dumps(dict(self.manifest, **changes)))))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            package.check(self.rewrite(self.with_manifest(json.dumps(self.manifest)[:-1] + ', "dirty": true}')))

    def test_package_under_another_name_is_rejected(self):
        renamed = self.path.with_name(self.path.name.replace('_openusd_', '_ov_openusd_'))
        with self.assertRaisesRegex(ValueError, 'package name'):
            package.check(self.rewrite(self.members.items(), renamed))


if __name__ == '__main__':
    unittest.main()
