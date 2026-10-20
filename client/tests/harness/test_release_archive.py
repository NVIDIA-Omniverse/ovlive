# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The release archive is reproducible, path-free and verified member by member."""
import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools.live import build
from tools.release import archive

ROOT = Path(__file__).resolve().parents[2]
HEAD = 'a' * 40
PLATFORM = 'linux-x86_64'
NAME = 'ovlive-0.1.0-stock-25.11-' + PLATFORM
COMMIT_TIME = archive.commit_time


def strings(value):
    """Every key and string value of a JSON document."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


class ReleaseArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.package = self.write_package(self.root / 'stage/usd/omniverse/live', PLATFORM)
        self.record = dict(
            platform=PLATFORM, configuration='release', inputs={'CMakeLists.txt': '1' * 64, 'source/live/a.cpp': '2' * 64},
            support_identity=dict(
                usd_profile='stock-25.11',
                tools={'cmake': dict(path='/sdk/tools/cmake', sha256='3' * 64),
                       'python': dict(path='/sdk/python/bin/python3.12', sha256='4' * 64)},
                trees={'usd_root/include': dict(path='/sdk/usd/include', sha256='5' * 64),
                       'dependency_root/grpc-release/lib': dict(path='/sdk/deps/grpc-release/lib', sha256='6' * 64)}))
        self.validate = self.start(patch.object(archive.build, 'validate_stage', return_value=self.record))
        self.start(patch.object(archive.platforms, 'host', return_value=archive.platforms.PLATFORMS[PLATFORM]))
        self.head = self.start(patch.object(archive, 'git_head', return_value=HEAD))
        self.dirty = self.start(patch.object(archive, 'git_dirty', return_value=False))
        self.tags = self.start(patch.object(archive, 'git_tags', return_value=[]))
        self.start(patch.object(archive, 'commit_time', return_value=1700000000))
        self.start(patch.object(archive, 'fingerprint', return_value='f' * 64))

    @staticmethod
    def write_package(package, platform):
        """A synthetic release stage payload of `platform` at `package`, with ELF headers of its machine."""
        elf = b'\x7fELF\x02\x01' + b'\0' * 12 + archive.platforms.get(platform).machine_code.to_bytes(2, 'little')
        for name in build.package_files(platform):
            path = package / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name in build.native_files(platform):
                content = elf + (HEAD.encode() if name == 'lib/libomni_usd_live.so' else b'')
            elif name == 'resources/plugInfo.json':
                content = json.dumps({'Plugins': [{'Root': '..', 'LibraryPath': 'lib/libomni_usd_live.so',
                                                   'Name': 'omni_usd_live'}]}).encode()
            elif name == 'share/omni_usd_live/BOM.json':
                content = (ROOT / 'BOM.json').read_bytes()
            elif name == build.PYTHON_PACKAGE + 'py.typed':
                content = b''
            elif name == 'include/omni/usd/live/Live.h':
                content = b'#define OMNI_USD_LIVE_ABI_VERSION UINT32_C(0x00010000)\n'
            else:
                content = f'synthetic {name}\n'.encode()
            path.write_bytes(content)
            path.chmod(0o755 if name in build.native_files(platform) else 0o644)
        return package

    def start(self, patcher):
        mock = patcher.start()
        self.addCleanup(patcher.stop)
        return mock

    def create(self, directory='out', **options):
        return archive.create(self.package, self.root / directory, **options)

    def manifest_of(self, path):
        with tarfile.open(path, 'r:gz') as source:
            return json.loads(source.extractfile(NAME + '/release-manifest.json').read())

    def rewrite(self, source, directory, edit):
        """A copy of `source` under its own name in `directory`, members passed through `edit`, with its checksum."""
        with tarfile.open(source, 'r:gz') as reader:
            members = [(info, reader.extractfile(info).read() if info.isfile() else None)
                       for info in reader.getmembers()]
        members = edit(members)
        target = self.root / directory / (NAME + '.tar.gz')
        target.parent.mkdir()
        with tarfile.open(target, 'w:gz', format=tarfile.PAX_FORMAT) as writer:
            for info, data in members:
                writer.addfile(info, io.BytesIO(data) if data is not None else None)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        target.with_name(target.name + '.sha256').write_text(f'{digest}  {target.name}\n')
        return target

    def test_create_then_check_passes(self):
        path = self.create()
        self.assertEqual(path, self.root / 'out' / (NAME + '.tar.gz'))
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), [NAME + '.tar.gz', NAME + '.tar.gz.sha256'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual((path.parent / (NAME + '.tar.gz.sha256')).read_text(), f'{digest}  {NAME}.tar.gz\n')
        self.validate.assert_called_once_with(self.package, archive.ROOT, expected_platform='linux-x86_64',
                                              expected_configuration='release')
        manifest = archive.check(path)
        self.assertEqual(manifest, self.manifest_of(path))
        inputs = json.dumps(self.record['inputs'], sort_keys=True, separators=(',', ':')).encode()
        bom = json.loads((ROOT / 'BOM.json').read_text())
        self.assertEqual({key: manifest[key] for key in (
            'schema_version', 'product', 'version', 'archive_root', 'extension_folder', 'platform', 'configuration',
            'usd_profile', 'sanitizer', 'abi_version', 'git_head', 'git_tag', 'dirty', 'source_fingerprint',
            'source_inputs_sha256', 'embedded_revision', 'stripped', 'sdk_identity', 'tool_identity', 'components')},
            dict(schema_version=1, product='ovlive', version='0.1.0', archive_root=NAME,
                 extension_folder='omni_usd_live',
                 platform='linux-x86_64', configuration='release', usd_profile='stock-25.11', sanitizer='none',
                 abi_version='0x00010000', git_head=HEAD, git_tag=None, dirty=False, source_fingerprint='f' * 64,
                 source_inputs_sha256=hashlib.sha256(inputs).hexdigest(), embedded_revision=HEAD, stripped=False,
                 sdk_identity={'usd_root/include': '5' * 64, 'dependency_root/grpc-release/lib': '6' * 64},
                 tool_identity={'cmake': '3' * 64, 'python': '4' * 64}, components=bom['components']))
        self.assertEqual(set(manifest['files']), {'omni_usd_live/' + name for name in build.package_files(PLATFORM)})
        library = (self.package / 'lib/libomni_usd_live.so').read_bytes()
        self.assertEqual(manifest['files']['omni_usd_live/lib/libomni_usd_live.so'],
                         dict(sha256=hashlib.sha256(library).hexdigest(), size=len(library), mode='0755'))
        self.assertEqual(manifest['files']['omni_usd_live/config/extension.toml']['mode'], '0644')
        self.assertEqual(path.read_bytes()[4:8], b'\0\0\0\0')
        with tarfile.open(path, 'r:gz') as source:
            members = source.getmembers()
        self.assertEqual([m.name for m in members][:2], [NAME, NAME + '/omni_usd_live'])
        self.assertEqual({(m.uid, m.gid, m.uname, m.gname, m.mtime) for m in members}, {(0, 0, '', '', 1700000000)})
        self.assertEqual([m.name for m in members], sorted((m.name for m in members), key=lambda n: n.split('/')))

    def test_two_creates_of_one_stage_are_byte_identical(self):
        first = self.create('one')
        # Stage file times and modes are not archive inputs.
        library = self.package / 'lib/libomni_usd_live.so'
        library.chmod(0o700)
        os.utime(library, (1, 1))
        second = self.create('two')
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_against_compares_two_archives_member_by_member(self):
        first = self.create('one')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(archive.main(['check', '--archive', str(first), '--against', str(self.create('two'))]), 0)
        self.assertEqual(output.getvalue(), f'{NAME}.tar.gz: 0 differing members, archive files byte-identical\n')
        # A build-dependent binary differs in its member, its manifest entry and nothing else.
        sample = self.package / 'bin/live_diagnostics_sample'
        sample.write_bytes(sample.read_bytes() + b'build id')
        self.dirty.return_value = True
        second = self.create('three', allow_dirty=True)
        self.assertEqual(archive.compare(first, second),
                         ([NAME + '/omni_usd_live/bin/live_diagnostics_sample', NAME + '/release-manifest.json'],
                          ['dirty'], False))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(archive.main(['check', '--archive', str(first), '--against', str(second)]), 1)
        self.assertEqual(output.getvalue().splitlines(), [
            f'differing member: {NAME}/omni_usd_live/bin/live_diagnostics_sample',
            f'differing member: {NAME}/release-manifest.json',
            'differing release manifest field: dirty',
            f'{NAME}.tar.gz: 2 differing members, archive files differ'])
        # Identical members in differently encoded archive files still fail.
        recompressed = self.rewrite(first, 'recompressed', lambda members: members)
        self.assertEqual(archive.compare(first, recompressed), ([], [], False))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(archive.main(['check', '--archive', str(first), '--against', str(recompressed)]), 1)
        # Both archives are checked, and archives of different roots are not compared.
        (recompressed.parent / (NAME + '.tar.gz.sha256')).unlink()
        with self.assertRaisesRegex(ValueError, 'checksum file is missing'):
            archive.compare(first, recompressed)
        other = self.root / 'other' / 'ovlive-0.1.0-kit-25.11-linux-x86_64.tar.gz'
        with self.assertRaisesRegex(ValueError, 'different roots'):
            archive.compare(first, other)

    def test_manifest_and_archive_hold_no_absolute_path(self):
        path = self.create()
        self.assertEqual([s for s in strings(self.manifest_of(path)) if s.startswith('/')], [])
        with gzip.open(path) as source:
            self.assertNotIn(b'/sdk/', source.read())
        leaking = json.loads(json.dumps(self.record))
        leaking['support_identity']['trees']['/sdk/usd/lib'] = dict(path='/sdk/usd/lib', sha256='7' * 64)
        self.validate.return_value = leaking
        with self.assertRaisesRegex(ValueError, 'absolute'):
            self.create('leak')
        self.assertFalse((self.root / 'leak').exists())

    def test_a_flipped_byte_fails_check_naming_the_member(self):
        member = NAME + '/omni_usd_live/python/omni/usd/live/sample.py'

        def flip(members):
            return [(info, bytes([data[0] ^ 1]) + data[1:] if info.name == member else data) for info, data in members]
        with self.assertRaisesRegex(ValueError, 'omni_usd_live/python/omni/usd/live/sample.py'):
            archive.check(self.rewrite(self.create(), 'flipped', flip))

    def test_a_build_machine_path_in_a_member_fails(self):
        checkout = '/' + 'home/x/checkout'
        library = self.package / 'lib/libomni_usd_live.so'
        library.write_bytes(library.read_bytes() + b'\0' + checkout.encode() + b'/source/live/scene/a.cpp\0'
                            + b'/builds/omniverse/agent/grpc/src/core/a.cc\0/usr/src/linux/a.h\0'
                            + b'"/builds/omniverse/agent/b.cc" "/sdk/usd/include/pxr/pxr.h"\0')
        path = self.create()
        with self.assertRaisesRegex(ValueError,
                                    'omni_usd_live/lib/libomni_usd_live.so names a path of the build machine'):
            archive.check(path, forbidden=[checkout])
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            self.assertEqual(archive.main(['check', '--archive', str(path), '--forbid', checkout]), 1)
        self.assertIn(checkout, error.getvalue())
        # The prebuilt dependencies' own build paths are theirs; one ends at whitespace or a quote.
        with self.assertRaisesRegex(ValueError, 'sdk/usd'):
            archive.check(path, forbidden=['/sdk/usd'])
        with self.assertRaisesRegex(ValueError, 'prebuilt dependency build paths'):
            archive.check(path, forbidden=['/builds/omniverse/agent'])
        # A forbidden path matches only as a whole path or a path prefix.
        archive.check(path, forbidden=['/src', checkout[:-1], '/sdk/us', checkout + '/source/live/scene/a'])
        # create forbids the stage's checkout, SDK roots and tool roots.
        leaking = json.loads(json.dumps(self.record))
        leaking['support_inputs'] = dict(root=checkout, sdks={'usd_root': '/sdk/usd'},
                                         tools={'cmake': '/usr/bin/cmake', 'protoc': '/sdk/deps/protobuf/bin/protoc'})
        self.assertTrue({checkout, '/sdk/usd', '/sdk/deps/protobuf'} <= set(archive.machine_paths(leaking)))
        self.assertNotIn('/usr', archive.machine_paths(leaking))
        self.validate.return_value = leaking
        # A failed self-check publishes nothing and removes its extraction.
        with self.assertRaisesRegex(ValueError, 'names a path of the build machine'):
            archive.create(self.package, self.root / 'leaking', extract_to=self.root / 'leaking-extract')
        self.assertEqual(list((self.root / 'leaking').iterdir()), [])
        self.assertFalse((self.root / 'leaking-extract').exists())

    def test_build_machine_paths_match_at_path_character_boundaries(self):
        checkout = ('/' + 'home/x/checkout').encode()
        library = self.package / 'lib/libomni_usd_live.so'
        head = library.read_bytes()
        found = {
            'parenthesis': b'(' + checkout + b'/a.cpp', 'angle': b'<' + checkout + b'/a>',
            'dollar': b'$' + checkout + b'/a.cpp',
            'length-byte': b'\x12' + checkout + b'/a.cpp', 'comma': b'at ' + checkout + b',',
            'closing': b'"' + checkout + b')"', 'end': b'\0' + checkout,
            'prefix-map': b'-fdebug-prefix-map=/builds/omniverse/a=' + checkout + b'/b',
            'list': b'/builds/omniverse/a:' + checkout + b'/lib',
        }
        for index, (name, probe) in enumerate(found.items()):
            library.write_bytes(head + b'\0' + probe)
            with self.subTest(probe=name), self.assertRaisesRegex(ValueError, 'names a path of the build machine: '
                                                                  + checkout.decode()):
                archive.check(self.create(f'found-{index}'), forbidden=[checkout.decode()])
        library.write_bytes(head + b'\0/usr/src/a.h\0' + checkout + b'2/a.cpp\0x' + checkout + b'/a\0'
                            + checkout[:-1] + b'\0/sdk/usd/include\0/builds/omniverse/a' + checkout + b'\0')
        archive.check(self.create('clean'), forbidden=['/src', checkout.decode(), checkout.decode()[:-2] + 'x',
                                                       '/sdk/us', '/sdk/usd/lib'])

    def test_a_failed_extraction_check_removes_only_what_create_made(self):
        with patch.object(archive.build, '_validate_layout', side_effect=ValueError('Unexpected or missing members')), \
                self.assertRaisesRegex(ValueError, 'Unexpected or missing members'):
            self.create('failed', extract_to=self.root / 'failed-extract')
        self.assertEqual(list((self.root / 'failed').iterdir()), [])
        self.assertFalse((self.root / 'failed-extract').exists())
        foreign = self.root / 'foreign'
        (foreign / 'kept').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'Extraction directory exists'):
            self.create('refused', extract_to=foreign)
        self.assertTrue((foreign / 'kept').is_dir())
        self.assertFalse(any((self.root / 'refused').glob('*')))

    def test_source_date_epoch_sets_the_member_time(self):
        with patch.dict(os.environ, {'SOURCE_DATE_EPOCH': '1234567890'}):
            self.assertEqual(COMMIT_TIME(), 1234567890)
        for value in ('12.5', '-1', 'soon'):
            with self.subTest(value=value), patch.dict(os.environ, {'SOURCE_DATE_EPOCH': value}), \
                    self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                COMMIT_TIME()

    def test_a_stale_checksum_fails_check(self):
        path = self.create()
        (path.parent / (NAME + '.tar.gz.sha256')).write_text('0' * 64 + f'  {NAME}.tar.gz\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            archive.check(path)

    def test_a_missing_checksum_file_fails_check(self):
        path = self.create()
        (path.parent / (NAME + '.tar.gz.sha256')).unlink()
        with self.assertRaisesRegex(ValueError, 'checksum file is missing'):
            archive.check(path)

    def test_every_member_and_manifest_rejection_fails_check(self):
        path = self.create()
        folder = NAME + '/omni_usd_live/'
        manifest_name = NAME + '/release-manifest.json'

        def member(name, kind=tarfile.REGTYPE, mode=0o644, data=b'x'):
            info = tarfile.TarInfo(name)
            info.type, info.mode, info.mtime = kind, mode, 1700000000
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            return info, data if kind == tarfile.REGTYPE else None

        def add(*extra):
            return lambda members: members + list(extra)

        def replace_manifest(data):
            def edit(members):
                result = []
                for info, content in members:
                    if info.name == manifest_name:
                        info.size, content = len(data), data
                    result.append((info, content))
                return result
            return edit

        def change_manifest(change):
            with tarfile.open(path, 'r:gz') as reader:
                record = json.loads(reader.extractfile(manifest_name).read())
            change(record)
            return replace_manifest(json.dumps(record).encode())

        def sample_mode(mode):
            def edit(members):
                for info, _ in members:
                    if info.name.endswith('sample.py'):
                        info.mode = mode
                return members
            return edit

        def consistent_0755(members):
            # The member and its manifest entry agree on 0755, which the per-name mode still refuses.
            members = sample_mode(0o755)(members)
            return change_manifest(lambda record: record['files']['omni_usd_live/python/omni/usd/live/sample.py']
                                   .update(mode='0755'))(members)

        cases = {
            'backslash': (add(member(folder + 'a\\b')), 'backslash in name'),
            'dot': (add(member(NAME + '/./dot.txt')), 'unnormalized name'),
            'empty': (add(member(NAME + '//empty.txt')), 'unnormalized name'),
            'outside': (add(member('other/escape.txt')), 'outside ' + NAME),
            'duplicate': (add(member(folder + 'config/extension.toml')), 'extension.toml: duplicate'),
            'no-manifest': (lambda members: [(i, d) for i, d in members if i.name != manifest_name],
                            'has no release-manifest.json'),
            'oversized': (replace_manifest(b' ' * (archive.MAX_MANIFEST_BYTES + 1)), 'too large'),
            'unrecognized': (replace_manifest(b'[]'), 'Unrecognized release manifest'),
            'name': (change_manifest(lambda record: record.update(product='other')), 'does not match the archive name'),
            'file-list': (change_manifest(lambda record: record['files'].pop('omni_usd_live/config/extension.toml')),
                          'does not list exactly the package files'),
            'directory': (add(member(folder + 'empty', tarfile.DIRTYPE, 0o755)), 'Unexpected release archive directories'),
            'per-name-mode': (consistent_0755, 'sample.py: mode 0755'),
            'bom': (change_manifest(lambda record: record.update(components=[])), 'bill of materials'),
            'platform-unknown': (change_manifest(lambda record: record.update(platform='windows-x86_64')),
                                 'does not match the archive name'),
            'platform-type': (change_manifest(lambda record: record.update(platform=[])), 'does not match the archive name'),
            'platform-other': (change_manifest(lambda record: record.update(platform='linux-aarch64')),
                               'does not match the archive name'),
        }
        for name, (edit, reason) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, reason):
                archive.check(self.rewrite(path, name, edit))

    def test_unsafe_extra_or_missing_members_fail_check(self):
        path = self.create()
        folder = NAME + '/omni_usd_live/'

        def member(name, kind=tarfile.REGTYPE, mode=0o644, data=b'x', link=''):
            info = tarfile.TarInfo(name)
            info.type, info.mode, info.linkname, info.mtime = kind, mode, link, 1700000000
            info.size = len(data) if kind == tarfile.REGTYPE else 0
            return info, data if kind == tarfile.REGTYPE else None

        def add(*extra):
            return lambda members: members + list(extra)

        def mode(members):
            for info, _ in members:
                if info.name.endswith('sample.py'):
                    info.mode = 0o777
            return members

        cases = {
            'extra': (add(member(folder + 'extra.txt')), r"extra=\['omni_usd_live/extra.txt'\]"),
            'missing': (lambda members: [(i, d) for i, d in members if not i.name.endswith('sample.py')],
                        r"missing=\['omni_usd_live/python/omni/usd/live/sample.py'\]"),
            'symlink': (add(member(folder + 'link', tarfile.SYMTYPE, link='lib/libomni_usd_live.so')),
                        'link: not a regular file or directory'),
            'hardlink': (add(member(folder + 'hard', tarfile.LNKTYPE, link=folder + 'lib/libomni_usd_live.so')),
                         'hard: not a regular file or directory'),
            'parent': (add(member(NAME + '/../escape.txt')), 'parent-relative'),
            'absolute': (add(member('/' + NAME + '/escape.txt')), 'absolute name'),
            'device': (add(member(folder + 'device', tarfile.CHRTYPE)), 'device: not a regular file or directory'),
            'mode': (mode, 'sample.py: mode 0777'),
        }
        for name, (edit, reason) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, reason):
                archive.check(self.rewrite(path, name, edit))

    def test_create_refuses_debug_dirty_tagged_mismatched_or_existing_output(self):
        self.validate.side_effect = ValueError('Package platform/configuration does not match its destination')
        with self.assertRaisesRegex(ValueError, 'configuration'):
            self.create('debug')
        self.validate.side_effect = None
        self.dirty.return_value = True
        with self.assertRaisesRegex(ValueError, 'modified'):
            self.create('dirty')
        self.assertTrue(self.manifest_of(self.create('allowed', allow_dirty=True))['dirty'])
        self.dirty.return_value = False
        self.tags.return_value = ['v0.2.0']
        with self.assertRaisesRegex(ValueError, 'v0.2.0'):
            self.create('tag')
        self.tags.return_value = ['v0.1.0', 'nightly']
        self.assertEqual(self.manifest_of(self.create('tagged'))['git_tag'], 'v0.1.0')
        self.tags.return_value = []
        self.head.return_value = 'b' * 40
        with self.assertRaisesRegex(ValueError, 'revision'):
            self.create('revision')
        self.head.return_value = HEAD
        self.create('existing')
        with self.assertRaisesRegex(ValueError, 'exists'):
            self.create('existing')
        for directory in ('debug', 'dirty', 'tag', 'revision'):
            self.assertFalse(any((self.root / directory).glob('*')), directory)
        self.assertEqual(sorted(p.name for p in (self.root / 'existing').iterdir()),
                         [NAME + '.tar.gz', NAME + '.tar.gz.sha256'])

    def test_an_aarch64_stage_archives_under_its_platform_name(self):
        aarch64 = 'linux-aarch64'
        package = self.write_package(self.root / 'aarch64/usd/omniverse/live', aarch64)
        self.validate.return_value = dict(self.record, platform=aarch64)
        name = 'ovlive-0.1.0-stock-25.11-' + aarch64
        extension = 'omni_usd_live/python/omni/usd/live/_live.cpython-312-aarch64-linux-gnu.so'
        with patch.object(archive.platforms, 'host', return_value=archive.platforms.PLATFORMS[aarch64]):
            path = archive.create(package, self.root / 'aarch64-out', extract_to=self.root / 'aarch64-check')
        self.validate.assert_called_once_with(package, archive.ROOT, expected_platform=aarch64,
                                              expected_configuration='release')
        self.assertEqual(path.name, name + '.tar.gz')
        # Checking needs no host of the archive's platform: it reads headers only.
        manifest = archive.check(path)
        self.assertEqual((manifest['platform'], manifest['archive_root']), (aarch64, name))
        self.assertEqual(set(manifest['files']), {'omni_usd_live/' + member for member in build.package_files(aarch64)})
        self.assertEqual(manifest['files'][extension]['mode'], '0755')
        self.assertTrue((self.root / 'aarch64-check' / name / extension).is_file())
        # An x86-64 library in an aarch64 archive fails the extracted layout check.
        library = package / 'lib/libomni_usd_live.so'
        library.write_bytes(library.read_bytes()[:18] + b'\x3e\x00' + library.read_bytes()[20:])
        with patch.object(archive.platforms, 'host', return_value=archive.platforms.PLATFORMS[aarch64]), \
                self.assertRaisesRegex(ValueError, 'linux-aarch64 ELF'):
            archive.create(package, self.root / 'aarch64-again', extract_to=self.root / 'aarch64-again-check')

    def test_extraction_yields_exactly_the_extension_folder_and_manifest(self):
        path = self.create()
        target = self.root / 'extracted'
        archive.check(path, target)
        self.assertEqual([p.name for p in target.iterdir()], [NAME])
        self.assertEqual(sorted(p.name for p in (target / NAME).iterdir()), ['omni_usd_live', 'release-manifest.json'])
        folder = target / NAME / 'omni_usd_live'
        self.assertEqual(len([p for p in folder.rglob('*') if p.is_file()]), 29)
        build._validate_layout(folder, PLATFORM, exact=True)
        with self.assertRaisesRegex(ValueError, 'exists'):
            archive.check(path, target)
        created = archive.create(self.package, self.root / 'again', extract_to=self.root / 'created')
        self.assertTrue((self.root / 'created' / NAME / 'release-manifest.json').is_file())
        self.assertEqual(created.name, NAME + '.tar.gz')

    def test_version_is_the_package_table_version(self):
        self.assertEqual(archive.version(), '0.1.0')
        path = self.root / 'extension.toml'
        for text in ('[package]\nversion = "0.1"\n', '[core]\nversion = "0.1.0"\n[package]\ntitle = "x"\n',
                     '[package]\ntitle = "x"\n[dependencies]\nversion = "0.1.0"\n'):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                archive.version(path)
        path.write_text('[core]\nreloadable = false\n\n[package]\ntitle = "x"\nversion = "1.22.333"\n')
        self.assertEqual(archive.version(path), '1.22.333')

    def test_cli_prints_one_reason_and_exits_one(self):
        path = self.create()
        (path.parent / (NAME + '.tar.gz.sha256')).write_text('0' * 64 + f'  {NAME}.tar.gz\n')
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            self.assertEqual(archive.main(['check', '--archive', str(path)]), 1)
        self.assertEqual(len(error.getvalue().splitlines()), 1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(archive.main(['create', '--package', str(self.package),
                                           '--output-dir', str(self.root / 'cli')]), 0)
        self.assertIn(NAME + '.tar.gz', output.getvalue())


if __name__ == '__main__':
    unittest.main()
