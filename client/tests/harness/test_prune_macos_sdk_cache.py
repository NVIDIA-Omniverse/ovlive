# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDK cache pruning keeps the validated selection and preserves unrecognized data."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.release import prune_macos_sdk_cache as cache


class MacSdkCachePruningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / 'sdk'
        self.active = self.completed('sdk-active')
        (self.root / 'install').symlink_to('builds/sdk-active/install', target_is_directory=True)

    def completed(self, name):
        build = self.root / 'builds' / name
        library = build / 'install/lib/libusd.dylib'
        library.parent.mkdir(parents=True)
        library.write_bytes(b'generated SDK library')
        record = dict(complete=True, identity=dict(architecture='arm64', configuration='Release',
                                                  recipe_sha256='a' * 64),
                      installed_artifacts={'lib/libusd.dylib': hashlib.sha256(library.read_bytes()).hexdigest()})
        (build / 'build-provenance.json').write_text(json.dumps(record))
        return build

    def test_prunes_multiple_completed_installs_and_preserves_selected_sdk_sources_and_builds(self):
        stale = [self.completed('sdk-old-one'), self.completed('sdk-old-two')]
        for build in stale:
            (build / 'OpenUSD-build').mkdir()
            (build / 'OpenUSD-build/local.txt').write_text('compiler evidence')
        for source in ('OpenUSD', 'oneTBB'):
            (self.root / source).mkdir()
            (self.root / source / 'local.txt').write_text('source checkout')
        active_record = (self.active / 'build-provenance.json').read_bytes()
        self.assertEqual(cache.prune(self.root), ['sdk-old-one', 'sdk-old-two'])
        self.assertEqual((self.root / 'install').readlink(), Path('builds/sdk-active/install'))
        self.assertEqual((self.active / 'build-provenance.json').read_bytes(), active_record)
        self.assertTrue((self.active / 'install/lib/libusd.dylib').is_file())
        for build in stale:
            self.assertFalse((build / 'install').exists())
            self.assertFalse((build / 'build-provenance.json').exists())
            self.assertEqual((build / 'OpenUSD-build/local.txt').read_text(), 'compiler evidence')
        for source in ('OpenUSD', 'oneTBB'):
            self.assertEqual((self.root / source / 'local.txt').read_text(), 'source checkout')
        self.assertEqual(cache.prune(self.root), [])

    def test_preserves_unknown_entries_and_symlinked_build_install_and_provenance(self):
        unknown = self.completed('user-build')
        (self.root / 'builds/sdk-user-file').write_text('user file')
        (self.root / 'builds/sdk-link').symlink_to(unknown, target_is_directory=True)
        install_link = self.completed('sdk-install-link')
        (install_link / 'install').rename(install_link / 'original')
        (install_link / 'install').symlink_to('original', target_is_directory=True)
        marker_link = self.completed('sdk-marker-link')
        (marker_link / 'build-provenance.json').rename(marker_link / 'original.json')
        (marker_link / 'build-provenance.json').symlink_to('original.json')
        self.assertEqual(cache.prune(self.root), [])
        self.assertTrue((unknown / 'install').is_dir())
        self.assertEqual((self.root / 'builds/sdk-user-file').read_text(), 'user file')
        self.assertTrue((self.root / 'builds/sdk-link').is_symlink())
        self.assertTrue((install_link / 'install').is_symlink())
        self.assertTrue((marker_link / 'build-provenance.json').is_symlink())

    def test_preserves_incomplete_malformed_changed_and_untracked_installs(self):
        for name, kind in (('sdk-incomplete', 'incomplete'), ('sdk-malformed', 'malformed'),
                           ('sdk-changed', 'changed'), ('sdk-extra-file', 'file'),
                           ('sdk-extra-directory', 'directory'), ('sdk-private-metadata', 'metadata')):
            build = self.completed(name)
            marker = build / 'build-provenance.json'
            if kind == 'incomplete':
                record = json.loads(marker.read_text())
                record['complete'] = False
                marker.write_text(json.dumps(record))
            elif kind == 'malformed':
                marker.write_text('not JSON')
            elif kind == 'changed':
                (build / 'install/lib/libusd.dylib').write_bytes(b'local changes')
            elif kind == 'file':
                (build / 'install/lib/local.txt').write_text('local file')
            elif kind == 'directory':
                (build / 'install/lib/local').mkdir()
            else:
                (build / 'install/lib/PACKAGE-INFO.yaml').touch()
        self.assertEqual(cache.prune(self.root), [])
        self.assertEqual(len(list((self.root / 'builds').iterdir())), 7)

    def test_preserves_active_and_obsolete_ancillary_install_content(self):
        stale = self.completed('sdk-old')
        for build in (self.active, stale):
            documentation = build / 'install/share/doc/TBB/license.txt'
            documentation.parent.mkdir(parents=True)
            documentation.write_text('ancillary SDK documentation')
            (build / 'install/user.txt').write_text('user file')
            (build / 'install/user-link').symlink_to('user.txt')
            (build / 'install/PACKAGE-INFO.yaml').touch()
        self.assertEqual(cache.prune(self.root), ['sdk-old'])
        self.assertFalse((stale / 'install/lib').exists())
        self.assertFalse((stale / 'build-provenance.json').exists())
        for build in (self.active, stale):
            self.assertEqual((build / 'install/share/doc/TBB/license.txt').read_text(),
                             'ancillary SDK documentation')
            self.assertEqual((build / 'install/user.txt').read_text(), 'user file')
            self.assertTrue((build / 'install/user-link').is_symlink())
            self.assertTrue((build / 'install/PACKAGE-INFO.yaml').exists())

    def test_preserves_symlink_to_outside_install_without_reading_it(self):
        build = self.completed('sdk-outside-link')
        sentinel = self.root / 'outside'
        sentinel.write_bytes(b'outside user data')
        (build / 'install/lib/libusd.dylib').unlink()
        (build / 'install/lib/libusd.dylib').symlink_to(sentinel)
        self.assertEqual(cache.prune(self.root), [])
        self.assertEqual(sentinel.read_bytes(), b'outside user data')
        self.assertTrue((build / 'install/lib/libusd.dylib').is_symlink())

    def test_prunes_generated_internal_library_symlinks(self):
        build = self.completed('sdk-internal-link')
        (build / 'install/lib/libalias.dylib').symlink_to('libusd.dylib')
        marker = build / 'build-provenance.json'
        record = json.loads(marker.read_text())
        record['installed_artifacts']['lib/libalias.dylib'] = record['installed_artifacts']['lib/libusd.dylib']
        marker.write_text(json.dumps(record))
        self.assertEqual(cache.prune(self.root), ['sdk-internal-link'])
        self.assertFalse(build.exists())

    def test_bad_selected_sdk_refuses_every_pruning_mutation(self):
        stale = self.completed('sdk-old')
        selected = self.root / 'install'
        for target in ('../elsewhere/install', 'builds/sdk-old/../sdk-active/install',
                       'builds/sdk-missing/install', str(self.active / 'install')):
            with self.subTest(target=target):
                selected.unlink()
                selected.symlink_to(target, target_is_directory=True)
                with self.assertRaises(ValueError):
                    cache.prune(self.root)
                self.assertTrue((stale / 'install/lib/libusd.dylib').is_file())
        selected.unlink()
        selected.symlink_to('builds/sdk-active/install', target_is_directory=True)
        (self.active / 'install/lib/libusd.dylib').write_bytes(b'invalid selected SDK')
        with self.assertRaisesRegex(ValueError, 'differs'):
            cache.prune(self.root)
        self.assertTrue((stale / 'build-provenance.json').is_file())

    def test_cli_accepts_an_explicit_sdk_root(self):
        self.completed('sdk-old')
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cache.main(['--root', str(self.root)]), 0)
        self.assertIn('Pruned 1 obsolete completed USD SDK installs', output.getvalue())


if __name__ == '__main__':
    unittest.main()
