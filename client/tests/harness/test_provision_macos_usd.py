# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned source SDK provisioning refuses changed sources and unsupported hosts."""
from pathlib import Path
import json
import os
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from tools.live import provision_macos_usd as provision


class MacUsdProvisionTests(unittest.TestCase):
    def test_sdk_validation_executes_native_loading_with_the_selected_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'install'
            python_root = Path(directory) / 'python'
            python = python_root / 'bin/python3.12'
            required = ('include/pxr/pxr.h', 'include/oneapi/tbb.h', 'lib/libtbb.dylib',
                        'lib/python/pxr/Usd/__init__.py', 'lib/libusd.dylib')
            for relative in required:
                path = prefix / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture artifact')
            with patch.object(provision.subprocess, 'check_output', return_value='arm64\n') as architecture, \
                    patch.object(provision.subprocess, 'run') as load:
                provision.validate_sdk(prefix, python, python_root)
            architecture.assert_called_once_with(['lipo', '-archs', str(prefix / 'lib/libusd.dylib')], text=True)
            command = load.call_args.args[0]
            self.assertEqual(command[:2], [str(python), '-c'])
            self.assertIn('Usd.GetVersion() == (0, 25, 11)', command[2])
            self.assertIn('Usd.Stage.CreateInMemory()', command[2])
            self.assertTrue(load.call_args.kwargs['check'])
            environment = load.call_args.kwargs['env']
            self.assertEqual(environment['PYTHONPATH'], str(prefix / 'lib/python'))
            self.assertEqual(environment['DYLD_LIBRARY_PATH'], os.pathsep.join(
                (str(prefix / 'lib'), str(python_root / 'lib'))))

    def test_cached_sdk_reuse_validates_identity_artifacts_and_native_loading_without_building(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / 'sdk'
            prefix = root / 'builds/cached/install'
            (prefix / 'lib').mkdir(parents=True)
            library = prefix / 'lib/libusd.dylib'
            library.write_bytes(b'cached library')
            (root / 'install').symlink_to(prefix.relative_to(root), target_is_directory=True)
            for name in provision.SOURCES:
                (root / name / '.git').mkdir(parents=True)
            python = root.parent / 'python'
            python.mkdir()
            (python / 'Python').write_bytes(b'framework')
            identity = dict(recipe='pinned', tools='selected')
            marker = prefix.parent / 'build-provenance.json'
            marker.write_text(json.dumps(dict(identity=identity, complete=True,
                                             installed_artifacts=provision.installed_artifacts(prefix))))
            argv = ['--root', str(root), '--python-root', str(python), '--cmake', '/cmake', '--reuse-only']
            with patch.object(provision.platform, 'system', return_value='Darwin'), \
                    patch.object(provision.platform, 'machine', return_value='arm64'), \
                    patch.object(provision.subprocess, 'check_output', return_value='3.12'), \
                    patch.object(provision, 'sdk_identity', return_value=identity), \
                    patch.object(provision, 'checkout') as checkout, \
                    patch.object(provision, 'run') as run, \
                    patch.object(provision, 'validate_sdk') as validate:
                provision.main(argv)
                run.assert_not_called()
                self.assertEqual(checkout.call_count, 2)
                self.assertTrue(all(call.kwargs == {'reuse_only': True} for call in checkout.call_args_list))
                validate.assert_called_once_with(prefix, python / 'bin/python3.12', python)
                library.write_bytes(b'changed library')
                with self.assertRaisesRegex(ValueError, 'artifacts differ'):
                    provision.main(argv)
                run.assert_not_called()
                self.assertEqual(validate.call_count, 1)
                library.write_bytes(b'cached library')
                validate.side_effect = ValueError('native import failed')
                with self.assertRaisesRegex(ValueError, 'native import failed'):
                    provision.main(argv)
                run.assert_not_called()
                for record in (dict(identity=identity, complete=False), dict(identity={'recipe': 'other'}, complete=True)):
                    marker.write_text(json.dumps(record))
                    with self.assertRaisesRegex(ValueError, 'No matching completed'):
                        provision.main(argv)
                run.assert_not_called()

    def test_failed_build_or_validation_keeps_selected_sdk(self):
        for failure in ('build', 'validation'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / 'sdk'
                selected = root / 'install'
                selected.mkdir(parents=True)
                (selected / 'sentinel').write_text('usable SDK')
                python = Path(directory) / 'python'
                python.mkdir()
                (python / 'Python').write_text('framework')
                with patch.object(provision.platform, 'system', return_value='Darwin'), \
                        patch.object(provision.platform, 'machine', return_value='arm64'), \
                        patch.object(provision.subprocess, 'check_output', return_value='3.12'), \
                        patch.object(provision, 'checkout'), \
                        patch.object(provision, 'run', side_effect=(subprocess.CalledProcessError(1, 'build')
                                     if failure == 'build' else None)), \
                        patch.object(provision, 'validate_sdk', side_effect=ValueError('invalid SDK')):
                    with self.assertRaises((subprocess.CalledProcessError, ValueError)):
                        provision.main(['--root', str(root), '--python-root', str(python), '--cmake', '/cmake'])
                self.assertFalse(selected.is_symlink())
                self.assertEqual((selected / 'sentinel').read_text(), 'usable SDK')

    def test_publication_retains_previous_sdk_and_selects_new_install(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            old = root / 'install'
            old.mkdir()
            (old / 'sentinel').write_text('old SDK')
            fresh = root / 'builds' / 'new' / 'install'
            fresh.mkdir(parents=True)
            (fresh / 'sentinel').write_text('new SDK')
            provision.publish_sdk(root, fresh)
            self.assertEqual((root / 'install').resolve(), fresh)
            self.assertEqual((root / 'install/sentinel').read_text(), 'new SDK')
            self.assertEqual(next((root / 'retired').glob('install-*/sentinel')).read_text(), 'old SDK')
            next_sdk = root / 'builds' / 'next' / 'install'
            next_sdk.mkdir(parents=True)
            if os.name == 'nt':
                # Windows cannot replace a directory symlink with os.replace. The macOS-only publisher's
                # selection boundary is observed here; POSIX hosts execute the atomic filesystem operation.
                with patch.object(Path, 'replace', autospec=True) as replace:
                    provision.publish_sdk(root, next_sdk)
                pending, target = replace.call_args.args
                self.assertEqual(target, root / 'install')
                self.assertEqual(pending.name, 'install')
                self.assertTrue(pending.parent.name.startswith('.install-next-'))
                self.assertFalse(pending.parent.exists())
                self.assertEqual((root / 'install').resolve(), fresh)
            else:
                provision.publish_sdk(root, next_sdk)
                self.assertEqual((root / 'install').resolve(), next_sdk)
            self.assertEqual((fresh / 'sentinel').read_text(), 'new SDK')

    def test_publication_ignores_interrupted_pending_entries_and_preserves_unexpected_paths(self):
        for kind in ('symlink', 'dangling-symlink', 'file', 'directory'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                fresh = root / 'builds/new/install'
                fresh.mkdir(parents=True)
                pending = root / '.install-next'
                if kind in ('symlink', 'dangling-symlink'):
                    target = root / ('builds/new/install' if kind == 'symlink' else 'builds/missing/install')
                    pending.symlink_to(target.relative_to(root), target_is_directory=True)
                    before = pending.readlink()
                elif kind == 'file':
                    pending.write_text('user file')
                else:
                    pending.mkdir()
                    (pending / 'sentinel').write_text('user directory')
                provision.publish_sdk(root, fresh)
                self.assertEqual((root / 'install').resolve(), fresh)
                if kind in ('symlink', 'dangling-symlink'):
                    self.assertEqual(pending.readlink(), before)
                elif kind == 'file':
                    self.assertEqual(pending.read_text(), 'user file')
                else:
                    self.assertEqual((pending / 'sentinel').read_text(), 'user directory')
                self.assertEqual(list(root.glob('.install-next-*')), [])

    def test_crashed_publication_cannot_block_next_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            old = root / 'builds/old/install'
            fresh = root / 'builds/new/install'
            old.mkdir(parents=True)
            fresh.mkdir(parents=True)
            (root / 'install').symlink_to(old.relative_to(root), target_is_directory=True)
            child = subprocess.run([sys.executable, '-B', '-c',
                                    'import os, sys; from pathlib import Path; '
                                    'from tools.live.provision_macos_usd import publish_sdk; '
                                    'Path.replace = lambda *args: os._exit(91); '
                                    'publish_sdk(Path(sys.argv[1]), Path(sys.argv[2]))',
                                    str(root), str(fresh)], check=False)
            self.assertEqual(child.returncode, 91)
            self.assertEqual((root / 'install').resolve(), old)
            orphan = next(root.glob('.install-next-*'))
            orphan_target = (orphan / 'install').readlink()
            if os.name == 'nt':
                with patch.object(Path, 'replace', autospec=True) as replace:
                    provision.publish_sdk(root, fresh)
                self.assertEqual(replace.call_args.args[1], root / 'install')
                self.assertEqual((root / 'install').resolve(), old)
            else:
                provision.publish_sdk(root, fresh)
                self.assertEqual((root / 'install').resolve(), fresh)
            self.assertEqual((orphan / 'install').readlink(), orphan_target)

    def test_failed_publication_cleans_its_pending_link_and_preserves_selected_sdk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            old = root / 'builds/old/install'
            fresh = root / 'builds/new/install'
            old.mkdir(parents=True)
            fresh.mkdir(parents=True)
            (root / 'install').symlink_to(old.relative_to(root), target_is_directory=True)
            with patch.object(Path, 'replace', side_effect=OSError('publication interrupted')):
                with self.assertRaisesRegex(OSError, 'publication interrupted'):
                    provision.publish_sdk(root, fresh)
            self.assertEqual((root / 'install').resolve(), old)
            self.assertEqual(list(root.glob('.install-next-*')), [])

    def test_publication_preserves_unexpected_selection_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fresh = root / 'builds/new/install'
            fresh.mkdir(parents=True)
            (root / 'install').write_text('user file')
            with self.assertRaisesRegex(ValueError, 'not an install directory'):
                provision.publish_sdk(root, fresh)
            self.assertEqual((root / 'install').read_text(), 'user file')
            self.assertFalse((root / 'retired').exists())
            self.assertEqual(list(root.glob('.install-next-*')), [])

    @staticmethod
    def source_history(root):
        source = root / 'upstream'
        subprocess.run(['git', 'init', '--quiet', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'config', 'user.name', 'SDK fixture'], check=True)
        subprocess.run(['git', '-C', str(source), 'config', 'user.email', 'sdk@example.invalid'], check=True)
        pins = []
        environment = dict(os.environ, GIT_AUTHOR_DATE='2000-01-01T00:00:00Z',
                           GIT_COMMITTER_DATE='2000-01-01T00:00:00Z')
        for contents in ('old source', 'new source'):
            (source / 'source.txt').write_text(contents)
            subprocess.run(['git', '-C', str(source), 'add', 'source.txt'], check=True)
            subprocess.run(['git', '-C', str(source), 'commit', '--quiet', '--no-gpg-sign', '-m', contents],
                           env=environment, check=True)
            pins.append(subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip())
        return source, pins

    @staticmethod
    def checkout_bytes(checkout):
        return {str(path.relative_to(checkout)): path.read_bytes()
                for path in checkout.rglob('*') if path.is_file()}

    def test_clean_restored_checkout_advances_to_new_exact_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, (old, new) = self.source_history(root)
            cached = root / 'cached'
            provision.checkout(cached, str(source), old)
            self.assertEqual((cached / 'source.txt').read_text(), 'old source')
            provision.checkout(cached, str(source), new)
            self.assertEqual((cached / 'source.txt').read_text(), 'new source')
            self.assertEqual(subprocess.check_output(['git', '-C', str(cached), 'rev-parse', 'HEAD'],
                                                    text=True).strip(), new)
            with patch.object(provision, 'run') as run:
                provision.checkout(cached, 'unavailable-upstream', new)
                run.assert_not_called()

    def test_failed_initial_fetch_keeps_unborn_checkout_for_an_explicit_later_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, (_, pin) = self.source_history(root)
            cached = root / 'cached'
            with patch.object(provision, 'run', wraps=provision.run) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    provision.checkout(cached, str(root / 'unavailable-upstream'), pin)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(run.call_args_list[0].args[0], ['git', 'init', cached])
                self.assertEqual(run.call_args_list[1].args[0][3], 'fetch')
            self.assertTrue((cached / '.git/HEAD').is_file())
            self.assertFalse((cached / 'source.txt').exists())
            with patch.object(provision, 'run', wraps=provision.run) as run:
                provision.checkout(cached, str(source), pin)
                self.assertEqual([call.args[0][3] for call in run.call_args_list], ['fetch', 'checkout'])
            self.assertEqual((cached / 'source.txt').read_text(), 'new source')
            self.assertEqual(subprocess.check_output(['git', '-C', str(cached), 'rev-parse', 'HEAD'],
                                                    text=True).strip(), pin)

    def test_dirty_unborn_checkout_preserves_staged_and_untracked_work_before_fetch(self):
        for staged in (False, True):
            with self.subTest(staged=staged), tempfile.TemporaryDirectory() as directory:
                cached = Path(directory) / 'cached'
                subprocess.run(['git', 'init', '--quiet', str(cached)], check=True)
                (cached / 'local.txt').write_text('local work')
                if staged:
                    subprocess.run(['git', '-C', str(cached), 'add', 'local.txt'], check=True)
                before = self.checkout_bytes(cached)
                for reuse_only in (False, True):
                    with patch.object(provision, 'run') as run:
                        with self.assertRaisesRegex(ValueError, 'must be clean at'):
                            provision.checkout(cached, 'unavailable-upstream', 'pin', reuse_only=reuse_only)
                        run.assert_not_called()
                    self.assertEqual(self.checkout_bytes(cached), before)

    def test_reuse_only_unborn_checkout_makes_no_mutation_or_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            cached = Path(directory) / 'cached'
            subprocess.run(['git', 'init', '--quiet', str(cached)], check=True)
            before = self.checkout_bytes(cached)
            with patch.object(provision, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'must be clean at'):
                    provision.checkout(cached, 'unavailable-upstream', 'pin', reuse_only=True)
                run.assert_not_called()
            self.assertEqual(self.checkout_bytes(cached), before)

    def test_invalid_repository_or_corrupt_head_fails_without_mutation(self):
        for kind in ('non-repository', 'invalid-head', 'missing-object'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                cached = Path(directory) / 'cached'
                cached.mkdir()
                if kind != 'non-repository':
                    subprocess.run(['git', 'init', '--quiet', str(cached)], check=True)
                    if kind == 'invalid-head':
                        (cached / '.git/HEAD').write_text('invalid HEAD\n')
                    else:
                        reference = subprocess.check_output(['git', '-C', str(cached), 'symbolic-ref', 'HEAD'],
                                                            text=True).strip()
                        target = cached / '.git' / reference
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text('1' * 40 + '\n')
                before = self.checkout_bytes(cached)
                with patch.object(provision, 'run') as run:
                    error = ValueError if kind == 'non-repository' else subprocess.CalledProcessError
                    with self.assertRaises(error):
                        provision.checkout(cached, 'unavailable-upstream', 'pin')
                    run.assert_not_called()
                self.assertEqual(self.checkout_bytes(cached), before)

    def test_dirty_checkout_preserves_tracked_and_untracked_work_before_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, (old, new) = self.source_history(root)
            cached = root / 'cached'
            provision.checkout(cached, str(source), old)
            (cached / 'source.txt').write_text('local edit')
            (cached / 'untracked.txt').write_text('local untracked work')
            before = self.checkout_bytes(cached)
            for reuse_only in (False, True):
                with patch.object(provision, 'run') as run:
                    with self.assertRaisesRegex(ValueError, 'must be clean at'):
                        provision.checkout(cached, str(source), new, reuse_only=reuse_only)
                    run.assert_not_called()
                self.assertEqual(self.checkout_bytes(cached), before)

    def test_reuse_only_pin_mismatch_or_missing_checkout_makes_no_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, (old, new) = self.source_history(root)
            cached = root / 'cached'
            provision.checkout(cached, str(source), old)
            before = self.checkout_bytes(cached)
            with patch.object(provision, 'run') as run:
                with self.assertRaisesRegex(ValueError, 'must be clean at'):
                    provision.checkout(cached, str(source), new, reuse_only=True)
                missing = root / 'missing'
                with self.assertRaisesRegex(ValueError, 'checkout is missing'):
                    provision.checkout(missing, str(source), old, reuse_only=True)
                run.assert_not_called()
            self.assertFalse(missing.exists())
            self.assertEqual(self.checkout_bytes(cached), before)

    def test_existing_checkout_requires_exact_clean_source_for_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / '.git').mkdir()
            for actual, dirty in (("other", ""), ("expected", " M file")):
                with self.subTest(actual=actual, dirty=dirty), \
                        patch.object(provision.subprocess, "check_output", side_effect=[actual, dirty]), \
                        patch.object(provision, "run") as run:
                    with self.assertRaisesRegex(ValueError, "must be clean at expected"):
                        provision.checkout(Path(directory), "url", "expected", reuse_only=True)
                    run.assert_not_called()

    def test_clean_pinned_checkout_needs_no_network(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(provision.subprocess, "check_output", side_effect=["expected", ""]), \
                patch.object(provision, "run") as run:
            (Path(directory) / '.git').mkdir()
            provision.checkout(Path(directory), "url", "expected")
            run.assert_not_called()

    def test_new_checkout_fetches_exact_commit(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(provision.subprocess, "check_output", side_effect=["expected", ""]), \
                patch.object(provision, "run") as run:
            target = Path(directory) / "source"
            provision.checkout(target, "url", "expected")
            self.assertEqual(run.call_args_list[1].args[0],
                             ["git", "-C", target, "fetch", "--depth=1", "url", "expected"])

    def test_unsupported_host_fails_before_any_build(self):
        with patch.object(provision.platform, "system", return_value="Linux"), \
                patch.object(provision, "run") as run:
            with self.assertRaises(SystemExit):
                provision.main(["--python-root", "/sdk", "--cmake", "/cmake"])
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
