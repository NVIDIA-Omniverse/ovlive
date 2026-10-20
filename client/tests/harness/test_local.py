# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The parameter-free local commands run exact presubmit CI steps against one reusable output directory."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from tools.live import ci, local

ROOT = Path(__file__).resolve().parents[2]


class FakeCompleted:
    def __init__(self, returncode):
        self.returncode = returncode


class LocalCommandTests(unittest.TestCase):
    def setUp(self):
        host = patch.object(ci.platforms, 'host', return_value=ci.platforms.get('linux-x86_64'))
        host.start()
        self.addCleanup(host.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def config(self):
        tools = {}
        for name in ci.TOOL_KEYS | ci.EXTRA_TOOLS | {'make'}:
            path = self.root / 'tools' / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('synthetic tool, never executed')
            path.chmod(0o700)
            tools[name] = str(path)
        sdks = {}
        for name in ci.SDK_KEYS:
            path = self.root / name
            path.mkdir(exist_ok=True)
            sdks[name] = str(path)
        return dict(schema_version=1, tools=tools, sdks=sdks)

    def native(self, config):
        """The CI tests' stand-in for build.plan(): no SDK tree is read."""
        return dict(commands=[['cmake', '-DBUILD_TESTING=OFF', '-DLIVE_DEPENDENCY_ROOT=/synthetic/deps',
                               '-DLIVE_ENABLE_SANITIZERS=OFF', '-DLIVE_BUILD_PYTHON=ON']],
                    tools={'python': '/synthetic/sdk/python'}, sdks=config['sdks'])

    def test_configuration_path_prefers_the_environment_then_xdg_then_home(self):
        self.assertEqual(local.config_path({local.CONFIG_ENV: '/worker/ovlive.json'}), Path('/worker/ovlive.json'))
        self.assertEqual(local.config_path({'XDG_CONFIG_HOME': '/xdg'}), Path('/xdg/ovlive/worker.json'))
        self.assertEqual(local.config_path({}), Path.home() / '.config/ovlive/worker.json')
        self.assertEqual(local.config_path({local.CONFIG_ENV: ''}), Path.home() / '.config/ovlive/worker.json')

    def test_parallelism_is_bounded_by_the_ci_limits(self):
        self.assertEqual(local.parallel(lambda: None), 2)
        self.assertEqual(local.parallel(lambda: 1), 1)
        self.assertEqual(local.parallel(lambda: 64), 8)

    def test_steps_are_the_presubmit_ci_steps_against_a_reusable_output(self):
        output = self.root / 'local'
        output.mkdir()
        config = self.config()
        with patch.object(ci.build, 'plan', return_value=self.native(config)), \
             patch.object(ci.test, 'fingerprint', return_value='source'):
            build = local.steps(config, 'build', output=output, workers=3)
            test = local.steps(config, 'test', output=output, workers=3)
            fresh = ci.plan(config, self.root / 'fresh', local.PROFILE, local.SEED, 3)
        expected = {step['name']: step['command'] for step in fresh['steps']}
        self.assertEqual([name for name, _ in build], ['configure', 'build'])
        self.assertEqual([name for name, _ in test], ['contracts'])
        for name, command in build + test:
            self.assertEqual(command, [arg.replace(str(self.root / 'fresh'), str(output)) for arg in expected[name]])
        self.assertIn('-DBUILD_TESTING=ON', build[0][1])
        self.assertIn(str(output / 'build'), build[1][1])
        self.assertIn('--artifact-dir', test[0][1])
        self.assertIn(str(output / 'tests'), test[0][1])
        with patch.object(ci.build, 'plan', return_value=self.native(config)), self.assertRaises(ValueError):
            ci.plan(config, output, local.PROFILE, local.SEED, 3)

    def test_steps_stop_at_the_first_failure_and_run_from_the_repository_root(self):
        calls = []

        def runner(command, cwd, env):
            calls.append((command, cwd, env))
            return FakeCompleted(3 if command[0] == 'second' else 0)

        planned = [('one', ['first']), ('two', ['second']), ('three', ['third'])]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(local.run_steps(planned, {'PATH': '/bin'}, runner), 3)
        self.assertEqual([command for command, _, _ in calls], [['first'], ['second']])
        self.assertEqual({cwd for _, cwd, _ in calls}, {ROOT})
        self.assertEqual({tuple(env.items()) for _, _, env in calls}, {(('PATH', '/bin'),)})

    def test_clean_removes_only_the_local_output(self):
        output = self.root / 'local'
        (output / 'build').mkdir(parents=True)
        (self.root / 'ci').mkdir()
        self.assertEqual(local.clean(output), 0)
        self.assertFalse(output.exists())
        self.assertTrue((self.root / 'ci').is_dir())
        self.assertEqual(local.clean(output), 0)

    def test_main_reports_a_missing_configuration_and_a_missing_build(self):
        missing = {local.CONFIG_ENV: str(self.root / 'absent.json')}
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as errors:
            local.main(['build'], environ=missing)
        self.assertIn(str(self.root / 'absent.json'), errors.getvalue())
        self.assertIn(local.CONFIG_ENV, errors.getvalue())
        path = self.root / 'worker.json'
        path.write_text(json.dumps(self.config()))
        with patch.object(local, 'OUTPUT', self.root / 'no-build'):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as errors:
                local.main(['test'], environ={local.CONFIG_ENV: str(path)})
        self.assertIn('build.sh', errors.getvalue())

    def test_main_plans_from_the_configuration_and_runs_the_steps(self):
        path = self.root / 'worker.json'
        path.write_text(json.dumps(self.config()))
        calls = []

        def runner(command, cwd, env):
            calls.append(command)
            return FakeCompleted(0)

        with patch.object(local, 'OUTPUT', self.root / 'local'), contextlib.redirect_stdout(io.StringIO()), \
             patch.object(ci.build, 'plan', return_value=self.native(self.config())), \
             patch.object(ci.test, 'fingerprint', return_value='source'):
            self.assertEqual(local.main(['build'], environ={local.CONFIG_ENV: str(path)}, runner=runner), 0)
        self.assertEqual(len(calls), 2)
        self.assertIn('-DBUILD_TESTING=ON', calls[0])
        self.assertEqual(calls[1][1:3], ['--build', str(self.root / 'local/build')])
        self.assertTrue((self.root / 'local').is_dir())

    def test_root_scripts_run_the_matching_command(self):
        for name in ('build', 'test', 'clean'):
            text = (ROOT / f'{name}.sh').read_text()
            self.assertIn('SPDX-License-Identifier: Apache-2.0', text)
            self.assertIn(f'tools/live/local.py {name}\n', text)
            # The mode the repository records, which a checkout on any platform carries; Windows file modes show no
            # execute bit.
            stage = subprocess.run(['git', 'ls-files', '--stage', f'{name}.sh'], cwd=ROOT, capture_output=True,
                                   text=True, check=True).stdout
            self.assertTrue(stage.startswith('100755 '), (name, stage))


if __name__ == '__main__':
    unittest.main()
