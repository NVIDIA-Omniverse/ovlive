# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI must reject missing coverage and preserve the first failed step."""
import contextlib
import copy
import io
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from tools.live import ci
from tools.live import sanitizer
from tools.live.test_support import ProcessResult


class CiTests(unittest.TestCase):
    def setUp(self):
        host = patch.object(ci.platforms, 'host', return_value=ci.platforms.PLATFORMS['linux-x86_64'])
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

    def test_clean_environment_preserves_windows_architecture_fallback(self):
        architecture = dict(PROCESSOR_ARCHITECTURE='x86', PROCESSOR_ARCHITEW6432='AMD64')
        for system, name in (('win32', 'nt'), ('linux', 'posix'), ('darwin', 'posix')):
            for cleaner in (ci.clean_environment, ci.test.clean_environment):
                with self.subTest(system=system, cleaner=cleaner.__module__), \
                        patch.object(sys, 'platform', system), patch.object(os, 'name', name), \
                        patch.dict(os.environ, dict(architecture, SERVICE_SECRET='synthetic-secret',
                                                   PYTHONPATH='ambient-shadow'), clear=True):
                    environment = cleaner()
                    for key, value in architecture.items():
                        self.assertEqual(environment.get(key), value if name == 'nt' else None)
                    self.assertNotIn('SERVICE_SECRET', environment)
                    self.assertNotEqual(environment.get('PYTHONPATH'), 'ambient-shadow')

    def test_filtered_python_identifies_host_without_wmi_or_a_shell(self):
        probe = (
            "import platform; from unittest.mock import patch\n"
            "with patch.object(platform, '_wmi_query', side_effect=OSError('WMI unavailable'), create=True), "
            "patch.object(platform, '_uname_cache', None):\n"
            "    print(platform.machine())\n")
        expected = 'AMD64' if sys.platform == 'win32' else platform.machine()
        for cleaner in (ci.clean_environment, ci.test.clean_environment):
            with self.subTest(cleaner=cleaner.__module__), \
                    patch.dict(os.environ, dict(PROCESSOR_ARCHITECTURE='x86', PROCESSOR_ARCHITEW6432='AMD64')):
                result = subprocess.run([sys.executable, '-c', probe], env=cleaner(),
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_path_only_config_rejects_duplicates_unknown_and_ambiguous_inputs(self):
        config = self.config()
        path = self.root / 'config.json'
        path.write_text(json.dumps(config))
        self.assertEqual(ci.read_config(path), config)
        kit = dict(config, usd_profile='kit-25.11')
        path.write_text(json.dumps(kit))
        self.assertEqual(ci.read_config(path), kit)
        bad_configs = []
        bad_configs.append(dict(config, usd_profile='unrecognized'))
        unknown = copy.deepcopy(config)
        unknown['commands'] = ['must never execute']
        bad_configs.append(unknown)
        ambiguous = copy.deepcopy(config)
        ambiguous['tools']['ninja'] = ambiguous['tools']['make']
        bad_configs.append(ambiguous)
        relative = copy.deepcopy(config)
        relative['tools']['cmake'] = 'ambient-cmake'
        bad_configs.append(relative)
        omitted = copy.deepcopy(config)
        del omitted['sdks']['usd_root']
        bad_configs.append(omitted)
        for value in bad_configs:
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                ci.read_config(path)
        path.write_text(json.dumps(config).replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            ci.read_config(path)
        path.write_text(' ' * 32769)
        with self.assertRaisesRegex(ValueError, '32 KiB'):
            ci.read_config(path)

    def make_plan(self, profile='presubmit', output='output'):
        config = self.config()
        native = dict(platform=ci.platforms.host().name, commands=[['cmake', '-DBUILD_TESTING=OFF', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
                                 '-DLIVE_DEPENDENCY_ROOT=/synthetic/deps', '-DLIVE_ENABLE_SANITIZERS=OFF',
                                 '-DLIVE_BUILD_PYTHON=ON']], tools={'python': '/synthetic/sdk/python'},
                      sdks=config['sdks'])
        with patch.object(ci.build, 'plan', return_value=native), patch.object(ci.test, 'fingerprint', return_value='source'):
            return ci.plan(config, self.root / output, profile, 42, 2)

    def test_fixed_profiles_include_required_work_and_never_reuse_output(self):
        kinds = {'presubmit': 'none', 'asan': 'address', 'nightly': 'none', 'tsan': 'thread'}
        for profile in ('presubmit', 'asan', 'nightly', 'tsan'):
            # The host may lack setarch; wrapping has its own test below.
            with patch.object(ci.platforms, 'host', return_value=ci.platforms.PLATFORMS['linux-x86_64']), \
                 patch.object(ci.shutil, 'which', return_value='/synthetic/setarch'):
                record = self.make_plan(profile)
            sanitized = kinds[profile] != 'none'
            self.assertTrue(set(ci.BASE_SUITES) <= set(record['suites']))
            self.assertIn('lock-noalloc', record['suites'])
            self.assertEqual('python-reference' in record['suites'], not sanitized)
            self.assertEqual('uri' in record['suites'], not sanitized)
            self.assertEqual(len(record['suites']), len(set(record['suites'])))
            steps = {step['name']: step for step in record['steps']}
            self.assertIn('package-native', steps)
            notices = steps['notices']['command']
            self.assertEqual(notices[notices.index('--dependency-root') + 1], '/synthetic/deps')
            self.assertEqual(notices[notices.index('--config') + 1], 'release')
            self.assertEqual(notices[notices.index('--pybind11-root') + 1], record['config']['sdks']['pybind11_root'])
            self.assertIn('--check', notices)
            self.assertNotIn('--write', notices)
            self.assertLess(list(steps).index('notices'), list(steps).index('configure'))
            self.assertEqual('package-python' in steps, not sanitized)
            self.assertTrue(all(step['state'] == 'pending' for step in steps.values()))
            configure = steps['configure']['command']
            self.assertEqual([arg for arg in configure if arg.startswith('-DBUILD_TESTING=')], ['-DBUILD_TESTING=ON'])
            self.assertEqual([arg for arg in configure if arg.startswith('-DCMAKE_BUILD_WITH_INSTALL_RPATH=')],
                             ['-DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF'])
            self.assertEqual([arg for arg in configure if arg.startswith('-DLIVE_ENABLE_SANITIZERS=')],
                             ['-DLIVE_ENABLE_SANITIZERS=OFF'])
            self.assertEqual([arg for arg in configure if arg.startswith('-DLIVE_SANITIZER=')],
                             ['-DLIVE_SANITIZER=' + kinds[profile]])
            self.assertEqual([arg for arg in configure if arg.startswith('-DLIVE_BUILD_PYTHON=')],
                             ['-DLIVE_BUILD_PYTHON=' + ('OFF' if sanitized else 'ON')])
            contract = steps['contracts']['command']
            self.assertEqual('--uri-library' in contract, not sanitized)
            self.assertEqual('--python-extension' in contract, not sanitized)
            long = profile in {'nightly', 'tsan'}
            self.assertEqual(contract[contract.index('--timeout-ms') + 1], '3000000' if long else '300000')
            self.assertEqual(steps['contracts']['timeout_ms'], 5400000 if long else 2400000)
            self.assertEqual(contract[contract.index('--lock-driver') + 1],
                             str(Path(record['output']) / 'build/live_lock_noalloc_test'))
            self.assertEqual(contract[contract.index('--writer-fuzz-profile') + 1],
                             'nightly' if profile == 'nightly' else 'ci')
            self.assertEqual(steps['stack-dump']['command'][-2:],
                             [str(Path(record['output']) / 'build/live_stack_dump_probe'), kinds[profile]])
        (self.root / 'output').mkdir()
        with self.assertRaisesRegex(ValueError, 'fresh'):
            self.make_plan()

    def test_soak_profile_runs_only_the_soak_campaign(self):
        record = self.make_plan('soak')
        self.assertEqual(record['suites'], ['writer-fuzz'])
        steps = {step['name']: step for step in record['steps']}
        contract = steps['contracts']['command']
        self.assertEqual(contract[contract.index('--suite') + 1], 'writer-fuzz')
        self.assertEqual(contract[contract.index('--writer-fuzz-profile') + 1], 'soak')
        self.assertEqual(contract[contract.index('--timeout-ms') + 1], '12300000')
        self.assertEqual(steps['contracts']['timeout_ms'], 12600000)
        inputs = sorted(arg for arg in contract if arg.startswith('--') and arg not in {
            '--suite', '--seed', '--artifact-dir', '--writer-fuzz-profile', '--timeout-ms'})
        self.assertEqual(inputs, sorted(['--plugin-driver', '--scene-transport-driver', '--reference-python',
                                         '--reference-openssl', '--diagnostic-python']))
        build = Path(record['output']) / 'build'
        exe = '.exe' if record['build_plan']['platform'] == 'windows-x86_64' else ''
        self.assertEqual(contract[contract.index('--plugin-driver') + 1], str(build / f'live_plugin_driver{exe}'))
        self.assertEqual(contract[contract.index('--scene-transport-driver') + 1],
                         str(build / f'live_scene_transport_driver{exe}'))
        self.assertFalse({'package-stage', 'package-archive', 'package-archive-native', 'package-archive-python'} & set(steps))
        if os.name != 'nt':  # The stack-dump probe is Linux-only.
            self.assertEqual(steps['stack-dump']['command'][-2:], [str(build / 'live_stack_dump_probe'), 'none'])
        configure = steps['configure']['command']
        self.assertIn('-DLIVE_SANITIZER=none', configure)
        # The replay names the soak campaign and only the two drivers it runs.
        run = self.root / 'run'
        run.mkdir()
        manifest = dict(seed=dict(schema_version=1, master_seed=42, writer_seeds=[ci.writer_seed(42, 0)]),
                        sanitizer_runtime={})
        replay = dict(master_seed=42, clients=1, suites=['writer-fuzz'], source_fingerprint='source',
                      writer_fuzz_profile='soak', sanitizer_runtime={})
        outputs = {}
        for key, path_key, name, boundary in ci.NATIVE_IDENTITIES:
            if key not in {'plugin_identity', 'scene_transport_identity'}:
                continue
            artifact = str((build / ci.build_artifact(ci.platforms.host().name, name)).resolve())
            identity = dict(configuration='Release', sanitizers=False, boundary=boundary, artifact=artifact)
            manifest[key] = replay[key] = outputs[(artifact, boundary)] = identity
            replay[path_key] = artifact
        with patch.object(ci.test, 'native_identity', side_effect=lambda path, boundary: outputs[(str(path), boundary)]):
            (run / 'replay.json').write_text(json.dumps(replay))
            ci.verify_profile(record, manifest, run)
            (run / 'replay.json').write_text(json.dumps(dict(replay, writer_fuzz_profile='nightly')))
            with self.assertRaises(ValueError):
                ci.verify_profile(record, manifest, run)
            (run / 'replay.json').write_text(json.dumps(replay))
            for key in ('plugin_identity', 'scene_transport_identity'):
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'required candidate'):
                    ci.verify_profile(record, {k: v for k, v in manifest.items() if k != key}, run)

    def test_macos_package_has_no_supervised_qualification(self):
        self.assertTrue(ci.build.packaged('macos-arm64'))
        with patch.object(ci.platforms, 'host', return_value=ci.platforms.PLATFORMS['macos-arm64']), \
             patch.object(ci.build, 'plan') as build_plan:
            for profile in ci.PROFILES:
                with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, 'no supported supervised CI'):
                    ci.plan(self.config(), self.root / profile, profile, 42, 2)
            build_plan.assert_not_called()

    def test_profile_lists_agree(self):
        import re
        source = (Path(ci.__file__)).read_text()
        choices = re.findall(r"'([\w-]+)'", re.search(r"'--profile', choices=\(([^)]*)\)", source).group(1))
        self.assertEqual(tuple(choices), ci.PROFILES)
        self.assertEqual(ci.QUALIFICATION_PLATFORMS, ('linux-x86_64', 'linux-aarch64', 'windows-x86_64'))
        self.assertTrue(all(ci.build.packaged(name) for name in ci.QUALIFICATION_PLATFORMS))

    def test_soak_contracts_need_no_diagnostic_schema_identity(self):
        # The soak runs no diagnostics-schema suite, so its manifest records no schema identity.
        record = self.make_plan('soak')
        output = self.root / 'output'
        seen = []
        def process(command, *, output, **options):
            name = output.name[:-len('.log')]
            seen.append(name)
            output.write_text(json.dumps({'identity': name}) if name.endswith('identity') else 'log\n')
            return ProcessResult('failed' if name == 'uri-native' else 'passed', 0, 1, 0, False, True)
        manifest = dict(reference_identity={'identity': 'reference-identity'})
        with patch.object(ci.build, '_support_identity', return_value={}), \
             patch.object(ci, 'run_process', side_effect=process), \
             patch.object(ci, 'verify_runner', return_value=(output / 'tests/run', manifest)), \
             patch.object(ci, 'verify_profile'), contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ci.execute(record), 1)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(seen[-1], 'uri-native')
        self.assertEqual(summary['reason'], 'CI step did not complete: uri-native')
        record = self.make_plan('presubmit', output='presubmit')
        with patch.object(ci.build, '_support_identity', return_value={}), \
             patch.object(ci, 'run_process', side_effect=process), \
             patch.object(ci, 'verify_runner', return_value=(self.root / 'presubmit/tests/run', manifest)), \
             patch.object(ci, 'verify_profile'), contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ci.execute(record), 1)
        summary = json.loads((self.root / 'presubmit/summary.json').read_text())
        self.assertEqual(summary['reason'], 'Dependency identity changed: diagnostic_schema_identity')

    def test_unsanitized_profiles_build_archive_and_load_the_release(self):
        archive_steps = ['package-stage', 'package-archive', 'package-archive-native', 'package-archive-python']
        root = 'ovlive-0.1.0-stock-25.11-linux-x86_64'
        for profile in ('presubmit', 'nightly', 'asan', 'tsan'):
            with patch.object(ci.platforms, 'host', return_value=ci.platforms.PLATFORMS['linux-x86_64']), \
                 patch.object(ci.shutil, 'which', return_value='/synthetic/setarch'):
                record = self.make_plan(profile, output=profile)
            names = [step['name'] for step in record['steps']]
            if profile in {'asan', 'tsan'}:
                self.assertEqual(sorted(set(archive_steps) & set(names)), [])
                continue
            self.assertEqual(names[-5:], ['package-python', *archive_steps])
            steps = {step['name']: step for step in record['steps']}
            output = Path(record['output'])
            config = record['config']
            stage = steps['package-stage']['command']
            self.assertEqual(stage[:3], [ci.sys.executable, '-B', str(ci.ROOT / 'tools/live/build.py')])
            for flag, value in [('--platform', 'linux-x86_64'), ('--config', 'release'), ('--usd-profile', 'stock-25.11'),
                                ('--target-dir', str(output / 'release-stage')),
                                ('--build-dir', str(output / 'release-build')), ('--parallel', '2'),
                                ('--usd-root', config['sdks']['usd_root']), ('--make', config['tools']['make'])]:
                self.assertEqual(stage[stage.index(flag) + 1], value, flag)
            self.assertIn('--grpc-cpp-plugin=' + config['tools']['grpc_cpp_plugin'], stage)
            self.assertEqual(steps['package-stage']['timeout_ms'], 1800000)
            self.assertEqual(steps['package-archive']['command'][2:],
                             [str(ci.ROOT / 'tools/release/archive.py'), 'create',
                              '--package', str(output / 'release-stage/usd/omniverse/live'),
                              '--output-dir', str(output / 'release'), '--extract-to', str(output / 'release-check'),
                              '--allow-dirty'])
            folder = str(output / 'release-check' / root / 'omni_usd_live')
            self.assertEqual(steps['package-archive-native']['command'][2:],
                             [str(ci.ROOT / 'tests/native/plugin/package.py'), folder,
                              str(output / 'build/live_plugin_load_probe'), str(ci.ROOT)])
            self.assertEqual(steps['package-archive-python']['command'],
                             ['/synthetic/sdk/python', '-B', str(ci.ROOT / 'tests/native/python/package.py'),
                              folder, config['sdks']['usd_root']])

    def test_tsan_profile_wraps_instrumented_steps_in_setarch(self):
        wrapped = {'contracts', 'uri-native', 'stack-dump', 'package-native'}
        with patch.object(ci.platforms, 'host', return_value=ci.platforms.PLATFORMS['linux-x86_64']), \
             patch.object(ci.shutil, 'which', return_value='/synthetic/setarch'):
            record = self.make_plan('tsan')
            steps = {step['name']: step for step in record['steps']}
            for name, step in steps.items():
                with self.subTest(step=name):
                    self.assertEqual(step['command'][:3] == ['/synthetic/setarch', 'x86_64', '-R'], name in wrapped)
            contract = steps['contracts']['command']
            self.assertEqual(contract[contract.index('--timeout-ms') + 1], '3000000')
            self.assertEqual(steps['contracts']['timeout_ms'], 5400000)
            self.assertEqual(steps['build']['command'][0], record['config']['tools']['cmake'])
            asan = self.make_plan('asan', output='asan')
            self.assertFalse(any(step['command'][0] == '/synthetic/setarch' for step in asan['steps']))
        with patch.object(ci.platforms, 'host', return_value=ci.platforms.PLATFORMS['linux-x86_64']), \
             patch.object(ci.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(ValueError, 'needs setarch'):
                self.make_plan('tsan', output='missing')
        with patch.object(ci.platforms, 'host', return_value=ci.platforms.PLATFORMS['linux-aarch64']), \
             patch.object(ci.shutil, 'which', return_value='/synthetic/setarch'):
            record = self.make_plan('tsan', output='other')
            for step in record['steps']:
                with self.subTest(step=step['name'], machine='aarch64'):
                    self.assertEqual(step['command'][:3] == ['/synthetic/setarch', 'aarch64', '-R'], step['name'] in wrapped)

    def test_tsan_profile_refuses_the_kit_usd_profile(self):
        config = self.config()
        config['usd_profile'] = 'kit-25.11'
        native = dict(commands=[['cmake', '-DLIVE_DEPENDENCY_ROOT=/synthetic/deps']],
                      tools={'python': '/synthetic/sdk/python'}, sdks=config['sdks'])
        with patch.object(ci.build, 'plan', return_value=native), patch.object(ci.test, 'fingerprint', return_value='source'), \
             patch.object(ci.shutil, 'which', return_value='/synthetic/setarch'):
            with self.assertRaisesRegex(ValueError, 'stock-25.11'):
                ci.plan(config, self.root / 'kit', 'tsan', 42, 2)
            self.assertEqual(ci.plan(config, self.root / 'kit-asan', 'asan', 42, 2)['usd_profile'], 'kit-25.11')

    def test_failed_step_is_preserved_and_later_steps_never_launch(self):
        record = self.make_plan()
        seen = []
        def process(command, **options):
            plan = json.loads((self.root / 'output/plan.json').read_text())
            self.assertEqual(plan['steps'][len(seen)]['state'], 'running')
            self.assertEqual(plan['seed'], 42)
            self.assertIn('package-python', [step['name'] for step in plan['steps']])
            seen.append(command)
            return ProcessResult('failed', 23, 1, 0, False, True)
        with patch.object(ci.build, '_support_identity', return_value={}), \
             patch.object(ci, 'run_process', side_effect=process), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(ci.execute(record), 1)
        summary = json.loads((self.root / 'output/summary.json').read_text())
        self.assertEqual(len(seen), 1)
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['results'][0]['returncode'], 23)
        self.assertFalse((self.root / 'output/build').exists())

    def test_summary_records_the_checkout_state(self):
        # The CI archive is a check of a possibly modified checkout, so the summary names the boundary.
        cases = ((0, dict(state='clean', changed_files=0)), (3, dict(state='modified', changed_files=3)),
                 (FileNotFoundError('git'), dict(state='unknown', changed_files=None)),
                 (ValueError('not a repository'), dict(state='unknown', changed_files=None)))
        for index, (changed, expected) in enumerate(cases):
            output = f'checkout-{index}'
            record = self.make_plan(output=output)
            with patch.object(ci.build, '_support_identity', return_value={}), \
                 patch.object(ci.archive, 'git_changes', side_effect=[changed]), \
                 patch.object(ci, 'run_process', return_value=ProcessResult('failed', 1, 1, 0, False, True)), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(ci.execute(record), 1)
            self.assertEqual(json.loads((self.root / output / 'summary.json').read_text())['checkout'], expected)

    def runner_fixture(self):
        directory = self.root / 'tests'
        run = directory / 'run-exact'
        run.mkdir(parents=True)
        result = dict(suite='plugin-reference', outcome='passed', returncode=0,
                      cleanup_complete=True, output_truncated=False, tests_run=1)
        manifest = dict(complete=True, passed=True, source_fingerprint='source',
                        results=[result], plugin_identity={'sanitizers': False})
        (run / 'manifest.json').write_text(json.dumps(manifest))
        (run / '0.log').write_text('Ran 1 test in 0.01s\n\nOK\n')
        self.write_junit(run, {'plugin-reference': ['test_plugin.PluginTests.test_one']})
        return directory, run, manifest

    @staticmethod
    def write_junit(run, suites):
        xml = ET.Element('testsuites', name='live', tests=str(sum(map(len, suites.values()))),
                         failures='0', errors='0')
        for suite, cases in suites.items():
            node = ET.SubElement(xml, 'testsuite', name=suite, tests=str(len(cases)), failures='0', errors='0')
            for case in cases:
                ET.SubElement(node, 'testcase', classname='live.' + suite, name=case)
        ET.ElementTree(xml).write(run / 'junit.xml')

    def test_incomplete_stale_wrong_sanitizer_or_failed_execution_never_passes(self):
        directory, run, manifest = self.runner_fixture()
        self.assertEqual(ci.verify_runner(directory, ['plugin-reference'], 'source', 'none')[0], run)
        for key, value in [('complete', False), ('passed', False), ('source_fingerprint', 'stale'),
                           ('results', []), ('plugin_identity', {'sanitizers': True}),
                           ('plugin_identity', {'sanitizers': False, 'sanitizer': 'thread'})]:
            broken = dict(manifest, **{key: value})
            (run / 'manifest.json').write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                ci.verify_runner(directory, ['plugin-reference'], 'source', 'none')
        for kind, identity in [('address', {'sanitizers': True}), ('address', {'sanitizers': True, 'sanitizer': 'address'}),
                               ('thread', {'sanitizers': True, 'sanitizer': 'thread'})]:
            (run / 'manifest.json').write_text(json.dumps(dict(manifest, plugin_identity=identity)))
            self.assertEqual(ci.verify_runner(directory, ['plugin-reference'], 'source', kind)[0], run)
        for kind, identity in [('thread', {'sanitizers': True}), ('address', {'sanitizers': True, 'sanitizer': 'thread'}),
                               ('none', {'sanitizers': True, 'sanitizer': 'address'})]:
            (run / 'manifest.json').write_text(json.dumps(dict(manifest, plugin_identity=identity)))
            with self.assertRaisesRegex(ValueError, 'sanitizer'):
                ci.verify_runner(directory, ['plugin-reference'], 'source', kind)
        for key, value in [('outcome', 'timeout'), ('returncode', 2), ('cleanup_complete', False),
                           ('output_truncated', True)]:
            broken = copy.deepcopy(manifest)
            broken['results'][0][key] = value
            (run / 'manifest.json').write_text(json.dumps(broken))
            with self.assertRaises(ValueError):
                ci.verify_runner(directory, ['plugin-reference'], 'source', 'none')

    def test_zero_skipped_missing_junit_and_multiple_runs_never_pass(self):
        directory, run, _ = self.runner_fixture()
        for log in ('', 'Ran 0 tests in 0.01s\nOK\n', 'Ran 1 test in 0.01s\nOK (skipped=1)\n'):
            (run / '0.log').write_text(log)
            with self.assertRaises(ValueError):
                ci.verify_runner(directory, ['plugin-reference'], 'source', 'none')
        (run / '0.log').write_text('Ran 1 test in 0.01s\nOK\n')
        (run / 'junit.xml').write_text('<testsuite tests="0" failures="0" errors="0"/>')
        with self.assertRaises(ValueError):
            ci.verify_runner(directory, ['plugin-reference'], 'source', 'none')
        (directory / 'unexpected-old-run').mkdir()
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            ci.verify_runner(directory, ['plugin-reference'], 'source', 'none')

    def test_junit_case_totals_must_equal_the_suite_evidence(self):
        directory, run, _ = self.runner_fixture()
        for cases in ([], ['test_plugin.PluginTests.test_one', 'test_plugin.PluginTests.test_two']):
            self.write_junit(run, {'plugin-reference': cases})
            with self.assertRaisesRegex(ValueError, 'JUnit'):
                ci.verify_runner(directory, ['plugin-reference'], 'source', 'none')
        self.write_junit(run, {'reference-service': ['test_plugin.PluginTests.test_one']})
        with self.assertRaisesRegex(ValueError, 'JUnit'):
            ci.verify_runner(directory, ['plugin-reference'], 'source', 'none')
        self.write_junit(run, {'plugin-reference': ['test_plugin.PluginTests.test_one']})
        tree = ET.parse(run / 'junit.xml')
        ET.SubElement(tree.getroot().find('testsuite/testcase'), 'skipped')
        tree.write(run / 'junit.xml')
        with self.assertRaisesRegex(ValueError, 'JUnit'):
            ci.verify_runner(directory, ['plugin-reference'], 'source', 'none')

    def test_native_component_logs_must_report_their_case_count(self):
        directory, run, manifest = self.runner_fixture()
        manifest['results'].append(dict(suite='wire-envelope', outcome='passed', returncode=0,
                                        cleanup_complete=True, output_truncated=False))
        (run / 'manifest.json').write_text(json.dumps(manifest))
        self.write_junit(run, {'plugin-reference': ['test_plugin.PluginTests.test_one'],
                               'wire-envelope': ['wire-envelope']})
        expected = ['plugin-reference', 'wire-envelope']
        for log in ('master_seed=42 scenario=ompf-envelope\n', 'live-cases passed=5\n',
                    'live-cases passed=6\ntrailing output\n', 'live-cases passed=6\nlive-cases passed=6\n'):
            (run / '1.log').write_text(log)
            with self.assertRaisesRegex(ValueError, 'case count'):
                ci.verify_runner(directory, expected, 'source', 'none')
        (run / '1.log').write_text('master_seed=42 scenario=ompf-envelope\nlive-cases passed=6\n')
        self.assertEqual(ci.verify_runner(directory, expected, 'source', 'none')[0], run)


    def test_profile_requires_every_fresh_candidate_and_exact_recipe(self):
        record = self.make_plan('asan')
        run = self.root / 'run'
        run.mkdir()
        runtime = sanitizer.provenance()
        self.assertEqual(runtime['ASAN_OPTIONS'], sanitizer.runtime_options()['ASAN_OPTIONS'])
        manifest = dict(seed=dict(schema_version=1, master_seed=42, writer_seeds=[ci.writer_seed(42, 0)]),
                        sanitizer_runtime=runtime)
        replay = dict(master_seed=42, clients=1, suites=record['suites'], source_fingerprint='source',
                      writer_fuzz_profile='ci', sanitizer_runtime=runtime)
        outputs = {}
        for key, path_key, name, boundary in ci.NATIVE_IDENTITIES:
            artifact = str((Path(record['output']) / 'build' / ci.build_artifact(ci.platforms.host().name, name)).resolve())
            identity = dict(configuration='Release', sanitizers=True, sanitizer='address', boundary=boundary,
                            artifact=artifact)
            manifest[key] = identity
            replay[key] = identity
            replay[path_key] = artifact
            outputs[(artifact, boundary)] = identity
        def validate(path, boundary):
            return outputs[(str(path), boundary)]
        with patch.object(ci.test, 'native_identity', side_effect=validate):
            (run / 'replay.json').write_text(json.dumps(replay))
            ci.verify_profile(record, manifest, run)
            for key, _, _, _ in ci.NATIVE_IDENTITIES:
                broken = dict(manifest)
                del broken[key]
                with self.assertRaisesRegex(ValueError, 'required candidate'):
                    ci.verify_profile(record, broken, run)
            for key, value in [('master_seed', 7), ('writer_fuzz_profile', 'nightly'),
                               ('plugin_driver', '/unrelated/build/live_plugin_driver'),
                               ('sanitizer_runtime', {}), ('sanitizer_runtime', sanitizer.runtime_options()),
                               ('suites', [])]:
                (run / 'replay.json').write_text(json.dumps(dict(replay, **{key: value})))
                with self.assertRaises(ValueError):
                    ci.verify_profile(record, manifest, run)
            (run / 'replay.json').write_text(json.dumps(replay))
            key = (replay['plugin_driver'], 'plugin')
            outputs[key] = dict(outputs[key], configuration='Debug')
            with self.assertRaisesRegex(ValueError, 'required candidate'):
                ci.verify_profile(record, manifest, run)
            outputs[key] = dict(outputs[key], configuration='Release', sanitizer='thread')
            with self.assertRaisesRegex(ValueError, 'required candidate'):
                ci.verify_profile(record, manifest, run)
