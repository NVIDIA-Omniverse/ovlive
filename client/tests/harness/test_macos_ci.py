# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The macOS source-build lane rejects incomplete checks and propagates failures."""
import copy
import contextlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from tools.live import macos_ci as ci


class MacCiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.output = self.root / 'evidence'
        self.config = dict(schema_version=1, purpose='macos-source-build', usd_profile='stock-25.11',
                           tools={name: str(self.root / 'tools' / name) for name in ci.TOOL_KEYS},
                           sdks={name: str(self.root / 'sdks' / name) for name in ci.SDK_KEYS})
        self.native = dict(root=str(ci.ROOT), platform='macos-arm64', config='release',
                           build_dir=str(self.output / 'build'), target=str(self.output / 'stage'),
                           sdks=self.config['sdks'], tools={**self.config['tools'], 'python': str(self.root / 'python')},
                           commands=[['cmake', '-DBUILD_TESTING=OFF', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON'],
                                     ['cmake', '--build', 'build', '--target', 'omni_usd_live_python'], []])
        self.recipe = dict(native=self.native, inventory=['ctest', '--show-only=json-v1'], ctest=['ctest'])
        self.inventory = dict(tests=[dict(name=name, properties=[]) for name in ci.TESTS])

    def test_workstation_config_is_explicit_and_rejects_supervisor_or_incomplete_inputs(self):
        path = self.root / 'worker.json'
        path.write_text(json.dumps(self.config))
        self.assertEqual(ci.read_config(path), self.config)
        for key in ('purpose', 'protoc', 'usd_profile', 'python_root'):
            config = copy.deepcopy(self.config)
            if key == 'purpose':
                del config[key]
            elif key == 'usd_profile':
                config[key] = 'kit-25.11'
            else:
                del config['tools' if key == 'protoc' else 'sdks'][key]
            path.write_text(json.dumps(config))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'macos-source-build'):
                ci.read_config(path)

    def test_native_plan_uses_sdk_python_build_rpath_and_direct_ctests(self):
        with patch.object(ci.platforms, 'host', return_value=ci.platforms.get('macos-arm64')), \
                patch.object(ci.build, 'plan', return_value=self.native) as plan, \
                patch.object(ci.build, '_path', side_effect=lambda path, **kwargs: path):
            recipe = ci.plan(self.config, self.output, 2)
        args = plan.call_args.args[0]
        self.assertEqual((args.platform, args.usd_profile, args.parallel), ('macos-arm64', 'stock-25.11', 2))
        self.assertEqual(args.python_root, self.config['sdks']['python_root'])
        self.assertIn('-DBUILD_TESTING=ON', recipe['native']['commands'][0])
        self.assertIn('-DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF', recipe['native']['commands'][0])
        self.assertIn('-DPython3_EXECUTABLE=' + self.native['tools']['python'], recipe['native']['commands'][0])
        self.assertIn('--no-tests=error', recipe['ctest'])

    def test_other_host_or_existing_output_rejects_before_build_planning(self):
        for host in ('linux-x86_64', 'windows-x86_64', 'macos-arm64'):
            self.output.mkdir(exist_ok=True)
            with self.subTest(host=host), \
                    patch.object(ci.platforms, 'host', return_value=ci.platforms.get(host)), \
                    patch.object(ci.build, 'plan') as plan:
                with self.assertRaises(ValueError):
                    ci.plan(self.config, self.output, 2)
                plan.assert_not_called()

    def test_inventory_rejects_missing_duplicate_extra_and_disabled_checks(self):
        ci.check_inventory(self.inventory)
        for change in ('missing', 'duplicate', 'extra', 'disabled'):
            inventory = copy.deepcopy(self.inventory)
            if change == 'missing':
                inventory['tests'].pop()
            elif change == 'duplicate':
                inventory['tests'][-1] = inventory['tests'][0]
            elif change == 'extra':
                inventory['tests'].append(dict(name='other'))
            else:
                inventory['tests'][0]['properties'].append(dict(name='DISABLED', value=True))
            with self.subTest(change=change), self.assertRaises(ValueError):
                ci.check_inventory(inventory)

    def execute_with_mocks(self, **kwargs):
        stack = self.stack.enter_context(patch.object(ci.build, 'execute'))
        validate = self.stack.enter_context(patch.object(ci.build, 'validate_stage'))
        self.stack.enter_context(patch.object(ci.subprocess, 'check_output', return_value=json.dumps(self.inventory)))
        self.stack.enter_context(patch.object(ci, 'check_results'))
        run = self.stack.enter_context(patch.object(ci.subprocess, 'run', **kwargs))
        return stack, validate, run

    def test_results_require_every_case_and_reject_skips_even_with_zero_exit_status(self):
        path = self.root / 'ctest.xml'
        def results():
            suite = ET.Element('testsuite')
            for name in ci.TESTS:
                ET.SubElement(suite, 'testcase', name=name, status='run')
            return suite
        ET.ElementTree(results()).write(path)
        ci.check_results(path)
        for change in ('missing', 'duplicate', 'failure', 'error', 'skipped', 'notrun'):
            suite = results()
            if change == 'missing':
                suite.remove(suite[-1])
            elif change == 'duplicate':
                suite[-1].set('name', suite[0].get('name'))
            elif change == 'notrun':
                suite[0].set('status', 'notrun')
            else:
                ET.SubElement(suite[0], change)
            ET.ElementTree(suite).write(path)
            with self.subTest(change=change), self.assertRaises(ValueError):
                ci.check_results(path)

    def test_failed_ctest_retains_inventory_and_never_runs_staged_checks(self):
        _, validate, run = self.execute_with_mocks(side_effect=subprocess.CalledProcessError(1, 'ctest'))
        with self.assertRaises(subprocess.CalledProcessError):
            ci.execute(self.recipe, self.output)
        self.assertTrue((self.output / 'ctest-inventory.json').is_file())
        self.assertTrue((self.output / 'plan.json').is_file())
        self.assertEqual(run.call_count, 1)
        validate.assert_not_called()

    def test_missing_ctest_never_launches_checks(self):
        self.inventory['tests'].pop()
        _, validate, run = self.execute_with_mocks()
        with self.assertRaisesRegex(ValueError, 'missing'):
            ci.execute(self.recipe, self.output)
        run.assert_not_called()
        validate.assert_not_called()

    def test_stage_checks_use_selected_python_and_run_outside_source(self):
        native, validate, run = self.execute_with_mocks()
        ci.execute(self.recipe, self.output)
        native.assert_called_once_with(self.native)
        self.assertEqual(run.call_count, 3)
        for call in run.call_args_list:
            self.assertFalse(Path(call.kwargs['cwd']).is_relative_to(ci.ROOT))
            self.assertTrue(call.kwargs['check'])
            self.assertEqual(call.kwargs['env']['DYLD_LIBRARY_PATH'],
                             os.pathsep.join((self.config['sdks']['usd_root'] + '/lib',
                                              self.config['sdks']['python_root'] + '/lib')))
        for call in run.call_args_list[1:]:
            self.assertEqual(call.args[0][0], self.native['tools']['python'])
            self.assertIn(self.native['target'], call.args[0])
        validate.assert_called_once_with(self.native['target'], ci.ROOT,
                                        expected_platform='macos-arm64', expected_configuration='release')

    def test_staged_import_failure_is_not_a_pass(self):
        _, validate, run = self.execute_with_mocks(side_effect=[None, None, subprocess.CalledProcessError(1, 'import')])
        with self.assertRaises(subprocess.CalledProcessError):
            ci.execute(self.recipe, self.output)
        self.assertEqual(run.call_count, 3)
        validate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
