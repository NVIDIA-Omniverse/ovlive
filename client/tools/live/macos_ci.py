#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build and check the macOS development package without supervised qualification."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live import build, platforms

TESTS = ('live_lock_noalloc', 'live_wire_envelope', 'live_uri_native_bounds', 'live_uri_contract',
         'live_participant_tests', 'live_diagnostic_log_tests', 'live_diagnostic_log_json',
         'live_plugin_package', 'live_python_package', 'live_installed_package')
COMPONENT_TARGETS = ('live_lock_noalloc_test', 'live_wire_envelope_test',
                     'live_uri_sanitize', 'live_uri_contract_probe')
SDK_KEYS = {'usd_root', 'python_root', 'dependency_root', 'pybind11_root'}
TOOL_KEYS = {'cmake', 'make', 'cc', 'cxx', 'protoc', 'grpc_cpp_plugin'}


def read_config(path):
    config = json.loads(Path(path).read_text())
    if (not isinstance(config, dict) or
            set(config) != {'schema_version', 'purpose', 'usd_profile', 'tools', 'sdks'} or
            type(config['schema_version']) is not int or config['schema_version'] != 1 or
            config['purpose'] != 'macos-source-build' or config['usd_profile'] != 'stock-25.11' or
            not isinstance(config['tools'], dict) or set(config['tools']) != TOOL_KEYS or
            not isinstance(config['sdks'], dict) or set(config['sdks']) != SDK_KEYS):
        raise ValueError('Expected a macos-source-build configuration from provision_macos.py')
    for key, value in {**config['tools'], **config['sdks']}.items():
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError('Expected an absolute input path: ' + key)
    return config


def plan(config, output, parallel):
    if platforms.host().name != 'macos-arm64':
        raise ValueError('requires a native macOS arm64 host')
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('macOS CI output must be a fresh directory')
    args = SimpleNamespace(root=str(ROOT), platform='macos-arm64', config='release',
                           usd_profile='stock-25.11', target_dir=str(output / 'stage'),
                           build_dir=str(output / 'build'), parallel=parallel, ninja=None,
                           **config['tools'], **config['sdks'])
    native = build.plan(args)
    # Direct CTests use the build RPATH; installation supplies the package RPATH.
    configure = native['commands'][0]
    configure[configure.index('-DBUILD_TESTING=OFF')] = '-DBUILD_TESTING=ON'
    configure[configure.index('-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON')] = '-DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF'
    configure.append('-DPython3_EXECUTABLE=' + native['tools']['python'])
    native['commands'][1].extend(COMPONENT_TARGETS)
    ctest = build._path(Path(native['tools']['cmake']).with_name('ctest'), executable=True)
    selection = [str(ctest), '--test-dir', native['build_dir'], '-R', '^(' + '|'.join(TESTS) + ')$']
    return dict(native=native, inventory=[*selection, '--show-only=json-v1'],
                ctest=[*selection, '--no-tests=error', '--timeout', '120', '--output-on-failure',
                       '--output-junit', str(output / 'ctest.xml')])


def check_inventory(inventory):
    tests = inventory['tests']
    if len(tests) != len(TESTS) or {test['name'] for test in tests} != set(TESTS):
        raise ValueError('Required macOS component/package CTests are missing')
    for test in tests:
        if any(prop['name'] == 'DISABLED' and prop['value'] for prop in test.get('properties', [])):
            raise ValueError('Required macOS CTest is disabled: ' + test['name'])


def check_results(path):
    cases = ET.parse(path).getroot().findall('.//testcase')
    if len(cases) != len(TESTS) or {case.get('name') for case in cases} != set(TESTS):
        raise ValueError('Required macOS CTest results are missing')
    for case in cases:
        if case.get('status') != 'run' or any(case.find(tag) is not None for tag in ('failure', 'error', 'skipped')):
            raise ValueError('Required macOS CTest did not pass: ' + case.get('name', 'unknown'))


def execute(recipe, output):
    output = Path(output)
    output.mkdir(parents=True)
    (output / 'plan.json').write_text(json.dumps(recipe, indent=2) + '\n')
    native = recipe['native']
    build.execute(native)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                       DYLD_LIBRARY_PATH=os.pathsep.join((native['sdks']['usd_root'] + '/lib',
                                                         native['sdks']['python_root'] + '/lib')))
    for key in ('PYTHONPATH', 'PYTHONHOME', 'PXR_PLUGINPATH_NAME', 'AR_DEFAULT_RESOLVER'):
        environment.pop(key, None)
    with tempfile.TemporaryDirectory(prefix='live-macos-ci-') as directory:
        inventory = json.loads(subprocess.check_output(recipe['inventory'], cwd=directory,
                                                       env=environment, text=True))
        (output / 'ctest-inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
        check_inventory(inventory)
        subprocess.run(recipe['ctest'], cwd=directory, env=environment, check=True)
        check_results(output / 'ctest.xml')
        # The staged bytes are checked independently of the CMake build package.
        for script, arguments in (
            ('plugin/package.py', [native['target'], str(Path(native['build_dir']) / 'live_plugin_load_probe'), str(ROOT)]),
            ('python/package.py', [native['target'], native['sdks']['usd_root']]),
        ):
            subprocess.run([native['tools']['python'], '-B', str(ROOT / 'tests/native' / script), *arguments],
                           cwd=directory, env=environment, check=True, timeout=120)
    build.validate_stage(native['target'], ROOT, expected_platform='macos-arm64', expected_configuration='release')
    print('macOS development package: ten direct CTests and staged native/Python package checks passed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        recipe = plan(read_config(args.config), args.output.resolve(), args.parallel)
        if args.dry_run:
            print(json.dumps(recipe, indent=2))
        else:
            execute(recipe, args.output.resolve())
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'macOS development package: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
