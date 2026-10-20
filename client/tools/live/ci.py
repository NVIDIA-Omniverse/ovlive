#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline CI entry point for an explicitly provisioned USD 25.11 worker.

No downloads, shell commands, test-selection overrides, artifact reuse or retries.
A passing run qualifies this profile only, not the complete product release.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live import build, platforms, sanitizer, stack_dump, test, writer_campaign
from tools.live.test_support import run_process, write_json, writer_seed
from tools.release import archive

BASE_SUITES = (
    'foundation', 'lock-noalloc', 'wire-envelope', 'usd-values',
    'wire-scene', 'scene-journal', 'sdf-scene', 'runtime-foundation',
    'diagnostics-schema', 'reference-service', 'native-reference', 'scene-reference',
    'resolver-reference', 'plugin-reference', 'writer-fuzz', 'resolver-probe',
)
PROFILES = ('presubmit', 'asan', 'nightly', 'tsan', 'soak')
# Qualification requires the runner's child-process containment on these hosts.
QUALIFICATION_PLATFORMS = ('linux-x86_64', 'linux-aarch64', 'windows-x86_64')
# The soak runs one long writer campaign and nothing else: its suite, drivers and watchdogs.
SOAK_SUITES = ['writer-fuzz']
SOAK_DRIVERS = ('plugin-driver', 'scene-transport-driver')
# Per-suite failure watchdogs: a campaign profile's own, which covers the slower suites on Windows as well.
SOAK_SUITE_TIMEOUT_MS = writer_campaign.PROFILES['soak']['campaign_timeout_ms']
NIGHTLY_SUITE_TIMEOUT_MS = writer_campaign.PROFILES['nightly']['campaign_timeout_ms']
SUITE_TIMEOUT_MS = writer_campaign.PROFILES['ci']['campaign_timeout_ms']
SOAK_CONTRACT_TIMEOUT_MS = 12_600_000
# Profiles that build instrumented binaries, and the sanitizer kind each selects.
SANITIZER_KINDS = {'asan': 'address', 'tsan': 'thread'}
# Suites whose manifest result carries `tests_run`, one JUnit testcase per recorded case.
CASE_SUITES = {'foundation', 'reference-service', 'native-reference', 'scene-reference', 'resolver-reference',
               'plugin-reference', 'python-reference', 'writer-fuzz', 'resolver-probe'}
NOTICE_FILES = build.NOTICES
SDK_KEYS = {'usd_root', 'python_root', 'dependency_root', 'pybind11_root'}
TOOL_KEYS = {'cmake', 'cc', 'cxx', 'protoc', 'grpc_cpp_plugin'}
EXTRA_TOOLS = {'reference_python', 'reference_openssl', 'diagnostic_python'}
DRIVERS = {
    'lock-driver': 'live_lock_noalloc_test',
    'wire-driver': 'live_wire_envelope_test',
    'usd-values-driver': 'live_usd_values_test',
    'wire-scene-driver': 'live_wire_scene_test', 'scene-journal-driver': 'live_scene_journal_test',
    'sdf-driver': 'live_sdf_scene_test', 'runtime-driver': 'live_runtime_foundation',
    'unary-driver': 'live_unary_driver', 'scene-transport-driver': 'live_scene_transport_driver',
    'resolver-runtime-driver': 'live_resolver_runtime_driver', 'plugin-driver': 'live_plugin_driver',
    'resolver-driver': 'live_resolver_probe',
}


# Manifest key, replay path key, build output, native validation boundary.
NATIVE_IDENTITIES = (
    ('lock_identity', 'lock_driver', 'live_lock_noalloc_test', 'lock_noalloc'),
    ('wire_identity', 'wire_driver', 'live_wire_envelope_test', 'wire_envelope'),
    ('usd_values_identity', 'usd_values_driver', 'live_usd_values_test', 'usd_values'),
    ('wire_scene_identity', 'wire_scene_driver', 'live_wire_scene_test', 'wire_scene'),
    ('scene_journal_identity', 'scene_journal_driver', 'live_scene_journal_test', 'scene_journal'),
    ('sdf_identity', 'sdf_driver', 'live_sdf_scene_test', 'sdf_scene'),
    ('runtime_identity', 'runtime_driver', 'live_runtime_foundation', 'runtime_foundation'),
    ('unary_identity', 'unary_driver', 'live_unary_driver', 'unary_transport'),
    ('scene_transport_identity', 'scene_transport_driver', 'live_scene_transport_driver', 'scene_transport'),
    ('resolver_runtime_identity', 'resolver_runtime_driver', 'live_resolver_runtime_driver', 'resolver_runtime'),
    ('plugin_identity', 'plugin_driver', 'live_plugin_driver', 'plugin'),
    ('resolver_identity', 'resolver_driver', 'live_resolver_probe', 'resolver_probe'),
)
URI_IDENTITY = ('native_identity', 'uri_library', 'liblive_uri_contract_probe.so', 'uri_component')


def build_artifact(platform, name):
    """The build tree file of the NATIVE_IDENTITIES or URI_IDENTITY artifact `name` on `platform`."""
    if platforms.get(platform).binary_format != 'pe':
        return name
    return 'live_uri_contract_probe.dll' if name == URI_IDENTITY[2] else name + '.exe'


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate configuration key: ' + key)
        value[key] = item
    return value


def read_config(path):
    with Path(path).open("rb") as source:
        payload = source.read(32769)
    if len(payload) > 32768:
        raise ValueError('CI configuration exceeds 32 KiB')
    config = json.loads(payload, object_pairs_hook=unique_object)
    if (not isinstance(config, dict) or set(config) not in ({'schema_version', 'tools', 'sdks'},
                                                         {'schema_version', 'tools', 'sdks', 'usd_profile'})
            or type(config['schema_version']) is not int or config['schema_version'] != 1):
        raise ValueError('Expected schema_version=1, tools, sdks and optional usd_profile only')
    if (not isinstance(config.get('usd_profile', 'stock-25.11'), str)
            or config.get('usd_profile', 'stock-25.11') not in build.USD_PROFILES):
        raise ValueError('Unknown USD profile')
    tools, sdks = config['tools'], config['sdks']
    if (not isinstance(tools, dict) or not isinstance(sdks, dict) or set(sdks) != SDK_KEYS
            or set(tools) not in (TOOL_KEYS | EXTRA_TOOLS | {'make'}, TOOL_KEYS | EXTRA_TOOLS | {'ninja'})):
        raise ValueError('Expected exact SDK/tool keys and one make or ninja')
    for key, value in {**tools, **sdks}.items():
        if not isinstance(value, str) or not Path(value).is_absolute() or any(c in value for c in '\0\r\n;'):
            raise ValueError('Expected an absolute input path: ' + key)
        build._path(value, directory=key in sdks, executable=key in tools, preserve_selection=True)
    return config


def clean_environment():
    environment = {key: value for key, value in os.environ.items()
                   if key in {'PATH', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'PATHEXT', 'TEMP', 'TMP', 'USERPROFILE'}
                   # Windows host identity and MSVC header/library discovery need these values.
                   or (os.name == 'nt' and key in {'INCLUDE', 'LIB', 'LIBPATH',
                                                'PROCESSOR_ARCHITECTURE', 'PROCESSOR_ARCHITEW6432'})}
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
    return environment


def plan(config, output, profile, seed, parallel, fresh=True):
    """The CI plan for one profile; `fresh` requires a new output directory, as every CI run does."""
    if profile not in PROFILES:
        raise ValueError('Unknown CI profile')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Seed must be a nonnegative signed 64-bit integer')
    if type(parallel) is not int or not 1 <= parallel <= 8:
        raise ValueError('Parallel compilation must be between 1 and 8')
    output = Path(output).absolute()
    if fresh and (output.exists() or output.is_symlink()):
        raise ValueError('CI output must be a fresh directory')
    tools, sdks = config['tools'], config['sdks']
    host = platforms.host()
    if host.name not in QUALIFICATION_PLATFORMS:
        raise ValueError(f'{host.name} has no supported supervised CI qualification; '
                         'native package builds and direct component checks are separate')
    args = SimpleNamespace(root=str(ROOT), platform=host.name, config='release',
                           usd_profile=config.get('usd_profile', 'stock-25.11'),
                           target_dir=str(output / 'stage'), build_dir=str(output / 'build'),
                           parallel=parallel, make=tools.get('make'), ninja=tools.get('ninja'),
                           **{key: tools[key] for key in TOOL_KEYS}, **sdks)
    native = build.plan(args)
    kind = SANITIZER_KINDS.get(profile, 'none')
    if kind == 'thread' and args.usd_profile != 'stock-25.11':
        raise ValueError('The tsan profile runs only on stock-25.11: tools/live/tsan.supp names its USD libraries and namespace')
    long = profile in {'nightly', 'tsan'}
    # The tests run from the build tree, so the test build keeps its build tree RPATH.
    settings = {'BUILD_TESTING': 'ON', 'CMAKE_BUILD_WITH_INSTALL_RPATH': 'OFF', 'LIVE_BUILD_RESOLVER_PROBE': 'ON',
                'LIVE_BUILD_RESOLVER_RUNTIME': 'ON', 'Python3_EXECUTABLE': sys.executable,
                'LIVE_BUILD_PYTHON': 'OFF' if kind != 'none' else 'ON',
                'LIVE_ENABLE_SANITIZERS': 'OFF', 'LIVE_SANITIZER': kind}
    configure = [arg for arg in native['commands'][0]
                 if not any(arg.startswith('-D' + key + '=') for key in settings)]
    dependency_root = [arg.split('=', 1)[1] for arg in configure if arg.startswith('-DLIVE_DEPENDENCY_ROOT=')]
    if len(dependency_root) != 1:
        raise ValueError('Build plan has no single dependency root')
    configure += [f'-D{key}={value}' for key, value in sorted(settings.items())]
    soak = profile == 'soak'
    suites = list(SOAK_SUITES if soak else BASE_SUITES)
    if kind == 'none' and not soak:
        suites += ['uri', 'python-reference']
    test_command = [sys.executable, '-B', str(ROOT / 'tools/live/test.py'), '--suite', ','.join(suites),
                    '--seed', str(seed), '--artifact-dir', str(output / 'tests'),
                    '--writer-fuzz-profile', writer_profile(profile),
                    '--timeout-ms', str(SOAK_SUITE_TIMEOUT_MS if soak else NIGHTLY_SUITE_TIMEOUT_MS if long
                                        else SUITE_TIMEOUT_MS)]
    # Windows executables and libraries carry their own suffixes and names.
    windows = host.binary_format == 'pe'
    exe = '.exe' if windows else ''
    for option, name in DRIVERS.items():
        if not soak or option in SOAK_DRIVERS:
            test_command += ['--' + option, str(output / 'build' / (name + exe))]
    for key in sorted(EXTRA_TOOLS):
        test_command += ['--' + key.replace('_', '-'), tools[key]]
    if kind == 'none' and not soak:
        uri_library = 'live_uri_contract_probe.dll' if windows else 'liblive_uri_contract_probe.so'
        test_command += ['--uri-library', str(output / 'build' / uri_library),
                         '--python-extension', str(output / 'build/plugin/omni_usd_live' / build.python_extension(host.name))]
    commands = [
        ('reference-identity', [tools['reference_python'], '-B', str(ROOT / 'tools/live/reference.py'),
                                '--openssl', tools['reference_openssl'], '--identity'], 30000),
        ('diagnostic-identity', [tools['diagnostic_python'], '-B', str(ROOT / 'tools/live/diagnostic_snapshot.py'),
                                 '--identity'], 30000),
        ('notices', [sys.executable, '-B', str(ROOT / 'tools/release/third_party_notices.py'), '--check',
                     '--dependency-root', dependency_root[0], '--config', args.config,
                     '--pybind11-root', sdks['pybind11_root']], 30000),
        ('configure', configure, 120000),
        ('build', [tools['cmake'], '--build', str(output / 'build'), '--parallel', str(parallel)], 1800000),
        ('contracts', test_command, SOAK_CONTRACT_TIMEOUT_MS if soak else 5400000 if long else 2400000),
        ('uri-native', [str(output / 'build' / ('live_uri_sanitize' + exe)), str(seed)], 60000),
    ]
    # Signal-driven stack capture and its probe exist on Linux only.
    if host.system == 'Linux':
        commands.append(('stack-dump', [sys.executable, '-B', str(ROOT / 'tests/native/support/stack_dump_check.py'),
                                        str(output / 'build/live_stack_dump_probe'), kind], 60000))
    commands += [
        ('install', [tools['cmake'], '--install', str(output / 'build'), '--prefix', str(output / 'install')], 120000),
    ]
    commands.append(('package-native', [sys.executable, '-B', str(ROOT / 'tests/native/plugin/package.py'),
                     str(output / 'package'), str(output / 'build' / ('live_plugin_load_probe' + exe)), str(ROOT)], 120000))
    if kind == 'none':
        commands.append(('package-python', [native['tools']['python'], '-B',
                         str(ROOT / 'tests/native/python/package.py'), str(output / 'package'), sdks['usd_root']], 120000))
    if profile in {'presubmit', 'nightly'}:
        # The release package: build.py in its own build directory, archived, then loaded from the extracted folder.
        root = archive.archive_root(archive.version(), args.usd_profile, host.name)
        release = output / 'release-check' / root / 'omni_usd_live'
        commands += [
            ('package-stage', [sys.executable, '-B', str(ROOT / 'tools/live/build.py'), '--platform', host.name,
                               '--config', 'release', '--usd-profile', args.usd_profile,
                               '--target-dir', str(output / 'release-stage'), '--build-dir', str(output / 'release-build'),
                               '--parallel', str(parallel), '--usd-root', sdks['usd_root'],
                               '--python-root', sdks['python_root'], '--dependency-root', sdks['dependency_root'],
                               '--pybind11-root', sdks['pybind11_root'],
                               *[f'--{key.replace("_", "-")}=' + tools[key]
                                 for key in ('cmake', 'cc', 'cxx', 'protoc', 'grpc_cpp_plugin')],
                               '--make' if 'make' in tools else '--ninja', tools.get('make') or tools['ninja']], 1800000),
            ('package-archive', [sys.executable, '-B', str(ROOT / 'tools/release/archive.py'), 'create',
                                 '--package', str(output / 'release-stage/usd/omniverse/live'),
                                 '--output-dir', str(output / 'release'),
                                 '--extract-to', str(output / 'release-check'), '--allow-dirty'], 300000),
            ('package-archive-native', [sys.executable, '-B', str(ROOT / 'tests/native/plugin/package.py'), str(release),
                                        str(output / 'build' / ('live_plugin_load_probe' + exe)), str(ROOT)], 120000),
            ('package-archive-python', [native['tools']['python'], '-B', str(ROOT / 'tests/native/python/package.py'),
                                        str(release), sdks['usd_root']], 120000),
        ]
    if kind == 'thread' and host.system == 'Linux':
        setarch = shutil.which('setarch')
        if not setarch:
            raise ValueError('The tsan profile needs setarch')
        # GCC 12's libtsan aborts with "unexpected memory mapping" under this kernel's high-entropy mmap ASLR and does not re-exec without it; setarch -R disables ASLR for the step and every descendant.
        commands = [(name, [setarch, host.machines[0], '-R', *command] if name in {'contracts', 'uri-native', 'stack-dump', 'package-native'}
                     else command, timeout) for name, command, timeout in commands]
    return dict(schema_version=1, boundary='linux-usd2511-ci', profile=profile,
                usd_profile=args.usd_profile,
                seed=seed, output=str(output), suites=suites, build_plan=native,
                config=config, source_fingerprint=test.fingerprint(),
                runner_identity=dict(path=sys.executable, sha256=build._hash(sys.executable), version=sys.version),
                steps=[dict(name=name, command=command, timeout_ms=timeout, state='pending')
                       for name, command, timeout in commands])


def writer_profile(profile):
    return profile if profile in {'nightly', 'soak'} else 'ci'


def verify_runner(directory, expected_suites, source_fingerprint, kind):
    runs = list(directory.iterdir())
    if len(runs) != 1 or runs[0].is_symlink() or not runs[0].is_dir():
        raise ValueError('Expected exactly one fresh test run')
    run = runs[0]
    manifest = json.loads((run / 'manifest.json').read_text())
    results = manifest.get('results', [])
    if (manifest.get('complete') is not True or manifest.get('passed') is not True
            or manifest.get('source_fingerprint') != source_fingerprint
            or [result.get('suite') for result in results] != expected_suites):
        raise ValueError('Missing, stale or incomplete mandatory contract execution')
    for index, result in enumerate(results):
        if (result.get('outcome') != 'passed' or result.get('returncode') != 0
                or result.get('cleanup_complete') is not True or result.get('output_truncated') is not False):
            raise ValueError('Failed/incomplete contract: ' + result['suite'])
        if result['suite'] in {'reference-service', 'native-reference', 'scene-reference',
                               'resolver-reference', 'plugin-reference', 'python-reference', 'uri'}:
            log = (run / f'{index}.log').read_text()
            counts = re.findall(r'^Ran ([0-9]+) tests? in ', log, re.M)
            if len(counts) != 1 or int(counts[0]) <= 0 or re.search(r'^OK \(|\.\.\. skipped ', log, re.M):
                raise ValueError('Missing, zero or skipped tests: ' + result['suite'])
        if result['suite'] in test.NATIVE_CASES and not test.native_cases_ran(
                (run / f'{index}.log').read_text(errors='replace'), test.NATIVE_CASES[result['suite']]):
            raise ValueError('Missing native case count: ' + result['suite'])
    # Suites with recorded cases carry one JUnit testcase per case; the others carry one suite-level testcase.
    expected_tests = [result.get('tests_run') if result['suite'] in CASE_SUITES else 1 for result in results]
    junit = ET.parse(run / 'junit.xml').getroot()
    suites = junit.findall('testsuite')
    if (junit.tag != 'testsuites' or junit.get('failures') != '0' or junit.get('errors') != '0'
            or junit.findall('.//skipped') or [node.get('name') for node in suites] != expected_suites
            or any(type(count) is not int or count <= 0 for count in expected_tests)
            or junit.get('tests') != str(sum(expected_tests))
            or any(node.get('tests') != str(count) or len(node.findall('testcase')) != count
                   or node.get('failures') != '0' or node.get('errors') != '0'
                   for node, count in zip(suites, expected_tests))):
        raise ValueError('Incomplete JUnit evidence')
    native_count = 0
    for key, value in manifest.items():
        if key.endswith('_identity') and isinstance(value, dict) and 'sanitizers' in value:
            native_count += 1
            if value['sanitizers'] is not (kind != 'none') or sanitizer.kind_of(value) != kind:
                raise ValueError('Wrong sanitizer candidate: ' + key)
    if not native_count:
        raise ValueError('Missing native candidate identities')
    return run, manifest


def verify_profile(record, manifest, run):
    replay = json.loads((run / 'replay.json').read_text())
    kind = SANITIZER_KINDS.get(record['profile'], 'none')
    sanitized = kind != 'none'
    runtime = sanitizer.provenance(kind) if sanitized else {}
    seed_record = dict(schema_version=1, master_seed=record['seed'],
                       writer_seeds=[writer_seed(record['seed'], 0)])
    soak = record['profile'] == 'soak'
    if (manifest.get('seed') != seed_record or replay.get('master_seed') != record['seed']
            or replay.get('writer_fuzz_profile') != writer_profile(record['profile']) or replay.get('clients') != 1
            or replay.get('suites') != record['suites']
            or replay.get('source_fingerprint') != record['source_fingerprint']
            or manifest.get('sanitizer_runtime') != runtime or replay.get('sanitizer_runtime') != runtime):
        raise ValueError('CI seed, campaign, suites or sanitizer recipe differs')
    identities = NATIVE_IDENTITIES if sanitized else (*NATIVE_IDENTITIES, URI_IDENTITY)
    if soak:
        identities = [entry for entry in NATIVE_IDENTITIES if entry[1].replace('_', '-') in SOAK_DRIVERS]
    for key, path_key, name, boundary in identities:
        artifact = (Path(record['output']) / 'build' / build_artifact(platforms.host().name, name)).resolve()
        if replay.get(path_key) != str(artifact):
            raise ValueError('Candidate is outside the fresh CI build: ' + path_key)
        actual = test.native_identity(artifact, boundary)
        if (actual.get('configuration') != 'Release' or actual.get('sanitizers') is not sanitized
                or sanitizer.kind_of(actual) != kind or manifest.get(key) != actual or replay.get(key) != actual):
            raise ValueError('Missing or different required candidate: ' + key)
    if soak:
        return
    if not sanitized:
        from tools.live.python_identity import validate
        artifact = (Path(record['output']) / 'build/plugin/omni_usd_live'
                    / build.python_extension(record['build_plan']['platform'])).resolve()
        actual = validate(artifact)
        if (replay.get('python_extension') != str(artifact) or manifest.get('python_binding_identity') != actual
                or replay.get('python_binding_identity') != actual):
            raise ValueError('Missing or different required Python candidate')
    elif manifest.get('python_binding_identity') is not None or manifest.get('native_identity') is not None:
        raise ValueError('Unexpected Python/URI ctypes candidate in the sanitizer profile')



def checkout_state():
    """The checkout record for the run summary: `state` is `clean`, `modified` or `unknown`, and `changed_files` counts modified, staged and untracked non-ignored files, or is None when git or the repository is unavailable."""
    try:
        changed = archive.git_changes()
    except (OSError, ValueError):
        return dict(state='unknown', changed_files=None)
    return dict(state='clean' if changed == 0 else 'modified', changed_files=changed)

def execute(record):
    output = Path(record['output'])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', record)
    summary = dict(schema_version=1, profile=record['profile'], source_fingerprint=record['source_fingerprint'],
                   seed=record['seed'], complete=False, passed=False, results=[])
    write_json(output / 'summary.json', summary)
    environment = clean_environment()
    kind = SANITIZER_KINDS.get(record['profile'], 'none')
    if kind != 'none':
        environment.update(sanitizer.runtime_options(kind, output / 'tsan'))
    native = record['build_plan']
    try:
        # The archive step accepts a modified checkout, so the summary names the checkout it qualified.
        summary['checkout'] = checkout_state()
        write_json(output / 'summary.json', summary)
        support = build._support_identity(native)
        write_json(output / 'support-identity.json', support)
        for step in record['steps']:
            print('CI step: ' + step['name'], flush=True)
            step['state'] = 'running'
            write_json(output / 'plan.json', record)
            # The release build keeps the configure environment and runs from the checkout.
            package_check = step['name'].startswith('package-') and step['name'] != 'package-stage'
            step_environment = dict(environment)
            usd, python = Path(native['sdks']['usd_root']), Path(native['sdks']['python_root'])
            if sys.platform == 'win32':
                # Windows has no RPATH, so every built binary finds the SDK DLLs on PATH; the package checks also run
                # dumpbin from the MSVC environment.
                step_environment['PATH'] = os.pathsep.join(
                    [str(usd / 'lib'), str(usd / 'bin'), str(python), os.environ.get('PATH', '')])
            elif package_check and step['name'] != 'package-archive':
                # Archiving runs git and hashes files; only the load checks need the SDK libraries.
                step_environment['LD_LIBRARY_PATH'] = os.pathsep.join([str(usd / 'lib'), str(python / 'lib')])
            result = asdict(run_process(step['command'], cwd=output if package_check else ROOT,
                                       output=output / (step['name'] + '.log'),
                                       timeout_ms=step['timeout_ms'], max_output_bytes=16 * 1024 * 1024,
                                       environment=step_environment))
            result['stack_dumps'] = test.artifact_names(result['stack_dumps'], output)
            summary['results'].append(dict(name=step['name'], **result))
            step['state'] = result['outcome']
            write_json(output / 'plan.json', record)
            write_json(output / 'summary.json', summary)
            if (result['outcome'] != 'passed' or result['returncode'] != 0
                    or result['cleanup_complete'] is not True or result['output_truncated'] is not False):
                raise ValueError('CI step did not complete: ' + step['name'])
            if step['name'] == 'contracts':
                run, manifest = verify_runner(output / 'tests', record['suites'], record['source_fingerprint'], kind)
                verify_profile(record, manifest, run)
                for key, name in [('reference_identity', 'reference-identity'),
                                  ('diagnostic_schema_identity', 'diagnostic-identity')]:
                    # Only the diagnostics-schema suite records the schema interpreter's identity.
                    if key == 'diagnostic_schema_identity' and 'diagnostics-schema' not in record['suites']:
                        continue
                    if manifest.get(key) != json.loads((output / (name + '.log')).read_text()):
                        raise ValueError('Dependency identity changed: ' + key)
                summary['run'] = str(run)
            if step['name'] == 'install':
                package = output / 'package'
                build.assemble_install(output / 'install', package)
                for name, source in NOTICE_FILES.items():
                    if (package / 'share/omni_usd_live' / name).read_bytes() != (ROOT / source).read_bytes():
                        raise ValueError('Installed notice differs: ' + name)
                if kind == 'none':
                    build._validate_layout(package, native['platform'], exact=True)
            if step['name'] == 'package-archive':
                archives = sorted((output / 'release').glob('*.tar.gz'))
                if len(archives) != 1:
                    raise ValueError('Expected exactly one release archive')
                digest = (output / 'release' / (archives[0].name + '.sha256')).read_text().split()[0]
                # The source revision and state pair two runs of one commit, whose archives `check --against` diffs.
                extracted = json.loads((output / 'release-check' / archives[0].name[:-len(archive.SUFFIX)]
                                        / archive.MANIFEST).read_text())
                summary['release_archive'] = dict(name=archives[0].name, sha256=digest,
                                                  git_head=extracted['git_head'], dirty=extracted['dirty'])
            if kind == 'thread':
                reports = sorted(path.name for path in output.glob('tsan.*'))
                if reports:
                    raise ValueError(f'ThreadSanitizer reported in step {step["name"]}: ' + ', '.join(reports))
        if (test.fingerprint() != record['source_fingerprint'] or build._support_identity(native) != support
                or build._hash(sys.executable) != record['runner_identity']['sha256']):
            raise ValueError('Source or SDK/tools changed during CI run')
        summary.update(complete=True, passed=True)
    except (OSError, ValueError, KeyError, TypeError, ET.ParseError, KeyboardInterrupt) as error:
        summary['reason'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else str(error)
    finally:
        write_json(output / 'plan.json', record)
        write_json(output / 'summary.json', summary)
    print('CI artifacts: ' + str(output), flush=True)
    if not summary['passed']:
        print('CI failed: ' + summary.get('reason', 'incomplete'), file=sys.stderr)
    return 0 if summary['passed'] else 1


def main():
    stack_dump.install()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='fresh owned run directory')
    parser.add_argument('--profile', choices=('presubmit', 'asan', 'nightly', 'tsan', 'soak'), default='presubmit')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        record = plan(read_config(args.config), args.output, args.profile, args.seed, args.parallel)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return 0
    return execute(record)


if __name__ == '__main__':
    raise SystemExit(main())
