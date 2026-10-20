#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute every required writer campaign case in a fresh supervised process."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests/reference_service')]
from tools.live import sanitizer, stack_dump
from tools.live.reference import identity as reference_identity
from tools.live.test import artifact_names, fingerprint, native_identity
from tools.live.test_support import run_process, write_json
from tools.live.writer_campaign import canonical, case_passed, generate, verify_summary
from writer_history import read_json


def artifact_bytes(directory):
    total = 0
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise ValueError('unexpected symlink in writer case artifacts')
        if path.is_file():
            total += path.stat().st_size
    return total


def execute(plan, root, commands, environment, *, runner=run_process):
    """Record progress before each case; stop on its first failure without retrying."""
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'plan.json', plan)
    summary = dict(schema_version=1, plan=plan, complete=False, passed=False, results=[])
    write_json(root / 'summary.json', summary)
    for case in plan['cases']:
        directory = root / case['id']
        directory.mkdir()
        specification = directory / 'case.json'
        write_json(specification, dict(case=case, context=plan['context']))
        command = [*commands[case['driver']], '--writer-case', str(specification)]
        entry = dict(id=case['id'], passed=False, process=dict(outcome='pending'), command=command)
        summary['results'].append(entry)
        write_json(root / 'summary.json', summary)
        print(json.dumps(dict(writer_case=case, command=command), sort_keys=True), flush=True)
        result = runner(command, cwd=ROOT, output=directory / 'process.log',
                        timeout_ms=case.get('timeout_ms', plan['budgets']['case_timeout_ms']), max_output_bytes=1024 * 1024,
                        environment=dict(environment, LIVE_TEST_ARTIFACT_DIR=str(directory),
                                         LIVE_TEST_SEED=str(case['seed'])))
        entry['process'] = dict(asdict(result), stack_dumps=artifact_names(result.stack_dumps, directory))
        try:
            entry['test'] = read_json(directory / 'writer-case-result.json')
            entry['artifact_bytes'] = artifact_bytes(directory)
            entry['passed'] = (result.outcome == 'passed' and type(result.returncode) is int and result.returncode == 0 and result.cleanup_complete and
                               not result.output_truncated and case_passed(case, entry['test']) and
                               entry['artifact_bytes'] <= plan['budgets']['case_artifact_bytes'])
        except (OSError, ValueError, KeyError, TypeError) as error:
            entry['evidence_error'] = str(error)
        write_json(root / 'summary.json', summary)
        if not entry['passed']:
            break
    summary['complete'] = True
    summary['passed'] = len(summary['results']) == len(plan['cases']) and all(e['passed'] for e in summary['results'])
    write_json(root / 'summary.json', summary)
    if summary['passed']:
        verify_summary(plan, summary)
    return summary


def main():
    stack_dump.install()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('ci', 'nightly', 'soak'), required=True)
    parser.add_argument('--plugin-driver', type=Path, required=True)
    parser.add_argument('--scene-driver', type=Path, required=True)
    parser.add_argument('--openssl', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    args = parser.parse_args()
    root = Path(os.environ['LIVE_TEST_ARTIFACT_DIR']) / 'writer-fuzz'
    expected = read_json(args.context, limit=16 * 1024 * 1024)

    def context():
        return dict(source=fingerprint(), plugin=native_identity(args.plugin_driver, 'plugin'),
                    scene=native_identity(args.scene_driver, 'scene_transport'),
                    reference=reference_identity(args.openssl),
                    sanitizer=sanitizer.recorded(os.environ))

    actual = context()
    if canonical(actual) != canonical(expected):
        raise ValueError('writer campaign preflight identity differs')
    plan = generate(args.profile, int(os.environ['LIVE_TEST_SEED']))
    plan['context'] = actual
    dependencies = {entry['role']: entry['path'] for entry in actual['plugin']['dependencies']}
    command = [sys.executable, '-B', str(ROOT / 'tools/live/reference.py'), '--openssl', str(args.openssl)]
    commands = dict(plugin=[*command, '--plugin-driver', str(args.plugin_driver),
                           '--plugin-plug-info', dependencies['plug_info'],
                           '--diagnostics-sample', dependencies['diagnostics_sample']],
                    scene=[*command, '--scene-transport-driver', str(args.scene_driver)])
    environment = dict(os.environ, PXR_PLUGINPATH_NAME='', PXR_AR_DEFAULT_SEARCH_PATH='', AR_DEFAULT_RESOLVER='')
    summary = execute(plan, root, commands, environment)
    # Concurrent source/artifact mutation invalidates a passing campaign.
    if canonical(context()) != canonical(expected):
        summary.update(passed=False, identity_error='writer campaign identity changed during execution')
        write_json(root / 'summary.json', summary)
    print(json.dumps(dict(profile=args.profile, passed=summary['passed'], cases=len(summary['results']),
                          required=len(plan['cases']), artifacts=str(root)), sort_keys=True))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
