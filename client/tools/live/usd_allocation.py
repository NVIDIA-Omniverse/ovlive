#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated public-USD allocation diagnostics; completion is never a safety pass."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live import sanitizer
from tools.live.test import native_identity
from tools.live.test_support import run_process, write_json


def records(output, backing, operation, index, sdk_lib):
    values = [json.loads(line) for line in output.splitlines() if line.startswith('{"phase":')]
    if not values or values[0].get('phase') != 'armed':
        raise ValueError('child did not reach the instrumented operation')
    armed = values[0]
    if (armed.get('backing'), armed.get('operation'), armed.get('index'), armed.get('usd_version')) != (
            backing, operation, index, 2511):
        raise ValueError('child identity or injection index mismatch')
    library = Path(armed['sdf_library'])
    if not library.is_absolute():
        raise ValueError('loaded Sdf library must have an absolute path')
    library.resolve().relative_to(sdk_lib.resolve())
    if library.name not in {'libov_25.11usd_ms.so', 'libusd_sdf.so'}:
        raise ValueError('unexpected loaded Sdf library')
    phases = [value['phase'] for value in values]
    if phases not in (['armed'], ['armed', 'returned'], ['armed', 'returned', 'observed']):
        raise ValueError('unexpected or duplicated diagnostic phase')
    return dict(records=values, sdf_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())


def terminal(entry):
    """Only a clean, uninjected success ends a sweep; timeout/death never does."""
    phases = entry['records']
    return (entry['process']['outcome'] == 'passed' and len(phases) == 3 and
            phases[1].get('injected') is False and phases[1].get('caught_bad_alloc') is False and
            phases[1].get('returned_success') is True and
            all(phases[2].get(key) is True for key in ('baseline_unchanged', 'retained_coherent', 'stage_namespace_coherent', 'stage_values_coherent', 'expected_effect')) and
            phases[2].get('sdk_diagnostics') is False)


def unique_selections(values):
    if values is not None and len(set(values)) != len(values):
        raise ValueError('duplicate selections would overwrite child evidence')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--backing', action='append', choices=('stock', 'projection'))
    parser.add_argument('--operation', action='append', choices=('move', 'create', 'field', 'dictionary', 'sample'))
    parser.add_argument('--max-index', type=int, default=128)
    parser.add_argument('--controls-only', action='store_true', help='Verify uninjected effects and clean exits only')
    args = parser.parse_args()
    try:
        unique_selections(args.backing)
        unique_selections(args.operation)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.max_index <= 4096:
        parser.error('--max-index must be 1..4096')
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='run-', dir=args.artifact_dir))
    report = dict(schema_version=1, product_qualified=False, complete=False, controls_only=args.controls_only, results=[], sweeps=[])
    write_json(directory / 'summary.json', report)
    print('Allocation diagnostic artifacts:', directory, flush=True)
    try:
        driver = args.driver.resolve()
        identity = native_identity(driver, 'wire_scene')
        report['candidate'] = identity
        trees = {item['role']: Path(item['path']) for item in identity['dependency_trees']}
        sdk_lib = trees['usd_lib']
        environment = {key: value for key, value in os.environ.items() if key in {'PATH', 'TMPDIR', 'TEMP', 'TMP'}}
        environment.update(LD_LIBRARY_PATH=os.pathsep.join((str(sdk_lib), str(trees['python_lib']))),
                           PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        if sanitizer.kind_of(identity) == 'thread' and not args.controls_only:
            # The thread build keeps the standard allocator, so no child can inject.
            raise ValueError('a ThreadSanitizer driver has no test allocator; run --controls-only')
        if identity['sanitizers']:
            environment.update(sanitizer.runtime_options(sanitizer.kind_of(identity), directory / 'tsan'))
        for backing in args.backing or ('stock', 'projection'):
            for operation in args.operation or ('move', 'create', 'field', 'dictionary', 'sample'):
                sweep = dict(backing=backing, operation=operation, complete=False, failed_children=0, incoherent=0)
                report['sweeps'].append(sweep)
                for index in ((-1,) if args.controls_only else range(-1, args.max_index)):
                    command = [str(driver), '--allocation-probe', backing, operation, str(index)]
                    entry = dict(backing=backing, operation=operation, index=index, command=command)
                    report['results'].append(entry)
                    write_json(directory / 'summary.json', report)
                    log = directory / f'{backing}-{operation}-{index}.log'
                    process = asdict(run_process(command, cwd=ROOT, output=log, timeout_ms=10000,
                                     max_output_bytes=131072, environment=environment))
                    entry['process'] = process
                    if process['cleanup_complete'] is not True or process['output_truncated'] is not False:
                        raise ValueError('child cleanup or evidence incomplete')
                    entry.update(records(log.read_text(), backing, operation, index, sdk_lib))
                    if index == -1:
                        if not terminal(entry):
                            raise ValueError('uninjected stock/projection control failed')
                        if args.controls_only:
                            sweep.update(complete=True, terminal_index=-1)
                    elif terminal(entry):
                        sweep.update(complete=True, terminal_index=index)
                        write_json(directory / 'summary.json', report)
                        break
                    else:
                        sweep['failed_children'] += process['outcome'] != 'passed'
                        observed = entry['records'][-1]
                        sweep['incoherent'] += observed.get('phase') == 'observed' and any(
                            observed.get(key) is False for key in ('baseline_unchanged', 'retained_coherent', 'stage_namespace_coherent', 'stage_values_coherent'))
                    write_json(directory / 'summary.json', report)
                print(json.dumps(sweep, sort_keys=True), flush=True)
        if native_identity(driver, 'wire_scene') != identity:
            raise ValueError('source, driver or SDK changed during diagnostic')
        if len({entry['sdf_library_sha256'] for entry in report['results']}) != 1:
            raise ValueError('diagnostic children loaded different SDK libraries')
        report['complete'] = all(sweep['complete'] for sweep in report['sweeps'])
        if not report['complete']:
            raise ValueError('allocation bound reached before every sweep finished')
        print('Diagnostic completed; observed crashes/inconsistency remain failures, not product qualification.')
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report['error'] = str(error)
        print('Allocation diagnostic incomplete:', error, file=sys.stderr)
        return 1
    finally:
        write_json(directory / 'summary.json', report)


if __name__ == '__main__':
    raise SystemExit(main())
