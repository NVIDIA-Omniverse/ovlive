#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repeatable storage characterization; successful execution is not qualification."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live.test import native_identity, fingerprint
from tools.live.test_support import record_seeds, run_process, write_json

METRICS = ('elapsed_ns', 'cpu_ns', 'latency_p50_ns', 'latency_p95_ns', 'latency_p99_ns',
           'process_peak_rss_before_oracle_kib', 'candidate_cleanup_ns', 'timer_pair_p50_ns')
KINDS = {'field_set', 'array_set', 'time_sample', 'move_pair', 'field_read', 'mixed_pair'}
MEASUREMENT_CONTRACT = {
    'field_set': 'one Set of a prebuilt scalar VtValue',
    'array_set': 'one Set of a prebuilt COW array VtValue; excludes generation/deep copy',
    'time_sample': 'one SetTimeSample of a prebuilt scalar at a precomputed key',
    'move_pair': 'two one-spec MoveSpec calls, each followed by HasSpec/GetSpecType validation',
    'field_read': 'one Get plus VtValue type validation and checksum accumulation',
    'mixed_pair': 'one writer transaction under production write guard: revision/payload Set pair; readers use production read guard for coherent pair, type/invariant/monotonic checks',
    'mixed_epochs': 'four first-write/exact-reader-witness/free-work/end-barrier epochs; synchronization included in total wall/CPU; thread creation/join excluded',
    'mixed_overlap': 'free reads seeing fewer than the epoch target committed writer transactions under data guard; not measured mutex contention; zero counts retained',
    'mixed_throughput': 'writer operations per joint epoch elapsed; each reader reported separately with same joint denominator; witness/free-read counts separate',
    'wall_interval': 'whole operation loop including dispatch, steady-clock samples and latency storage',
    'cpu_interval': 'process CPU readings bracket the wall interval; includes its timer sampling overhead',
    'latency': 'per-operation steady-clock sample pair, no overhead subtraction',
    'rss': 'Linux process-lifetime peak through measured work, includes startup/setup/warmup; excludes later oracle',
    'cleanup': 'candidate backing destruction after correctness verification, excludes SDK/process exit',
    'supervisor_elapsed': 'child lifetime plus process-guard cleanup; not native teardown or drain latency',
}


def integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('integer outside workload bounds')
    return value


def workloads(content):
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError('workload definition must be an object')
    if (data.get('schema_version') != 1 or data.get('boundary') != 'sdf_storage' or
            data.get('qualification') != 'characterization_only'):
        raise ValueError('only version-1 storage characterization workloads are supported')
    warmup = integer(data.get('warmup_operations'), 1, 1000000)
    cases = data.get('cases')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 32:
        raise ValueError('bounded nonempty case matrix required')
    names = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {'id', 'kind', 'specs', 'operations', 'array_elements', 'readers'}:
            raise ValueError('invalid case definition')
        name = case['id']
        if (not isinstance(name, str) or not name or len(name) > 64 or
                any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name) or name in names):
            raise ValueError('unique safe case IDs required')
        names.add(name)
        if case['kind'] not in KINDS:
            raise ValueError('unsupported case kind')
        integer(case['specs'], 1, 100000)
        integer(case['operations'], warmup, 1000000)
        readers = integer(case['readers'], 0, 8)
        if (case['kind'] == 'mixed_pair') != bool(readers):
            raise ValueError('reader count differs from workload kind')
        if readers and (warmup < 4 or warmup % 4 or case['operations'] % 4):
            raise ValueError('mixed counts require four complete epochs')
        elements = integer(case['array_elements'], 0, 65536)
        if (case['kind'] == 'array_set') != bool(elements) or case['operations'] * elements > 16777216:
            raise ValueError('invalid array preparation budget')
    return data


def validate_trial(row, case, seed, warmup):
    if not isinstance(row, dict):
        raise ValueError('native trial must be an object')
    expected = dict(schema_version=1, boundary='sdf_storage', correctness='passed', seed=seed,
                    warmup_operations=warmup, **{k: case[k] for k in ('kind', 'specs', 'operations', 'array_elements', 'readers')})
    for key, value in expected.items():
        if type(row.get(key)) is not type(value) or row[key] != value:
            raise ValueError('native trial differs from requested work or failed correctness: ' + key)
    for key in METRICS:
        integer(row.get(key), 0, (1 << 64) - 1)
    if not row['elapsed_ns'] or not row['process_peak_rss_before_oracle_kib']:
        raise ValueError('missing elapsed time or process RSS')
    if not row['latency_p50_ns'] <= row['latency_p95_ns'] <= row['latency_p99_ns'] <= row['elapsed_ns']:
        raise ValueError('inconsistent latency percentiles')
    readers = row.get('reader_progress')
    if not isinstance(readers, list) or len(readers) != case['readers']:
        raise ValueError('missing per-reader progress')
    for index, reader in enumerate(readers):
        if not isinstance(reader, dict):
            raise ValueError('invalid reader progress')
        for key, value in dict(index=index, operations=case['operations'], witness_reads=4,
                               free_reads=case['operations'] - 4).items():
            if type(reader.get(key)) is not int or reader[key] != value:
                raise ValueError('incorrect reader work: ' + key)
        integer(reader.get('free_reads_while_writer_pending'), 0, reader['free_reads'])
        integer(reader.get('max_revision'), 3 * (case['operations'] // 4) + 1, case['operations'])
        integer(reader.get('elapsed_ns'), 1, row['elapsed_ns'])
        for key in ('latency_p50_ns', 'latency_p95_ns', 'latency_p99_ns'):
            integer(reader.get(key), 0, reader['elapsed_ns'])
        if not reader['latency_p50_ns'] <= reader['latency_p95_ns'] <= reader['latency_p99_ns']:
            raise ValueError('inconsistent reader latency percentiles')
    return row


def summarize(rows):
    if not rows:
        raise ValueError('no measured trials')
    result = {key: {'median': statistics.median(r[key] for r in rows),
                    'min': min(r[key] for r in rows), 'max': max(r[key] for r in rows)} for key in METRICS}
    result['operations_per_second'] = {'median': statistics.median(r['operations'] * 1e9 / r['elapsed_ns'] for r in rows)}
    if rows[0]['readers']:
        result['reader_progress'] = []
        for index in range(rows[0]['readers']):
            result['reader_progress'].append(dict(index=index,
                operations_per_second_median=statistics.median(r['reader_progress'][index]['operations'] * 1e9 / r['elapsed_ns'] for r in rows),
                latency_p99_ns_median=statistics.median(r['reader_progress'][index]['latency_p99_ns'] for r in rows),
                free_reads_while_writer_pending_min=min(r['reader_progress'][index]['free_reads_while_writer_pending'] for r in rows),
                witness_reads_per_trial=4))
    # These are descriptive trial summaries, never a pooled latency distribution
    # or a confidence bound on improvement over another delegate/revision.
    return result


def machine():
    cpu = 'unavailable'
    source = Path('/proc/cpuinfo')
    if source.exists():
        for line in source.read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip(); break
    return dict(system=platform.system(), release=platform.release(), machine=platform.machine(),
                cpu_model=cpu, logical_cpus=os.cpu_count(),
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                stable_worker_attested=False)


def python_identity():
    executable = Path(sys.executable).resolve()
    return dict(path=str(executable), version=sys.version, implementation=sys.implementation.name,
                optimization=sys.flags.optimize, executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest())


def child_environment():
    environment = {k: v for k, v in os.environ.items() if k in {'PATH', 'SYSTEMROOT', 'WINDIR', 'TEMP', 'TMP'}}
    environment.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PXR_PLUGINPATH_NAME='',
                       PXR_AR_DEFAULT_SEARCH_PATH='', AR_DEFAULT_RESOLVER='')
    return environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--workloads', type=Path, default=ROOT / 'tests/benchmarks/scene-workloads.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trials', type=int, default=7)
    parser.add_argument('--timeout-ms', type=int, default=30000)
    parser.add_argument('--artifact-dir', type=Path, default=ROOT.parent / '.cache/live-benchmarks')
    args = parser.parse_args(argv)
    expected = None
    try:
        if args.replay:
            expected = json.loads(args.replay.read_text())
            args.driver = Path(expected['driver'])
            args.workloads = args.replay.resolve().parent / 'workloads.json'
            args.seed, args.trials = expected['master_seed'], expected['requested_trials']
            args.timeout_ms = expected['timeout_ms']
        if args.driver is None:
            raise ValueError('--driver or --replay is required')
        integer(args.seed, 0, (1 << 64) - 1); integer(args.trials, 2, 100)
        integer(args.timeout_ms, 1, 3600000)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='run-', dir=args.artifact_dir)).resolve()
    record_seeds(directory, args.seed, 1)  # Before native work, including failed preflight.
    report = dict(schema_version=1, boundary='sdf_storage', delegate_version=2,
                  execution_passed=False, complete=False, source_fingerprint=fingerprint(),
                  qualification=dict(state='unverified', reasons=[
                      'characterization only; numerical budgets are not frozen',
                      'no comparable earlier storage report supplied',
                      'worker stability and statistical comparison are not qualified',
                      'transport, convergence, concurrency, recovery and instrumentation matrix incomplete']),
                  machine=machine(), python_identity=python_identity(), child_environment=child_environment(),
                  measurement_contract=MEASUREMENT_CONTRACT, trials=[], summaries={})
    try:
        workload_bytes = args.workloads.read_bytes()
        definition = workloads(workload_bytes)
        driver = args.driver.resolve()
        identity = native_identity(driver, 'scene_benchmark')
        if identity.get('configuration') != 'Release' or identity.get('sanitizers'):
            raise ValueError('performance characterization requires an unsanitized Release build')
        if expected and (identity != expected['native_identity'] or
                         report['source_fingerprint'] != expected['source_fingerprint'] or
                         definition != expected['workloads'] or report['machine'] != expected['machine'] or
                         report['python_identity'] != expected['python_identity'] or
                         report['child_environment'] != expected['child_environment']):
            raise ValueError('replay candidate or workload identity differs')
        report['driver'] = str(driver)
        report['timeout_ms'] = args.timeout_ms
        report['native_identity'] = identity
        report['workloads'] = definition
        report['workloads_sha256'] = hashlib.sha256(workload_bytes).hexdigest()
        write_json(directory / 'workloads.json', definition)
        report['master_seed'] = args.seed
        report['requested_trials'] = args.trials
        write_json(directory / 'report.json', report)
        environment = report['child_environment']
        order = random.Random(args.seed)
        for trial in range(args.trials):
            cases = list(definition['cases']); order.shuffle(cases)
            for case in cases:
                case_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{case['id']}".encode()).digest()[:8], 'big')
                entry = dict(case=case['id'], trial=trial, seed=case_seed)
                report['trials'].append(entry)
                write_json(directory / 'report.json', report)  # Retain scheduled seed even if child crashes.
                log = directory / f"{trial}-{case['id']}.log"
                result = run_process([str(driver), case['kind'], str(case_seed), str(case['specs']),
                    str(case['operations']), str(case['array_elements']), str(definition['warmup_operations']), str(case['readers'])],
                    cwd=ROOT, output=log, timeout_ms=args.timeout_ms, max_output_bytes=65536, environment=environment)
                entry['process'] = asdict(result)
                if result.outcome != 'passed' or not result.cleanup_complete or result.output_truncated:
                    raise ValueError(f"native trial did not complete cleanly: {case['id']}")
                entry['measurement'] = validate_trial(json.loads(log.read_text().splitlines()[-1]), case,
                                                     case_seed, definition['warmup_operations'])
                write_json(directory / 'report.json', report)
        for case in definition['cases']:
            rows = [r['measurement'] for r in report['trials'] if r['case'] == case['id']]
            if len(rows) != args.trials:
                raise ValueError('missing repeated trial')
            report['summaries'][case['id']] = summarize(rows)
        # Identity verification here catches source, SDK and binary changes
        # during the run instead of comparing a mixed candidate silently.
        if native_identity(driver, 'scene_benchmark') != identity or fingerprint() != report['source_fingerprint']:
            raise ValueError('candidate changed during measurement')
        if (python_identity() != report['python_identity'] or child_environment() != report['child_environment'] or
                machine() != report['machine']):
            raise ValueError('execution environment changed during measurement')
        if hashlib.sha256(args.workloads.read_bytes()).hexdigest() != report['workloads_sha256']:
            raise ValueError('workload definition changed during measurement')
        report['execution_passed'] = True
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        report['error'] = str(error)
    finally:
        report['complete'] = True
        write_json(directory / 'report.json', report)
    print('Artifacts:', directory)
    print('Replay:', sys.executable, '-B tools/live/benchmark.py --replay', directory / 'report.json')
    print('Execution:', 'passed' if report['execution_passed'] else 'failed', '; qualification: unverified')
    return 0 if report['execution_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
