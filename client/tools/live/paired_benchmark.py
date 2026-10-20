#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interleaved storage revision measurements; never a release qualification."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
import statistics
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live import benchmark as bench
from tools.live.test_support import record_seeds, run_process, write_json

POLICY = dict(schema_version=1, minimum_pairs=20, family_confidence=0.95,
              improvement_ratio=0.99, regression_ratio=1.01,
              estimator='paired candidate/reference ratio median',
              interval='two-sided binomial order-statistic median interval; Bonferroni across case/metric comparisons',
              metrics=['elapsed_ns', 'latency_p99_ns', 'each_reader_latency_p99_ns'],
              assumptions='independent representative pairs; stable measurement distribution; no post-hoc trial selection')


def median_interval(values, comparisons):
    """Distribution-free median interval; ties make binomial coverage conservative."""
    if (not values or any(not math.isfinite(v) or v <= 0 for v in values) or
            type(comparisons) is not int or comparisons < 1):
        raise ValueError('positive finite ratios and comparison count required')
    ordered = sorted(values)
    n, tail, index = len(values), 0, None
    alpha = (1 - POLICY['family_confidence']) / comparisons
    for i in range(n // 2):
        tail += math.comb(n, i) / (2 ** n)
        if tail <= alpha / 2:
            index = i
        else:
            break
    result = dict(median=statistics.median(ordered), pairs=n, lower=None, upper=None,
                  family_confidence=POLICY['family_confidence'], comparisons=comparisons,
                  observation='insufficient_pairs')
    if index is not None:
        lower, upper = ordered[index], ordered[n - index - 1]
        result.update(lower=lower, upper=upper, order_index=index,
                      marginal_coverage=1 - 2 * sum(math.comb(n, j) for j in range(index + 1)) / (2 ** n),
                      observation=('improvement' if upper < POLICY['improvement_ratio'] else
                                   'regression' if lower > POLICY['regression_ratio'] else 'inconclusive'))
    return result


def schedule(definition, seed, trials):
    rng = random.Random(seed)
    roles = {}
    for case in definition['cases']:
        # Balanced first position per case; odd counts differ by at most one.
        starts = ['reference', 'candidate'] * (trials // 2)
        if trials % 2: starts.append(rng.choice(['reference', 'candidate']))
        rng.shuffle(starts)
        roles[case['id']] = starts
    result = []
    for trial in range(trials):
        cases = list(definition['cases']); rng.shuffle(cases)
        for case in cases:
            first = roles[case['id']][trial]
            result.append(dict(case=case['id'], trial=trial,
                seed=int.from_bytes(hashlib.sha256(f"{seed}:{case['id']}".encode()).digest()[:8], 'big'),
                order=[first, 'candidate' if first == 'reference' else 'reference']))
    return result


def build_signature(identity, source_root):
    """Compare recorded SDK/compiler inputs and actual normalized compile commands."""
    if identity.get('configuration') != 'Release' or identity.get('sanitizers'):
        raise ValueError('paired measurements require unsanitized Release builds')
    compile_files = [Path(d['path']) for d in identity['dependencies']
                     if Path(d['path']).name == 'compile_commands.json']
    if len(compile_files) != 1:
        raise ValueError('one recorded compile-command file required')
    compile_file = compile_files[0]
    content = compile_file.read_bytes()
    expected = next(d['sha256'] for d in identity['dependencies'] if d['path'] == str(compile_file))
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('compile commands changed')
    commands = {}
    for row in json.loads(content):
        command = row.get('arguments') or shlex.split(row['command'])
        # A Windows compile command separates path components with backslashes.
        if not any(marker in t.replace('\\', '/') for t in command
                   for marker in ('CMakeFiles/live_scene_core.dir/', 'CMakeFiles/live_scene_benchmark.dir/')):
            continue
        source = Path(row['file']).resolve().relative_to(source_root).as_posix()
        if source in commands:
            raise ValueError('duplicate measured compile source')
        commands[source] = [t.replace(str(compile_file.parent), '<BUILD>').replace(str(source_root), '<SOURCE>')
                            for t in command]
    if set(commands) != {r['source'] for r in identity['compile_commands']}:
        raise ValueError('measured compile inventory differs')
    inputs = {r['path']: r['sha256'] for r in identity['inputs']}
    return dict(configuration=identity['configuration'], compiler_id=identity['compiler_id'],
                compiler_version=identity['compiler_version'], compile_commands=commands,
                benchmark_source_sha256=inputs['tests/native/usd/scene_benchmark.cpp'],
                dependency_trees=identity['dependency_trees'],
                dependencies=[d for d in identity['dependencies'] if Path(d['path']).name != 'compile_commands.json'])


def comparisons(pairs, definition, trials):
    result = {}
    count = sum(2 + case['readers'] for case in definition['cases'])
    for case in definition['cases']:
        selected = [p for p in pairs if p['case'] == case['id']]
        if len(selected) != trials or {p['trial'] for p in selected} != set(range(trials)):
            raise ValueError('missing or duplicate pair')
        metrics = {}
        for metric in ['elapsed_ns', 'latency_p99_ns'] + [f'reader_{i}_latency_p99_ns' for i in range(case['readers'])]:
            values = []
            for pair in selected:
                rows = [pair['results'][role]['measurement'] for role in ('reference', 'candidate')]
                reference, candidate = ([row['reader_progress'][int(metric.split('_')[1])]['latency_p99_ns'] for row in rows]
                                        if metric.startswith('reader_') else [row[metric] for row in rows])
                if reference <= 0 or candidate <= 0:
                    break
                values.append(candidate / reference)
            metrics[metric] = (median_interval(values, count) if len(values) == trials else
                               dict(observation='unmeasurable', reason='zero timer metric; no samples discarded'))
        result[case['id']] = metrics
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-driver', type=Path)
    parser.add_argument('--candidate-root', type=Path, default=ROOT)
    parser.add_argument('--reference-driver', type=Path)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--workloads', type=Path, default=ROOT / 'tests/benchmarks/scene-workloads.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--timeout-ms', type=int, default=30000)
    parser.add_argument('--artifact-dir', type=Path, default=ROOT.parent / '.cache/live-paired-benchmarks')
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args(argv)
    expected = None
    try:
        if args.replay:
            if args.replay.stat().st_size > 32 * 1024 * 1024:
                raise ValueError('replay report exceeds limit')
            expected = json.loads(args.replay.read_bytes())
            args.workloads = args.replay.resolve().parent / 'workloads.json'
            args.seed, args.trials, args.timeout_ms = (expected[k] for k in ('master_seed', 'requested_trials', 'timeout_ms'))
            for role in ('candidate', 'reference'):
                setattr(args, role + '_driver', Path(expected['builds'][role]['driver']))
                setattr(args, role + '_root', Path(expected['builds'][role]['source_root']))
        if not args.candidate_driver or not args.reference_driver or not args.reference_root:
            raise ValueError('both drivers and the reference source root are required')
        bench.integer(args.seed, 0, (1 << 64) - 1)
        bench.integer(args.trials, POLICY['minimum_pairs'], 100)
        bench.integer(args.timeout_ms, 1, 3600000)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='run-', dir=args.artifact_dir)).resolve()
    record_seeds(directory, args.seed, 1)
    report = dict(schema_version=1, boundary='paired_sdf_storage', delegate_version=2,
        execution_passed=False, complete=False, source_fingerprint=bench.fingerprint(),
        master_seed=args.seed, requested_trials=args.trials, timeout_ms=args.timeout_ms,
        policy=POLICY, measurement_contract=bench.MEASUREMENT_CONTRACT,
        machine=bench.machine(), python_identity=bench.python_identity(), child_environment=bench.child_environment(),
        qualification=dict(state='unverified', reasons=['storage component only; full workload/budget matrix incomplete',
            'worker stability and independence of trials are not attested', 'no comparison with an earlier storage report',
            'compile/SDK matching is recorded; full toolchain/linker comparability is not qualified']),
        builds={}, pairs=[], comparisons={})
    try:
        content = args.workloads.read_bytes()
        definition = bench.workloads(content)
        report.update(workloads=definition, workloads_sha256=hashlib.sha256(content).hexdigest())
        # Persist exactly the bytes whose digest is recorded, including formatting.
        (directory / 'workloads.json').write_bytes(content)
        for role in ('candidate', 'reference'):
            driver = getattr(args, role + '_driver').resolve()
            root = getattr(args, role + '_root').resolve()
            identity = bench.native_identity(driver, 'scene_benchmark', source_root=root)
            report['builds'][role] = dict(driver=str(driver), source_root=str(root), native_identity=identity,
                                         signature=build_signature(identity, root))
        if report['builds']['candidate']['signature'] != report['builds']['reference']['signature']:
            raise ValueError('SDK, compiler commands or benchmark implementation differs between revisions')
        report['schedule'] = schedule(definition, args.seed, args.trials)
        if expected:
            for key in ('schema_version', 'boundary', 'source_fingerprint', 'master_seed', 'requested_trials',
                        'timeout_ms', 'policy', 'measurement_contract', 'machine', 'python_identity',
                        'child_environment', 'builds', 'workloads', 'workloads_sha256', 'schedule'):
                if report[key] != expected[key]:
                    raise ValueError('replay identity differs: ' + key)
        write_json(directory / 'report.json', report)  # Entire order/seed plan before native work.
        cases = {c['id']: c for c in definition['cases']}
        for plan in report['schedule']:
            pair = dict(plan, results={})
            report['pairs'].append(pair)
            case = cases[plan['case']]
            for role in plan['order']:
                log = directory / f"{plan['trial']}-{case['id']}-{role}.log"
                build = report['builds'][role]
                command = [build['driver'], case['kind'], str(plan['seed']), str(case['specs']),
                           str(case['operations']), str(case['array_elements']),
                           str(definition['warmup_operations']), str(case['readers'])]
                pair['results'][role] = dict(command=command)
                write_json(directory / 'report.json', report)
                outcome = run_process(command, cwd=directory, output=log, timeout_ms=args.timeout_ms,
                                      max_output_bytes=65536, environment=report['child_environment'])
                result = pair['results'][role]
                result['process'] = asdict(outcome)
                if (outcome.outcome != 'passed' or outcome.returncode != 0 or
                        not outcome.cleanup_complete or outcome.output_truncated):
                    raise ValueError('paired child did not complete cleanly: ' + log.name)
                result['measurement'] = bench.validate_trial(json.loads(log.read_text().splitlines()[-1]), case,
                                                             plan['seed'], definition['warmup_operations'])
                write_json(directory / 'report.json', report)
        report['comparisons'] = comparisons(report['pairs'], definition, args.trials)
        for build in report['builds'].values():
            if bench.native_identity(Path(build['driver']), 'scene_benchmark', source_root=Path(build['source_root'])) != build['native_identity']:
                raise ValueError('build changed during paired measurements')
        if (bench.fingerprint() != report['source_fingerprint'] or bench.machine() != report['machine'] or
                bench.python_identity() != report['python_identity'] or bench.child_environment() != report['child_environment']):
            raise ValueError('runner or environment changed during paired measurements')
        if hashlib.sha256(args.workloads.read_bytes()).hexdigest() != report['workloads_sha256']:
            raise ValueError('workload changed during paired measurements')
        report['execution_passed'] = True
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        report['error'] = str(error)
    finally:
        report['complete'] = True
        write_json(directory / 'report.json', report)
    print('Artifacts:', directory)
    print('Replay:', sys.executable, '-B tools/live/paired_benchmark.py --replay', directory / 'report.json')
    print('Execution:', 'passed' if report['execution_passed'] else 'failed', '; qualification: unverified')
    return 0 if report['execution_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
