# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.live import paired_benchmark as paired
from tools.live import test as runner
from tools.live.test_support import ProcessResult


class PairedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.definition = dict(schema_version=1, boundary='sdf_storage', qualification='characterization_only',
            warmup_operations=2, cases=[dict(id='small', kind='field_set', specs=4, operations=8, array_elements=0, readers=0)])
        self.row = dict(schema_version=1, boundary='sdf_storage', correctness='passed', seed=42,
            kind='field_set', specs=4, operations=8, array_elements=0, warmup_operations=2, readers=0, reader_progress=[],
            elapsed_ns=800, cpu_ns=700, latency_p50_ns=70, latency_p95_ns=90, latency_p99_ns=100,
            process_peak_rss_before_oracle_kib=1000, candidate_cleanup_ns=20, timer_pair_p50_ns=5)

    def test_exact_median_interval_and_family_adjustment(self):
        result = paired.median_interval(list(range(1, 21)), 1)
        self.assertEqual((result['lower'], result['upper'], result['median']), (6, 15, 10.5))
        adjusted = paired.median_interval(list(range(1, 21)), 16)
        self.assertLess(adjusted['lower'], result['lower'])
        self.assertGreater(adjusted['upper'], result['upper'])
        self.assertGreaterEqual(1 - 16 * (1 - adjusted['marginal_coverage']), .95)
        self.assertEqual(paired.median_interval([.8] * 20, 16)['observation'], 'improvement')
        self.assertEqual(paired.median_interval([1.2] * 20, 16)['observation'], 'regression')
        self.assertEqual(paired.median_interval([1.] * 20, 16)['observation'], 'inconclusive')
        self.assertEqual(paired.median_interval([.8] * 2, 16)['observation'], 'insufficient_pairs')
        for values in ([], [0], [-1], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError): paired.median_interval(values, 1)

    def test_schedule_balances_order_and_preserves_seeded_work(self):
        definition = dict(self.definition, cases=[dict(self.definition['cases'][0], id=str(i)) for i in range(8)])
        for rounds in (20, 21):
            plan = paired.schedule(definition, 42, rounds)
            self.assertEqual(plan, paired.schedule(definition, 42, rounds))
            self.assertNotEqual(plan, paired.schedule(definition, 43, rounds))
            for case in definition['cases']:
                rows = [r for r in plan if r['case'] == case['id']]
                first = sum(r['order'][0] == 'candidate' for r in rows)
                self.assertLessEqual(abs(first - (rounds - first)), 1)
                self.assertEqual({r['trial'] for r in rows}, set(range(rounds)))
                self.assertEqual(len({r['seed'] for r in rows}), 1)
                self.assertTrue(all(set(r['order']) == {'candidate', 'reference'} for r in rows))

    def fixture(self, root, *, failure=None, replay=None):
        workloads = root / 'input.json'
        workloads.write_text(json.dumps(self.definition))
        artifacts = root / ('replayed' if replay else 'artifacts')
        def child(command, **kwargs):
            report = json.loads((kwargs['cwd'] / 'report.json').read_text())
            self.assertEqual(len(report['schedule']), 20)
            self.assertEqual(report['qualification']['state'], 'unverified')
            row = dict(self.row, seed=int(command[2]))
            if failure == 'seed': row['seed'] += 1
            kwargs['output'].write_text(json.dumps(row) + '\n')
            return ProcessResult('crash' if failure == 'crash' else 'passed',
                                 1 if failure in ('crash', 'exit') else 0, 1, 100,
                                 failure == 'truncated', failure != 'cleanup')
        identity = dict(configuration='Release', sanitizers=False, artifact_sha256='fixture')
        def identify(path, boundary, **kwargs):
            self.assertEqual(boundary, 'scene_benchmark')
            self.assertIn('source_root', kwargs)
            return dict(identity, artifact_sha256='changed' if failure == 'identity' else 'fixture')
        with patch.object(paired.bench, 'native_identity', side_effect=identify), \
             patch.object(paired.bench, 'fingerprint', return_value='runner'), \
             patch.object(paired.bench, 'machine', return_value={'stable_worker_attested': False}), \
             patch.object(paired, 'build_signature', side_effect=(lambda *a: {'same': True})), \
             patch.object(paired, 'run_process', side_effect=child) as spawn, contextlib.redirect_stdout(io.StringIO()):
            args = ['--artifact-dir', str(artifacts)]
            args += (['--replay', str(replay)] if replay else ['--candidate-driver', str(root/'candidate'),
                     '--reference-driver', str(root/'reference'), '--reference-root', str(root),
                     '--workloads', str(workloads)])
            result = paired.main(args)
        path = next(artifacts.glob('run-*/report.json'))
        return result, json.loads(path.read_text()), path, spawn.call_count

    def test_paired_execution_and_exact_replay_never_grant_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, report, path, count = self.fixture(root)
            self.assertEqual((result, count), (0, 40))
            self.assertTrue(report['complete'] and report['execution_passed'])
            self.assertEqual(len(report['pairs']), 20)
            self.assertEqual(report['qualification']['state'], 'unverified')
            self.assertTrue(all(v['observation'] == 'inconclusive' for v in report['comparisons']['small'].values()))
            replay_result, replay, _, count = self.fixture(root, replay=path)
            self.assertEqual((replay_result, count), (0, 40))
            self.assertEqual(replay['schedule'], report['schedule'])
            self.assertEqual(hashlib.sha256((path.parent/'workloads.json').read_bytes()).hexdigest(), report['workloads_sha256'])

    def test_failed_pair_stops_without_retry_or_partial_comparison(self):
        for failure in ('crash', 'exit', 'cleanup', 'truncated', 'seed'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                result, report, _, count = self.fixture(Path(directory), failure=failure)
                self.assertEqual((result, count), (1, 1))
                self.assertFalse(report['execution_passed'])
                self.assertEqual(report['comparisons'], {})
                self.assertEqual(len(report['pairs']), 1)

    def test_replay_identity_or_schedule_change_rejects_before_children(self):
        for change in ('identity', 'schedule', 'policy'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, original, path, _ = self.fixture(root)
                if change == 'schedule': original['schedule'][0]['order'].reverse()
                if change == 'policy': original['policy']['regression_ratio'] = 2
                path.write_text(json.dumps(original))
                result, report, _, count = self.fixture(root, replay=path, failure=change)
                self.assertEqual((result, count), (1, 0))
                self.assertIn('replay identity', report['error'])

    def test_reader_tails_zero_metrics_and_missing_pairs_are_not_hidden(self):
        definition = dict(self.definition, cases=[dict(self.definition['cases'][0], readers=1)])
        pairs = []
        for trial in range(20):
            rows = {role: dict(measurement=dict(self.row, reader_progress=[dict(latency_p99_ns=tail)]))
                    for role, tail in [('reference', 10), ('candidate', 20)]}
            pairs.append(dict(case='small', trial=trial, results=rows))
        result = paired.comparisons(pairs, definition, 20)['small']
        self.assertEqual(result['reader_0_latency_p99_ns']['observation'], 'regression')
        self.assertEqual(result['elapsed_ns']['observation'], 'inconclusive')
        pairs[0]['results']['candidate']['measurement']['elapsed_ns'] = 0
        self.assertEqual(paired.comparisons(pairs, definition, 20)['small']['elapsed_ns']['observation'], 'unmeasurable')
        with self.assertRaises(ValueError): paired.comparisons(pairs[:-1], definition, 20)

    def test_alternate_source_root_validates_bytes_and_remains_benchmark_only(self):
        from tools.live import scene_identity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source'; source.mkdir()
            (source/'fixture').write_bytes(b'correct source')
            driver = root/'driver'; driver.write_bytes(b'fixture binary')
            commands = root/'compile_commands.json'; commands.write_bytes(b'[]')
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            manifest = dict(schema_version=1, boundary='scene_benchmark', sanitizers=False,
                artifact_sha256=digest(driver), inputs=[dict(path='fixture', sha256=digest(source/'fixture'))],
                compile_commands=[{}], dependencies=[dict(path=str(commands),sha256=digest(commands))],
                dependency_trees=[dict(path=str(source),sha256='sdk')]*3)
            Path(str(driver)+'.manifest.json').write_text(json.dumps(manifest))
            with patch.object(scene_identity, 'inputs', return_value=['fixture']), patch.object(scene_identity,'tree_hash',return_value='sdk'):
                self.assertEqual(runner.native_identity(driver,'scene_benchmark',source_root=source), manifest)
                (source/'fixture').write_bytes(b'changed source')
                with self.assertRaises(ValueError): runner.native_identity(driver,'scene_benchmark',source_root=source)
            with self.assertRaises(ValueError): runner.native_identity(driver,'runtime_foundation',source_root=source)

    def test_build_signature_normalizes_only_checkout_and_build_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            signatures = []
            for role in ('candidate', 'reference'):
                root = base/role; build = root/'build'; build.mkdir(parents=True)
                source = 'tests/native/usd/scene_benchmark.cpp'
                command = ['/usr/bin/c++', '-O3', '-DNDEBUG', '-I'+str(root/'include'),
                           '-o', str(build/'CMakeFiles/live_scene_benchmark.dir/test.o'), '-c', str(root/source)]
                file = build/'compile_commands.json'
                file.write_text(json.dumps([dict(file=str(root/source), arguments=command)]))
                identity = dict(configuration='Release', sanitizers=False, compiler_id='GNU', compiler_version='12',
                    dependencies=[dict(path=str(file),sha256=hashlib.sha256(file.read_bytes()).hexdigest())],
                    dependency_trees=[], compile_commands=[dict(source=source)], inputs=[dict(path=source,sha256='same driver')])
                signatures.append(paired.build_signature(identity,root))
            self.assertEqual(*signatures)
            identity['sanitizers'] = True
            with self.assertRaises(ValueError): paired.build_signature(identity, root)
