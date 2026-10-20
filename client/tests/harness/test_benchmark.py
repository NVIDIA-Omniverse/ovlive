# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.live import benchmark as bench
from tools.live.test_support import ProcessResult


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.definition = dict(schema_version=1, boundary='sdf_storage', qualification='characterization_only',
            warmup_operations=2, cases=[dict(id='small', kind='field_set', specs=4, operations=8, array_elements=0, readers=0)])
        self.case = self.definition['cases'][0]
        self.row = dict(schema_version=1, boundary='sdf_storage', correctness='passed', seed=42,
            kind='field_set', specs=4, operations=8, array_elements=0, warmup_operations=2, readers=0, reader_progress=[],
            elapsed_ns=800, cpu_ns=700, latency_p50_ns=70, latency_p95_ns=90, latency_p99_ns=100,
            process_peak_rss_before_oracle_kib=1000, candidate_cleanup_ns=20, timer_pair_p50_ns=5)

    def test_workload_limits_reject_duplicates_unsafe_ids_and_excess_preparation(self):
        for change in ('duplicate', 'unsafe', 'excess', 'warmup', 'bool'):
            definition = copy.deepcopy(self.definition)
            case = definition['cases'][0]
            if change == 'duplicate': definition['cases'].append(dict(case))
            if change == 'unsafe': case['id'] = '../escape'
            if change == 'excess': case.update(kind='array_set', array_elements=65536, operations=1000000)
            if change == 'warmup': definition['warmup_operations'] = 9
            if change == 'bool': case['specs'] = True
            with self.subTest(change=change), self.assertRaises(ValueError): bench.workloads(json.dumps(definition))
        for value in (None, [], 1, 'text'):
            with self.assertRaises(ValueError): bench.workloads(json.dumps(value))
        self.assertEqual(bench.workloads(json.dumps(self.definition)), self.definition)

    def test_measurements_require_matching_work_correctness_and_finite_integer_metrics(self):
        for key, value in (('seed', 3), ('operations', 1), ('correctness', 'failed'), ('elapsed_ns', 0),
                           ('cpu_ns', float('nan')), ('latency_p95_ns', 900), ('peak_rss', None)):
            row = dict(self.row)
            if key == 'peak_rss': del row['process_peak_rss_before_oracle_kib']
            else: row[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): bench.validate_trial(row, self.case, 42, 2)
        self.assertEqual(bench.validate_trial(self.row, self.case, 42, 2), self.row)
        self.assertEqual(bench.summarize([self.row, dict(self.row, elapsed_ns=1600)])['elapsed_ns'],
                         dict(median=1200, min=800, max=1600))

    def run_fixture(self, directory, *, outcome='passed', configuration='Release', mutate=None, malformed=False):
        workload = directory / 'workloads-input.json'
        workload.write_text('[]' if malformed else json.dumps(self.definition))
        def child(command, **kwargs):
            row = dict(self.row, seed=int(command[2]))
            if mutate == 'wrong_seed': row['seed'] += 1
            if mutate == 'workload_changed': workload.write_text('{}')
            kwargs['output'].write_text(json.dumps(row) + '\n')
            return ProcessResult('passed' if outcome == 'cleanup_failed' else outcome,
                                 0 if outcome in ('passed', 'cleanup_failed') else 1, 1, 100, False, outcome != 'cleanup_failed')
        identities = dict(configuration=configuration, sanitizers=False, artifact_sha256='candidate')
        with patch.object(bench, 'native_identity', return_value=identities), patch.object(bench, 'fingerprint', return_value='source'), \
             patch.object(bench, 'machine', return_value={'fixture': True}), patch.object(bench, 'run_process', side_effect=child), \
             contextlib.redirect_stdout(io.StringIO()):
            result = bench.main(['--driver', str(directory / 'driver'), '--workloads', str(workload),
                                 '--artifact-dir', str(directory / 'artifacts'), '--trials', '2'])
        report_path = next((directory / 'artifacts').glob('run-*/report.json'))
        return result, json.loads(report_path.read_text()), report_path

    def test_success_is_unqualified_and_retains_every_trial_and_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, report, path = self.run_fixture(Path(temporary))
            self.assertEqual(result, 0)
            self.assertTrue(report['execution_passed'] and report['complete'])
            self.assertEqual(report['qualification']['state'], 'unverified')
            self.assertEqual(len(report['trials']), 2)
            self.assertEqual(len({r['seed'] for r in report['trials']}), 1)
            self.assertTrue((path.parent / 'seeds.json').exists())
            self.assertTrue((path.parent / 'workloads.json').exists())
            self.assertEqual(report['measurement_contract'], bench.MEASUREMENT_CONTRACT)

    def test_crashes_cleanup_failures_wrong_work_debug_and_malformed_input_cannot_pass(self):
        for kwargs in (dict(outcome='crash'), dict(outcome='cleanup_failed'), dict(mutate='wrong_seed'),
                       dict(configuration='Debug'), dict(malformed=True), dict(mutate='workload_changed')):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as temporary:
                result, report, _ = self.run_fixture(Path(temporary), **kwargs)
                self.assertEqual(result, 1)
                self.assertFalse(report['execution_passed'])
                self.assertTrue(report['complete'])
                self.assertIn('error', report)
                self.assertEqual(report['qualification']['state'], 'unverified')

    def test_replay_refuses_changed_candidate_before_spawning(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, original, path = self.run_fixture(root)
            with patch.object(bench, 'native_identity', return_value=original['native_identity']), \
                 patch.object(bench, 'fingerprint', return_value='different source'), \
                 patch.object(bench, 'machine', return_value=original['machine']), \
                 patch.object(bench, 'run_process') as spawn, contextlib.redirect_stdout(io.StringIO()):
                result = bench.main(['--replay', str(path), '--artifact-dir', str(root / 'replayed')])
            self.assertEqual(result, 1)
            spawn.assert_not_called()
            failed = json.loads(next((root / 'replayed').glob('run-*/report.json')).read_text())
            self.assertIn('replay candidate', failed['error'])
            self.assertFalse(failed['execution_passed'])

    def test_replay_refuses_changed_python_environment_and_machine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, original, path = self.run_fixture(root)
            for field in ('python_identity', 'child_environment', 'machine'):
                changed = copy.deepcopy(original)
                changed[field]['changed'] = True
                path.write_text(json.dumps(changed))
                with patch.object(bench, 'native_identity', return_value=original['native_identity']), \
                     patch.object(bench, 'fingerprint', return_value=original['source_fingerprint']), \
                     patch.object(bench, 'machine', return_value=original['machine']), \
                     patch.object(bench, 'run_process') as spawn, contextlib.redirect_stdout(io.StringIO()):
                    result = bench.main(['--replay', str(path), '--artifact-dir', str(root / field)])
                self.assertEqual(result, 1)
                spawn.assert_not_called()

    def test_mixed_reader_counts_witnesses_bounds_and_zero_overlap(self):
        case = dict(self.case, kind='mixed_pair', readers=2)
        definition = dict(self.definition, warmup_operations=4, cases=[case])
        self.assertEqual(bench.workloads(json.dumps(definition)), definition)
        row = dict(self.row, kind='mixed_pair', readers=2, warmup_operations=4)
        row['reader_progress'] = [dict(index=i, operations=8, witness_reads=4, free_reads=4,
            free_reads_while_writer_pending=0, max_revision=7, elapsed_ns=700,
            latency_p50_ns=50, latency_p95_ns=80, latency_p99_ns=90) for i in range(2)]
        self.assertEqual(bench.validate_trial(row, case, 42, 4), row)
        self.assertEqual(bench.summarize([row])['reader_progress'][0]['free_reads_while_writer_pending_min'], 0)
        for field, value in (('index', 7), ('witness_reads', 3), ('free_reads', 8), ('operations', 7),
                             ('max_revision', 6), ('elapsed_ns', 900), ('free_reads_while_writer_pending', 5),
                             ('latency_p95_ns', -1)):
            changed = copy.deepcopy(row); changed['reader_progress'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): bench.validate_trial(changed, case, 42, 4)
        row['reader_progress'].pop()
        with self.assertRaises(ValueError): bench.validate_trial(row, case, 42, 4)
        for change in (dict(readers=0), dict(operations=9), dict(readers=9)):
            invalid = dict(definition, cases=[dict(case, **change)])
            with self.assertRaises(ValueError): bench.workloads(json.dumps(invalid))
