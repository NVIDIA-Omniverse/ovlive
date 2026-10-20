# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reducer mutation and budget tests use immediate synthetic outcomes, no sleeps."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests/reference_service')]
from graph_reduction import Case, Trials, canonical, difference, expected, mismatch, reduce_history, validate_record
from scene_graph_wire import Command as C, Field as F, Sample as S, TimeSamples as T, Delta, decode, encode
from scene_graph_wire import Real, String, Strings, TokenArray
from tools.live.test_support import ProcessResult


def witness():
    return [Case(0, 'setup', encode(Delta(commands=(C(1, 2, 1, 'Required', 6), C(1, 3, 1, 'Noise', 6)),
                                        fields=(F(1, 'noise', 1),))))] + [
        Case(i, 'noise', encode(Delta(fields=(F(1, 'noise', i),)))) for i in range(1, 12)] + [
        Case(12, 'lost-field', encode(Delta(fields=(F(2, 'reducerMarker', 17), F(1, 'unrelated', 99)))))]


def faulty(rows):
    rows = copy.deepcopy(rows)
    for row in rows:
        for node in row['scene']['graph']:
            node['fields'].pop('reducerMarker', None)
    return rows


def runner(command, *, output, **kwargs):
    cases = [Case(i, 'candidate', Path(path).read_bytes()) for i, path in enumerate(command[1:])]
    text = '\n'.join(canonical(row) for row in faulty(expected(cases))) + '\n'
    output.write_text(text)
    return ProcessResult('passed', 0, 0, len(text), False, True)


class ReductionTests(unittest.TestCase):
    def target(self, cases):
        wanted = expected(cases)
        return mismatch(cases, wanted, faulty(wanted))['signature']

    def test_batch_and_record_reduction_has_literal_minimum_and_preserves_original(self):
        cases = witness()
        original = tuple(c.payload for c in cases)
        with tempfile.TemporaryDirectory() as temporary:
            trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=runner)
            result = reduce_history(cases, self.target(cases), trials, max_trials=80)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['verified'] and result['deletion_minimal'])
            self.assertEqual([c['origin'] for c in result['cases']], [0, 12])
            reduced = [decode((trials.root / 'blobs' / (c['sha256'] + '.ompf')).read_bytes()) for c in result['cases']]
            self.assertEqual(reduced, [Delta(commands=(C(1, 2, 1, 'Required', 6),)),
                                       Delta(fields=(F(2, 'reducerMarker', 17),))])
            self.assertEqual(tuple(c.payload for c in cases), original)
            for c in cases:
                self.assertEqual((trials.root / 'blobs' / (c.reference()['sha256'] + '.ompf')).read_bytes(), c.payload)
            self.assertEqual(len(list(trials.root.glob('trial-*'))), result['attempts'])

    def test_budget_reserves_fresh_verification_without_claiming_minimum(self):
        cases = witness()
        with tempfile.TemporaryDirectory() as temporary:
            trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=runner)
            result = reduce_history(cases, self.target(cases), trials, max_trials=2)
            self.assertEqual((result['status'], result['attempts'], result['verified'], result['deletion_minimal']),
                             ('budget_exhausted', 2, True, False))

    def test_crash_timeout_bad_output_and_cleanup_are_unresolved_not_reproductions(self):
        cases = witness()
        for fault in ('crash', 'timeout', 'truncated', 'cleanup', 'json', 'duplicate', 'nonfinite', 'missing'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                count = 0
                def injected(command, *, output, **kwargs):
                    nonlocal count
                    count += 1
                    if count in (1, 3):
                        return runner(command, output=output, **kwargs)
                    text = {'json': 'not-json', 'duplicate': '{"scene":1,"scene":2}',
                            'nonfinite': '{"value":NaN}', 'missing': ''}.get(fault, '')
                    output.write_text(text)
                    return ProcessResult('timeout' if fault == 'timeout' else 'failed' if fault == 'crash' else 'passed',
                                         1 if fault == 'crash' else 0, 0, len(text), fault == 'truncated', fault != 'cleanup')
                trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=injected)
                result = reduce_history(cases, self.target(cases), trials, max_trials=3)
                self.assertTrue(result['verified'] and result['unresolved_trials'])
                self.assertFalse(result['deletion_minimal'])
                self.assertEqual(result['cases'], result['original'])

    def test_unrelated_failure_cannot_replace_original(self):
        cases = witness()
        count = 0
        def injected(command, *, output, **kwargs):
            nonlocal count
            count += 1
            process = runner(command, output=output, **kwargs)
            if count == 2:
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                rows[1]['result'] = 'unsupported'
                output.write_text('\n'.join(canonical(row) for row in rows))
            return process
        with tempfile.TemporaryDirectory() as temporary:
            trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=injected)
            result = reduce_history(cases, self.target(cases), trials, max_trials=3)
            self.assertTrue(result['verified'])
            self.assertEqual(result['cases'], result['original'])

    def test_disappearing_failure_is_not_a_verified_reproducer(self):
        cases = witness()
        count = 0
        def injected(command, *, output, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                return runner(command, output=output, **kwargs)
            rows = expected([Case(i, '', Path(path).read_bytes()) for i, path in enumerate(command[1:])])
            output.write_text('\n'.join(canonical(row) for row in rows))
            return ProcessResult('passed', 0, 0, 0, False, True)
        with tempfile.TemporaryDirectory() as temporary:
            trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=injected)
            result = reduce_history(cases, self.target(cases), trials, max_trials=2)
            self.assertEqual(result['status'], 'unstable')
            self.assertFalse(result['verified'] or result['deletion_minimal'])

    def test_signature_keeps_identity_category_and_exact_json_types(self):
        a = dict(result='ok', scene=dict(graph=[dict(id=2, fields=dict(x=1)), dict(id=5, fields=dict(x=7))]))
        b = copy.deepcopy(a); b['scene']['graph'][1]['fields']['x'] = 9
        first = difference(a, b)
        self.assertEqual(first['path'], ('scene', 'graph', '5', 'fields', 'x'))
        a['scene']['graph'].pop(0); b['scene']['graph'].pop(0)
        self.assertEqual(difference(a, b), first)
        self.assertEqual(difference(1, True)['kind'], 'type')
        self.assertEqual(difference(0.0, -0.0)['kind'], 'value')
        self.assertNotEqual(difference(0.0, -0.0), difference(0.0, 1.0))
        self.assertNotEqual(difference(['A', 'B'], ['A']), difference(['A', 'B'], ['Z']))
        self.assertNotEqual(difference(['A', 'B'], ['A']), difference(['A', 'B'], []))
        self.assertNotEqual(difference({'missing': 17}, {}), difference({'missing': 99}, {}))
        self.assertNotEqual(difference(1, True), difference(2, True))
        self.assertEqual(difference(dict(scene=dict(graph=[dict(id=1), dict(id=2)])),
                                    dict(scene=dict(graph=[dict(id=2), dict(id=1)])))['kind'], 'order')

    def test_same_location_different_wrong_value_cannot_replace_original(self):
        self.assert_value_substitution(17, 18, 99)

    def test_same_location_different_wrong_vector_cannot_replace_original(self):
        self.assert_value_substitution(('A', 'B'), ['A'], ['Z'])
        self.assert_value_substitution(('A', 'B'), ['A'], [])

    def assert_value_substitution(self, original, first_wrong, other_wrong):
        cases = witness()
        cases[-1] = replace(cases[-1], payload=encode(Delta(fields=(F(2, 'reducerMarker', original),
                                                                  F(1, 'unrelated', 99)))))
        wanted = expected(cases)
        def corrupt(rows, value):
            rows = copy.deepcopy(rows)
            for row in rows:
                for node in row['scene']['graph']:
                    if 'reducerMarker' in node['fields']:
                        node['fields']['reducerMarker']['value'] = value
            return rows
        target = mismatch(cases, wanted, corrupt(wanted, first_wrong))['signature']
        count = 0
        def injected(command, *, output, **kwargs):
            nonlocal count
            count += 1
            rows = expected([Case(i, '', Path(p).read_bytes()) for i, p in enumerate(command[1:])])
            output.write_text('\n'.join(canonical(row) for row in corrupt(rows, other_wrong if count == 2 else first_wrong)))
            return ProcessResult('passed', 0, 0, 0, False, True)
        with tempfile.TemporaryDirectory() as temporary:
            trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=injected)
            result = reduce_history(cases, target, trials, max_trials=3)
            self.assertTrue(result['verified'])
            self.assertEqual(result['cases'], result['original'])

    def test_native_record_schema_accepts_exact_presence_families_only(self):
        values = dict(camera=String('/Camera'), selection=Strings(('/A', '')), order=TokenArray(('xformOp:transform',)),
                      focal=Real('float', (0x80000000,)), floats=Real('float', (0x7fc00123, 0), True),
                      doubles=Real('double', (2**64 - 1,), True), pose=Real('matrix4d', (0,) * 16),
                      poses=Real('matrix4d', (), True), clip=Real('vec2f', (0, 1)), pivot=Real('vec3d', (0, 0, 1)),
                      plane=Real('vec4f', (0,) * 4), planes=Real('vec4f', (0,) * 8, True))
        valid = expected([Case(0, 'presence', encode(Delta(fields=tuple(F(1, k, v) for k, v in values.items()))))])[0]
        validate_record(valid)
        for key, raw in (('pose', [0] * 15), ('planes', [0] * 6), ('focal', [2**32]), ('pivot', [0.0, 0, 0]),
                         ('camera', ['/Camera']), ('selection', '/A')):
            bad = copy.deepcopy(valid)
            bad['scene']['graph'][0]['fields'][key]['value'] = raw
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_record(bad)
        bad = copy.deepcopy(valid)
        bad['scene']['graph'][0]['fields']['pose']['type'] = 'quatd'
        with self.assertRaises(ValueError):
            validate_record(bad)

    def test_valid_bootstrap_with_malformed_later_record_is_unresolved(self):
        case = Case(0, 'one', encode(Delta(fields=(F(1, 'reducerMarker', 17),))))
        valid = expected([case])[0]
        duplicate = copy.deepcopy(valid)
        duplicate['scene']['graph'].append(copy.deepcopy(duplicate['scene']['graph'][0]))
        wrong_id = copy.deepcopy(valid); wrong_id['scene']['graph'][0]['id'] = True
        wrong_field = copy.deepcopy(valid)
        wrong_field['scene']['graph'][0]['fields']['reducerMarker']['value'] = True
        invalid = [None, [], {}, {'result': 'ok'}, dict(valid, result=7),
                   dict(valid, scene=None), dict(valid, scene=dict(source_format=7, graph=[])),
                   duplicate, wrong_id, wrong_field]
        for bad in invalid:
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as temporary:
                def injected(command, *, output, **kwargs):
                    rows = expected([Case(-1, '', Path(command[1]).read_bytes())]) + [bad]
                    output.write_text('\n'.join(canonical(row) for row in rows))
                    return ProcessResult('passed', 0, 0, 0, False, True)
                trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=injected)
                self.assertEqual(trials.evaluate([case])['outcome'], 'unresolved')

        # A complete deletion pass cannot certify minimality if its only smaller
        # candidate returned invalid data, even when the original reproduces.
        def malformed_empty(command, *, output, **kwargs):
            result = runner(command, output=output, **kwargs)
            if not decode(Path(command[-1]).read_bytes()).fields:
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                rows[-1] = None
                output.write_text('\n'.join(canonical(row) for row in rows))
            return result
        with tempfile.TemporaryDirectory() as temporary:
            trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=malformed_empty)
            result = reduce_history([case], self.target([case]), trials)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['verified'] and result['unresolved_trials'])
            self.assertFalse(result['deletion_minimal'])

    def test_fixed_point_revisits_batch_after_later_field_removal(self):
        cases = [Case(0, 'early', encode(Delta(fields=(F(1, 'noiseA', 1),)))),
                 Case(1, 'required', encode(Delta(commands=(C(1, 2, 1, 'Required', 6),),
                                                 fields=(F(1, 'noiseB', 1),)))),
                 Case(2, 'target', encode(Delta(fields=(F(2, 'reducerMarker', 17),))))]
        def injected(command, *, output, **kwargs):
            rows = expected([Case(i, '', Path(p).read_bytes()) for i, p in enumerate(command[1:])])
            for row in rows:
                root = row['scene']['graph'][0]['fields']
                if 'noiseB' not in root or 'noiseA' in root:
                    for node in row['scene']['graph']:
                        node['fields'].pop('reducerMarker', None)
            output.write_text('\n'.join(canonical(row) for row in rows))
            return ProcessResult('passed', 0, 0, 0, False, True)
        with tempfile.TemporaryDirectory() as temporary:
            trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=injected)
            result = reduce_history(cases, self.target(cases), trials, max_trials=80)
            self.assertTrue(result['verified'] and result['deletion_minimal'])
            self.assertEqual([c['origin'] for c in result['cases']], [1, 2])
            first = decode((trials.root / 'blobs' / (result['cases'][0]['sha256'] + '.ompf')).read_bytes())
            self.assertEqual(first, Delta(commands=(C(1, 2, 1, 'Required', 6),)))

    def test_invalid_history_rejects_before_spawning(self):
        cases = witness()
        invalid = [[], [cases[0], cases[0]], [replace(cases[-1], origin=True)],
                   [replace(cases[-1], payload=bytearray(cases[-1].payload))],
                   [replace(cases[-1], name=None)], cases[:-1],
                   [replace(cases[-1], payload=b'x' * (1024*1024 + 1))],
                   [replace(cases[-1], origin=i, payload=b'x' * (1024*1024)) for i in range(65)],
                   [replace(cases[-1], origin=i) for i in range(1024)]]
        with tempfile.TemporaryDirectory() as temporary:
            trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=runner)
            for history in invalid:
                with self.assertRaises(ValueError):
                    reduce_history(history, self.target(cases), trials)
            self.assertEqual(trials.count, 0)
            trials.evaluate(cases)
            with self.assertRaises(ValueError):
                reduce_history(cases, self.target(cases), trials)
            self.assertEqual(trials.count, 1)

    def test_sample_deletion_keeps_the_original_time_identity(self):
        cases = [Case(0, 'samples', encode(Delta(samples=(S(1, 1.0, 17), S(1, 2.0, 17)))))]
        def faulty_samples(rows):
            rows = copy.deepcopy(rows)
            for row in rows:
                for node in row['scene']['graph']:
                    for value in node['fields'].values():
                        if value['type'] == 'time_samples':
                            for sample in value['value']:
                                sample['value']['value'] = 18
            return rows
        wanted = expected(cases)
        target = mismatch(cases, wanted, faulty_samples(wanted))['signature']
        def injected(command, *, output, **kwargs):
            rows = expected([Case(i, '', Path(p).read_bytes()) for i, p in enumerate(command[1:])])
            output.write_text('\n'.join(canonical(row) for row in faulty_samples(rows)))
            return ProcessResult('passed', 0, 0, 0, False, True)
        with tempfile.TemporaryDirectory() as temporary:
            trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=injected)
            result = reduce_history(cases, target, trials)
            self.assertTrue(result['verified'] and result['deletion_minimal'])
            only = decode((trials.root / 'blobs' / (result['cases'][0]['sha256'] + '.ompf')).read_bytes())
            self.assertEqual(only, Delta(samples=(S(1, 1.0, 17),)))

    def test_malformed_sample_map_reports_are_unresolved(self):
        case = Case(0, 'samples', encode(Delta(fields=(F(1, 'timeSamples', T(((0.0, 17),))),))))
        for extra_bits in (0, 2**63, 0x7ff8000000000000):
            with self.subTest(bits=extra_bits), tempfile.TemporaryDirectory() as temporary:
                def injected(command, *, output, **kwargs):
                    rows = expected([Case(i, '', Path(p).read_bytes()) for i, p in enumerate(command[1:])])
                    values = rows[-1]['scene']['graph'][0]['fields']['timeSamples']['value']
                    values.append(dict(time_bits=extra_bits, value=dict(type='int32', value=99)))
                    output.write_text('\n'.join(canonical(row) for row in rows))
                    return ProcessResult('passed', 0, 0, 0, False, True)
                trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=injected)
                self.assertEqual(trials.evaluate([case])['outcome'], 'unresolved')

    def test_invalid_budget_rejects_before_spawning(self):
        with tempfile.TemporaryDirectory() as temporary:
            trials = Trials(Path(temporary) / 'reduce', ['fake'], ROOT, {}, runner=runner)
            for budget in (0, 1, 257):
                with self.assertRaises(ValueError):
                    reduce_history(witness(), self.target(witness()), trials, max_trials=budget)
            self.assertEqual(trials.count, 0)
