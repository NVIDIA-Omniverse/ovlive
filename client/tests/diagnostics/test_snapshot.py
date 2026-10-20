# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from decimal import Decimal
import json
import importlib.util
import marshal
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.live import diagnostic_snapshot as api
from tools.live.diagnostic_schema_suite import validate_corpus


def baseline():
    # Independent consumer envelope: no optional metric groups or configuration
    # additions are necessary to understand a schema-version-1 capture.
    return {'schema_version': 1, 'runtime_instance_id': 9007199254740993,
            'capture_time_ms': 0, 'endpoint_filter': 0,
            'provenance': {'abi_version': 65536, 'proto': 'nvidia/omniverse/live/v1alpha/live.proto',
                           'proto_sha256': 'fixture', 'source_revision': 'fixture',
                           'source_sha256': 'fixture', 'compiler': 'fixture', 'artifact_scope': 'native_runtime_component'},
            'effective_configuration': {}, 'statuses': [], 'closed_incidents': [],
            'metrics': {'scope': 'runtime'}, 'recent_records': [],
            'history': {'first_sequence': 19, 'last_sequence': 71, 'truncated': False}}


def status():
    return {'endpoint': 1, 'session': 0, 'subject_kind': 3, 'state': 8, 'subject_id': 27,
            'revision': 3, 'changed_at_ms': 0, 'result': 12, 'operation': 8,
            'grpc_status_present': 0, 'recovery_action': 5, 'progress_known': 0,
            'incident_id': 90, 'pending_write_count': 1, 'pending_bytes': 2,
            'has_unconfirmed_edits': 1, 'blocked_layers': 0, 'failed_layers': 0,
            'blocked_sessions': 0, 'failed_sessions': 0}


def record():
    return {'event_code': 1, 'event_name': 'endpoint_registered', 'severity': 2,
            'timestamp_ms': 0, 'elapsed_ms': 0, 'runtime_instance_id': 3, 'sequence': 30,
            'endpoint': 1, 'subject_id': 1, 'incident_id': 0, 'fields': {}}


class ConsumerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.consumer = api.SnapshotValidator()
        cls.current = api.SnapshotValidator(True)

    def test_versioned_envelope_and_additive_fields(self):
        value = baseline()
        value.update(future={'nested': [True, 'opaque']}, **{'$ref': 'https://invalid.invalid/never-fetch'})
        value['metrics']['future_metric'] = {'arbitrary': 'opaque'}
        with patch.object(socket, 'socket', side_effect=AssertionError('unexpected network')):
            self.consumer.validate(value)
        for key in baseline():
            candidate = baseline(); del candidate[key]
            with self.subTest(key=key), self.assertRaises(api.InvalidSnapshot):
                self.consumer.validate(candidate)
        for version in [True, 0, 2, '1']:
            value['schema_version'] = version
            with self.subTest(version=version), self.assertRaises(api.InvalidSnapshot):
                self.consumer.validate(value)

    def test_exact_integers_and_fractional_rejection(self):
        for token, expected in [('9007199254740993', 9007199254740993),
                                ('18446744073709551615', (1 << 64) - 1),
                                ('18446744073709551615.0', (1 << 64) - 1), ('1e0', 1)]:
            value = baseline()
            data = json.dumps(value).replace('9007199254740993', token).encode()
            parsed = api.load_document(data)
            self.assertEqual(parsed['runtime_instance_id'], expected)
            self.consumer.validate(parsed)
        for number in [True, -1, 1 << 64, '17', Decimal('1.000000000000000000000001')]:
            value = baseline(); value['runtime_instance_id'] = number
            with self.subTest(number=number), self.assertRaises(api.InvalidSnapshot):
                self.consumer.validate(value)

    def test_flag_types_and_grpc_presence(self):
        value = baseline(); value['statuses'] = [status()]
        self.consumer.validate(value)
        value['statuses'][0].update(grpc_status_present=1, grpc_status=0)
        self.consumer.validate(value)
        for change in [dict(grpc_status_present=0), dict(grpc_status=True), dict(grpc_status_present=True),
                       dict(progress_known=True), dict(has_unconfirmed_edits=False)]:
            candidate = copy.deepcopy(value); candidate['statuses'][0].update(change)
            with self.subTest(change=change), self.assertRaises(api.InvalidSnapshot):
                self.consumer.validate(candidate)
        del value['statuses'][0]['grpc_status']
        with self.assertRaises(api.InvalidSnapshot):
            self.consumer.validate(value)
        value = baseline(); value['history']['truncated'] = 0
        with self.assertRaises(api.InvalidSnapshot):
            self.consumer.validate(value)

    def test_records_unknown_ids_and_optional_fields(self):
        value = baseline(); value['recent_records'] = [record()]
        value['recent_records'][0]['fields'] = {'result': 21, 'new_field': {'opaque': 'extension'}}
        self.consumer.validate(value)
        value['recent_records'][0].update(event_code=4000, event_name='future', severity=99)
        self.consumer.validate(value)
        value['recent_records'][0]['fields']['result'] = True
        with self.assertRaises(api.InvalidSnapshot):
            self.consumer.validate(value)
        value['recent_records'] = [record()]
        value['recent_records'][0]['event_name'] = 'wrong-name'
        with self.assertRaisesRegex(api.InvalidSnapshot, 'code/name'):
            self.consumer.validate(value)

    def test_histogram_shape_sparse_uniqueness_and_saturation(self):
        value = baseline()
        row = {'result': 22, 'count': (1 << 64) - 1, 'sum_ms': (1 << 64) - 1,
               'counts': [(1 << 64) - 1] * 9}
        hist = {'bounds': [1, 5, 10, 50, 100, 500, 1000, 5000], 'entries': [row]}
        value['metrics']['save_duration_ms_by_result'] = hist
        self.consumer.validate(value)  # Independent saturation defeats arithmetic identities.
        hist['entries'].append(dict(row, result=9000))
        self.consumer.validate(value)  # Future dimensions remain opaque.
        hist['entries'].append(dict(row, count=2))
        with self.assertRaisesRegex(api.InvalidSnapshot, 'x-unique-dimensions'):
            self.consumer.validate(value)
        hist['entries'] = [row]
        for change in [dict(count=0), dict(counts=[0] * 8), dict(counts=[False] * 9)]:
            candidate = copy.deepcopy(value); candidate['metrics']['save_duration_ms_by_result']['entries'][0].update(change)
            with self.subTest(change=change), self.assertRaises(api.InvalidSnapshot):
                self.consumer.validate(candidate)
        hist['entries'] = []
        self.consumer.validate(value)
        hist['bounds'][1] = 1
        with self.assertRaises(api.InvalidSnapshot):
            self.consumer.validate(value)

    def test_no_cross_capture_or_global_history_assertions(self):
        value = baseline(); value['statuses'] = [status()]; value['closed_incidents'] = [status()]
        value['recent_records'] = [record()]
        value['recent_records'][0].update(subject_id=9999, sequence=40)
        self.consumer.validate(value)

    def test_input_syntax_and_resource_budgets(self):
        for data in [b'{"secret":1,"secret":2}', b'{"x":{"a":1,"a":2}}', b'NaN', b'Infinity',
                     b'-Infinity', b'{} {}', b'\xff', b'[' * 65 + b']' * 65,
                     b'1' * 129, b'1e' + b'1' * 129]:
            with self.subTest(data=data[:20]), self.assertRaises(api.InvalidSnapshot):
                api.load_document(data)
        with self.assertRaisesRegex(api.InvalidSnapshot, 'byte budget'):
            api.load_document(b'{}', max_bytes=1)
        with patch.object(api, 'MAX_NODES', 3), self.assertRaisesRegex(api.InvalidSnapshot, 'node budget'):
            api.load_document(b'[0,0,0]')
        self.assertEqual(api.load_document(b'{"a":"[\\\"{]"}'), {'a': '["{]'})

    def test_error_messages_never_echo_values_or_input_paths(self):
        value = baseline(); value['statuses'] = [{'sensitive-key-canary': 'sensitive-value-canary'}]
        with self.assertRaises(api.InvalidSnapshot) as captured:
            self.consumer.validate(value)
        self.assertNotIn('sensitive', str(captured.exception))
        with self.assertRaises(api.InvalidSnapshot) as captured:
            api.load_document(b'{"sensitive-canary":0,"sensitive-canary":1}')
        self.assertNotIn('sensitive', str(captured.exception))


class NativeCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ['LIVE_TEST_DIAGNOSTIC_CORPUS'])
        cls.lines = path.read_bytes().splitlines(keepends=True)
        cls.native = api.load_document(cls.lines[0])['snapshot']
        cls.consumer = api.SnapshotValidator()
        cls.current = api.SnapshotValidator(True)

    def test_current_profile_rejects_each_missing_metric_and_option(self):
        self.current.validate(self.native)
        for group in ['metrics', 'effective_configuration']:
            for key in self.native[group]:
                value = copy.deepcopy(self.native); del value[group][key]
                with self.subTest(group=group, key=key), self.assertRaises(api.InvalidSnapshot):
                    self.current.validate(value)
                if key != 'scope':
                    self.consumer.validate(value)

    def test_current_profile_requires_dense_kind_and_state_rows(self):
        for group in ['subjects', 'readiness', 'backpressure', 'applications', 'operation_auth_waits']:
            value = copy.deepcopy(self.native); value['metrics'][group].pop()
            with self.subTest(group=group), self.assertRaises(api.InvalidSnapshot):
                self.current.validate(value)
        value = copy.deepcopy(self.native); value['metrics']['subjects'][0]['states'].pop()
        with self.assertRaises(api.InvalidSnapshot):
            self.current.validate(value)

    def test_corpus_rejects_empty_missing_footer_and_malformed_final_frame(self):
        candidates = [b'', self.lines[0], self.lines[0] + b'bad\n',
                      b'{"complete":true,"count":0}\n',
                      self.lines[0] + b'{"complete":true,"count":2}\n',
                      self.lines[0] + b'{"complete":true,"count":1}\n' + self.lines[0]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'corpus.jsonl'
            with self.assertRaises(OSError):
                validate_corpus(path)
            for index, data in enumerate(candidates):
                path.write_bytes(data)
                with self.subTest(candidate=index), self.assertRaises(api.InvalidSnapshot):
                    validate_corpus(path)


class ValidatorIdentityTests(unittest.TestCase):
    def test_lane_ignores_existing_repository_helper_bytecode(self):
        source = Path(api.__file__)
        with tempfile.TemporaryDirectory() as temporary:
            previous = sys.pycache_prefix
            try:
                sys.pycache_prefix = temporary
                cached = Path(importlib.util.cache_from_source(str(source)))
            finally:
                sys.pycache_prefix = previous
            cached.parent.mkdir(parents=True)
            stat = source.stat()
            code = compile('raise RuntimeError("STALE-REPOSITORY-BYTECODE")', str(source), 'exec')
            cached.write_bytes(importlib.util.MAGIC_NUMBER +
                               struct.pack('<III', 0, int(stat.st_mtime) & 0xffffffff, stat.st_size) +
                               marshal.dumps(code))
            environment = dict(os.environ, PYTHONPYCACHEPREFIX=temporary)
            probe = subprocess.run([sys.executable, '-B', '-c', 'import tools.live.diagnostic_snapshot'],
                                   env=environment, capture_output=True, timeout=10)
            self.assertNotEqual(probe.returncode, 0)
            self.assertIn(b'STALE-REPOSITORY-BYTECODE', probe.stderr)
            checked = subprocess.run([sys.executable, '-B', '-c',
                                      'import runpy,sys; runpy.run_path(sys.argv[1], run_name="cache_probe"); '
                                      'print("repository-source-loaded")',
                                      str(api.ROOT / 'tools/live/diagnostic_schema_suite.py')],
                                     env=environment, capture_output=True, timeout=10)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(checked.stdout.strip(), b'repository-source-loaded')

    def test_dependency_version_and_module_origin_are_bound(self):
        import jsonschema
        identity = api.identity()
        self.assertEqual(len(identity['interpreter_sha256']), 64)
        with patch.object(jsonschema, '__file__', '/untrusted/validator.py'), self.assertRaises(RuntimeError):
            api.identity()
        with patch.object(api.importlib.metadata, 'distribution') as distribution:
            distribution.return_value.version = 'unapproved'
            with self.assertRaises(RuntimeError):
                api.identity()

    def test_existing_timestamp_valid_bytecode_is_not_executed(self):
        import jsonschema
        source = Path(jsonschema.__file__)
        with tempfile.TemporaryDirectory() as temporary:
            previous = sys.pycache_prefix
            try:
                sys.pycache_prefix = temporary
                cached = Path(importlib.util.cache_from_source(str(source)))
            finally:
                sys.pycache_prefix = previous
            cached.parent.mkdir(parents=True)
            stat = source.stat()
            code = compile('raise RuntimeError("STALE-VALIDATOR-BYTECODE")', str(source), 'exec')
            cached.write_bytes(importlib.util.MAGIC_NUMBER +
                               struct.pack('<III', 0, int(stat.st_mtime) & 0xffffffff, stat.st_size) +
                               marshal.dumps(code))
            environment = dict(os.environ, PYTHONPYCACHEPREFIX=temporary)
            # First prove this is a usable, timestamp-valid poisoned cache.
            probe = subprocess.run([sys.executable, '-B', '-c', 'import jsonschema'],
                                   env=environment, capture_output=True, timeout=10)
            self.assertNotEqual(probe.returncode, 0)
            self.assertIn(b'STALE-VALIDATOR-BYTECODE', probe.stderr)
            checked = subprocess.run([sys.executable, '-B', str(api.ROOT / 'tools/live/diagnostic_snapshot.py'),
                                      '--identity'], env=environment, capture_output=True, timeout=10)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(json.loads(checked.stdout)['bytecode_policy'],
                             'fresh-private-cache-prefix-with-writes-disabled')
