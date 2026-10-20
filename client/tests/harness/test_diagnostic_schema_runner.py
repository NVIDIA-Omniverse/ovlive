# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.live.test import diagnostic_schema_evidence


class DiagnosticSchemaEvidenceTests(unittest.TestCase):
    def test_missing_evidence_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(OSError):
                diagnostic_schema_evidence(Path(temporary), {})

    def test_completion_binds_corpus_dependencies_and_native_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); lane = root / 'diagnostics-schema'; lane.mkdir()
            corpus = b'{"synthetic":"corpus-fixture"}\n'
            identity = {'fixture': 'dependency-hash'}
            native = {'outcome': 'passed', 'cleanup_complete': True, 'output_truncated': False}
            summary = {'schema_version': 1, 'boundary': 'diagnostic_snapshot', 'profile': 'currentProducer',
                       'passed': True, 'snapshots': 1, 'bytes': len(corpus),
                       'sha256': hashlib.sha256(corpus).hexdigest()}
            files = {'summary.json': summary, 'validator-identity.json': identity, 'native-result.json': native}
            def restore():
                for name, value in files.items():
                    (lane / name).write_text(json.dumps(value))
                (lane / 'snapshots.jsonl').write_bytes(corpus)
            restore()
            self.assertEqual(diagnostic_schema_evidence(root, identity), summary)
            for field, bad in [('passed', False), ('snapshots', 0), ('snapshots', True), ('profile', 'consumer'), ('sha256', 'wrong')]:
                restore(); (lane / 'summary.json').write_text(json.dumps(dict(summary, **{field: bad})))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    diagnostic_schema_evidence(root, identity)
            for field, bad in [('outcome', 'timed_out'), ('cleanup_complete', False), ('output_truncated', True)]:
                restore(); (lane / 'native-result.json').write_text(json.dumps(dict(native, **{field: bad})))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    diagnostic_schema_evidence(root, identity)
            restore(); (lane / 'snapshots.jsonl').write_bytes(corpus + b'changed\n')
            with self.assertRaises(ValueError):
                diagnostic_schema_evidence(root, identity)
            restore()
            with self.assertRaises(ValueError):
                diagnostic_schema_evidence(root, {'fixture': 'changed-dependency'})
