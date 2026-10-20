#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate real native snapshots and independent malformed consumer fixtures."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

# Establish this before repository imports: -B alone would still execute a
# stale diagnostic_snapshot/test_support cache while identity hashes its source.
_repository_cache = tempfile.TemporaryDirectory(prefix='live-schema-suite-imports-')
sys.dont_write_bytecode = True
sys.pycache_prefix = _repository_cache.name
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.live.diagnostic_snapshot import InvalidSnapshot, MAX_BYTES, SnapshotValidator, identity, load_document
from tools.live.test_support import run_process, write_json

SCENARIOS = {'options', 'executors', 'subjects', 'backpressure', 'applications',
             'jitter', 'capacity', 'durations', 'status-history', 'auth-waits', 'callback-failures'}
MAX_CORPUS_BYTES = 64 * 1024 * 1024
MAX_SNAPSHOTS = 1024


def validate_corpus(path):
    validator = SnapshotValidator(current=True)
    total = count = 0
    complete = False
    scenarios = set()
    coverage = set()
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while line := source.readline(MAX_BYTES + 257):
            digest.update(line)
            total += len(line)
            if total > MAX_CORPUS_BYTES or count > MAX_SNAPSHOTS:
                raise InvalidSnapshot('corpus budget exceeded')
            if complete or not line.endswith(b'\n'):
                raise InvalidSnapshot('corpus missing framing or data after completion')
            try:
                frame = load_document(line, max_bytes=MAX_BYTES + 256)
                if type(frame) is not dict:
                    raise InvalidSnapshot('invalid corpus frame')
                if 'complete' in frame:
                    if set(frame) != {'complete', 'count'} or frame['complete'] is not True or type(frame['count']) is not int or frame['count'] != count:
                        raise InvalidSnapshot('invalid corpus completion')
                    complete = True
                    continue
                if set(frame) != {'scenario', 'snapshot'} or frame['scenario'] not in SCENARIOS:
                    raise InvalidSnapshot('invalid corpus scenario')
                snapshot = validator.validate(frame['snapshot'])
            except (InvalidSnapshot, TypeError) as error:
                reason = str(error) if isinstance(error, InvalidSnapshot) else 'invalid corpus types'
                raise InvalidSnapshot(f'corpus frame {count + 1}: {reason}') from None
            count += 1
            scenarios.add(frame['scenario'])
            if snapshot['endpoint_filter']:
                coverage.add('filtered')
            if snapshot['recent_records']:
                coverage.add('history')
            else:
                coverage.add('empty-history')
            if snapshot['effective_configuration']['flags'] & 1:
                coverage.add('history-disabled')
            if snapshot['closed_incidents']:
                coverage.add('closed-incidents')
            if any('canonical_resource' in r for r in snapshot['statuses']):
                coverage.add('resource-opt-in')
            if snapshot['metrics']['capacity']['pending_bytes']['current_bytes_overflow']:
                coverage.add('saturated-bytes')
    required = {'filtered', 'history', 'empty-history', 'history-disabled', 'closed-incidents',
                'resource-opt-in', 'saturated-bytes'}
    if not complete or count == 0 or scenarios != SCENARIOS or not coverage >= required:
        raise InvalidSnapshot('incomplete corpus scenarios or coverage')
    return {'snapshots': count, 'bytes': total, 'sha256': digest.hexdigest(),
            'scenarios': sorted(scenarios), 'coverage': sorted(coverage)}


def main():
    driver = Path(sys.argv[1]).resolve()
    timeout_ms = int(sys.argv[2])
    max_output = int(sys.argv[3])
    directory = Path(os.environ['LIVE_TEST_ARTIFACT_DIR']) / 'diagnostics-schema'
    directory.mkdir()  # Never reuse an earlier corpus.
    dependencies = identity()
    replay = json.loads((directory.parent / 'replay.json').read_text())
    if dependencies != replay['diagnostic_schema_identity']:
        print('Diagnostic validator identity changed after preflight', file=sys.stderr)
        return 125
    write_json(directory / 'validator-identity.json', dependencies)
    corpus = directory / 'snapshots.jsonl'
    result = asdict(run_process([str(driver)], cwd=ROOT, output=directory / 'native.log',
                               timeout_ms=timeout_ms, max_output_bytes=max_output,
                               environment=dict(os.environ, LIVE_TEST_DIAGNOSTIC_CORPUS=str(corpus))))
    write_json(directory / 'native-result.json', result)
    if result['outcome'] != 'passed' or not result['cleanup_complete'] or result['output_truncated']:
        print('Native diagnostic corpus producer did not complete cleanly', file=sys.stderr)
        return 1
    try:
        summary = validate_corpus(corpus)
    except (OSError, InvalidSnapshot) as error:
        print(str(error) if isinstance(error, InvalidSnapshot) else 'Cannot read native diagnostic corpus', file=sys.stderr)
        return 1
    os.environ['LIVE_TEST_DIAGNOSTIC_CORPUS'] = str(corpus)
    tests = unittest.defaultTestLoader.discover(str(ROOT / 'tests/diagnostics'), pattern='test_*.py')
    if tests.countTestCases() == 0:
        print('No diagnostic consumer tests discovered', file=sys.stderr)
        return 1
    if not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful():
        return 1
    if identity() != dependencies:
        print('Diagnostic validator identity changed during execution', file=sys.stderr)
        return 125
    summary.update(schema_version=1, boundary='diagnostic_snapshot', profile='currentProducer', passed=True)
    write_json(directory / 'summary.json', summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
