#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The real probe's named defect must fail, reduce, and fail on standalone replay."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests/reference_service')]
from graph_reference import environment
from graph_reduction import canonical
from scene_graph_wire import Command as C, Field as F, Delta, decode
from tools.live.test_support import run_process, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    args = parser.parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='reduce-check-', dir=args.artifact_root.resolve()))
    print('Artifacts:', root, flush=True)
    env = environment()
    # Python child must be able to import the repository's tool module too.
    command = [sys.executable, '-B', str(Path(__file__).with_name('graph_reference.py')),
               '--probe', str(args.probe.resolve()), '--artifact-root', str(root / 'failure'), '--test-reducer-fault']
    failed = run_process(command, cwd=ROOT, output=root / 'failure.log', timeout_ms=45000,
                         max_output_bytes=1024*1024, environment=env)
    assert failed.returncode == 1 and failed.cleanup_complete and not failed.output_truncated, failed
    original = next((root / 'failure').iterdir())
    evidence = json.loads((original / 'evidence.json').read_text())
    reduction = original / 'reduction' / 'reduction.json'
    reduced = json.loads(reduction.read_text())
    assert evidence['comparison'] == 'failed' and reduced['status'] == 'complete', (evidence, reduced)
    assert reduced['verified'] and reduced['deletion_minimal'], reduced
    assert len(reduced['original']) == 14 and [c['origin'] for c in reduced['cases']] == [0, 13], reduced
    payloads = [decode((reduction.parent / 'blobs' / (c['sha256'] + '.ompf')).read_bytes()) for c in reduced['cases']]
    assert payloads == [Delta(commands=(C(1, 2, 1, 'Required', 6),)), Delta(fields=(F(2, 'reducerMarker', 17),))], payloads
    # Without the intentional defect, the same real native probe agrees with
    # the independent oracle on the entire original history.
    cases = json.loads((original / 'cases.json').read_text())
    good = run_process([str(args.probe.resolve()), *[str(original / c['file']) for c in cases]], cwd=ROOT,
                       output=root / 'correct.log', timeout_ms=10000, max_output_bytes=1024*1024, environment=env)
    assert good.outcome == 'passed' and good.cleanup_complete and not good.output_truncated, good
    actual = [json.loads(line) for line in (root / 'correct.log').read_text().splitlines()]
    assert canonical(actual) == canonical([c['expected'] for c in cases])
    replay = run_process([sys.executable, '-B', str(Path(__file__).with_name('graph_reference.py')),
                          '--probe', str(args.probe.resolve()), '--artifact-root', str(root / 'replay'),
                          '--replay-reduction', str(reduction)], cwd=ROOT, output=root / 'replay.log', timeout_ms=10000,
                         max_output_bytes=1024*1024, environment=env)
    assert replay.returncode == 1 and replay.cleanup_complete and not replay.output_truncated, replay
    replayed = json.loads((next((root / 'replay').iterdir()) / 'replay.json').read_text())
    assert replayed['reproduced'] and replayed['identity_unchanged'], replayed
    # Changed input must fail before starting the native probe.
    damaged = reduction.parent / 'blobs' / (reduced['cases'][-1]['sha256'] + '.ompf')
    original_bytes = damaged.read_bytes()
    try:
        damaged.write_bytes(original_bytes + b'changed')
        rejected = run_process([sys.executable, '-B', str(Path(__file__).with_name('graph_reference.py')),
                               '--probe', str(args.probe.resolve()), '--artifact-root', str(root / 'tampered'),
                               '--replay-reduction', str(reduction)], cwd=ROOT, output=root / 'tampered.log', timeout_ms=10000,
                              max_output_bytes=1024*1024, environment=env)
        assert rejected.returncode != 0 and rejected.cleanup_complete
        assert not list((root / 'tampered').glob('*/replay/trial-*'))
    finally:
        damaged.write_bytes(original_bytes)
    # Every recorded execution identity is required before launching a trial.
    # Keep the original recipe bytes unchanged after each negative check.
    recipe_bytes = reduction.read_bytes()
    for key in ('source_fingerprint', 'probe', 'environment', 'interpreter'):
        try:
            altered = copy.deepcopy(reduced)
            altered['context'][key] = {'stale': True}
            write_json(reduction, altered)
            artifact_root = root / ('stale-' + key)
            rejected = run_process([sys.executable, '-B', str(Path(__file__).with_name('graph_reference.py')),
                                   '--probe', str(args.probe.resolve()), '--artifact-root', str(artifact_root),
                                   '--replay-reduction', str(reduction)], cwd=ROOT, output=root / (key + '.log'),
                                  timeout_ms=10000, max_output_bytes=1024*1024, environment=env)
            assert rejected.returncode != 0 and rejected.cleanup_complete and not rejected.output_truncated
            assert not list(artifact_root.glob('*/replay/trial-*'))
            assert 'reduction source, probe, environment, interpreter or verification differs' in (root / (key + '.log')).read_text()
        finally:
            reduction.write_bytes(recipe_bytes)
    write_json(root / 'evidence.json', dict(passed=True, original_failure=str(original), replay=str(root / 'replay'),
                                          attempts=reduced['attempts'], fixture_fault='native --test-drop-marker'))
    print('Native defect failed; 14 batches reduced to one Create and one field edit; standalone replay still fails')


if __name__ == '__main__':
    main()
