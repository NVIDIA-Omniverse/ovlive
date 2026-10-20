# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reducer algorithm checks; the public-plugin suite also runs a real defect."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
from writer_history import generate, load
from writer_reduction import reduce_history

TARGET = dict(reason='lost edit', edit='w0:2')


async def evaluate(plan, directory):
    assert load(directory / 'input.plan.json') == plan  # Persist before execution.
    assert json.loads((directory / 'result.json').read_text())['outcome'] == 'pending'
    if any(op['edit'] == TARGET['edit'] for writer in plan['writers'] for op in writer):
        return dict(outcome='mismatch', signature=TARGET, cleanup_complete=True)
    return dict(outcome='passed', cleanup_complete=True)


class WriterReductionTests(unittest.IsolatedAsyncioTestCase):
    async def run_reduction(self, runner=evaluate, budget=32):
        original = generate(42, clients=2, steps=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'reduction'
            summary = await reduce_history(original, TARGET, root, runner, max_trials=budget)
            self.assertEqual(load(root / 'original.plan.json'), original)
            self.assertEqual(summary, json.loads((root / 'reduction.json').read_text()))
            self.assertEqual(len(list(root.glob('trial-*'))), summary['attempts'])
            return summary, load(root / 'best.plan.json')

    async def test_literal_minimum_preserves_original_ids_and_population(self):
        summary, best = await self.run_reduction()
        self.assertEqual(summary['status'], 'complete')
        self.assertTrue(summary['verified'] and summary['operation_deletion_minimal'])
        self.assertEqual([len(writer) for writer in best['writers']], [1, 1])
        self.assertEqual(best['writers'][0][0]['edit'], 'w0:2')
        self.assertEqual(best['client_count'], 2)

    async def test_budget_reserves_fresh_final_trial(self):
        summary, best = await self.run_reduction(budget=2)
        self.assertEqual((summary['status'], summary['attempts']), ('budget_exhausted', 2))
        self.assertTrue(summary['verified'])
        self.assertFalse(summary['operation_deletion_minimal'])
        self.assertEqual(best, generate(42, clients=2, steps=4))

    async def test_failed_cleanup_timeout_crash_and_bad_reports_cannot_reduce(self):
        for failure in (dict(outcome='timeout', cleanup_complete=True),
                        dict(outcome='crash', cleanup_complete=True),
                        dict(outcome='mismatch', signature=TARGET, cleanup_complete=False),
                        dict(outcome='mismatch', signature='malformed', cleanup_complete=True), None):
            with self.subTest(failure=failure):
                count = 0
                async def injected(plan, directory):
                    nonlocal count
                    count += 1
                    return copy.deepcopy(failure) if count == 2 else await evaluate(plan, directory)
                summary, best = await self.run_reduction(injected, budget=3)
                self.assertTrue(summary['verified'])
                self.assertEqual(summary['unresolved_trials'], 1)
                self.assertFalse(summary['operation_deletion_minimal'])
                self.assertEqual(best, generate(42, clients=2, steps=4))

    async def test_unrelated_signature_does_not_reduce(self):
        count = 0
        async def injected(plan, directory):
            nonlocal count
            count += 1
            result = await evaluate(plan, directory)
            if count == 2:
                result['signature'] = dict(TARGET, edit='w1:2')
            return result
        summary, best = await self.run_reduction(injected, budget=3)
        self.assertTrue(summary['verified'])
        self.assertEqual(best, generate(42, clients=2, steps=4))

    async def test_not_reproduced_and_unstable_are_explicit(self):
        for failure_at, status in ((1, 'not_reproduced'), (2, 'unstable')):
            with self.subTest(status=status):
                count = 0
                async def injected(plan, directory):
                    nonlocal count
                    count += 1
                    if count == failure_at:
                        return dict(outcome='passed', cleanup_complete=True)
                    return await evaluate(plan, directory)
                summary, _ = await self.run_reduction(injected, budget=2)
                self.assertEqual(summary['status'], status)
                self.assertFalse(summary['verified'] or summary['operation_deletion_minimal'])

    async def test_successful_deletion_restarts_search_for_nonmonotonic_failure(self):
        seen = []
        async def nonmonotonic(plan, directory):
            edits = {op['edit'] for writer in plan['writers'] for op in writer}
            seen.append(edits)
            # Deleting w0:0 while w1:0 remains hides the bug. Once w1:0 is
            # removed, w0:0 can be removed too; an earlier rejected deletion
            # must not establish minimality of the new candidate.
            fails = TARGET['edit'] in edits and ('w0:0' in edits or 'w1:0' not in edits)
            return dict(outcome='mismatch' if fails else 'passed', cleanup_complete=True,
                        **({'signature': TARGET} if fails else {}))
        summary, best = await self.run_reduction(nonmonotonic, budget=64)
        self.assertTrue(summary['verified'] and summary['operation_deletion_minimal'], summary)
        self.assertEqual(sum(map(len, best['writers'])), 2)
        self.assertNotIn('w0:0', summary['retained_edits'])
        self.assertTrue(any('w0:0' not in edits and 'w1:0' in edits for edits in seen))
