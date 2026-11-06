# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reducer algorithm checks; the public-plugin suite also runs a real defect."""
import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
import round_plans
import sample_history
import structural_history
from writer_history import WriterMismatch, generate, load
from writer_reduction import reduce_history, round_domain

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
        self.assertTrue(summary['verified'] and summary['deletion_minimal'])
        self.assertEqual(summary['schema_version'], 2)
        self.assertEqual(summary['retained_units'], summary['retained_edits'])
        self.assertNotIn('operation_deletion_minimal', summary)
        self.assertEqual([len(writer) for writer in best['writers']], [1, 1])
        self.assertEqual(best['writers'][0][0]['edit'], 'w0:2')
        self.assertEqual(best['client_count'], 2)

    async def test_budget_reserves_fresh_final_trial(self):
        summary, best = await self.run_reduction(budget=2)
        self.assertEqual((summary['status'], summary['attempts']), ('budget_exhausted', 2))
        self.assertTrue(summary['verified'])
        self.assertFalse(summary['deletion_minimal'])
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
                self.assertFalse(summary['deletion_minimal'])
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
                self.assertFalse(summary['verified'] or summary['deletion_minimal'])

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
        self.assertTrue(summary['verified'] and summary['deletion_minimal'], summary)
        self.assertEqual(sum(map(len, best['writers'])), 2)
        self.assertNotIn('w0:0', summary['retained_edits'])
        self.assertTrue(any('w0:0' not in edits and 'w1:0' in edits for edits in seen))


class RoundReductionTests(unittest.IsolatedAsyncioTestCase):
    """Writer, round and rotation deletion of structural and sample plans."""
    async def reduce(self, history, original, target_edit, budget=32, fails=None):
        """Reduce `original`, whose trials fail while they retain `target_edit` or, given `fails(plan)`, while
        that is true."""
        domain = round_domain(history.validate, history.origin)
        signature = dict(reason='corrupted', edit=target_edit)
        proposed = []

        async def evaluate(plan, directory):
            history.validate(plan)  # Every proposed plan can execute.
            proposed.append(plan)
            edits = {op['edit'] for writer in plan['writers'] for op in writer}
            if fails(plan) if fails else target_edit in edits:
                return dict(outcome='mismatch', signature=signature, cleanup_complete=True)
            return dict(outcome='passed', cleanup_complete=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'reduction'
            summary = await reduce_history(original, signature, root, evaluate, max_trials=budget, domain=domain)
            self.assertEqual(json.loads((root / 'original.plan.json').read_text()), original)
            best = history.validate(json.loads((root / 'best.plan.json').read_text()))
        self.assertEqual(summary['attempts'], len(proposed))
        return summary, best

    async def test_samples_reduce_to_the_one_failing_edit(self):
        summary, best = await self.reduce(sample_history, sample_history.generate(42, clients=4, rounds=12), 'w2:6')
        self.assertEqual(summary['status'], 'complete')
        self.assertTrue(summary['verified'] and summary['deletion_minimal'], summary)
        self.assertEqual((summary['retained_units'], summary['retained_edits']), (['writer:2', 'round:6'], ['w2:6']))
        self.assertEqual((best['client_count'], best['rounds'], best['rotation_after']), (1, 1, []))

    async def test_structural_reduction_keeps_the_branch_state_a_failing_edit_needs(self):
        plan = structural_history.generate(42, clients=3)
        # Writer 1's round-9 reorder of /P needs a branch under /P: its round-0 creation there. Its round-8 move from
        # /Q needs its round-6 creation under /Q, which needs the erase of its round-0 branch; deleting any one of
        # those makes the plan unexecutable. Writer 0's round-4 rename under /P needs a creation there before it, and
        # deletion keeps its round-3 one.
        cases = (('w1:9', ['w1:0', 'w1:9']), ('w1:8', ['w1:0', 'w1:5', 'w1:6', 'w1:8']), ('w0:4', ['w0:3', 'w0:4']))
        for edit, retained in cases:
            with self.subTest(edit=edit):
                summary, best = await self.reduce(structural_history, plan, edit)
                self.assertTrue(summary['verified'] and summary['deletion_minimal'], summary)
                self.assertEqual(summary['unresolved_trials'], 0)
                self.assertEqual(summary['retained_edits'], retained)

    async def test_rotation_after_a_retained_round_is_a_unit(self):
        original = sample_history.generate(42, clients=2, rounds=12)
        plan = round_plans.select(original, [0, 1], list(range(12)), [5, 11])
        domain = round_domain(sample_history.validate, sample_history.origin)
        # Deleting a rotation keeps its round; deleting a round deletes the rotation after it.
        self.assertEqual(domain.remove(plan, {'rotation:5'})['rotation_after'], [11])
        without_round = domain.remove(plan, {'round:5'})
        self.assertEqual([op['ordinal'] for op in without_round['writers'][0]], [o for o in range(12) if o != 5])
        self.assertEqual(without_round['rotation_after'], [10])  # Round 11's rotation, at its new position.
        # A failure of round 5's edit alone reduces to that round without the rotation after it.
        summary, best = await self.reduce(sample_history, plan, 'w0:5')
        self.assertTrue(summary['verified'] and summary['deletion_minimal'], summary)
        self.assertEqual(summary['retained_units'], ['writer:0', 'round:5'])
        self.assertEqual(best['rotation_after'], [])

    async def test_rotation_failure_keeps_its_rotation_its_round_and_one_writer(self):
        """A failure of the rotation after round 5 alone, as the misrouted rotation of the scene-reference self-test,
        keeps that fault group, the round it follows and the fewest writers a trial can run."""
        def rotates_after_five(plan):
            ordinals = [op['ordinal'] for op in plan['writers'][0]]
            return 5 in [ordinals[position] for position in plan['rotation_after']]
        summary, best = await self.reduce(sample_history, sample_history.generate(42, clients=2, rounds=12), None,
                                          budget=16, fails=rotates_after_five)
        self.assertEqual(summary['status'], 'complete')
        self.assertTrue(summary['verified'] and summary['deletion_minimal'], summary)
        self.assertEqual((summary['retained_units'], summary['retained_edits']),
                         (['rotation:5', 'writer:0', 'round:5'], ['w0:5']))
        self.assertEqual((best['client_count'], best['rounds'], best['rotation_after']), (1, 1, [0]))

    async def test_interrupted_reduction_is_visibly_incomplete(self):
        """A watchdog that ends the run mid-trial leaves `running` and that trial's `pending` result on disk."""
        original = sample_history.generate(42, clients=2, rounds=12)
        domain = round_domain(sample_history.validate, sample_history.origin)
        signature = dict(reason='corrupted', edit='w0:6')
        count = 0

        async def evaluate(plan, directory):
            nonlocal count
            count += 1
            if count == 3:
                raise asyncio.CancelledError
            return dict(outcome='mismatch', signature=signature, cleanup_complete=True)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'reduction'
            with self.assertRaises(asyncio.CancelledError):
                await reduce_history(original, signature, root, evaluate, domain=domain)
            summary = json.loads((root / 'reduction.json').read_text())
            self.assertEqual((summary['status'], summary['attempts'], summary['verified']), ('running', 3, False))
            self.assertFalse(summary['deletion_minimal'])
            self.assertEqual(json.loads((root / 'trial-0003' / 'result.json').read_text())['outcome'], 'pending')
            self.assertEqual(sample_history.validate(json.loads((root / 'best.plan.json').read_text()))['writers'],
                             json.loads((root / 'trial-0002' / 'input.plan.json').read_text())['writers'])


class RotationCheckTests(unittest.TestCase):
    """A backend rotation's own oracle names the rotation, never a writer, backend or cursor."""
    def failure(self, generations, resumed, reads, cursor=9):
        with self.assertRaises(WriterMismatch) as caught:
            round_plans.check_rotation(5, generations, resumed, reads, 'B', cursor)
        return caught.exception

    def test_rotation_that_resumes_every_writer_passes(self):
        reads = [dict(backend='A', cursor=1, writer='w0'), dict(backend='B', cursor=9, writer='w0'),
                 dict(backend='B', cursor=9, writer='w1')]
        self.assertIsNone(round_plans.check_rotation(5, [1, 1], [2, 3], reads, 'B', 9))

    def test_unadvanced_generation_names_the_rotation(self):
        reads = [dict(backend='B', cursor=9, writer=writer) for writer in ('w0', 'w1')]
        first, second = self.failure([1, 1], [1, 2], reads), self.failure([1, 1], [2, 1], reads)
        self.assertEqual(first.signature, dict(reason='rotation did not advance a writer read generation', rotation=5))
        # Which writer stalled is evidence, so deleting the other writer keeps the signature.
        self.assertEqual((first.evidence['writers'], second.evidence['writers']), ([0], [1]))
        self.assertEqual(second.signature, first.signature)

    def test_resume_elsewhere_names_the_rotation(self):
        reason = 'writers did not resume reading on the rotation destination at its frontier'
        misrouted = self.failure([1, 1], [2, 2], [dict(backend='A', cursor=9, writer=w) for w in ('w0', 'w1')])
        self.assertEqual(misrouted.signature, dict(reason=reason, rotation=5))
        self.assertEqual(misrouted.evidence, dict(backend='B', cursor=9, matching=0, writers=2))
        # The frontier before the rotation depends on the retained rounds; it is evidence.
        shifted = self.failure([1], [2], [dict(backend='A', cursor=4, writer='w0')], cursor=4)
        self.assertEqual(shifted.signature, misrouted.signature)
        partial = self.failure([1, 1], [2, 2], [dict(backend='B', cursor=9, writer='w0'),
                                                dict(backend='B', cursor=8, writer='w1')])
        self.assertEqual((partial.signature, partial.evidence['matching']), (misrouted.signature, 1))
