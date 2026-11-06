# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root-field rotation plans, their reduction units and the signatures of their fault-schedule oracles; the
plugin-reference suite runs each with real clients and a named defect."""
import copy
import itertools
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
import round_plans
import rotation_history
import writer_history
from writer_history import WriterMismatch
from writer_reduction import reduce_history, round_domain

DOMAIN = round_domain(rotation_history.validate, rotation_history.origin)
RESUME = 'writers did not resume reading on the rotation destination at its frontier'


def commit(op, value=None):
    return dict(fields={op['field']: op['value'] if value is None else value})


class RotationPlanTests(unittest.TestCase):
    def test_queued_plan_is_three_phases_of_seeded_root_field_edits(self):
        plan = rotation_history.generate_queued(42)
        seeded = writer_history.generate(42, 4, 12)
        self.assertEqual(plan['writers'], seeded['writers'])
        self.assertEqual(plan['release_orders'], [seeded['release_order']] * 12)
        self.assertEqual(plan['rotation_after'], [3, 7])
        self.assertEqual(rotation_history.phases(plan), [(0, 4), (4, 8), (8, 12)])
        self.assertEqual([rotation_history.rotation_ordinal(plan, i) for i in range(2)], [3, 7])
        self.assertIs(rotation_history.validate(plan), plan)

    def test_uncertain_plan_has_no_rotation_unit(self):
        plan = rotation_history.generate_uncertain(42)
        self.assertEqual((plan['client_count'], plan['rounds'], plan['rotation_after']), (4, 4, []))
        self.assertEqual(rotation_history.phases(plan), [(0, 4)])
        self.assertEqual(DOMAIN.units(plan), [f'writer:{w}' for w in range(4)] + [f'round:{o}' for o in range(4)])

    def test_selection_keeps_seeded_edits_and_names_rotations_by_seeded_round(self):
        original = rotation_history.generate_queued(42)
        plan = DOMAIN.remove(original, {'rotation:3', 'writer:1', 'writer:2', 'writer:3', 'round:8', 'round:9',
                                        'round:10', 'round:11'})
        self.assertEqual(plan, round_plans.select(original, [0], list(range(8)), [7]))
        self.assertEqual(rotation_history.phases(plan), [(0, 8), (8, 8)])  # The rotation's phase is empty.
        self.assertEqual(rotation_history.rotation_ordinal(plan, 0), 7)
        self.assertEqual(plan['release_orders'], [[0]] * 8)
        self.assertIs(rotation_history.validate(plan), plan)
        # Deleting a round deletes the rotation after it.
        self.assertEqual(DOMAIN.remove(original, {'round:7'})['rotation_after'], [3])

    def test_plans_reject_changed_edits_and_foreign_defects(self):
        queued, uncertain = rotation_history.generate_queued(42), rotation_history.generate_uncertain(42)
        changed = copy.deepcopy(queued)
        changed['writers'][0][1]['value'] += 1
        for plan in (changed, dict(queued, scenario='public-plugin.root-fields.v1'),
                     dict(queued, test_service_fault='apply-cut-w0:0'),
                     dict(uncertain, test_service_fault='misroute-rotation-7'),
                     dict(uncertain, test_service_fault='corrupt-owned-w0:6'),
                     dict(uncertain, rotation_after=[1]), None):
            with self.subTest(plan=None if plan is None else plan.get('scenario')), self.assertRaises(ValueError):
                rotation_history.validate(plan)
        self.assertTrue(rotation_history.validate(dict(queued, test_service_fault='misroute-rotation-7')))
        self.assertTrue(rotation_history.validate(dict(queued, test_service_fault='corrupt-owned-w0:6')))
        self.assertTrue(rotation_history.validate(dict(uncertain, test_service_fault='lose-cut-w0:0')))


class RotationReductionTests(unittest.IsolatedAsyncioTestCase):
    async def reduce(self, original, fails):
        signature = dict(reason='failed', edit=None)

        async def evaluate(plan, directory):
            rotation_history.validate(plan)  # Every proposed plan can execute.
            outcome = 'mismatch' if fails(plan) else 'passed'
            return dict(outcome=outcome, cleanup_complete=True, **({'signature': signature} if fails(plan) else {}))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'reduction'
            summary = await reduce_history(original, signature, root, evaluate, domain=DOMAIN)
            best = rotation_history.validate(json.loads((root / 'best.plan.json').read_text()))
        self.assertEqual(summary['status'], 'complete', summary)
        self.assertTrue(summary['verified'] and summary['deletion_minimal'], summary)
        return summary, best

    async def test_misrouted_rotation_keeps_its_rotation_its_round_and_one_writer(self):
        """As the plugin-reference self-test: the rotation after round 7 fails while the plan retains it."""
        def rotates_after_seven(plan):
            return 7 in [rotation_history.rotation_ordinal(plan, i) for i in range(len(plan['rotation_after']))]
        summary, best = await self.reduce(rotation_history.generate_queued(42), rotates_after_seven)
        self.assertEqual((summary['retained_units'], summary['retained_edits']),
                         (['rotation:7', 'writer:0', 'round:7'], ['w0:7']))
        self.assertEqual(rotation_history.phases(best), [(0, 1), (1, 1)])

    async def test_corrupted_owned_edit_keeps_that_edit_alone(self):
        """As the coalesced plugin-reference self-test: writer 0's round-6 edit fails while the plan retains it,
        whichever rotations and so phases the selection keeps."""
        def retains_w0_6(plan):
            return any(op['edit'] == 'w0:6' for writer in plan['writers'] for op in writer)
        for seed in (42, 7, 0):
            with self.subTest(seed=seed):
                summary, best = await self.reduce(rotation_history.generate_queued(seed), retains_w0_6)
                self.assertEqual((summary['retained_units'], summary['retained_edits']),
                                 (['writer:0', 'round:6'], ['w0:6']))
                self.assertEqual(rotation_history.phases(best), [(0, 1)])

    async def test_cut_write_failure_keeps_that_write_alone(self):
        """As the plugin-reference self-tests: writer 0's round-0 write fails while it is the cut write."""
        def cuts_w0_0(plan):
            return plan['writers'][0][0]['edit'] == 'w0:0'
        summary, best = await self.reduce(rotation_history.generate_uncertain(42), cuts_w0_0)
        self.assertEqual((summary['retained_units'], summary['retained_edits']), (['writer:0', 'round:0'], ['w0:0']))
        self.assertEqual((best['client_count'], best['rounds']), (1, 1))

    async def test_leaked_cut_write_keeps_that_write_and_one_queued_behind_it(self):
        """As the plugin-reference leak self-test: writer 0's queued writes follow its released round-0 write."""
        def leaks(plan):
            # The signature names the first queued edit, so only a plan whose first one is w0:1 reproduces it.
            return [op['edit'] for op in plan['writers'][0][:2]] == ['w0:0', 'w0:1']
        summary, best = await self.reduce(rotation_history.generate_uncertain(42), leaks)
        self.assertEqual((summary['retained_units'], summary['retained_edits']),
                         (['writer:0', 'round:0', 'round:1'], ['w0:0', 'w0:1']))


class CutOracleTests(unittest.TestCase):
    """The cut's ledger oracle names the first discrepant edit in plan order, never the commit order."""
    plan = rotation_history.generate_uncertain(42)

    def first(self, plan=None):
        return [writer[0] for writer in (plan or self.plan)['writers']]

    def failure(self, boundary, commits, plan=None):
        with self.assertRaises(WriterMismatch) as caught:
            rotation_history.check_cut(plan or self.plan, boundary, commits)
        return caught.exception

    def test_correct_ledgers_pass_in_any_commit_order(self):
        self.assertIsNone(rotation_history.check_cut(self.plan, 'before', []))
        for order in itertools.permutations(self.first()):
            rotation_history.check_cut(self.plan, 'after_commit', [commit(op) for op in order])

    def test_write_cut_before_execution_names_the_first_committed_edit(self):
        w0, w1, w2, _ = self.first()
        signatures = {json.dumps(self.failure('before', [commit(op) for op in order]).signature)
                      for order in itertools.permutations((w2, w1, w0))}
        self.assertEqual(signatures, {json.dumps(dict(reason='write cut before execution committed', edit='w0:0'))})
        alone = self.failure('before', [commit(w2)])
        self.assertEqual((alone.signature['edit'], alone.evidence['committed']), ('w2:0', ['w2:0']))

    def test_missing_cut_write_names_the_first_missing_edit_and_survives_deletion(self):
        w0, w1, w2, w3 = self.first()
        missing = self.failure('after_commit', [commit(w3), commit(w0)])
        reason = 'write cut after its commit missing from the ledger'
        self.assertEqual(missing.signature, dict(reason=reason, edit='w1:0'))
        self.assertEqual(missing.evidence, dict(missing=['w1:0', 'w2:0']))
        # Deleting another writer and a later round keeps the signature; the evidence shrinks.
        selected = DOMAIN.remove(self.plan, {'writer:2', 'round:3'})
        smaller = self.failure('after_commit', [commit(w0), commit(w3)], selected)
        self.assertEqual((smaller.signature, smaller.evidence), (missing.signature, dict(missing=['w1:0'])))
        twice = self.failure('after_commit', [commit(w2), commit(w0), commit(w1), commit(w3), commit(w2)])
        self.assertEqual(twice.signature, dict(reason='write cut after its commit committed more than once',
                                               edit='w2:0'))

    def test_queued_write_names_the_first_queued_edit_at_either_boundary(self):
        later, earlier = self.plan['writers'][0][2], self.plan['writers'][3][1]
        for boundary in ('before', 'after_commit'):
            for order in ((later, earlier), (earlier, later)):
                cut = [] if boundary == 'before' else [commit(op) for op in self.first()]
                error = self.failure(boundary, cut + [commit(op) for op in order])
                self.assertEqual(error.signature, dict(reason='queued write behind a cut write reached the service',
                                                       edit='w3:1'))

    def test_unexplained_commit_names_its_fields(self):
        w0 = self.first()[0]
        for commits in ([commit(w0, value=-1)], [dict(fields={w0['field']: w0['value'], 'shared_9': 1})]):
            error = self.failure('after_commit', commits)
            self.assertEqual(error.signature['reason'], 'unexplained commit across a write cut')
            self.assertEqual(error.evidence, dict(commit=commits[0]['fields'], unexplained=[commits[0]['fields']]))
        self.assertEqual(self.failure('before', [commit(w0, value=-1)]).signature,
                         dict(reason='unexplained commit across a write cut', fields=[w0['field']]))

    def test_several_unexplained_commits_are_named_independently_of_their_order(self):
        """A defect that corrupts every cut write commits them in OS-scheduled order; the signature is the same."""
        w0, w1, _, _ = self.first()
        corrupted = [commit(w1, value=-1), commit(w0, value=-2)]
        for boundary in ('before', 'after_commit'):
            errors = [self.failure(boundary, order) for order in (corrupted, corrupted[::-1])]
            self.assertEqual([error.signature for error in errors],
                             [dict(reason='unexplained commit across a write cut', fields=['owned_0_0'])] * 2)
            # The commit order is evidence.
            self.assertEqual([error.evidence['unexplained'] for error in errors],
                             [[c['fields'] for c in corrupted], [c['fields'] for c in corrupted[::-1]]])


class CoalescedPhaseSignatureTests(unittest.TestCase):
    """The coalesced queued-rotation accounting of the phase holding writer 0's corrupted round-6 edit names its
    field whatever runs the writers' windows drain, their interleaving and the selection a reduction proposes."""
    original = rotation_history.generate_queued(42)

    @staticmethod
    def ledger(plan, end, seed, corrupt):
        """Coalesced commits of every edit before round position `end`: each writer's edits in seeded contiguous
        runs, the writers interleaved by a seeded shuffle; with `corrupt`, owned_0_6 commits -1 in its run."""
        rng = random.Random(seed)
        queues = []
        for writer in plan['writers']:
            ops, runs, start = writer[:end], [], 0
            while start < len(ops):
                stop = rng.randint(start + 1, len(ops))
                runs.append(ops[start:stop])
                start = stop
            queues.append(runs)
        commits, sequence = [], 1
        while any(queues):
            run = rng.choice([queue for queue in queues if queue]).pop(0)
            fields = {}
            for op in run:
                fields[op['field']] = op['value']
            if corrupt and 'owned_0_6' in fields:
                fields['owned_0_6'] = -1
            sequence += 1
            commits.append(dict(sequence=sequence, writer=f"wire-{run[0]['writer']}", fields=fields))
        return commits

    def account(self, plan, seed, corrupt=True):
        # The phase that holds round 6 ends at its first phase bound after round 6's position.
        position = next(i for i, op in enumerate(plan['writers'][0]) if op['ordinal'] == 6)
        end = next(end for first, end in rotation_history.phases(plan) if first <= position < end)
        partial = dict(plan, writers=[writer[:end] for writer in plan['writers']])
        admissions = [dict(edit=op['edit']) for writer in partial['writers'] for op in writer]
        commits = self.ledger(plan, end, seed, corrupt)
        if not corrupt:
            return writer_history.account(partial, admissions, commits, coalesced=True)
        with self.assertRaises(WriterMismatch) as caught:
            writer_history.account(partial, admissions, commits, coalesced=True)
        return caught.exception

    def test_corrupted_owned_edit_names_its_field_across_run_shapes_and_selections(self):
        selections = [set(), {'rotation:3'}, {'rotation:3', 'rotation:7'}, {'writer:1', 'writer:3'},
                      {'round:0', 'round:1', 'round:2', 'round:3'}, {'writer:2', 'round:5', 'round:7', 'rotation:7'}]
        for removed in selections:
            plan = DOMAIN.remove(self.original, removed) if removed else self.original
            runs = set()
            for seed in range(8):
                with self.subTest(removed=sorted(removed), seed=seed):
                    self.account(plan, seed, corrupt=False)  # The same ledger without the defect accounts.
                    error = self.account(plan, seed)
                    self.assertEqual(error.signature, dict(reason='unexplained committed field', field='owned_0_6'))
                    runs.add(json.dumps(sorted(error.evidence['commits'][0]['fields'])))
            # The corrupted edit's run differs between shapes; it is evidence only.
            self.assertGreater(len(runs), 1, runs)


class ResumeCheckTests(unittest.TestCase):
    def test_resume_check_names_the_rotation_and_keeps_the_rest_as_evidence(self):
        reads = [dict(backend='B', cursor=9, writer=f'w{index}') for index in range(3)]
        self.assertIsNone(round_plans.check_resume(7, reads, 3, 'B', 9))
        with self.assertRaises(WriterMismatch) as misrouted:
            round_plans.check_resume(7, reads, 3, 'A', 9)
        self.assertEqual(misrouted.exception.signature, dict(reason=RESUME, rotation=7))
        self.assertEqual(misrouted.exception.evidence, dict(backend='A', cursor=9, matching=0, writers=3))
        # A selection with fewer writers and an earlier frontier keeps the signature.
        with self.assertRaises(WriterMismatch) as smaller:
            round_plans.check_resume(7, [dict(backend='B', cursor=2, writer='w0')], 1, 'A', 2)
        self.assertEqual(smaller.exception.signature, misrouted.exception.signature)

    def test_a_second_read_of_one_writer_passes(self):
        reads = [dict(backend='B', cursor=9, writer=f'w{index}') for index in range(2)]
        # A second reconnect's read is left to the case's plain checks.
        self.assertIsNone(round_plans.check_resume(7, reads + reads[:1], 2, 'B', 9))

    def test_a_second_read_of_one_writer_never_covers_another_writers_resume(self):
        """w0 resumes correctly once or twice, as a second reconnect would make it, and w1 at the wrong frontier or
        backend: writers are counted, not reads, so the rotation signature holds whether or not the extra read
        happened, in every read order."""
        w0 = dict(backend='B', cursor=9, writer='w0', attempt=40)
        for wrong in (dict(w0, writer='w1', attempt=41, cursor=8), dict(w0, writer='w1', attempt=41, backend='A')):
            for reads in ([w0, wrong], [w0, dict(w0, attempt=50), wrong]):
                for order in itertools.permutations(reads):
                    attempts = [read['attempt'] for read in order]
                    with self.subTest(wrong=wrong, order=attempts), self.assertRaises(WriterMismatch) as caught:
                        round_plans.check_resume(7, list(order), 2, 'B', 9)
                    self.assertEqual(caught.exception.signature, dict(reason=RESUME, rotation=7))
                    self.assertEqual(caught.exception.evidence, dict(backend='B', cursor=9, matching=1, writers=2))


if __name__ == '__main__':
    unittest.main()
