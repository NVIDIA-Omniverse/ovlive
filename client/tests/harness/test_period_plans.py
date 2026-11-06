# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Period plans of the long-session plugin cases, their truncation and reduction, and the backend-rotation and
stream-age expiry oracles; the plugin-reference suite reduces real defects."""
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
import period_plans
import writer_history
from writer_history import WriterMismatch
from writer_reduction import period_domain, reduce_history


def stream_age(periods=8, clients=4, fault=None):
    plan = period_plans.generate(42, clients, periods, period_plans.STREAM_AGE)
    return dict(plan, test_service_fault=fault) if fault else plan


def long_rotations(periods=8, clients=4, fault=None):
    plan = period_plans.generate(42, clients, periods)
    return dict(plan, test_service_fault=fault) if fault else plan


def writers(plan):
    return [writer[0]['writer'] for writer in plan['writers']]


class PeriodPlanTests(unittest.TestCase):
    def test_generated_plan_is_one_seeded_edit_per_writer_per_period(self):
        plan = period_plans.generate(42, 4, 12)
        self.assertEqual(plan['scenario'], period_plans.LONG_ROTATIONS)
        self.assertEqual(period_plans.periods(plan), list(range(1, 13)))
        self.assertEqual([op['field'] for op in plan['writers'][1][::2]],
                         ['owned_1_0', 'owned_1_1', 'owned_1_2', 'owned_1_3', 'owned_1_0', 'owned_1_1'])
        self.assertTrue(all(op['field'].startswith('shared_') for op in plan['writers'][1][1::2]))
        values = [op['value'] for writer in plan['writers'] for op in writer]
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(period_plans.validate(plan), plan)
        with self.assertRaises(ValueError):
            period_plans.generate(42, 4, 12, 'public-plugin.other.v1')

    def test_selection_keeps_seeded_edits_whole_cycles_and_a_cut_last_cycle(self):
        original = stream_age()
        plan = period_plans.select(original, [1, 3], [2, 3, 4])
        self.assertEqual(period_plans.periods(plan), [3, 4, 5])
        self.assertEqual(writers(plan), [1, 3])
        self.assertEqual(plan['writers'][1][2], original['writers'][3][4])
        self.assertEqual((plan['client_count'], plan['origin']), (2, dict(client_count=4, periods=8)))
        self.assertEqual(period_plans.origin(plan), original)
        for ordinals in ([1, 2, 3], [2, 4, 5], [3], [2, 3, 3], []):
            with self.subTest(ordinals=ordinals), self.assertRaises(ValueError):
                period_plans.select(original, [0], ordinals)
        for chosen in ([], [1, 0], [4]):
            with self.subTest(writers=chosen), self.assertRaises(ValueError):
                period_plans.select(original, chosen, [0, 1])

    def test_plans_reject_changed_edits_and_foreign_defects(self):
        plan = stream_age(fault='corrupt-w0:4')
        self.assertEqual(period_plans.validate(plan), plan)
        changed = json.loads(json.dumps(plan))
        changed['writers'][0][1]['value'] += 1
        foreign = long_rotations(fault='unannounced-peer-6')
        for bad in (changed, dict(plan, test_service_fault='misroute-rotation-7'), foreign,
                    dict(plan, test_service_fault='misroute-rotation-6'),
                    long_rotations(fault='misroute-rotation-7'),
                    dict(plan, origin=dict(client_count=4)), dict(plan, scenario='other'), []):
            with self.subTest(plan=str(bad)[:60]), self.assertRaises(ValueError):
                period_plans.validate(bad)

    def test_long_rotation_plans_carry_their_own_defects(self):
        for fault in ('corrupt-w0:4', 'misroute-rotation-6'):
            with self.subTest(fault=fault):
                plan = long_rotations(fault=fault)
                self.assertEqual(period_plans.validate(plan), plan)
                cut = period_plans.truncate(plan, 6)
                self.assertEqual((period_plans.periods(cut), cut['test_service_fault']), ([1, 2, 3, 4, 5, 6], fault))

    def test_every_selection_alternates_backends_from_an_odd_period(self):
        """A rotation goes to B in an odd period and to A in an even one, and every retained run of periods starts odd
        and alternates parity, so each rotation of a selection changes backend, from A, where writers join."""
        domain = period_domain()
        plan = long_rotations(periods=10)
        units = domain.units(plan)
        for count in range(len(units) + 1):
            for removed in itertools.combinations(units, count):
                candidate = domain.remove(plan, set(removed))
                if candidate is None:
                    continue
                periods = period_plans.periods(candidate)
                self.assertEqual(periods[0] % 2, 1, periods)
                self.assertTrue(all((a + b) % 2 == 1 for a, b in zip(periods, periods[1:])), periods)
                self.assertEqual(period_plans.last_period(candidate), 10)
        self.assertEqual(period_plans.last_period(plan), 10)
        self.assertEqual(period_plans.last_period(period_plans.truncate(plan, 5)), 10)

    def test_truncation_cuts_after_the_named_period_only(self):
        plan = stream_age(fault='unannounced-peer-6')
        cut = period_plans.truncate(plan, 6)
        self.assertEqual(period_plans.periods(cut), [1, 2, 3, 4, 5, 6])
        self.assertEqual(cut['test_service_fault'], 'unannounced-peer-6')
        self.assertEqual(period_plans.periods(period_plans.truncate(plan, 5)), [1, 2, 3, 4, 5])
        self.assertEqual(period_plans.validate(cut)['writers'][2], plan['writers'][2][:6])
        # Not a period the plan runs, its last period, or no period at all.
        for period in (0, 9, 8, None, '6'):
            with self.subTest(period=period):
                self.assertIsNone(period_plans.truncate(plan, period))
        selected = period_plans.select(stream_age(), [0], [2, 3, 4, 5])
        self.assertEqual(period_plans.periods(period_plans.truncate(selected, 5)), [3, 4, 5])
        self.assertIsNone(period_plans.truncate(selected, 2))

    def test_domain_never_deletes_the_last_cycle(self):
        domain = period_domain()
        plan = period_plans.truncate(stream_age(), 5)
        self.assertEqual(domain.units(plan), ['cycle:0', 'cycle:1', 'cycle:2', 'writer:0', 'writer:1', 'writer:2',
                                              'writer:3'])
        self.assertIsNone(domain.remove(plan, {'cycle:2'}))
        self.assertIsNone(domain.remove(plan, {'writer:0', 'writer:1', 'writer:2', 'writer:3'}))
        reduced = domain.remove(plan, {'cycle:0', 'writer:1'})
        self.assertEqual((period_plans.periods(reduced), writers(reduced)), ([3, 4, 5], [0, 2, 3]))
        self.assertEqual(domain.retained(reduced)[:3], ['w0:2', 'w0:3', 'w0:4'])
        self.assertEqual(domain.truncate(stream_age(), dict(reason='x', period=5)), plan)
        self.assertIsNone(domain.truncate(stream_age(), dict(reason='x', edit='w0:4')))


class PeriodReductionTests(unittest.IsolatedAsyncioTestCase):
    async def reduce(self, original, fails, target):
        """Reduce `original` with an evaluator whose trial fails with `target` when `fails(plan)`; returns the summary
        and the periods each trial ran."""
        ran = []

        async def evaluate(plan, directory):
            period_plans.validate(plan)
            ran.append(period_plans.periods(plan))
            if fails(plan):
                return dict(outcome='mismatch', signature=target, cleanup_complete=True)
            return dict(outcome='passed', cleanup_complete=True)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'reduction'
            summary = await reduce_history(original, target, root, evaluate, domain=period_domain())
            self.assertEqual(json.loads((root / 'original.plan.json').read_text()), original)
            self.assertEqual(period_plans.periods(json.loads((root / 'trial-0001/input.plan.json').read_text())),
                             ran[0])
            return summary, ran

    async def test_corrupted_edit_reduces_from_the_cut_plan_to_that_edit(self):
        target = dict(reason='unexplained committed field', field='owned_0_2', period=5)
        def holds_w0_4(plan):
            return any(op['edit'] == 'w0:4' for writer in plan['writers'] for op in writer)
        summary, ran = await self.reduce(stream_age(fault='corrupt-w0:4'), holds_w0_4, target)
        self.assertEqual((summary['status'], summary['truncated']), ('complete', True))
        self.assertTrue(summary['verified'] and summary['deletion_minimal'], summary)
        self.assertEqual((summary['retained_units'], summary['retained_edits']), (['cycle:2', 'writer:0'], ['w0:4']))
        # The first trial is the cut plan, and no trial runs a period after the failing one.
        self.assertEqual(ran[0], [1, 2, 3, 4, 5])
        self.assertTrue(all(periods[-1] == 5 for periods in ran), ran)
        self.assertLessEqual(summary['attempts'], 16)

    async def test_expiry_failure_of_every_writer_reduces_to_its_cycle_and_one_writer(self):
        target = dict(reason='read did not resume at its coverage frontier after a stream-age expiry', period=6)
        summary, ran = await self.reduce(stream_age(fault='unannounced-peer-6'),
                                         lambda plan: 6 in period_plans.periods(plan), target)
        self.assertEqual((summary['status'], summary['truncated']), ('complete', True))
        self.assertTrue(summary['verified'] and summary['deletion_minimal'], summary)
        self.assertEqual((summary['retained_units'], summary['retained_edits']),
                         (['cycle:2', 'writer:3'], ['w3:4', 'w3:5']))
        self.assertEqual(ran[0], [1, 2, 3, 4, 5, 6])

    async def test_a_long_plan_shortens_its_trials_and_fits_the_trial_budget(self):
        # A ci-scale stream-age plan failing at period 98: the first trial runs 98 periods, never 100, and the
        # leading cycles go first.
        target = dict(reason='read did not resume at its coverage frontier after a stream-age expiry', period=98)
        summary, ran = await self.reduce(stream_age(periods=100), lambda plan: 98 in period_plans.periods(plan),
                                         target)
        self.assertEqual(len(ran[0]), 98)
        self.assertEqual(summary['retained_units'][0], 'cycle:48')
        self.assertLess(sum(map(len, ran[1:])), 2 * 98, [len(periods) for periods in ran])

    async def test_long_rotation_defects_reduce_from_the_cut_plan(self):
        corrupt = dict(reason='unexplained committed field', field='owned_0_2', period=5)
        summary, ran = await self.reduce(
            long_rotations(fault='corrupt-w0:4'),
            lambda plan: any(op['edit'] == 'w0:4' for writer in plan['writers'] for op in writer), corrupt)
        self.assertEqual((summary['status'], summary['truncated'], ran[0]), ('complete', True, [1, 2, 3, 4, 5]))
        self.assertTrue(summary['verified'] and summary['deletion_minimal'], summary)
        self.assertEqual((summary['retained_units'], summary['retained_edits']), (['cycle:2', 'writer:0'], ['w0:4']))
        resume = dict(reason='writers did not resume reading on the rotation destination at its frontier', period=6)
        summary, ran = await self.reduce(long_rotations(fault='misroute-rotation-6'),
                                         lambda plan: 6 in period_plans.periods(plan), resume)
        self.assertEqual((summary['status'], summary['truncated'], ran[0]), ('complete', True, [1, 2, 3, 4, 5, 6]))
        self.assertTrue(summary['verified'] and summary['deletion_minimal'], summary)
        self.assertEqual((summary['retained_units'], summary['retained_edits']),
                         (['cycle:2', 'writer:3'], ['w3:4', 'w3:5']))
        self.assertTrue(all(periods[-1] == 6 for periods in ran), ran)

    async def test_a_nightly_long_rotation_failure_shortens_its_trials(self):
        # The nightly 500-rotation case failing at rotation 498: the first trial runs 498 rotations, never 500, and
        # the leading cycles go first.
        resume = dict(reason='writers did not resume reading on the rotation destination at its frontier', period=498)
        summary, ran = await self.reduce(long_rotations(periods=500), lambda plan: 498 in period_plans.periods(plan),
                                         resume)
        self.assertEqual((summary['status'], len(ran[0])), ('complete', 498))
        self.assertEqual(summary['retained_units'][0], 'cycle:248')
        self.assertLess(sum(map(len, ran[1:])), 2 * 498, [len(periods) for periods in ran])

    async def test_a_cut_plan_that_does_not_reproduce_is_never_replayed_whole(self):
        target = dict(reason='read did not resume at its coverage frontier after a stream-age expiry', period=6)
        summary, ran = await self.reduce(stream_age(), lambda plan: False, target)
        self.assertEqual((summary['status'], summary['attempts'], summary['truncated']), ('not_reproduced', 1, True))
        self.assertEqual(ran, [[1, 2, 3, 4, 5, 6]])

    async def test_a_signature_without_a_period_replays_the_whole_plan_first(self):
        target = dict(reason='unexplained committed field', field='owned_0_2')
        summary, ran = await self.reduce(stream_age(), lambda plan: True, target)
        self.assertFalse(summary['truncated'])
        self.assertEqual(ran[0], list(range(1, 9)))


class ExpiryResumeTests(unittest.TestCase):
    @staticmethod
    def records(frontier, generation, count=1):
        return [dict(stream_generation=generation, resume_sequence=frontier)] * count

    def failure(self, *arguments):
        with self.assertRaises(WriterMismatch) as caught:
            period_plans.check_expiry_resume(*arguments)
        return caught.exception

    def test_resumed_reads_at_the_frontier_pass(self):
        period_plans.check_expiry_resume(6, [0, 1, 2], [self.records(19, 5)] * 3, [3, 3, 4], 19)

    def test_each_class_names_its_period_whichever_writers_fail_and_in_any_order(self):
        cases = [
            ('read stream generation did not advance across a stream-age expiry',
             lambda index: self.records(19, 3 if index == 0 else 5)),
            ('read did not resume at its coverage frontier after a stream-age expiry',
             lambda index: self.records(20 if index != 1 else 19, 5)),
        ]
        for reason, resumed in cases:
            for order in itertools.permutations(range(3)):
                with self.subTest(reason=reason, order=order):
                    error = self.failure(6, list(order), [resumed(index) for index in order], [3, 3, 3], 19)
                    self.assertEqual(error.signature, dict(reason=reason, period=6))
            # Deleting another writer keeps the signature; only the evidence changes.
            failing = [index for index in range(3) if resumed(index) != self.records(19, 5)]
            error = self.failure(6, failing, [resumed(index) for index in failing], [3] * len(failing), 19)
            self.assertEqual(error.signature, dict(reason=reason, period=6))
            self.assertEqual(error.evidence['writers'], failing)

    def test_a_second_resume_gives_the_same_outcome_in_either_read_batch(self):
        """Whether a duplicate resume shares the first resume's read batch or arrives in a later read is scheduling.
        Behind a correct first resume it is never a WriterMismatch: in the batch the check raises a plain error, and
        later the case's plain check that no read resumed outside an expiry does. Behind a first resume at the wrong
        frontier, the signature is the frontier class's, whichever batch the duplicate lands in."""
        def outcome(batch):
            try:
                period_plans.check_expiry_resume(6, [0, 1], batch, [3, 3], 19)
            except WriterMismatch as error:
                return error.signature
            except AssertionError:
                return 'plain'
            return 'later plain check'
        first, wrong = self.records(19, 5), self.records(20, 5)
        duplicate = [dict(stream_generation=6, resume_sequence=19)]
        self.assertEqual(outcome([first + duplicate, first]), 'plain')
        self.assertEqual(outcome([first, first]), 'later plain check')
        frontier = dict(reason='read did not resume at its coverage frontier after a stream-age expiry', period=6)
        self.assertEqual(outcome([wrong + duplicate, first]), frontier)
        self.assertEqual(outcome([wrong, first]), frontier)

    def test_a_writer_without_a_resume_is_a_plain_error(self):
        with self.assertRaises(AssertionError) as caught:
            period_plans.check_expiry_resume(6, [0, 1], [self.records(19, 5), []], [3, 3], 19)
        self.assertNotIsInstance(caught.exception, WriterMismatch)


class RotationResumeTests(unittest.TestCase):
    REASON = 'writers did not resume reading on the rotation destination at its frontier'

    @staticmethod
    def reads(backends, cursor=19):
        return [dict(backend=backend, cursor=cursor, writer=f'wire-{index}', object=1, attempt=40 + index)
                for index, backend in enumerate(backends)]

    def test_reads_on_the_destination_at_the_frontier_pass(self):
        period_plans.check_rotation_resume(6, self.reads('AAA'), 3, 'A', 19)
        # A second reconnect's read is left to the case's plain count check.
        period_plans.check_rotation_resume(6, self.reads('AAA') + self.reads('A'), 3, 'A', 19)

    def test_a_second_read_of_one_writer_never_covers_another_writers_resume(self):
        """w0 resumes correctly twice, as a second reconnect would make it, and w1 at the wrong frontier or backend:
        writers are counted, not reads, so the period signature holds whether or not the extra read happened."""
        w0 = self.reads('A')[0]
        for wrong in (dict(w0, writer='wire-1', cursor=18), dict(w0, writer='wire-1', backend='B')):
            for reads in ([w0, wrong], [w0, dict(w0, attempt=50), wrong]):
                for order in itertools.permutations(reads):
                    with self.subTest(wrong=wrong, reads=len(reads)), self.assertRaises(WriterMismatch) as caught:
                        period_plans.check_rotation_resume(6, list(order), 2, 'A', 19)
                    self.assertEqual(caught.exception.signature, dict(reason=self.REASON, period=6))
                    self.assertEqual(caught.exception.evidence['matching'], 1)

    def test_a_misrouted_or_moved_resume_names_its_period_in_any_read_order(self):
        for reads in (self.reads('BBB'), self.reads('ABA'), self.reads('AAA', cursor=20),
                      self.reads('AA') + self.reads('A', cursor=20)):
            for order in itertools.permutations(reads):
                with self.subTest(reads=[(r['backend'], r['cursor']) for r in order]):
                    with self.assertRaises(WriterMismatch) as caught:
                        period_plans.check_rotation_resume(6, list(order), 3, 'A', 19)
                    self.assertEqual(caught.exception.signature, dict(reason=self.REASON, period=6))
            # Deleting a writer, whose read goes with it, keeps the signature; only the evidence changes.
            wrong = [read for read in reads if (read['backend'], read['cursor']) != ('A', 19)]
            with self.assertRaises(WriterMismatch) as caught:
                period_plans.check_rotation_resume(6, wrong, len(wrong), 'A', 19)
            self.assertEqual(caught.exception.signature, dict(reason=self.REASON, period=6))
            self.assertEqual(caught.exception.evidence['matching'], 0)


class RotationReconnectTests(unittest.TestCase):
    """One writer's read records across a rotation, with reconnect_initial_ms 50 and reconnect_maximum_ms 1000."""
    @staticmethod
    def records(*events):
        return [dict(event_name=name, fields=dict(fields, attempt_id=1)) for name, fields in events]

    def check(self, *events):
        return period_plans.check_rotation_reconnects(self.records(*events), 73, 50, 1000)

    def test_one_first_window_retry_and_one_resume_pass(self):
        retries, resumed = self.check(('rpc_failed', dict(stream_generation=73)),
                                      ('retry_scheduled', dict(stream_generation=73, delay_ms=99)),
                                      ('stream_resumed', dict(stream_generation=74, resume_sequence=285)))
        self.assertEqual(([r['delay_ms'] for r in retries], resumed[0]['stream_generation']), ([99], 74))

    def test_an_immediate_retry_that_fails_before_a_backend_is_the_next_attempt(self):
        # The #413 trace: a 0 ms first retry failed UNAVAILABLE at once on the cut connection, and the second
        # attempt resumed after a wait in its own, doubled window.
        retries, _ = self.check(('retry_scheduled', dict(stream_generation=73, delay_ms=0)),
                                ('rpc_failed', dict(stream_generation=74, grpc_status=14, duration_ms=0)),
                                ('retry_scheduled', dict(stream_generation=74, delay_ms=160)),
                                ('stream_resumed', dict(stream_generation=75, resume_sequence=289)))
        self.assertEqual([r['delay_ms'] for r in retries], [0, 160])

    def test_a_first_retry_outside_the_first_window_shows_an_unreset_attempt_count(self):
        with self.assertRaisesRegex(AssertionError, 'outside its consecutive reconnect attempt window'):
            self.check(('retry_scheduled', dict(stream_generation=73, delay_ms=100)),
                       ('stream_resumed', dict(stream_generation=74)))

    def test_a_later_retry_outside_its_attempt_window_fails(self):
        with self.assertRaisesRegex(AssertionError, 'outside its consecutive reconnect attempt window'):
            self.check(('retry_scheduled', dict(stream_generation=73, delay_ms=0)),
                       ('retry_scheduled', dict(stream_generation=74, delay_ms=200)),
                       ('stream_resumed', dict(stream_generation=75)))

    def test_later_attempt_windows_stop_growing_at_reconnect_maximum_ms(self):
        # Attempt 7's window is 2 * min(1000, 50 * 2^6) = 2000 ms; without the cap it would be 6400 ms.
        def run(last_delay):
            delays = [0, 0, 0, 0, 0, 0, last_delay]
            events = [('retry_scheduled', dict(stream_generation=73 + i, delay_ms=d)) for i, d in enumerate(delays)]
            return self.check(*events, ('stream_resumed', dict(stream_generation=73 + len(delays))))
        self.assertEqual(run(1999)[0][-1]['delay_ms'], 1999)
        with self.assertRaisesRegex(AssertionError, 'outside its consecutive reconnect attempt window'):
            run(2000)

    def test_retries_must_advance_the_stream_generation(self):
        with self.assertRaisesRegex(AssertionError, 'outside its consecutive reconnect attempt window'):
            self.check(('retry_scheduled', dict(stream_generation=73, delay_ms=0)),
                       ('retry_scheduled', dict(stream_generation=73, delay_ms=10)),
                       ('stream_resumed', dict(stream_generation=74)))
        with self.assertRaisesRegex(AssertionError, 'already retried from'):
            self.check(('retry_scheduled', dict(stream_generation=73, delay_ms=0)),
                       ('stream_resumed', dict(stream_generation=73)))

    def test_a_first_retry_from_another_stream_fails(self):
        with self.assertRaisesRegex(AssertionError, 'not from the stream the rotation ended'):
            self.check(('retry_scheduled', dict(stream_generation=72, delay_ms=0)),
                       ('stream_resumed', dict(stream_generation=74)))

    def test_no_retry_or_a_second_resume_fails(self):
        for events in ([('stream_resumed', dict(stream_generation=74))],
                       [('retry_scheduled', dict(stream_generation=73, delay_ms=0))],
                       [('retry_scheduled', dict(stream_generation=73, delay_ms=0)),
                        ('stream_resumed', dict(stream_generation=74)), ('stream_resumed', dict(stream_generation=75))]):
            with self.subTest(events=events), self.assertRaisesRegex(AssertionError, 'did not reconnect once'):
                self.check(*events)


class PeriodAccountingTests(unittest.TestCase):
    def test_a_period_ledger_mismatch_names_its_period_in_any_commit_order(self):
        for plan in (stream_age(), long_rotations()):
            with self.subTest(scenario=plan['scenario']):
                self.check_period_ledger(plan)

    def check_period_ledger(self, plan):
        ops = [writer[4] for writer in plan['writers']]
        partial = dict(plan, writers=[[op] for op in ops])
        commits = [dict(sequence=0, writer=f'wire-{op["writer"]}',
                        fields={op['field']: -1 if op['edit'] == 'w0:4' else op['value']}) for op in ops]
        for order in itertools.permutations(commits):
            ledger = [dict(commit, sequence=30 + position) for position, commit in enumerate(order, 1)]
            with self.subTest(order=[commit['writer'] for commit in order]):
                with self.assertRaises(WriterMismatch) as caught, period_plans.in_period(5):
                    writer_history.account(partial, [{'edit': op['edit']} for op in ops], ledger, base_sequence=30)
                self.assertEqual(caught.exception.signature,
                                 dict(reason='unexplained committed field', field='owned_0_2', period=5))
                self.assertEqual(caught.exception.evidence['fields'], ['owned_0_2'])

    def test_a_named_period_is_kept(self):
        with self.assertRaises(WriterMismatch) as caught, period_plans.in_period(7):
            raise WriterMismatch('read did not resume at its coverage frontier after a stream-age expiry', period=6)
        self.assertEqual(caught.exception.signature['period'], 6)


if __name__ == '__main__':
    unittest.main()
