# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deliberate data-loss/replay mutations must fail the many-writer oracle."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
import rotation_history
import round_plans
from writer_history import account, generate, load, validate, verify_provenance, read_json, WriterMismatch

COVER = "writer's commits do not cover its admitted edits in contiguous FIFO runs"


class WriterHistoryTests(unittest.TestCase):
    def test_every_replay_identity_is_required_and_type_sensitive(self):
        plan = generate(42, clients=2, steps=4)
        identities = dict(source_fingerprint='source', candidate={'version': 1}, reference_identity={'python': 'test'},
                          sanitizer_runtime={}, initial_fixture_sha256='fixture')
        plan.update(identities)
        expected = dict(source='source', candidate={'version': 1}, reference={'python': 'test'},
                        sanitizer={}, fixture='fixture')
        verify_provenance(plan, **expected)
        for key in identities:
            for mode in ('missing', 'changed'):
                with self.subTest(key=key, mode=mode):
                    changed = copy.deepcopy(plan)
                    if mode == 'missing': del changed[key]
                    else: changed[key] = 'different'
                    with self.assertRaises(ValueError): verify_provenance(changed, **expected)
        plan['candidate']['version'] = True
        with self.assertRaises(ValueError): verify_provenance(plan, **expected)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'replay.json'
            path.write_text('{"writer_plan":{"master_seed":42,"master_seed":43}}')
            with self.assertRaisesRegex(ValueError, 'duplicate'): read_json(path, limit=16 * 1024 * 1024)

    def test_reduced_origins_remain_fifo_without_renumbering(self):
        plan = generate(42, clients=2, steps=8)
        plan['writers'] = [[writer[i] for i in (2, 5, 7)] for writer in plan['writers']]
        validate(plan)
        ops = [op for pair in zip(*plan['writers']) for op in pair]
        admissions = [dict(edit=op['edit']) for op in ops]
        commits = [dict(sequence=i + 2, writer=str(op['writer']), fields={op['field']: op['value']})
                   for i, op in enumerate(ops)]
        _, lineage = account(plan, admissions, commits)
        self.assertEqual(set(lineage), {op['edit'] for op in ops})
        commits[0], commits[2] = commits[2], commits[0]
        for i, commit in enumerate(commits):
            commit['sequence'] = i + 2
        with self.assertRaisesRegex(AssertionError, 'FIFO'):
            account(plan, admissions, commits)

    def test_replay_rejects_ambiguous_modified_and_unbounded_inputs(self):
        original = generate(42, clients=2, steps=4)
        for kind in ('value', 'edit', 'bool', 'order', 'duplicate', 'empty', 'release', 'seed', 'fault', 'unknown'):
            with self.subTest(kind=kind):
                plan = copy.deepcopy(original)
                if kind == 'value': plan['writers'][0][0]['value'] += 1
                elif kind == 'edit': plan['writers'][0][0]['edit'] = 'different'
                elif kind == 'bool': plan['writers'][0][0]['writer'] = False
                elif kind == 'order': plan['writers'][0].reverse()
                elif kind == 'duplicate': plan['writers'][0].append(plan['writers'][0][0])
                elif kind == 'empty': plan['writers'][0].clear()
                elif kind == 'release': plan['release_order'] = [0, 0]
                elif kind == 'seed': plan['master_seed'] += 1
                elif kind == 'fault': plan['test_service_fault'] = 'unknown'
                else: plan['unknown'] = True
                with self.assertRaises(ValueError): validate(plan)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'plan.json'
            path.write_text(json.dumps(original))
            self.assertEqual(load(path), original)
            for text in ('{"schema_version":1,"schema_version":1}', '{"seed":NaN}',
                         '[' * 2000 + ']' * 2000, ' ' * (2 * 1024 * 1024 + 1)):
                path.write_text(text)
                with self.assertRaises(ValueError): load(path)

    def fixture(self):
        plan = generate(42, clients=2, steps=4)
        ops = [op for pair in zip(*plan['writers']) for op in pair]
        admissions = [{'edit': op['edit']} for op in ops]
        commits = [{'sequence': i + 2, 'writer': f"native-{op['writer']}",
                    'fields': {op['field']: op['value']}} for i, op in enumerate(ops)]
        return plan, admissions, commits

    def test_every_edit_accounted_including_conflict_losers(self):
        plan, admissions, commits = self.fixture()
        self.assertEqual(plan, generate(42, clients=2, steps=4))
        self.assertNotEqual(plan['writers'], generate(43, clients=2, steps=4)['writers'])
        expected = {}
        for commit in commits:
            expected.update(commit['fields'])
        final, lineage = account(plan, admissions, commits)
        self.assertEqual(final, expected)
        self.assertEqual(len(lineage), len(admissions))
        self.assertTrue(any(entry['outcome'] == 'superseded' for entry in lineage.values()))
        for edit, entry in lineage.items():
            if entry['outcome'] == 'superseded':
                later = lineage[entry['superseded_by']]
                self.assertGreater(later['sequence'], entry['sequence'])
                self.assertEqual(later['field'], entry['field'])

    def test_dropped_superseded_duplicate_invented_and_reordered_commits_fail(self):
        plan, admissions, original = self.fixture()
        _, lineage = account(plan, admissions, original)
        loser = next(v['sequence'] - 2 for v in lineage.values() if v['outcome'] == 'superseded')
        for kind in ('lost-loser', 'duplicate', 'invented', 'fifo', 'same-writer', 'sequence'):
            with self.subTest(kind=kind):
                commits = copy.deepcopy(original)
                if kind == 'lost-loser':
                    del commits[loser]
                elif kind == 'duplicate':
                    commits.insert(1, copy.deepcopy(commits[0]))
                elif kind == 'invented':
                    commits[0]['fields'] = {'alien': 777}
                elif kind == 'fifo':
                    commits[0], commits[2] = commits[2], commits[0]
                elif kind == 'same-writer':
                    for commit in commits:
                        commit['writer'] = 'shared'
                # Renumbering prevents sequence gaps alone from detecting loss.
                for i, commit in enumerate(commits):
                    commit['sequence'] = i + 2
                if kind == 'sequence':
                    commits[1]['sequence'] += 1
                with self.assertRaises(AssertionError):
                    account(plan, admissions, commits)

    @staticmethod
    def coalesced(plan, runs):
        """Commits that fold each (writer, first, end) run of the writer's edits, in the given order."""
        commits = []
        for index, (writer, first, end) in enumerate(runs):
            fields = {}
            for op in plan['writers'][writer][first:end]:
                fields[op['field']] = op['value']
            commits.append(dict(sequence=index + 2, writer=f'wire-{writer}', fields=fields))
        return commits

    def test_coalesced_runs_account_every_edit_and_name_the_folded_ones(self):
        plan = generate(7, clients=2, steps=8)
        admissions = [dict(edit=op['edit']) for writer in plan['writers'] for op in writer]
        runs = [(0, 0, 1), (1, 0, 3), (0, 1, 8), (1, 3, 4), (1, 4, 8)]
        commits = self.coalesced(plan, runs)
        expected = {}
        for commit in commits:
            expected.update(commit['fields'])
        final, lineage = account(plan, admissions, commits, coalesced=True)
        self.assertEqual(final, expected)
        self.assertEqual(set(lineage), {a['edit'] for a in admissions})
        # Writer 0's run of edits 1..7 writes a shared field more than once, so one of them folds away.
        folded = [entry for entry in lineage.values() if entry['outcome'] == 'coalesced']
        self.assertTrue(folded)
        for entry in folded:
            later = lineage[entry['superseded_by']]
            self.assertEqual((later['sequence'], later['field']), (entry['sequence'], entry['field']))
        # One edit per commit is the degenerate run and accounts as the uncoalesced ledger does.
        single = self.coalesced(plan, [(w, i, i + 1) for i in range(8) for w in (0, 1)])
        self.assertEqual(account(plan, admissions, single, coalesced=True)[0],
                         account(plan, admissions, single)[0])

    def test_coalesced_runs_reject_loss_gaps_mixing_and_reordering(self):
        plan = generate(7, clients=2, steps=8)
        admissions = [dict(edit=op['edit']) for writer in plan['writers'] for op in writer]
        runs = [(0, 0, 4), (1, 0, 4), (0, 4, 8), (1, 4, 8)]
        account(plan, admissions, self.coalesced(plan, runs), coalesced=True)
        cases = {
            'lost-run': runs[:-1],
            'skipped-edit': [(0, 0, 3), (1, 0, 4), (0, 4, 8), (1, 4, 8)],
            'reordered': [(0, 4, 8), (1, 0, 4), (0, 0, 4), (1, 4, 8)],
            'repeated': runs + [(1, 4, 8)],
        }
        for kind, shape in cases.items():
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                account(plan, admissions, self.coalesced(plan, shape), coalesced=True)
        for kind in ('mixed', 'stale-value', 'same-writer', 'empty', 'sequence'):
            with self.subTest(kind=kind):
                commits = self.coalesced(plan, runs)
                if kind == 'mixed':
                    commits[0]['fields'].update(commits[1]['fields'])
                elif kind == 'stale-value':
                    # A field the run writes twice carries the earlier, folded-away value: no contiguous run folds
                    # to it, so the run's later edits never count as committed.
                    run = plan['writers'][0][4:8]
                    writes = [op['value'] for op in run if op['field'] == 'shared_0']
                    self.assertGreater(len(writes), 1)
                    commits[2]['fields']['shared_0'] = writes[0]
                    with self.assertRaises(WriterMismatch) as caught:
                        account(plan, admissions, commits, coalesced=True)
                    self.assertEqual(caught.exception.signature, dict(reason=COVER, writer=0))
                    continue
                elif kind == 'same-writer':
                    for commit in commits:
                        commit['writer'] = 'shared'
                elif kind == 'empty':
                    commits.insert(1, dict(sequence=0, writer='wire-0', fields={}))
                    for i, commit in enumerate(commits):
                        commit['sequence'] = i + 2
                elif kind == 'sequence':
                    commits[1]['sequence'] += 1
                with self.assertRaises(AssertionError):
                    account(plan, admissions, commits, coalesced=True)

    def test_missing_and_duplicate_admission_fail(self):
        plan, original, commits = self.fixture()
        for kind in ('missing', 'duplicate'):
            with self.subTest(kind=kind):
                admissions = copy.deepcopy(original)
                if kind == 'missing':
                    admissions.pop()
                else:
                    admissions[-1] = admissions[0]
                with self.assertRaises(AssertionError):
                    account(plan, admissions, commits)


def ledger(ops, *, lose=(), corrupt=(), repeat=(), rewire=(), skip=(), gap=False, wires=None):
    """One commit per op of `ops`, in that order: `lose` drops edits (consuming their sequences with `gap`), `corrupt`
    commits an edit's field with the value -1, `repeat` commits an edit twice in a row, `rewire` commits an edit
    under another wire writer ID, `skip` skips the sequence before an edit, and `wires` maps a seeded writer to its
    wire writer ID."""
    commits, sequence = [], 1
    for op in ops:
        sequence += op['edit'] in skip
        for _ in range(2 if op['edit'] in repeat else 1):
            sequence += 1
            if op['edit'] in lose:
                if not gap:
                    sequence -= 1
                continue
            wire = (wires or {}).get(op['writer'], f"native-{op['writer']}")
            commits.append(dict(sequence=sequence, writer='other' if op['edit'] in rewire else wire,
                                fields={op['field']: -1 if op['edit'] in corrupt else op['value']}))
    return commits


def signature(plan, commits, **options):
    admissions = [dict(edit=op['edit']) for writer in plan['writers'] for op in writer]
    try:
        account(plan, admissions, commits, **options)
    except WriterMismatch as error:
        return error.signature, error.evidence
    return None, None


class RootFieldSignatureTests(unittest.TestCase):
    """Root-field accounting signatures are deterministic across trials of one plan and survive its reduction."""

    @staticmethod
    def interleavings(plan, count=8):
        """Commit orders of every edit of a root-field plan, which authors every writer's edits at once: each writer's
        in its FIFO order, interleaved by seeded shuffles as the OS schedules them."""
        orders = []
        for seed in range(count):
            rng = random.Random(seed)
            queues = [list(writer) for writer in plan['writers']]
            order = []
            while any(queues):
                order.append(rng.choice([queue for queue in queues if queue]).pop(0))
            orders.append(order)
        return orders

    @staticmethod
    def phase_orders(plan, count=8):
        """Commit orders of a round-based phase: rounds are barriers, and within each round the writers' order is a
        seeded shuffle."""
        orders = []
        for seed in range(count):
            rng = random.Random(seed)
            order = []
            for column in zip(*plan['writers']):
                order += rng.sample(column, len(column))
            orders.append(order)
        return orders

    def test_signature_ignores_the_commit_order_within_a_phase(self):
        root = generate(42, clients=4, steps=4)
        phase = rotation_history.generate_queued(42)
        phase = dict(phase, writers=[writer[:4] for writer in phase['writers']])
        cases = [
            (root, self.interleavings(root), dict(lose={'w2:2', 'w1:1'}),
             dict(reason='admitted edit missing from service ledger', edit='w1:1')),
            (root, self.interleavings(root), dict(lose={'w2:2', 'w1:1', 'w3:3'}, gap=True),
             dict(reason='admitted edit missing from service ledger', edit='w1:1')),
            (root, self.interleavings(root), dict(skip={'w2:2', 'w1:1'}),
             dict(reason='missing sequence in writer ledger')),
            (root, self.interleavings(root), dict(repeat={'w0:3', 'w3:2'}),
             dict(reason='duplicate committed edit', edit='w3:2')),
            (root, self.interleavings(root), dict(corrupt={'w2:0', 'w1:2'}),
             dict(reason='unexplained committed field', field='owned_1_2')),
            (root, self.interleavings(root), dict(rewire={'w1:3', 'w3:1'}),
             dict(reason='writer identity changed within workload', writer=1)),
            (phase, self.phase_orders(phase), dict(lose={'w3:2', 'w2:1', 'w0:3'}),
             dict(reason='admitted edit missing from service ledger', edit='w2:1')),
            (phase, self.phase_orders(phase), dict(lose={'w3:2', 'w2:1', 'w0:3'}, gap=True),
             dict(reason='admitted edit missing from service ledger', edit='w2:1')),
        ]
        for plan, orders, defect, expected in cases:
            with self.subTest(defect=defect):
                results = [signature(plan, ledger(order, **defect)) for order in orders]
                self.assertEqual([found for found, _ in results], [expected] * len(orders))
                # The commit order is evidence: the orders differ, and so does where a gap breaks the ledger.
                self.assertGreater(len({json.dumps([op['edit'] for op in order]) for order in orders}), 1)
        # Two writers each commit their own edits out of FIFO order; the first violation in plan order is named.
        def reordered(order):
            order = list(order)
            for a, b in (('w2:1', 'w2:2'), ('w0:2', 'w0:3')):
                i, j = (next(k for k, op in enumerate(order) if op['edit'] == name) for name in (a, b))
                order[i], order[j] = order[j], order[i]
            return order
        found = {json.dumps(signature(root, ledger(reordered(order)))[0]) for order in self.interleavings(root)}
        self.assertEqual(found, {json.dumps(dict(reason='per-writer commit FIFO violated', edit='w2:1'))})

    def test_signature_survives_deleting_a_later_missing_edit_or_an_earlier_edit(self):
        """A root-field reduction deletes single edits, so the plan order is the seeded step and writer, never the
        position in the writer's retained edits."""
        plan = generate(42, clients=4, steps=4)
        lost = {'w1:2', 'w0:3'}
        reason = 'admitted edit missing from service ledger'
        for gap in (False, True):
            with self.subTest(gap=gap):
                def found(candidate):
                    order = [op for column in zip(*candidate['writers']) for op in column] + [
                        op for writer in candidate['writers'] for op in writer[min(map(len, candidate['writers'])):]]
                    return signature(validate(candidate), ledger(order, lose=lost, gap=gap))
                original, evidence = found(plan)
                self.assertEqual(original, dict(reason=reason, edit='w1:2'))
                self.assertEqual(evidence['missing'], ['w0:3', 'w1:2'])
                for deleted in ({'w0:3'}, {'w0:0', 'w0:1'}, {'w1:0', 'w2:3', 'w3:1'}):
                    candidate = copy.deepcopy(plan)
                    candidate['writers'] = [[op for op in writer if op['edit'] not in deleted]
                                            for writer in candidate['writers']]
                    smaller, evidence = found(candidate)
                    self.assertEqual(smaller, original, deleted)
                    self.assertEqual(evidence['missing'], sorted(lost - deleted))

    def test_distinct_missing_edits_give_distinct_signatures(self):
        root = generate(42, clients=4, steps=4)
        phase = rotation_history.generate_queued(42)
        phase = dict(phase, writers=[writer[:4] for writer in phase['writers']])
        for plan in (root, phase):
            edits = [op['edit'] for writer in plan['writers'] for op in writer]
            order = [op for column in zip(*plan['writers']) for op in column]
            for gap in (False, True):
                found = [signature(plan, ledger(order, lose={edit}, gap=gap))[0] for edit in edits]
                self.assertEqual(found, [dict(reason='admitted edit missing from service ledger', edit=edit)
                                         for edit in edits])

    def test_coalesced_signatures_ignore_run_boundaries_and_interleaving(self):
        """Which blocks coalesce into one commit is OS-scheduled, so a coalesced signature names a field or a writer,
        never a run, its first edit or its sequence."""
        plan = generate(7, clients=2, steps=8)
        shapes = [[(0, 8)], [(0, 1), (1, 3), (3, 8)], [(0, 2), (2, 3), (3, 5), (5, 8)], [(0, 4), (4, 8)],
                  [(i, i + 1) for i in range(8)], [(0, 3), (3, 6), (6, 8)]]

        def commits(shape, other, seed, *, lose=False, corrupt=False, gap=False, skip=False):
            """Writer 0 commits runs `shape`, writer 1 runs `other`, interleaved by a seeded shuffle; the defect
            hits the run carrying owned_0_2."""
            queues = [[(0, first, end) for first, end in shape], [(1, first, end) for first, end in other]]
            rng, result, sequence = random.Random(seed), [], 1
            while any(queues):
                writer, first, end = rng.choice([queue for queue in queues if queue]).pop(0)
                fields = {}
                for op in plan['writers'][writer][first:end]:
                    fields[op['field']] = op['value']
                sequence += 1 + (skip and 'owned_0_2' in fields)
                if 'owned_0_2' in fields:
                    if lose:
                        sequence -= 0 if gap else 1
                        continue
                    if corrupt:
                        fields['owned_0_2'] = -1
                result.append(dict(sequence=sequence, writer=f'wire-{writer}', fields=fields))
            return result

        cases = [(dict(corrupt=True), dict(reason='unexplained committed field', field='owned_0_2')),
                 (dict(lose=True), dict(reason=COVER, writer=0)),
                 (dict(lose=True, gap=True), dict(reason=COVER, writer=0)),
                 (dict(skip=True), dict(reason='missing sequence in writer ledger'))]
        for defect, expected in cases:
            with self.subTest(defect=defect):
                first_uncovered = set()
                for index, shape in enumerate(shapes):
                    for seed in range(4):
                        found, evidence = signature(plan, commits(shape, shapes[(index + seed) % len(shapes)], seed,
                                                                  **defect), coalesced=True)
                        self.assertEqual(found, expected, (shape, seed))
                        first_uncovered.update(entry['first_uncovered'] for entry in evidence.get('uncovered', ()))
                if defect.get('lose'):
                    # The lost run's first edit depends on the boundaries, so it is evidence only.
                    self.assertGreater(len(first_uncovered), 1, first_uncovered)

    def test_shared_wire_identity_names_the_least_writer_and_survives_deleting_another(self):
        """Writers 0, 1 and 2 share one wire ID; deleting writer 1, as a round-domain reduction does, keeps writer 0."""
        original = rotation_history.generate_queued(42)
        wires = {0: 'shared', 1: 'shared', 2: 'shared'}
        for writers in ([0, 1, 2, 3], [0, 2, 3], [0, 2]):
            plan = round_plans.select(original, writers, [0, 1, 2, 3], [])
            order = [op for column in zip(*plan['writers']) for op in column]
            for coalesced in (False, True):
                with self.subTest(writers=writers, coalesced=coalesced):
                    found, evidence = signature(plan, ledger(order, wires=wires), coalesced=coalesced)
                    self.assertEqual(found, dict(reason='independent clients share a wire writer identity', writer=0))
                    self.assertEqual(evidence['pairs'], [[a, b] for a in writers for b in writers
                                                         if a < b and b in wires])

    def test_unplanned_field_names_nothing_and_a_planned_one_is_named(self):
        """A field no planned edit authors comes from the defect, so it is evidence only, in both accounting modes."""
        plan = generate(42, clients=4, steps=4)
        order = [op for column in zip(*plan['writers']) for op in column]
        for coalesced in (False, True):
            with self.subTest(coalesced=coalesced):
                found = []
                for name in ('invented_a', 'invented_b'):
                    commits = ledger(order)
                    commits.append(dict(sequence=len(commits) + 2, writer='native-0', fields={name: 1}))
                    signed, evidence = signature(plan, commits, coalesced=coalesced)
                    self.assertEqual(evidence['fields'], [name])
                    found.append(signed)
                self.assertEqual(found, [dict(reason='committed field no planned edit authors')] * 2)
                commits = ledger(order, corrupt={'w1:2'})
                commits.append(dict(sequence=len(commits) + 2, writer='native-0', fields={'invented_a': 1}))
                self.assertEqual(signature(plan, commits, coalesced=coalesced)[0],
                                 dict(reason='unexplained committed field', field='owned_1_2'))

    def test_identity_change_on_a_writers_first_commit_names_the_writer(self):
        """The ledger does not say which of a writer's commits has the wrong identity, so the writer is named, and
        deleting its unaltered next edit keeps the signature."""
        plan = generate(42, clients=4, steps=4)
        for deleted in (set(), {'w3:1'}, {'w3:1', 'w3:2'}):
            candidate = copy.deepcopy(plan)
            candidate['writers'] = [[op for op in writer if op['edit'] not in deleted] for writer in plan['writers']]
            order = [op for writer in validate(candidate)['writers'] for op in writer]
            for coalesced in (False, True):
                with self.subTest(deleted=deleted, coalesced=coalesced):
                    self.assertEqual(signature(candidate, ledger(order, rewire={'w3:0'}), coalesced=coalesced)[0],
                                     dict(reason='writer identity changed within workload', writer=3))

    def test_coalesced_foreign_value_gives_one_signature_across_run_shapes(self):
        """A defect puts one writer's planned value in place of another writer's edit; whether that run carries other
        edits depends on when the window drains, and the signature is the same either way."""
        plan = generate(7, clients=2, steps=8)
        shapes = [[(i, i + 1) for i in range(8)], [(0, 2), (2, 8)], [(0, 1), (1, 3), (3, 8)], [(0, 8)],
                  [(0, 1), (1, 2), (2, 8)]]
        for victim, donor in ((0, 1), (1, 0)):
            # Owned fields, which no later edit of either writer rewrites, so no run folds the defect away.
            replaced = plan['writers'][victim][2]
            foreign = plan['writers'][donor][2]
            with self.subTest(victim=victim):
                found = set()
                for index, shape in enumerate(shapes):
                    for seed in range(3):
                        runs = {victim: shape, donor: shapes[(index + seed + 1) % len(shapes)]}
                        queues = [[(w, a, b) for a, b in runs[w]] for w in (0, 1)]
                        rng, commits = random.Random(seed), []
                        while any(queues):
                            writer, first, end = rng.choice([queue for queue in queues if queue]).pop(0)
                            fields = {}
                            for op in plan['writers'][writer][first:end]:
                                if op is replaced:
                                    fields[foreign['field']] = foreign['value']
                                else:
                                    fields[op['field']] = op['value']
                            commits.append(dict(sequence=len(commits) + 2, writer=f'wire-{writer}', fields=fields))
                        found.add(json.dumps(signature(plan, commits, coalesced=True)[0], sort_keys=True))
                self.assertEqual(found, {json.dumps(dict(reason=COVER, writer=0), sort_keys=True)})
