# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deliberate loss/identity/type mutations of structural workload evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
import round_plans
import structural_history as history
from writer_history import WriterMismatch


def swapped(commits, first, second):
    """`commits` with the commits at indices `first` and `second` exchanged, sequences kept in ledger order, as an
    OS schedule may commit one round's writers either way."""
    commits = [dict(commit) for commit in commits]
    commits[first], commits[second] = (dict(commits[second], sequence=commits[first]['sequence']),
                                       dict(commits[first], sequence=commits[second]['sequence']))
    return commits


class StructuralHistoryTests(unittest.TestCase):
    def fixture(self, clients=2, plan=None):
        plan = plan or history.generate(42, clients)
        admissions = []
        peers = [dict(bytes='sentinel', effects=[dict(kind='create', id=1000, parent=5, name='Sentinel', value=1)])]
        commits = [dict(sequence=2, bytes='sentinel', writer='independent-peer')]
        oracle = history.account(plan, admissions, peers, commits, 0)
        identity = 10000
        for ordinal in range(plan['rounds']):
            nodes = {n['id']: n for n in oracle['scene']}
            for writer in plan['writers']:
                op = writer[ordinal]
                admission = dict(edit=op['edit'], accepted_id=ordinal + 1, bytes=op['edit'])
                if op['kind'] == 'create':
                    prefix = nodes[op['parent']]['path'] + '/' + op['name']
                    specs = history.template(op['shape'], prefix, op['value'])
                    ids = {suffix: identity + i for i, suffix in enumerate(specs)}
                    identity += len(ids)
                    admission.update(captured_prefix=prefix, ids=ids)
                admissions.append(admission)
            branched = sorted(oracle['branches'])
            if ordinal in (2, 4, 7, 9) and branched:
                # A peer creates a child under the first writer's branch that exists.
                effect = dict(kind='create', id=2000 + ordinal, parent=oracle['branches'][branched[0]]['ids'][''],
                              name=f'Late{ordinal}', value=500 + ordinal)
                peers.append(dict(bytes=f'peer{ordinal}', effects=[effect]))
                commits.append(dict(sequence=len(commits) + 2, bytes=f'peer{ordinal}', writer='independent-peer'))
            for writer in reversed(plan['writers']):
                op = writer[ordinal]
                commits.append(dict(sequence=len(commits) + 2, bytes=op['edit'], writer=f"native-{op['writer']}"))
            oracle = history.account(plan, admissions, peers, commits, ordinal + 1)
        return plan, admissions, peers, commits, oracle

    def test_seeded_lifecycles_preserve_history_after_deletion_and_supersession(self):
        plan, admissions, peers, commits, oracle = self.fixture()
        self.assertEqual(plan, history.generate(42, 2))
        self.assertNotEqual(plan, history.generate(43, 2))
        lineage = oracle['lineage']
        self.assertEqual(len(lineage), 24)
        self.assertTrue(lineage['w0:0']['deleted_ids'])
        self.assertTrue(lineage['w0:1']['deleted_ids'])
        # Writer 0's shared-field writes supersede each other; round 11's commit order puts writer 1's first.
        self.assertEqual(lineage['w0:2']['superseded_by'], 'w0:5')
        self.assertEqual(lineage['w1:11']['superseded_by'], 'w0:11')
        # Round 2's peer child under writer 1's branch leaves with the branch's erase in round 5.
        self.assertIn(2002, lineage['w1:5']['deleted_ids'])
        # Round 4's peer child under writer 0's branch moves with its rename, and writer 1's later reorder of /P
        # supersedes its earlier one.
        self.assertIn(2004, lineage['w0:4']['live_ids'])
        self.assertEqual(lineage['w1:2']['superseded_by'], 'w1:9')
        first = admissions[0]['ids']['']
        replacement = next(a for a in admissions if a['edit'] == 'w0:3')['ids']['']
        self.assertNotEqual(first, replacement)
        self.assertIn(first, oracle['used_ids'])
        history.verify_scene(oracle['scene'], oracle['scene'])

    def test_generated_plans_draw_every_class_and_keep_a_reordered_parent_to_its_edit(self):
        """Each seed draws its writers' edits, covering every edit class and value type, with unique names; a
        rename or reorder edits /P, and no other edit of its round changes /P's children."""
        plans = {}
        for seed in range(40):
            for clients in (2, 4, 8):
                plan = history.generate(seed, clients)
                self.assertEqual(plan, history.generate(seed, clients))
                self.assertIs(history.validate(plan), plan)
                self.assertTrue(history.covers(plan['writers']))
                names = [op['name'] for writer in plan['writers'] for op in writer if 'name' in op]
                self.assertEqual(len(names), len(set(names)))
                branches = [None] * clients
                for ordinal in range(plan['rounds']):
                    changed, ordered = [], []
                    for op in (writer[ordinal] for writer in plan['writers']):
                        kind, parent = op['kind'], branches[op['writer']]
                        if kind in ('rename', 'reorder'):
                            self.assertEqual((op['parent'], parent), (history.ORDERED_PARENT,) * 2)
                            ordered.append(op['parent'])
                        elif kind == 'create':
                            changed.append(op['parent'])
                            branches[op['writer']] = op['parent']
                        elif kind == 'move':
                            changed += [parent, op['parent']]
                            branches[op['writer']] = op['parent']
                        elif kind == 'erase':
                            changed.append(parent)
                            branches[op['writer']] = None
                    self.assertFalse(set(ordered) & set(changed), (seed, clients, ordinal))
                    self.assertEqual(len(ordered), len(set(ordered)), (seed, clients, ordinal))
                plans[seed, clients] = plan['writers']
        self.assertEqual(len({repr(writers) for writers in plans.values()}), len(plans))

    def test_a_plan_of_another_scenario_is_refused(self):
        plan = history.generate(42, 2)
        for changed in (dict(plan, scenario='private-attachment.structural-v1'), dict(plan, schema_version=1)):
            with self.subTest(changed={key: changed[key] for key in ('scenario', 'schema_version')}):
                with self.assertRaises(ValueError):
                    history.validate(changed)

    def test_rename_ends_last_and_reorder_reverses_the_parents_children(self):
        plan = round_plans.select(history.generate(42, 2), [0, 1], [0, 1, 2, 3, 4], [])
        _, admissions, _, _, oracle = self.fixture(plan=plan)
        ops = {op['edit']: op for writer in plan['writers'] for op in writer}
        nodes = {node['id']: node for node in oracle['scene']}
        renamed = ops['w0:4']['name']
        # Writer 1's round-2 reorder put its branch before /P/Child; round 4's rename moves writer 0's later branch
        # to the end of /P's children, and round 4's peer child under it moves with it.
        self.assertEqual(nodes[2]['fields']['primChildren']['value'], [ops['w1:0']['name'], 'Child', renamed])
        self.assertEqual(nodes[2]['fields'][history.REORDER_FIELD], history.value('int32', ops['w1:2']['value']))
        self.assertEqual(nodes[2004]['path'], f'/P/{renamed}/Late4')
        root = next(a for a in admissions if a['edit'] == 'w0:3')['ids']['']
        lineage = oracle['lineage']
        self.assertEqual((lineage['w0:4']['parents'], set(lineage['w0:4']['live_ids']) >= {root, 2004}),
                         ([2], True))
        self.assertEqual((lineage['w1:2']['parents'], 2 in lineage['w1:2']['ids']), ([2], True))
        history.verify_scene(oracle['scene'], oracle['scene'], lineage)

    def test_rename_moves_a_branch_before_a_later_sibling_to_the_end(self):
        """The host's rename appends the renamed prim, so a branch created before another writer's ends after it."""
        plan = round_plans.select(history.generate(42, 2), [0, 1], [0, 1, 2, 3, 4], [])
        shared = history.value('int32', 7)
        writers = plan['writers']
        # Writer 0 keeps its round-0 branch under /P and renames it in round 4; writer 1 creates its branch under /P
        # in round 1, after writer 0's.
        for ordinal in (1, 3):
            writers[0][ordinal] = dict(writers[0][ordinal], kind='field', value=history.value('int32', ordinal))
        writers[1][0] = dict(writers[1][0], kind='shared', value=shared)
        writers[1][1] = dict(writers[0][0], writer=1, ordinal=1, edit='w1:1', name='Writer1_late')
        writers[1][2] = dict(writers[1][2], kind='shared', value=history.value('int32', 8))
        _, _, _, _, oracle = self.fixture(plan=plan)
        nodes = {node['id']: node for node in oracle['scene']}
        self.assertEqual(nodes[2]['fields']['primChildren']['value'],
                         ['Child', 'Writer1_late', writers[0][4]['name']])

    def test_children_order_difference_names_the_reorder_that_changed_them(self):
        """A reader whose /P lists its children in the order before writer 1's round-2 reorder names that reorder,
        the round's only edit of /P."""
        plan = round_plans.select(history.generate(42, 2), [1], [0, 2], [])
        plan, _, _, _, oracle = self.fixture(plan=plan)
        scene = copy.deepcopy(oracle['scene'])
        next(node for node in scene if node['id'] == 2)['fields']['primChildren']['value'].reverse()
        with self.assertRaises(WriterMismatch) as caught:
            round_plans.check_replicas(plan, [dict(authoritative=scene, visible=scene)], oracle['scene'],
                                       lambda actual, expected, path: history.scene_edit(actual, expected, path,
                                                                                         oracle['lineage']))
        self.assertEqual(caught.exception.signature,
                         dict(reason='writer authoritative scene differs from the service ledger', edit='w1:2'))

    def test_unexplained_commit_names_its_writers_next_planned_edit(self):
        plan, admissions, peers, commits, _ = self.fixture()
        corrupted = copy.deepcopy(commits)
        next(commit for commit in corrupted if commit['bytes'] == 'w0:3')['bytes'] = 'corrupted'
        for bindings, edit in ((['native-0', 'native-1'], 'w0:3'), (None, None)):
            with self.subTest(bindings=bindings):
                with self.assertRaises(WriterMismatch) as caught:
                    history.account(plan, admissions, peers, corrupted, plan['rounds'], bindings=bindings)
                self.assertEqual(caught.exception.signature,
                                 dict(reason='unexplained or repeated structural commit', edit=edit))

    def test_lost_edit_signature_names_only_the_first_missing_edit(self):
        def lost_from(plan, position):
            """The signature when every commit from round `position` on is lost."""
            plan, admissions, peers, commits, _ = self.fixture(plan=plan)
            later = {op['edit'] for writer in plan['writers'] for op in writer[position:]}
            kept = next(i for i, commit in enumerate(commits) if commit['bytes'] in later)
            with self.assertRaises(WriterMismatch) as caught:
                history.account(plan, admissions, peers, commits[:kept], plan['rounds'])
            return caught.exception
        full = lost_from(history.generate(42, 2), 10)
        reason = 'lost admitted edit/peer effect or shared writer identity'
        self.assertEqual(full.signature, dict(reason=reason, edit='w0:10'))
        self.assertEqual(full.evidence, dict(missing=['w0:10', 'w0:11', 'w1:10', 'w1:11']))
        # Deleting the later missing round changes the evidence, never the signature.
        reduced = lost_from(round_plans.select(history.generate(42, 2), [0, 1], list(range(11)), [5, 9]), 10)
        self.assertEqual(reduced.signature, full.signature)
        self.assertEqual(reduced.evidence, dict(missing=['w0:10', 'w1:10']))

    def test_distinct_missing_admissions_have_distinct_signatures(self):
        plan, admissions, peers, commits, _ = self.fixture()
        signatures = []
        for edit in ('w0:0', 'w1:7'):
            with self.assertRaises(WriterMismatch) as caught:
                history.account(plan, [a for a in admissions if a['edit'] != edit], peers, commits, plan['rounds'])
            signatures.append(caught.exception.signature)
        self.assertEqual(signatures, [dict(reason='missing or duplicate structural admission', edit=edit)
                                      for edit in ('w0:0', 'w1:7')])

    def test_reused_identity_in_a_create_names_the_edit(self):
        plan, admissions, peers, commits, _ = self.fixture()
        reused = copy.deepcopy(admissions)
        create = next(a for a in reused if a['edit'] == 'w1:6')
        create['ids'][''] = next(a for a in reused if a['edit'] == 'w0:0')['ids']['']
        with self.assertRaises(WriterMismatch) as caught:
            history.account(plan, reused, peers, commits, plan['rounds'])
        self.assertEqual(caught.exception.signature, dict(reason='reused or invalid graph identity', edit='w1:6'))

    def scene_signature(self, plan, *, swap=False, change=None):
        """The signature when /P differs from the oracle's scene: an extra field, or `change` applied to it; `swap`
        exchanges the last round's two commits first."""
        plan, admissions, peers, commits, _ = self.fixture(plan=plan)
        if swap:
            commits = swapped(commits, len(commits) - 2, len(commits) - 1)
        oracle = history.account(plan, admissions, peers, commits, plan['rounds'])
        scene = copy.deepcopy(oracle['scene'])
        node = next(node for node in scene if node['id'] == 2)
        (change or (lambda node: node['fields'].update(corrupt=history.value('int32', 1))))(node)
        with self.assertRaises(WriterMismatch) as caught:
            history.verify_scene(scene, oracle['scene'], oracle['lineage'])
        return caught.exception

    def replica_signature(self, value, position, *, swap=False):
        """The check_replicas mismatch of two readers of the seed-42 plan cut after round 3 whose scenes start as the
        oracle's, the service's, with reader `position`'s /P shared field set to `value`; `swap` exchanges round 3's
        two commits first."""
        plan = round_plans.select(history.generate(42, 2), [0, 1], [0, 1, 2, 3], [])
        plan, admissions, peers, commits, _ = self.fixture(plan=plan)
        if swap:
            commits = swapped(commits, len(commits) - 2, len(commits) - 1)
        oracle = history.account(plan, admissions, peers, commits, plan['rounds'])
        reports = [dict(authoritative=copy.deepcopy(oracle['scene']), visible=copy.deepcopy(oracle['scene']))
                   for _ in range(2)]
        for view in ('authoritative', 'visible'):
            next(node for node in reports[position][view] if node['id'] == 2)['fields']['shared'] = history.value(
                'int32', value)
        with self.assertRaises(WriterMismatch) as caught:
            round_plans.check_replicas(plan, reports, oracle['scene'],
                                       lambda actual, expected, path: history.scene_edit(actual, expected, path,
                                                                                         oracle['lineage']))
        return caught.exception

    def test_replica_mismatch_ignores_the_reader_value_and_commit_order(self):
        """A reader whose /P differs from the service after round 3, in which writer 0's creation under /P is the
        latest edit to change it, names w0:3 whichever reader it is, whatever value it holds and whichever writer
        committed last; the path and nodes are evidence."""
        variants = [self.replica_signature(value, position, swap=swap)
                    for value in (-1, 7) for position in (0, 1) for swap in (False, True)]
        self.assertEqual({repr(v.signature) for v in variants}, {repr(dict(
            reason='writer authoritative scene differs from the service ledger', edit='w0:3'))})
        self.assertEqual([(r['position'], r['path']) for r in variants[0].evidence['readers']], [(0, '/P')])
        self.assertEqual(variants[0].evidence['readers'][0]['actual']['fields']['shared'], history.value('int32', -1))

    def test_scene_mismatch_names_the_latest_changing_edit_in_plan_order(self):
        reason = 'observed scene differs from independent structural intents'
        full = self.scene_signature(history.generate(42, 2))
        # Round 11's shared-field writes are the last to change /P, and writer 0 is the first writer of the round.
        self.assertEqual((full.signature, full.evidence), (dict(reason=reason, edit='w0:11'), dict(path='/P')))
        reduced = self.scene_signature(round_plans.select(history.generate(42, 2), [0, 1], list(range(8)), [5]))
        self.assertEqual(reduced.signature, dict(reason=reason, edit='w0:7'))

    def test_scene_mismatch_signature_ignores_the_commit_order_within_a_round(self):
        plan = round_plans.select(history.generate(42, 2), [0, 1], list(range(4)), [])
        _, _, _, commits, _ = self.fixture(plan=plan)
        self.assertEqual([commit['bytes'] for commit in commits[-2:]], ['w1:3', 'w0:3'])
        self.assertEqual(self.scene_signature(plan, swap=True).signature, self.scene_signature(plan).signature)

    def test_scene_mismatch_names_the_edit_that_changed_a_parents_children(self):
        def drop_child(node):
            node['fields']['primChildren']['value'].pop()
        for writer in (0, 1):
            with self.subTest(writer=writer):
                plan = round_plans.select(history.generate(42, 2), [writer], [0], [])
                self.assertEqual(self.scene_signature(plan, change=drop_child).signature,
                                 dict(reason='observed scene differs from independent structural intents',
                                      edit=f'w{writer}:0'))

    def test_sequence_gap_names_the_first_lost_admitted_edit(self):
        plan, admissions, peers, commits, _ = self.fixture()

        def lost(commits, edit):
            with self.assertRaises(WriterMismatch) as caught:
                history.account(plan, admissions, peers, [c for c in commits if c['bytes'] != edit], plan['rounds'])
            self.assertEqual(caught.exception.evidence, dict(gap=2))
            return caught.exception.signature
        reason = 'missing structural sequence'
        round_one = [i for i, commit in enumerate(commits) if commit['bytes'] in ('w0:1', 'w1:1')]
        for edit in ('w1:1', 'w0:1', 'w1:3'):
            with self.subTest(edit=edit):
                self.assertEqual(lost(commits, edit), dict(reason=reason, edit=edit))
                # Round 1's commits in the other order: the commit after the gap differs, the signature does not.
                self.assertEqual(lost(swapped(commits, *round_one), edit), dict(reason=reason, edit=edit))

    def test_stopped_round_names_the_edit_the_ledger_lacks(self):
        """A round whose ledger lacks one acknowledged edit, the round's later commits taking its sequence."""
        plan = round_plans.select(history.generate(42, 3), [0, 1, 2], [0, 1, 2, 3], [])
        plan, admissions, peers, commits, _ = self.fixture(plan=plan)

        def stopped(commits, edit):
            kept = [dict(c, sequence=index + 2) for index, c in enumerate(c for c in commits if c['bytes'] != edit)]
            with self.assertRaises(WriterMismatch) as caught:
                round_plans.stopped_short(lambda: history.account(plan, admissions, peers, kept, plan['rounds']),
                                          frontier=len(commits) + 1, sequence=len(kept) + 1, in_flight=0, refused=0,
                                          stopped=[])
            return caught.exception.signature
        expected = dict(reason='round commits stopped short of its frontier',
                        cause='lost admitted edit/peer effect or shared writer identity', edit='w1:3')
        self.assertEqual(stopped(commits, 'w1:3'), expected)
        # The round's other commits in another order keep the signature.
        last = [i for i, commit in enumerate(commits) if commit['bytes'] in ('w0:3', 'w2:3')]
        self.assertEqual(stopped(swapped(commits, *last), 'w1:3'), expected)

    def test_selected_rounds_account_fifo_by_position(self):
        plan = round_plans.select(history.generate(42, 3), [0, 2], list(range(3, 10)), [5, 9])
        self.assertIs(history.validate(plan), plan)
        _, _, _, _, oracle = self.fixture(plan=plan)
        self.assertEqual(sorted(oracle['lineage']), sorted(f'w{w}:{o}' for w in (0, 2) for o in range(3, 10)))

    def test_selection_requires_each_edit_to_have_its_created_branch(self):
        original = history.generate(42, 2)
        self.assertIs(history.validate(original), original)
        # Writer 0: create /P, erase, shared, create /P, rename, shared, move to /Q, shared, shared, field, move to
        # /P, shared. Writer 1: create /P, field, reorder /P, field, field, erase, create /Q, field, move to /P,
        # reorder /P, move to /Q, shared.
        valid = {0: ([2], [3, 4], [0, 4], [0, 4, 5, 6, 9], [5, 7, 11]),
                 1: ([0, 3], [0, 1, 2, 3, 4, 5], [6, 7, 8, 9], [0, 2, 9], [6, 8, 9, 10, 11])}
        # An edit before its branch's creation, a second creation before an erase, a move to the branch's own
        # parent, and a rename or reorder at a parent its branch is not under cannot execute as generated.
        invalid = {0: ([4], [0, 3], [3, 4, 10], [10]),
                   1: ([3], [2], [0, 6], [0, 8], [6, 9], [6, 7, 9])}
        for writer in (0, 1):
            for rounds in valid[writer]:
                with self.subTest(writer=writer, rounds=rounds):
                    history.validate(round_plans.select(original, [writer], rounds, []))
            for rounds in invalid[writer]:
                with self.subTest(writer=writer, rounds=rounds):
                    with self.assertRaises(ValueError):
                        history.validate(round_plans.select(original, [writer], rounds, []))

    def test_edit_naming_defect_requires_its_edit_kind_in_the_generated_plan(self):
        # At SELFTEST_SEED, w0:2 is a shared write and w1:2 a reorder, so each defect, and a selection that keeps or
        # drops its edit, validates.
        selftest = history.generate(history.SELFTEST_SEED, 2)
        for fault in history.FAULTS:
            with self.subTest(fault=fault):
                history.validate(dict(selftest, test_service_fault=fault))
                for writer in (0, 1):
                    history.validate(dict(round_plans.select(selftest, [writer], list(range(12)), [5, 9]),
                                          test_service_fault=fault))
        # At seed 1, w1:2 is a move; at seed 2, w0:2 is a move. A selection without the edit is refused too.
        for seed, fault in ((1, 'corrupt-reorder-w1:2'), (2, 'corrupt-shared-w0:2')):
            plan = history.generate(seed, 2)
            for candidate in (plan, round_plans.select(plan, [0], list(range(12)), [5, 9])):
                with self.subTest(seed=seed, fault=fault, writers=len(candidate['writers'])):
                    with self.assertRaisesRegex(ValueError, 'names no'):
                        history.validate(dict(candidate, test_service_fault=fault))
        self.assertTrue(history.validate(dict(history.generate(1, 2), test_service_fault='relay-shared')))

    def test_same_path_create_replaces_earlier_branch_and_supersedes_its_later_edits(self):
        plan = history.generate(42, 2)
        loser, winner = plan['writers'][1], plan['writers'][0]
        # Writer 1 commits first in the fixture's reversed order; writer 0's Create then replaces it.
        for op in loser[:5]:
            if op['kind'] != 'move':
                op.update(parent=winner[0]['parent'], name=winner[0]['name'])
        _, admissions, _, _, oracle = self.fixture(plan=plan)
        lineage, ids = oracle['lineage'], {a['edit']: a['ids'] for a in admissions if 'ids' in a}
        self.assertEqual((lineage['w1:0']['outcome'], lineage['w1:0']['superseded_by']), ('superseded', 'w0:0'))
        self.assertEqual(set(lineage['w1:0']['deleted_ids']), set(ids['w1:0'].values()))
        # The replaced branch's field edits, reorder and erase are each superseded by the replacing creation.
        self.assertEqual([(lineage[f'w1:{i}']['outcome'], lineage[f'w1:{i}']['superseded_by']) for i in range(1, 6)],
                         [('superseded', 'w0:0')] * 5)
        self.assertEqual(lineage['w0:0']['outcome'], 'reflected')
        self.assertEqual(lineage['w1:6']['outcome'], 'reflected')  # A new branch after erasing the lost one.
        self.assertIn(ids['w1:6'][''], {node['id'] for node in oracle['scene']})
        paths = [node['path'] for node in oracle['scene']]
        self.assertEqual(len(paths), len(set(paths)))

    def test_move_to_missing_parent_or_onto_occupant_follows_service_rule(self):
        plan = history.generate(42, 2)
        _, _, peers, commits, _ = self.fixture(plan=plan)
        base = history.account(plan, [], peers[:1], commits[:1], 0)
        nodes = {n['id']: n for n in base['scene']}
        self.assertEqual(nodes[1000]['path'], '/Q/Sentinel')
        effects = [dict(kind='create', id=1001, parent=2, name='Sentinel', value=2),
                   dict(kind='move', id=1001, parent=5, name='Sentinel', old_parent=2),
                   dict(kind='move', id=3, parent=999, name='Gone', old_parent=2)]
        peers = peers[:1] + [dict(bytes='conflicts', effects=effects)]
        commits = commits[:1] + [dict(sequence=3, bytes='conflicts', writer='independent-peer')]
        scene = {n['id']: n for n in history.account(plan, [], peers, commits, 0)['scene']}
        self.assertEqual(scene[1001]['path'], '/Q/Sentinel')
        self.assertNotIn(1000, scene)
        self.assertFalse({3, 4} & set(scene))
        self.assertEqual(scene[2]['fields']['primChildren']['value'], [])

    def test_rename_onto_a_live_sibling_name_replaces_the_sibling(self):
        plan = history.generate(42, 2)
        _, _, peers, commits, _ = self.fixture(plan=plan)
        effects = [dict(kind='create', id=1001, parent=2, name='Extra', value=2),
                   dict(kind='move', id=1001, parent=2, name='Child', old_parent=2),
                   dict(kind='move', id=1001, parent=2, name='Child', old_parent=2)]
        peers = peers[:1] + [dict(bytes='rename', effects=effects)]
        commits = commits[:1] + [dict(sequence=3, bytes='rename', writer='independent-peer')]
        scene = {n['id']: n for n in history.account(plan, [], peers, commits, 0)['scene']}
        # The rename deletes /P/Child{3} with its attribute; the repeated rename to its own name changes nothing.
        self.assertFalse({3, 4} & set(scene))
        self.assertEqual((scene[1001]['path'], scene[1001]['fields']['marker']['value']), ('/P/Child', 2))
        self.assertEqual(scene[2]['fields']['primChildren']['value'], ['Child'])

    def test_lost_deleted_edits_duplicate_invented_fifo_writer_and_sequence_fail(self):
        plan, admissions, peers, original, _ = self.fixture()
        for kind in ('lost-deleted-create', 'lost-deleted-field', 'duplicate', 'invented', 'fifo', 'same-writer', 'sequence'):
            with self.subTest(kind=kind):
                commits = copy.deepcopy(original)
                if kind.startswith('lost-'):
                    target = 'w0:0' if kind.endswith('create') else 'w1:1'
                    commits = [c for c in commits if c['bytes'] != target]
                elif kind == 'duplicate':
                    commits.insert(2, copy.deepcopy(commits[1]))
                elif kind == 'invented':
                    commits[1]['bytes'] = 'unplanned'
                elif kind == 'fifo':
                    a = next(i for i, c in enumerate(commits) if c['bytes'] == 'w0:1')
                    b = next(i for i, c in enumerate(commits) if c['bytes'] == 'w0:3')
                    commits[a], commits[b] = commits[b], commits[a]
                elif kind == 'same-writer':
                    for c in commits:
                        if c['writer'] != 'independent-peer':
                            c['writer'] = 'shared-native'
                for i, c in enumerate(commits):
                    c['sequence'] = i + 2
                if kind == 'sequence':
                    commits[1]['sequence'] += 1
                with self.assertRaises(AssertionError):
                    history.account(plan, admissions, peers, commits, plan['rounds'])

    def test_identity_reuse_and_missing_admissions_fail(self):
        plan, original, peers, commits, _ = self.fixture()
        for kind in ('missing', 'duplicate', 'zero', 'accepted-reuse', 'identity-reuse', 'aliased-ids'):
            with self.subTest(kind=kind):
                admissions = copy.deepcopy(original)
                if kind == 'missing':
                    admissions.pop()
                elif kind == 'duplicate':
                    admissions[-1] = admissions[0]
                elif kind == 'zero':
                    admissions[0]['accepted_id'] = 0
                elif kind == 'accepted-reuse':
                    admissions[2]['accepted_id'] = admissions[0]['accepted_id']
                elif kind == 'identity-reuse':
                    next(a for a in admissions if a['edit'] == 'w0:3')['ids'][''] = admissions[0]['ids']['']
                else:
                    keys = list(admissions[0]['ids'])
                    admissions[0]['ids'][keys[1]] = admissions[0]['ids'][keys[0]]
                with self.assertRaises(AssertionError):
                    history.account(plan, admissions, peers, commits, plan['rounds'])

    def test_wrong_typed_scene_and_stale_late_child_fail(self):
        _, _, _, _, oracle = self.fixture()
        for kind in ('missing-node', 'wrong-parent', 'wrong-type', 'bool-coercion', 'stale-child'):
            with self.subTest(kind=kind):
                scene = copy.deepcopy(oracle['scene'])
                if kind == 'missing-node':
                    scene.pop()
                elif kind == 'wrong-parent':
                    scene[-1]['parent'] = 1
                elif kind == 'wrong-type':
                    scene[-1]['type'] = 7
                elif kind == 'bool-coercion':
                    sentinel = next(n for n in scene if n['id'] == 1000)
                    sentinel['fields']['marker']['value'] = True  # Same Python equality as int 1.
                else:
                    scene.append(dict(id=2009, parent=2, type=6, path='/P/Late9', fields={}))
                with self.assertRaises(AssertionError):
                    history.verify_scene(scene, oracle['scene'])
