# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deliberate loss/identity/type mutations of structural workload evidence."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
import structural_history as history


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
            if ordinal in (2, 4, 7, 9):
                effect = dict(kind='create', id=2000 + ordinal, parent=oracle['branches'][0]['ids'][''],
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
        self.assertEqual(len(oracle['lineage']), 24)
        self.assertTrue(oracle['lineage']['w0:0']['deleted_ids'])
        self.assertTrue(oracle['lineage']['w0:1']['deleted_ids'])
        self.assertEqual(oracle['lineage']['w1:3']['superseded_by'], 'w0:3')
        self.assertIn(2004, oracle['lineage']['w0:4']['deleted_ids'])
        self.assertIn(2009, oracle['lineage']['w0:9']['deleted_ids'])
        first = admissions[0]['ids']['']
        replacement = next(a for a in admissions if a['edit'] == 'w0:5')['ids']['']
        self.assertNotEqual(first, replacement)
        self.assertIn(first, oracle['used_ids'])
        history.verify_scene(oracle['scene'], oracle['scene'])

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
        self.assertEqual([(lineage[f'w1:{i}']['outcome'], lineage[f'w1:{i}']['superseded_by']) for i in (1, 2, 4)],
                         [('superseded', 'w0:0')] * 3)
        self.assertEqual(lineage['w1:3']['outcome'], 'superseded')  # The shared field still lands, then w0:3 overwrites it.
        self.assertEqual(lineage['w0:0']['outcome'], 'reflected')
        self.assertEqual(lineage['w1:5']['outcome'], 'reflected')  # A new branch after erasing the lost one.
        self.assertIn(ids['w1:10'][''], {node['id'] for node in oracle['scene']})
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

    def test_lost_deleted_edits_duplicate_invented_fifo_writer_and_sequence_fail(self):
        plan, admissions, peers, original, _ = self.fixture()
        for kind in ('lost-deleted-create', 'lost-deleted-field', 'duplicate', 'invented', 'fifo', 'same-writer', 'sequence'):
            with self.subTest(kind=kind):
                commits = copy.deepcopy(original)
                if kind.startswith('lost-'):
                    target = 'w0:0' if kind.endswith('create') else 'w0:1'
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
                    next(a for a in admissions if a['edit'] == 'w0:5')['ids'][''] = admissions[0]['ids']['']
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
