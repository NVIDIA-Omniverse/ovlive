# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deliberate data-loss/replay mutations must fail the many-writer oracle."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
from writer_history import account, generate, load, validate, verify_provenance, read_json


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
