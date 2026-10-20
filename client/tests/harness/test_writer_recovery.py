# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deliberately corrupt exported-layer evidence to test the offline oracle."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
from writer_history import generate
from writer_recovery import EvidenceError, main, verify, verify_ledger
from scene_wire import update


def usda(fields):
    return '#usda 1.0\n(\n' + ''.join(f'    {key} = {value}\n' for key, value in sorted(fields.items())) + ')\n'


class WriterRecoveryTests(unittest.TestCase):
    def fixture(self):
        plan = generate(42, clients=3, steps=4)
        ops = plan['writers'][0]
        local = {}
        for op in ops:
            local[op['field']] = op['value']
        first, second = plan['writers'][1][0], plan['writers'][2][0]
        commits = [dict(sequence=2, writer='peer-1', wire=update({first['field']: first['value']}).hex()),
                   dict(sequence=3, writer='original', wire=update({ops[0]['field']: ops[0]['value']}).hex()),
                   dict(sequence=4, writer='peer-2', wire=update({second['field']: second['value']}).hex())]
        # The host installed the first peer write before the cut and the second
        # one after it; every local edit overlays both.
        before = dict({first['field']: first['value']}, **local)
        host = dict({first['field']: first['value'], second['field']: second['value']}, **local)
        expected = dict(status=dict(subject='77', state=7, incident=9, pending=4),
                        state=dict(state=7, result=19, pending=4, fields=dict(host)), before_cut=before)
        return usda(host), expected, ops, commits, plan

    def test_frozen_export_with_installed_peer_prefix_passes(self):
        text, expected, ops, commits, _ = self.fixture()
        self.assertEqual(verify(text, expected, ops, commits)['operations'], 4)

    def test_export_at_the_before_cut_prefix_passes(self):
        _, expected, ops, commits, _ = self.fixture()
        expected['state']['fields'] = dict(expected['before_cut'])
        self.assertEqual(verify(usda(expected['before_cut']), expected, ops, commits)['fields'], 5)

    def test_export_that_lost_a_local_edit_fails(self):
        _, expected, ops, commits, _ = self.fixture()
        fields = dict(expected['state']['fields'])
        del fields[ops[-1]['field']]
        expected['state']['fields'] = fields
        with self.assertRaisesRegex(EvidenceError, 'lost a local edit'):
            verify(usda(fields), expected, ops, commits)

    def test_export_that_differs_from_the_frozen_host_fails(self):
        text, expected, ops, commits, _ = self.fixture()
        expected['state']['fields'] = dict(expected['state']['fields'], shared_extra=5)
        with self.assertRaisesRegex(EvidenceError, 'differs from the frozen host layer'):
            verify(text, expected, ops, commits)

    def test_export_below_the_before_cut_prefix_fails(self):
        _, expected, ops, commits, _ = self.fixture()
        local = {op['field']: op['value'] for op in ops}
        expected['state']['fields'] = dict(local)
        with self.assertRaisesRegex(EvidenceError, 'below the ledger prefix shown before the cut'):
            verify(usda(local), expected, ops, commits)

    def test_unapplied_peer_value_fails(self):
        _, expected, ops, commits, _ = self.fixture()
        fields = dict(expected['state']['fields'], owned_1_9=5)
        expected['state']['fields'] = fields
        with self.assertRaisesRegex(EvidenceError, 'unapplied peer value or lost an installed one'):
            verify(usda(fields), expected, ops, commits)

    def test_later_peer_write_without_the_earlier_one_fails(self):
        _, expected, ops, commits, plan = self.fixture()
        first = plan['writers'][1][0]
        fields = dict(expected['state']['fields'])
        del fields[first['field']]
        expected['state']['fields'] = fields
        expected['before_cut'] = {op['field']: op['value'] for op in ops}
        with self.assertRaisesRegex(EvidenceError, 'unapplied peer value or lost an installed one'):
            verify(usda(fields), expected, ops, commits)

    def test_non_usda_export_fails(self):
        _, expected, ops, commits, _ = self.fixture()
        with self.assertRaisesRegex(EvidenceError, 'not a USDA layer'):
            verify('PK\x03\x04', expected, ops, commits)

    def test_export_line_outside_the_profile_fails(self):
        text, expected, ops, commits, _ = self.fixture()
        with self.assertRaisesRegex(EvidenceError, 'outside the root-int32 profile'):
            verify(text.replace('(\n', '(\n    doc = "extra"\n', 1), expected, ops, commits)
        with self.assertRaisesRegex(EvidenceError, 'outside the root-int32 profile'):
            verify(text.replace('(\n', '(\n    owned_0_0 = 1.5\n', 1), expected, ops, commits)

    def test_ledger_accepts_only_first_writes(self):
        _, _, _, commits, plan = self.fixture()
        verify_ledger(plan, commits, [])
        queued = plan['writers'][0][1]
        with self.assertRaisesRegex(EvidenceError, 'queued write was issued'):
            verify_ledger(plan, commits + [dict(sequence=5, writer='original-2',
                                                wire=update({queued['field']: queued['value']}).hex())], [])
        with self.assertRaisesRegex(EvidenceError, 'replayed'):
            verify_ledger(plan, commits + [dict(commits[1], sequence=5, writer='original-2')], [])

    def test_ledger_accepts_each_harness_peer_write_once(self):
        _, _, _, commits, plan = self.fixture()
        peer = {'shared_peer': 7}
        verify_ledger(plan, commits + [dict(sequence=5, writer='peer-3', wire=update(peer).hex())], [peer])
        with self.assertRaisesRegex(EvidenceError, 'queued write was issued'):
            verify_ledger(plan, commits + [dict(sequence=5, writer='peer-3', wire=update(peer).hex())], [])

    def test_ledger_requires_unique_permitted_fingerprints(self):
        _, _, _, commits, plan = self.fixture()
        first = plan['writers'][1][0]
        with self.assertRaisesRegex(EvidenceError, 'fingerprints are not unique'):
            verify_ledger(plan, commits, [{first['field']: first['value']}])

    def test_ledger_rejects_a_wire_writer_that_commits_twice(self):
        _, _, _, commits, plan = self.fixture()
        with self.assertRaisesRegex(EvidenceError, 'wire writer committed more than once'):
            verify_ledger(plan, [dict(commit, writer='original') for commit in commits], [])

    def test_evidence_without_peer_writes_fails(self):
        _, _, _, commits, plan = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'evidence.json'
            path.write_text(json.dumps(dict(clients=[], plan=plan, commits=commits)))
            with self.assertRaisesRegex(EvidenceError, 'omits the harness peer writes'):
                main(path)


if __name__ == '__main__':
    unittest.main()
