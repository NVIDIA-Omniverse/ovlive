# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loss/replay mutations must fail even when sample payloads are identical."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
import sample_history as history


def fixture(rounds=8):
    plan = history.generate(42, clients=2)
    bindings = ['native-0', 'native-1']
    admissions, commits = [], []
    for ordinal in range(rounds):
        for writer in range(2):
            op = plan['writers'][writer][ordinal]
            payload = f"{op['kind']}:{op['section']}:{op['time_bits']}:{op['value']}"
            admissions.append(dict(edit=op['edit'], writer=bindings[writer], accepted_id=ordinal*3+1,
                                   local_ordinal=ordinal+1, bytes=payload))
            commits.append(dict(sequence=len(commits)+2, writer=bindings[writer], bytes=payload))
    return plan, bindings, admissions, [], commits, rounds


class SampleHistoryTests(unittest.TestCase):
    def test_repeated_equal_edits_and_erasures_have_complete_lineage(self):
        args = fixture()
        result = history.account(*args)
        self.assertEqual(result['native_commits'], 16)
        self.assertEqual(len(result['lineage']), 16)
        self.assertEqual(args[2][0]['bytes'], args[2][1]['bytes'])
        self.assertEqual(args[2][4]['bytes'], args[2][6]['bytes'])
        for identity, stamp in ((4, 2**63), (6, history.bits(3.0))):
            node = next(n for n in result['scene'] if n['id']==identity)
            self.assertEqual(node['fields']['timeSamples'], dict(type='time_samples',
                value=[dict(time_bits=stamp, value=dict(type='int32', value=7))]))
        self.assertEqual(result['lineage']['w0:0']['outcome'], 'superseded')
        self.assertEqual(result['lineage']['w1:1']['outcome'], 'erased')
        self.assertEqual(result['lineage']['w0:3']['outcome'], 'no_effect')
        self.assertEqual(result['lineage']['w0:2']['displaced'], ['w1:1'])
        self.assertEqual(result['lineage']['w0:7']['displaced'], ['w1:5'])
        self.assertEqual(history.generate(42), history.generate(42))
        self.assertNotEqual(history.generate(42)['writers'], history.generate(43)['writers'])

    def test_missing_loser_repeated_commit_payload_writer_and_barrier_mutations_fail(self):
        args = fixture()
        for kind in ('lost-overwritten', 'lost-noop', 'duplicate', 'fifo', 'writer', 'payload', 'sequence', 'round'):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(args); commits = changed[4]
                if kind == 'lost-overwritten': del commits[0]
                elif kind == 'lost-noop': del commits[6]
                elif kind == 'duplicate': commits.insert(1, copy.deepcopy(commits[0]))
                elif kind == 'fifo': commits[0], commits[2] = commits[2], commits[0]
                elif kind == 'writer': commits[0]['writer'] = commits[1]['writer']
                elif kind == 'payload': commits[0]['bytes'] += 'changed'
                elif kind == 'round': commits[1], commits[2] = commits[2], commits[1]
                for i, commit in enumerate(commits): commit['sequence'] = i+2
                if kind == 'sequence': commits[0]['sequence'] = 3
                with self.assertRaises(AssertionError): history.account(*changed)

    def test_admission_occurrence_and_prebound_writer_identity_mutations_fail(self):
        for kind in ('missing', 'duplicate', 'same-id', 'ordinal', 'writer', 'bindings'):
            with self.subTest(kind=kind):
                args = copy.deepcopy(fixture())
                admissions = args[2]
                if kind == 'missing': admissions.pop()
                elif kind == 'duplicate': admissions[-1] = admissions[0]
                elif kind == 'same-id': admissions[2]['accepted_id'] = admissions[0]['accepted_id']
                elif kind == 'ordinal': admissions[2]['local_ordinal'] = 1
                elif kind == 'writer': admissions[2]['writer'] = 'alien'
                else: args[1][1] = args[1][0]
                with self.assertRaises(AssertionError): history.account(*args)

    def test_key_bits_replacement_semantics_and_scene_corruption(self):
        args = fixture(rounds=2)
        result = history.account(*args)
        field = result['scene'][3]['fields']['timeSamples']
        self.assertEqual(field['value'][0]['time_bits'], 2**63)
        for kind in ('type', 'zero', 'missing', 'metadata'):
            with self.subTest(kind=kind):
                actual = copy.deepcopy(result['scene'])
                if kind == 'type': actual[3]['fields']['timeSamples']['value'][0]['value'] = dict(type='bool',value=True)
                elif kind == 'zero': actual[3]['fields']['timeSamples']['value'][0]['time_bits'] = 0
                elif kind == 'missing': del actual[3]['fields']['timeSamples']
                else: actual[3]['path'] = '/wrong'
                with self.assertRaises(AssertionError): history.verify_scene(actual, result['scene'])

    def test_peer_rotation_boundary_is_accounted_separately(self):
        plan, bindings, admissions, peers, commits, rounds = fixture(rounds=6)
        peer = dict(edit='peer:5', kind='sample', section=6, time_bits=history.bits(1.0), value=999,
                    after_round=5, bytes='peer-original')
        peers.append(peer)
        commits.append(dict(sequence=len(commits)+2, writer='independent-peer', bytes=peer['bytes']))
        result = history.account(plan, bindings, admissions, peers, commits, rounds)
        self.assertEqual(result['peer_commits'], 1)
        self.assertEqual(result['scene'][5]['fields']['timeSamples']['value'][0]['value']['value'], 999)
        for mutated in (commits[:-1], commits+[dict(commits[-1],sequence=commits[-1]['sequence']+1)]):
            with self.assertRaises(AssertionError): history.account(plan, bindings, admissions, peers, mutated, rounds)
        shifted = [commits[-1], *commits[:-1]]
        shifted = [dict(commit, sequence=i+2) for i,commit in enumerate(shifted)]
        with self.assertRaises(AssertionError): history.account(plan, bindings, admissions, peers, shifted, rounds)
        peers[0]['edit'] = 'w0:0'
        with self.assertRaises(AssertionError): history.account(plan, bindings, admissions, peers, commits, rounds)

    def test_repeated_peer_payloads_match_distinct_boundary_occurrences(self):
        plan, bindings, admissions, peers, commits, rounds = fixture()
        peers.extend(dict(edit=f'peer:{r}', kind='sample', section=6, time_bits=history.bits(1.0), value=999,
                          after_round=r, bytes='same-peer-payload') for r in (5, 7))
        commits.insert(12, dict(writer='independent-peer', bytes='same-peer-payload'))
        commits.append(dict(writer='independent-peer', bytes='same-peer-payload'))
        commits = [dict(c, sequence=i+2) for i,c in enumerate(commits)]
        result = history.account(plan, bindings, admissions, peers, commits, rounds)
        self.assertEqual(result['peer_commits'], 2)
        self.assertEqual(result['lineage']['peer:5']['superseded_by'], 'peer:7')

    def test_wrong_encoded_losing_intent_is_rejected_before_ledger_accounting(self):
        plan = history.generate(42, clients=2)
        for index in (0, 2, 5):
            op = plan['writers'][0][index]
            value = None if index==2 else dict(type='int32', value=op['value'])
            item = dict(section=4, time_bits=op['time_bits'], value=value)
            valid = dict(is_diff=False, base_version=0, source_format='', commands=0, fields=[], samples=[item])
            history.verify_effect(op, valid)
            if index == 5:
                # A whole-map edit erases each key its writer saw, except one of identical bits, before its own.
                seen = [history.bits(-1.0), op['time_bits'] ^ 2**63, op['time_bits']]
                erases = [dict(section=4, time_bits=key, value=None) for key in sorted(seen[:2], key=history.time)]
                history.verify_effect(op, dict(valid, samples=erases + [item]), seen)
                with self.assertRaises(AssertionError): history.verify_effect(op, valid, seen)
            for fault in ('value', 'time', 'section', 'extra', 'diff', 'missing'):
                with self.subTest(index=index, fault=fault):
                    bad = copy.deepcopy(valid)
                    record = sample = bad['samples'][0]
                    if fault=='value': sample['value'] = dict(type='bool',value=True)
                    elif fault=='time': sample['time_bits'] ^= 2**63
                    elif fault=='section': record['section'] = 6
                    elif fault=='extra': bad['commands'] = 1
                    elif fault=='diff': bad['is_diff'] = True
                    else: bad['fields'] = []; bad['samples'] = []
                    with self.assertRaises(AssertionError): history.verify_effect(op, bad)
