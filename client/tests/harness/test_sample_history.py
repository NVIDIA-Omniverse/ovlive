# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loss/replay mutations must fail even when sample payloads are identical."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
import round_plans
import sample_history as history
from writer_history import WriterMismatch


def fixture(rounds=8, plan=None):
    """Each writer's rounds, each edit committed unless it changes nothing in the map its writer compiled against;
    two writers of the seed-42 plan unless `plan` is given."""
    plan = plan or history.generate(42, clients=2)
    bindings = [f'native-{writer}' for writer in range(len(plan['writers']))]
    admissions, commits, admitted = [], [], [0] * len(bindings)
    for ordinal in range(rounds):
        scene = history.account(plan, bindings, admissions, [], commits, ordinal)['scene']
        for writer in range(len(bindings)):
            op = plan['writers'][writer][ordinal]
            if not history.compiled_samples(op, history.sample_view(scene, op['section'])):
                admissions.append(dict(edit=op['edit'], writer=bindings[writer], accepted_id=None, bytes=None,
                                       noop=True))
                continue
            admitted[writer] += 1
            payload = f"{op['kind']}:{op['section']}:{op['time_bits']}:{op['value']}"
            admissions.append(dict(edit=op['edit'], writer=bindings[writer], accepted_id=ordinal*3+1,
                                   local_ordinal=admitted[writer], bytes=payload))
            commits.append(dict(sequence=len(commits)+2, writer=bindings[writer], bytes=payload))
    return plan, bindings, admissions, [], commits, rounds


def swapped(commits, first, second):
    """`commits` with the commits at indices `first` and `second` exchanged, sequences kept in ledger order, as an
    OS schedule may commit one round's writers either way."""
    commits = [dict(commit) for commit in commits]
    commits[first], commits[second] = (dict(commits[second], sequence=commits[first]['sequence']),
                                       dict(commits[first], sequence=commits[second]['sequence']))
    return commits


class SampleHistoryTests(unittest.TestCase):
    def test_compiled_samples_leave_unchanged_keys_out(self):
        bits, zero, negative = history.bits, history.bits(0.0), history.bits(-0.0)
        def compiled(kind, stamp, value, view):
            return history.compiled_samples(dict(kind=kind, section=4, time_bits=bits(stamp), value=value), view)
        # A set keeps an equivalent key's bits, so the same value changes nothing; an erase removes it whatever its
        # bits; erasing an absent key changes nothing.
        self.assertEqual(compiled('sample', -0.0, 7, {zero: 7}), [])
        self.assertEqual(compiled('sample', -0.0, 8, {zero: 7}), [(negative, 8)])
        self.assertEqual(compiled('erase-sample', -0.0, None, {zero: 7}), [(negative, None)])
        self.assertEqual(compiled('erase-sample', 1.0, None, {zero: 7}), [])
        # A whole-map edit erases the other keys; its own key is left out when its bits and value already match, and
        # a key of other bits is erased and authored again.
        self.assertEqual(compiled('sample-map', 0.0, 7, {zero: 7, bits(1.0): 3}), [(bits(1.0), None)])
        self.assertEqual(compiled('sample-map', 0.0, 7, {zero: 7}), [])
        self.assertEqual(compiled('sample-map', -0.0, 7, {zero: 7}), [(zero, None), (negative, 7)])


    def test_ledger_loss_signature_names_only_the_first_missing_edit(self):
        def lost_from(plan, rounds, position):
            """The signature when every commit from round `position` on is lost."""
            plan, bindings, admissions, peers, commits, rounds = fixture(rounds, plan)
            kept = sum(not a.get('noop') for a in admissions[:position * len(bindings)])
            with self.assertRaises(WriterMismatch) as caught:
                history.account(plan, bindings, admissions, peers, commits[:kept], rounds)
            return caught.exception
        full = lost_from(None, 8, 6)
        self.assertEqual(full.signature, dict(reason='admitted sample occurrence missing from ledger', edit='w0:6'))
        self.assertEqual(full.evidence, dict(missing=['w0:6', 'w0:7', 'w1:6', 'w1:7']))
        # Deleting the later missing round changes the evidence, never the signature.
        selected = round_plans.select(history.generate(42, clients=2), [0, 1], list(range(7)), [5])
        reduced = lost_from(selected, 7, 6)
        self.assertEqual(reduced.signature, full.signature)
        self.assertEqual(reduced.evidence, dict(missing=['w0:6', 'w1:6']))

    def test_distinct_missing_admissions_have_distinct_signatures(self):
        plan, bindings, admissions, peers, commits, rounds = fixture()
        signatures = []
        for edit in ('w0:0', 'w1:7'):
            with self.assertRaises(WriterMismatch) as caught:
                history.account(plan, bindings, [a for a in admissions if a['edit'] != edit], peers, commits, rounds)
            signatures.append(caught.exception.signature)
            self.assertEqual(caught.exception.evidence, dict(missing=[edit], repeated=[], unplanned=[]))
        self.assertEqual(signatures, [dict(reason='missing or duplicate sample admission', edit=edit)
                                      for edit in ('w0:0', 'w1:7')])

    def scene_signature(self, plan, rounds, section, *, swap=None, key=None):
        """The signature when one sample of `section` (the first, or the one at time bits `key`) reads another value
        than the oracle's; `swap` exchanges two commits of one round first."""
        plan, bindings, admissions, peers, commits, rounds = fixture(rounds, plan)
        if swap:
            commits = swapped(commits, *swap)
        result = history.account(plan, bindings, admissions, peers, commits, rounds)
        scene = copy.deepcopy(result['scene'])
        samples = next(item for item in scene if item['id'] == section)['fields']['timeSamples']['value']
        next(sample for sample in samples if key is None or sample['time_bits'] == key)['value']['value'] = -99
        with self.assertRaises(WriterMismatch) as caught:
            history.verify_scene(scene, result['scene'], result['lineage'])
        return caught.exception

    def test_scene_mismatch_names_the_latest_emitting_edit_in_plan_order(self):
        reason = 'scene differs from independent sample intents'
        first = self.scene_signature(None, 8, 4)
        self.assertEqual((first.signature, first.evidence), (dict(reason=reason, edit='w0:7'),
                                                             dict(path='/P/Child.value')))
        self.assertEqual(self.scene_signature(None, 8, 6).signature, dict(reason=reason, edit='w0:6'))
        # Deleting the named edit's writer changes the signature, so a reduction cannot keep that deletion.
        without = round_plans.select(history.generate(42, clients=2), [1], list(range(16)), [5, 11])
        self.assertEqual(self.scene_signature(without, 8, 6).signature, dict(reason=reason, edit='w1:6'))

    def test_scene_mismatch_signature_ignores_the_commit_order_within_a_round(self):
        _, _, _, _, commits, _ = fixture()
        self.assertEqual([commit['writer'] for commit in commits[-2:]], ['native-0', 'native-1'])  # Round 7.
        last = len(commits) - 1
        self.assertEqual(self.scene_signature(None, 8, 4, swap=(last - 1, last)).signature,
                         self.scene_signature(None, 8, 4).signature)

    def test_scene_mismatch_names_only_edits_that_emitted_the_key(self):
        # Seed 5's w1:13 is a whole-map edit at time 0 that leaves that key as it was and erases the others, so it
        # emits no record there; the key's signature is the same before and after round 13.
        plan = history.generate(5, clients=2)
        before, after = (self.scene_signature(plan, rounds, 6, key=0) for rounds in (13, 14))
        self.assertEqual(before.signature, dict(reason='scene differs from independent sample intents', edit='w0:12'))
        self.assertEqual(after.signature, before.signature)

    def replica_signature(self, changes, *, swap=None):
        """The check_replicas mismatch of two readers whose scenes start as the oracle's scene after round 7, the
        service's, with `changes` (reader position to a list of (view, section, time bits, value)) applied; `swap`
        exchanges two commits of one round first."""
        plan, bindings, admissions, peers, commits, rounds = fixture()
        if swap:
            commits = swapped(commits, *swap)
        result = history.account(plan, bindings, admissions, peers, commits, rounds)
        reports = [dict(authoritative=copy.deepcopy(result['scene']), visible=copy.deepcopy(result['scene']))
                   for _ in range(2)]
        for position, edits in changes.items():
            for view, section, key, value in edits:
                node = next(item for item in reports[position][view] if item['id'] == section)
                next(sample for sample in node['fields']['timeSamples']['value']
                     if sample['time_bits'] == key)['value']['value'] = value
        with self.assertRaises(WriterMismatch) as caught:
            round_plans.check_replicas(plan, reports, result['scene'],
                                       lambda actual, expected, path: history.scene_edit(actual, expected, path,
                                                                                         result['lineage']))
        return caught.exception

    def test_replica_mismatch_ignores_the_reader_value_and_commit_order(self):
        """A reader whose round-6 sample of section 6 differs from the service names w0:6 whichever reader it is,
        whatever value it holds and whichever round-6 writer committed last; the path and nodes are evidence."""
        reason = 'writer authoritative scene differs from the service ledger'
        three = history.bits(3.0)
        _, _, _, _, commits, _ = fixture()
        round_six = [i for i, commit in enumerate(commits) if commit['bytes'].startswith('sample:6:')]
        self.assertEqual(len(round_six), 2)
        both = ('authoritative', 'visible')
        variants = [self.replica_signature({position: [(view, 6, three, value) for view in both]}, swap=swap)
                    for position in (0, 1) for value in (-1, 99) for swap in (None, round_six)]
        self.assertEqual({repr(v.signature) for v in variants}, {repr(dict(reason=reason, edit='w0:6'))})
        first = variants[0]
        self.assertEqual([(r['position'], r['path']) for r in first.evidence['readers']], [(0, '/Q.other')])
        self.assertEqual(first.evidence['readers'][0]['actual']['fields']['timeSamples']['value'][0]['value']['value'],
                         -1)

    def test_replica_mismatch_names_the_first_edit_in_plan_order_over_every_reader(self):
        """Readers that differ at different keys name the edit first in plan order, not the first reader's; a
        difference in the visible scene alone has its own reason."""
        three, zero = history.bits(3.0), history.bits(-0.0)
        late = [('authoritative', 4, zero, -1)]  # Section 4's key at time 0, which w0:7 last emitted.
        early = [('authoritative', 6, three, -1)]  # w0:6's key.
        for changes in ({0: late, 1: early}, {0: early, 1: late}):
            self.assertEqual(self.replica_signature(changes).signature,
                             dict(reason='writer authoritative scene differs from the service ledger', edit='w0:6'))
        self.assertEqual(self.replica_signature({0: late}).signature['edit'], 'w0:7')
        self.assertEqual(self.replica_signature({1: [('visible', 6, three, -1)]}).signature,
                         dict(reason='writer visible scene differs from the service ledger', edit='w0:6'))

    def test_matching_replicas_pass(self):
        plan, bindings, admissions, peers, commits, rounds = fixture()
        result = history.account(plan, bindings, admissions, peers, commits, rounds)
        scene = result['scene']
        round_plans.check_replicas(plan, [dict(authoritative=scene, visible=scene)] * 2, scene,
                                   lambda *_: self.fail('a matching scene is never attributed'))

    def test_sequence_gap_names_the_first_lost_admitted_edit(self):
        plan, bindings, admissions, peers, commits, rounds = fixture()
        by_edit = {admission['edit']: admission for admission in admissions}

        def lost(commits, edit):
            index = next(i for i, commit in enumerate(commits) if commit['bytes'] == by_edit[edit]['bytes'] and
                         commit['writer'] == by_edit[edit]['writer'])
            with self.assertRaises(WriterMismatch) as caught:
                history.account(plan, bindings, admissions, peers, commits[:index] + commits[index + 1:], rounds)
            self.assertEqual(caught.exception.evidence, dict(gap=2))
            return caught.exception.signature
        reason = 'missing sample sequence'
        self.assertEqual(lost(commits, 'w0:0'), dict(reason=reason, edit='w0:0'))
        # Round 0's commits in the other order: the commit after the gap differs, the signature does not.
        self.assertEqual(lost(swapped(commits, 0, 1), 'w0:0'), dict(reason=reason, edit='w0:0'))
        # The commit after w1:0's is round 2's; round 1's no-op edits, which commit nothing, are never named.
        self.assertEqual(lost(commits, 'w1:0'), dict(reason=reason, edit='w1:0'))

    def stopped(self, plan, rounds, edit, *, swap=None, **final):
        """The classification of a round whose ledger lacks `edit`'s commit, as when the service acknowledges it
        without committing it and the round's later commits take its sequence; `swap` exchanges two commits first."""
        plan, bindings, admissions, peers, commits, rounds = fixture(rounds, plan)
        lost = next(a for a in admissions if a['edit'] == edit)
        kept = [c for c in commits if (c['bytes'], c['writer']) != (lost['bytes'], lost['writer'])]
        kept = [dict(commit, sequence=index + 2) for index, commit in enumerate(swapped(kept, *swap) if swap else kept)]
        conditions = dict(in_flight=0, refused=0, stopped=[]) | final
        with self.assertRaises(AssertionError) as caught:
            round_plans.stopped_short(lambda: history.account(plan, bindings, admissions, peers, kept, rounds),
                                      frontier=len(commits) + 1, sequence=len(kept) + 1, **conditions)
        return caught.exception

    def test_stopped_round_names_the_edit_the_ledger_lacks(self):
        four = history.generate(42, clients=4)
        failure = self.stopped(four, 7, 'w1:6')
        self.assertEqual((failure.signature, failure.evidence['frontier'] - failure.evidence['sequence']),
                         (dict(reason='round commits stopped short of its frontier',
                               cause='admitted sample occurrence missing from ledger', edit='w1:6'), 1))
        self.assertEqual(failure.evidence['accounting'], dict(missing=['w1:6']))
        # The round's other commits in another order, as an OS schedule may commit them, keep the signature.
        _, _, _, _, commits, _ = fixture(7, four)
        last = len(commits) - 2  # The last commit of the ledger without w1:6.
        self.assertEqual(self.stopped(four, 7, 'w1:6', swap=(last - 1, last)).signature, failure.signature)
        # Deleting an earlier round changes the frontier, which is evidence, never the signature.
        selected = round_plans.select(four, [0, 1, 2, 3], [0, 1, 2, 3, 5, 6], [5])
        reduced = self.stopped(selected, 6, 'w1:6')
        self.assertEqual(reduced.signature, failure.signature)
        self.assertNotEqual(reduced.evidence['frontier'], failure.evidence['frontier'])

    def test_stopped_round_without_a_final_ledger_has_no_signature(self):
        # A call in flight, a retryable refusal or a stopped writer process leaves the ledger open to change, so the
        # stall is a watchdog or crash matter that no trial can match.
        for final in (dict(in_flight=1), dict(refused=1), dict(stopped=[0])):
            with self.subTest(final=final):
                failure = self.stopped(None, 7, 'w0:6', **final)
                self.assertNotIsInstance(failure, WriterMismatch)
        plan, bindings, admissions, peers, commits, rounds = fixture(7)
        with self.assertRaises(AssertionError) as caught:
            round_plans.stopped_short(lambda: history.account(plan, bindings, admissions, peers, commits, rounds),
                                      frontier=len(commits) + 2, sequence=len(commits) + 1, in_flight=0, refused=0,
                                      stopped=[])
        self.assertNotIsInstance(caught.exception, WriterMismatch)  # Accounting finds no lost edit.

    def test_changed_commit_bytes_name_the_edit(self):
        plan, bindings, admissions, peers, commits, rounds = fixture()
        changed = copy.deepcopy(commits)
        changed[0]['bytes'] = 'changed'
        with self.assertRaises(WriterMismatch) as caught:
            history.account(plan, bindings, admissions, peers, changed, rounds)
        self.assertEqual(caught.exception.signature, dict(reason='sample FIFO or original bytes changed', edit='w0:0'))

    def test_selection_keeps_seeded_edits_release_orders_and_rotations(self):
        original = history.generate(42, clients=4, rounds=12)
        plan = round_plans.select(original, [1, 3], [2, 5, 6, 11], [5, 11])
        self.assertIs(history.validate(plan), plan)
        self.assertEqual(history.origin(plan), original)
        self.assertEqual((plan['client_count'], plan['rounds'], plan['rotation_after']), (2, 4, [1, 3]))
        self.assertEqual(plan['writers'], [[original['writers'][w][o] for o in (2, 5, 6, 11)] for w in (1, 3)])
        self.assertEqual(plan['release_orders'], [[{1: 0, 3: 1}[w] for w in original['release_orders'][o] if w in (1, 3)]
                                                  for o in (2, 5, 6, 11)])
        self.assertEqual(plan['origin'], dict(client_count=4, rounds=12))

    def test_selection_rejects_any_plan_its_seed_does_not_reproduce(self):
        original = history.generate(42, clients=4, rounds=12)
        selected = round_plans.select(original, [1, 3], [2, 5, 6], [5])
        def value(plan): plan['writers'][0][1]['value'] = 99
        def order(plan): plan['release_orders'][0].reverse()
        def rotation(plan): plan['rotation_after'] = [2]
        def writer(plan): plan['writers'].reverse()
        def population(plan): plan['origin']['client_count'] = 5
        def unselected(plan): del plan['origin']
        def defect(plan): plan['test_service_fault'] = 'corrupt-everything'
        for mutation in (value, order, rotation, writer, population, unselected, defect):
            with self.subTest(mutation=mutation.__name__):
                plan = copy.deepcopy(selected)
                mutation(plan)
                with self.assertRaises(ValueError):
                    history.validate(plan)
        history.validate(dict(selected, test_service_fault='corrupt-sample-w0:6'))

    def test_repeated_equal_edits_and_erasures_have_complete_lineage(self):
        args = fixture()
        result = history.account(*args)
        # Rounds 1 (a set of +0 to the -0 key's value) and 3 (a repeated erase) change nothing and admit nothing.
        self.assertEqual(result['native_commits'], 12)
        self.assertEqual(len(result['lineage']), 16)
        self.assertEqual([a['edit'] for a in args[2] if a.get('noop')], ['w0:1', 'w1:1', 'w0:3', 'w1:3'])
        self.assertEqual(args[2][0]['bytes'], args[2][1]['bytes'])
        self.assertEqual(args[2][4]['bytes'], args[2][5]['bytes'])
        for identity, stamp in ((4, 2**63), (6, history.bits(3.0))):
            node = next(n for n in result['scene'] if n['id']==identity)
            self.assertEqual(node['fields']['timeSamples'], dict(type='time_samples',
                value=[dict(time_bits=stamp, value=dict(type='int32', value=7))]))
        self.assertEqual(result['lineage']['w0:0']['outcome'], 'superseded')
        self.assertEqual(result['lineage']['w1:0']['outcome'], 'erased')
        self.assertEqual(result['lineage']['w1:1']['outcome'], 'no_effect')
        self.assertEqual(result['lineage']['w0:3']['outcome'], 'no_effect')
        self.assertEqual(result['lineage']['w0:2']['displaced'], ['w1:0'])
        self.assertEqual(result['lineage']['w0:7']['displaced'], ['w1:5'])
        self.assertEqual(history.generate(42), history.generate(42))
        self.assertNotEqual(history.generate(42)['writers'], history.generate(43)['writers'])

    def test_missing_loser_repeated_commit_payload_writer_and_barrier_mutations_fail(self):
        args = fixture()
        for kind in ('lost-overwritten', 'lost-noop', 'duplicate', 'fifo', 'writer', 'payload', 'sequence', 'round'):
            with self.subTest(kind=kind):
                changed = copy.deepcopy(args); commits = changed[4]
                if kind == 'lost-overwritten': del commits[0]
                elif kind == 'lost-noop': del commits[3]  # w1:2, whose erase found the key already gone.
                elif kind == 'duplicate': commits.insert(1, copy.deepcopy(commits[0]))
                elif kind == 'fifo': commits[0], commits[2] = commits[2], commits[0]
                elif kind == 'writer': commits[0]['writer'] = commits[1]['writer']
                elif kind == 'payload': commits[0]['bytes'] += 'changed'
                elif kind == 'round': commits[1], commits[2] = commits[2], commits[1]
                for i, commit in enumerate(commits): commit['sequence'] = i+2
                if kind == 'sequence': commits[0]['sequence'] = 3
                with self.assertRaises(AssertionError): history.account(*changed)

    def test_admission_occurrence_and_prebound_writer_identity_mutations_fail(self):
        for kind in ('missing', 'duplicate', 'same-id', 'ordinal', 'writer', 'bindings', 'noop-id', 'noop-dropped'):
            with self.subTest(kind=kind):
                args = copy.deepcopy(fixture())
                admissions = args[2]
                if kind == 'missing': admissions.pop()
                elif kind == 'duplicate': admissions[-1] = admissions[0]
                elif kind == 'same-id': admissions[4]['accepted_id'] = admissions[0]['accepted_id']
                elif kind == 'ordinal': admissions[4]['local_ordinal'] = 1
                elif kind == 'writer': admissions[2]['writer'] = 'alien'
                elif kind == 'noop-id': admissions[2]['accepted_id'] = 99
                elif kind == 'noop-dropped':
                    # An edit that changes its writer's map must not be accounted as a no-op.
                    removed = admissions[4]
                    admissions[4] = dict(edit=removed['edit'], writer=removed['writer'], accepted_id=None,
                                         bytes=None, noop=True)
                    args[4][:] = [dict(c, sequence=i+2) for i, c in
                                  enumerate(c for c in args[4] if c['bytes'] is not removed['bytes'])]
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

    def test_lost_peers_are_ranked_by_round_not_name(self):
        plan, bindings, admissions, peers, commits, rounds = fixture(rounds=12)
        peers.extend(dict(edit=f'peer:{r}', kind='sample', section=6, time_bits=history.bits(1.0), value=999,
                          after_round=r, bytes=f'peer-{r}') for r in (5, 11))
        with self.assertRaises(WriterMismatch) as caught:
            history.account(plan, bindings, admissions, peers, commits, rounds)
        self.assertEqual(caught.exception.signature,
                         dict(reason='admitted sample occurrence missing from ledger', edit='peer:5'))
        self.assertEqual(caught.exception.evidence, dict(missing=['peer:11', 'peer:5']))
        self.assertEqual(round_plans.first_edit(plan, {'peer:11', 'peer:5', 'w1:6'}, {'peer:5': 5, 'peer:11': 11}),
                         'peer:5')
        self.assertEqual(round_plans.first_edit(plan, {'peer:5', 'w1:6', 'w0:5'}, {'peer:5': 5}), 'w0:5')
        self.assertEqual(round_plans.first_edit(plan, {'unplanned', 'w1:6'}), 'w1:6')

    def test_sequence_gap_ranks_lost_peers_by_round_not_name(self):
        plan, bindings, admissions, peers, commits, rounds = fixture(rounds=12)
        peers.extend(dict(edit=f'peer:{r}', kind='sample', section=6, time_bits=history.bits(1.0), value=999,
                          after_round=r, bytes=f'peer-{r}') for r in (5, 11))
        after_five = sum(1 for a in admissions if not a.get('noop') and int(a['edit'].split(':')[1]) <= 5)
        commits.insert(after_five, dict(writer='independent-peer', bytes='peer-5'))
        commits.append(dict(writer='independent-peer', bytes='peer-11'))
        commits = [dict(c, sequence=i+2) for i, c in enumerate(commits)]
        history.account(plan, bindings, admissions, peers, commits, rounds)
        # Both peer commits are lost; the commit after peer:5's keeps its sequence, so the ledger has a gap.
        lost = [c for c in commits if c['writer'] != 'independent-peer']
        with self.assertRaises(WriterMismatch) as caught:
            history.account(plan, bindings, admissions, peers, lost, rounds)
        self.assertEqual(caught.exception.signature, dict(reason='missing sample sequence', edit='peer:5'))

    def test_repeated_peer_payloads_match_distinct_boundary_occurrences(self):
        plan, bindings, admissions, peers, commits, rounds = fixture()
        peers.extend(dict(edit=f'peer:{r}', kind='sample', section=6, time_bits=history.bits(1.0), value=999,
                          after_round=r, bytes='same-peer-payload') for r in (5, 7))
        after_five = sum(1 for a in admissions if not a.get('noop') and int(a['edit'].split(':')[1]) <= 5)
        commits.insert(after_five, dict(writer='independent-peer', bytes='same-peer-payload'))
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
                seen = {history.bits(-1.0): 0, op['time_bits'] ^ 2**63: 0, op['time_bits']: 0}
                erases = [dict(section=4, time_bits=key, value=None)
                          for key in sorted(list(seen)[:2], key=history.time)]
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
