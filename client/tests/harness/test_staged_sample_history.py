# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The staged-samples case accepts only exact planned payloads, accounts each layer's ledger with signatures that
survive deleting other samples, writers and rounds, reduces by selections of its seeded plan and reads the diagnostics
summary on every platform."""
import copy
import unittest

import scene_graph_model as graph_model
import scene_graph_wire as wire
from scene_graph_wire import Command, Delta, Field, Sample, Specifier, Token
import staged_sample_history as history
from staged_sample_history import listed_sessions, planned_edit
from writer_history import WriterMismatch
from writer_reduction import staged_domain

PRIM, ATTRIBUTE = 900, 901


def encoded(delta):
    """`delta` through the repository's real wire encoder and decoder, as a commit's payload reaches the oracle."""
    return wire.decode(wire.encode(delta))


SET = Delta(fields=(Field(3, 'default', 11, set_order=1),))
DEFINE = Delta(commands=(Command(1, PRIM, 1, 'W0', graph_model.PRIM), Command(1, ATTRIBUTE, PRIM, 'value', graph_model.ATTRIBUTE)),
               fields=(Field(PRIM, 'specifier', Specifier(0), set_order=4), Field(PRIM, 'typeName', Token('Scope'), set_order=5),
                       Field(ATTRIBUTE, 'typeName', Token('int'), set_order=8), Field(ATTRIBUTE, 'default', 22, set_order=9)))


class StagedSampleHistoryTests(unittest.TestCase):
    def setUp(self):
        self.planned = {11: ('native', 0, dict(path='/Example', kind='set', value=11)),
                        22: ('writer-0', 0, dict(path='/W0', kind='define', value=22))}
        self.attributes = {'/Example': 3, '/W0': ATTRIBUTE}

    def rejects(self, delta):
        planned = copy.deepcopy(self.planned)
        with self.assertRaises(AssertionError):
            planned_edit(encoded(delta), planned, self.attributes)
        self.assertEqual(planned, self.planned, 'a rejected commit consumed its planned edit')

    def test_exact_set_and_define_payloads_carry_their_planned_edit(self):
        self.assertEqual(planned_edit(encoded(SET), self.planned, self.attributes)[:2], ('native', 0))
        self.assertEqual(planned_edit(encoded(DEFINE), self.planned, self.attributes)[:2], ('writer-0', 0))
        self.assertEqual(self.planned, {})

    def test_a_planned_value_with_any_unplanned_mutation_is_rejected(self):
        # The planned /Example.value with an unplanned /Example.typeName change, the shape a defect that also
        # rewrites metadata takes; the converged scenes project neither field, so only the commit oracle sees it.
        self.rejects(Delta(fields=(*SET.fields, Field(2, 'typeName', Token('UNPLANNED_TYPE'), set_order=2))))
        self.rejects(Delta(fields=SET.fields, samples=(Sample(3, 1.0, 5),)))
        self.rejects(Delta(commands=(Command(2, 7),), fields=SET.fields))
        self.rejects(Delta(fields=(Field(4, 'default', 11, set_order=1),)))  # Another prim's attribute.
        self.rejects(Delta(commands=DEFINE.commands, fields=(*DEFINE.fields, Field(PRIM, 'kind', Token('component')))))
        self.rejects(Delta(commands=(*DEFINE.commands, Command(1, 902, PRIM, 'extra', graph_model.ATTRIBUTE)),
                           fields=DEFINE.fields))
        self.rejects(Delta(commands=DEFINE.commands, fields=DEFINE.fields[1:]))
        self.rejects(Delta(commands=DEFINE.commands, fields=DEFINE.fields, samples=(Sample(ATTRIBUTE, 0.0, 1),)))
        self.rejects(Delta(fields=(Field(3, 'default', 12, set_order=1),)))  # No planned edit has this value.
        self.rejects(Delta(fields=(*SET.fields, Field(PRIM, 'default', 22))))  # Two planned values in one commit.

    def test_diagnostics_summary_reads_lf_and_crlf_lines(self):
        for ending in (b'\n', b'\r\n'):
            with self.subTest(ending=ending):
                stderr = ending.join([b'endpoint_status query=0 result=0 state=2 incident=0',
                                      b'list_sessions result=0 count=2', b'sample_result=0'])
                self.assertEqual(listed_sessions(stderr + ending), [(0, 2)])
        self.assertEqual(listed_sessions(b'list_sessions result=0 count=2 extra\r\n'), [])


# The section IDs of /Example.value and of writer i's prim /W<i> and its value attribute in a synthetic layer.
EXAMPLE = 3


def sections(writer):
    return 900 + 2 * writer, 901 + 2 * writer


def payload(op):
    """The exact payload the participant authoring `op` uploads."""
    if op['path'] == '/Example':
        return wire.encode(Delta(fields=(Field(EXAMPLE, 'default', op['value'], set_order=1),)))
    prim, attribute = sections(op['writer'])
    if op['kind'] == 'set':
        return wire.encode(Delta(fields=(Field(attribute, 'default', op['value'], set_order=1),)))
    return wire.encode(Delta(
        commands=(Command(1, prim, 1, op['path'][1:], graph_model.PRIM),
                  Command(1, attribute, prim, 'value', graph_model.ATTRIBUTE)),
        fields=(Field(prim, 'specifier', Specifier(0)), Field(prim, 'typeName', Token('Scope')),
                Field(attribute, 'typeName', Token('int')), Field(attribute, 'default', op['value']))))


def ledger(plan, layer, *, reverse=False):
    """The layer's commits after its creation, each participant's edit of a round in one commit under its own wire
    identity, in participant order or, with `reverse`, the reverse within each round; returns the commits, the
    attribute sections of every prim and the commit count after each round."""
    members = history.participants(plan, layer)
    commits, boundaries = [], []
    for position in range(plan['rounds']):
        ops = [(name, ops[position]) for name, ops in members]
        for name, op in (reversed(ops) if reverse else ops):
            commits.append((len(commits) + 2, f'wire-{name}', payload(op)))
        boundaries.append(len(commits))
    attributes = {'/Example': EXAMPLE}
    attributes.update({f'/W{edits[0]["writer"]}': sections(edits[0]['writer'])[1] for edits in plan['writers']
                       if any(op['kind'] == 'define' for op in edits)})
    return commits, attributes, boundaries


def define_round(plan, writer):
    return next(op['ordinal'] for op in plan['writers'][writer] if op['kind'] == 'define')


class StagedSamplePlanTests(unittest.TestCase):
    def test_a_generated_plan_is_seeded_with_distinct_values_and_defines_before_sets(self):
        plan = history.generate(42, 4, 8)
        self.assertEqual(plan, history.generate(42, 4, 8))
        self.assertNotEqual(plan, history.generate(43, 4, 8))
        self.assertIs(history.validate(plan), plan)
        values = [op['value'] for _, ops in history.participants(plan) for op in ops]
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(plan['layers'], ['native', 'python', 'native', 'python'])
        for edits in plan['writers']:
            kinds = [op['kind'] for op in edits if op['path'] != '/Example']
            self.assertEqual(kinds, ['define'] + ['set'] * (len(kinds) - 1) if kinds else [])

    def test_a_selection_keeps_seeded_edits_and_refuses_a_set_without_its_define(self):
        plan = history.generate(42, 4, 8)
        writer = next(w for w, edits in enumerate(plan['writers'])
                      if any(op['kind'] == 'set' and op['path'] != '/Example' for op in edits))
        defined = define_round(plan, writer)
        later = next(op['ordinal'] for op in plan['writers'][writer]
                     if op['kind'] == 'set' and op['path'] != '/Example')
        layer = plan['layers'][writer]
        selected = history.reselect(plan, [layer], [writer], [defined, later])
        self.assertIs(history.validate(selected), selected)
        self.assertEqual(selected['origin'], dict(client_count=4, rounds=8))
        self.assertEqual(history.edits(selected), [f'{layer}:{defined}', f'w{writer}:{defined}', f'{layer}:{later}',
                                                   f'w{writer}:{later}'])
        with self.assertRaisesRegex(ValueError, 'no retained round defines'):
            history.reselect(plan, [layer], [writer], [later])
        for invalid in (([], [], [0]), (['python', 'native'], [], [0]), ([layer], [writer], []),
                        ([layer], [writer], [8]), ([layer], [writer, writer], [defined]),
                        ([next(name for name in history.SAMPLES if name != layer)], [writer], [defined])):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                history.reselect(plan, *invalid)
        for tampered in (dict(selected, rounds=3), dict(selected, test_service_fault='corrupt-w9:9'),
                         dict(selected, scenario='other'), dict(selected, layers=['python'])):
            with self.subTest(tampered=tampered), self.assertRaises(ValueError):
                history.validate(tampered)
        changed = copy.deepcopy(selected)
        changed['samples'][layer][0]['value'] += 1
        with self.assertRaisesRegex(ValueError, 'seeded origin'):
            history.validate(changed)

    def test_the_domain_deletes_samples_with_their_writers_writers_and_rounds(self):
        plan = dict(history.generate(42, 2, 4), test_service_fault=history.CORRUPT)
        domain = staged_domain()
        self.assertEqual(domain.units(plan), ['sample:native', 'sample:python', 'writer:0', 'writer:1', 'round:0',
                                              'round:1', 'round:2', 'round:3'])
        alone = domain.remove(plan, {'sample:python'})
        self.assertEqual((domain.units(alone), alone['layers'], alone['test_service_fault']),
                         (['sample:native', 'writer:0', 'round:0', 'round:1', 'round:2', 'round:3'], ['native'],
                          history.CORRUPT))
        self.assertEqual(domain.units(domain.remove(plan, {'writer:0', 'writer:1'})),
                         ['sample:native', 'sample:python', 'round:0', 'round:1', 'round:2', 'round:3'])
        self.assertIsNone(domain.remove(plan, {'sample:native', 'sample:python'}))
        self.assertIsNone(domain.remove(plan, {f'round:{o}' for o in range(4)}))
        # Writer 0 sets /W0 in round 3, which the define of round 1 creates.
        self.assertEqual((define_round(plan, 0), plan['writers'][0][3]['kind']), (1, 'set'))
        self.assertIsNone(domain.remove(plan, {'round:1'}))
        self.assertEqual(domain.retained(alone), [f'{name}:{o}' for o in range(4) for name in ('native', 'w0')])


class StagedSampleAccountingTests(unittest.TestCase):
    def setUp(self):
        self.plan = history.generate(42, 4, 4)

    def account(self, plan, layer, commits, attributes, boundaries):
        return history.account(plan, layer, commits, attributes, boundaries[:-1], len(boundaries))

    def signature(self, plan, layer, commits, attributes, boundaries):
        with self.assertRaises(WriterMismatch) as raised:
            self.account(plan, layer, commits, attributes, boundaries)
        return raised.exception.signature

    def test_a_clean_ledger_in_any_order_within_rounds_makes_the_last_value_of_each_prim(self):
        for layer in history.SAMPLES:
            commits, attributes, boundaries = ledger(self.plan, layer)
            scene = self.account(self.plan, layer, commits, attributes, boundaries)
            final = {}
            for _, ops in history.participants(self.plan, layer):
                for op in ops:
                    final[op['path']] = op['value']
            # The last commit of /Example in the round wins, in commit order.
            for position in range(self.plan['rounds']):
                for _, ops in history.participants(self.plan, layer):
                    if ops[position]['path'] == '/Example':
                        last = ops[position]['value']
            self.assertEqual(scene['values']['/Example'], last)
            self.assertEqual(scene['paths'], sorted({'/Example', *final}))
            reversed_commits, _, _ = ledger(self.plan, layer, reverse=True)
            self.assertEqual(self.account(self.plan, layer, reversed_commits, attributes, boundaries)['paths'],
                             scene['paths'])
            # After its first round, the ledger of that round alone accounts.
            self.account(self.plan, layer, commits[:boundaries[0]], attributes, boundaries[:1])

    def corrupt(self, commits, value):
        """`commits` with the commit carrying `value` committing -1 in its place."""
        def changed(payload):
            delta = wire.decode(payload)
            return wire.encode(Delta(commands=delta.commands, fields=tuple(
                Field(f.section, f.key, -1 if f.value == value else f.value, f.set_order) for f in delta.fields)))
        return [(sequence, writer, changed(payload)) for sequence, writer, payload in commits]

    def test_a_corrupted_value_names_the_layer_and_the_path_whatever_else_the_plan_retains(self):
        edit = self.plan['writers'][0][2]
        expected = dict(reason='commit carries no single planned edit', layer='native', path=edit['path'])
        commits, attributes, boundaries = ledger(self.plan, 'native')
        self.assertEqual(self.signature(self.plan, 'native', self.corrupt(commits, edit['value']), attributes,
                                        boundaries), expected)
        rounds = sorted({2, define_round(self.plan, 0)}) if edit['path'] != '/Example' else [2]
        smaller = history.reselect(self.plan, ['native'], [0], rounds)
        commits, attributes, boundaries = ledger(smaller, 'native', reverse=True)
        self.assertEqual(self.signature(smaller, 'native', self.corrupt(commits, edit['value']), attributes,
                                        boundaries), expected)
        # An unplanned path names only the layer.
        stray = (99, 'wire-native', wire.encode(Delta(fields=(Field(77, 'default', 5),))))
        self.assertEqual(self.signature(smaller, 'native', [*commits, stray], attributes, boundaries),
                         dict(reason='commit writes no path the layer authors', layer='native'))
        garbage = (99, 'wire-native', b'not a delta')
        self.assertEqual(self.signature(smaller, 'native', [*commits, garbage], attributes, boundaries),
                         dict(reason='commit writes no path the layer authors', layer='native'))

    def test_a_repeated_missing_or_misplaced_edit_names_the_first_in_plan_order(self):
        commits, attributes, boundaries = ledger(self.plan, 'python')
        first = history.participants(self.plan, 'python')
        self.assertEqual(self.signature(self.plan, 'python', [*commits, commits[1]], attributes, boundaries),
                         dict(reason='duplicate committed edit', edit=first[1][1][0]['edit']))
        self.assertEqual(self.signature(self.plan, 'python', commits[:2] + commits[3:], attributes, boundaries),
                         dict(reason='admitted edit missing from service ledger', edit=first[2][1][0]['edit']))
        # The python sample's round-0 and round-1 edits trade places across the round boundary.
        swapped = list(commits)
        swapped[0], swapped[boundaries[0]] = swapped[boundaries[0]], swapped[0]
        self.assertEqual(self.signature(self.plan, 'python', swapped, attributes, boundaries),
                         dict(reason='edit committed outside its round', edit='python:0'))

    def test_a_changed_or_shared_wire_identity_names_the_first_participant(self):
        commits, attributes, boundaries = ledger(self.plan, 'native')
        changed = list(commits)
        sequence, _, data = changed[-1]
        changed[-1] = (sequence, 'wire-other', data)
        owner = history.participants(self.plan, 'native')[-1][0]
        self.assertEqual(self.signature(self.plan, 'native', changed, attributes, boundaries),
                         dict(reason='participant identity changed within workload', participant=owner))
        shared = [(sequence, 'wire-native' if writer == f'wire-{owner}' else writer, data)
                  for sequence, writer, data in commits]
        self.assertEqual(self.signature(self.plan, 'native', shared, attributes, boundaries),
                         dict(reason='participants share a wire writer identity', participant='native'))
        anonymous = [(sequence, '' if writer == 'wire-native' else writer, data) for sequence, writer, data in commits]
        self.assertEqual(self.signature(self.plan, 'native', anonymous, attributes, boundaries),
                         dict(reason='participant identity changed within workload', participant='native'))


if __name__ == '__main__':
    unittest.main()
