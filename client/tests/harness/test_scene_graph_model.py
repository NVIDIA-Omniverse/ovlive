# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent graph-contract examples and reproducible identity histories."""
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
import scene_graph_model as graph
from scene_graph_wire import Command as C, Field as F, Sample, Delta, Paths, TimeSamples, encode


def apply(model, commands=(), fields=(), **kwargs):
    return model.apply(encode(Delta(commands=tuple(commands), fields=tuple(fields), **kwargs)))


def by_id(model):
    return {node['id']: node for node in model.state}


class GraphModelTests(unittest.TestCase):
    def assertRejected(self, model, delta, kind='unsupported'):
        before, history, sequence, source_format = model.state, model.history, model.sequence, model.source_format
        with self.assertRaises(graph.GraphError) as caught:
            model.apply(encode(delta), 999)
        self.assertEqual(caught.exception.kind, kind)
        self.assertEqual((model.state, model.history, model.sequence, model.source_format),
                         (before, history, sequence, source_format))

    def test_source_format_reset_preserve_replace_and_atomic_failure(self):
        model = graph.Model()
        apply(model, is_diff=True, source_format="usda")
        apply(model, fields=[F(1, "value", 7)])
        self.assertEqual(model.source_format, "usda")
        apply(model, is_diff=True, base_version=1)
        self.assertEqual(model.source_format, "usda")
        apply(model, is_diff=True, base_version=2, source_format="usdc")
        self.assertEqual(model.source_format, "usdc")
        for base in (0, 3):
            self.assertRejected(model, Delta(True, base, (C(5, 2, 1, 'A', 6, 0), C(5, 2, 1, 'B', 6, 1)),
                                             source_format="rejected"))
        apply(model, is_diff=True)
        self.assertEqual((model.state, model.source_format), ([graph.root()], ""))
        apply(model, is_diff=True, source_format="new")
        self.assertEqual(model.source_format, "new")

    def test_recursive_move_and_delete_preserve_identity_and_values(self):
        model = graph.Model()
        apply(model, [C(1, 2, 1, 'P', 6), C(1, 3, 2, 'C', 6), C(1, 4, 3, 'x', 1),
                      C(1, 5, 1, 'Q', 6), C(1, 6, 2, 'r', 8)], [F(4, 'v', -17), F(6, 'v', 2.5)])
        old = model.state
        apply(model, [C(3, 2, 5, 'Moved', old_parent=1)])
        state = by_id(model)
        self.assertEqual((state[2]['path'], state[3]['path'], state[4]['path'], state[6]['path']),
                         ('/Q/Moved', '/Q/Moved/C', '/Q/Moved/C.x', '/Q/Moved.r'))
        self.assertEqual((state[4]['fields'], state[6]['fields']), ({'v': -17}, {'v': 2.5}))
        self.assertEqual(old[1]['path'], '/P')
        apply(model, [C(2, 2, 5)])
        self.assertEqual(set(by_id(model)), {1, 5})
        self.assertEqual(by_id(model)[5]['fields'], {})

    def test_erase_recreate_fresh_identity_and_no_inherited_fields(self):
        model = graph.Model()
        apply(model, [C(1, 2, 1, 'P', 6)], [F(2, 'v', 9)])
        apply(model, [C(2, 2, 1), C(1, 3, 1, 'P', 6)])
        self.assertEqual(set(by_id(model)), {1, 3})
        self.assertEqual(by_id(model)[3]['fields'], {})
        apply(model, fields=[F(3, 'v', 4)])
        apply(model, [C(2, 3, 1), C(1, 3, 1, 'P', 6)])
        self.assertEqual(by_id(model)[3]['fields'], {})

    def test_reorder_stores_entire_children_list_verbatim(self):
        model = graph.Model()
        apply(model, [C(1, i + 2, 1, name, 6) for i, name in enumerate('ABCD')] + [C(1, 6, 4, 'child', 6)],
              [F(4, 'v', 3)])
        apply(model, [C(4, 1, children=(5, 3, 2, 4))])
        self.assertEqual(by_id(model)[1]['fields']['primChildren'], ('D', 'B', 'A', 'C'))
        # An omitted child keeps its row but is unreachable with its subtree.
        apply(model, [C(4, 1, children=(5, 3))])
        self.assertEqual(by_id(model)[1]['fields']['primChildren'], ('D', 'B'))
        self.assertEqual(set(by_id(model)), {1, 3, 5})
        self.assertEqual(set(model.detached), {2, 4, 6})
        # Edits to an unlisted section are stored on it, not as orphans.
        apply(model, fields=[F(4, 'v', 8)])
        self.assertEqual((model.detached[4]['fields']['v'], model.orphans), (8, {}))
        # Listing it again makes it and its subtree reachable; a Create appends.
        apply(model, [C(4, 1, children=(4, 5, 3)), C(1, 7, 1, 'E', 6)])
        state = by_id(model)
        self.assertEqual(state[1]['fields']['primChildren'], ('C', 'D', 'B', 'E'))
        self.assertEqual((state[4]['fields'], state[6]['path']), ({'primChildren': ('child',), 'v': 8}, '/C/child'))
        self.assertEqual(set(model.detached), {2})
        # A Create at an unlisted section's name replaces it, as at any occupied slot.
        apply(model, [C(1, 8, 1, 'A', 6)])
        self.assertEqual((model.detached, by_id(model)[1]['fields']['primChildren']), ({}, ('C', 'D', 'B', 'E', 'A')))

    def test_delete_keeps_unlisted_descendants_as_unreachable_rows(self):
        model = graph.Model()
        apply(model, [C(1, 2, 1, 'P', 6), C(1, 3, 2, 'A', 6), C(1, 4, 2, 'B', 6), C(1, 5, 4, 'Child', 6),
                      C(1, 6, 1, 'Q', 6)], [F(4, 'v', 4)])
        apply(model, [C(4, 2, children=(3,))])
        # The cascade walks listed children only, as the service's does.
        apply(model, [C(2, 2, 1)])
        self.assertEqual(set(by_id(model)), {1, 6})
        self.assertEqual(set(model.detached), {4, 5})
        # A peer's later Move of the surviving row makes it and its subtree reachable again.
        apply(model, [C(3, 4, 6, 'B', old_parent=2)])
        state = by_id(model)
        self.assertEqual((state[4]['path'], state[5]['path'], state[4]['fields']['v']), ('/Q/B', '/Q/B/Child', 4))
        self.assertEqual((model.detached, state[6]['fields']['primChildren']), ({}, ('B',)))

    def test_diff_orders_are_sibling_indices_across_child_lists(self):
        model = graph.Model()
        apply(model, [C(5, 10, 1, 'P', 6, 7), C(5, 11, 10, 'child', 6, 19),
                      C(5, 12, 10, 'attr', 1, 23)], is_diff=True)
        state = by_id(model)
        self.assertEqual(state[10]['fields'], {'primChildren': ('child',), 'properties': ('attr',)})
        self.assertEqual((state[11]['order'], state[12]['order']), (0, 1))
        # A restated section takes its index among the unchanged siblings, even one an unchanged sibling holds.
        apply(model, [C(5, 12, 10, 'attr', 1, 0)], is_diff=True, base_version=1)
        state = by_id(model)
        self.assertEqual((state[12]['order'], state[11]['order']), (0, 1))
    def test_reorder_naming_absent_or_moved_members_is_invalid_argument(self):
        model = graph.Model()
        apply(model, [C(1, 2, 1, 'P', 6), C(1, 7, 1, 'Q', 6)] +
              [C(1, i, 2, name, 6) for i, name in zip(range(3, 7), 'ABCD')] + [C(1, 8, 2, 'attr', 1)])
        apply(model, [C(3, 4, 7, 'B', old_parent=2), C(2, 5, 2), C(1, 9, 2, 'E', 6)])
        for commands, message in (
                ((C(4, 2, children=(6, 5, 3)),), 'ReorderChildren contained unknown section_id 5 for parent 2'),
                ((C(4, 2, children=(6, 99)),), 'ReorderChildren contained unknown section_id 99 for parent 2'),
                ((C(4, 2, children=(6, 4, 3)),), 'ReorderChildren(2) listed section_id 4 whose actual parent_id is 7'),
                ((C(2, 9, 2), C(4, 2, children=(9, 6))), 'ReorderChildren contained unknown section_id 9 for parent 2'),
                ((C(4, 99, children=(4,)),), 'ReorderChildren(99) listed section_id 4 whose actual parent_id is 7')):
            with self.subTest(commands=commands):
                before = model.state, model.history
                with self.assertRaises(graph.GraphError) as caught:
                    model.apply(encode(Delta(commands=commands, fields=(F(1, 'v', 1),))), 999)
                self.assertEqual((caught.exception.kind, caught.exception.status_message),
                                 ('malformed', 'delta error: ' + message))
                self.assertEqual((model.state, model.history), before)
        for parent in (2, 999):
            for children in ((0,), (1,), (parent,), (998, 998)):
                self.assertRejected(model, Delta(commands=(C(4, parent, children=children),)), 'malformed')
        # childrenListId is ignored: one list orders every child list of the parent.
        apply(model, [C(4, 2, child_list=1, children=(6, 8, 3, 9))])
        state = by_id(model)
        self.assertEqual((state[2]['fields']['primChildren'], state[2]['fields']['properties']), (('D', 'A', 'E'), ('attr',)))
        # A moved-away child that returns is appended; a list that omits the attribute unlists it.
        apply(model, [C(3, 4, 2, 'B', old_parent=7), C(4, 2, children=(4, 6, 3, 9))])
        self.assertEqual(by_id(model)[2]['fields'], {'primChildren': ('B', 'D', 'A', 'E')})
        self.assertEqual(set(model.detached), {8})
    def test_unchanged_reorder_preserves_authored_empty_and_absent_fields(self):
        for authored in (False, True):
            model = graph.Model()
            if authored:
                apply(model, fields=[F(1, 'primChildren', ())])
            before = model.state
            apply(model, [C(4, 1, children=())])
            self.assertEqual(model.state, before)
            self.assertEqual(len(model.history), 2 if authored else 1)
            apply(model, [C(1, 2, 1, 'A', 6), C(1, 3, 1, 'B', 6)], [F(1, 'x', 1)])
            before = model.state
            apply(model, [C(4, 1, children=(2, 3))])
            self.assertEqual(model.state, before)
    def test_diff_moves_survivor_out_before_parent_disappears_and_swaps(self):
        model = graph.Model()
        apply(model, [C(1, 2, 1, 'A', 6), C(1, 3, 2, 'child', 6), C(1, 4, 1, 'B', 6)], [F(3, 'v', 7)])
        commands = (C(6, 2), C(5, 3, 4, 'survivor', 6, 0))
        apply(model, commands, is_diff=True, base_version=1)
        self.assertEqual(by_id(model)[3]['path'], '/B/survivor')
        self.assertEqual(by_id(model)[3]['fields'], {'v': 7})
        apply(model, [C(5, 3, 1, 'B', 6, 1), C(5, 4, 1, 'A', 6, 0)], is_diff=True, base_version=2)
        self.assertEqual(by_id(model)[1]['fields']['primChildren'], ('A', 'B'))
        self.assertEqual((by_id(model)[3]['path'], by_id(model)[4]['path']), ('/B', '/A'))

    def test_reset_replaces_state_and_command_order_does_not_define_diff_order(self):
        model = graph.Model()
        apply(model, [C(1, 2, 1, 'old', 6)], [F(1, 'gone', 8)])
        apply(model, [C(5, 4, 1, 'later', 6, 9), C(5, 3, 1, 'first', 6, 2), C(5, 1, 0, '', 7, 0)], is_diff=True)
        self.assertEqual(set(by_id(model)), {1, 3, 4})
        self.assertEqual(by_id(model)[1]['fields'], {'primChildren': ('first', 'later')})

    def test_final_child_field_precedence_and_presence(self):
        model = graph.Model()
        apply(model, fields=[F(1, 'primChildren', ())])
        self.assertIn('primChildren', by_id(model)[1]['fields'])
        apply(model, fields=[F(1, 'primChildren', None)])
        self.assertNotIn('primChildren', by_id(model)[1]['fields'])
        apply(model, [C(1, 2, 1, 'A', 6), C(1, 3, 1, 'B', 6)],
              [F(1, 'primChildren', ('missing',)), F(1, 'primChildren', None), F(1, 'primChildren', ('B', 'A'))])
        self.assertEqual(by_id(model)[1]['fields']['primChildren'], ('B', 'A'))
        self.assertEqual((by_id(model)[2]['order'], by_id(model)[3]['order']), (1, 0))
        for value in (None, (), ('A',), ('A', 'A'), ('A', 'missing')):
            self.assertRejected(model, Delta(fields=(F(1, 'primChildren', value),)))
        self.assertRejected(model, Delta(fields=(F(1, 'primChildren', 3),)), 'malformed')

    def test_atomic_rejection_late_conflict_and_identity_not_path(self):
        model = graph.Model()
        model.apply(encode(Delta(commands=(C(1, 2, 1, 'A', 6), C(1, 3, 1, 'B', 6)))), 10)
        failures = (
            (Delta(commands=(C(1, 4, 1, 'valid', 6), C(1, 5, 1, 'A', 6), C(1, 3, 1, 'other', 6)),
                   fields=(F(2, 'v', 42), F(99, 'v', 1))), 'malformed'),
            (Delta(commands=(C(3, 2, 2, 'cycle', old_parent=1),)), 'malformed'),
            (Delta(commands=(C(1, 4, 99, 'missingParent', 6), C(4, 99, children=(3,)))), 'malformed'),
            (Delta(True, 1, (C(5, 2, 1, 'A', 6, 0), C(6, 2))), 'unsupported'),
        )
        for delta, kind in failures:
            self.assertRejected(model, delta, kind)
        self.assertEqual(model.orphans, {})
        returned = model.state
        returned[0]['fields']['injected'] = 1
        self.assertNotIn('injected', model.state[0]['fields'])

    def test_retained_child_field_type_change_rejected(self):
        model = graph.Model()
        apply(model, [C(1, 2, 1, 'P', 6), C(1, 3, 1, 'Q', 6)], [F(2, 'primChildren', ())])
        self.assertRejected(model, Delta(True, 1, (C(5, 2, 3, 'attr', 1, 0),)), 'malformed')

    def test_order_exhaustion_profile_rejection_and_quota_atomicity(self):
        model = graph.Model()
        apply(model, [C(5, 2, 1, 'A', 6, 2**32 - 1)], is_diff=True)
        self.assertEqual(by_id(model)[2]['order'], 0)  # A diff's order is an index; past the end appends.
        self.assertRejected(model, Delta(True, 1, (C(5, 3, 2, 'v', 12, 0),)))
        self.assertRejected(model, Delta(fields=(F(2, 'timeSamples', ()),)), 'malformed')
        for name, value in (('MAX_WORK', 1), ('MAX_HISTORY_BYTES', 1), ('MAX_COMMITS', 1), ('MAX_SECTIONS', 1)):
            with patch.object(graph, name, value):
                self.assertRejected(model, Delta(), 'limit')

    def test_nested_variants_paths_lists_and_latest_subtree(self):
        model = graph.Model()
        apply(model, [C(1, 2, 1, 'P', 6), C(1, 3, 2, 'look', 11), C(1, 4, 3, '1', 10),
                      C(1, 5, 3, 'high-low', 10), C(1, 6, 5, 'Child', 6), C(1, 7, 5, 'amount', 1),
                      C(1, 8, 5, 'detail', 11), C(1, 9, 8, '2', 10), C(1, 10, 9, 'link', 8)],
              [F(7, 'value', 42)])
        state = by_id(model)
        expected = {2: '/P', 3: '/P{look=}', 4: '/P{look=1}', 5: '/P{look=high-low}',
                    6: '/P{look=high-low}Child', 7: '/P{look=high-low}.amount',
                    8: '/P{look=high-low}{detail=}', 9: '/P{look=high-low}{detail=2}',
                    10: '/P{look=high-low}{detail=2}.link'}
        self.assertEqual({i: state[i]['path'] for i in expected}, expected)
        self.assertEqual(state[5]['fields'], {'primChildren': ('Child',), 'properties': ('amount',),
                                             'variantSetChildren': ('detail',)})
        self.assertEqual(state[3]['fields'], {'variantChildren': ('1', 'high-low')})
        # Include a newly added descendant in the entire moved/deleted subtree.
        apply(model, [C(1, 11, 9, 'Peer', 6), C(3, 2, 1, 'Moved', old_parent=1)])
        moved = by_id(model)
        self.assertEqual(moved[11]['path'], '/Moved{look=high-low}{detail=2}Peer')
        for identity, path in expected.items():
            self.assertEqual(moved[identity]['path'], '/Moved' + path[2:])
        self.assertEqual(state[7]['fields'], moved[7]['fields'])
        apply(model, [C(3, 11, 1, 'Survivor', old_parent=9), C(2, 2, 1)])
        self.assertEqual(set(by_id(model)), {1, 11})
        self.assertEqual(by_id(model)[11]['path'], '/Survivor')

    def test_variant_cross_list_orders_presence_and_atomic_rejection(self):
        model = graph.Model()
        apply(model, [C(5, 2, 1, 'P', 6, 0), C(5, 3, 2, 'look', 11, 2), C(5, 4, 3, '1', 10, 4),
                      C(5, 5, 4, 'Child', 6, 11), C(5, 6, 4, 'attr', 1, 23),
                      C(5, 7, 4, 'first', 11, 31), C(5, 8, 4, 'middle', 11, 45),
                      C(5, 9, 4, 'last', 11, 61)], is_diff=True)
        apply(model, [C(4, 4, child_list=2, children=(5, 6, 9, 8, 7))])
        self.assertEqual(by_id(model)[4]['fields'], {'primChildren': ('Child',), 'properties': ('attr',),
            'variantSetChildren': ('last', 'middle', 'first')})
        self.assertEqual([by_id(model)[i]['order'] for i in (5, 6, 7, 8, 9)], [0, 1, 4, 3, 2])
        apply(model, fields=[F(7, 'variantChildren', ())])
        self.assertEqual(by_id(model)[7]['fields'], {'variantChildren': ()})
        apply(model, fields=[F(7, 'variantChildren', None)])
        self.assertEqual(by_id(model)[7]['fields'], {})
        apply(model, [C(5, 9, 4, 'last', 11, 1)], is_diff=True, base_version=1)
        self.assertEqual([by_id(model)[i]['order'] for i in (5, 9, 6, 8, 7)], [0, 1, 2, 3, 4])
        self.assertRejected(model, Delta(fields=(F(4, 'variantSetChildren', ('first',)),)))
        self.assertRejected(model, Delta(commands=(C(4, 4, child_list=2, children=(3,)),)), 'malformed')
        self.assertRejected(model, Delta(commands=(C(1, 10, 7, '', 10),)), 'malformed')
        self.assertRejected(model, Delta(commands=(C(1, 10, 7, 'Child', 6),)), 'malformed')
        self.assertRejected(model, Delta(fields=(F(7, 'primChildren', ()),)), 'malformed')
        # Depth follows graph nodes, including both set and selection nodes.
        with patch.object(graph, 'MAX_DEPTH', 3):
            self.assertRejected(model, Delta(), 'limit')

    def test_path_children_expression_mapper_and_relational_attribute(self):
        model = graph.Model()
        apply(model, [C(1, 2, 1, 'P', 6), C(1, 3, 2, 'a', 1), C(1, 4, 3, '/Target.a', 2),
            C(1, 5, 3, 'expression', 3), C(1, 6, 3, '/Target', 4), C(1, 7, 6, 'arg', 5),
            C(1, 8, 2, 'rel', 8), C(1, 9, 8, '/Target', 9), C(1, 10, 9, 'weight', 1),
            C(1, 11, 8, 'expression', 3)], [F(10, 'value', 42)])
        expected = {4: '/P.a[/Target.a]', 5: '/P.a.expression', 6: '/P.a.mapper[/Target]',
            7: '/P.a.mapper[/Target].arg', 9: '/P.rel[/Target]', 10: '/P.rel[/Target].weight',
            11: '/P.rel.expression'}
        state = by_id(model)
        self.assertEqual({i: state[i]['path'] for i in expected}, expected)
        self.assertEqual(state[3]['fields'], {'connectionChildren': Paths(('/Target.a',)),
            'expressionChildren': ('expression',), 'mapperChildren': Paths(('/Target',))})
        self.assertEqual(state[8]['fields'], {'targetChildren': Paths(('/Target',)), 'expressionChildren': ('expression',)})
        apply(model, [C(3, 2, 1, 'Moved', old_parent=1)])
        moved = by_id(model)
        for i, path in expected.items():
            self.assertEqual(moved[i]['path'], '/Moved' + path[2:])
        self.assertEqual(moved[6]['name'], '/Target')
        self.assertEqual(moved[10]['fields'], {'value': 42})
        # Relational attributes are allowed; relationships under targets aren't.
        self.assertRejected(model, Delta(commands=(C(1, 12, 9, 'bad', 8),)), 'malformed')
        self.assertRejected(model, Delta(commands=(C(1, 12, 8, 'other', 3),)), 'malformed')
        self.assertRejected(model, Delta(fields=(F(8, 'targetChildren', ('/Target',)),)), 'malformed')
        apply(model, [C(3, 10, 2, 'survivor', old_parent=9), C(2, 8, 2)])
        self.assertEqual(by_id(model)[10]['path'], '/Moved.survivor')
        self.assertNotIn(9, by_id(model))

    def test_path_children_reorders_and_exact_presence(self):
        model = graph.Model()
        apply(model, [C(1, 2, 1, 'P', 6), C(1, 3, 2, 'a', 1), C(1, 4, 3, '/A.x', 2),
                      C(1, 5, 3, 'expression', 3), C(1, 6, 3, '/B.x', 2), C(1, 7, 3, '/C.x', 2)])
        apply(model, [C(4, 3, children=(7, 5, 6, 4))])
        self.assertEqual(by_id(model)[3]['fields']['connectionChildren'], Paths(('/C.x', '/B.x', '/A.x')))
        self.assertEqual(by_id(model)[5]['order'], 1)
        apply(model, fields=[F(3, 'connectionChildren', Paths(('/bad',))),
            F(3, 'connectionChildren', None), F(3, 'connectionChildren', Paths(('/B.x', '/A.x', '/C.x')))])
        self.assertEqual([by_id(model)[i]['order'] for i in (4, 6, 7)], [2, 0, 3])
        self.assertRejected(model, Delta(fields=(F(3, 'connectionChildren', Paths(('/A.x',))),)))
        self.assertRejected(model, Delta(fields=(F(3, 'connectionChildren', Paths(('/A.x', '/A.x', '/C.x'))),)))
        apply(model, [C(2, i, 3) for i in (4, 6, 7)])
        self.assertNotIn('connectionChildren', by_id(model)[3]['fields'])
        apply(model, fields=[F(3, 'connectionChildren', Paths(()))])
        self.assertEqual(by_id(model)[3]['fields']['connectionChildren'], Paths(()))
        apply(model, fields=[F(3, 'connectionChildren', None)])
        self.assertNotIn('connectionChildren', by_id(model)[3]['fields'])

    def test_root_order_and_accounting_preparation(self):
        model = graph.Model()
        apply(model, [C(5, 1, 0, '', 7, 17)], is_diff=True)
        apply(model, fields=[F(1, 'v', 3)])
        self.assertEqual(by_id(model)[1]['order'], 17)
        before = model.state, model.history, model.sequence
        class FailedCounter(int):
            def __add__(self, other):
                raise MemoryError('injected accounting allocation failure')
        model._history_bytes = FailedCounter(model._history_bytes)
        with self.assertRaises(MemoryError):
            apply(model, [C(1, 2, 1, 'new', 6)])
        self.assertEqual((model.state, model.history, model.sequence), before)
        class OnceCounter(int):
            calls = 0
            def __add__(self, other):
                self.calls += 1
                if self.calls > 1:
                    raise MemoryError('unexpected allocation after publication')
                return int(self) + other
        counter = OnceCounter(int(model._history_bytes))
        model._history_bytes = counter
        apply(model, [C(1, 2, 1, 'new', 6)])
        self.assertEqual(counter.calls, 1)
        self.assertIn(2, by_id(model))

    def test_seeded_identity_histories_against_direct_list_oracle(self):
        seed = 82731
        rng = random.Random(seed)
        model, expected, order, unlisted, next_id = graph.Model(), {}, [], {}, 2
        for step in range(180):
            actions = ('create', 'erase', 'rename', 'reorder', 'field') + (('relist',) if unlisted else ())
            action = rng.choice(actions) if order else 'create'
            if action == 'create':
                identity, next_id = next_id, next_id + 1
                name = f'node{identity}'
                apply(model, [C(1, identity, 1, name, 6)])
                expected[identity] = {'name': name, 'fields': {}}
                order.append(identity)
            elif action == 'erase':
                identity = rng.choice(order)
                apply(model, [C(2, identity, 1)])
                del expected[identity]
                order.remove(identity)
            elif action == 'rename':
                identity = rng.choice(order)
                name = f'renamed{step}'
                apply(model, [C(3, identity, 1, name, old_parent=1)])
                expected[identity]['name'] = name
            elif action == 'reorder':
                # Mostly complete permutations; an incomplete list unlists what it omits.
                count = len(order) if rng.random() < 0.8 else rng.randint(0, len(order))
                chosen = rng.sample(order, count)
                apply(model, [C(4, 1, children=tuple(chosen))])
                for identity in order:
                    if identity not in chosen:
                        unlisted[identity] = expected.pop(identity)
                order = chosen
            elif action == 'relist':
                identity = rng.choice(sorted(unlisted))
                order.insert(rng.randint(0, len(order)), identity)
                apply(model, [C(4, 1, children=tuple(order))])
                expected[identity] = unlisted.pop(identity)
            else:
                identity = rng.choice(order)
                value = rng.choice((None, rng.randint(-1000, 1000)))
                apply(model, fields=[F(identity, 'v', value)])
                if value is None:
                    expected[identity]['fields'].pop('v', None)
                else:
                    expected[identity]['fields']['v'] = value
            state = by_id(model)
            self.assertEqual(set(state), {1, *expected}, f'seed={seed} step={step}')
            self.assertEqual(set(model.detached), set(unlisted), f'seed={seed} step={step}')
            self.assertEqual(state[1]['fields'].get('primChildren', ()), tuple(expected[i]['name'] for i in order))
            for identity, wanted in expected.items():
                self.assertEqual((state[identity]['path'], state[identity]['fields']), ('/' + wanted['name'], wanted['fields']))
class ServiceConflictRuleTests(unittest.TestCase):
    """The Live Edit service's rules for commands that meet a changed graph.

    Missing targets are skipped and orphan field data is stored but never
    projected; a Create at an occupied (parent, name) slot replaces the
    occupant; Move edge cases delete a subtree. Each update is all-or-nothing.
    """
    def setUp(self):
        self.model = graph.Model()
        # /A{2} with child /A/Child{3} and attribute /A/Child.x{4}; /B{5}.
        self.model.apply(encode(Delta(commands=(C(1, 2, 1, 'A', 6), C(1, 3, 2, 'Child', 6), C(1, 4, 3, 'x', 1),
                                                C(1, 5, 1, 'B', 6)), fields=(F(4, 'default', 1),))), 1)

    def commit(self, commands=(), fields=(), samples=(), sequence=None):
        sequence = sequence or len(self.model.history) + 1
        self.model.apply(encode(Delta(commands=tuple(commands), fields=tuple(fields), samples=tuple(samples))), sequence)
        self.assertEqual(self.model.history[-1][0], sequence)
        return by_id(self.model)

    def test_field_and_sample_on_missing_section_are_orphaned_and_never_projected(self):
        self.commit([C(2, 3, 2)])
        before = self.model.state
        state = self.commit(fields=[F(4, 'default', 2), F(99, 'marker', 7), F(99, 'gone', None),
                                    F(99, 'primChildren', ('ghost',))],
                            samples=[Sample(4, 1.0, 5), Sample(99, 2.0, 6)])
        self.assertEqual(self.model.state, before)
        self.assertEqual(set(state), {1, 2, 5})
        self.assertEqual(len(self.model.history), 3)
        self.assertEqual(self.model.orphans, {4: {'default': 2, 'timeSamples': TimeSamples(((1.0, 5),))},
                                              99: {'marker': 7, 'primChildren': ('ghost',),
                                                   'timeSamples': TimeSamples(((2.0, 6),))}})
        self.commit(fields=[F(99, 'marker', None)], samples=[Sample(99, 2.0, None)])
        self.assertEqual(self.model.orphans[99], {'primChildren': ('ghost',)})
        self.assertEqual(self.model.state, before)

    def test_structural_commands_on_missing_targets_are_skipped(self):
        before = self.model.state
        self.commit([C(1, 6, 99, 'Orphan', 6), C(2, 99, 1), C(3, 99, 5, 'Moved', old_parent=1),
                     C(4, 99, children=())])
        self.assertEqual(self.model.state, before)
        self.assertEqual(len(self.model.history), 2)
        # A stale delete parent is ignored: the section's stored parent wins.
        state = self.commit([C(2, 3, 5)])
        self.assertEqual(set(state), {1, 2, 5})
        self.assertNotIn('primChildren', state[2]['fields'])

    def test_same_parent_rename_onto_occupied_name_keeps_duplicate_siblings(self):
        state = self.commit([C(3, 5, 1, 'A', old_parent=1)], [F(5, 'marker', 1)])
        self.assertEqual((state[2]['path'], state[5]['path']), ('/A', '/A'))
        self.assertEqual(state[1]['fields']['primChildren'], ('A', 'A'))
        # A later Create at the slot replaces only the lower identity.
        state = self.commit([C(1, 6, 1, 'A', 6)])
        self.assertEqual(set(state), {1, 5, 6})
        self.assertEqual((state[5]['fields'], state[1]['fields']['primChildren']), ({'marker': 1}, ('A', 'A')))
        # Snapshots carry both siblings and rebuild the same state.
        import scene_graph_snapshot as projection
        rebuilt = graph.Model()
        rebuilt.apply(projection.snapshot(self.model.state))
        self.assertEqual(graph.ranked(rebuilt.state), graph.ranked(self.model.state))

    def test_reorder_listing_members_of_a_missing_parent_is_invalid_argument(self):
        with self.assertRaises(graph.GraphError) as caught:
            self.commit([C(4, 99, children=(3,))])
        self.assertEqual(caught.exception.kind, 'malformed')

    def test_update_is_all_or_nothing(self):
        before = (self.model.state, self.model.history, self.model.sequence, self.model.orphans)
        with self.assertRaises(graph.GraphError):
            self.model.apply(encode(Delta(commands=(C(1, 6, 1, 'A', 6), C(2, 5, 1), C(1, 6, 1, 'Late', 6)),
                                          fields=(F(99, 'marker', 1), F(1, 'counter', 2)))), 2)
        self.assertEqual((self.model.state, self.model.history, self.model.sequence, self.model.orphans), before)

    def test_create_at_occupied_name_replaces_earlier_subtree(self):
        state = self.commit([C(1, 6, 1, 'A', 6), C(1, 7, 6, 'x', 1)], [F(7, 'default', 9), F(4, 'default', 3)])
        self.assertEqual(set(state), {1, 5, 6, 7})
        self.assertEqual((state[6]['path'], state[7]['path'], state[7]['fields']), ('/A', '/A.x', {'default': 9}))
        self.assertEqual(state[1]['fields']['primChildren'], ('B', 'A'))
        # The replaced attribute's later field is orphan data, not a new node.
        self.assertEqual(self.model.orphans, {4: {'default': 3}})
        # The slot is (parent, name), not the derived path: a prim replaces a same-named attribute.
        state = self.commit([C(1, 8, 6, 'x', 6)])
        self.assertEqual(set(state), {1, 5, 6, 8})
        self.assertEqual((state[8]['path'], state[6]['fields']), ('/A/x', {'primChildren': ('x',)}))

    def test_create_with_live_identity_is_invalid_argument_with_service_message(self):
        before = self.model.state
        for commands in ([C(1, 2, 1, 'Other', 6)], [C(1, 6, 1, 'N', 6), C(1, 6, 1, 'M', 6)]):
            with self.assertRaises(graph.GraphError) as caught:
                self.commit(commands)
            identity = commands[-1].section
            self.assertEqual(caught.exception.kind, 'malformed')
            self.assertEqual(caught.exception.status_message, f'delta error: duplicate section_id {identity} in CreateSection')
            self.assertEqual(self.model.state, before)

    def test_create_with_retired_identity_is_accepted_without_old_fields(self):
        self.commit([C(2, 2, 1)])
        state = self.commit([C(1, 3, 1, 'Again', 6), C(1, 4, 3, 'x', 1)])
        self.assertEqual((state[3]['path'], state[4]['path'], state[4]['fields']), ('/Again', '/Again.x', {}))
        # Delete and re-create of one identity within a batch is also accepted.
        state = self.commit([C(2, 3, 1), C(1, 3, 1, 'Third', 6)])
        self.assertEqual(state[3]['path'], '/Third')
        self.assertNotIn(4, state)

    def test_move_to_missing_parent_or_into_own_subtree_deletes_moved_subtree(self):
        for move in (C(3, 2, 99, 'Gone', old_parent=1), C(3, 2, 3, 'Cycle', old_parent=1)):
            with self.subTest(move=move):
                self.setUp()
                state = self.commit([move])
                self.assertEqual(set(state), {1, 5})
                self.assertEqual(state[1]['fields']['primChildren'], ('B',))

    def test_move_onto_occupied_name_follows_service_priority(self):
        # A non-ancestor occupant is deleted with its subtree, then the move proceeds.
        self.commit([C(1, 6, 5, 'Child', 6), C(1, 7, 6, 'y', 1)])
        state = self.commit([C(3, 3, 5, 'Child', old_parent=2)])
        self.assertEqual(set(state), {1, 2, 3, 4, 5})
        self.assertEqual((state[3]['path'], state[4]['path']), ('/B/Child', '/B/Child.x'))
        # An occupant that is an ancestor of the mover deletes the mover instead.
        self.setUp()
        self.commit([C(1, 6, 3, 'A', 6)])
        state = self.commit([C(3, 6, 1, 'A', old_parent=3)])
        self.assertEqual(set(state), {1, 2, 3, 4, 5})

    def test_move_with_matching_command_parents_is_rename_in_place(self):
        # old_parent == parent selects a rename; the stored parent is not re-checked.
        state = self.commit([C(3, 3, 5, 'Renamed', old_parent=5)])
        self.assertEqual((state[3]['parent'], state[3]['path']), (2, '/A/Renamed'))

    def test_captured_move_after_peer_move_follows_command_parents(self):
        # A peer moved /A/Child{3} to /B{5}; a captured Move naming the old parent /A{2} replays.
        peer = [C(3, 3, 5, 'Child', old_parent=2)]
        # Equal command parents rename under the stored parent, without a sibling-name check.
        self.commit(peer + [C(1, 6, 5, 'Taken', 6)])
        state = self.commit([C(3, 3, 2, 'N2', old_parent=2)])
        self.assertEqual((state[3]['parent'], state[3]['path']), (5, '/B/N2'))
        state = self.commit([C(3, 3, 2, 'Taken', old_parent=2)])
        self.assertEqual((state[3]['path'], state[6]['path']), ('/B/Taken', '/B/Taken'))
        # Differing command parents take the move path: occupancy applies at the new parent.
        self.setUp()
        self.commit(peer + [C(1, 6, 5, 'Taken', 6)])
        state = self.commit([C(3, 3, 5, 'Renamed', old_parent=2)])
        self.assertEqual((state[3]['path'], state[5]['fields']['primChildren']), ('/B/Renamed', ('Renamed', 'Taken')))
        state = self.commit([C(3, 3, 5, 'Taken', old_parent=2)])
        self.assertEqual((set(state), state[3]['path']), ({1, 2, 3, 4, 5}, '/B/Taken'))

    def test_move_with_stale_old_parent_leaves_the_section_listed_under_its_actual_parent(self):
        # A peer moved /A/Child{3} to /B{5}; a Move carrying old parent /A{2} replays to /R{6}.
        self.commit([C(3, 3, 5, 'Child', old_parent=2), C(1, 6, 1, 'R', 6)])
        state = self.commit([C(3, 3, 6, 'Child', old_parent=2)])
        self.assertEqual((state[3]['path'], state[4]['path']), ('/R/Child', '/R/Child.x'))
        self.assertNotIn('primChildren', state[5]['fields'])
        # B's list still names the section, so deleting B deletes it and its subtree.
        state = self.commit([C(2, 5, 1)])
        self.assertEqual((set(state), self.model.detached), ({1, 2, 6}, {}))
        self.assertNotIn('primChildren', state[6]['fields'])
        # A surviving stale listing leaves the section reachable through its stored parent.
        self.setUp()
        self.commit([C(3, 3, 5, 'Child', old_parent=2), C(1, 6, 1, 'R', 6), C(3, 3, 6, 'Child', old_parent=2),
                     C(1, 7, 5, 'Other', 6)])
        state = self.commit([C(2, 7, 5)])
        self.assertEqual(state[3]['path'], '/R/Child')
        # A Reorder of B replaces its list, so deleting B afterwards keeps the section.
        state = self.commit([C(4, 5, children=()), C(2, 5, 1)])
        self.assertEqual((set(state), state[3]['path']), ({1, 2, 3, 4, 6}, '/R/Child'))
        # A Move carrying the current old parent leaves no stale listing.
        self.setUp()
        self.commit([C(3, 3, 5, 'Child', old_parent=2), C(1, 6, 1, 'R', 6), C(3, 3, 6, 'Child', old_parent=5)])
        state = self.commit([C(2, 5, 1)])
        self.assertEqual(state[3]['path'], '/R/Child')

    def test_same_batch_delete_sees_the_listing_an_earlier_move_dropped(self):
        # /A/Child{3} stored at /R{6}, still listed by /B{5}; one batch moves it off B, then deletes B.
        self.commit([C(3, 3, 5, 'Child', old_parent=2), C(1, 6, 1, 'R', 6), C(3, 3, 6, 'Child', old_parent=2),
                     C(1, 7, 1, 'X', 6)])
        state = self.commit([C(3, 3, 7, 'Child', old_parent=5), C(2, 5, 1)])
        self.assertEqual((5 in state, state[3]['path']), (False, '/X/Child'))
        # The same holds for a section whose stored parent lists it.
        self.setUp()
        self.commit([C(3, 3, 5, 'Child', old_parent=2), C(1, 6, 1, 'R', 6)])
        state = self.commit([C(3, 3, 6, 'Child', old_parent=5), C(2, 5, 1)])
        self.assertEqual((5 in state, state[3]['path']), (False, '/R/Child'))

    def test_self_parenting_structural_command_is_invalid_argument(self):
        for command in (C(1, 6, 6, 'Self', 6), C(3, 2, 2, 'Self', old_parent=1)):
            with self.subTest(command=command), self.assertRaises(graph.GraphError) as caught:
                self.commit([command])
            self.assertEqual(caught.exception.kind, 'malformed')


if __name__ == '__main__':
    unittest.main()
