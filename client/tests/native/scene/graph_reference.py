#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare every native graph transition against an independent bounded oracle."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests/reference_service')]
from scene_graph_wire import Command as C, Field as F, Sample as S, TimeSamples as T, Delta, Token, Specifier, Variability, Paths, PathOp, typed_value, encode, decode
from scene_graph_wire import String, Strings, TokenArray, Real, real
from scene_graph_model import Model, GraphError, CHILD_FIELDS, group
from scene_graph_snapshot import snapshot, difference as graph_difference
from graph_reduction import Case, Trials, reduce_history, mismatch, expected, canonical, validate_record, reject_constant
from tools.live import sanitizer
from tools.live.test import fingerprint, native_identity
from tools.live.test_support import record_seeds, run_process, write_json


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate native JSON key: {key!r}')
        result[key] = value
    return result


def typed(state):
    return [dict(node, fields={key: typed_value(value)
                              for key, value in node['fields'].items()}) for node in state]


def shadowed_projection(state):
    """Design 4.7: among siblings that share a derived path, the lowest section ID
    owns it; the others and their subtrees stay out of the visible projection.
    The model reports every duplicate at the shared path, so the expected native
    projection drops the shadowed sections and their parents' child-field entries."""
    nodes = {node['id']: node for node in state}
    owners, shadowed = {}, set()
    for identity in sorted(nodes):
        if owners.setdefault(nodes[identity]['path'], identity) != identity:
            shadowed.add(identity)
    if not shadowed:
        return state

    def hidden(identity):
        while identity in nodes and identity != 1:
            if identity in shadowed:
                return True
            identity = nodes[identity]['parent']
        return False

    members = {}
    for identity in sorted(nodes, key=lambda identity: nodes[identity]['order']):
        if identity != 1:
            members.setdefault(group(nodes, identity), []).append(identity)
    result = []
    for node in state:
        if hidden(node['id']):
            continue
        fields = dict(node['fields'])
        for key, value in node['fields'].items():
            listed = members.get((node['id'], key), ())
            if key not in CHILD_FIELDS or not shadowed.intersection(listed):
                continue
            names = value.value if type(value) is Paths else value
            assert len(names) == len(listed), (node['id'], key)
            names = tuple(name for name, identity in zip(names, listed) if identity not in shadowed)
            fields[key] = Paths(names) if type(value) is Paths else names
        result.append(dict(node, fields=fields))
    return result


def scenarios(seed):
    yield 'empty', Delta()
    yield from sample_scenarios(seed)
    yield 'membership_reorder_setup', Delta(commands=(C(1, 2, 1, 'P', 6), C(1, 7, 1, 'Q', 6),
        C(1, 3, 2, 'A', 6), C(1, 4, 2, 'B', 6), C(1, 5, 2, 'C', 6), C(1, 6, 2, 'D', 6), C(1, 8, 2, 'attr', 1)))
    yield 'membership_reorder_peer', Delta(commands=(C(3, 4, 7, 'B', old_parent=2), C(2, 5, 2), C(1, 9, 2, 'E', 6)))
    yield 'membership_reorder_survivors', Delta(commands=(C(4, 2, children=(6, 5, 4, 3)),))
    yield 'membership_reorder_all_filtered', Delta(commands=(C(4, 2, children=(4, 5, 999)),))
    yield 'membership_reorder_wrong_list_atomic', Delta(commands=(C(4, 2, children=(3, 8, 6)),))
    yield 'membership_reorder_invalid_list', Delta(commands=(C(4, 1, children=(999,), child_list=1),))
    yield 'membership_reorder_missing_parent', Delta(commands=(C(4, 999, children=(4,)),))
    for parent in (2, 999):
        # A reader ignores IDs that are not children and repeats, as USD's list ordering does.
        for children in ((0,), (1,), (parent,), (998, 998), (3, 3)):
            yield f'membership_reorder_ignored_{parent}_{children}', Delta(commands=(C(4, parent, children=children),))
    yield 'membership_reorder_returned', Delta(commands=(C(3, 4, 2, 'B', old_parent=7), C(4, 2, children=(4, 6))))
    yield 'membership_reorder_deleted_in_batch', Delta(commands=(C(2, 9, 2), C(4, 2, children=(9, 999))))
    yield 'membership_reorder_reset', Delta(True)
    yield 'membership_reorder_absent_field', Delta(commands=(C(4, 1, children=(999,)),))
    yield 'membership_reorder_explicit_empty', Delta(fields=(F(1, 'primChildren', ()),))
    yield 'membership_reorder_empty_field', Delta(commands=(C(4, 1, children=(999,)),))
    # P's children interleave prims A and B, properties x and y, and variant set v; a Reorder orders each child
    # list it names within the positions that list holds.
    yield 'interleaved_reorder_reset', Delta(True)
    yield 'interleaved_reorder_setup', Delta(commands=(C(1, 2, 1, 'P', 6), C(1, 3, 2, 'A', 6), C(1, 4, 2, 'x', 1),
        C(1, 5, 2, 'B', 6), C(1, 6, 2, 'y', 8), C(1, 7, 2, 'v', 11)))
    yield 'interleaved_reorder_prims', Delta(commands=(C(4, 2, children=(5, 3)),))
    yield 'interleaved_reorder_properties', Delta(commands=(C(4, 2, child_list=1, children=(6, 4)),))
    yield 'interleaved_reorder_full', Delta(commands=(C(4, 2, children=(3, 5, 7, 4, 6)),))
    yield 'interleaved_reorder_both_partial', Delta(commands=(C(4, 2, children=(6, 5, 3)),))
    # Each child's first occurrence orders it; repeats and unknown IDs are ignored.
    yield 'interleaved_reorder_repeats', Delta(commands=(C(4, 2, children=(3, 5, 3, 999, 6, 4, 6, 2, 1)),))
    yield 'source_format_initial', Delta(True, source_format='usda')
    yield 'source_format_ordinary_preserve', Delta(fields=(F(1, 'value', 17),))
    yield 'source_format_diff_preserve', Delta(True, 1)
    yield 'source_format_diff_replace', Delta(True, 1, source_format='usdc')
    yield 'source_format_failed_reset', Delta(True, commands=(C(5, 2, 999, 'missingParent', 6),),
                                             source_format='rejected')
    yield 'source_format_failed_diff', Delta(True, 1, fields=(F(999, 'value', 1),), source_format='rejected')
    yield 'source_format_unicode_reset', Delta(True, source_format='format-\u03bb-\U0001f680')
    yield 'source_format_absent_reset', Delta(True)
    yield 'public_scalar_types', Delta(fields=(F(1, 'boolean', True), F(1, 'integer', 1),
        F(1, 'token', Token('λ')), F(1, 'empty_token', Token('')), F(1, 'specifier', Specifier(0)),
        F(1, 'variability', Variability(1))))
    yield 'public_scalar_type_changes', Delta(True, 1, fields=(F(1, 'boolean', 1), F(1, 'integer', True),
        F(1, 'specifier', Variability(0)), F(1, 'variability', Specifier(2)), F(1, 'empty_token', ())))
    yield 'public_scalar_erasure', Delta(fields=(F(1, 'token', None),))
    yield 'public_scalar_reset', Delta(True)
    yield from presence_scenarios()
    yield 'variant_graph', Delta(commands=(C(1, 2, 1, 'P', 6), C(1, 3, 2, 'look', 11),
        C(1, 4, 3, '1', 10), C(1, 5, 3, 'high-low', 10), C(1, 6, 5, 'Child', 6),
        C(1, 7, 5, 'amount', 1), C(1, 8, 5, 'first', 11), C(1, 9, 8, '2', 10),
        C(1, 10, 9, 'link', 8), C(1, 11, 5, 'middle', 11), C(1, 12, 5, 'last', 11)),
        fields=(F(7, 'value', 42), F(10, 'custom', True)))
    yield 'variant_partial_reorder', Delta(commands=(C(4, 5, child_list=2, children=(12, 8)),))
    yield 'variant_selection_reorder', Delta(commands=(C(4, 3, children=(5, 4)),))
    yield 'variant_cross_list_collision', Delta(True, 1, (C(5, 8, 5, 'first', 11, 1),))
    yield 'variant_wrong_list_member', Delta(commands=(C(4, 5, child_list=2, children=(7,)),))
    yield 'variant_incomplete_children', Delta(fields=(F(5, 'variantSetChildren', ('first',)),))
    yield 'variant_empty_children', Delta(fields=(F(11, 'variantChildren', ()),))
    yield 'variant_erase_empty_children', Delta(fields=(F(11, 'variantChildren', None),))
    yield 'variant_empty_name', Delta(commands=(C(1, 13, 11, '', 10),))
    yield 'variant_wrong_parent_type', Delta(commands=(C(1, 13, 11, 'Child', 6),))
    yield 'variant_wrong_field_type', Delta(fields=(F(11, 'primChildren', ()),))
    yield 'variant_latest_subtree', Delta(commands=(C(1, 13, 9, 'Peer', 6), C(3, 2, 1, 'Moved', old_parent=1)))
    yield 'variant_move_out_before_delete', Delta(commands=(C(3, 13, 1, 'Survivor', old_parent=9), C(2, 5, 3)))
    yield 'variant_diff_survivor', Delta(True, 1, (C(6, 2), C(6, 3), C(5, 14, 1, 'Q', 6, 2),
        C(5, 15, 14, 'saved', 11, 0), C(5, 4, 15, '1', 10, 0)))
    yield 'variant_reset', Delta(True)
    yield 'path_children_create', Delta(commands=(C(1, 2, 1, 'P', 6), C(1, 3, 2, 'a', 1),
        C(1, 4, 3, '/Target.a', 2), C(1, 5, 3, 'expression', 3), C(1, 6, 3, '/T.x', 4),
        C(1, 7, 6, 'arg', 5), C(1, 8, 2, 'rel', 8), C(1, 9, 8, '/Target', 9),
        C(1, 10, 9, 'weight', 1), C(1, 11, 8, 'expression', 3), C(1, 12, 3, '/Other.x', 2)),
        fields=(F(10, 'default', 42), F(3, 'connectionPaths', PathOp(True, ('/Target.a', '/Other.x'))),
                F(8, 'targetPaths', PathOp(added=('/Target',), deleted=('/Missing',)))))
    yield 'path_children_partial_reorder', Delta(commands=(C(4, 3, children=(12, 4)),))
    yield 'path_children_cross_list_collision', Delta(True, 1, (C(5, 6, 3, '/T.x', 4, 1),))
    yield 'path_children_wrong_vector_type', Delta(fields=(F(8, 'targetChildren', ('/Target',)),))
    yield 'path_children_incomplete_members', Delta(fields=(F(3, 'connectionChildren', Paths(('/Target.a',))),))
    yield 'path_children_wrong_parent_type', Delta(commands=(C(1, 13, 9, 'bad', 8),))
    yield 'path_children_expression_name', Delta(commands=(C(1, 13, 3, 'bad', 3),))
    yield 'path_children_move_owner', Delta(commands=(C(3, 2, 1, 'Moved', old_parent=1),))
    yield 'path_children_diff_survivor', Delta(True, 1, (C(6, 8), C(6, 9), C(6, 11),
        C(5, 10, 2, 'saved', 1, 2)))
    yield 'path_list_all_buckets', Delta(fields=(F(3, 'connectionPaths', PathOp(added=('/A',),
        prepended=('/P',), appended=('/Z',), deleted=('/D',), ordered=('/O',))),))
    yield 'path_list_explicit_empty', Delta(fields=(F(3, 'connectionPaths', PathOp(True)),))
    yield 'path_list_nonexplicit_empty', Delta(fields=(F(3, 'connectionPaths', PathOp()),))
    yield 'path_list_erasure', Delta(fields=(F(3, 'connectionPaths', None),))
    yield 'path_children_superseded_vectors', Delta(fields=(F(3, 'connectionChildren', Paths(('/bad',))),
        F(3, 'connectionChildren', None), F(3, 'connectionChildren', Paths(('/Target.a', '/Other.x')))))
    yield 'path_children_remove', Delta(commands=(C(2, 4, 3), C(2, 12, 3)))
    yield 'path_children_explicit_empty', Delta(fields=(F(3, 'connectionChildren', Paths(())),))
    yield 'path_children_empty_erasure', Delta(fields=(F(3, 'connectionChildren', None),))
    yield 'path_children_reset', Delta(True)
    yield 'create_subtree', Delta(commands=(C(1, 2, 1, 'P', 6), C(1, 3, 2, 'C', 6), C(1, 4, 3, 'a:x', 1),
                                           C(1, 5, 1, 'Q', 6), C(1, 6, 2, 'r', 8)),
                                  fields=(F(4, 'value', -17), F(6, 'value', -0.0)))
    yield 'move_subtree', Delta(commands=(C(3, 2, 5, 'Moved', old_parent=1),))
    yield 'late_destination_conflict', Delta(commands=(C(1, 7, 1, 'new', 6), C(1, 8, 5, 'Moved', 6)), fields=(F(4, 'value', 99),))
    yield 'stale_parent', Delta(commands=(C(2, 2, 1),))
    yield 'cycle', Delta(commands=(C(3, 2, 3, 'cycle', old_parent=5),))
    yield 'nonmember_reorder_noop', Delta(commands=(C(4, 1, children=(3,99)),))
    yield 'missing_field_owner', Delta(fields=(F(99, 'value', 1),))
    yield 'preserve_then_delete', Delta(True, 1, (C(6, 2), C(5, 3, 5, 'survivor', 6, 1), C(6, 6)))
    yield 'recursive_delete', Delta(commands=(C(2, 3, 5),))
    yield 'explicit_empty', Delta(fields=(F(5, 'primChildren', ()),))
    yield 'empty_erasure', Delta(fields=(F(5, 'primChildren', None),))
    yield 'new_identity_same_path', Delta(commands=(C(1, 10, 5, 'survivor', 6),))
    yield 'same_batch_reused_identity', Delta(commands=(C(2, 10, 5), C(1, 10, 5, 'survivor', 6)))
    yield 'reset_sparse_order', Delta(True, 0, (C(5, 15, 1, 'D', 6, 50), C(5, 13, 1, 'B', 6, 10),
                                             C(5, 12, 1, 'A', 6, 2), C(5, 14, 1, 'C', 6, 30)))
    yield 'partial_reorder', Delta(commands=(C(4, 1, children=(15, 13)),))
    yield 'superseded_child_vectors', Delta(fields=(F(1, 'primChildren', ('bad',)), F(1, 'primChildren', None),
                                                   F(1, 'primChildren', ('D', 'C', 'B', 'A'))))
    yield 'contradictory_child_vector', Delta(fields=(F(1, 'primChildren', ('A',)),))
    yield 'wrong_child_value_type', Delta(fields=(F(1, 'primChildren', 1),))
    yield 'duplicate_diff_id', Delta(True, 1, (C(5, 12, 1, 'A', 6, 2), C(6, 12)))
    yield 'diff_swap', Delta(True, 1, (C(5, 12, 1, 'D', 6, 2), C(5, 15, 1, 'A', 6, 50)))
    yield 'root_metadata_order', Delta(True, 1, (C(5, 1, 0, '', 7, 17),))
    yield 'root_field_after_metadata', Delta(fields=(F(1, 'value', 1.0),))
    yield 'retained_child_setup', Delta(True, 0, (C(5, 2, 1, 'P', 6, 0), C(5, 3, 1, 'Q', 6, 1)),
                                      (F(2, 'primChildren', ()),))
    yield 'retained_child_type_conflict', Delta(True, 1, (C(5, 2, 3, 'attr', 1, 0),))
    yield 'cross_list_order_setup', Delta(True, 0, (C(5, 2, 1, 'P', 6, 0), C(5, 3, 2, 'C', 6, 19), C(5, 4, 2, 'attr', 1, 23)))
    yield 'cross_list_order_collision', Delta(True, 1, (C(5, 4, 2, 'attr', 1, 19),))
    yield 'max_order', Delta(True, 0, (C(5, 2, 1, 'last', 6, 2**32-1),))
    yield 'exhausted_order', Delta(commands=(C(1, 3, 1, 'overflow', 6),))
    yield 'duplicate_sibling_reset', Delta(True)
    yield 'duplicate_sibling_setup', Delta(commands=(C(1, 2, 1, 'P', 6), C(1, 3, 2, 'A', 6), C(1, 4, 2, 'B', 6),
        C(1, 5, 4, 'Child', 6), C(1, 6, 2, 'C', 6)), fields=(F(4, 'value', 4),))
    yield 'duplicate_sibling_rename', Delta(commands=(C(3, 4, 2, 'A', old_parent=2),))
    yield 'duplicate_sibling_hidden_field', Delta(fields=(F(4, 'value', 5), F(5, 'value', 6)))
    yield 'duplicate_sibling_reorder', Delta(commands=(C(4, 2, children=(6, 4, 3)),))
    yield 'duplicate_sibling_separate', Delta(commands=(C(3, 3, 2, 'Z', old_parent=2),))
    yield 'duplicate_sibling_rejoin', Delta(commands=(C(3, 3, 2, 'A', old_parent=2),))
    yield 'duplicate_sibling_reveal', Delta(commands=(C(2, 3, 2),))
    yield 'random_reset', Delta(True)
    rng, identities, next_id = random.Random(seed), [], 100
    for step in range(160):
        action = rng.choice(('create', 'delete', 'rename', 'field', 'reorder')) if identities else 'create'
        if action == 'create':
            identity, next_id = next_id, next_id + 1
            identities.append(identity)
            delta = Delta(commands=(C(1, identity, 1, f'node{identity}', 6),))
        elif action == 'delete':
            identity = rng.choice(identities)
            identities.remove(identity)
            delta = Delta(commands=(C(2, identity, 1),))
        elif action == 'rename':
            delta = Delta(commands=(C(3, rng.choice(identities), 1, f'renamed{step}', old_parent=1),))
        elif action == 'field':
            delta = Delta(fields=(F(rng.choice(identities), 'value', rng.choice((None, -0.0, 7.0, rng.randint(-100000, 100000))), step),))
        else:
            delta = Delta(commands=(C(4, 1, children=tuple(rng.sample(identities, rng.randint(0, len(identities))))),))
        yield f'random_{step}_{action}', delta
    # A separate seeded hierarchy explores all three Variant child lists and
    # nested selections. Its generator tracks identities without consulting the
    # oracle's inferred paths or children, and every generated delta is saved.
    yield 'variant_random_reset', Delta(True, commands=(C(5, 2, 1, 'P', 6),
        C(5, 3, 2, 'look', 11), C(5, 4, 3, '1', 10), C(5, 5, 4, 'detail', 11),
        C(5, 6, 5, 'high-low', 10)))
    lists = {(parent, kind): [] for parent in (4, 6) for kind in (6, 1, 11)}
    for step in range(80):
        parent, kind = rng.choice(tuple(lists))
        members = lists[parent, kind]
        action = rng.choice(('create', 'delete', 'rename', 'field', 'reorder')) if members else 'create'
        if action == 'create':
            identity, next_id = next_id, next_id + 1
            members.append(identity)
            delta = Delta(commands=(C(1, identity, parent, f'child{identity}', kind),))
        elif action == 'delete':
            identity = rng.choice(members)
            members.remove(identity)
            delta = Delta(commands=(C(2, identity, parent),))
        elif action == 'rename':
            delta = Delta(commands=(C(3, rng.choice(members), parent, f'renamed{step}', old_parent=parent),))
        elif action == 'field':
            delta = Delta(fields=(F(rng.choice(members), 'value', rng.randint(-100000, 100000)),))
        else:
            delta = Delta(commands=(C(4, parent, child_list={6: 0, 1: 1, 11: 2}[kind],
                children=tuple(rng.sample(members, rng.randint(0, len(members))))),))
        yield f'variant_random_{step}_{action}', delta


def environment():
    return {key: value for key, value in os.environ.items() if key in
            {'PATH', 'LD_LIBRARY_PATH', 'TEMP', 'TMP', *sanitizer.PASSTHROUGH}}


def presence_scenarios():
    """Fixed cases for the value families Kit's presence layer authors; they consume no seeded randomness."""
    negative_zero, nan_payload = 0x80000000, 0x7fc00123
    values = {'bound_camera': String('/OmniverseKit_Persp'), 'following_user': String(''),
              'unicode': String('λ-\U0001f680'), 'selected_prim_paths': Strings(('/World/Cube', '/World/λ', '')),
              'xformOpOrder': TokenArray(('xformOp:transform',)), 'focalLength': real('float', 50.0),
              'exposure': Real('float', (nan_payload,)), 'floats': Real('float', (negative_zero, 0), True),
              'doubles': real('double', -0.0, 2.5, array=True),
              'xformOp:transform': real('matrix4d', *(float(i == j) for i in range(4) for j in range(4))),
              'transforms': Real('matrix4d', (), True), 'clippingRange': real('vec2f', 1.0, 1000000.0),
              'ranges': real('vec2f', 0.5, 2.0, array=True), 'omni:kit:centerOfInterest': real('vec3d', 0.0, -0.0, -500.0),
              'pivots': real('vec3d', 1.0, 2.0, 3.0, array=True), 'plane': real('vec4f', 0.0, 1.0, 0.0, -0.0),
              'clippingPlanes': real('vec4f', 1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 1.0, -5.0, array=True)}
    yield 'presence_values', Delta(fields=tuple(F(1, key, value) for key, value in values.items()))
    # Representation-only changes: a zero changes sign and a NaN payload changes.
    yield 'presence_sign_only_diff', Delta(True, 1, fields=(F(1, 'focalLength', Real('float', (negative_zero,))),
        F(1, 'exposure', Real('float', (nan_payload + 1,))),
        F(1, 'omni:kit:centerOfInterest', real('vec3d', 0.0, 0.0, -500.0))))
    yield 'presence_erasure', Delta(fields=(F(1, 'selected_prim_paths', None), F(1, 'following_user', None)))
    yield 'presence_empty_arrays', Delta(fields=(F(1, 'selected_prim_paths', Strings(())),
        F(1, 'xformOpOrder', TokenArray(())), F(1, 'clippingPlanes', Real('vec4f', (), True))))
    yield 'presence_reset', Delta(True)


def sample_scenarios(seed):
    yield 'samples_setup', Delta(commands=(C(1, 2, 1, 'P', 6), C(1, 3, 2, 'x', 1)))
    yield 'samples_map_then_keys', Delta(fields=(F(3, 'timeSamples', T(((1.0, 1), (2.0, 2))), 99),),
        samples=(S(3, 1.0, 10), S(3, 2.0, None), S(3, 3.0, True), S(3, 3.0, 31)))
    yield 'samples_bad_type_atomic', Delta(fields=(F(3, 'value', 99), F(3, 'timeSamples', ())))
    yield 'samples_superseded_bad_type', Delta(fields=(F(3, 'timeSamples', 7), F(3, 'timeSamples', T())))
    yield 'samples_missing_section_atomic', Delta(samples=(S(3, 1.0, 99), S(999, 2.0, 1)))
    yield 'samples_explicit_empty', Delta(fields=(F(3, 'timeSamples', T()),), samples=(S(3, 99.0, None),))
    yield 'samples_keep_empty', Delta(samples=(S(3, 100.0, None),))
    yield 'samples_last_key_erases_field', Delta(samples=(S(3, 1.0, 7), S(3, 1.0, None)))
    yield 'samples_signed_zero_insert', Delta(samples=(S(3, -0.0, False), S(3, 0.0, 1)))
    yield 'samples_signed_zero_replace_map', Delta(fields=(F(3, 'timeSamples', T(((0.0, True),))),))
    yield 'samples_infinite_times', Delta(samples=(S(3, float('-inf'), Token('start')), S(3, float('inf'), Token('end'))))
    yield 'samples_typed_values', Delta(samples=(S(3, 1.0, ()), S(3, 2.0, Paths(('/A',))),
        S(3, 3.0, PathOp(True, ('/A.x',))), S(3, 4.0, Specifier(2)), S(3, 5.0, Variability(0))))
    yield 'samples_move_owner', Delta(commands=(C(3, 2, 1, 'Moved', old_parent=1),), samples=(S(3, 6.0, -0.0),))
    yield 'samples_field_erase_then_insert', Delta(fields=(F(3, 'timeSamples', None),), samples=(S(3, 7.0, 7),))
    # Exercise the independent service's actual snapshot/diff projection bytes
    # through the native parser, including a signed-zero key/type-only change.
    model = Model()
    model.apply(encode(Delta(fields=(F(1, 'timeSamples', T(((-0.0, 1), (2.0, Token('old'))))),))))
    before = model.state
    yield 'samples_resume_snapshot', decode(snapshot(before))
    model.apply(encode(Delta(fields=(F(1, 'timeSamples', T(((0.0, True), (3.0, Token('new'))))),))))
    yield 'samples_resume_diff', decode(graph_difference(before, model.state, 1))
    for name, old, new in (('key_zero_sign_only', T(((-0.0, 7),)), T(((0.0, 7),))),
                           ('scalar_type_only', T(((1.0, True),)), T(((1.0, 1),)))):
        model = Model()
        model.apply(encode(Delta(fields=(F(1, 'timeSamples', old),))))
        before = model.state
        yield 'samples_' + name + '_snapshot', decode(snapshot(before))
        model.apply(encode(Delta(fields=(F(1, 'timeSamples', new),))))
        yield 'samples_' + name + '_diff', decode(graph_difference(before, model.state, 1))
    rng = random.Random(seed ^ 0x53414d504c45)
    for index in range(80):
        time = rng.choice((-2.5, -0.0, 0.0, 1.0, 7.25))
        choice = rng.randrange(5)
        value = rng.choice((True, False, -7, 0, 99, -0.0, 2.5, Token('sample'), ()))
        if choice == 0:
            delta = Delta(fields=(F(1, 'timeSamples', T(((time, value),))),))
        elif choice == 1:
            delta = Delta(fields=(F(1, 'timeSamples', T()),))
        elif choice == 2:
            delta = Delta(fields=(F(1, 'timeSamples', None),), samples=(S(1, time, value),))
        else:
            delta = Delta(samples=(S(1, time, None if choice == 3 else value),))
        yield f'samples_random_{index}_{choice}', delta
    yield 'samples_reset', Delta(True)


def input_metadata(value):
    # Diagnostic JSON remains standards-compliant for legal infinite times.
    # Exact input bytes are separately preserved and hashed.
    if isinstance(value, float) and not math.isfinite(value):
        return dict(float64_bits=struct.unpack('<Q', struct.pack('<d', value))[0])
    if isinstance(value, dict):
        return {key: input_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [input_metadata(item) for item in value]
    return value


def interpreter_identity():
    path = Path(sys.executable).resolve()
    return dict(path=str(path), version=sys.version, implementation=sys.implementation.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), optimize=sys.flags.optimize)


def reduction_witness(seed):
    yield 'required-and-noise', Delta(commands=(C(1, 2, 1, 'Required', 6), C(1, 3, 1, 'Noise', 6)),
                                      fields=(F(1, 'noise', 99),))
    rng = random.Random(seed)
    for index in range(12):
        yield f'noise-{index}', Delta(fields=(F(1, 'noise', rng.randrange(1000)),))
    yield 'missing-marker', Delta(fields=(F(2, 'reducerMarker', 17), F(1, 'unrelated', 99)))


def read_bounded(path, limit):
    with path.open('rb') as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError('reduction input byte limit')
    return data


def reduce_failure(cases, target, artifact, probe, identity, source, env, *, interpreter, probe_args=(), max_trials=48):
    context = dict(source_fingerprint=source, probe=identity, probe_args=list(probe_args), environment=env,
                   interpreter=interpreter)
    trials = Trials(artifact / 'reduction', [str(probe), *probe_args], ROOT, env)
    result = reduce_history(cases, target, trials, max_trials=max_trials, context=context)
    if (fingerprint() != source or native_identity(probe, 'wire_scene') != identity or
            interpreter_identity() != interpreter):
        result.update(status='identity_changed', verified=False, deletion_minimal=False)
        write_json(trials.root / 'reduction.json', result)
    return result


def replay_reduction(recipe, artifact, probe):
    record = json.loads(read_bounded(recipe, 1024*1024), object_pairs_hook=unique_object)
    context = record['context']
    identity, source, env = native_identity(probe, 'wire_scene'), fingerprint(), environment()
    interpreter = interpreter_identity()
    if (record['schema_version'] != 1 or not record['verified'] or
            context['source_fingerprint'] != source or context['probe'] != identity or context['environment'] != env or
            context.get('interpreter') != interpreter):
        raise ValueError('reduction source, probe, environment, interpreter or verification differs')
    flags = context['probe_args']
    if flags not in ([], ['--test-drop-marker']):
        raise ValueError('unsupported native reduction mode')
    if not 1 <= len(record['cases']) <= 1023:
        raise ValueError('reduction history count limit')
    cases, size = [], 0
    for item in record['cases']:
        digest = item['sha256']
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid reduction payload identity')
        path = recipe.parent / 'blobs' / (digest + '.ompf')
        payload = read_bounded(path, 1024*1024); size += len(payload)
        if size > 64*1024*1024 or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError('changed or oversized reduction payload')
        if type(item['origin']) is not int or not isinstance(item['name'], str):
            raise ValueError('invalid reduction case identity')
        cases.append(Case(item['origin'], item['name'], payload))
    if len({c.origin for c in cases}) != len(cases):
        raise ValueError('duplicate reduction origin')
    trials = Trials(artifact / 'replay', [str(probe), *flags], ROOT, env)
    result = trials.evaluate(cases)
    reproduced = (result['outcome'] == 'mismatch' and
                  canonical(result['mismatch']['signature']) == canonical(record['target']))
    unchanged = (fingerprint() == source and native_identity(probe, 'wire_scene') == identity and
                 interpreter_identity() == interpreter)
    write_json(artifact / 'replay.json', dict(reproduced=reproduced and unchanged, identity_unchanged=unchanged,
                                            recipe=str(recipe.resolve()), trial=result))
    print('Reduced discrepancy reproduced' if reproduced and unchanged else 'Reduced discrepancy NOT reproduced')
    # A reproduced defect remains a failing check, never a green qualification.
    return 1 if reproduced and unchanged else 2


def main():
    if sys.flags.optimize:
        raise RuntimeError('graph qualification requires Python assertions enabled')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--replay-reduction', type=Path)
    parser.add_argument('--test-reducer-fault', action='store_true', help='deliberate native reporting defect; self-test only')
    parser.add_argument('--reduction-trials', type=int, default=48, choices=range(2, 257), metavar='2..256')
    args = parser.parse_args()
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    artifact = Path(tempfile.mkdtemp(prefix='graph-', dir=args.artifact_root.resolve()))
    print('Artifacts:', artifact, flush=True)
    if args.replay_reduction:
        return replay_reduction(args.replay_reduction, artifact, args.probe.resolve())
    seeds = record_seeds(artifact, args.seed, 1)
    probe = args.probe.resolve()
    identity = native_identity(probe, 'wire_scene')
    source = fingerprint()
    model, records, files, cases = Model(), [], [], []
    probe_args = ['--test-drop-marker'] if args.test_reducer_fault else []
    inputs = reduction_witness if args.test_reducer_fault else scenarios
    for index, (name, delta) in enumerate(inputs(seeds['writer_seeds'][0])):
        payload = encode(delta)
        cases.append(Case(index, name, payload))
        path = artifact / f'{index:04d}-{name}.ompf'
        path.write_bytes(payload)
        before = model.state, model.source_format
        try:
            state = model.apply(payload, delivered=True)
            result = 'ok'
        except GraphError as error:
            result, state = error.kind, model.state
            assert (state, model.source_format) == before
        records.append({'name': name, 'file': path.name, 'sha256': hashlib.sha256(payload).hexdigest(),
                        'delta': input_metadata(asdict(delta)), 'expected': {'result': result,
                            'scene': {'graph': typed(shadowed_projection(state)), 'source_format': model.source_format}}})
        files.append(str(path))
    write_json(artifact / 'cases.json', records)
    command = [str(probe), *probe_args, *files]
    env = environment()
    evidence = {'source_fingerprint': source, 'probe': identity, 'command': command, 'environment': env, 'comparison': 'pending',
                'interpreter': interpreter_identity()}
    write_json(artifact / 'evidence.json', evidence)
    result = run_process(command, cwd=ROOT, output=artifact / 'native.log', timeout_ms=45000,
                         max_output_bytes=8*1024*1024, environment=env)
    evidence['result'] = asdict(result)
    write_json(artifact / 'evidence.json', evidence)
    assert result.outcome == 'passed' and not result.output_truncated and result.cleanup_complete, result
    actual = [json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
              for line in (artifact / 'native.log').read_text().splitlines()]
    assert len(actual) == len(records), (len(actual), len(records))
    for row in actual:
        validate_record(row)
    for index, (record, observed) in enumerate(zip(records, actual)):
        if json.dumps(observed, sort_keys=True) != json.dumps(record['expected'], sort_keys=True):
            evidence.update(comparison='failed', mismatch_index=index)
            write_json(artifact / 'evidence.json', evidence)
            write_json(artifact / 'mismatch.json', {'index': index, 'case': record, 'actual': observed})
            target = mismatch(cases, [r['expected'] for r in records], actual)['signature']
            try:
                reduced = reduce_failure(cases[:index + 1], target, artifact, probe, identity, source, env,
                                         interpreter=evidence['interpreter'], max_trials=args.reduction_trials,
                                         probe_args=probe_args)
                evidence['reduction'] = {k: reduced[k] for k in ('status', 'verified', 'attempts', 'deletion_minimal')}
            except Exception as error:
                evidence['reduction'] = dict(status='failed', error=type(error).__name__)
            write_json(artifact / 'evidence.json', evidence)
            raise AssertionError(f"graph mismatch at {index} {record['name']}; see {artifact / 'mismatch.json'}")
    assert (fingerprint() == source and native_identity(probe, 'wire_scene') == identity and
            interpreter_identity() == evidence['interpreter'])
    evidence.update(comparison='passed', comparisons=len(records), source_unchanged=True, probe_unchanged=True)
    write_json(artifact / 'evidence.json', evidence)
    print(f'{len(records)} complete native graph transitions match independent model; retained snapshots unchanged')


if __name__ == '__main__':
    raise SystemExit(main())
