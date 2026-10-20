# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent final-state witnesses for graph reset/diff projection."""
import copy
import importlib.util
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parents[1] / "reference_service"
for name in ("scene_wire", "scene_graph_wire", "scene_graph_model", "scene_graph_snapshot"):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, DIRECTORY / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
wire, graph, projection = (sys.modules[name] for name in
                         ("scene_graph_wire", "scene_graph_model", "scene_graph_snapshot"))


def node(identity=1, parent=0, spec_type=7, name="", order=0, path="/", **fields):
    return dict(id=identity, parent=parent, type=spec_type, name=name, order=order, path=path, fields=fields)


def signature(state):
    # This test oracle independently distinguishes Python numeric equality and
    # IEEE-754 equality; 7 != 7.0 and -0.0 != +0.0 in an authored state.
    # Order values are not observable; the sibling sequence they give is.
    records = []
    for item in sorted(graph.ranked(state), key=lambda item: item["id"]):
        values = tuple((key, type(value).__name__, struct.pack("<d", value) if type(value) is float else value)
                       for key, value in sorted(item["fields"].items()))
        records.append((item["id"], item["parent"], item["type"], item["name"], item["order"], item["path"], values))
    return tuple(records)


class SceneGraphSnapshotTests(unittest.TestCase):
    def test_path_vectors_list_modes_and_target_survivor_diffs(self):
        before = [node(primChildren=('P',)), node(2, 1, 6, 'P', 0, '/P', properties=('rel',)),
            node(3, 2, 8, 'rel', 0, '/P.rel', targetChildren=wire.Paths(('/Target',)),
                 targetPaths=wire.PathOp(True, ('/Target',))),
            node(4, 3, 9, '/Target', 0, '/P.rel[/Target]', properties=('weight',)),
            node(5, 4, 1, 'weight', 0, '/P.rel[/Target].weight', value=42)]
        after = [node(primChildren=('P',)), node(2, 1, 6, 'P', 0, '/P', properties=('weight',)),
                 node(5, 2, 1, 'weight', 0, '/P.weight', value=42)]
        self.exact_replay(before, after)
        values = (wire.Paths(()), (), wire.PathOp(), wire.PathOp(True), wire.PathOp(added=('/T',)),
                  wire.PathOp(prepended=('/T',)), wire.PathOp(deleted=('/T',)))
        for old, new in zip(values, values[1:]):
            delta = self.exact_replay([node(value=old)], [node(value=new)])
            self.assertEqual(delta.fields, (wire.Field(1, 'value', new),))
        for invalid in (wire.PathOp(True, added=('/A',)), wire.PathOp(added=('/A', '/A'))):
            self.rejected('malformed', lambda: projection.snapshot([node(value=invalid)]))

    def test_presence_value_families_snapshot_and_diff_by_representation(self):
        # A sign-only or NaN-payload change is a field change; equal bits are not.
        values = (wire.String('/Camera'), wire.String(''), wire.Strings(('/A', 'λ')), wire.Strings(()),
                  wire.TokenArray(('xformOp:transform',)), wire.real('float', 0.0), wire.real('float', -0.0),
                  wire.Real('float', (0x7fc00001,)), wire.Real('float', (0x7fc00002,)),
                  wire.real('matrix4d', *([0.0] * 16)), wire.real('matrix4d', *([0.0] * 15 + [-0.0])),
                  wire.real('vec3d', 0.0, -0.0, 1.0, array=True), wire.Real('vec4f', (), True))
        for old, new in zip(values, values[1:]):
            with self.subTest(old=old, new=new):
                delta = self.exact_replay([node(value=old)], [node(value=new)])
                self.assertEqual(delta.fields, (wire.Field(1, 'value', new),))
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(self.exact_replay([node(value=value)], [node(value=value)]).fields, ())
                model = graph.Model()
                self.assertEqual(model.apply(projection.snapshot([node(value=value)]))[0]['fields'], {'value': value})
        versions = projection.Versions().advanced([node(value=values[5])], [node(value=values[6])], 3)
        self.assertEqual(versions.fields, {(1, 'value'): 3})
        self.assertEqual(projection.Versions().advanced([node(value=values[7])], [node(value=values[7])], 3).fields, {})
        self.rejected('unsupported', lambda: projection.snapshot([node(value=wire.String('nul\0'))]))
        self.rejected('malformed', lambda: projection.snapshot([node(value=wire.Real('matrix4d', (0,) * 15))]))
        with patch.object(wire, 'MAX_RECORDS', 4):
            self.rejected('limit', lambda: projection.snapshot([node(value=wire.Strings(('a', 'b', 'c')))]))

    def test_nested_variant_snapshot_and_diff_survivor(self):
        before = [node(primChildren=('P',)), node(2, 1, 6, 'P', 0, '/P', variantSetChildren=('look',)),
            node(3, 2, 11, 'look', 9, '/P{look=}', variantChildren=('high-low',)),
            node(4, 3, 10, 'high-low', 3, '/P{look=high-low}', variantSetChildren=('detail',)),
            node(5, 4, 11, 'detail', 7, '/P{look=high-low}{detail=}', variantChildren=('1',)),
            node(6, 5, 10, '1', 13, '/P{look=high-low}{detail=1}', primChildren=('Child',)),
            node(7, 6, 6, 'Child', 17, '/P{look=high-low}{detail=1}Child', value=42)]
        model = graph.Model()
        self.assertEqual(signature(model.apply(projection.snapshot(before))), signature(before))
        after = [node(primChildren=('Survivor',)), node(7, 1, 6, 'Survivor', 17, '/Survivor', value=42)]
        delta = self.exact_replay(before, after)
        self.assertEqual({c.section for c in delta.commands if c.tag == 6}, {2, 3, 4, 5, 6})
        self.assertEqual([(c.section, c.parent) for c in delta.commands if c.tag == 5], [(7, 1)])
        invalid = copy.deepcopy(before)
        invalid[-1]['path'] = '/P{look=high-low}{detail=1}/Child'
        self.rejected('malformed', lambda: projection.snapshot(invalid))

    def test_source_format_snapshot_resume_and_unrepresentable_clear(self):
        state = [node(value=17)]
        saved = signature(state)
        for before, after in (("", "usda"), ("usda", "usdc"), ("usda", "usda"), ("", "")):
            with self.subTest(before=before, after=after):
                model = graph.Model()
                model.apply(projection.snapshot(state, source_format=before), 1)
                self.assertEqual(model.source_format, before)
                delta = projection.difference(state, state, 1,
                    before_source_format=before, after_source_format=after)
                self.assertEqual(wire.decode(delta).source_format, after if before != after else "")
                model.apply(delta, 2)
                self.assertEqual(model.source_format, after)
                self.assertEqual(signature(model.state), saved)
        self.rejected("unsupported", lambda: projection.difference(state, state, 1,
            before_source_format="usda", after_source_format=""))
        self.assertEqual(signature(state), saved)
        for invalid in (None, "bad\0format", "\ud800"):
            self.rejected("malformed", lambda: projection.snapshot(state, source_format=invalid))
        with patch.object(wire, "MAX_STRING_BYTES", 8):
            self.rejected("limit", lambda: projection.snapshot(state, source_format="usda"))
        with patch.object(wire, "MAX_STRING_BYTES", 9):
            self.assertEqual(wire.decode(projection.snapshot(state, source_format="usda")).source_format, "usda")

    def exact_replay(self, before, after, base=11):
        saved = signature(before), signature(after)
        encoded = projection.difference(before, after, base)
        model = graph.Model()
        model.apply(projection.snapshot(before), base)
        self.assertEqual(signature(model.apply(encoded, base + 1)), signature(after))
        self.assertEqual((signature(before), signature(after)), saved)
        return wire.decode(encoded)

    def rejected(self, kind, operation):
        with self.assertRaises(graph.GraphError) as error:
            operation()
        self.assertEqual(error.exception.kind, kind, str(error.exception))

    def test_complete_snapshot_records_root_sparse_order_and_owned_empty_fields(self):
        state = [node(9, 1, 6, "Later", 2**32-1, "/Later", primChildren=(), value=7.0),
                 node(order=17, primChildren=("First", "Later"), custom=17),
                 node(5, 1, 6, "First", 7, "/First", properties=("ns:a", "rel")),
                 node(11, 5, 8, "rel", 99, "/First.rel", value=-0.0),
                 node(10, 5, 1, "ns:a", 13, "/First.ns:a", value=-27)]
        saved = copy.deepcopy(state)
        data = projection.snapshot(state)
        delta = wire.decode(data)
        self.assertTrue(delta.is_diff)
        self.assertEqual(delta.base_version, 0)
        # A nonroot section's order is its index among its parent's children; the root keeps its stored order.
        self.assertEqual(delta.commands, (
            wire.Command(5, 1, order=17, spec_type=7),
            wire.Command(5, 5, parent=1, name="First", spec_type=6, order=0),
            wire.Command(5, 9, parent=1, name="Later", spec_type=6, order=1),
            wire.Command(5, 10, parent=5, name="ns:a", spec_type=1, order=0),
            wire.Command(5, 11, parent=5, name="rel", spec_type=8, order=1)))
        self.assertEqual([(f.section, f.key, f.value) for f in delta.fields],
                         [(1, "custom", 17), (1, "primChildren", ("First", "Later")),
                          (5, "properties", ("ns:a", "rel")), (9, "primChildren", ()),
                          (9, "value", 7.0), (10, "value", -27), (11, "value", -0.0)])
        self.assertEqual(signature(graph.Model().apply(data)), signature(saved))
        self.assertEqual(signature(state), signature(saved))
        # Deterministic projection ignores input dictionary/list iteration order.
        reordered = [dict(reversed(tuple(item.items()))) for item in reversed(state)]
        for item in reordered:
            item["fields"] = dict(reversed(tuple(item["fields"].items())))
        self.assertEqual(projection.snapshot(reordered), data)
        state.clear()
        self.assertEqual(signature(graph.Model().apply(data)), signature(saved))

    def test_diff_survivor_move_ancestor_delete_and_inactive_child_presence(self):
        before = [node(order=13, primChildren=("P", "Q")),
                  node(2, 1, 6, "P", 5, "/P", primChildren=("C",), properties=("a",)),
                  node(3, 2, 6, "C", 9, "/P/C", value=17, primChildren=()),
                  node(4, 2, 1, "a", 50, "/P.a", value=91),
                  node(5, 1, 6, "Q", 20, "/Q", primChildren=(), properties=())]
        after = [node(order=13, primChildren=("Q",)),
                 node(3, 5, 6, "Moved", 2**32-1, "/Q/Moved", value=17, primChildren=()),
                 node(5, 1, 6, "Q", 20, "/Q", primChildren=("Moved",), properties=())]
        delta = self.exact_replay(before, after)
        self.assertEqual(delta.commands, (wire.Command(6, 2), wire.Command(6, 4),
                                         wire.Command(5, 3, 5, "Moved", 6, 0), wire.Command(5, 5, 1, "Q", 6, 0)))
        self.assertEqual(delta.fields, (wire.Field(1, "primChildren", ("Q",)),
                                       wire.Field(5, "primChildren", ("Moved",))))
        self.assertEqual(delta.base_version, 11)
        self.assertNotIn(3, [c.section for c in delta.commands if c.tag == 6])

    def test_diff_exact_numeric_type_signed_zero_deletions_and_new_fields(self):
        before = [node(integer=7, negative=-0.0, positive=0.0, remove=42, stable=1.25, tokens=())]
        after = [node(integer=7.0, negative=0.0, positive=-0.0, stable=1.25, tokens=("a",), added=-11)]
        delta = self.exact_replay(before, after)
        self.assertEqual(delta.commands, ())
        values = {field.key: field.value for field in delta.fields}
        self.assertEqual(set(values), {"integer", "negative", "positive", "remove", "tokens", "added"})
        self.assertIs(type(values["integer"]), float)
        self.assertEqual(struct.pack("<d", values["negative"]), bytes.fromhex("0000000000000000"))
        self.assertEqual(struct.pack("<d", values["positive"]), bytes.fromhex("0000000000000080"))
        self.assertIsNone(values["remove"])
        self.assertEqual(values["tokens"], ("a",))
        for first, second, value in (([node()], [node(primChildren=())], ()),
                                     ([node(primChildren=())], [node()], None)):
            change = self.exact_replay(first, second)
            self.assertEqual(change.fields, (wire.Field(1, "primChildren", value),))
        self.assertEqual(self.exact_replay(after, copy.deepcopy(after)).fields, ())

    def test_public_scalar_types_survive_snapshot_and_diff_without_numeric_coercion(self):
        before = [node(a=1, b=True, c=wire.Specifier(0), d=wire.Variability(0),
                       token=wire.Token(''), vector=(), stable=wire.Token('Scope'))]
        after = [node(a=True, b=1, c=wire.Variability(0), d=wire.Specifier(0),
                      token=wire.Token('λ'), vector=wire.Token(''), stable=wire.Token('Scope'))]
        delta = self.exact_replay(before, after)
        self.assertEqual({field.key for field in delta.fields}, {'a', 'b', 'c', 'd', 'token', 'vector'})
        self.assertEqual(self.exact_replay(after, after).fields, ())
        for value in (wire.Specifier(True), wire.Variability(2), wire.Token('bad\0token')):
            self.rejected('malformed', lambda value=value: projection.snapshot([node(value=value)]))
        state = [node(k=wire.Token('λ'))]
        with patch.object(wire, 'MAX_STRING_BYTES', 3):
            self.assertEqual(graph.Model().apply(projection.snapshot(state)), state)
        with patch.object(wire, 'MAX_STRING_BYTES', 2):
            self.rejected('limit', lambda: projection.snapshot(state))

    def test_new_ids_same_path_and_supported_type_transition(self):
        before = [node(primChildren=("P",)), node(2, 1, 6, "P", 17, "/P", properties=("a",)),
                  node(3, 2, 1, "a", 5, "/P.a", value=11, stale=8)]
        after = [node(primChildren=("P",)), node(2, 1, 6, "P", 17, "/P", properties=("a",)),
                 node(9, 2, 8, "a", 5, "/P.a", value=22)]
        delta = self.exact_replay(before, after)
        self.assertEqual(delta.commands, (wire.Command(6, 3), wire.Command(5, 9, 2, "a", 8, 0)))
        self.assertEqual(delta.fields, (wire.Field(9, "value", 22),))
        same_identity = copy.deepcopy(after)
        same_identity[-1]["type"] = 1
        self.assertEqual(self.exact_replay(after, same_identity).commands,
                         (wire.Command(5, 9, 2, "a", 1, 0),))
        # Valid endpoint states do not authorize silently changing an existing
        # Prim carrying an authored child field into an Attribute. The current
        # model forbids that child-field record even when it requests erasure.
        original = [node(primChildren=("P",)), node(2, 1, 6, "P", 0, "/P", primChildren=("C",)),
                    node(3, 2, 6, "C", 0, "/P/C", primChildren=())]
        target = [node(primChildren=("P",)), node(2, 1, 6, "P", 0, "/P", properties=("C",)),
                  node(3, 2, 1, "C", 0, "/P.C")]
        saved = signature(original), signature(target)
        self.rejected("malformed", lambda: projection.difference(original, target, 1))
        self.assertEqual((signature(original), signature(target)), saved)

    def test_bad_state_is_not_silently_normalized(self):
        valid = [node(primChildren=("A", "B")), node(2, 1, 6, "A", 1, "/A"),
                 node(3, 1, 6, "B", 2, "/B")]
        wrong_path = copy.deepcopy(valid); wrong_path[1]["path"] = "/guessed"
        missing_children = copy.deepcopy(valid); missing_children[0]["fields"].clear()
        wrong_order = copy.deepcopy(valid); wrong_order[0]["fields"]["primChildren"] = ("B", "A")
        duplicate = copy.deepcopy(valid); duplicate.append(copy.deepcopy(valid[1]))
        wrong_shape = copy.deepcopy(valid); wrong_shape[0]["extra"] = 1
        bool_id = copy.deepcopy(valid); bool_id[0]["id"] = True
        stored_erase = copy.deepcopy(valid); stored_erase[0]["fields"]["value"] = None
        for state in ([], wrong_path, missing_children, wrong_order, duplicate, wrong_shape, bool_id, stored_erase):
            with self.subTest(state=state):
                saved = copy.deepcopy(state)
                self.rejected("malformed", lambda: projection.snapshot(state))
                self.assertEqual(state, saved)
        cycle = copy.deepcopy(valid)
        cycle[1]["parent"], cycle[2]["parent"] = 3, 2
        self.rejected("unsupported", lambda: projection.snapshot(cycle))
        collision = copy.deepcopy(valid); collision[2]["order"] = 1
        self.rejected("unsupported", lambda: projection.snapshot(collision))
        missing = copy.deepcopy(valid); missing[2]["parent"] = 99
        self.rejected("unsupported", lambda: projection.snapshot(missing))
        for base in (0, -1, True, 2**64, "1"):
            self.rejected("malformed", lambda: projection.difference(valid, valid, base))
        self.rejected("malformed", lambda: projection.snapshot([node(timeSamples=())]))
        self.rejected("unsupported", lambda: projection.snapshot([node(value=float("inf"))]))

    def test_shared_input_output_model_and_history_limits(self):
        state = [node(primChildren=("P",)), node(2, 1, 6, "P", 17, "/P", properties=(), value=-0.0)]
        saved = signature(state)
        encoded = projection.snapshot(state)
        for module, setting, maximum in ((graph, "MAX_SECTIONS", 1), (graph, "MAX_FIELDS", 2),
                                         (wire, "MAX_RECORDS", 5), (wire, "MAX_STRING_BYTES", 1),
                                         (graph, "MAX_PATH_BYTES", 2), (graph, "MAX_DEPTH", 0),
                                         (graph, "MAX_WORK", 1), (graph, "MAX_HISTORY_BYTES", 1)):
            with self.subTest(limit=setting), patch.object(module, setting, maximum):
                self.rejected("limit", lambda: projection.snapshot(state))
                self.rejected("limit", lambda: projection.difference(state, state, 1))
                self.assertEqual(signature(state), saved)
        with patch.object(wire, "MAX_BYTES", len(encoded)):
            self.assertEqual(projection.snapshot(state), encoded)
        with patch.object(wire, "MAX_BYTES", len(encoded) - 1):
            self.rejected("limit", lambda: projection.snapshot(state))
        # An output diff can exceed the cap even though each complete endpoint
        # snapshot fits: all old fields become deletion records plus new ones.
        before = [node(**{f"before{i}": i for i in range(20)})]
        after = [node(**{f"after{i}": i for i in range(20)})]
        byte_limit = max(len(projection.snapshot(before)), len(projection.snapshot(after)))
        with patch.object(wire, "MAX_BYTES", byte_limit):
            projection.snapshot(before); projection.snapshot(after)
            self.rejected("limit", lambda: projection.difference(before, after, 1))
        with patch.object(wire, "MAX_RECORDS", 21):
            projection.snapshot(before); projection.snapshot(after)
            self.rejected("limit", lambda: projection.difference(before, after, 1))
        self.assertEqual(signature(state), saved)


if __name__ == "__main__":
    unittest.main()
