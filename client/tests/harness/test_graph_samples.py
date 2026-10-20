# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent sample framing, phase/presence and resume witnesses."""
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reference_service'))
import scene_graph_wire as wire
import scene_graph_snapshot as projection
from scene_graph_model import Model, GraphError
from scene_graph_wire import Delta, Field as F, Sample as S, TimeSamples as T, Token, encode, decode, typed_value
from test_scene_graph_wire import root, record, location, target, replace


def samples(model):
    return model.state[0]['fields'].get('timeSamples')


class GraphSampleTests(unittest.TestCase):
    def test_literal_sample_map_bytes_and_shared_limits(self):
        value = T(((-0.0, 17), (2.0, Token('hi'))))
        raw = (b'\x2e\0' + struct.pack('<Qd', 2, -0.0) + b'\x03\0' + struct.pack('<i', 17) +
               struct.pack('<d', 2.0) + b'\x0b\0' + struct.pack('<I', 2) + b'hi')
        self.assertEqual(wire._value(value, wire._Budget()), raw)
        self.assertEqual(typed_value(wire._decode_value(raw, wire._Budget())), typed_value(value))
        self.assertEqual(wire._value(T(), wire._Budget()), b'\x2e\0' + bytes(8))
        for invalid in (raw[:-1], raw + b'x', raw[:9], replace(raw, 2, 'Q', 2**64-1),
                        replace(raw, 10, 'd', float('nan')), replace(raw, 24, 'd', 0.0)):
            with self.subTest(raw=invalid), self.assertRaises(wire.GraphWireError):
                wire._decode_value(invalid, wire._Budget())
        for value in (T(((0.0, 1), (-0.0, 2))), T(((float('nan'), 1),)), T(((1.0, None),)),
                      T(((1.0, T()),)), T(((1, 7),))):
            with self.assertRaises(wire.GraphWireError):
                wire._value(value, wire._Budget())
        for key, limit in (('MAX_RECORDS', 1), ('MAX_STRING_BYTES', 1)):
            with patch.object(wire, key, limit):
                with self.assertRaises(wire.GraphWireError) as error:
                    wire._decode_value(raw, wire._Budget())
                self.assertEqual(error.exception.kind, 'limit')
        unsorted = b'\x2e\0' + struct.pack('<Q', 2) + raw[24:] + raw[10:24]
        self.assertEqual(typed_value(wire._decode_value(unsorted, wire._Budget())),
                         typed_value(T(((-0.0, 17), (2.0, Token('hi'))))))

    def test_sample_table_layout_defaults_and_invalid_records(self):
        delta = Delta(samples=(S(7, -0.0, 17), S(7, float('inf'), None)))
        data = encode(delta)
        table = record(data, location(data, root(data), 5))
        self.assertEqual(struct.unpack_from('<Q', data, location(data, table, 0))[0], 7)
        self.assertEqual(data[location(data, table, 1):location(data, table, 1)+8], struct.pack('<d', -0.0))
        blob = target(data, location(data, table, 2))
        self.assertEqual(data[blob:blob+10], struct.pack('<IHi', 6, 3, 17))
        decoded = decode(data)
        self.assertEqual(encode(decoded), data)
        vtable = table - struct.unpack_from('<i', data, table)[0]
        omitted = replace(data, vtable + 6, 'H', 0)
        self.assertEqual(struct.pack('<d', decode(omitted).samples[0].time), bytes(8))
        for broken in (replace(data, location(data, table, 0), 'Q', 0),
                       replace(data, location(data, table, 1), 'd', float('nan'))):
            with self.assertRaises(wire.GraphWireError) as error:
                decode(broken)
            self.assertEqual(error.exception.kind, 'malformed')
        external = replace(data, vtable + 10, 'H', location(data, table, 0)-table)
        with self.assertRaises(wire.GraphWireError) as error:
            decode(external)
        self.assertEqual(error.exception.kind, 'unsupported')
        with patch.object(wire, 'MAX_RECORDS', 1):
            for operation in (lambda: encode(delta), lambda: decode(data)):
                with self.assertRaises(wire.GraphWireError) as error:
                    operation()
                self.assertEqual(error.exception.kind, 'limit')

    def test_whole_map_phase_repeated_keys_and_presence(self):
        model = Model()
        model.apply(encode(Delta(fields=(F(1, 'timeSamples', T(((1.0, 1), (2.0, 2))), 99),),
            samples=(S(1, 1.0, 10), S(1, 2.0, None), S(1, 3.0, True), S(1, 3.0, 31)))))
        self.assertEqual(samples(model), T(((1.0, 10), (3.0, 31))))
        model.apply(encode(Delta(fields=(F(1, 'timeSamples', T()),), samples=(S(1, 99.0, None),))))
        self.assertEqual(samples(model), T())
        model.apply(encode(Delta(samples=(S(1, 100.0, None),))))
        self.assertEqual(samples(model), T())
        model.apply(encode(Delta(samples=(S(1, 1.0, 7), S(1, 1.0, None)))))
        self.assertIsNone(samples(model))
        model.apply(encode(Delta(fields=(F(1, 'timeSamples', None),), samples=(S(1, 2.0, False),))))
        self.assertEqual(typed_value(samples(model)), typed_value(T(((2.0, False),))))
        before = model.state, model.history
        for delta in (Delta(samples=(S(1, 2.0, 9),), fields=(F(99, 'timeSamples', 7),)),
                      Delta(fields=(F(1, 'timeSamples', 7),)),
                      Delta(fields=(F(1, 'timeSamples', 7), F(1, 'timeSamples', T())))):
            with self.assertRaises(GraphError):
                model.apply(encode(delta))
            self.assertEqual((model.state, model.history), before)
        # A sample for a missing section commits as orphan data and is never projected.
        model.apply(encode(Delta(samples=(S(99, 1.0, 2),))))
        self.assertEqual((model.state, model.orphans), (before[0], {99: {'timeSamples': T(((1.0, 2),))}}))

    def test_signed_zero_key_bits_and_snapshot_diff_values(self):
        model = Model()
        model.apply(encode(Delta(samples=(S(1, -0.0, True), S(1, 0.0, 1), S(1, float('inf'), Token('end'))))))
        self.assertEqual(typed_value(samples(model))['value'][0],
                         dict(time_bits=2**63, value=dict(type='int32', value=1)))
        before = model.state
        model.apply(encode(Delta(fields=(F(1, 'timeSamples', T(((0.0, True),))),))))
        after = model.state
        diff = projection.difference(before, after, 1)
        self.assertEqual(decode(diff).fields, (F(1, 'timeSamples', T(((0.0, True),))),))
        fresh = Model(); fresh.apply(projection.snapshot(before)); fresh.apply(diff)
        self.assertEqual(typed_value(samples(fresh)), typed_value(samples(model)))
        self.assertEqual(typed_value(samples(fresh))['value'][0]['time_bits'], 0)
        model.apply(encode(Delta(fields=(F(1, 'timeSamples', T()),))))
        fresh.apply(projection.difference(after, model.state, 2))
        self.assertEqual(samples(fresh), T())
        after = model.state
        model.apply(encode(Delta(fields=(F(1, 'timeSamples', None),))))
        fresh.apply(projection.difference(after, model.state, 3))
        self.assertIsNone(samples(fresh))

    def test_diffs_preserve_zero_sign_only_and_scalar_type_only_changes(self):
        for old, new in ((T(((-0.0, 7),)), T(((0.0, 7),))),
                         (T(((1.0, True),)), T(((1.0, 1),)))):
            with self.subTest(old=old, new=new):
                model = Model()
                model.apply(encode(Delta(fields=(F(1, 'timeSamples', old),))))
                before = model.state
                model.apply(encode(Delta(fields=(F(1, 'timeSamples', new),))))
                data = projection.difference(before, model.state, 1)
                self.assertEqual(decode(data).fields, (F(1, 'timeSamples', new),))
                restored = Model(); restored.apply(projection.snapshot(before)); restored.apply(data)
                self.assertEqual(typed_value(samples(restored)), typed_value(new))

    def test_snapshot_aggregate_sample_and_string_bounds(self):
        model = Model()
        model.apply(encode(Delta(fields=(F(1, 'timeSamples', T(((1.0, Token('abc')), (2.0, ('x','y'))))),))))
        for key, limit in (('MAX_RECORDS', 5), ('MAX_STRING_BYTES', 15)):
            with patch.object(wire, key, limit):
                with self.assertRaises(GraphError) as error:
                    projection.snapshot(model.state)
                self.assertEqual(error.exception.kind, 'limit')
