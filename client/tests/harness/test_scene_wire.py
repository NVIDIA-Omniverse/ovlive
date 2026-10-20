# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No gRPC, production codec, generated FlatBuffers, or third-party imports."""
import importlib.util
from pathlib import Path
import random
import struct
import sys
import unittest

PATH = Path(__file__).resolve().parents[1] / "reference_service" / "scene_wire.py"
SPEC = importlib.util.spec_from_file_location("reference_scene_wire", PATH)
wire = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = wire
SPEC.loader.exec_module(wire)


def put(data, position, fmt, value):
    result = bytearray(data)
    struct.pack_into("<" + fmt, result, position, value)
    return bytes(result)


def field_locations(data):
    """Locate our handbuilt fixture's slots without the codec's parser."""
    # Encoder layout: 16-byte framing, root at relative32, field vector at56.
    vector = 16 + 56
    positions = []
    for i in range(struct.unpack_from("<I", data, vector)[0]):
        slot = vector + 4 + i * 4
        positions.append(slot + struct.unpack_from("<I", data, slot)[0])
    return positions


class SceneWireTests(unittest.TestCase):
    def test_provider_literal_noop(self):
        # Pinned provider gen_noop_delta output, independently supplied literal.
        words = [0x46504D4F, 1, 16, 0, 24, 0x46444F4F, 0x00140010,
                 0, 0x00080004, 0x0010000C, 16, 28, 20, 12, 4, 0, 0, 0, 0]
        data = struct.pack("<19I", *words)
        self.assertEqual(wire.decode_update(data), {})
        for n in range(len(data)):
            with self.assertRaises(wire.WireError, msg=str(n)):
                wire.decode_update(data[:n])

    def test_manual_layout_and_values(self):
        data = wire.update({"count": -123}, base_version=99)
        self.assertEqual(data[:16], b"OMPF\x01\0\0\0\x10\0\0\0\0\0\0\0")
        self.assertEqual(data[20:24], b"OODF")
        self.assertEqual(struct.unpack_from("<Q", data, 16 + 32 + 8)[0], 99)
        record, = field_locations(data)
        self.assertEqual(struct.unpack_from("<Q", data, record + 8)[0], 1)
        value = record + 20 + struct.unpack_from("<I", data, record + 20)[0]
        self.assertEqual(data[value:value + 10], b"\x06\0\0\0\x03\0\x85\xff\xff\xff")
        expected = {"a": -(2**31), "b": 2**31 - 1, "é": 0, "gone": None}
        delta = wire.decode_envelope(wire.update(expected))
        self.assertEqual([(f.name, f.value, f.set_order) for f in delta.fields],
                         [(key, value, i) for i, (key, value) in enumerate(expected.items())])
        self.assertEqual(wire.decode_update(wire.update(expected)), expected)

    def test_reset_aggregate_and_model(self):
        model = wire.Model()
        self.assertEqual(model.apply(wire.snapshot({"a": 1, "b": 2}), 10), {"a": 1, "b": 2})
        self.assertEqual(model.apply(wire.update({"a": 1}), 11), {"a": 1, "b": 2})
        self.assertEqual(model.apply(wire.aggregate({"b": None, "c": 3}, 11), 12), {"a": 1, "c": 3})
        self.assertEqual(model.commits, 3)
        self.assertEqual(model.edits, 5)
        self.assertTrue(model.history[2][1].is_diff)
        self.assertEqual(model.history[2][1].base_version, 11)
        borrowed = model.state
        borrowed.clear()
        self.assertEqual(model.state, {"a": 1, "c": 3})
        self.assertEqual(model.apply(wire.snapshot({"d": 4}), 13), {"d": 4})
        self.assertEqual(model.apply(wire.empty_snapshot(), 14), {})
        self.assertEqual(model.commits, 5)
        self.assertEqual(model.edits, 6)
        with self.assertRaises(wire.WireError):
            wire.decode_update(wire.empty_snapshot())
        with self.assertRaises(wire.WireError):
            wire.aggregate({}, 0)

    def test_double_wire_bytes_and_exact_numeric_types(self):
        # Exact scalar type/version header and IEEE-754 little-endian bytes,
        # independent of the oracle's value decoder.
        for number, payload in (
            (24.0, bytes.fromhex("09000000000000003840")),
            (-0.0, bytes.fromhex("09000000000000000080")),
            (-1.5, bytes.fromhex("0900000000000000f8bf")),
        ):
            with self.subTest(number=number):
                data = wire.update({"framesPerSecond": number})
                record, = field_locations(data)
                vector = record + 20 + struct.unpack_from("<I", data, record + 20)[0]
                self.assertEqual(data[vector:vector + 14], b"\x0a\0\0\0" + payload)
                decoded = wire.decode_update(data)["framesPerSecond"]
                self.assertIs(type(decoded), float)
                self.assertEqual(struct.pack("<d", decoded), payload[2:])
        mixed = wire.decode_update(wire.update({"integer": 24, "double": 24.0, "erase": None}))
        self.assertIs(type(mixed["integer"]), int)
        self.assertIs(type(mixed["double"]), float)
        self.assertIsNone(mixed["erase"])

    def test_double_malformed_and_atomic_model(self):
        data = wire.update({"integer": 99, "startTimeCode": -0.0})
        _, record = field_locations(data)
        vector = record + 20 + struct.unpack_from("<I", data, record + 20)[0]
        # Fail only the second record to prove the earlier valid record is not
        # partially applied. Also reject another scalar/array/version tag.
        malformed = [put(data, vector, "I", length) for length in (0, 6, 9, 11)]
        malformed += [put(data, vector + 4, "H", tag) for tag in (3, 8, 0x89, 0x109)]
        malformed += [put(data, vector + 6, "Q", bits) for bits in
                      (0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000001)]
        malformed += [data[:n] for n in range(vector + 4, len(data))]
        model = wire.Model()
        model.apply(wire.snapshot({"integer": 1, "startTimeCode": -0.0}), 10)
        before = model.state, model.history, model.commits, model.edits
        for bad in malformed:
            with self.subTest(data=bad.hex()):
                with self.assertRaises(wire.WireError):
                    model.apply(bad, 11)
                self.assertEqual((model.state, model.history, model.commits, model.edits), before)
                self.assertEqual(struct.pack("<d", model.state["startTimeCode"]), bytes.fromhex("0000000000000080"))
        self.assertEqual(model.apply(wire.aggregate({"integer": None, "timeCodesPerSecond": 48.0}, 10), 11),
                         {"startTimeCode": -0.0, "timeCodesPerSecond": 48.0})
        self.assertEqual(model.apply(wire.update({"startTimeCode": 1.25}), 12),
                         {"startTimeCode": 1.25, "timeCodesPerSecond": 48.0})
        self.assertEqual((model.commits, model.edits), (3, 5))

    def test_duplicate_edits_preserved(self):
        data = wire.update({"a": 1, "b": 2})
        first, second = field_locations(data)
        key = second + 16 + struct.unpack_from("<I", data, second + 16)[0] + 4
        data = put(data, key, "B", ord("a"))
        # Notification setOrder does not change scene application wire order.
        data = put(data, first + 24, "Q", 100)
        delta = wire.decode_envelope(data)
        self.assertEqual([(f.name, f.value) for f in delta.fields], [("a", 1), ("a", 2)])
        self.assertEqual(wire.decode_update(data), {"a": 2})
        model = wire.Model()
        self.assertEqual(model.apply(data), {"a": 2})
        self.assertEqual(model.edits, 2)
        model.apply(wire.update({"a": 2}))
        self.assertEqual((model.commits, model.edits), (2, 3))

    def test_truncation_and_malformed(self):
        data = wire.update({"a": 9})
        record, = field_locations(data)
        name = record + 16 + struct.unpack_from("<I", data, record + 16)[0]
        value = record + 20 + struct.unpack_from("<I", data, record + 20)[0]
        vt = record - struct.unpack_from("<i", data, record)[0]
        for n in range(len(data)):
            with self.assertRaises(wire.WireError, msg=f"truncate {n}"):
                wire.decode_envelope(data[:n])
        cases = [(0, "I", 0), (4, "I", 0), (4, "I", 2), (8, "Q", 0),
                 (8, "Q", len(data)), (16, "I", 0xFFFFFFFF), (20, "I", 0),
                 (48, "i", 0), (52, "B", 2), (72, "I", 0xFFFFFFFF),
                 (76, "I", 0), (record + 8, "Q", 2), (record + 16, "I", 0),
                 (vt, "H", 3), (vt + 2, "H", 4), (vt + 4, "H", 31),
                 (name, "I", 0xFFFFFFFF), (name + 4, "B", 0),
                 (name + 4, "B", 0xFF), (name + 5, "B", 1),
                 (value, "I", 0), (value, "I", 5), (value + 4, "H", 0x103),
                 (value + 4, "H", 0x83)]
        for offset, fmt, number in cases:
            with self.subTest(offset=offset, fmt=fmt, number=number):
                with self.assertRaises(wire.WireError):
                    wire.decode_envelope(put(data, offset, fmt, number))

    def test_subset_rejections_and_limits(self):
        data = wire.update({"a": 7})
        record, = field_locations(data)
        vt = record - struct.unpack_from("<i", data, record)[0]
        # Reuse the valid existing setOrder slot as external size/index metadata.
        external = put(put(data, vt + 10, "H", 24), record + 24, "Q", 6)
        external_index = put(put(data, vt + 12, "H", 24), record + 24, "I", 1)
        # Delta vtable begins at part offset8 (absolute24). Reuse a bounded
        # existing vector offset so these failures are semantic, not truncation.
        samples = put(data, 24 + 4 + 5 * 2, "H", 16)
        commands = put(put(data, 24 + 4 + 2 * 2, "H", 16), 24 + 4 + 3 * 2, "H", 16)
        unpaired = put(data, 24 + 4 + 2 * 2, "H", 16)
        for bad in (external, external_index, samples, commands, unpaired):
            with self.assertRaises(wire.WireError):
                wire.decode_envelope(bad)
        with self.assertRaises(wire.WireError):
            wire.update({f"f{i}": i for i in range(wire.MAX_FIELDS + 1)})
        with self.assertRaises(wire.WireError):
            wire.update({"x" * wire.MAX_BYTES: 1})
        with self.assertRaises(wire.WireError):
            wire.decode_envelope(data + b"x" * wire.MAX_BYTES)

    def test_negative_vtable_displacement_and_padding(self):
        data = wire.update({"a": 7, "b": 8})
        first, second = field_locations(data)
        # Legal shared vtable after the first object; parser must use signed int32.
        shared = second - struct.unpack_from("<i", data, second)[0]
        data = put(data, first, "i", first - shared)
        self.assertLess(first - shared, 0)
        self.assertEqual(wire.decode_update(data), {"a": 7, "b": 8})
        padded = put(data[:16] + b"xyz" + data[16:], 8, "Q", 19)
        self.assertEqual(wire.decode_update(padded), {"a": 7, "b": 8})

    def test_input_rejection_and_atomic_model(self):
        for fields in ({"": 1}, {"a\0b": 1}, {"x": True}, {"x": 2**31},
                       {"x": -(2**31)-1}, {"x": float("nan")}, {"x": float("inf")},
                       {"x": -float("inf")}, {"x": "1"}, {"x": object()}, {"\ud800": 1}):
            with self.assertRaises(wire.WireError):
                wire.update(fields)
        for base in (-1, 2**64, True):
            with self.assertRaises(wire.WireError):
                wire.update({}, base)
        model = wire.Model()
        model.apply(wire.update({"a": 1}), 10)
        before = model.state, model.history, model.commits, model.edits
        for data, sequence in ((b"broken", 11), (wire.update({"a": 2}), 10),
                               (wire.update({"a": 2}), -1)):
            with self.assertRaises(wire.WireError):
                model.apply(data, sequence)
            self.assertEqual((model.state, model.history, model.commits, model.edits), before)

    def test_seeded_histories(self):
        seed = 42
        rng = random.Random(seed)
        model, expected = wire.Model(), {}
        edits = 0
        for sequence in range(1, 129):
            fields = {f"field{rng.randrange(8)}": (None if rng.randrange(4) == 0 else rng.randint(-100, 100))
                      for _ in range(rng.randrange(1, 5))}
            for name, value in fields.items():
                if value is None:
                    expected.pop(name, None)
                else:
                    expected[name] = value
            edits += len(fields)
            self.assertEqual(model.apply(wire.update(fields), sequence), expected, f"seed={seed}, commit={sequence}")
            self.assertEqual((model.commits, model.edits), (sequence, edits))


if __name__ == "__main__":
    unittest.main()
