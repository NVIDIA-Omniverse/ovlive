# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent byte witnesses; no native/generated/USD imports or service."""
import importlib.util
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parents[1] / "reference_service"
for name in ("scene_wire", "scene_graph_wire"):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, DIRECTORY / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
wire = sys.modules["scene_graph_wire"]


def replace(data, offset, fmt, value):
    result = bytearray(data)
    struct.pack_into("<" + fmt, result, offset, value)
    return bytes(result)


def location(data, table, index):
    """Fixture introspection independent of the production/test reader class."""
    vtable = table - struct.unpack_from("<i", data, table)[0]
    offset = struct.unpack_from("<H", data, vtable + 4 + 2 * index)[0]
    return table + offset if offset else None


def target(data, slot):
    return slot + struct.unpack_from("<I", data, slot)[0]


def root(data):
    part = struct.unpack_from("<Q", data, 8)[0]
    return target(data, part)


def record(data, vector_slot, index=0):
    vector = target(data, vector_slot)
    return target(data, vector + 4 + 4 * index)


def literal_create():
    # Complete hand-authored part. Root has absent scalar defaults and two
    # command-union vectors; no encoder output is used for this witness.
    part = bytearray(102)
    struct.pack_into("<I4s", part, 0, 24, b"OODF")
    struct.pack_into("<6H", part, 8, 12, 12, 0, 0, 4, 8)
    struct.pack_into("<iII", part, 24, 16, 8, 12)
    struct.pack_into("<IB", part, 36, 1, 1)
    struct.pack_into("<II", part, 44, 1, 16)
    struct.pack_into("<6H", part, 52, 12, 32, 8, 16, 24, 28)
    struct.pack_into("<i", part, 64, 12)
    struct.pack_into("<QQIB", part, 72, 1, 0x20000000000007, 8, 6)
    struct.pack_into("<I2s", part, 96, 1, b"P\0")
    return struct.pack("<4sIQ", b"OMPF", 1, 16) + part


class SceneGraphWireTests(unittest.TestCase):
    def test_path_list_buckets_exact_wire_and_invalid_combinations(self):
        def vector(path):
            return struct.pack('<QI', 1, len(path)) + path.encode()
        op = wire.PathOp(added=('/A',), prepended=('/P',), appended=('/Z',), deleted=('/D',), ordered=('/O',))
        expected = b'\x22\0\x7c' + b''.join(vector(p) for p in ('/A', '/P', '/Z', '/D', '/O'))
        for value, raw in ((op, expected), (wire.PathOp(), b'\x22\0\0'),
                           (wire.PathOp(True), b'\x22\0\x01'),
                           (wire.PathOp(True, ('/A',)), b'\x22\0\x03' + vector('/A'))):
            self.assertEqual(wire._value(value, wire._Budget()), raw)
            self.assertEqual(wire._decode_value(raw, wire._Budget()), value)
        for raw in (expected[:-1], expected + b'\0', b'\x22\0', b'\x22\0\x80',
                    b'\x22\0\x02', b'\x22\0\x05',
                    b'\x22\0\x04' + struct.pack('<Q', 2) + vector('/A')[8:] * 2):
            with self.subTest(raw=raw), self.assertRaises(wire.GraphWireError) as error:
                wire._decode_value(raw, wire._Budget())
            self.assertEqual(error.exception.kind, 'malformed')
        for invalid in (wire.PathOp(True, added=('/A',)), wire.PathOp(explicit_items=('/A',)),
                        wire.PathOp(added=('/A', '/A')), wire.PathOp(explicit=1)):
            with self.assertRaises(wire.GraphWireError):
                wire._value(invalid, wire._Budget())
        # Identical entries in different nonexplicit buckets remain authored.
        overlapping = wire.PathOp(added=('/A',), deleted=('/A',))
        self.assertEqual(wire._decode_value(wire._value(overlapping, wire._Budget()), wire._Budget()), overlapping)
        for key, limit in (('MAX_STRING_BYTES', 9), ('MAX_RECORDS', 4)):
            with patch.object(wire, key, limit):
                for operation in (lambda: wire._value(op, wire._Budget()),
                                  lambda: wire._decode_value(expected, wire._Budget())):
                    with self.assertRaises(wire.GraphWireError) as error:
                        operation()
                    self.assertEqual(error.exception.kind, 'limit')

    def test_path_vector_bytes_types_and_shared_budgets(self):
        paths = wire.Paths(('/A.x', '/B/ns:property'))
        # Namespaced properties require a dot, so this path is deliberately
        # outside the grammar and must not pass merely as token text.
        with self.assertRaises(wire.GraphWireError):
            wire._value(paths, wire._Budget())
        paths = wire.Paths(('/A.x', '/B.ns:property'))
        expected = b'\x28\0' + struct.pack('<Q', 2) + struct.pack('<I', 4) + b'/A.x' + struct.pack('<I', 14) + b'/B.ns:property'
        self.assertEqual(wire._value(paths, wire._Budget()), expected)
        self.assertEqual(wire._decode_value(expected, wire._Budget()), paths)
        self.assertEqual(wire._decode_value(b'\x28\0' + bytes(8), wire._Budget()), wire.Paths(()))
        self.assertNotEqual(wire.typed_value(paths), wire.typed_value(paths.value))
        for raw in (expected[:-1], expected + b'x', b'\x28\0' + bytes(7),
                    b'\x28\0' + struct.pack('<QI', 1, 2) + b'/\xff'):
            with self.subTest(raw=raw), self.assertRaises(wire.GraphWireError) as error:
                wire._decode_value(raw, wire._Budget())
            self.assertEqual(error.exception.kind, 'malformed')
        delta = wire.Delta(fields=(wire.Field(1, 'k', wire.Paths(('/A', '/B'))),))
        data = wire.encode(delta)
        for key, limit in (('MAX_STRING_BYTES', 4), ('MAX_RECORDS', 2)):
            with patch.object(wire, key, limit):
                for operation in (lambda: wire.encode(delta), lambda: wire.decode(data)):
                    with self.assertRaises(wire.GraphWireError) as error:
                        operation()
                    self.assertEqual(error.exception.kind, 'limit')
        with patch.object(wire, 'MAX_STRING_BYTES', 5), patch.object(wire, 'MAX_RECORDS', 3):
            self.assertEqual(wire.encode(delta), data)
            self.assertEqual(wire.decode(data), delta)

    def test_public_authoring_scalars_have_independent_byte_witnesses(self):
        witnesses = ((False, b'\x01\0\0'), (True, b'\x01\0\x01'),
                     (wire.Token(''), b'\x0b\0\0\0\0\0'),
                     (wire.Token('int'), b'\x0b\0\x03\0\0\0int'),
                     (wire.Token('λ'), b'\x0b\0\x02\0\0\0\xce\xbb'),
                     (wire.Specifier(0), b'\x2a\0\0\0\0\0'),
                     (wire.Specifier(2), b'\x2a\0\x02\0\0\0'),
                     (wire.Variability(0), b'\x2c\0\0\0\0\0'),
                     (wire.Variability(1), b'\x2c\0\x01\0\0\0'))
        for value, expected in witnesses:
            with self.subTest(value=value):
                data = wire.encode(wire.Delta(fields=(wire.Field(1, 'k', value),)))
                table = record(data, location(data, root(data), 4))
                blob = target(data, location(data, table, 2))
                size = struct.unpack_from('<I', data, blob)[0]
                self.assertEqual(data[blob + 4:blob + 4 + size], expected)
                actual = wire._decode_value(expected, wire._Budget())
                self.assertIs(type(actual), type(value))
                self.assertEqual(actual, value)
                self.assertIs(type(wire.decode(data).fields[0].value), type(value))
        for raw in (b'\x01\0\x02', b'\x01\0', b'\x01\0\x01\0',
                    b'\x2a\0\x03\0\0\0', b'\x2c\0\x02\0\0\0',
                    b'\x2a\0\xff\xff\xff\xff', b'\x2c\0\x01',
                    b'\x0b\0\x02\0\0\0x', b'\x0b\0\x01\0\0\0\xff',
                    b'\x0b\0\x01\0\0\0\0'):
            with self.subTest(raw=raw), self.assertRaises(wire.GraphWireError) as caught:
                wire._decode_value(raw, wire._Budget())
            self.assertEqual(caught.exception.kind, 'malformed')
        for value in (wire.Specifier(True), wire.Specifier(-1), wire.Specifier(3),
                      wire.Variability(2), wire.Token('bad\0token'), wire.Token('\ud800')):
            with self.subTest(value=value), self.assertRaises(wire.GraphWireError) as caught:
                wire.encode(wire.Delta(fields=(wire.Field(1, 'k', value),)))
            self.assertEqual(caught.exception.kind, 'malformed')

    def test_presence_families_have_independent_byte_witnesses(self):
        # Kit's presence layer authors string, string-array, token-array, float, vec2f, vec4f-array, vec3d and
        # matrix4d values; each family's scalar or array counterpart has a witness too. Layouts follow the version
        # zero USD value table: a u16 type with 0x80 for arrays, u32-length strings, u64 array counts and
        # little-endian IEEE-754 components, row-major for a matrix.
        def strings(tag, *items):
            return struct.pack('<HQ', tag, len(items)) + b''.join(struct.pack('<I', len(i)) + i for i in items)

        def f32(*values):
            return struct.pack('<' + 'f' * len(values), *values)

        def f64(*values):
            return struct.pack('<' + 'd' * len(values), *values)

        identity = [float(i == j) for i in range(4) for j in range(4)]
        witnesses = (
            (wire.String('/Camera'), b'\x0a\0\x07\0\0\0/Camera'),
            (wire.String(''), b'\x0a\0\0\0\0\0'),
            (wire.String('λ'), b'\x0a\0\x02\0\0\0\xce\xbb'),
            (wire.Strings(('/World/Cube', '')), strings(0x8a, b'/World/Cube', b'')),
            (wire.Strings(()), strings(0x8a)),
            (wire.TokenArray(('xformOp:transform',)), strings(0x8b, b'xformOp:transform')),
            (wire.TokenArray(()), strings(0x8b)),
            (wire.real('float', 50.0), b'\x08\0' + f32(50.0)),
            (wire.Real('float', (0x80000000,)), b'\x08\0\0\0\0\x80'),
            (wire.Real('float', (0x7fc00123,)), b'\x08\0' + struct.pack('<I', 0x7fc00123)),
            (wire.real('float', 1.5, -0.0, array=True), b'\x88\0' + struct.pack('<Q', 2) + f32(1.5, -0.0)),
            (wire.real('double', -0.0, array=True), b'\x89\0' + struct.pack('<Q', 1) + f64(-0.0)),
            (wire.Real('double', (0x7ff0000000000000, 0x7ff8000000000001), True),
             b'\x89\0' + struct.pack('<QQQ', 2, 0x7ff0000000000000, 0x7ff8000000000001)),
            (wire.real('matrix4d', *identity), b'\x0f\0' + f64(*identity)),
            (wire.Real('matrix4d', (), True), b'\x8f\0' + bytes(8)),
            (wire.real('matrix4d', *identity, *identity[::-1], array=True),
             b'\x8f\0' + struct.pack('<Q', 2) + f64(*identity, *identity[::-1])),
            (wire.real('vec2f', 1.0, 1000000.0), b'\x14\0' + f32(1.0, 1000000.0)),
            (wire.real('vec2f', 1.0, 2.0, 3.0, 4.0, array=True), b'\x94\0' + struct.pack('<Q', 2) + f32(1.0, 2.0, 3.0, 4.0)),
            (wire.real('vec3d', 0.0, -0.0, -500.0), b'\x17\0' + f64(0.0, -0.0, -500.0)),
            (wire.real('vec3d', 1.0, 2.0, 3.0, array=True), b'\x97\0' + struct.pack('<Q', 1) + f64(1.0, 2.0, 3.0)),
            (wire.real('vec4f', 0.0, 1.0, 0.0, 5.0), b'\x1c\0' + f32(0.0, 1.0, 0.0, 5.0)),
            (wire.Real('vec4f', (), True), b'\x9c\0' + bytes(8)))
        for value, expected in witnesses:
            with self.subTest(value=value):
                data = wire.encode(wire.Delta(fields=(wire.Field(1, 'k', value),)))
                table = record(data, location(data, root(data), 4))
                blob = target(data, location(data, table, 2))
                size = struct.unpack_from('<I', data, blob)[0]
                self.assertEqual(data[blob + 4:blob + 4 + size], expected)
                actual = wire._decode_value(expected, wire._Budget())
                self.assertIs(type(actual), type(value))
                self.assertEqual(actual, value)
                self.assertEqual(wire.decode(data).fields[0].value, value)
        # Equality is by representation: signed zeros and NaN payloads differ.
        self.assertNotEqual(wire.real('float', 0.0), wire.real('float', -0.0))
        self.assertNotEqual(wire.Real('float', (0x7fc00001,)), wire.Real('float', (0x7fc00002,)))
        self.assertNotEqual(wire.real('vec3d', 0.0, 0.0, 0.0), wire.real('vec3d', 0.0, 0.0, 0.0, array=True))
        # Each family projects to its own type, distinct from token, token-vector and double values.
        self.assertNotEqual(wire.typed_value(wire.TokenArray(('a',))), wire.typed_value(('a',)))
        self.assertNotEqual(wire.typed_value(wire.String('a')), wire.typed_value(wire.Token('a')))
        self.assertEqual(wire.typed_value(wire.real('vec2f', 1.0, -0.0)),
                         {'type': 'vec2f', 'value': [0x3f800000, 0x80000000]})
        self.assertEqual(wire.typed_value(wire.Real('vec4f', (), True)), {'type': 'vec4f[]', 'value': []})
        self.assertEqual(wire.typed_value(wire.Strings(('a',))), {'type': 'string[]', 'value': ['a']})

    def test_presence_families_reject_malformed_unsupported_and_over_limit_values(self):
        def rejected(kind, call):
            with self.assertRaises(wire.GraphWireError) as error:
                call()
            self.assertEqual(error.exception.kind, kind, str(error.exception))

        def strings(tag, *items):
            return struct.pack('<HQ', tag, len(items)) + b''.join(struct.pack('<I', len(i)) + i for i in items)

        malformed = (b'\x0a\0', b'\x0a\0\x01\0\0', b'\x0a\0\x02\0\0\0x', b'\x0a\0\x01\0\0\0xy',
                     b'\x8a\0' + bytes(7), b'\x8a\0' + struct.pack('<QI', 1, 2) + b'x',
                     strings(0x8a, b'a') + b'\0', strings(0x8b, b'bad\0token'),
                     # A count beyond the content is malformed, never charged as a limit.
                     b'\x8a\0' + struct.pack('<Q', 2**64 - 1), b'\x8b\0' + struct.pack('<Q', 2**64 - 1),
                     b'\x8a\0' + struct.pack('<QI', 2, 0) + bytes(3), b'\x8b\0' + struct.pack('<QI', 2, 0) + bytes(3),
                     b'\x08\0' + bytes(3), b'\x08\0' + bytes(5), b'\x0f\0' + bytes(15 * 8), b'\x0f\0' + bytes(17 * 8),
                     b'\x14\0' + bytes(4), b'\x17\0' + bytes(16), b'\x1c\0' + bytes(12),
                     b'\x88\0' + bytes(7), b'\x88\0' + struct.pack('<Q', 2) + bytes(4),
                     b'\x88\0' + struct.pack('<Q', 2**64 - 1), b'\x8f\0' + struct.pack('<Q', 1) + bytes(15 * 8),
                     b'\x9c\0' + struct.pack('<Q', 1) + bytes(17))
        for raw in malformed:
            with self.subTest(raw=raw):
                rejected('malformed', lambda raw=raw: wire._decode_value(raw, wire._Budget()))
        # Valid wire values outside the profile: non-UTF-8 or NUL string bytes, the remaining Gf, half, integer
        # and asset families and their arrays, dictionaries, type 0, a nonzero version byte and a nonfinite
        # scalar double.
        unsupported = (b'\x0a\0\x01\0\0\0\xff', b'\x0a\0\x01\0\0\0\0', strings(0x8a, b'\xff'), strings(0x8a, b'\0'),
                       b'\x07\0\0\0', b'\x0e\0' + bytes(72), b'\x10\0' + bytes(32), b'\x11\0' + bytes(16),
                       b'\x13\0' + bytes(16), b'\x15\0' + bytes(4), b'\x18\0' + bytes(12), b'\x1b\0' + bytes(32),
                       b'\x0c\0\0\0\0\0', b'\x1f\0' + bytes(8), b'\0\0', b'\x81\0' + bytes(8), b'\x83\0' + bytes(8),
                       b'\x8c\0' + bytes(8), b'\x8e\0' + bytes(8), b'\x98\0' + bytes(8), b'\x08\x01' + bytes(4),
                       b'\x88\x01' + bytes(8), b'\x09\0' + struct.pack('<d', float('inf')))
        for raw in unsupported:
            with self.subTest(raw=raw):
                rejected('unsupported', lambda raw=raw: wire._decode_value(raw, wire._Budget()))
        for value in (wire.String('nul\0'), wire.Strings(('ok', '\0'))):
            rejected('unsupported', lambda value=value: wire.encode(wire.Delta(fields=(wire.Field(1, 'k', value),))))
        for value in (wire.Real('matrix4d', (0,) * 15), wire.Real('vec2f', (0,) * 3, True), wire.Real('double', (0,)),
                      wire.Real('float', (2**32,)), wire.Real('float', (-1,)), wire.Real('float', (True,)),
                      wire.Real('float', [0]), wire.Real('quatd', (0,) * 4), wire.Real('float', (0,), 1),
                      wire.String(b'x'), wire.String('\ud800'), wire.Strings(['a']), wire.Strings((b'a',)),
                      wire.TokenArray(('bad\0',)), wire.TokenArray(['a'])):
            with self.subTest(value=value):
                rejected('malformed', lambda value=value: wire.encode(wire.Delta(fields=(wire.Field(1, 'k', value),))))
        for call in (lambda: wire.real('float', 0.1), lambda: wire.real('float', 1e300), lambda: wire.real('float', 1),
                     lambda: wire.real('quatd', 0.0)):
            rejected('malformed', call)
        # Strings and token arrays share the string-byte budget; every array charges its elements.
        for value, strings_needed, records_needed in ((wire.String('abcd'), 5, 1), (wire.Strings(('abc', 'λ')), 6, 3),
                                                      (wire.TokenArray(('ab', 'c')), 4, 3),
                                                      (wire.real('vec3d', *([1.0] * 6), array=True), 1, 3)):
            delta = wire.Delta(fields=(wire.Field(1, 'k', value),))
            data = wire.encode(delta)
            for setting, exact in (('MAX_STRING_BYTES', strings_needed), ('MAX_RECORDS', records_needed)):
                with self.subTest(value=value, limit=setting):
                    with patch.object(wire, setting, exact):
                        self.assertEqual(wire.encode(delta), data)
                        self.assertEqual(wire.decode(data), delta)
                    with patch.object(wire, setting, exact - 1):
                        rejected('limit', lambda: wire.encode(delta))
                        rejected('limit', lambda: wire.decode(data))

    def test_scalar_tokens_share_utf8_budget_with_fields_and_vectors(self):
        source = wire.Delta(fields=(wire.Field(1, 'a', wire.Token('λ')),
                                   wire.Field(1, 'b', ('int',))))
        data = wire.encode(source)  # 2 key bytes + 2 scalar bytes + 3 vector bytes.
        with patch.object(wire, 'MAX_STRING_BYTES', 7):
            self.assertEqual(wire.decode(data), source)
            self.assertEqual(wire.encode(source), data)
        with patch.object(wire, 'MAX_STRING_BYTES', 6):
            for operation in (lambda: wire.decode(data), lambda: wire.encode(source)):
                with self.assertRaises(wire.GraphWireError) as caught:
                    operation()
                self.assertEqual(caught.exception.kind, 'limit')

    def test_source_format_bytes_defaults_and_shared_limits(self):
        delta = wire.Delta(True, source_format="usd-\u03bb-\U0001f680", fields=(wire.Field(1, "k", 1),))
        data = wire.encode(delta)
        self.assertEqual(wire.decode(data), delta)
        slot = location(data, root(data), 6)
        string = target(data, slot)
        raw = delta.source_format.encode("utf-8")
        self.assertEqual(struct.unpack_from("<I", data, string)[0], len(raw))
        self.assertEqual(data[string + 4:string + 5 + len(raw)], raw + b"\0")
        with patch.object(wire, "MAX_STRING_BYTES", len(raw) + 1):
            self.assertEqual(wire.encode(delta), data)
            self.assertEqual(wire.decode(data), delta)
        with patch.object(wire, "MAX_STRING_BYTES", len(raw)):
            for operation in (lambda: wire.encode(delta), lambda: wire.decode(data)):
                with self.assertRaises(wire.GraphWireError) as error:
                    operation()
                self.assertEqual(error.exception.kind, "limit")
        for value in (None, 1, "bad\0format", "\ud800"):
            with self.subTest(value=value), self.assertRaises(wire.GraphWireError) as error:
                wire.encode(wire.Delta(True, source_format=value))
            self.assertEqual(error.exception.kind, "malformed")
        with self.assertRaises(wire.GraphWireError) as error:
            wire.encode(wire.Delta(source_format="usda"))
        self.assertEqual(error.exception.kind, "malformed")
        for offset, value in ((string + 4, 0xff), (string + 4, 0), (string + 4 + len(raw), 1)):
            with self.assertRaises(wire.GraphWireError) as error:
                wire.decode(replace(data, offset, "B", value))
            self.assertEqual(error.exception.kind, "malformed")
        # Explicit empty strings and absence have the same wire semantics,
        # including on an ordinary delta. Construct the empty-present witness
        # without the encoder, which deliberately elides empty metadata.
        empty = replace(replace(data, string, "I", 0), string + 4, "B", 0)
        for is_diff in (False, True):
            decoded = wire.decode(replace(empty, location(empty, root(empty), 0), "B", is_diff))
            self.assertEqual(decoded.source_format, "")
            self.assertEqual(decoded.is_diff, is_diff)
            absent = wire.encode(wire.Delta(is_diff, fields=delta.fields))
            self.assertIsNone(location(absent, root(absent), 6))
            self.assertEqual(wire.decode(absent), decoded)

    def test_independent_literal_create_and_scalar_defaults(self):
        data = literal_create()
        self.assertEqual(wire.decode(data), wire.Delta(commands=(
            wire.Command(1, 0x20000000000007, parent=1, name="P", spec_type=6),)))
        # Scalar defaults are decoded, except Reorder's required parent/vector.
        for tag in range(1, 7):
            command = wire.Command(tag, 0)
            encoded = wire.encode(wire.Delta(commands=(command,)))
            table = record(encoded, location(encoded, root(encoded), 3))
            vt = table - struct.unpack_from("<i", encoded, table)[0]
            size = struct.unpack_from("<H", encoded, vt)[0]
            for offset in range(vt + 4, vt + size, 2):
                encoded = replace(encoded, offset, "H", 0)
            if tag == 4:
                with self.assertRaises(wire.GraphWireError):
                    wire.decode(encoded)
            else:
                self.assertEqual(wire.decode(encoded).commands, (command,))
        words = [0x46504D4F, 1, 16, 0, 24, 0x46444F4F, 0x00140010,
                 0, 0x00080004, 0x0010000C, 16, 28, 20, 12, 4, 0, 0, 0, 0]
        self.assertEqual(wire.decode(struct.pack("<19I", *words)), wire.Delta())

    def test_all_command_slot_layouts(self):
        cases = (
            (wire.Command(1, 17, parent=9, name="Child", spec_type=6), ((0, "Q", 9), (1, "Q", 17), (3, "B", 6))),
            (wire.Command(2, 18, parent=10), ((0, "Q", 10), (1, "Q", 18))),
            (wire.Command(3, 19, parent=11, old_parent=12, name="Renamed"), ((0, "Q", 12), (1, "Q", 11), (2, "Q", 19))),
            (wire.Command(4, 20, child_list=2, children=(77, 55)), ((0, "Q", 20), (1, "B", 2))),
            (wire.Command(5, 21, parent=13, name="Final", spec_type=1, order=2**32-1),
             ((0, "Q", 13), (1, "Q", 21), (3, "B", 1), (4, "I", 2**32-1))),
            (wire.Command(6, 22), ((0, "Q", 22),)),
        )
        for command, expected in cases:
            with self.subTest(tag=command.tag):
                data = wire.encode(wire.Delta(True, 2**64-1, (command,)))
                rt = root(data)
                self.assertEqual(struct.unpack_from("<Q", data, location(data, rt, 1))[0], 2**64-1)
                tags = target(data, location(data, rt, 2))
                self.assertEqual(data[tags:tags + 5], struct.pack("<IB", 1, command.tag))
                table = record(data, location(data, rt, 3))
                for index, fmt, value in expected:
                    self.assertEqual(struct.unpack_from("<" + fmt, data, location(data, table, index))[0], value)
                if command.tag in (1, 3, 5):
                    string = target(data, location(data, table, 3 if command.tag == 3 else 2))
                    self.assertEqual(data[string:string + 5 + len(command.name)],
                                     struct.pack("<I", len(command.name)) + command.name.encode() + b"\0")
                if command.tag == 4:
                    vector = target(data, location(data, table, 2))
                    self.assertEqual((vector + 4 - 16) % 8, 0)
                    self.assertEqual(data[vector:vector + 20], struct.pack("<IQQ", 2, 77, 55))
                self.assertEqual(wire.decode(data), wire.Delta(True, 2**64-1, (command,)))

    def test_value_bytes_deletion_empty_and_order(self):
        fields = (wire.Field(17, "integer", -(2**31), 100), wire.Field(17, "negativeZero", -0.0, 2),
                  wire.Field(19, "primChildren", ("B", "é", ""), 1),
                  wire.Field(19, "primChildren", (), 0), wire.Field(19, "primChildren", None, 0))
        data = wire.encode(wire.Delta(fields=fields))
        expected = (bytes.fromhex("030000000080"), bytes.fromhex("09000000000000000080"),
                    bytes.fromhex("29000300000000000000010000004202000000c3a900000000"),
                    bytes.fromhex("29000000000000000000"), None)
        for index, value in enumerate(expected):
            table = record(data, location(data, root(data), 4), index)
            self.assertEqual(struct.unpack_from("<Q", data, location(data, table, 0))[0], fields[index].section)
            self.assertEqual(struct.unpack_from("<Q", data, location(data, table, 5))[0], fields[index].set_order)
            slot = location(data, table, 2)
            if value is None:
                self.assertIsNone(slot)
            else:
                vector = target(data, slot)
                self.assertEqual(data[vector:vector + 4 + len(value)], struct.pack("<I", len(value)) + value)
        decoded = wire.decode(data)
        self.assertEqual(decoded.fields, fields)
        self.assertEqual(struct.pack("<d", decoded.fields[1].value), bytes.fromhex("0000000000000080"))
        # A present zero-byte USD blob also denotes deletion, not an empty vector.
        table = record(data, location(data, root(data), 4), 3)
        vector = target(data, location(data, table, 2))
        self.assertIsNone(wire.decode(replace(data, vector, "I", 0)).fields[3].value)

    def test_literal_truncation_union_offsets_and_malformed_tables(self):
        data = literal_create()
        for length in range(len(data)):
            with self.subTest(length=length), self.assertRaises(wire.GraphWireError):
                wire.decode(data[:length])
        # Offsets below are independently fixed by literal_create, not decode.
        cases = ((0, "I", 0), (4, "I", 2), (8, "Q", 0), (20, "I", 0),
                 (16 + 8 + 8, "H", 0), (16 + 36, "I", 0), (16 + 40, "B", 0),
                 (16 + 40, "B", 7), (16 + 48, "I", 0), (16 + 48, "I", 2**32-1),
                 (16 + 64, "i", 0), (16 + 52, "H", 3), (16 + 52, "H", 14), (16 + 54, "H", 8),
                 (16 + 96, "I", 2**32-1), (16 + 100, "B", 0xff), (16 + 101, "B", 1))
        for offset, fmt, value in cases:
            with self.subTest(offset=offset, value=value), self.assertRaises(wire.GraphWireError):
                wire.decode(replace(data, offset, fmt, value))
        # Reader must accept legal signed shared-vtable displacement and outer padding.
        commands = (wire.Command(2, 3, parent=1), wire.Command(2, 4, parent=1))
        data = wire.encode(wire.Delta(commands=commands))
        first, second = (record(data, location(data, root(data), 3), i) for i in (0, 1))
        shared = second - struct.unpack_from("<i", data, second)[0]
        self.assertLess(first - shared, 0)
        data = replace(data, first, "i", first - shared)
        padded = replace(data[:16] + b"pad" + data[16:], 8, "Q", 19)
        self.assertEqual(wire.decode(padded).commands, commands)

    def test_unsupported_value_metadata_and_malformed_token_lengths(self):
        data = wire.encode(wire.Delta(fields=(wire.Field(2, "child", ("P",)),)))
        rt = root(data)
        table = record(data, location(data, rt, 4))
        vector = target(data, location(data, table, 2))
        malformed = [replace(data, vector + 4, "H", tag) for tag in (40, 0x29 | 0x80, 0x129, 11)]
        malformed += [replace(data, vector + 6, "Q", 2), replace(data, vector + 14, "I", 100)]
        malformed += [replace(data, vector + 18, "B", 0), replace(data, vector + 18, "B", 0xff)]
        malformed += [replace(data, vector, "I", n) for n in (1, 2, 9, 14)]
        # Reuse existing bounded scalar storage as external size/index; root
        # field vector as malformed sourceFormat or an unsupported sample vector.
        vt = table - struct.unpack_from("<i", data, table)[0]
        scalar_offset = location(data, table, 0) - table
        malformed += [replace(data, vt + 4 + index * 2, "H", scalar_offset) for index in (3, 4)]
        rv = rt - struct.unpack_from("<i", data, rt)[0]
        fs_offset = location(data, rt, 4) - rt
        malformed += [replace(data, rv + 4 + index * 2, "H", fs_offset) for index in (5, 6)]
        for bad in malformed:
            with self.subTest(data=bad.hex()), self.assertRaises(wire.GraphWireError):
                wire.decode(bad)
        for bits in (0x7ff0000000000000, 0xfff0000000000000, 0x7ff8000000000001):
            value = wire.encode(wire.Delta(fields=(wire.Field(1, "number", 1.0),)))
            table = record(value, location(value, root(value), 4))
            blob = target(value, location(value, table, 2))
            with self.assertRaises(wire.GraphWireError):
                wire.decode(replace(value, blob + 6, "Q", bits))

    def test_shared_limits_charge_repeated_references(self):
        delta = wire.Delta(commands=(wire.Command(4, 1, children=(2, 3)),),
                           fields=(wire.Field(2, "abc", ("def", "ghi")),))
        data = wire.encode(delta)
        for budget in ("MAX_RECORDS", "MAX_STRING_BYTES"):
            exact = 6 if budget == "MAX_RECORDS" else 9
            with patch.object(wire, budget, exact):
                self.assertEqual(wire.decode(data), delta)
                self.assertEqual(wire.encode(delta), data)
            with patch.object(wire, budget, exact - 1):
                with self.assertRaises(wire.GraphWireError):
                    wire.decode(data)
                with self.assertRaises(wire.GraphWireError):
                    wire.encode(delta)
        with patch.object(wire, "MAX_BYTES", len(data)):
            self.assertEqual(wire.encode(delta), data)
            self.assertEqual(wire.decode(data), delta)
        with patch.object(wire, "MAX_BYTES", len(data) - 1):
            with self.assertRaises(wire.GraphWireError):
                wire.encode(delta)
            with self.assertRaises(wire.GraphWireError):
                wire.decode(data)
        fields = (wire.Field(1, "name", ("token",)), wire.Field(2, "name", ("token",)))
        shared = wire.encode(wire.Delta(fields=fields))
        vector = target(shared, location(shared, root(shared), 4))
        first = target(shared, vector + 4)
        shared = replace(shared, vector + 8, "I", first - (vector + 8))
        with patch.object(wire, "MAX_STRING_BYTES", 9):
            with self.assertRaises(wire.GraphWireError):
                wire.decode(shared)

    def test_error_categories_do_not_turn_profile_limits_into_malformed_wire(self):
        def rejected(kind, call):
            with self.assertRaises(wire.GraphWireError) as error:
                call()
            self.assertEqual(error.exception.kind, kind, str(error.exception))

        self.assertEqual(wire.GraphWireError("default").kind, "malformed")
        source = wire.Delta(fields=(wire.Field(2, "key", ("token",)),))
        data = wire.encode(source)
        for setting, maximum in (("MAX_BYTES", len(data) - 1), ("MAX_RECORDS", 1), ("MAX_STRING_BYTES", 7)):
            with self.subTest(limit=setting), patch.object(wire, setting, maximum):
                rejected("limit", lambda: wire.encode(source))
                rejected("limit", lambda: wire.decode(data))
        for bad in (b"", b"broken", bytearray(data), replace(data, 4, "I", 0),
                    replace(data, 4, "I", 2), replace(data, 8, "Q", len(data))):
            rejected("malformed", lambda bad=bad: wire.decode(bad))
        # Two correctly framed parts are outside this profile, unlike merely
        # changing the part count without providing a valid offset table.
        part = literal_create()[16:]
        multipart = struct.pack("<4sIQQ", b"OMPF", 2, 24, 24 + len(part)) + part + b"external"
        rejected("unsupported", lambda: wire.decode(multipart))

        rt = root(data)
        table = record(data, location(data, rt, 4))
        blob = target(data, location(data, table, 2))
        rejected("malformed", lambda: wire.decode(replace(data, blob, "I", 1)))
        rejected("malformed", lambda: wire.decode(replace(data, blob + 14, "I", 2**32-1)))
        rejected("unsupported", lambda: wire.decode(replace(data, blob + 4, "H", 0x129)))
        # A fully formed boolean belongs to the public authoring profile.
        boolean = replace(replace(data, blob, "I", 3), blob + 4, "H", 1)
        boolean = replace(boolean, blob + 6, "B", 1)
        self.assertIs(wire.decode(boolean).fields[0].value, True)
        vt = table - struct.unpack_from("<i", data, table)[0]
        external = replace(data, vt + 4 + 3 * 2, "H", location(data, table, 0) - table)
        rejected("unsupported", lambda: wire.decode(external))
        rv = rt - struct.unpack_from("<i", data, rt)[0]
        samples = replace(data, rv + 4 + 5 * 2, "H", location(data, rt, 4) - rt)
        rejected("malformed", lambda: wire.decode(samples))
        # Use the real bounded field-key string for a well-formed sourceFormat.
        source_format = replace(data, rv + 4 + 6 * 2, "H", 32)
        key = target(data, location(data, table, 1))
        source_format = replace(source_format, rt + 32, "I", key - rt - 32)
        rejected("malformed", lambda: wire.decode(source_format))  # Nonempty ordinary metadata.
        diff_format = replace(source_format, location(source_format, rt, 0), "B", 1)
        self.assertEqual(wire.decode(diff_format).source_format, "key")
        rejected("malformed", lambda: wire.decode(replace(source_format, rt + 32, "I", 2**32-1)))
        rejected("unsupported", lambda: wire.encode(wire.Delta(fields=(wire.Field(2, "x", float("inf")),))))
        # Bare text is neither a token nor a string value.
        for value in (2**31, ["wrong container"], object(), "bare text"):
            rejected("malformed", lambda value=value: wire.encode(wire.Delta(fields=(wire.Field(2, "x", value),))))

    def test_encode_rejects_unrepresentable_input_without_changing_owners(self):
        command = wire.Command(1, 2, parent=1, name="P", spec_type=6)
        fields = (wire.Field(2, "n", 7),)
        source = wire.Delta(commands=(command,), fields=fields)
        before = wire.encode(source)
        invalid = (wire.Delta(commands=(wire.Command(7, 2),)),
                   wire.Delta(commands=(wire.Command(6, 2, name="discarded"),)),
                   wire.Delta(commands=(wire.Command(4, 1, children=(True,)),)),
                   wire.Delta(commands=(wire.Command(1, 2, name="\ud800"),)),
                   wire.Delta(commands=(wire.Command(1, -1),)),
                   wire.Delta(is_diff=1), wire.Delta(base_version=2**64),
                   wire.Delta(fields=(wire.Field(2, "x", wire.Specifier(3)),)),
                   wire.Delta(fields=(wire.Field(2, "x", 2**31),)),
                   wire.Delta(fields=(wire.Field(2, "x", float("nan")),)),
                   wire.Delta(fields=(wire.Field(2, "x", ("bad\0token",)),)),
                   wire.Delta(fields=(wire.Field(2, "x", ["wrong container"]),)))
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(wire.GraphWireError):
                wire.encode(value)
            self.assertEqual(wire.encode(source), before)
        self.assertIs(source.fields, fields)
        self.assertIs(source.commands[0], command)


if __name__ == "__main__":
    unittest.main()
